"""
Flattening scenes into one table, named the way EQL names the same attributes, so a
query built for the relational pipeline reads the flat table's columns unchanged.

A scene holds between two and twenty object instances and no canonical order over them,
so a flat table has to choose what to do with the parts. The three layouts are the
choices a flat learner has: keep only the scene's own scalars, add the aggregation
counts the relational model derives from the parts, or unroll the parts into one block
of columns per position and pad the positions a scene does not fill. Values are kept as
they are, an enum member stays a member, which is also how a fitted tree's leaves and a
query's conditions read them.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field, fields
from enum import StrEnum

import pandas as pd
from krrood.utils import get_class_and_attribute_name
from typing_extensions import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    get_args,
    get_type_hints,
)

from experiments.causal_reasoning.graspclutter6d.domain import (
    GraspClutterScene,
    GraspClutterSceneAggregations,
)
from experiments.causal_reasoning.graspclutter6d.exceptions import (
    FlatTableSchemaMismatchError,
)


class TableLayout(StrEnum):
    """
    What a flat table holds of a scene besides its own scalars.
    """

    SCALARS = "scalars-only"
    """
    The scene's own scalars and nothing of its parts.
    """

    PROPOSITIONAL = "propositional"
    """
    The scalars and the aggregation counts over the parts, the classic
    propositionalisation of a relational example.
    """

    UNROLLED = "unrolled"
    """
    The scalars, the counts, and every part's attributes under the part's position,
    padded where a scene has fewer parts than the widest one.
    """

    @property
    def has_counts(self) -> bool:
        """
        Whether the layout carries the aggregation counts.
        """
        return self is not TableLayout.SCALARS

    @property
    def has_parts(self) -> bool:
        """
        Whether the layout carries the parts' own attributes.
        """
        return self is TableLayout.UNROLLED


class SceneView(StrEnum):
    """
    How much of a scene a likelihood is taken over.
    """

    SCALARS = "scalars"
    """
    The scene's own scalars.
    """

    SCALARS_AND_COUNTS = "scalars and counts"
    """
    The scalars and the aggregation counts over the parts.
    """

    WHOLE_SCENE = "whole scene"
    """
    The scalars, the counts, and every object and viewpoint.
    """


class AbsentPart(StrEnum):
    """
    The symbol an unrolled column holds where a scene has no part at that position.
    """

    ABSENT = "absent"


@dataclass(frozen=True)
class PartPadding:
    """
    What an unrolled column holds at a position the scene has no part for, per kind of
    attribute.
    """

    symbol: AbsentPart = AbsentPart.ABSENT
    """
    For an enum attribute: a symbol of its own next to the enum's members.
    """

    integer: int = -1
    """
    For an integer attribute: a value no real part shows, since every count and index
    the parts carry is a count of something.
    """

    real: float = -1.0
    """
    For a continuous attribute: a value outside every real one, since a part's diameter,
    visibility and distance are all positive.
    """

    def value_for(self, attribute_type: Type) -> Any:
        """
        :param attribute_type: The attribute's type.
        :return: The padding value of that kind.
        """
        if issubclass(attribute_type, enum.Enum):
            return self.symbol
        if issubclass(attribute_type, float):
            return self.real
        return self.integer


@dataclass(frozen=True)
class PartAttribute:
    """
    One attribute of one of a scene's exchangeable parts, as a query names it.
    """

    part_field: str
    """
    The exchangeable-part field of the scene, ``objects`` or ``viewpoints``.
    """

    index: int
    """
    The part's position in that field's list.
    """

    attribute: str
    """
    The part's attribute.
    """


@dataclass(frozen=True)
class SceneSchema:
    """
    The attributes of a scene as EQL names them: the scene's own scalars, its
    aggregation counts, and each object's or viewpoint's attributes under the part's
    index.
    """

    @property
    def part_fields(self) -> Tuple[str, ...]:
        """
        The exchangeable-part fields of
        :class:`~experiments.causal_reasoning.graspclutter6d.domain.GraspClutterScene`,
        which are the fields
        :class:`~experiments.causal_reasoning.graspclutter6d.domain.GraspClutterSceneAggregations`
        aggregates over.
        """
        return tuple(GraspClutterSceneAggregations.aggregation_registry)

    def part_class(self, part_field: str) -> Type:
        """
        :param part_field: An exchangeable-part field.
        :return: The class of that field's parts.
        """
        [part_class] = get_args(get_type_hints(GraspClutterScene)[part_field])
        return part_class

    def part_attribute_types(self, part_field: str) -> Dict[str, Type]:
        """
        :param part_field: An exchangeable-part field.
        :return: The attributes of that field's parts and their types.
        """
        return get_type_hints(self.part_class(part_field))

    @property
    def aggregation_statistics(self) -> Tuple[Callable[..., Any], ...]:
        """
        The aggregation statistics over every part field; a flat table carries them as
        columns, so it holds the same summaries the relational model derives from its
        parts.
        """
        return tuple(
            statistic
            for part_field in self.part_fields
            for statistic in GraspClutterSceneAggregations.aggregation_registry[
                part_field
            ]
        )

    @property
    def scalar_fields(self) -> Tuple[str, ...]:
        """
        The scene's own scalar attributes.
        """
        return tuple(
            scene_field.name
            for scene_field in fields(GraspClutterScene)
            if scene_field.name not in self.part_fields
        )

    def scalar_column(self, field_name: str) -> str:
        """
        :param field_name: A scene scalar field.
        :return: The column, and EQL variable, name of that field.
        """
        return get_class_and_attribute_name(GraspClutterScene.__name__, field_name)

    @property
    def scalar_columns(self) -> Tuple[str, ...]:
        """
        The column names of every scalar field.
        """
        return tuple(self.scalar_column(name) for name in self.scalar_fields)

    def part_column(self, part: PartAttribute) -> str:
        """
        :param part: One part's attribute.
        :return: The column, and EQL variable, name of that attribute.
        """
        return self.scalar_column(f"{part.part_field}[{part.index}].{part.attribute}")

    def part_attribute(self, variable_name: str) -> Optional[PartAttribute]:
        """
        :param variable_name: A variable name, as EQL names it.
        :return: The part attribute it names, or ``None`` if it names a scene-level
            variable.
        """
        pattern = re.compile(
            rf"^{re.escape(GraspClutterScene.__name__)}\."
            rf"({'|'.join(map(re.escape, self.part_fields))})\[(\d+)\]\.(\w+)$"
        )
        match = pattern.match(variable_name)
        if match is None:
            return None
        return PartAttribute(match.group(1), int(match.group(2)), match.group(3))

    def aggregation_column(self, statistic_name: str) -> str:
        """
        :param statistic_name: The name of one of the :attr:`aggregation_statistics`.
        :return: The column, and EQL variable, name of that statistic, which grounding
            names by its class and its call.
        """
        return get_class_and_attribute_name(
            GraspClutterSceneAggregations.__name__, f"{statistic_name}()"
        )

    @property
    def aggregation_columns(self) -> Tuple[str, ...]:
        """
        The column names of every aggregation statistic.
        """
        return tuple(
            self.aggregation_column(statistic.__name__)
            for statistic in self.aggregation_statistics
        )


@dataclass
class FlatTable:
    """
    Scenes as one row each, holding what the layout says of them.
    """

    layout: TableLayout = TableLayout.PROPOSITIONAL
    """
    What the rows hold besides the scene's own scalars.
    """

    part_widths: Dict[str, int] = field(default_factory=dict)
    """
    Per part field, how many positions an unrolled row has; a scene with more parts than
    that cannot be a row, one with fewer is padded.
    """

    padding: PartPadding = PartPadding()
    """
    What a position without a part holds.
    """

    schema: SceneSchema = field(default_factory=SceneSchema)
    """
    How the columns are named.
    """

    @classmethod
    def unrolled_for(
        cls,
        scenes: Sequence[GraspClutterScene],
        schema: SceneSchema = SceneSchema(),
    ) -> FlatTable:
        """
        :param scenes: The scenes the table has to hold.
        :param schema: How the columns are named.
        :return: An unrolled table wide enough for the largest of them.
        """
        return cls(
            layout=TableLayout.UNROLLED,
            part_widths={
                part_field: max(len(vars(scene)[part_field]) for scene in scenes)
                for part_field in schema.part_fields
            },
            schema=schema,
        )

    @property
    def part_columns(self) -> List[str]:
        """
        The unrolled parts' columns, in position order.
        """
        return [
            self.schema.part_column(PartAttribute(part_field, index, attribute))
            for part_field in self.schema.part_fields
            for index in range(self.part_widths.get(part_field, 0))
            for attribute in self.schema.part_attribute_types(part_field)
        ]

    @property
    def columns(self) -> List[str]:
        """
        The table's columns, in order.
        """
        columns = list(self.schema.scalar_columns)
        if self.layout.has_counts:
            columns += list(self.schema.aggregation_columns)
        if self.layout.has_parts:
            columns += self.part_columns
        return columns

    def columns_of(self, view: SceneView) -> Optional[List[str]]:
        """
        :param view: How much of a scene to look at.
        :return: The columns holding that much, or ``None`` if the layout holds less.
        """
        if view is SceneView.SCALARS:
            return list(self.schema.scalar_columns)
        if view is SceneView.SCALARS_AND_COUNTS and self.layout.has_counts:
            return list(self.schema.scalar_columns) + list(
                self.schema.aggregation_columns
            )
        if view is SceneView.WHOLE_SCENE and self.layout.has_parts:
            return self.columns
        return None

    def fits(self, scene: GraspClutterScene) -> bool:
        """
        :param scene: A scene.
        :return: Whether the table has a position for every one of its parts.
        """
        return not self._overflowing_columns(scene)

    def _overflowing_columns(self, scene: GraspClutterScene) -> List[str]:
        """
        :param scene: A scene.
        :return: The columns its parts past the table's width would need.
        """
        if not self.layout.has_parts:
            return []
        return [
            self.schema.part_column(PartAttribute(part_field, index, attribute))
            for part_field in self.schema.part_fields
            for index in range(
                self.part_widths[part_field], len(vars(scene)[part_field])
            )
            for attribute in self.schema.part_attribute_types(part_field)
        ]

    def row(self, scene: GraspClutterScene) -> Dict[str, Any]:
        """
        :param scene: The scene to flatten.
        :return: The scene's values, keyed by column.
        :raises FlatTableSchemaMismatchError: If the scene has more parts than the table
            has positions.
        """
        overflowing = self._overflowing_columns(scene)
        if overflowing:
            raise FlatTableSchemaMismatchError(overflowing)
        scene_values = vars(scene)
        row = {
            self.schema.scalar_column(name): scene_values[name]
            for name in self.schema.scalar_fields
        }
        if self.layout.has_counts:
            aggregations = GraspClutterSceneAggregations(instance=scene)
            row.update(
                {
                    self.schema.aggregation_column(statistic.__name__): statistic(
                        aggregations
                    )
                    for statistic in self.schema.aggregation_statistics
                }
            )
        if self.layout.has_parts:
            for part_field in self.schema.part_fields:
                row.update(self._part_values(part_field, scene_values[part_field]))
        return row

    def _part_values(self, part_field: str, parts: Sequence[Any]) -> Dict[str, Any]:
        """
        :param part_field: The exchangeable-part field the parts belong to.
        :param parts: The scene's parts of that kind, in the scene's order.
        :return: Every position's attribute values, padded past the last part.
        """
        attribute_types = self.schema.part_attribute_types(part_field)
        values = {}
        for index in range(self.part_widths[part_field]):
            part_values = vars(parts[index]) if index < len(parts) else None
            for attribute, attribute_type in attribute_types.items():
                column = self.schema.part_column(
                    PartAttribute(part_field, index, attribute)
                )
                values[column] = (
                    self.padding.value_for(attribute_type)
                    if part_values is None
                    else part_values[attribute]
                )
        return values

    def dataframe(self, scenes: Iterable[GraspClutterScene]) -> pd.DataFrame:
        """
        :param scenes: The scenes to flatten.
        :return: One row per scene, columns in :attr:`columns` order.
        """
        return pd.DataFrame([self.row(scene) for scene in scenes], columns=self.columns)
