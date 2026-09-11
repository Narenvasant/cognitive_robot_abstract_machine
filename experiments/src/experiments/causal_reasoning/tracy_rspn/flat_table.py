"""
Flattening recorded attempts into one fixed-width table, named the way EQL names the
same attributes, so a query built for the relational pipeline reads the flat table's
columns unchanged.

Values are kept as they are -- an enum member stays a member -- which is also how a
fitted tree's leaves and a query's conditions read them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd
from krrood.entity_query_language.factories import variable
from krrood.utils import get_class_and_attribute_name
from typing_extensions import Any, Dict, Iterable, List

from experiments.causal_reasoning.tracy_rspn.domain import (
    ClutteredObject,
    ClutterPickScene,
    ClutterPickSceneAggregations,
)
from experiments.causal_reasoning.tracy_rspn.exceptions import (
    FlatTableSchemaMismatchError,
)

NEIGHBOURS_FIELD = "neighbours"
"""
The exchangeable-part field of
:class:`~experiments.causal_reasoning.tracy_rspn.domain.ClutterPickScene` the table
unrolls into indexed columns.
"""

SCENE_SCALAR_FIELDS = tuple(
    field.name for field in fields(ClutterPickScene) if field.name != NEIGHBOURS_FIELD
)
"""
The scene's own scalar attributes, one column each.
"""

NEIGHBOUR_FIELDS = tuple(field.name for field in fields(ClutteredObject))
"""
A neighbour's attributes, one column each per neighbour index.
"""

AGGREGATION_STATISTICS = ("crowding_count",)
"""
The aggregation statistics of
:class:`~experiments.causal_reasoning.tracy_rspn.domain.ClutterPickSceneAggregations`,
one column each, so the flat table carries the same summaries the relational model
derives from its parts.
"""


def scene_column(field_name: str) -> str:
    """
    :param field_name: A scene scalar field.
    :return: The column, and EQL variable, name of that field.
    """
    return get_class_and_attribute_name(ClutterPickScene.__name__, field_name)


def neighbour_column(index: int, field_name: str) -> str:
    """
    :param index: The neighbour's position in the scene's neighbour list.
    :param field_name: A neighbour field.
    :return: The column, and EQL variable, name of that neighbour's field.
    """
    return scene_column(f"{NEIGHBOURS_FIELD}[{index}].{field_name}")


def aggregation_column(statistic_name: str) -> str:
    """
    :param statistic_name: One of :data:`AGGREGATION_STATISTICS`.
    :return: The column, and EQL variable, name of that statistic.
    """
    return getattr(variable(ClutterPickSceneAggregations), statistic_name)()._name_


@dataclass
class FlatTable:
    """
    Recorded attempts as one row each, every scene with the same neighbour count.
    """

    neighbour_count: int
    """
    How many neighbours every row unrolls; scenes with another count cannot be rows.
    """

    @property
    def columns(self) -> List[str]:
        """
        The table's columns, in order.
        """
        return (
            [scene_column(name) for name in SCENE_SCALAR_FIELDS]
            + [aggregation_column(name) for name in AGGREGATION_STATISTICS]
            + [
                neighbour_column(index, name)
                for index in range(self.neighbour_count)
                for name in NEIGHBOUR_FIELDS
            ]
        )

    def row(self, scene: ClutterPickScene) -> Dict[str, Any]:
        """
        :param scene: The attempt to flatten.
        :return: The attempt's encoded values, keyed by column.
        :raises FlatTableSchemaMismatchError: If ``scene`` has another neighbour count.
        """
        if len(scene.neighbours) != self.neighbour_count:
            raise FlatTableSchemaMismatchError(
                [
                    neighbour_column(index, name)
                    for index in range(
                        min(len(scene.neighbours), self.neighbour_count),
                        max(len(scene.neighbours), self.neighbour_count),
                    )
                    for name in NEIGHBOUR_FIELDS
                ]
            )
        aggregations = ClutterPickSceneAggregations(instance=scene)
        row = {scene_column(name): getattr(scene, name) for name in SCENE_SCALAR_FIELDS}
        row.update(
            {
                aggregation_column(name): getattr(aggregations, name)()
                for name in AGGREGATION_STATISTICS
            }
        )
        for index, neighbour in enumerate(scene.neighbours):
            row.update(
                {
                    neighbour_column(index, name): getattr(neighbour, name)
                    for name in NEIGHBOUR_FIELDS
                }
            )
        return row

    def dataframe(self, scenes: Iterable[ClutterPickScene]) -> pd.DataFrame:
        """
        :param scenes: The attempts to flatten.
        :return: One row per attempt, columns in :attr:`columns` order.
        """
        return pd.DataFrame([self.row(scene) for scene in scenes], columns=self.columns)
