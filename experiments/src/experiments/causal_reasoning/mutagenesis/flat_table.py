"""
Flattening molecules into one table of their own scalars and aggregation counts, named
the way EQL names the same attributes, so a query built for the relational pipeline
reads the flat table's columns unchanged.

A molecule has anywhere from 14 to 40 atoms and no canonical atom order, so there is no
fixed-width unrolling in which one column means the same thing in two molecules. The
table is therefore propositional: the classic aggregation summary of a relational
example, which is exactly the information the relational circuit's class-level circuit
is fitted on as well. Values are kept as they are, an enum member stays a member, which
is also how a fitted tree's leaves and a query's conditions read them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields

import pandas as pd
from krrood.utils import get_class_and_attribute_name
from typing_extensions import Any, Callable, Dict, Iterable, List, Optional, Tuple

from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
)


@dataclass(frozen=True)
class PartAttribute:
    """
    One attribute of one of a molecule's exchangeable parts, as a query names it.
    """

    part_field: str
    """
    The exchangeable-part field of the molecule, ``atoms`` or ``bonds``.
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
class MoleculeSchema:
    """
    The attributes of a molecule as EQL names them: the molecule's own scalars, its
    aggregation counts, and each atom's or bond's attributes under the part's index.
    """

    @property
    def part_fields(self) -> Tuple[str, ...]:
        """
        The exchangeable-part fields of
        :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMolecule`,
        which are the fields
        :class:`~experiments.causal_reasoning.mutagenesis.domain.MutagenesisMoleculeAggregations`
        aggregates over.
        """
        return tuple(MutagenesisMoleculeAggregations.aggregation_registry)

    @property
    def aggregation_statistics(self) -> Tuple[Callable[..., Any], ...]:
        """
        The aggregation statistics over every part field; the flat table carries them as
        columns, so it holds the same summaries the relational model derives from its
        parts.
        """
        return tuple(
            statistic
            for part_field in self.part_fields
            for statistic in MutagenesisMoleculeAggregations.aggregation_registry[
                part_field
            ]
        )

    @property
    def scalar_fields(self) -> Tuple[str, ...]:
        """
        The molecule's own scalar attributes.
        """
        return tuple(
            molecule_field.name
            for molecule_field in fields(MutagenesisMolecule)
            if molecule_field.name not in self.part_fields
        )

    def scalar_column(self, field_name: str) -> str:
        """
        :param field_name: A molecule scalar field.
        :return: The column, and EQL variable, name of that field.
        """
        return get_class_and_attribute_name(MutagenesisMolecule.__name__, field_name)

    def part_column(self, part: PartAttribute) -> str:
        """
        :param part: One part's attribute.
        :return: The EQL variable name of that attribute; the flat table has no such
            column.
        """
        return self.scalar_column(f"{part.part_field}[{part.index}].{part.attribute}")

    def part_attribute(self, variable_name: str) -> Optional[PartAttribute]:
        """
        :param variable_name: A variable name, as EQL names it.
        :return: The part attribute it names, or ``None`` if it names a molecule-level
            variable.
        """
        pattern = re.compile(
            rf"^{re.escape(MutagenesisMolecule.__name__)}\."
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
            MutagenesisMoleculeAggregations.__name__, f"{statistic_name}()"
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
    Molecules as one row each: the molecule's own scalars and its aggregation counts.
    """

    schema: MoleculeSchema = field(default_factory=MoleculeSchema)
    """
    How the columns are named.
    """

    @property
    def columns(self) -> List[str]:
        """
        The table's columns, in order.
        """
        return [
            self.schema.scalar_column(name) for name in self.schema.scalar_fields
        ] + list(self.schema.aggregation_columns)

    def row(self, molecule: MutagenesisMolecule) -> Dict[str, Any]:
        """
        :param molecule: The molecule to flatten.
        :return: The molecule's values, keyed by column.
        """
        aggregations = MutagenesisMoleculeAggregations(instance=molecule)
        molecule_values = vars(molecule)
        row = {
            self.schema.scalar_column(name): molecule_values[name]
            for name in self.schema.scalar_fields
        }
        row.update(
            {
                self.schema.aggregation_column(statistic.__name__): statistic(
                    aggregations
                )
                for statistic in self.schema.aggregation_statistics
            }
        )
        return row

    def dataframe(self, molecules: Iterable[MutagenesisMolecule]) -> pd.DataFrame:
        """
        :param molecules: The molecules to flatten.
        :return: One row per molecule, columns in :attr:`columns` order.
        """
        return pd.DataFrame(
            [self.row(molecule) for molecule in molecules], columns=self.columns
        )
