"""
Checking, on the training data, the assumption a relational circuit makes about its
causal structure: an instance's attributes reach its parts' attributes only through the
aggregation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from typing_extensions import Any

from probabilistic_model.causal_structure.discretization import Discretization
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
    TrainingTableColumn,
)

# %% one dependence the assumption does not allow


@dataclass(frozen=True)
class ExampleToPartDependence:
    """
    An instance attribute that the data says depends on a part attribute even given the
    aggregation statistics.
    """

    example_attribute: str
    """
    The name of the instance attribute.
    """

    part_attribute: str
    """
    The name of the part attribute, as its part's table names it.
    """

    part: str
    """
    The exchangeable part the attribute belongs to.
    """

    p_value: float
    """
    The permutation p-value of the dependence.
    """


# %% the tables the check reads


@dataclass
class CodedTables:
    """
    An instance table and one of its part tables, cut into levels.
    """

    example_codes: pd.DataFrame
    """
    One row of levels per instance, over its attributes and statistics.
    """

    part_codes: pd.DataFrame
    """
    One row of levels per part, over the part's own attributes.
    """

    positions: np.ndarray
    """
    The position of each part's instance among the instances.
    """

    statistics: list[str]
    """
    The names of the statistics both tables carry.
    """

    @property
    def varying_example_attributes(self) -> list[str]:
        """
        :return: The instance attributes, other than the statistics, that take more than
            one level.
        """
        return [
            column
            for column in self.example_codes.columns
            if column not in self.statistics
            and self.example_codes[column].nunique() > 1
        ]

    @property
    def varying_part_attributes(self) -> list[str]:
        """
        :return: The part attributes that take more than one level.
        """
        return [
            column
            for column in self.part_codes.columns
            if self.part_codes[column].nunique() > 1
        ]

    def strata_given(self, conditioning: list[str]) -> np.ndarray:
        """
        :param conditioning: Names of instance columns.
        :return: One integer per instance, the same for instances whose levels agree on
            the columns.
        """
        if not conditioning:
            return np.zeros(len(self.example_codes), dtype=int)
        return pd.factorize(pd.MultiIndex.from_frame(self.example_codes[conditioning]))[
            0
        ]


# %% the check


@dataclass
class ExampleToPartCheck:
    """
    Tests every instance attribute against every part attribute given the aggregation
    statistics, with a G² statistic whose null distribution comes from permuting the
    instance attribute across whole instances that share the statistics' values.

    The parts of one instance are not independent of each other, so permuting rows
    instead would understate the p-values.
    """

    significance_level: float = 0.05
    """
    The level below which a p-value rejects independence.
    """

    permutation_count: int = 199
    """
    How many permutations the null distribution is drawn from.
    """

    discretization: Discretization = field(default_factory=Discretization)
    """
    How the columns are cut into levels.
    """

    random_seed: int = 0
    """
    Seed of the permutations.
    """

    def rejected_pairs_of_circuit(
        self, circuit: RelationalProbabilisticCircuit, instances: list[Any]
    ) -> list[ExampleToPartDependence]:
        """
        :param circuit: A fitted relational circuit.
        :param instances: The instances it was fitted on.
        :return: The dependences its training tables show between an instance attribute
            and a part attribute given the statistics.
        """
        tables = circuit.training_tables(instances)
        return [
            dependence
            for part, rows in tables.part_rows.items()
            for dependence in self.rejected_pairs(tables.example_rows, rows, part)
        ]

    def rejected_pairs(
        self, example_rows: pd.DataFrame, part_rows: pd.DataFrame, part: str
    ) -> list[ExampleToPartDependence]:
        """
        :param example_rows: One row per instance.
        :param part_rows: One row per part, with the statistics its instance carries and
            the instance's position in :attr:`TrainingTableColumn.EXAMPLE_POSITION`.
        :param part: The part's name.
        :return: The dependences between an instance attribute and a part attribute
            given the statistics shared by both tables.
        """
        tables = self.coded(example_rows, part_rows)
        random_state = np.random.default_rng(self.random_seed)
        return [
            ExampleToPartDependence(example_attribute, part_attribute, part, p_value)
            for example_attribute in tables.varying_example_attributes
            for part_attribute in tables.varying_part_attributes
            if (
                p_value := self.p_value(
                    tables, example_attribute, part_attribute, [], random_state
                )
            )
            < self.significance_level
        ]

    def dependences_given_another_attribute(
        self, example_rows: pd.DataFrame, part_rows: pd.DataFrame, part: str
    ) -> list[ExampleToPartDependence]:
        """
        :param example_rows: One row per instance.
        :param part_rows: One row per part, as :meth:`rejected_pairs` takes it.
        :param part: The part's name.
        :return: The dependences of :meth:`rejected_pairs` that remain given the
            statistics and any one other instance attribute that also depends on the
            part attribute, each with the largest p-value it was tested with.
        """
        rejected = self.rejected_pairs(example_rows, part_rows, part)
        tables = self.coded(example_rows, part_rows)
        random_state = np.random.default_rng(self.random_seed)
        remaining = []
        for dependence in rejected:
            p_values = [dependence.p_value] + [
                self.p_value(
                    tables,
                    dependence.example_attribute,
                    dependence.part_attribute,
                    [other.example_attribute],
                    random_state,
                )
                for other in rejected
                if other.part_attribute == dependence.part_attribute
                and other.example_attribute != dependence.example_attribute
            ]
            if max(p_values) < self.significance_level:
                remaining.append(replace(dependence, p_value=max(p_values)))
        return remaining

    def coded(self, example_rows: pd.DataFrame, part_rows: pd.DataFrame) -> CodedTables:
        """
        :param example_rows: One row per instance.
        :param part_rows: One row per part, as :meth:`rejected_pairs` takes it.
        :return: Both tables cut into levels.
        """
        statistics = [
            column for column in part_rows.columns if column in example_rows.columns
        ]
        example_attributes = [
            column for column in example_rows.columns if column not in statistics
        ]
        part_attributes = [
            column
            for column in part_rows.columns
            if column not in statistics
            and column != TrainingTableColumn.EXAMPLE_POSITION
        ]
        return CodedTables(
            example_codes=self.discretization.apply(
                example_rows, example_attributes + statistics
            ),
            part_codes=self.discretization.apply(part_rows, part_attributes),
            positions=part_rows[TrainingTableColumn.EXAMPLE_POSITION].to_numpy(),
            statistics=statistics,
        )

    def p_value(
        self,
        tables: CodedTables,
        example_attribute: str,
        part_attribute: str,
        conditioning: list[str],
        random_state: np.random.Generator,
    ) -> float:
        """
        :param tables: The coded tables.
        :param example_attribute: The instance attribute.
        :param part_attribute: The part attribute.
        :param conditioning: The instance attributes conditioned on besides the
            statistics.
        :param random_state: Source of the permutations.
        :return: The permutation p-value of the two's dependence given the statistics
            and the conditioning attributes.
        """
        strata = tables.strata_given(tables.statistics + conditioning)
        values = tables.example_codes[example_attribute].to_numpy()
        part_values = tables.part_codes[part_attribute].to_numpy()
        part_strata = strata[tables.positions]
        observed = self.g_squared(values[tables.positions], part_values, part_strata)
        exceeding = sum(
            self.g_squared(
                self.permuted_within(values, strata, random_state)[tables.positions],
                part_values,
                part_strata,
            )
            >= observed
            for _ in range(self.permutation_count)
        )
        return (1 + exceeding) / (1 + self.permutation_count)

    @staticmethod
    def permuted_within(
        values: np.ndarray, strata: np.ndarray, random_state: np.random.Generator
    ) -> np.ndarray:
        """
        :param values: One value per instance.
        :param strata: One stratum per instance.
        :param random_state: Source of randomness.
        :return: The values shuffled among the instances of each stratum.
        """
        permuted = values.copy()
        for stratum in np.unique(strata):
            members = np.flatnonzero(strata == stratum)
            permuted[members] = values[random_state.permutation(members)]
        return permuted

    @staticmethod
    def g_squared(first: np.ndarray, second: np.ndarray, strata: np.ndarray) -> float:
        """
        :param first: Integer levels.
        :param second: Integer levels.
        :param strata: Integer strata.
        :return: The G² statistic of the two given the strata, summed over the strata.
        """
        statistic = 0.0
        for stratum in np.unique(strata):
            members = strata == stratum
            observed = pd.crosstab(first[members], second[members]).to_numpy(float)
            expected = (
                observed.sum(axis=1, keepdims=True)
                * observed.sum(axis=0, keepdims=True)
                / observed.sum()
            )
            nonzero = observed > 0
            statistic += 2 * float(
                (
                    observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
                ).sum()
            )
        return statistic
