"""
Checking, on the training data, the assumption a relational circuit makes about its
causal structure: an instance's attributes reach its parts' attributes only through the
aggregation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        example_codes = self.discretization.apply(
            example_rows, example_attributes + statistics
        )
        part_codes = self.discretization.apply(part_rows, part_attributes)
        positions = part_rows[TrainingTableColumn.EXAMPLE_POSITION].to_numpy()
        strata = (
            self.strata_of(example_codes[statistics])
            if statistics
            else np.zeros(len(example_rows), dtype=int)
        )
        random_state = np.random.default_rng(self.random_seed)
        rejected = []
        for example_attribute in example_attributes:
            if example_codes[example_attribute].nunique() < 2:
                continue
            values = example_codes[example_attribute].to_numpy()
            permutations = [
                self.permuted_within(values, strata, random_state)
                for _ in range(self.permutation_count)
            ]
            for part_attribute in part_attributes:
                part_values = part_codes[part_attribute].to_numpy()
                if len(np.unique(part_values)) < 2:
                    continue
                observed = self.g_squared(
                    values[positions], part_values, strata[positions]
                )
                exceeding = sum(
                    self.g_squared(permuted[positions], part_values, strata[positions])
                    >= observed
                    for permuted in permutations
                )
                p_value = (1 + exceeding) / (1 + self.permutation_count)
                if p_value < self.significance_level:
                    rejected.append(
                        ExampleToPartDependence(
                            example_attribute, part_attribute, part, p_value
                        )
                    )
        return rejected

    @staticmethod
    def strata_of(codes: pd.DataFrame) -> np.ndarray:
        """
        :param codes: Integer columns.
        :return: One integer per row, the same for rows with the same values.
        """
        return pd.factorize(pd.MultiIndex.from_frame(codes))[0]

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
