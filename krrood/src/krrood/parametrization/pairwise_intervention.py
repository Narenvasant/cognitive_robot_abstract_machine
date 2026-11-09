"""
Answering an intervention on a relation between two parts, or on one part, whose effect
sits on another part or on the whole instance, through the one statistic the
intervention changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from random_events.product_algebra import SimpleEvent
from typing_extensions import Any, Callable

from krrood.entity_query_language.query.match import Match
from krrood.parametrization.exceptions import (
    ExposureInputNotEvidenceError,
    InterventionReachesEffectThroughSeveralStatisticsError,
)
from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.parametrization.feature_extraction.exposures import exposure_class_of
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)

# %% the question


@dataclass
class PartOfInstance:
    """
    One part of an instance, by its collection and position.
    """

    collection_name: str
    """
    The collection of the instance holding the part.
    """

    position: int
    """
    The position of the part in the collection.
    """


@dataclass
class PairwiseQuestion:
    """
    What happens to an attribute of one part, or of the instance, when a relation
    between two parts or an attribute of another part is set.
    """

    query_with_cause: Callable[[Any], Match]
    """
    Builds the unresolved query of the instance with the cause set to the given value,
    or left open when given ``...``.
    """

    cause_values: list[Any]
    """
    The values the cause is set to.
    """

    effect_attribute: str
    """
    The attribute whose value the question reads.
    """

    effect_value: Any
    """
    The value of the effect whose probability the question asks for.
    """

    effect_part: PartOfInstance | None = None
    """
    The part the effect sits on, or ``None`` for the instance itself.
    """

    adjustment_names: list[str] = field(default_factory=list)
    """
    The names of the variables to adjust for.
    """

    evidence: dict[str, Any] = field(default_factory=dict)
    """
    The values the query gives, by the name of their variable, that the answer is
    conditioned on.
    """


@dataclass
class PairwiseInterventionAnswer:
    """
    The answer at one value of the cause.
    """

    cause_value: Any
    """
    The value the cause is set to.
    """

    statistic_value: Any
    """
    The value it gives the statistic it reaches the effect through.
    """

    probability: float
    """
    The probability of the effect under the intervention.
    """


@dataclass
class PairwiseInterventionResult:
    """
    The answers of a pairwise question at every value of its cause.
    """

    statistic_name: str | None
    """
    The variable of the statistic the cause reaches the effect through, or ``None`` if
    it reaches the effect through none.
    """

    answers: list[PairwiseInterventionAnswer]
    """
    The answer at every value of the cause.
    """


# %% answering it


@dataclass
class PairwiseInterventionResolver:
    """
    Maps every value of a question's cause to the statistic it reaches the effect
    through, grounds the circuit with that statistic open, and adjusts for the
    question's adjustment variables on the grounded circuit.
    """

    relational_circuit: RelationalProbabilisticCircuit
    """
    The fitted circuit.
    """

    def resolve(self, question: PairwiseQuestion) -> PairwiseInterventionResult:
        """
        :param question: The question.
        :return: The probability of the effect at every value of the cause.
        :raises InterventionReachesEffectThroughSeveralStatisticsError: If the cause
            changes more than one statistic the effect depends on.
        :raises ExposureInputNotEvidenceError: If the query leaves an input of that
            statistic other than the cause open.
        """
        statistics_by_value = [
            self.statistics_of_effect(question, question.query_with_cause(value))
            for value in question.cause_values
        ]
        changed = sorted(
            name
            for name in statistics_by_value[0]
            if len({str(statistics[name]) for statistics in statistics_by_value}) > 1
        )
        if len(changed) > 1:
            raise InterventionReachesEffectThroughSeveralStatisticsError(changed)
        open_query = question.query_with_cause(...)
        open_query.resolve()
        circuit = self.relational_circuit.ground(open_query)
        if not changed:
            probability = self.adjusted_probability(circuit, question, None, None)
            return PairwiseInterventionResult(
                None,
                [
                    PairwiseInterventionAnswer(value, None, probability)
                    for value in question.cause_values
                ],
            )
        [name] = changed
        for statistics in statistics_by_value:
            if statistics[name] is ...:
                raise ExposureInputNotEvidenceError(name)
        variable_name = self.variable_name(question, name)
        return PairwiseInterventionResult(
            variable_name,
            [
                PairwiseInterventionAnswer(
                    value,
                    statistics[name],
                    self.adjusted_probability(
                        circuit, question, variable_name, statistics[name]
                    ),
                )
                for value, statistics in zip(question.cause_values, statistics_by_value)
            ],
        )

    def statistics_of_effect(
        self, question: PairwiseQuestion, query: Match
    ) -> dict[str, Any]:
        """
        :param question: The question.
        :param query: The query with the cause set.
        :return: The statistics the effect depends on, by name: the exposures of the
            effect's part, or the counts of the instance.
        """
        instance = query.construct_instance()
        if question.effect_part is None:
            return {
                variable.name: value
                for collection_name, template in self.relational_circuit.exchangeable_distribution_templates.items()
                for variable, value in compute_aggregation_statistics(
                    instance, collection_name, template.latent_variables
                ).items()
            }
        collection_name = question.effect_part.collection_name
        part = getattr(instance, collection_name)[question.effect_part.position]
        template = self.relational_circuit.exchangeable_distribution_templates[
            collection_name
        ]
        values = exposure_class_of(type(part))(
            role_taker=part, instance=instance
        ).values(collection_name)
        return {name: values[name] for name in template.exposure_names}

    def variable_name(self, question: PairwiseQuestion, statistic_name: str) -> str:
        """
        :param question: The question.
        :param statistic_name: The name of the statistic the cause reaches the effect
            through.
        :return: The name the grounded circuit gives the statistic.
        """
        if question.effect_part is None:
            return statistic_name
        return (
            f"{self.relational_circuit.class_.__name__}."
            f"{question.effect_part.collection_name}"
            f"[{question.effect_part.position}].{statistic_name}"
        )

    def effect_name(self, question: PairwiseQuestion) -> str:
        """
        :param question: The question.
        :return: The name the grounded circuit gives the effect.
        """
        return self.variable_name(question, question.effect_attribute)

    def adjusted_probability(
        self,
        circuit: Any,
        question: PairwiseQuestion,
        statistic_name: str | None,
        statistic_value: Any,
    ) -> float:
        """
        :param circuit: The circuit grounded with the statistic open.
        :param question: The question.
        :param statistic_name: The name of the statistic, or ``None`` to read the effect
            without an intervention.
        :param statistic_value: The value the intervention gives it.
        :return: The effect's probability with the statistic set, given the evidence,
            summed over the adjustment variables weighted by their probability given the
            evidence; strata in which the statistic never takes the value are left out
            and the rest renormalized.
        """
        variable_of = {variable.name: variable for variable in circuit.variables}
        effect = SimpleEvent.from_data(
            {variable_of[self.effect_name(question)]: question.effect_value}
        ).as_composite_set()
        intervention = (
            effect | effect.complement()
            if statistic_name is None
            else SimpleEvent.from_data(
                {variable_of[statistic_name]: statistic_value}
            ).as_composite_set()
        )
        evidence = effect | effect.complement()
        for name, value in question.evidence.items():
            evidence = (
                evidence
                & SimpleEvent.from_data({variable_of[name]: value}).as_composite_set()
            )
        adjustment = [variable_of[name] for name in question.adjustment_names]
        strata = [evidence]
        for variable in adjustment:
            strata = [
                stratum & SimpleEvent.from_data({variable: value}).as_composite_set()
                for stratum in strata
                for value in variable.domain.hash_map.values()
            ]
        total, weight = 0.0, 0.0
        for stratum in strata:
            stratum_probability = circuit.probability(stratum)
            given = circuit.probability(intervention & stratum)
            if stratum_probability <= 0 or given <= 0:
                continue
            total += (
                circuit.probability(effect & intervention & stratum)
                / given
                * stratum_probability
            )
            weight += stratum_probability
        return float(total / weight) if weight > 0 else float(np.nan)
