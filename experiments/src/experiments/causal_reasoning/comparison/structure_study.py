"""
How well a causal graph learned from the training rows chooses what a causal question
adjusts for, against the adjustment set the question declares by hand and against no
adjustment at all, scored against a synthetic model's true interventional
probabilities and its true graph.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import networkx
import numpy as np
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.causal_structure_registries import (
    DictCausalStructureRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.causal_structure.background_knowledge import (
    BackgroundKnowledge,
    DeterministicCopy,
)
from probabilistic_model.causal_structure.exceptions import InvalidAdjustmentSet
from krrood.entity_query_language.exceptions import (
    EffectNotIdentified,
    VariableNotInCausalGraph,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import AdjustmentCheck
from probabilistic_model.causal_structure.learning import (
    LearnedCausalStructure,
    RelationalStructureLearner,
)
from random_events.variable import Continuous
from typing_extensions import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from experiments.causal_reasoning.comparison.domain import RelationalDomain
from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    KnownTruth,
    Refusal,
    _effect_probability_in,
    _ordinal,
    _support_count,
    describe_region,
)
from experiments.causal_reasoning.comparison.pipelines import RelationalPipeline
from experiments.causal_reasoning.comparison.queries import CausalQueryCase

# %% what the study knows about each domain


class KnowsItsGraph(ABC):
    """
    A model of the domain whose causal graph over the recorded variables is known.
    """

    @abstractmethod
    def true_edges(self, configuration: Hashable) -> List[Tuple[str, str]]:
        """
        :param configuration: One setting of the model.
        :return: The directed edges of its causal graph, each as cause and effect, over
            the variables as a learned graph names them: example attributes and
            aggregation statistics by their query names, part attributes by their
            template name ``Class.collection.attribute``.
        """


def background_knowledge_of(
    domain: RelationalDomain,
    first: Sequence[str] = (),
    outcome_fields: Sequence[str] = (),
    derived_fields: Optional[Dict[str, str]] = None,
) -> BackgroundKnowledge:
    """
    The knowledge a domain states, in the learned graph's names: the given variables
    are settled before everything else; the effect, the example's other outcomes and
    the parts' outcome attributes are recorded after everything else; and a derived
    field or part attribute only copies its source.

    :param domain: The domain.
    :param first: The variables settled before every other, by their graph names.
    :param outcome_fields: The example's fields recorded after its outcome, besides the
        effect.
    :param derived_fields: The example's fields that only copy another field, mapped to
        that field.
    :return: The knowledge.
    """
    owner = domain.example_class.__name__
    derived_fields = derived_fields or {}
    statistics = [
        f"{domain.aggregation_class.__name__}.{function.__name__}()"
        for part_field in domain.part_fields
        for function in domain.aggregation_class.aggregation_registry[part_field]
    ]
    late = {f"{owner}.{name}" for name in (domain.effect_field, *outcome_fields)}
    late |= {
        f"{owner}.{part_field}.{attribute}"
        for part_field, attributes in domain.outcome_part_attributes.items()
        for attribute in attributes
    }
    every_name = (
        [f"{owner}.{name}" for name in domain.scalar_fields]
        + statistics
        + [
            f"{owner}.{part_field}.{attribute}"
            for part_field in domain.part_fields
            for attribute in domain.part_attribute_types(part_field)
        ]
    )
    early = set(first)
    return BackgroundKnowledge(
        tiers=[
            sorted(early),
            [name for name in every_name if name not in late and name not in early],
            sorted(late),
        ],
        deterministic_copies=[
            DeterministicCopy(
                copy=f"{owner}.{part_field}.{computed}",
                source=f"{owner}.{part_field}.{source}",
            )
            for part_field, derived in domain.derived_part_attributes.items()
            for computed, source in derived.items()
        ]
        + [
            DeterministicCopy(copy=f"{owner}.{copy}", source=f"{owner}.{source}")
            for copy, source in derived_fields.items()
        ],
    )


# %% one question answered three ways


@dataclass
class RegionAnswer:
    """
    One cause region's effect probability under each adjustment, and the truth.
    """

    cause_region: str
    """
    The region, written out.
    """

    support_count: int
    """
    How many training rows the region holds.
    """

    true_probability: float
    """
    The model's interventional probability.
    """

    declared: Optional[float]
    """
    The answer adjusting for the declared set; ``None`` if the declared arm refused.
    """

    learned: Optional[float]
    """
    The answer adjusting for the learned set; ``None`` if the learned arm refused.
    """

    unadjusted: float
    """
    The answer adjusting for nothing.
    """


@dataclass
class QuestionResult:
    """
    One question, answered on one fit, three ways.
    """

    configuration: Hashable
    """
    The model setting the examples were sampled under.
    """

    example_count: int
    """
    How many examples the pipeline was fitted on.
    """

    repeat: int
    """
    Which repeat of the sampling this is.
    """

    case_name: str
    """
    The question.
    """

    declared_set: List[str]
    """
    The variables the question declares, by name.
    """

    learned_set: Optional[List[str]] = None
    """
    The variables the learned graph chose, by name; ``None`` if it refused.
    """

    learned_refusal: Optional[str] = None
    """
    Why the learned arm gave no answer, if it did not.
    """

    declared_refusal: Optional[str] = None
    """
    Why the declared arm gave no answer, if it did not.
    """

    declared_valid: Optional[bool] = None
    """
    Whether the declared set meets the adjustment criterion in the true graph.
    """

    learned_valid: Optional[bool] = None
    """
    Whether the learned set meets the adjustment criterion in the true graph.
    """

    regions: List[RegionAnswer] = field(default_factory=list)
    """
    The supported regions' answers.
    """

    seconds_to_choose: float = 0.0
    """
    How long choosing the learned set took.
    """


@dataclass
class StructureScore:
    """
    How close one learned graph is to the true one over the example's own attributes
    and statistics.
    """

    configuration: Hashable
    """
    The model setting.
    """

    example_count: int
    """
    How many examples the graph was learned from.
    """

    repeat: int
    """
    Which repeat this is.
    """

    structural_hamming_distance: int
    """
    How many pairs of nodes the learned and the true graph join differently.
    """

    adjacency_precision: float
    """
    The share of learned adjacencies that are true.
    """

    adjacency_recall: float
    """
    The share of true adjacencies that were learned.
    """

    orientation_precision: float
    """
    The share of learned directed edges that point the true way.
    """

    seconds_to_learn: float
    """
    How long learning the graph took.
    """

    learned_edges: List[str]
    """
    The learned graph's edges over the example's attributes, written out.
    """


@dataclass
class StructureStudyReport:
    """
    Every question answered and every graph scored.
    """

    questions: List[QuestionResult] = field(default_factory=list)
    """
    The questions.
    """

    structures: List[StructureScore] = field(default_factory=list)
    """
    The learned graphs.
    """

    descriptions: Dict[Hashable, Dict[str, str]] = field(default_factory=dict)
    """
    Each model setting's parameters, by name.
    """


# %% the study


@dataclass
class StructureStudy:
    """
    Fits the relational circuit on examples sampled from a model with known truth,
    learns a causal graph from the same fit, and answers every question three ways.
    """

    domain: RelationalDomain
    """
    The example and its parts.
    """

    truth: KnownTruth
    """
    The model and its settings, which also knows its graph.
    """

    cases: Sequence[CausalQueryCase]
    """
    The questions.
    """

    knowledge: BackgroundKnowledge
    """
    What the learner is told in advance.
    """

    example_counts: Sequence[int] = (400, 1000)
    """
    How many examples each fit draws.
    """

    repeat_count: int = 3
    """
    How many independent samples are drawn per setting and size.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region may hold for its answer to be scored.
    """

    random_seed: int = 0
    """
    Seed of the sampling and of the Monte-Carlo grounding.
    """

    def run(self, log=print) -> StructureStudyReport:
        """
        :param log: Where progress is reported.
        :return: The study.
        """
        report = StructureStudyReport()
        for configuration in self.truth.configurations:
            report.descriptions[configuration] = self.truth.describe(configuration)
            true_dag = self.true_dag(configuration)
            for example_count in self.example_counts:
                for repeat in range(self.repeat_count):
                    log(f"{configuration} n={example_count} repeat {repeat}")
                    self.run_one(report, configuration, true_dag, example_count, repeat)
        return report

    def true_dag(self, configuration: Hashable) -> networkx.DiGraph:
        """
        :param configuration: One setting.
        :return: The model's true graph.
        """
        return networkx.DiGraph(self.truth.true_edges(configuration))

    def run_one(
        self,
        report: StructureStudyReport,
        configuration: Hashable,
        true_dag: networkx.DiGraph,
        example_count: int,
        repeat: int,
    ) -> None:
        """
        Fit, learn and ask once.
        """
        examples = self.truth.examples(
            configuration,
            example_count,
            np.random.default_rng([self.random_seed, example_count, repeat]),
        )
        pipeline = RelationalPipeline(domain=self.domain)
        pipeline.fit(examples)
        started = time.perf_counter()
        structure = RelationalStructureLearner().learn(
            pipeline.plain_model, pipeline._training_rows(), self.knowledge
        )
        seconds_to_learn = time.perf_counter() - started
        report.structures.append(
            self.score_structure(
                structure.selector.graph,
                true_dag,
                configuration,
                example_count,
                repeat,
                seconds_to_learn,
            )
        )
        for case in self.cases:
            report.questions.append(
                self.ask(
                    pipeline,
                    structure,
                    case,
                    configuration,
                    true_dag,
                    example_count,
                    repeat,
                )
            )

    def ask(
        self,
        pipeline: RelationalPipeline,
        structure: LearnedCausalStructure,
        case: CausalQueryCase,
        configuration: Hashable,
        true_dag: networkx.DiGraph,
        example_count: int,
        repeat: int,
    ) -> QuestionResult:
        """
        Answer one question with the declared set, the learned set and no set.
        """
        graph = structure.selector.graph
        declared_parameters = UnderspecifiedParameters(case.build())
        result = QuestionResult(
            configuration=configuration,
            example_count=example_count,
            repeat=repeat,
            case_name=case.name,
            declared_set=[
                variable.name
                for variable in declared_parameters.search_confounder_variables
            ],
        )
        [cause] = declared_parameters.search_cause_variables
        [effect] = declared_parameters.effect_variables_from_causes_effect
        result.declared_valid = self.valid_in_truth(
            graph, true_dag, cause.name, effect.name, result.declared_set
        )
        declared = self.answers(
            pipeline,
            case,
            declared_parameters,
            declared_parameters.search_confounder_variables,
        )
        if isinstance(declared, str):
            result.declared_refusal = declared
            declared = {}
        learned_parameters = UnderspecifiedParameters(case.build())
        learned_parameters.search_confounder_variables.clear()
        backend = ProbabilisticBackend(
            model_registry=pipeline.registry,
            causal_structure_registry=DictCausalStructureRegistry(
                {self.domain.example_class: structure}
            ),
        )
        started = time.perf_counter()
        choice = self.call(
            lambda: backend.choose_adjustment_sets(learned_parameters, case.build())
        )
        result.seconds_to_choose = time.perf_counter() - started
        learned: Dict[str, InterventionalEffect] | str
        if isinstance(choice, str):
            result.learned_refusal = choice
            learned = {}
        else:
            [learned_cause] = learned_parameters.search_cause_variables
            adjustment = learned_parameters.adjustment_variables_of(learned_cause)
            result.learned_set = [variable.name for variable in adjustment]
            result.learned_valid = self.valid_in_truth(
                graph, true_dag, cause.name, effect.name, result.learned_set
            )
            learned = self.answers(pipeline, case, learned_parameters, adjustment)
            if isinstance(learned, str):
                result.learned_refusal = learned
                learned = {}
        unadjusted = self.answers(
            pipeline, case, declared_parameters, [], adjusted=False
        )
        if isinstance(unadjusted, str):
            return result
        for region, effect_answer in unadjusted.items():
            if effect_answer.support_count < self.min_region_support:
                continue
            result.regions.append(
                RegionAnswer(
                    cause_region=region,
                    support_count=effect_answer.support_count,
                    true_probability=self.truth.probability(
                        configuration, case, effect_answer
                    ),
                    declared=(
                        declared[region].adjusted_probability
                        if region in declared
                        else None
                    ),
                    learned=(
                        learned[region].adjusted_probability
                        if region in learned
                        else None
                    ),
                    unadjusted=effect_answer.naive_probability,
                )
            )
        return result

    def answers(
        self,
        pipeline: RelationalPipeline,
        case: CausalQueryCase,
        parameters: UnderspecifiedParameters,
        adjustment: List[Any],
        adjusted: bool = True,
    ) -> Dict[str, InterventionalEffect] | str:
        """
        :return: Per cause region, the effect's probability adjusting for the given
            set, or why there is none.
        """
        np.random.seed(self.random_seed)
        outcome = self.call(lambda: pipeline.registry.get_model(parameters))
        if isinstance(outcome, str):
            return outcome
        causal_circuit = outcome
        [cause] = causal_circuit.causal_variables
        [effect] = causal_circuit.effect_variables
        effect_event = parameters.truncation_assignments_from_where_conditions
        interventional = self.call(
            lambda: causal_circuit.backdoor_adjustment(
                cause, effect, adjustment_variables=adjustment if adjusted else []
            )
        )
        if isinstance(interventional, str):
            return interventional
        training_values = pipeline.training_values_of(cause.name)
        answers = {}
        for region in causal_circuit._extract_disjoint_regions_for_variable(cause):
            probability = _effect_probability_in(
                interventional, region.event, effect_event
            )
            answers[describe_region(region.event, cause)] = InterventionalEffect(
                cause_region=describe_region(region.event, cause),
                region_probability=region.probability,
                naive_probability=probability,
                adjusted_probability=probability,
                support_count=_support_count(region.event, cause, training_values),
                ordinal=_ordinal(region.event, cause),
            )
        return answers

    @staticmethod
    def call(action: Callable[[], Any]) -> Any:
        """
        :param action: What to do.
        :return: Its result, or the name of the refusal it raised: one the comparison
            already reads as a refusal, or the learned structure's.
        """
        try:
            return action()
        except Refusal.exception_types() + (
            EffectNotIdentified,
            VariableNotInCausalGraph,
            InvalidAdjustmentSet,
        ) as refusal:
            return type(refusal).__name__

    @staticmethod
    def valid_in_truth(
        graph: CausalGraph,
        true_dag: networkx.DiGraph,
        cause_name: str,
        effect_name: str,
        adjustment: List[str],
    ) -> Optional[bool]:
        """
        :return: Whether adjusting for the named variables identifies the effect in the
            true graph; ``None`` if a name has no node in it.
        """
        names = []
        for name in [cause_name, effect_name, *adjustment]:
            reference = graph.reference_of(name)
            if reference is None or reference.node.name not in true_dag:
                return None
            names.append(reference.node.name)
        cause, effect, *rest = names
        return (
            AdjustmentCheck(true_dag, cause, effect).failure_reason(frozenset(rest))
            is None
        )

    @staticmethod
    def score_structure(
        learned: CausalGraph,
        true_dag: networkx.DiGraph,
        configuration: Hashable,
        example_count: int,
        repeat: int,
        seconds_to_learn: float,
    ) -> StructureScore:
        """
        :return: How close the learned graph is to the true one, over the example's own
            attributes and statistics.
        """
        names = sorted(
            node.name
            for node in learned.nodes
            if node.level is not NodeLevel.PART_ATTRIBUTE and node.name in true_dag
        )
        kept = set(names)
        true_graph = CausalGraph(
            nodes=[
                GraphNode(Continuous(name), NodeLevel.EXAMPLE_ATTRIBUTE)
                for name in names
            ],
            edges=[
                GraphEdge.directed(parent, child)
                for parent, child in true_dag.edges
                if parent in kept and child in kept
            ],
        )
        learned_graph = CausalGraph(
            nodes=true_graph.nodes,
            edges=[
                edge
                for edge in learned.edges
                if edge.first in kept and edge.second in kept
            ],
        )
        true_pairs = {frozenset((edge.first, edge.second)) for edge in true_graph.edges}
        learned_pairs = {
            frozenset((edge.first, edge.second)) for edge in learned_graph.edges
        }
        directed = [edge for edge in learned_graph.edges if edge.is_directed]
        return StructureScore(
            configuration=configuration,
            example_count=example_count,
            repeat=repeat,
            structural_hamming_distance=learned_graph.structural_hamming_distance(
                true_graph
            ),
            adjacency_precision=(
                len(true_pairs & learned_pairs) / len(learned_pairs)
                if learned_pairs
                else float("nan")
            ),
            adjacency_recall=(
                len(true_pairs & learned_pairs) / len(true_pairs)
                if true_pairs
                else float("nan")
            ),
            orientation_precision=(
                sum(true_dag.has_edge(edge.parent, edge.child) for edge in directed)
                / len(directed)
                if directed
                else float("nan")
            ),
            seconds_to_learn=seconds_to_learn,
            learned_edges=sorted(
                f"{edge.first} {edge.mark_at_first.name}-{edge.mark_at_second.name} {edge.second}"
                for edge in learned_graph.edges
            ),
        )
