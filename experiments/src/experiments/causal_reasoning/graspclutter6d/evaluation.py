"""
Asking every question of the catalogue to every pipeline and writing down what each
answered, how fast, and how much model it took.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.exceptions import DoRequiresCausalCircuitModel
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    EmptyInterventionalCircuitError,
    SupportDeterminismVerificationResult,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    ClassCircuitGroundingFailedError,
    PartCircuitGroundingFailedError,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.interval import Interval
from random_events.product_algebra import Event
from random_events.variable import Variable
from scipy.stats import spearmanr
from typing_extensions import Any, Dict, List, Optional, Sequence, Tuple, Type

from experiments.causal_reasoning.graspclutter6d.dataset import (
    GraspClutterDataset,
    GraspableRate,
)
from experiments.causal_reasoning.graspclutter6d.domain import (
    GraspClutterScene,
    GraspClutterSceneAggregations,
)
from experiments.causal_reasoning.graspclutter6d.exceptions import (
    FlatTableSchemaMismatchError,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import SceneView
from experiments.causal_reasoning.graspclutter6d.pipelines import (
    CausalQueryPipeline,
    FitReport,
    LikelihoodReport,
    pipelines,
)
from experiments.causal_reasoning.graspclutter6d.queries import (
    CausalQueryCase,
    object_level_cases,
    query_catalogue,
)

# %% what one question yields on one pipeline


class Refusal(StrEnum):
    """
    Why a pipeline could not answer a question.
    """

    SCHEMA_MISMATCH = "the fitted table has no column for the queried variables"
    """
    The question constrains an object's or a viewpoint's attribute, which the flat table
    has no column for.
    """

    NOT_SUPPORT_DETERMINISTIC = "the model is not support-deterministic over the cause"
    """
    The fitted circuit mixes branches that overlap on the cause, so backdoor adjustment
    has no disjoint regions to intervene on.
    """

    GROUNDING_FAILED = "grounding the query left no circuit"
    """
    Conditioning the relational circuit on the query emptied it.
    """

    NO_INTERVENTION_REGION = "no cause region carries probability"
    """
    Every region of the cause truncates the circuit to nothing.
    """

    NOT_A_CAUSAL_MODEL = "the registry returned no causal circuit"
    """
    The pipeline served a plain circuit for a question that marks a cause.
    """

    EFFECT_NEVER_OCCURS = "the effect has zero probability under every cause region"
    """
    No region of the cause gives the effect any probability, so the search over
    interventions has nothing to rank.
    """

    OVERLAPPING_REGIONS = "the cause regions read off the model overlap"
    """
    The regions of the cause the model distinguishes carry more probability together than
    one, so they overlap instead of partitioning the cause, and the effect read off each
    of them is not an interventional probability.
    """

    @classmethod
    def by_exception(cls) -> Dict[Type[Exception], Refusal]:
        """
        :return: The exceptions a pipeline refuses a question with, and what each one
            means.
        """
        return {
            FlatTableSchemaMismatchError: cls.SCHEMA_MISMATCH,
            SupportDeterminismVerificationResult: cls.NOT_SUPPORT_DETERMINISTIC,
            ClassCircuitGroundingFailedError: cls.GROUNDING_FAILED,
            PartCircuitGroundingFailedError: cls.GROUNDING_FAILED,
            EmptyInterventionalCircuitError: cls.NO_INTERVENTION_REGION,
            DoRequiresCausalCircuitModel: cls.NOT_A_CAUSAL_MODEL,
        }

    @classmethod
    def exception_types(cls) -> Tuple[Type[Exception], ...]:
        """
        :return: Every exception a pipeline refuses a question with.
        """
        return tuple(cls.by_exception())


@dataclass(frozen=True)
class ProportionInterval:
    """
    A confidence interval for a probability read off a region of the training data.
    """

    lower: float
    """
    The lower bound.
    """

    upper: float
    """
    The upper bound.
    """

    @classmethod
    def wilson(
        cls, probability: float, count: int, z: float = 1.96
    ) -> ProportionInterval:
        """
        The Wilson score interval of a proportion over ``count`` observations.

        :param probability: The proportion.
        :param count: How many observations it is read off; zero gives the whole unit
            interval.
        :param z: The standard-normal quantile of the confidence level; 1.96 for 95%.
        :return: The interval.
        """
        if count == 0:
            return cls(0.0, 1.0)
        centre = (probability + z**2 / (2 * count)) / (1 + z**2 / count)
        half_width = (
            z
            * math.sqrt(probability * (1 - probability) / count + z**2 / (4 * count**2))
            / (1 + z**2 / count)
        )
        return cls(max(0.0, centre - half_width), min(1.0, centre + half_width))

    def difference_from(
        self, other: ProportionInterval, probability: float, other_probability: float
    ) -> ProportionInterval:
        """
        Newcombe's interval for the difference between two proportions, this one minus
        the other.

        :param other: The other proportion's interval.
        :param probability: This proportion.
        :param other_probability: The other proportion.
        :return: The interval of the difference.
        """
        difference = probability - other_probability
        return ProportionInterval(
            difference
            - math.sqrt(
                (probability - self.lower) ** 2 + (other.upper - other_probability) ** 2
            ),
            difference
            + math.sqrt(
                (self.upper - probability) ** 2 + (other_probability - other.lower) ** 2
            ),
        )


@dataclass(frozen=True)
class InterventionalEffect:
    """
    The effect's probability under an intervention setting the cause to one region.
    """

    cause_region: str
    """
    The region of the cause, written out.
    """

    region_probability: float
    """
    The region's own probability under the fitted model.
    """

    naive_probability: float
    """
    ``P(effect | cause in region)``, read off the model with no adjustment.
    """

    adjusted_probability: float
    """
    ``P(effect | do(cause in region))``, backdoor-adjusted for the question's
    confounders; equal to the naive one when the question names none.
    """

    support_count: int
    """
    How many training rows of the cause fall in the region: scenes for a scene-level
    cause, parts for a part's own attribute.
    """

    ordinal: Optional[float] = None
    """
    Where the region sits on the cause's scale, for a numeric cause; ``None`` for a
    symbolic one, whose regions have no order.
    """

    @property
    def adjusted_interval(self) -> ProportionInterval:
        """
        The Wilson interval of the adjusted probability over the region's support.
        """
        return ProportionInterval.wilson(self.adjusted_probability, self.support_count)

    @property
    def naive_interval(self) -> ProportionInterval:
        """
        The Wilson interval of the naive probability over the region's support.
        """
        return ProportionInterval.wilson(self.naive_probability, self.support_count)

    def is_supported(self, min_region_support: int) -> bool:
        """
        :param min_region_support: The fewest training rows a region may hold to be
            read.
        :return: Whether the region holds at least that many.
        """
        return self.support_count >= min_region_support


@dataclass(frozen=True)
class Contrast:
    """
    The adjusted probability at the high end of the cause against the low end.
    """

    high_region: str
    """
    The region at the high end, written out.
    """

    low_region: str
    """
    The region at the low end, written out.
    """

    difference: float
    """
    The adjusted probability at the high end minus at the low end.
    """

    interval: ProportionInterval
    """
    Newcombe's interval of the difference.
    """


@dataclass
class QueryOutcome:
    """
    What one pipeline made of one question.
    """

    case: CausalQueryCase
    """
    The question.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    duration: float
    """
    Wall-clock seconds from asking to the answer or the refusal, the first time the
    question was asked, including the fit of the cause-specific model if that cause had
    not been asked about before.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region may hold for its effect to be read as an
    answer; a region below it is reported but takes no part in the summary.
    """

    repeat_duration: float = float("nan")
    """
    Wall-clock seconds the same question took asked again, with every model fitted:
    grounding, verification and adjustment alone.
    """

    refusal: Optional[Refusal] = None
    """
    Why the pipeline could not answer, or ``None`` if it did.
    """

    best_region: Optional[str] = None
    """
    The cause region EQL's own search settles on, written out: the region most probable
    once the effect is required to hold, which is the one the query's samples are drawn
    from.
    """

    effect_probability_given_best_region: Optional[float] = None
    """
    ``P(effect | do(cause in best region))``.
    """

    effects: List[InterventionalEffect] = field(default_factory=list)
    """
    The effect's probability under every cause region the model distinguishes.
    """

    @property
    def answered(self) -> bool:
        """
        Whether the pipeline answered the question.
        """
        return self.refusal is None

    @property
    def supported_effects(self) -> List[InterventionalEffect]:
        """
        The effects read off regions holding at least :attr:`min_region_support`
        training rows.
        """
        return [
            effect
            for effect in self.effects
            if effect.is_supported(self.min_region_support)
        ]

    @property
    def most_effective(self) -> Optional[InterventionalEffect]:
        """
        The supported cause region whose intervention gives the effect the highest
        adjusted probability, or ``None`` if the question was refused or no region is
        supported.
        """
        supported = self.supported_effects
        if not supported:
            return None
        return max(supported, key=lambda effect: effect.adjusted_probability)

    @property
    def least_effective(self) -> Optional[InterventionalEffect]:
        """
        The supported cause region whose intervention gives the effect the lowest
        adjusted probability, or ``None`` if the question was refused or no region is
        supported.
        """
        supported = self.supported_effects
        if not supported:
            return None
        return min(supported, key=lambda effect: effect.adjusted_probability)

    @property
    def trend(self) -> Optional[float]:
        """
        Spearman's rank correlation between the cause's value and the adjusted
        probability over the supported regions; ``None`` for a symbolic cause, whose
        regions have no order, or with fewer than three supported regions.
        """
        ordered = [
            effect for effect in self.supported_effects if effect.ordinal is not None
        ]
        if len(ordered) < 3:
            return None
        correlation = spearmanr(
            [effect.ordinal for effect in ordered],
            [effect.adjusted_probability for effect in ordered],
        ).statistic
        return None if math.isnan(correlation) else float(correlation)

    @property
    def contrast(self) -> Optional[Contrast]:
        """
        The adjusted probability at the highest supported region of a numeric cause
        against the lowest, or at the most effective supported region of a symbolic
        cause against the least; ``None`` with fewer than two supported regions.
        """
        supported = self.supported_effects
        if len(supported) < 2:
            return None
        if all(effect.ordinal is not None for effect in supported):
            low = min(supported, key=lambda effect: effect.ordinal)
            high = max(supported, key=lambda effect: effect.ordinal)
        else:
            low, high = self.least_effective, self.most_effective
        return Contrast(
            high_region=high.cause_region,
            low_region=low.cause_region,
            difference=high.adjusted_probability - low.adjusted_probability,
            interval=high.adjusted_interval.difference_from(
                low.adjusted_interval,
                high.adjusted_probability,
                low.adjusted_probability,
            ),
        )


# %% asking


def describe_region(event: Event, variable: Variable) -> str:
    """
    :param event: An event restricting ``variable``.
    :param variable: The variable to describe the restriction of.
    :return: The restriction written out: a point as its value, a range as its bounds, a
        set of symbols as their names.
    """
    parts = []
    for simple_event in event.simple_sets:
        value = simple_event[variable]
        if isinstance(value, Interval):
            for interval in value.simple_sets:
                if interval.lower == interval.upper:
                    parts.append(f"{interval.lower:g}")
                else:
                    parts.append(f"[{interval.lower:g}, {interval.upper:g}]")
        else:
            parts.extend(str(element) for element in value.simple_sets)
    return " or ".join(parts)


def regions_partition_the_cause(
    effects: Sequence[InterventionalEffect], tolerance: float
) -> bool:
    """
    :param effects: The effect under every region of the cause the model distinguishes.
    :param tolerance: How far the regions' probabilities may sum away from one.
    :return: Whether the regions partition the cause, so that their probabilities sum to
        one.
    """
    return abs(sum(effect.region_probability for effect in effects) - 1) <= tolerance


@dataclass
class QuestionAsker:
    """
    Asks the questions and records the outcomes.
    """

    random_seed: int = 0
    """
    Seed applied to the global NumPy random state before each question, since relational
    grounding draws Monte-Carlo samples for open aggregation counts.
    """

    region_probability_tolerance: float = 1e-6
    """
    How far the probabilities of the cause regions may sum away from one before the
    regions count as overlapping.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region may hold for its effect to be read as an
    answer.
    """

    def ask(self, pipeline: CausalQueryPipeline, case: CausalQueryCase) -> QueryOutcome:
        """
        Ask one pipeline one question.

        :param pipeline: The pipeline to ask.
        :param case: The question.
        :return: What came of it.
        """
        backend = ProbabilisticBackend(model_registry=pipeline.registry)
        np.random.seed(self.random_seed)
        started = time.perf_counter()
        try:
            ranked = backend.rank_causes(case.build())
        except Refusal.exception_types() as refusal:
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                min_region_support=self.min_region_support,
                duration=time.perf_counter() - started,
                refusal=Refusal.by_exception()[type(refusal)],
            )
        duration = time.perf_counter() - started
        if not ranked:
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                min_region_support=self.min_region_support,
                duration=duration,
                refusal=Refusal.EFFECT_NEVER_OCCURS,
            )
        [primary] = ranked
        cause = primary.cause_variable
        outcome = QueryOutcome(
            case=case,
            pipeline_name=pipeline.name,
            min_region_support=self.min_region_support,
            duration=duration,
            best_region=describe_region(
                primary.narrowed_circuit.marginal([cause]).support, cause
            ),
            effect_probability_given_best_region=(
                primary.effect_probability_given_region
            ),
        )
        np.random.seed(self.random_seed)
        effects = self._effects(pipeline, case)
        if not regions_partition_the_cause(effects, self.region_probability_tolerance):
            return QueryOutcome(
                case=case,
                pipeline_name=pipeline.name,
                min_region_support=self.min_region_support,
                duration=duration,
                refusal=Refusal.OVERLAPPING_REGIONS,
            )
        outcome.effects = effects
        return outcome

    def time_repeat(
        self, pipeline: CausalQueryPipeline, case: CausalQueryCase
    ) -> float:
        """
        Ask a question again and time it alone.

        :param pipeline: The pipeline to ask, with every model it needs fitted.
        :param case: The question.
        :return: Wall-clock seconds to the answer or the refusal.
        """
        backend = ProbabilisticBackend(model_registry=pipeline.registry)
        np.random.seed(self.random_seed)
        started = time.perf_counter()
        try:
            backend.rank_causes(case.build())
        except Refusal.exception_types():
            pass
        return time.perf_counter() - started

    @staticmethod
    def _effects(
        pipeline: CausalQueryPipeline, case: CausalQueryCase
    ) -> List[InterventionalEffect]:
        """
        Read the effect's naive and adjusted probability off every cause region, with
        how many training rows the region holds.

        :param pipeline: The pipeline whose model to read.
        :param case: The question.
        :return: One row per region, in the model's own order.
        """
        parameters = UnderspecifiedParameters(case.build())
        causal_circuit: CausalCircuit = pipeline.registry.get_model(parameters)
        [cause] = causal_circuit.causal_variables
        [effect] = causal_circuit.effect_variables
        effect_event = parameters.truncation_assignments_from_where_conditions
        naive = causal_circuit.backdoor_adjustment(cause, effect)
        adjusted = causal_circuit.backdoor_adjustment(
            cause, effect, adjustment_variables=parameters.search_confounder_variables
        )
        training_values = pipeline.training_values_of(cause.name)
        return [
            InterventionalEffect(
                cause_region=describe_region(region.event, cause),
                region_probability=region.probability,
                naive_probability=_effect_probability_in(
                    naive, region.event, effect_event
                ),
                adjusted_probability=_effect_probability_in(
                    adjusted, region.event, effect_event
                ),
                support_count=_support_count(region.event, cause, training_values),
                ordinal=_ordinal(region.event, cause),
            )
            for region in causal_circuit._extract_disjoint_regions_for_variable(cause)
        ]


def _support_count(region: Event, variable: Variable, values: Sequence[Any]) -> int:
    """
    :param region: An event restricting ``variable``.
    :param variable: The variable the region restricts.
    :param values: The training values of that variable.
    :return: How many of the values fall in the region.
    """
    [simple_region] = region.simple_sets
    restriction = simple_region[variable]
    return sum(
        not variable.make_value(value).intersection_with(restriction).is_empty()
        for value in values
    )


def _ordinal(region: Event, variable: Variable) -> Optional[float]:
    """
    :param region: An event restricting ``variable``.
    :param variable: The variable the region restricts.
    :return: The region's lower bound for a numeric variable, ``None`` for a symbolic
        one.
    """
    [simple_region] = region.simple_sets
    restriction = simple_region[variable]
    if not isinstance(restriction, Interval):
        return None
    return float(min(interval.lower for interval in restriction.simple_sets))


def _effect_probability_in(
    interventional: ProbabilisticCircuit, region: Event, effect_event: Event
) -> float:
    """
    :param interventional: A joint circuit over the cause and the effect.
    :param region: The cause region to restrict to.
    :param effect_event: The effect's condition.
    :return: The effect condition's probability within the region.
    """
    truncated, _ = interventional.truncated(
        region.fill_missing_variables_pure(interventional.variables)
    )
    if truncated is None:
        return 0.0
    return float(
        truncated.probability(
            effect_event.fill_missing_variables_pure(truncated.variables)
        )
    )


# %% the whole comparison


@dataclass
class PipelineReport:
    """
    Everything one pipeline reported over the comparison.
    """

    name: str
    """
    The pipeline's name.
    """

    fit: FitReport
    """
    What its fits cost.
    """

    likelihoods: Dict[SceneView, Optional[LikelihoodReport]] = field(
        default_factory=dict
    )
    """
    How well its plain model explains the held-out scenes, per view; ``None`` for a view
    the pipeline models less than.
    """

    outcomes: List[QueryOutcome] = field(default_factory=list)
    """
    What it made of every question, in catalogue order.
    """


@dataclass
class EvaluationReport:
    """
    The comparison of every pipeline on one split of one dataset.
    """

    random_seed: int
    """
    The seed of the split and of the questions' Monte-Carlo grounding.
    """

    training_scene_count: int
    """
    How many scenes the pipelines were fitted on.
    """

    test_scene_count: int
    """
    How many held-out scenes they were scored on.
    """

    graspable_rate: float
    """
    Share of all scenes that leave every object graspable.
    """

    graspable_by_catalogue: Dict[str, GraspableRate] = field(default_factory=dict)
    """
    The share per object catalogue.
    """

    graspable_by_small_object_count: Dict[int, GraspableRate] = field(
        default_factory=dict
    )
    """
    The share per small-object count.
    """

    graspable_by_occluded_object_count: Dict[int, GraspableRate] = field(
        default_factory=dict
    )
    """
    The share per occluded-object count.
    """

    graspable_by_object_count: Dict[int, GraspableRate] = field(default_factory=dict)
    """
    The share per number of objects.
    """

    min_samples_per_leaf: float = 0.0
    """
    The fewest training rows a leaf of a cause-specific model was allowed to hold.
    """

    plain_min_samples_per_leaf: float = 0.0
    """
    The fewest training rows a leaf of the plain model was allowed to hold.
    """

    min_region_support: int = 10
    """
    The fewest training rows a cause region was allowed to hold for its effect to be
    read as an answer.
    """

    pipelines: List[PipelineReport] = field(default_factory=list)
    """
    One report per pipeline.
    """

    shared_coverage_log_likelihoods: Dict[SceneView, Dict[str, float]] = field(
        default_factory=dict
    )
    """
    Per view, each pipeline's mean log-likelihood over the held-out scenes every pipeline
    modelling that view covers, so the numbers are comparable, keyed by pipeline name.
    """

    def pipeline(self, name: str) -> PipelineReport:
        """
        :param name: A pipeline's name.
        :return: Its report.
        """
        [report] = [report for report in self.pipelines if report.name == name]
        return report


def shared_coverage_log_likelihoods(
    log_likelihoods: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    :param log_likelihoods: Each pipeline's log-likelihood per held-out scene, by
        pipeline name.
    :return: Each pipeline's mean over the scenes every pipeline covers.
    """
    covered_by_all = np.all(
        [np.isfinite(values) for values in log_likelihoods.values()], axis=0
    )
    return {
        name: (
            float(values[covered_by_all].mean())
            if covered_by_all.any()
            else float("nan")
        )
        for name, values in log_likelihoods.items()
    }


def configured_pipelines(
    scenes: Sequence[GraspClutterScene],
    min_samples_per_leaf: Optional[float],
    plain_min_samples_per_leaf: Optional[float],
) -> List[CausalQueryPipeline]:
    """
    :param scenes: The scenes the pipelines will be fitted on.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :return: Every pipeline, unfitted, with those settings.
    """
    configured = pipelines(scenes)
    for pipeline in configured:
        if min_samples_per_leaf is not None:
            pipeline.min_samples_per_leaf = min_samples_per_leaf
        if plain_min_samples_per_leaf is not None:
            pipeline.plain_min_samples_per_leaf = plain_min_samples_per_leaf
    return configured


def score_every_view(
    pipeline: CausalQueryPipeline, scenes: Sequence[GraspClutterScene]
) -> Dict[SceneView, Optional[LikelihoodReport]]:
    """
    :param pipeline: A fitted pipeline.
    :param scenes: The held-out scenes.
    :return: The pipeline's likelihood report per view.
    """
    return {view: pipeline.log_likelihood(scenes, view) for view in SceneView}


def evaluate(
    dataset: GraspClutterDataset,
    train_fraction: float = 0.8,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    cases: Optional[Sequence[CausalQueryCase]] = None,
    time_repeats: bool = True,
    min_region_support: int = 10,
) -> EvaluationReport:
    """
    Fit every pipeline on part of the dataset, score them on the rest, and ask them every
    question.

    :param dataset: The scenes.
    :param train_fraction: Share of scenes to fit on.
    :param random_seed: Seed of the split and of the questions' Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param cases: The questions to ask; defaults to :func:`query_catalogue`.
    :param time_repeats: Whether to ask every question a second time to time it alone.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The comparison.
    """
    training, test = dataset.split(train_fraction, np.random.default_rng(random_seed))
    cases = list(cases or query_catalogue())
    report = EvaluationReport(
        random_seed=random_seed,
        training_scene_count=len(training.scenes),
        test_scene_count=len(test.scenes),
        min_region_support=min_region_support,
        graspable_rate=dataset.graspable_rate,
        graspable_by_catalogue=dataset.graspable_rate_by(
            lambda scene: scene.object_catalogue
        ),
        graspable_by_small_object_count=dataset.graspable_rate_by(
            lambda scene: GraspClutterSceneAggregations(
                instance=scene
            ).small_object_count()
        ),
        graspable_by_occluded_object_count=dataset.graspable_rate_by(
            lambda scene: GraspClutterSceneAggregations(
                instance=scene
            ).occluded_object_count()
        ),
        graspable_by_object_count=dataset.graspable_rate_by(
            lambda scene: len(scene.objects)
        ),
    )
    asker = QuestionAsker(
        random_seed=random_seed, min_region_support=min_region_support
    )
    log_likelihoods: Dict[SceneView, Dict[str, np.ndarray]] = {
        view: {} for view in SceneView
    }
    for pipeline in configured_pipelines(
        training.scenes, min_samples_per_leaf, plain_min_samples_per_leaf
    ):
        report.min_samples_per_leaf = pipeline.min_samples_per_leaf
        report.plain_min_samples_per_leaf = pipeline.plain_min_samples_per_leaf
        fit = pipeline.fit(training.scenes)
        pipeline_report = PipelineReport(
            name=pipeline.name,
            fit=fit,
            likelihoods=score_every_view(pipeline, test.scenes),
        )
        for view, likelihood in pipeline_report.likelihoods.items():
            if likelihood is not None:
                log_likelihoods[view][pipeline.name] = likelihood.log_likelihoods
        for case in cases:
            pipeline_report.outcomes.append(asker.ask(pipeline, case))
        if time_repeats:
            for outcome in pipeline_report.outcomes:
                outcome.repeat_duration = asker.time_repeat(pipeline, outcome.case)
        report.pipelines.append(pipeline_report)
    report.shared_coverage_log_likelihoods = {
        view: shared_coverage_log_likelihoods(values)
        for view, values in log_likelihoods.items()
    }
    return report


# %% the order of the parts


@dataclass
class PermutedOutcomes:
    """
    What one pipeline made of one question over several orderings of the objects and
    viewpoints.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    case: CausalQueryCase
    """
    The question.
    """

    outcomes: List[QueryOutcome] = field(default_factory=list)
    """
    One outcome per ordering.
    """

    @property
    def answered(self) -> List[QueryOutcome]:
        """
        The outcomes that were answered.
        """
        return [outcome for outcome in self.outcomes if outcome.answered]

    @property
    def best_regions(self) -> List[str]:
        """
        The distinct most effective cause regions found over the orderings.
        """
        return sorted(
            {
                outcome.most_effective.cause_region
                for outcome in self.answered
                if outcome.most_effective is not None
            }
        )

    @property
    def largest_adjusted_difference(self) -> float:
        """
        Over the cause regions every answered ordering distinguishes, the largest
        difference between orderings in the effect's adjusted probability; ``nan`` if
        fewer than two orderings were answered.
        """
        answered = self.answered
        if len(answered) < 2:
            return float("nan")
        adjusted_by_region = [
            {
                effect.cause_region: effect.adjusted_probability
                for effect in outcome.effects
            }
            for outcome in answered
        ]
        shared_regions = set.intersection(
            *(set(by_region) for by_region in adjusted_by_region)
        )
        if not shared_regions:
            return float("nan")
        return max(
            max(by_region[region] for by_region in adjusted_by_region)
            - min(by_region[region] for by_region in adjusted_by_region)
            for region in shared_regions
        )


@dataclass
class PermutationReport:
    """
    How the pipelines' answers and likelihoods move when the objects and viewpoints of
    every scene are put in another order.
    """

    ordering_count: int
    """
    How many random orderings were tried.
    """

    questions: List[PermutedOutcomes] = field(default_factory=list)
    """
    One entry per pipeline and question.
    """

    whole_scene_likelihoods: Dict[str, List[LikelihoodReport]] = field(
        default_factory=dict
    )
    """
    Per pipeline that models whole scenes, its held-out likelihood report under each
    ordering.
    """

    def largest_likelihood_drop(
        self, pipeline_name: str, in_dataset_order: float
    ) -> float:
        """
        :param pipeline_name: A pipeline modelling whole scenes.
        :param in_dataset_order: Its mean whole-scene log-likelihood with the parts in
            the order the dataset lists them.
        :return: How far below that the worst ordering took it.
        """
        means = [
            report.mean_log_likelihood
            for report in self.whole_scene_likelihoods[pipeline_name]
        ]
        return float(in_dataset_order - np.nanmin(means))


def permutation_study(
    dataset: GraspClutterDataset,
    ordering_count: int = 3,
    train_fraction: float = 0.8,
    random_seed: int = 0,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    cases: Optional[Sequence[CausalQueryCase]] = None,
    min_region_support: int = 10,
) -> PermutationReport:
    """
    Put every scene's objects and viewpoints in a random order, several times over, and
    each time refit the pipelines that model the parts and ask them the questions about
    objects. The split is the same every time; only the order within a scene changes.

    :param dataset: The scenes.
    :param ordering_count: How many random orderings to try.
    :param train_fraction: Share of scenes to fit on.
    :param random_seed: Seed of the split, the orderings and the Monte-Carlo grounding.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param cases: The questions to ask; defaults to :func:`object_level_cases`.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The study.
    """
    training, test = dataset.split(train_fraction, np.random.default_rng(random_seed))
    cases = list(cases or object_level_cases())
    asker = QuestionAsker(
        random_seed=random_seed, min_region_support=min_region_support
    )
    report = PermutationReport(ordering_count=ordering_count)
    questions: Dict[Tuple[str, str], PermutedOutcomes] = {}
    for ordering in range(ordering_count):
        random_state = np.random.default_rng([random_seed, ordering])
        shuffled_training = training.with_shuffled_parts(random_state)
        shuffled_test = test.with_shuffled_parts(random_state)
        for pipeline in configured_pipelines(
            shuffled_training.scenes, min_samples_per_leaf, plain_min_samples_per_leaf
        ):
            if not pipeline.models_parts:
                continue
            pipeline.fit(shuffled_training.scenes)
            likelihood = pipeline.log_likelihood(
                shuffled_test.scenes, SceneView.WHOLE_SCENE
            )
            report.whole_scene_likelihoods.setdefault(pipeline.name, []).append(
                likelihood
            )
            for case in cases:
                questions.setdefault(
                    (pipeline.name, case.name),
                    PermutedOutcomes(pipeline_name=pipeline.name, case=case),
                ).outcomes.append(asker.ask(pipeline, case))
    report.questions = list(questions.values())
    return report


# %% several splits


@dataclass
class SplitReport:
    """
    The comparison repeated over several random splits of the dataset.
    """

    reports: List[EvaluationReport] = field(default_factory=list)
    """
    One comparison per split.
    """

    @property
    def pipeline_names(self) -> List[str]:
        """
        The pipelines' names, in the order they were run.
        """
        return [pipeline.name for pipeline in self.reports[0].pipelines]

    def coverage(self, pipeline_name: str, view: SceneView) -> List[float]:
        """
        :param pipeline_name: A pipeline's name.
        :param view: How much of a scene to look at.
        :return: The pipeline's held-out coverage per split.
        """
        return [
            report.pipeline(pipeline_name).likelihoods[view].coverage
            for report in self.reports
        ]

    def shared_log_likelihood(self, pipeline_name: str, view: SceneView) -> List[float]:
        """
        :param pipeline_name: A pipeline's name.
        :param view: How much of a scene to look at.
        :return: The pipeline's mean log-likelihood over the scenes every pipeline
            modelling that view covers, per split.
        """
        return [
            report.shared_coverage_log_likelihoods[view][pipeline_name]
            for report in self.reports
        ]

    def outcomes(self, pipeline_name: str, case_index: int) -> List[QueryOutcome]:
        """
        :param pipeline_name: A pipeline's name.
        :param case_index: A question's position in the catalogue.
        :return: What the pipeline made of that question, per split.
        """
        return [
            report.pipeline(pipeline_name).outcomes[case_index]
            for report in self.reports
        ]


def split_study(
    dataset: GraspClutterDataset,
    random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train_fraction: float = 0.8,
    min_samples_per_leaf: Optional[float] = None,
    plain_min_samples_per_leaf: Optional[float] = None,
    cases: Optional[Sequence[CausalQueryCase]] = None,
    min_region_support: int = 10,
) -> SplitReport:
    """
    Repeat the comparison over several random splits.

    :param dataset: The scenes.
    :param random_seeds: One seed per split.
    :param train_fraction: Share of scenes to fit on.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :param cases: The questions to ask; defaults to :func:`query_catalogue`.
    :param min_region_support: The fewest training rows a cause region may hold for its
        effect to be read as an answer.
    :return: The study.
    """
    return SplitReport(
        reports=[
            evaluate(
                dataset,
                train_fraction=train_fraction,
                random_seed=random_seed,
                min_samples_per_leaf=min_samples_per_leaf,
                plain_min_samples_per_leaf=plain_min_samples_per_leaf,
                cases=cases,
                time_repeats=False,
                min_region_support=min_region_support,
            )
            for random_seed in random_seeds
        ]
    )


# %% how much training data it takes


@dataclass(frozen=True)
class LearningCurvePoint:
    """
    One pipeline's held-out likelihood at one training-set size on one split.
    """

    pipeline_name: str
    """
    The pipeline.
    """

    train_fraction: float
    """
    Share of the scenes it was fitted on.
    """

    random_seed: int
    """
    The seed of the split.
    """

    likelihoods: Dict[SceneView, Optional[LikelihoodReport]]
    """
    Its likelihood report per view.
    """


@dataclass
class LearningCurveReport:
    """
    How the pipelines' held-out likelihoods grow with the training set.
    """

    points: List[LearningCurvePoint] = field(default_factory=list)
    """
    Every measurement.
    """

    @property
    def train_fractions(self) -> List[float]:
        """
        The training-set sizes measured, ascending.
        """
        return sorted({point.train_fraction for point in self.points})

    @property
    def pipeline_names(self) -> List[str]:
        """
        The pipelines measured, in the order they were run.
        """
        return list(dict.fromkeys(point.pipeline_name for point in self.points))

    def points_of(
        self, pipeline_name: str, train_fraction: float
    ) -> List[LearningCurvePoint]:
        """
        :param pipeline_name: A pipeline's name.
        :param train_fraction: A training-set size.
        :return: The pipeline's measurements at that size, one per split.
        """
        return [
            point
            for point in self.points
            if point.pipeline_name == pipeline_name
            and point.train_fraction == train_fraction
        ]


def learning_curve(
    dataset: GraspClutterDataset,
    train_fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    random_seeds: Sequence[int] = (0, 1, 2),
    plain_min_samples_per_leaf: Optional[float] = None,
) -> LearningCurveReport:
    """
    Fit every pipeline's plain model on growing shares of the dataset and score the same
    held-out scenes each time.

    The held-out scenes are the last fifth of every split's shuffle, whatever the training
    share, so every size is scored on the same scenes.

    :param dataset: The scenes.
    :param train_fractions: The training-set sizes to measure.
    :param random_seeds: One seed per split.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain model
        may hold; the pipelines' own default if not given.
    :return: The curve.
    """
    report = LearningCurveReport()
    for random_seed in random_seeds:
        available, test = dataset.split(
            max(train_fractions), np.random.default_rng(random_seed)
        )
        for train_fraction in train_fractions:
            training = available.scenes[
                : round(train_fraction / max(train_fractions) * len(available.scenes))
            ]
            for pipeline in configured_pipelines(
                training, None, plain_min_samples_per_leaf
            ):
                pipeline.fit(training)
                report.points.append(
                    LearningCurvePoint(
                        pipeline_name=pipeline.name,
                        train_fraction=train_fraction,
                        random_seed=random_seed,
                        likelihoods=score_every_view(pipeline, test.scenes),
                    )
                )
    return report
