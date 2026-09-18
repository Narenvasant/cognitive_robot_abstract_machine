"""
The two causal-query pipelines under comparison, behind one interface: fit on scenes,
serve ``cause``/``causes_effect`` EQL queries through a model registry, and report what
the fits cost and how well they explain held-out scenes.

Backdoor adjustment needs the circuit it runs on to be support-deterministic over the
cause variable, which a fit guarantees by stratifying its training rows on that
variable's exact value. Stratifying on two variables at once cannot serve both, since
two partitions sharing a value of one of them overlap on it, so each pipeline keeps one
plain model for everything that is not a causal query, and fits one further model per
cause variable it is asked about, the first time it is asked.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import pandas as pd
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import (
    ModelRegistry,
    RelationalCircuitRegistry,
)
from krrood.parametrization.parameterizer import (
    ModelQueryParameters,
    UnderspecifiedParameters,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.learning_method import (
    LearningMethod,
    StratifiedLearning,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    ExchangeableDistributionTemplate,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel
from typing_extensions import Any, Dict, List, Optional, Sequence, Set

from experiments.causal_reasoning.graspclutter6d.domain import GraspClutterScene
from experiments.causal_reasoning.graspclutter6d.exceptions import (
    FlatTableSchemaMismatchError,
    OneCausePerQueryError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import (
    FlatTable,
    PartAttribute,
    SceneSchema,
    SceneView,
    TableLayout,
)

# %% what a fit reports


@dataclass(frozen=True)
class CircuitSize:
    """
    How big a fitted circuit is.
    """

    node_count: int
    """
    Number of units, leaves included.
    """

    edge_count: int
    """
    Number of edges between units.
    """

    @classmethod
    def of(cls, circuit: ProbabilisticCircuit) -> CircuitSize:
        """
        :param circuit: The circuit to measure.
        :return: Its size.
        """
        return cls(node_count=len(circuit.nodes()), edge_count=len(circuit.edges()))

    def __add__(self, other: CircuitSize) -> CircuitSize:
        return CircuitSize(
            self.node_count + other.node_count, self.edge_count + other.edge_count
        )


@dataclass
class FitReport:
    """
    What fitting a pipeline cost and produced, over the plain model and every cause-
    specific model fitted on demand.
    """

    training_scene_count: int
    """
    How many scenes the pipeline was fitted on.
    """

    model_count: int = 0
    """
    How many models were fitted.
    """

    training_duration: float = 0.0
    """
    Wall-clock seconds all the fits took together.
    """

    size: CircuitSize = CircuitSize(0, 0)
    """
    The size of every fitted circuit together.
    """

    def record(self, duration: float, size: CircuitSize) -> None:
        """
        Add one model's fit.

        :param duration: Wall-clock seconds the fit took.
        :param size: The size of the fitted circuit.
        """
        self.model_count += 1
        self.training_duration += duration
        self.size = self.size + size


@dataclass(frozen=True)
class LikelihoodReport:
    """
    How well a fitted pipeline explains held-out scenes.
    """

    scene_count: int
    """
    How many scenes were scored.
    """

    covered_scene_count: int
    """
    How many of them lie inside the model's support at all.
    """

    mean_log_likelihood: float
    """
    Mean log-likelihood over the covered scenes; ``nan`` if none is covered.
    """

    log_likelihoods: np.ndarray = field(compare=False, repr=False)
    """
    One log-likelihood per scored scene, in the scenes' order, ``-inf`` for a scene
    outside the support.
    """

    @property
    def coverage(self) -> float:
        """
        Share of scenes inside the model's support.
        """
        return self.covered_scene_count / self.scene_count

    @classmethod
    def from_log_likelihoods(cls, log_likelihoods: np.ndarray) -> LikelihoodReport:
        """
        :param log_likelihoods: One log-likelihood per scored scene, ``-inf`` for a
            scene outside the support.
        :return: The report.
        """
        finite = log_likelihoods[np.isfinite(log_likelihoods)]
        return cls(
            scene_count=len(log_likelihoods),
            covered_scene_count=len(finite),
            mean_log_likelihood=float(finite.mean()) if len(finite) else float("nan"),
            log_likelihoods=log_likelihoods,
        )


# %% which variable a query marks as its cause


@dataclass(frozen=True)
class CauseStratification:
    """
    What a fit has to be stratified by to register one variable as a cause.
    """

    class_columns: Optional[List[str]]
    """
    The scene-level columns to stratify the class circuit by, or ``None`` to leave it to
    the plain fit.
    """

    part_attributes: Dict[str, List[str]]
    """
    Per exchangeable-part field, the attributes to stratify that part's template by; a
    part absent from the mapping is left to the plain fit.
    """

    @classmethod
    def for_variable(
        cls, variable_name: str, schema: SceneSchema = SceneSchema()
    ) -> CauseStratification:
        """
        :param variable_name: The cause variable's name, as EQL names it.
        :param schema: How the scene's attributes are named.
        :return: The stratification that makes a fit support-deterministic over it.
        """
        part = schema.part_attribute(variable_name)
        if part is None:
            return cls(class_columns=[variable_name], part_attributes={})
        return cls(
            class_columns=None, part_attributes={part.part_field: [part.attribute]}
        )


def cause_variable_name(parameters: ModelQueryParameters) -> Optional[str]:
    """
    :param parameters: The parameters extracted from a queried statement.
    :return: The name of the one variable the query marks as its cause, or ``None`` if it
        marks none.
    :raises OneCausePerQueryError: If the query marks more than one cause.
    """
    if not isinstance(parameters, UnderspecifiedParameters):
        return None
    causes = parameters.search_cause_variables
    if not causes:
        return None
    if len(causes) > 1:
        raise OneCausePerQueryError([cause.name for cause in causes])
    return causes[0].name


def constrained_variable_names(parameters: ModelQueryParameters) -> Set[str]:
    """
    :param parameters: The parameters extracted from a queried statement.
    :return: The names of the variables the query says something about: a value it sets,
        a condition it truncates to, or a cause, confounder or effect it marks. A
        variable the query merely lists and leaves open is not among them.
    """
    if not isinstance(parameters, UnderspecifiedParameters):
        return set(parameters.variables)
    names = {
        variable.name
        for variable in (
            parameters.search_cause_variables
            + parameters.search_confounder_variables
            + parameters.effect_variables_from_causes_effect
            + list(parameters.conditioning_assignments_from_literal_values)
        )
    }
    events = list(parameters.truncation_assignments_from_krrood_variables)
    if parameters.truncation_assignments_from_where_conditions is not None:
        events.append(parameters.truncation_assignments_from_where_conditions)
    for event in events:
        names.update(variable.name for variable in event.variables)
    return names


# %% the shared interface


@dataclass
class CausalQueryPipeline(ABC):
    """
    One way of turning scenes into models that answer causal EQL queries.
    """

    min_samples_per_leaf: float = 0.05
    """
    The fewest training rows a leaf of a cause-specific model may hold, as a share of the
    rows it is fitted on: enough for a continuous attribute's leaf to span a range rather
    than pin the values it saw, few enough for a stratum of one cause value to still
    split on what else drives the effect.

    See
    :attr:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree.min_samples_per_leaf`,
    which reads a share below one as a share of its training rows, so the same setting
    holds for a class circuit over a thousand scenes and for a part template over their
    tens of thousands of parts.
    """

    plain_min_samples_per_leaf: float = 0.15
    """
    The fewest training rows a leaf of the plain model may hold, as a share of the rows
    it is fitted on.

    The plain model scores held-out scenes, and a leaf spans only the ranges it saw, so
    wider leaves cover more of them.
    """

    schema: SceneSchema = field(default_factory=SceneSchema)
    """
    How the scene's attributes are named.
    """

    training_scenes: List[GraspClutterScene] = field(default_factory=list)
    """
    The scenes :meth:`fit` was given, kept for the cause-specific fits.
    """

    fit_report: Optional[FitReport] = None
    """
    What the fits so far cost, once :meth:`fit` ran.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        What the pipeline is called in reports.
        """

    @property
    def registry(self) -> ModelRegistry:
        """
        The registry a
        :class:`~krrood.entity_query_language.backends.ProbabilisticBackend` resolves
        queries against.
        """
        return PipelineRegistry(pipeline=self)

    @property
    @abstractmethod
    def table(self) -> FlatTable:
        """
        The scenes as rows of what the pipeline's plain model is fitted on.
        """

    @property
    @abstractmethod
    def models_parts(self) -> bool:
        """
        Whether the pipeline models the objects and viewpoints themselves, so that the
        order a scene lists them in can matter to it.
        """

    @property
    @abstractmethod
    def order_invariant(self) -> bool:
        """
        Whether fitting on the same scenes with their parts in another order gives the
        same model, so that a study over reorderings need fit it once.
        """

    def fit(self, scenes: Sequence[GraspClutterScene]) -> FitReport:
        """
        Fit the plain model, and keep the scenes for the cause-specific fits.

        :param scenes: The scenes to fit on.
        :return: The report the later fits keep adding to.
        """
        self.training_scenes = list(scenes)
        self.fit_report = FitReport(training_scene_count=len(scenes))
        started = time.perf_counter()
        size = self._fit_plain_model()
        self.fit_report.record(time.perf_counter() - started, size)
        return self.fit_report

    @abstractmethod
    def _fit_plain_model(self) -> CircuitSize:
        """
        Fit the model that serves every query without a cause, on
        :attr:`training_scenes`.

        :return: The size of the fitted circuit.
        """

    @abstractmethod
    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        """
        Fit the model that serves queries marking the named variable as their cause, on
        :attr:`training_scenes`.

        :param cause_name: The cause variable's name, as EQL names it.
        :return: The size of the fitted circuit.
        """

    @abstractmethod
    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        """
        :param cause_name: The cause variable's name, or ``None`` for the plain model.
        :return: The registry over the model fitted for it.
        """

    @abstractmethod
    def _has_cause_model(self, cause_name: str) -> bool:
        """
        :param cause_name: The cause variable's name.
        :return: Whether its model was fitted already.
        """

    def registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        """
        The registry over the model serving queries with the given cause, fitting that
        model first if it was not asked for before.

        :param cause_name: The cause variable's name, or ``None`` for the plain model.
        :return: The registry.
        :raises PipelineNotFittedError: If :meth:`fit` never ran.
        """
        if self.fit_report is None:
            raise PipelineNotFittedError(self.name)
        if cause_name is not None and not self._has_cause_model(cause_name):
            started = time.perf_counter()
            size = self._fit_cause_model(cause_name)
            self.fit_report.record(time.perf_counter() - started, size)
        return self._registry_for(cause_name)

    def training_values_of(self, variable_name: str) -> List[Any]:
        """
        The values the training scenes hold for a variable, one per row the variable
        is fitted on: one per scene for a scene attribute or count, one per part for a
        part's own attribute.

        :param variable_name: The variable's name, as EQL names it.
        :return: The values.
        :raises FlatTableSchemaMismatchError: If the pipeline's table has no column for
            the variable.
        """
        part = self.schema.part_attribute(variable_name)
        if part is not None and self.models_parts:
            return self._part_training_values(part)
        if variable_name not in self.table.columns:
            raise FlatTableSchemaMismatchError([variable_name])
        return [self.table.row(scene)[variable_name] for scene in self.training_scenes]

    @abstractmethod
    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        """
        :param part: One part's attribute, as a query names it.
        :return: The values the training scenes hold for it, one per row the part
            template or column is fitted on.
        """

    @abstractmethod
    def plain_circuit_of(self, view: SceneView) -> Optional[ProbabilisticCircuit]:
        """
        The plain model's joint over as much of a scene as the view asks for.

        :param view: How much of a scene to look at.
        :return: The circuit, or ``None`` if the pipeline models less than that.
        :raises PipelineNotFittedError: If :meth:`fit` never ran.
        """

    def log_likelihood(
        self, scenes: Sequence[GraspClutterScene], view: SceneView
    ) -> Optional[LikelihoodReport]:
        """
        Score held-out scenes under the plain model, on as much of them as the view asks
        for.

        :param scenes: The scenes to score.
        :param view: How much of a scene to look at.
        :return: The report, or ``None`` if the pipeline models less than the view; a
            scene the pipeline's table has no row for counts as outside the support.
        """
        circuit = self.plain_circuit_of(view)
        if circuit is None:
            return None
        log_likelihoods = np.full(len(scenes), -np.inf)
        fitting = [
            index for index, scene in enumerate(scenes) if self.table.fits(scene)
        ]
        if fitting:
            rows = [self.table.row(scenes[index]) for index in fitting]
            log_likelihoods[fitting] = log_likelihoods_of_rows(circuit, rows)
        return LikelihoodReport.from_log_likelihoods(log_likelihoods)


def log_likelihoods_of_rows(
    circuit: ProbabilisticCircuit, rows: Sequence[Dict[str, Any]]
) -> np.ndarray:
    """
    :param circuit: The circuit to score under.
    :param rows: One value per variable of the circuit, keyed by variable name, per row.
    :return: One log-likelihood per row, ``-inf`` outside the support.
    """
    events = np.array(
        [[row[variable.name] for variable in circuit.variables] for row in rows],
        dtype=object,
    )
    return circuit.log_likelihood(events)


@dataclass
class PipelineRegistry(ModelRegistry):
    """
    Routes each query to the pipeline's model fitted for the cause it marks.
    """

    pipeline: CausalQueryPipeline
    """
    The pipeline whose models are served.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        return self.pipeline.registry_for(cause_variable_name(parameters)).get_model(
            parameters
        )


# %% relational pipeline


@dataclass
class RelationalPipeline(CausalQueryPipeline):
    """
    A relational probabilistic circuit fitted on the scenes' relational structure and
    grounded per query into a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    monte_carlo_sample_count: int = 2000
    """
    How many samples grounding draws for an aggregation count a query leaves open.
    """

    plain_model: Optional[RelationalProbabilisticCircuit] = None
    """
    The model fitted without any stratification, once :meth:`fit` ran.
    """

    cause_models: Dict[str, RelationalProbabilisticCircuit] = field(
        default_factory=dict
    )
    """
    The model fitted for each cause variable asked about so far, by variable name.
    """

    @property
    def name(self) -> str:
        return "relational circuit"

    @staticmethod
    def _learning_method(
        min_samples_per_leaf: float, stratified_columns: Optional[List[str]]
    ) -> LearningMethod:
        """
        :param min_samples_per_leaf: The fewest training rows a leaf may hold.
        :param stratified_columns: The columns to stratify by, or ``None`` for a plain
            fit.
        :return: The learning method fitting a circuit that way.
        """
        tree_learning = JointProbabilityTree(min_samples_per_leaf=min_samples_per_leaf)
        if stratified_columns is None:
            return tree_learning
        return StratifiedLearning(variables=stratified_columns, method=tree_learning)

    def _new_model(
        self,
        min_samples_per_leaf: float,
        stratification: CauseStratification = CauseStratification(None, {}),
    ) -> RelationalProbabilisticCircuit:
        """
        :param min_samples_per_leaf: The fewest training rows a leaf of the class circuit
            and of every part template may hold.
        :param stratification: What the fit is stratified by; nothing by default.
        :return: The model, not yet fitted.
        """
        return RelationalProbabilisticCircuit(
            GraspClutterScene,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
            learning_method=self._learning_method(
                min_samples_per_leaf, stratification.class_columns
            ),
            part_learning_methods={
                part_field: self._learning_method(
                    min_samples_per_leaf,
                    stratification.part_attributes.get(part_field),
                )
                for part_field in self.schema.part_fields
            },
        )

    @staticmethod
    def _size_of(model: RelationalProbabilisticCircuit) -> CircuitSize:
        """
        :param model: A fitted relational circuit.
        :return: The size of its class circuit and every part template together.
        """
        size = CircuitSize.of(model.class_probabilistic_circuit)
        for template in model.exchangeable_distribution_templates.values():
            size = size + CircuitSize.of(
                template.template_distribution.class_probabilistic_circuit
            )
        return size

    def _training_rows(self) -> List[Any]:
        """
        The training scenes as data access objects, each with its parts in a canonical
        order, so that the fit is the same whatever order the scenes list their parts
        in. The tree learner the templates are fitted with breaks ties by row order, and
        the parts of every scene are pooled into its rows.

        :return: One data access object per training scene.
        """
        return [to_dao(self._canonical(scene)) for scene in self.training_scenes]

    def _canonical(self, scene: GraspClutterScene) -> GraspClutterScene:
        """
        :param scene: A scene.
        :return: The scene with every part list sorted by the parts' attribute values.
        """
        return replace(
            scene,
            **{
                part_field: sorted(
                    vars(scene)[part_field],
                    key=lambda part: tuple(
                        str(value) if isinstance(value, Enum) else value
                        for value in vars(part).values()
                    ),
                )
                for part_field in self.schema.part_fields
            },
        )

    def _fit_plain_model(self) -> CircuitSize:
        self.plain_model = self._new_model(self.plain_min_samples_per_leaf)
        self.plain_model.fit(self._training_rows())
        return self._size_of(self.plain_model)

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        stratification = CauseStratification.for_variable(cause_name, self.schema)
        model = self._new_model(self.min_samples_per_leaf, stratification)
        model.fit(self._training_rows())
        self.cause_models[cause_name] = model
        return self._size_of(model)

    def _has_cause_model(self, cause_name: str) -> bool:
        return cause_name in self.cause_models

    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        model = (
            self.plain_model if cause_name is None else self.cause_models[cause_name]
        )
        return RelationalCircuitRegistry(relational_probabilistic_circuit=model)

    @property
    def table(self) -> FlatTable:
        return FlatTable(TableLayout.PROPOSITIONAL, schema=self.schema)

    @property
    def models_parts(self) -> bool:
        return True

    @property
    def order_invariant(self) -> bool:
        return True

    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        return [
            vars(one)[part.attribute]
            for scene in self.training_scenes
            for one in vars(scene)[part.part_field]
        ]

    def plain_circuit_of(self, view: SceneView) -> Optional[ProbabilisticCircuit]:
        if self.plain_model is None:
            raise PipelineNotFittedError(self.name)
        class_circuit = self.plain_model.class_probabilistic_circuit
        if view is SceneView.SCALARS:
            scalar_columns = set(self.schema.scalar_columns)
            return class_circuit.marginal(
                [
                    variable
                    for variable in class_circuit.variables
                    if variable.name in scalar_columns
                ]
            )
        return class_circuit

    def log_likelihood(
        self, scenes: Sequence[GraspClutterScene], view: SceneView
    ) -> Optional[LikelihoodReport]:
        """
        Score held-out scenes under the plain model, on as much of them as the view asks
        for. A whole scene is scored the way the relational circuit factorizes it: the
        class circuit over its scalars and counts, times each part template over one
        object or viewpoint given those counts, the parts taken in canonical order so
        that the sum does not depend on the order the scene lists them in.

        :param scenes: The scenes to score.
        :param view: How much of a scene to look at.
        :return: The report.
        """
        if view is not SceneView.WHOLE_SCENE:
            return super().log_likelihood(scenes, view)
        log_likelihoods = (
            super()
            .log_likelihood(scenes, SceneView.SCALARS_AND_COUNTS)
            .log_likelihoods.copy()
        )
        for (
            part_field,
            template,
        ) in self.plain_model.exchangeable_distribution_templates.items():
            for index, scene in enumerate(scenes):
                log_likelihoods[index] += self._part_log_likelihood(
                    template, scene, vars(self._canonical(scene))[part_field]
                )
        return LikelihoodReport.from_log_likelihoods(log_likelihoods)

    def _part_log_likelihood(
        self,
        template: ExchangeableDistributionTemplate,
        scene: GraspClutterScene,
        parts: Sequence[Any],
    ) -> float:
        """
        :param template: The fitted template of one exchangeable part.
        :param scene: The scene the parts belong to.
        :param parts: The scene's parts of that kind.
        :return: The summed log-likelihood of every part given the scene's counts.
        """
        circuit = template.template_distribution.class_probabilistic_circuit
        latents = template.latent_variables
        counts = {
            variable.name: self.table.row(scene)[variable.name] for variable in latents
        }
        if not parts:
            return 0.0
        given_counts = log_likelihoods_of_rows(circuit.marginal(latents), [counts])[0]
        if not np.isfinite(given_counts):
            return -np.inf
        rows = [{**counts, **vars(part)} for part in parts]
        joint = log_likelihoods_of_rows(circuit, rows)
        return float(joint.sum() - len(parts) * given_counts)


# %% flat-table pipeline


@dataclass
class FlatTableRegistry(ModelRegistry):
    """
    Serves a circuit fitted on the flat table for queries over that table's columns,
    wrapped as a causal circuit when the query marks a cause.

    A query may list a scene's objects and viewpoints, which the table has no columns
    for, as long as it says nothing about them; the served circuit then simply lacks
    them.
    """

    circuit: ProbabilisticCircuit
    """
    The fitted circuit.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        """
        :param parameters: The parameters extracted from the queried statement.
        :return: The circuit, renamed to the query's own variables, and wrapped as a
            verified causal circuit if the query marks a cause.
        :raises FlatTableSchemaMismatchError: If the query constrains a variable the
            table never had.
        """
        fitted_names = {variable.name for variable in self.circuit.variables}
        missing = sorted(constrained_variable_names(parameters) - fitted_names)
        if missing:
            raise FlatTableSchemaMismatchError(missing)
        renamed = self.circuit.__deepcopy__()
        renamed.update_variables(
            {
                variable: parameters.variables[variable.name]
                for variable in renamed.variables
                if variable.name in parameters.variables
            }
        )
        if cause_variable_name(parameters) is None:
            return renamed
        return RelationalCausalCircuit().from_grounded_circuit(
            renamed,
            parameters.search_cause_variables,
            list(parameters.effect_variables_from_causes_effect),
            adjustment_variables=parameters.search_confounder_variables,
            trim_to_registered_variables=True,
        )


@dataclass
class FlatTablePipeline(CausalQueryPipeline):
    """
    Joint probability trees fitted on the scenes flattened into one table, each wrapped
    as a
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    flat_table: FlatTable = field(default_factory=FlatTable)
    """
    The table the scenes are flattened into.
    """

    plain_circuit: Optional[ProbabilisticCircuit] = None
    """
    The tree fitted without any stratification, once :meth:`fit` ran.
    """

    cause_circuits: Dict[str, ProbabilisticCircuit] = field(default_factory=dict)
    """
    The tree fitted for each cause variable asked about so far, by variable name.
    """

    @property
    def name(self) -> str:
        return f"{self.flat_table.layout} tree"

    @property
    def table(self) -> FlatTable:
        return self.flat_table

    @property
    def models_parts(self) -> bool:
        return self.flat_table.layout.has_parts

    @property
    def order_invariant(self) -> bool:
        return not self.flat_table.layout.has_parts

    def _part_training_values(self, part: PartAttribute) -> List[Any]:
        column = self.schema.part_column(part)
        return [self.table.row(scene)[column] for scene in self.training_scenes]

    def _training_dataframe(self) -> pd.DataFrame:
        return self.table.dataframe(self.training_scenes)

    def _fit_plain_model(self) -> CircuitSize:
        dataframe = self._training_dataframe()
        self.plain_circuit = JointProbabilityTree(
            min_samples_per_leaf=self.plain_min_samples_per_leaf
        ).fit(dataframe, infer_variables_from_dataframe(dataframe))
        return CircuitSize.of(self.plain_circuit)

    def _fit_cause_model(self, cause_name: str) -> CircuitSize:
        if cause_name not in self.table.columns:
            raise FlatTableSchemaMismatchError([cause_name])
        dataframe = self._training_dataframe()
        self.cause_circuits[cause_name] = StratifiedLearning(
            variables=[cause_name],
            method=JointProbabilityTree(min_samples_per_leaf=self.min_samples_per_leaf),
        ).fit(dataframe, infer_variables_from_dataframe(dataframe))
        return CircuitSize.of(self.cause_circuits[cause_name])

    def _has_cause_model(self, cause_name: str) -> bool:
        return cause_name in self.cause_circuits

    def _registry_for(self, cause_name: Optional[str]) -> ModelRegistry:
        circuit = (
            self.plain_circuit
            if cause_name is None
            else self.cause_circuits[cause_name]
        )
        return FlatTableRegistry(circuit=circuit)

    def plain_circuit_of(self, view: SceneView) -> Optional[ProbabilisticCircuit]:
        if self.plain_circuit is None:
            raise PipelineNotFittedError(self.name)
        columns = self.table.columns_of(view)
        if columns is None:
            return None
        column_set = set(columns)
        return self.plain_circuit.marginal(
            [
                variable
                for variable in self.plain_circuit.variables
                if variable.name in column_set
            ]
        )


def pipelines(scenes: Sequence[GraspClutterScene]) -> List[CausalQueryPipeline]:
    """
    :param scenes: The scenes the pipelines will be fitted on, which size the unrolled
        table.
    :return: Every pipeline, unfitted: the relational circuit and one flat-table tree per
        layout.
    """
    return [
        RelationalPipeline(),
        FlatTablePipeline(flat_table=FlatTable(TableLayout.PROPOSITIONAL)),
        FlatTablePipeline(flat_table=FlatTable.unrolled_for(scenes)),
        FlatTablePipeline(flat_table=FlatTable(TableLayout.SCALARS)),
    ]
