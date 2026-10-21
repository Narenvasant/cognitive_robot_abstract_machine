"""
Tests that a causal query takes its adjustment set from the causal structure learned for
its class: chosen when the query marks no ``confounder``, checked when it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from probabilistic_model.causal_structure.exceptions import InvalidAdjustmentSet
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import AdjustmentSelector
from probabilistic_model.causal_structure.learning import LearnedCausalStructure
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from random_events.variable import Integer

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import (
    EffectNotIdentified,
    NoCauseVariablesForRanking,
)
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.parametrization.causal_structure_registries import (
    DictCausalStructureRegistry,
)
from krrood.parametrization.model_registries import (
    CausalCircuitRegistry,
    RelationalCircuitRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters

from ...dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneRoom,
)
from ...test_feature_extraction.test_relational_probabilistic_circuit import (  # noqa: F401  # fixtures
    layered_relational_probabilistic_circuit,
    relational_probabilistic_circuit,
    scenario,
)

# %% a summer whose season drives both ice cream sales and drownings


@dataclass
class SummerDay:
    season: int
    ice_cream: int
    sunscreen: int
    heat: int
    drowning: int


def summer_rows(row_count: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows where the season drives the ice cream sales and the drownings, and
        neither ice cream nor sunscreen causes a drowning.
    """
    random_state = np.random.default_rng(seed)
    season = random_state.integers(0, 2, row_count)
    heat = random_state.integers(0, 2, row_count)

    def noisy(values: np.ndarray) -> np.ndarray:
        return values ^ (random_state.random(row_count) < 0.2)

    return pd.DataFrame(
        {
            "SummerDay.season": season,
            "SummerDay.ice_cream": noisy(season),
            "SummerDay.sunscreen": noisy(heat),
            "SummerDay.heat": heat,
            "SummerDay.drowning": noisy(season | heat),
        }
    )


def summer_graph(
    directed: list[tuple[str, str]], undirected: list[tuple[str, str]] = ()
) -> CausalGraph:
    """
    :param directed: The directed edges between ``SummerDay`` fields.
    :param undirected: The undirected edges between them.
    :return: The graph over every ``SummerDay`` field.
    """
    return CausalGraph(
        nodes=[
            GraphNode(
                variable=Integer(f"SummerDay.{name}"),
                level=NodeLevel.EXAMPLE_ATTRIBUTE,
            )
            for name in ("season", "ice_cream", "sunscreen", "heat", "drowning")
        ],
        edges=[
            GraphEdge.directed(f"SummerDay.{parent}", f"SummerDay.{child}")
            for parent, child in directed
        ]
        + [
            GraphEdge.undirected(f"SummerDay.{first}", f"SummerDay.{second}")
            for first, second in undirected
        ],
    )


TRUE_SUMMER_EDGES = [
    ("season", "ice_cream"),
    ("season", "drowning"),
    ("heat", "sunscreen"),
    ("heat", "drowning"),
]


def structure_of(graph: CausalGraph) -> DictCausalStructureRegistry:
    """
    :param graph: The causal graph of ``SummerDay``.
    :return: A registry holding it, with the summer's rows to rank sets by.
    """
    return DictCausalStructureRegistry(
        {SummerDay: LearnedCausalStructure(AdjustmentSelector(graph), summer_rows())}
    )


@pytest.fixture
def summer_backend_registry() -> CausalCircuitRegistry:
    """
    :return: A causal circuit over the summer's rows with the ice cream sales and the
        season as causes, so that both can be intervened on and adjusted for.
    """
    return summer_circuit_registry(["ice_cream", "season"])


def ice_cream_query(declares_the_season: bool):
    """
    :param declares_the_season: Whether the season is marked ``confounder``.
    :return: Does ice cream cause drowning?
    """
    query = a(SummerDay)(
        season=confounder if declares_the_season else ...,
        ice_cream=cause,
        sunscreen=...,
        heat=...,
        drowning=...,
    )
    query.causes_effect(query.drowning == 1)
    return query


# %% choosing the adjustment set


def test_learned_graph_supplies_the_confounder(summer_backend_registry):
    learned = ProbabilisticBackend(
        model_registry=summer_backend_registry,
        causal_structure_registry=structure_of(summer_graph(TRUE_SUMMER_EDGES)),
    ).rank_causes(ice_cream_query(declares_the_season=False))
    declared = ProbabilisticBackend(model_registry=summer_backend_registry).rank_causes(
        ice_cream_query(declares_the_season=True)
    )

    assert [candidate.effect_probability_given_region for candidate in learned] == [
        candidate.effect_probability_given_region for candidate in declared
    ]


def test_backend_without_structure_registry_chooses_nothing():
    parameters = UnderspecifiedParameters(ice_cream_query(declares_the_season=False))

    ProbabilisticBackend().choose_adjustment_sets(
        parameters, ice_cream_query(declares_the_season=False)
    )

    assert parameters.derived_adjustment_variables == {}


def test_each_cause_candidate_gets_its_own_adjustment_set():
    query = a(SummerDay)(
        season=..., ice_cream=cause, sunscreen=cause, heat=..., drowning=...
    )
    query.causes_effect(query.drowning == 1)
    parameters = UnderspecifiedParameters(query)
    backend = ProbabilisticBackend(
        causal_structure_registry=structure_of(summer_graph(TRUE_SUMMER_EDGES))
    )

    backend.choose_adjustment_sets(parameters, query)

    assert {
        cause_variable.name: [
            variable.name
            for variable in parameters.adjustment_variables_of(cause_variable)
        ]
        for cause_variable in parameters.search_cause_variables
    } == {
        "SummerDay.ice_cream": ["SummerDay.season"],
        "SummerDay.sunscreen": ["SummerDay.heat"],
    }


# %% refusing


def test_declared_invalid_confounder_is_refused():
    mediated = summer_graph([("ice_cream", "season"), ("season", "drowning")])
    query = ice_cream_query(declares_the_season=True)

    with pytest.raises(InvalidAdjustmentSet):
        ProbabilisticBackend(
            causal_structure_registry=structure_of(mediated)
        ).choose_adjustment_sets(UnderspecifiedParameters(query), query)


def test_partially_identified_effect_raises_effect_not_identified():
    undecided = summer_graph(
        [("season", "drowning"), ("ice_cream", "drowning")],
        [("season", "ice_cream")],
    )
    query = ice_cream_query(declares_the_season=False)

    with pytest.raises(EffectNotIdentified):
        ProbabilisticBackend(
            causal_structure_registry=structure_of(undecided)
        ).choose_adjustment_sets(UnderspecifiedParameters(query), query)


# %% the relational path


def room_query():
    """
    :return: Does a room's x position cause its y position, among rooms of four
        objects?
    """
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=cause, y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.causes_effect(query.position.y == 1.0)
    return query


@pytest.mark.parametrize(
    "fitted_circuit",
    ["relational_probabilistic_circuit", "layered_relational_probabilistic_circuit"],
)
def test_derived_count_survives_the_relational_trim(fitted_circuit, request):
    circuit = request.getfixturevalue(fitted_circuit)
    count = circuit.exchangeable_distribution_templates["objects"].latent_variables[0]
    query = room_query()
    parameters = UnderspecifiedParameters(query)
    [x] = parameters.search_cause_variables
    [y] = parameters.effect_variables_from_causes_effect
    graph = CausalGraph(
        nodes=[
            GraphNode(variable=x, level=NodeLevel.EXAMPLE_ATTRIBUTE),
            GraphNode(variable=y, level=NodeLevel.EXAMPLE_ATTRIBUTE),
            GraphNode(variable=count, level=NodeLevel.AGGREGATION_STATISTIC),
        ],
        edges=[
            GraphEdge.directed(count.name, x.name),
            GraphEdge.directed(count.name, y.name),
        ],
    )
    backend = ProbabilisticBackend(
        causal_structure_registry=DictCausalStructureRegistry(
            {SceneRoom: LearnedCausalStructure(AdjustmentSelector(graph), None)}
        )
    )

    backend.choose_adjustment_sets(parameters, query)
    np.random.seed(0)
    causal_circuit = RelationalCircuitRegistry(
        relational_probabilistic_circuit=circuit
    ).get_model(parameters)

    assert count.name in {
        variable.name for variable in causal_circuit.probabilistic_circuit.variables
    }


# %% an effect identified only up to a few adjustment sets


def summer_circuit_registry(causes: list[str]) -> CausalCircuitRegistry:
    """
    :param causes: The ``SummerDay`` fields to stratify by and register as causes.
    :return: A causal circuit over the summer's rows with those causes.
    """
    rows = summer_rows()
    variables = infer_variables_from_dataframe(rows)
    by_name = {annotated.variable.name: annotated.variable for annotated in variables}
    cause_variables = [by_name[f"SummerDay.{name}"] for name in causes]
    effect_variable = by_name["SummerDay.drowning"]
    circuit = StratifiedLearning(
        variables=[variable.name for variable in cause_variables],
        method=JointProbabilityTree(),
    ).fit(rows, variables)
    return CausalCircuitRegistry(
        {
            SummerDay: CausalCircuit.from_probabilistic_circuit(
                circuit,
                MarginalDeterminismTreeNode.from_variable_order(cause_variables),
                cause_variables,
                [effect_variable],
            )
        }
    )


def effect_probability_within(interventional, region, effect_event) -> float:
    """
    :return: The effect's probability within one region of the cause.
    """
    truncated, _ = interventional.truncated(
        region.fill_missing_variables_pure(interventional.variables)
    )
    return float(
        truncated.probability(
            effect_event.fill_missing_variables_pure(truncated.variables)
        )
    )


def test_range_spans_the_answers_of_each_candidate():
    registry = summer_circuit_registry(["ice_cream", "season"])
    undecided = summer_graph(
        [("season", "drowning"), ("ice_cream", "drowning")],
        [("season", "ice_cream")],
    )
    query = ice_cream_query(declares_the_season=False)
    [answer] = ProbabilisticBackend(
        model_registry=registry, causal_structure_registry=structure_of(undecided)
    ).interventional_range(query)
    causal_circuit = registry.circuits[SummerDay]
    [ice_cream, season] = causal_circuit.causal_variables
    [drowning] = causal_circuit.effect_variables
    effect_event = UnderspecifiedParameters(
        query
    ).truncation_assignments_from_where_conditions
    candidates = [
        causal_circuit.backdoor_adjustment(ice_cream, drowning, adjustment)
        for adjustment in ([], [season])
    ]

    assert [
        (region.lowest_effect_probability, region.highest_effect_probability)
        for region in answer.regions
    ] == [
        (
            min(probabilities),
            max(probabilities),
        )
        for probabilities in (
            [
                effect_probability_within(candidate, region.event, effect_event)
                for candidate in candidates
            ]
            for region in causal_circuit._extract_disjoint_regions_for_variable(
                ice_cream
            )
        )
    ]


# %% ranking every possible cause of an effect


def drowning_query(**causes):
    """
    :return: What causes drowning, with the given fields marked.
    """
    fields = dict(season=..., ice_cream=..., sunscreen=..., heat=..., drowning=...)
    fields.update(causes)
    query = a(SummerDay)(**fields)
    query.causes_effect(query.drowning == 1)
    return query


def test_effect_only_ranking_equals_marking_every_possible_ancestor():
    backend = ProbabilisticBackend(
        model_registry=summer_circuit_registry(["season", "heat"]),
        causal_structure_registry=structure_of(summer_graph(TRUE_SUMMER_EDGES)),
    )

    effect_only = backend.rank_causes(drowning_query())
    marked = backend.rank_causes(drowning_query(season=cause, heat=cause))

    assert [
        (candidate.cause_variable.name, candidate.effect_probability_given_region)
        for candidate in effect_only
    ] == [
        (candidate.cause_variable.name, candidate.effect_probability_given_region)
        for candidate in marked
    ]


def test_effect_only_ranking_without_structure_registry_raises():
    with pytest.raises(NoCauseVariablesForRanking):
        ProbabilisticBackend(
            model_registry=summer_circuit_registry(["season", "heat"])
        ).rank_causes(drowning_query())
