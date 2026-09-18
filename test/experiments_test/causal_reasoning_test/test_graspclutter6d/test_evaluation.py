"""
Asking every pipeline every question, the studies around that, and rendering the result.
"""

from __future__ import annotations

import math

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import pytest

from experiments.causal_reasoning.graspclutter6d.evaluation import (
    Refusal,
    evaluate,
    learning_curve,
    permutation_study,
    split_study,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import (
    SceneView,
    TableLayout,
)
from experiments.causal_reasoning.graspclutter6d.queries import (
    object_level_cases,
    query_catalogue,
    scene_level_cases,
)
from experiments.causal_reasoning.graspclutter6d.report import MarkdownReport

LEAF_SHARE = 0.2
"""
The share of its training rows a leaf may hold in these tests: wide enough that a
stratum of eight synthetic scenes still spans an interval of the confounder for the
adjustment to sum over.
"""


@pytest.fixture(scope="module")
def comparison(shared_synthetic_dataset):
    """
    Every pipeline fitted on the synthetic scenes and asked every question.
    """
    return evaluate(
        shared_synthetic_dataset,
        random_seed=0,
        min_samples_per_leaf=LEAF_SHARE,
        plain_min_samples_per_leaf=LEAF_SHARE,
    )


@pytest.fixture(scope="module")
def reorderings(shared_synthetic_dataset):
    """
    The questions about one object asked again under three orderings of the parts.
    """
    return permutation_study(
        shared_synthetic_dataset,
        ordering_count=3,
        min_samples_per_leaf=LEAF_SHARE,
        plain_min_samples_per_leaf=LEAF_SHARE,
    )


def _questions_of(study, pipeline_name):
    """
    :param study: A permutation study.
    :param pipeline_name: A pipeline's name.
    :return: What that pipeline made of each question over the orderings.
    """
    return [
        question
        for question in study.questions
        if question.pipeline_name == pipeline_name
    ]


def _outcomes(comparison, pipeline_name):
    return {
        outcome.case.name: outcome
        for outcome in comparison.pipeline(pipeline_name).outcomes
    }


def test_every_question_of_the_catalogue_is_asked_of_every_pipeline(comparison):
    names = [case.name for case in query_catalogue()]
    for pipeline in comparison.pipelines:
        assert [outcome.case.name for outcome in pipeline.outcomes] == names


def test_the_scalars_only_tree_refuses_every_question_about_a_count(comparison):
    outcomes = _outcomes(comparison, f"{TableLayout.SCALARS} tree")
    counted = [
        case.name
        for case in scene_level_cases()
        if "count" in case.name or "count" in getattr(case, "confounder_name", "")
    ]
    assert {outcomes[name].refusal for name in counted} == {Refusal.SCHEMA_MISMATCH}


def test_the_propositional_tree_refuses_every_question_about_one_object(comparison):
    outcomes = _outcomes(comparison, f"{TableLayout.PROPOSITIONAL} tree")
    assert {outcomes[case.name].refusal for case in object_level_cases()} == {
        Refusal.SCHEMA_MISMATCH
    }


def test_the_relational_circuit_answers_every_scene_level_question(comparison):
    outcomes = _outcomes(comparison, "relational circuit")
    assert all(outcomes[case.name].answered for case in scene_level_cases())


def test_an_answer_ranks_the_cause_regions_it_distinguishes(comparison):
    outcome = _outcomes(comparison, "relational circuit")[
        "small_object_count_causes_graspability_adjusting_extent"
    ]
    assert len(outcome.effects) > 1
    assert (
        outcome.most_effective.adjusted_probability
        >= outcome.least_effective.adjusted_probability
    )


def test_the_cause_regions_of_an_answer_partition_the_cause(comparison):
    for pipeline in comparison.pipelines:
        for outcome in pipeline.outcomes:
            if not outcome.answered:
                continue
            total = sum(effect.region_probability for effect in outcome.effects)
            assert total == pytest.approx(1.0)


def test_the_pipelines_that_share_the_columns_give_the_same_answer(comparison):
    relational = _outcomes(comparison, "relational circuit")
    propositional = _outcomes(comparison, f"{TableLayout.PROPOSITIONAL} tree")
    for case in scene_level_cases():
        if not propositional[case.name].answered:
            continue
        assert (
            relational[case.name].most_effective.cause_region
            == propositional[case.name].most_effective.cause_region
        )


def test_every_pipeline_is_scored_on_the_views_it_models(comparison):
    for pipeline in comparison.pipelines:
        assert pipeline.likelihoods[SceneView.SCALARS] is not None


def test_reordering_the_parts_leaves_the_relational_answers_where_they_were(
    reorderings,
):
    questions = _questions_of(reorderings, "relational circuit")
    assert questions
    for question in questions:
        assert len(question.best_regions) <= 1
        assert question.largest_adjusted_difference == pytest.approx(0.0, abs=1e-9)


def test_reordering_the_parts_moves_the_unrolled_trees_answers(reorderings):
    questions = _questions_of(reorderings, f"{TableLayout.UNROLLED} tree")
    assert questions
    assert (
        max(
            question.largest_adjusted_difference
            for question in questions
            if not math.isnan(question.largest_adjusted_difference)
        )
        > 0.0
    )


def test_a_split_study_reports_one_comparison_per_seed(shared_synthetic_dataset):
    study = split_study(
        shared_synthetic_dataset,
        random_seeds=(0, 1),
        min_samples_per_leaf=LEAF_SHARE,
        plain_min_samples_per_leaf=LEAF_SHARE,
    )
    assert [report.random_seed for report in study.reports] == [0, 1]
    assert len(study.coverage("relational circuit", SceneView.SCALARS)) == 2


def test_a_learning_curve_measures_every_share_on_every_pipeline(
    shared_synthetic_dataset,
):
    curve = learning_curve(
        shared_synthetic_dataset,
        train_fractions=(0.4, 0.8),
        random_seeds=(0,),
        plain_min_samples_per_leaf=LEAF_SHARE,
    )
    assert curve.train_fractions == [0.4, 0.8]
    for name in curve.pipeline_names:
        assert len(curve.points_of(name, 0.4)) == 1


def test_the_report_names_every_pipeline_and_every_question(comparison):
    rendered = MarkdownReport(comparison).render()
    for pipeline in comparison.pipelines:
        assert pipeline.name in rendered
    for case in query_catalogue():
        assert case.question in rendered


def test_the_reorderings_are_measured_against_the_datasets_own_order(
    comparison, reorderings
):
    in_dataset_order = reorderings.in_dataset_order("relational circuit")
    assert (
        in_dataset_order
        == comparison.pipeline("relational circuit").likelihoods[SceneView.WHOLE_SCENE]
    )
    assert len(reorderings.reordered("relational circuit")) == 3
    assert reorderings.largest_likelihood_drop("relational circuit") == pytest.approx(
        in_dataset_order.mean_log_likelihood
        - min(
            report.mean_log_likelihood
            for report in reorderings.reordered("relational circuit")
        )
    )


def test_the_relational_circuits_answers_never_move_under_reordering(reorderings):
    for question in _questions_of(reorderings, "relational circuit"):
        assert question.largest_adjusted_difference == pytest.approx(0.0, abs=1e-9)
        assert math.isnan(question.argmax_flip_share) or question.argmax_flip_share == 0


def test_the_report_renders_the_reordering_distribution(comparison, reorderings):
    rendered = MarkdownReport(comparison, permutations=reorderings).render()
    assert "dataset order, coverage / mean log-likelihood" in rendered
    assert "argmax moved" in rendered
