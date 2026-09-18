"""
The synthetic model: what it samples, what forcing a cause does, and how an answer is
scored against it.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest

from experiments.causal_reasoning.graspclutter6d.dataset import GraspClutterDataset
from experiments.causal_reasoning.graspclutter6d.domain import (
    GraspClutterSceneAggregations,
    ObjectCatalogue,
    ObjectSize,
)
from experiments.causal_reasoning.graspclutter6d.evaluation import (
    GroundTruthConfiguration,
    GroundTruthReport,
    ground_truth_study,
)
from experiments.causal_reasoning.graspclutter6d.queries import (
    ground_truth_cases,
    query_catalogue,
)
from experiments.causal_reasoning.graspclutter6d.synthetic_scm import (
    Cause,
    CausalQuestion,
    Effect,
    Intervention,
    SceneStructuralCausalModel,
)


@pytest.fixture
def model() -> SceneStructuralCausalModel:
    return SceneStructuralCausalModel(object_count_centre=10, confounding_strength=0.6)


def test_every_scene_holds_a_number_of_objects_in_the_models_range(model):
    scenes = model.scenes(50, np.random.default_rng(0))
    assert all(len(scene.objects) in model.object_count_range for scene in scenes)
    assert all(len(scene.viewpoints) == model.viewpoint_count for scene in scenes)


def test_the_small_objects_are_listed_first(model):
    for scene in model.scenes(50, np.random.default_rng(0)):
        sizes = [one.size for one in scene.objects]
        first_large = next(
            (index for index, size in enumerate(sizes) if size is ObjectSize.LARGE),
            len(sizes),
        )
        assert all(size is ObjectSize.LARGE for size in sizes[first_large:])


def test_graspability_of_every_object_is_the_scene_flag(model):
    for scene in model.scenes(50, np.random.default_rng(0)):
        assert scene.all_objects_graspable == all(
            one.graspability.value == "graspable" for one in scene.objects
        )


def test_forcing_the_small_object_count_sets_it_exactly(model):
    forced = model.sample(
        2000, np.random.default_rng(0), Intervention(Cause.SMALL_OBJECT_COUNT, 4)
    )
    assert np.all(forced.object_small.sum(axis=1) == 4)
    assert np.all(forced.object_count >= 4)


def test_forcing_the_occluded_object_count_sets_it_exactly(model):
    forced = model.sample(
        2000, np.random.default_rng(0), Intervention(Cause.OCCLUDED_OBJECT_COUNT, 3)
    )
    assert np.all(forced.object_occluded.sum(axis=1) == 3)


def test_forcing_the_catalogue_sets_every_scene(model):
    forced = model.sample(
        500,
        np.random.default_rng(0),
        Intervention(Cause.CATALOGUE, ObjectCatalogue.GRASP),
    )
    assert np.all(forced.catalogue_is_grasp)


def test_a_large_object_loses_its_grasps_more_often_than_a_small_one(model):
    random_state = np.random.default_rng(0)
    small = model.interventional_probability(
        Intervention(Cause.OBJECT_SIZE, ObjectSize.SMALL),
        Effect.OBJECT_BLOCKED,
        random_state,
        50_000,
    )
    large = model.interventional_probability(
        Intervention(Cause.OBJECT_SIZE, ObjectSize.LARGE),
        Effect.OBJECT_BLOCKED,
        random_state,
        50_000,
    )
    assert large > small + 0.05


def test_clear_viewpoints_have_no_effect(model):
    random_state = np.random.default_rng(0)
    rates = [
        model.interventional_probability(
            Intervention(Cause.CLEAR_VIEWPOINT_COUNT, count),
            Effect.ALL_OBJECTS_GRASPABLE,
            random_state,
            50_000,
        )
        for count in (0, 4, 8)
    ]
    assert max(rates) - min(rates) < 0.02


def test_the_crowded_scenes_hold_more_small_objects_at_full_strength():
    for strength in (0.0, 0.6):
        model = SceneStructuralCausalModel(
            object_count_centre=10, confounding_strength=strength
        )
        scenes = GraspClutterDataset(model.scenes(600, np.random.default_rng(0)))
        share = {}
        for scene in scenes.scenes:
            count = len(scene.objects)
            small = GraspClutterSceneAggregations(instance=scene).small_object_count()
            share.setdefault(count, []).append(small / count)
        smallest, largest = min(share), max(share)
        gap = float(np.mean(share[largest]) - np.mean(share[smallest]))
        assert gap == pytest.approx(strength, abs=0.15)


def test_every_question_of_the_catalogue_has_a_reading_in_the_model():
    for case in query_catalogue() + ground_truth_cases():
        question = CausalQuestion.of(case)
        assert isinstance(question.cause, Cause)
        assert isinstance(question.effect, Effect)


def test_the_study_scores_every_pipeline_under_every_setting():
    report = ground_truth_study(
        configurations=[GroundTruthConfiguration(5, 0.3)],
        scene_count=60,
        ordering_count=1,
        min_samples_per_leaf=0.2,
        plain_min_samples_per_leaf=0.2,
        cases=ground_truth_cases()[3:5],
        min_region_support=5,
        truth_sample_count=20_000,
    )
    assert report.configurations == [GroundTruthConfiguration(5, 0.3)]
    assert len(report.pipeline_names) == 4
    relational = report.of("relational circuit", ordering=0)
    assert [outcome.case.name for outcome in relational] == [
        case.name for case in ground_truth_cases()[3:5]
    ]
    assert all(
        0.0 <= error.true_probability <= 1.0
        for outcome in relational
        for error in outcome.errors
    )
    assert report.of("relational circuit", ordering=1) == []
    assert len(report.of("unrolled tree", ordering=1)) == 2
    weighted = GroundTruthReport.weighted_absolute_error(relational)
    assert 0.0 <= weighted <= 1.0
