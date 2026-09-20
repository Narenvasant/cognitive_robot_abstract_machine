"""
The levels a measurement is read as, which camera recorded a frame, and what the scene's
aggregation statistics count.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.graspclutter6d.dataset import graspability_summaries
from experiments.causal_reasoning.graspclutter6d.domain import (
    CameraModel,
    GraspClutterSceneAggregations,
    MeasurementLevels,
    ObjectSize,
    Occlusion,
    Proximity,
    ViewClarity,
)


def test_each_level_holds_an_equal_share_of_the_values():
    values = list(range(300))
    levels = MeasurementLevels.from_values(values, list(ObjectSize))
    counts = {size: 0 for size in ObjectSize}
    for value in values:
        counts[levels.level_of(value)] += 1
    assert set(counts.values()) == {100}


def test_a_value_below_every_boundary_is_the_lowest_level():
    levels = MeasurementLevels.from_values([1.0, 2.0, 3.0], list(Proximity))
    assert levels.level_of(0.0) is Proximity.NEAR


def test_a_value_above_every_boundary_is_the_highest_level():
    levels = MeasurementLevels.from_values([1.0, 2.0, 3.0], list(Proximity))
    assert levels.level_of(10.0) is Proximity.FAR


def test_a_boundary_itself_falls_in_the_higher_level():
    levels = MeasurementLevels(levels=tuple(Proximity), boundaries=(1.0,))
    assert levels.level_of(1.0) is Proximity.FAR


def test_the_first_four_frames_are_the_four_cameras_in_order():
    assert [CameraModel.of_frame(image_id) for image_id in range(1, 5)] == list(
        CameraModel
    )


def test_a_viewpoints_four_frames_are_the_four_cameras_again():
    assert CameraModel.of_frame(5) is CameraModel.of_frame(1)


def test_the_aggregations_count_the_parts_matching_them(synthetic_scenes):
    scene = synthetic_scenes[0]
    aggregations = GraspClutterSceneAggregations(instance=scene)
    assert aggregations.small_object_count() == sum(
        one.size is ObjectSize.SMALL for one in scene.objects
    )
    assert aggregations.occluded_object_count() == sum(
        one.occlusion is not Occlusion.VISIBLE for one in scene.objects
    )
    assert aggregations.near_viewpoint_count() == sum(
        one.proximity is Proximity.NEAR for one in scene.viewpoints
    )
    assert aggregations.clear_viewpoint_count() == sum(
        one.clarity is ViewClarity.CLEAR for one in scene.viewpoints
    )


def test_the_aggregations_are_registered_over_both_part_fields():
    assert set(GraspClutterSceneAggregations.aggregation_registry) == {
        "objects",
        "viewpoints",
    }


def test_shuffling_the_parts_keeps_the_aggregations(synthetic_dataset):
    shuffled = synthetic_dataset.with_shuffled_parts(np.random.default_rng(1))
    for before, after in zip(synthetic_dataset.examples, shuffled.examples):
        assert (
            GraspClutterSceneAggregations(instance=after).small_object_count()
            == GraspClutterSceneAggregations(instance=before).small_object_count()
        )


def test_splitting_keeps_every_scene(synthetic_dataset):
    training, test = synthetic_dataset.split(0.8, np.random.default_rng(0))
    assert len(training.examples) + len(test.examples) == len(
        synthetic_dataset.examples
    )


def test_the_graspable_rate_by_a_key_sums_to_the_whole(synthetic_dataset):
    by_catalogue = synthetic_dataset.effect_rate_by(
        lambda scene: scene.object_catalogue
    )
    assert sum(rate.example_count for rate in by_catalogue.values()) == len(
        synthetic_dataset.examples
    )
    assert sum(rate.effect_count for rate in by_catalogue.values()) == pytest.approx(
        synthetic_dataset.effect_rate * len(synthetic_dataset.examples)
    )


def test_the_graspability_summaries_cover_every_scene(synthetic_dataset):
    summaries = graspability_summaries(synthetic_dataset)
    assert list(summaries) == [
        "object catalogue",
        "small objects",
        "occluded objects",
        "objects",
    ]
    for rates in summaries.values():
        assert sum(rate.example_count for rate in rates.values()) == len(
            synthetic_dataset.examples
        )
