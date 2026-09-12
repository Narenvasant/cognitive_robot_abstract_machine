"""
Tests for the synthetic attempts and their on-disk dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.tracy_rspn.dataset import ClutterPickDataset
from experiments.causal_reasoning.tracy_rspn.domain import (
    ClutterEnvironment,
    ClutterPickSceneAggregations,
    DistanceBand,
    FrictionLadder,
)
from experiments.causal_reasoning.tracy_rspn.layout_sampler import (
    ClutterLayoutSampler,
    EnvironmentDistribution,
)
from experiments.causal_reasoning.tracy_rspn.synthetic import (
    SyntheticPickOutcomes,
    synthetic_clutter_pick_scenes,
)


@pytest.fixture
def scenes():
    return synthetic_clutter_pick_scenes(np.random.default_rng(0), scene_count=30)


# %% layout sampling


def test_sampled_layout_has_the_requested_object_count():
    layout = ClutterLayoutSampler(np.random.default_rng(0), object_count=7).sample()
    assert len(layout.objects) == 7
    assert len(layout.neighbours) == 6


def test_sampled_friction_comes_from_the_environments_own_levels():
    distributions = EnvironmentDistribution.of_mock_environments()
    bin_only = {ClutterEnvironment.BIN: distributions[ClutterEnvironment.BIN]}
    sampler = ClutterLayoutSampler(np.random.default_rng(0), distributions=bin_only)
    for _ in range(20):
        layout = sampler.sample()
        assert layout.environment == ClutterEnvironment.BIN
        assert (
            layout.friction_coefficient
            in distributions[ClutterEnvironment.BIN].friction_levels
        )


# %% synthetic outcomes


def test_highest_friction_without_adjacent_neighbours_always_lifts():
    distributions = EnvironmentDistribution.of_mock_environments()
    table_only = {ClutterEnvironment.TABLE: distributions[ClutterEnvironment.TABLE]}
    sampler = ClutterLayoutSampler(np.random.default_rng(1), distributions=table_only)
    outcomes = SyntheticPickOutcomes(np.random.default_rng(1))
    lifted_count = 0
    for _ in range(20):
        layout = sampler.sample()
        layout.friction_coefficient = FrictionLadder().highest
        outcome = outcomes.simulate(layout)
        scene = outcome.to_scene(layout)
        if ClutterPickSceneAggregations(instance=scene).crowding_count() == 0:
            assert outcome.lifted
            assert outcome.lift_height == outcomes.lifted_height
            lifted_count += 1
    assert lifted_count > 0


def test_synthetic_scenes_record_every_neighbour(scenes):
    assert len(scenes) == 30
    for scene in scenes:
        assert len(scene.neighbours) == 9


# %% dataset persistence


def test_dataset_round_trips_through_json(tmp_path, scenes):
    path = tmp_path / "attempts.json"
    ClutterPickDataset(scenes).save(path)

    loaded = ClutterPickDataset.load(path)

    assert loaded.scenes == scenes
    assert type(loaded.scenes[0].environment) is ClutterEnvironment
    assert type(loaded.scenes[0].neighbours[0].distance_band) is DistanceBand


def test_dataset_split_keeps_every_scene_once(scenes):
    dataset = ClutterPickDataset(scenes)

    first, second = dataset.split(0.8, np.random.default_rng(0))

    assert len(first.scenes) == 24
    assert len(second.scenes) == 6
    assert sorted(map(id, first.scenes + second.scenes)) == sorted(map(id, scenes))


def test_success_rate_is_the_share_of_lifted_scenes(scenes):
    dataset = ClutterPickDataset(scenes)
    assert dataset.success_rate == pytest.approx(
        sum(scene.lifted for scene in scenes) / len(scenes)
    )


def test_success_rate_by_environment_counts_every_scene_once(scenes):
    dataset = ClutterPickDataset(scenes)

    rates = dataset.success_rate_by(lambda scene: scene.environment)

    assert sum(rate.attempt_count for rate in rates.values()) == len(scenes)
    assert sum(rate.lifted_count for rate in rates.values()) == sum(
        scene.lifted for scene in scenes
    )
    for environment, rate in rates.items():
        assert rate.rate == pytest.approx(
            sum(scene.lifted for scene in scenes if scene.environment == environment)
            / rate.attempt_count
        )
