"""
Tests running one pick attempt in MuJoCo; skipped where Tracy's description is not
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.tracy_rspn.domain import FRICTION_LEVELS
from experiments.causal_reasoning.tracy_rspn.episode import LIFT_THRESHOLD, PickEpisode
from experiments.causal_reasoning.tracy_rspn.layout_sampler import (
    ClutterLayoutSampler,
)
from experiments.causal_reasoning.tracy_rspn.scene import MilkClutterWorld, milk_name
from semantic_digital_twin.utils import tracy_installed

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)


@pytest.fixture
def layout():
    return ClutterLayoutSampler(np.random.default_rng(0)).sample()


def test_scene_stands_every_carton_on_the_table(layout):
    scene = MilkClutterWorld(layout)

    assert [milk.name.name for milk in scene.milks] == [
        milk_name(index) for index in range(len(layout.objects))
    ]
    assert scene.target is scene.milks[layout.target_index]
    assert len(scene.neighbours) == len(layout.objects) - 1
    assert scene.actuators


def test_high_friction_uncrowded_target_is_lifted(layout, tmp_path):
    """
    The surest attempt there is: the highest friction level, and the target moved out to
    the edge of the clutter so no neighbour stands in the fingers' way.
    """
    layout.friction_coefficient = max(FRICTION_LEVELS)
    layout.target_index = 0
    layout.objects[0].x -= 0.15
    layout.objects[0].y -= 0.15

    outcome = PickEpisode(screenshot_directory=tmp_path).run(layout)

    assert outcome.lifted
    assert outcome.lift_height > LIFT_THRESHOLD
    assert len(outcome.neighbour_displacements) == len(layout.neighbours)
    assert (tmp_path / "before_pick.png").exists()
    assert (tmp_path / "after_pick.png").exists()
