"""
Tests for reading a GraspClutter6D scene into a layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.causal_reasoning.tracy_rspn.domain import (
    ClutterEnvironment,
    ObjectCategory,
)
from experiments.causal_reasoning.tracy_rspn.exceptions import UnknownSceneRecordError
from experiments.causal_reasoning.tracy_rspn.graspclutter6d import (
    GraspClutter6DScene,
)

SCENE_DIRECTORY = Path(__file__).parent.parent / "dataset" / "graspclutter6d_scene"
"""
A BOP-format scene of three objects seen by one camera standing a metre along x and
turned a quarter turn, so world and camera poses differ in both position and yaw.
"""

IMAGE_ID = 1
"""
The one image the scene fixture carries records for.
"""


@pytest.fixture
def scene() -> GraspClutter6DScene:
    return GraspClutter6DScene.from_directory(SCENE_DIRECTORY, IMAGE_ID)


def test_objects_are_read_back_into_the_world_frame(scene):
    assert [pose.object_id for pose in scene.objects] == [7, 7, 42]
    first = scene.objects[0].world_transform
    assert first.translation == pytest.approx([0.8, 0.25, 0.93])
    assert first.yaw == pytest.approx(0.1)
    assert scene.objects[1].world_transform.yaw == pytest.approx(-0.2)


def test_missing_image_record_raises(scene):
    with pytest.raises(UnknownSceneRecordError):
        GraspClutter6DScene.from_json(
            json.loads((SCENE_DIRECTORY / "scene_gt.json").read_text()),
            json.loads((SCENE_DIRECTORY / "scene_camera.json").read_text()),
            image_id=IMAGE_ID + 1,
        )


def test_layout_stands_every_object_on_the_table(scene):
    layout = scene.to_layout(
        target_index=1,
        environment=ClutterEnvironment.BIN,
        friction_coefficient=0.5,
        grasp_yaw=0.0,
        category_of={7: ObjectCategory.MILK, 42: ObjectCategory.MILK},
    )

    assert layout.environment == ClutterEnvironment.BIN
    assert layout.target.x == pytest.approx(0.87)
    assert layout.target.yaw == pytest.approx(-0.2)
    assert len(layout.neighbours) == 2
    assert layout.neighbours[0].distance_to(layout.target) == pytest.approx(0.07)
