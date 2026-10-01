"""
Counting the grasps a scene leaves its objects, and keeping those counts in an index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.causal_reasoning.graspclutter6d.exceptions import (
    GraspLabelsMissingError,
    SceneObjectCountMismatchError,
)
from experiments.causal_reasoning.graspclutter6d.grasp_labels import (
    TOOLKIT_FRICTION_COEFFICIENT,
    GraspCountIndex,
    GraspLabelField,
    GraspLabels,
    LabelFolder,
)

SCENE_ID = "000042"
"""
The scene the written labels stand for.
"""


@pytest.fixture
def written_labels(tmp_path) -> GraspLabels:
    """
    Labels for two object models in one scene: the first has one candidate of each kind
    (too much friction, not antipodal, colliding, and one usable grasp), the second has
    none that survive.
    """
    friction = {
        1: np.array([TOOLKIT_FRICTION_COEFFICIENT + 0.5, -1.0, 0.2, 0.1]),
        2: np.array([0.1, 0.2]),
    }
    collisions = {1: np.array([False, False, True, False]), 2: np.array([True, True])}
    grasp_directory = tmp_path / LabelFolder.GRASP_CANDIDATES
    grasp_directory.mkdir()
    for object_id, values in friction.items():
        np.savez(
            grasp_directory / f"obj_{object_id:06d}_labels.npz",
            **{GraspLabelField.FRICTION_COEFFICIENT.value: values},
        )
    collision_directory = tmp_path / LabelFolder.SCENE_COLLISIONS
    collision_directory.mkdir()
    np.savez(collision_directory / f"{SCENE_ID}.npz", collisions[1], collisions[2])
    return GraspLabels(directory=tmp_path)


def test_a_grasp_counts_when_it_is_antipodal_enough_and_does_not_collide(
    written_labels,
):
    assert written_labels.grasp_counts(SCENE_ID, [1, 2]) == [1, 0]


def test_raising_the_friction_it_may_need_admits_more_grasps(written_labels):
    written_labels.friction_coefficient = 1.0
    written_labels.antipodal_candidates.clear()
    assert written_labels.grasp_counts(SCENE_ID, [1, 2])[0] == 2


def test_labels_that_are_not_there_say_which_file_is_missing(written_labels):
    with pytest.raises(GraspLabelsMissingError) as raised:
        written_labels.grasp_counts("000999", [1])
    assert raised.value.missing_paths == [written_labels.scene_label_path("000999")]


def test_a_scene_whose_labels_do_not_match_its_objects_says_so(written_labels):
    with pytest.raises(SceneObjectCountMismatchError) as raised:
        written_labels.grasp_counts(SCENE_ID, [1, 2, 1])
    assert (raised.value.instance_count, raised.value.collision_array_count) == (3, 2)


def test_the_index_reads_a_scene_once(written_labels):
    index = GraspCountIndex()
    first = index.counts(written_labels, SCENE_ID, [1, 2])
    written_labels.directory = Path("/nowhere")
    assert index.counts(written_labels, SCENE_ID, [1, 2]) == first


def test_the_index_survives_being_written_and_read(written_labels, tmp_path):
    index = GraspCountIndex()
    index.counts(written_labels, SCENE_ID, [1, 2])
    path = tmp_path / "index.json"
    index.write(path)
    assert GraspCountIndex.read(path).counts_by_scene == index.counts_by_scene


def test_an_index_that_was_never_written_is_empty(tmp_path):
    assert GraspCountIndex.read(tmp_path / "absent.json").counts_by_scene == {}
