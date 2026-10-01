"""
Reading a scene's BOP annotation files: what the files say, and the geometry derived
from them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.causal_reasoning.graspclutter6d.annotations import (
    MILLIMETERS_PER_METER,
    BopField,
    ExtractedScenes,
    SceneAnnotationFile,
)
from experiments.causal_reasoning.graspclutter6d.exceptions import (
    SceneAnnotationsUnavailableError,
)

from .conftest import SAMPLE_SCENE_ID


def _recorded(scenes_directory: Path, annotation_file: SceneAnnotationFile) -> dict:
    """
    :param scenes_directory: The checked-in scenes directory.
    :param annotation_file: Which annotation file to read.
    :return: What the file itself holds, to compare a parsed scene against.
    """
    return json.loads(
        (scenes_directory / SAMPLE_SCENE_ID / annotation_file.value).read_text()
    )


def test_every_frame_the_visibility_file_covers_is_parsed(
    sample_annotations, scenes_directory
):
    recorded = _recorded(scenes_directory, SceneAnnotationFile.VISIBILITY)
    assert [frame.image_id for frame in sample_annotations.frames] == sorted(
        int(image_id) for image_id in recorded
    )


def test_the_object_instances_are_the_ones_the_ground_truth_lists(
    sample_annotations, scenes_directory
):
    recorded = _recorded(scenes_directory, SceneAnnotationFile.GROUND_TRUTH)
    first_frame = min(recorded, key=int)
    assert sample_annotations.object_ids == [
        entry[BopField.OBJECT_ID] for entry in recorded[first_frame]
    ]


def test_every_frame_lists_the_same_object_instances_in_the_same_order(
    sample_annotations,
):
    orders = {
        tuple(placement.object_id for placement in frame.placements)
        for frame in sample_annotations.frames
    }
    assert orders == {tuple(sample_annotations.object_ids)}


def test_the_mean_visible_fraction_is_the_mean_of_the_recorded_ones(
    sample_annotations, scenes_directory
):
    recorded = _recorded(scenes_directory, SceneAnnotationFile.VISIBILITY)
    expected = np.array(
        [
            [entry[BopField.VISIBLE_FRACTION] for entry in recorded[image_id]]
            for image_id in sorted(recorded, key=int)
        ]
    ).mean(axis=0)
    assert sample_annotations.mean_visible_fractions() == pytest.approx(expected)


def test_a_position_is_read_in_meters(sample_annotations, scenes_directory):
    recorded = _recorded(scenes_directory, SceneAnnotationFile.GROUND_TRUTH)
    first_frame = min(recorded, key=int)
    expected = (
        np.asarray(recorded[first_frame][0][BopField.TRANSLATION_MODEL_TO_CAMERA])
        / MILLIMETERS_PER_METER
    )
    assert sample_annotations.frames[0].placements[0].camera_position == pytest.approx(
        expected
    )


def test_the_frames_agree_on_where_the_objects_lie_relative_to_each_other(
    sample_annotations,
):
    """
    The dataset calibrates each camera's world frame separately, so the frames need not
    agree on where the origin sits, but they do see the same arrangement of objects.
    """
    centred = [
        frame.world_positions() - frame.world_positions().mean(axis=0)
        for frame in sample_annotations.frames
    ]
    deviations = [np.abs(centred[0] - other).max() for other in centred[1:]]
    assert max(deviations) < 0.05


def test_the_spread_is_the_diagonal_and_the_height_of_the_objects_box(
    sample_annotations,
):
    expected = np.array(
        [np.ptp(frame.world_positions(), axis=0) for frame in sample_annotations.frames]
    ).mean(axis=0)
    extent, height_span = sample_annotations.object_position_spread()
    assert extent == pytest.approx(float(np.linalg.norm(expected)))
    assert height_span == pytest.approx(float(expected[2]))


def test_a_scene_that_is_not_there_says_so(scenes_directory):
    with pytest.raises(SceneAnnotationsUnavailableError) as raised:
        ExtractedScenes(directory=scenes_directory).scene("999999")
    assert raised.value.scene_id == "999999"
