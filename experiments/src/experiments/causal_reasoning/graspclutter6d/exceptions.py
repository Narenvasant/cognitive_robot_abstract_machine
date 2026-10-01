"""
Exceptions for the GraspClutter6D causal-reasoning experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException
from typing_extensions import List


@dataclass
class SceneAnnotationsUnavailableError(DataclassException):
    """
    Raised when a scene's BOP annotation files can be reached neither on disk nor on the
    dataset server.
    """

    scene_id: str
    """
    The scene that was asked for.
    """

    reason: str
    """
    What the read failed with.
    """

    def error_message(self) -> str:
        return (
            f"Could not read the annotations of GraspClutter6D scene "
            f"{self.scene_id}: {self.reason}"
        )

    def suggest_correction(self) -> str:
        return (
            "Point SEMANTIC_DIGITAL_TWIN_DATASET_SERVER at a server holding the "
            "extracted dataset, or extract scenes.7z and pass its scenes directory."
        )


@dataclass
class GraspLabelsMissingError(DataclassException):
    """
    Raised when the grasp or collision labels a scene's graspability is read off are not
    on disk.
    """

    missing_paths: List[Path]
    """
    The files that were looked for.
    """

    def error_message(self) -> str:
        return f"The GraspClutter6D grasp labels {self.missing_paths} are missing."

    def suggest_correction(self) -> str:
        return (
            "Download and extract grasp_label.7z and collision_label.7z from "
            "https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D."
        )


@dataclass
class SceneObjectCountMismatchError(DataclassException):
    """
    Raised when a scene's collision labels do not hold one array per object instance the
    scene's ground truth lists, so the two cannot be read side by side.
    """

    scene_id: str
    """
    The scene whose labels were read.
    """

    instance_count: int
    """
    How many object instances the ground truth lists.
    """

    collision_array_count: int
    """
    How many arrays the collision labels hold.
    """

    def error_message(self) -> str:
        return (
            f"GraspClutter6D scene {self.scene_id} lists {self.instance_count} object "
            f"instances but its collision labels hold "
            f"{self.collision_array_count} arrays."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the collision labels and the scene annotations come from the "
            "same release of the dataset."
        )
