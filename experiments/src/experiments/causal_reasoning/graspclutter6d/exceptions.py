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


@dataclass
class FlatTableSchemaMismatchError(DataclassException):
    """
    Raised when a query constrains a flat-table model on variables the table it was
    fitted on never had, such as one object's size: the table carries a scene's own
    scalars and its aggregation counts, not its objects or viewpoints.
    """

    missing_variable_names: List[str]
    """
    The constrained variable names the fitted table does not carry.
    """

    def error_message(self) -> str:
        return (
            "The flat-table model has no column for the queried variables "
            f"{self.missing_variable_names}."
        )

    def suggest_correction(self) -> str:
        return (
            "Ask about the scene's own scalars and aggregation counts, or use the "
            "relational pipeline, which grounds a circuit over the queried objects and "
            "viewpoints."
        )


@dataclass
class OneCausePerQueryError(DataclassException):
    """
    Raised when a query marks more than one variable as its cause: each pipeline fits
    one support-deterministic model per cause variable, which cannot serve two at once.
    """

    cause_names: List[str]
    """
    The names of the variables the query marked.
    """

    def error_message(self) -> str:
        return f"The query marks {self.cause_names} as causes; only one is supported."

    def suggest_correction(self) -> str:
        return "Ask one query per candidate cause."


@dataclass
class PipelineNotFittedError(DataclassException):
    """
    Raised when a pipeline is asked for a model before it was fitted.
    """

    pipeline_name: str
    """
    The pipeline that was asked.
    """

    def error_message(self) -> str:
        return f"The {self.pipeline_name} pipeline has not been fitted yet."

    def suggest_correction(self) -> str:
        return "Call fit(scenes) first."
