"""
Reading how many of an object's annotated grasps survive the scene it stands in.

GraspClutter6D annotates every object model with antipodal grasp candidates and the
friction coefficient each needs, and every scene with which of its objects' candidates
collide with the rest of that scene. A candidate counts as a grasp of the object in the
scene when it is antipodal at the friction the dataset's own toolkit asks for and does
not collide, which is what makes graspability a property of the clutter rather than of
the object alone.

The labels are large (about a third of a terabyte extracted), so the counts read off them
are kept in an index beside the dataset and computed only once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import Any, Dict, List, Self, Sequence

from experiments.causal_reasoning.graspclutter6d.exceptions import (
    GraspLabelsMissingError,
    SceneObjectCountMismatchError,
)

TOOLKIT_FRICTION_COEFFICIENT = 0.4
"""
The friction coefficient below which a candidate counts as a grasp, as
``graspclutter6dAPI.GraspClutter6D.loadGrasp`` defaults to; the lower the coefficient a
grasp needs, the more firmly it holds.
"""

NUMPY_UNNAMED_ARRAY_PREFIX = "arr_"
"""
The name NumPy gives the arrays of an archive written without keywords, which is how the
dataset writes one array of collision flags per object instance.
"""


class LabelFolder(StrEnum):
    """
    The folders the dataset's label archives extract into.
    """

    GRASP_CANDIDATES = "grasp_label"
    """
    One file per object model, holding its grasp candidates.
    """

    SCENE_COLLISIONS = "collision_label"
    """
    One file per scene, holding which of its objects' candidates collide with the rest
    of it.
    """


class GraspLabelField(StrEnum):
    """
    The arrays a grasp label file holds.
    """

    FRICTION_COEFFICIENT = "scores"
    """
    Per candidate, the friction coefficient it needs, or a value of zero or less where
    the candidate is not antipodal at all.
    """


@dataclass
class GraspLabels:
    """
    The dataset's grasp and collision label archives as extracted on this machine.
    """

    directory: Path
    """
    The directory holding the extracted ``grasp_label`` and ``collision_label``
    folders.
    """

    friction_coefficient: float = TOOLKIT_FRICTION_COEFFICIENT
    """
    The highest friction coefficient a candidate may need to count as a grasp.
    """

    antipodal_candidates: Dict[int, np.ndarray] = field(
        default_factory=dict, repr=False
    )
    """
    Per object model, which of its candidates are antipodal at
    :attr:`friction_coefficient`, kept because a scene's objects recur across scenes and
    an object's labels are hundreds of megabytes to read.
    """

    @property
    def grasp_label_directory(self) -> Path:
        """
        The folder holding one file of grasp candidates per object model.
        """
        return self.directory / LabelFolder.GRASP_CANDIDATES

    @property
    def collision_label_directory(self) -> Path:
        """
        The folder holding one file of collision flags per scene.
        """
        return self.directory / LabelFolder.SCENE_COLLISIONS

    def object_label_path(self, object_id: int) -> Path:
        """
        :param object_id: An object model's id.
        :return: The file holding its grasp candidates.
        """
        return self.grasp_label_directory / f"obj_{object_id:06d}_labels.npz"

    def scene_label_path(self, scene_id: str) -> Path:
        """
        :param scene_id: A scene's id.
        :return: The file holding which of its objects' candidates collide.
        """
        return self.collision_label_directory / f"{scene_id}.npz"

    @property
    def available(self) -> bool:
        """
        Whether both label folders are on this machine.
        """
        return (
            self.grasp_label_directory.is_dir()
            and self.collision_label_directory.is_dir()
        )

    def _antipodal(self, object_id: int) -> np.ndarray:
        """
        :param object_id: An object model's id.
        :return: Which of its candidates are antipodal at :attr:`friction_coefficient`.
        :raises GraspLabelsMissingError: If the object's labels are not on this machine.
        """
        if object_id in self.antipodal_candidates:
            return self.antipodal_candidates[object_id]
        path = self.object_label_path(object_id)
        if not path.is_file():
            raise GraspLabelsMissingError([path])
        friction = np.load(path)[GraspLabelField.FRICTION_COEFFICIENT]
        self.antipodal_candidates[object_id] = (friction > 0) & (
            friction <= self.friction_coefficient
        )
        return self.antipodal_candidates[object_id]

    def grasp_counts(self, scene_id: str, object_ids: Sequence[int]) -> List[int]:
        """
        Count the grasps every object instance of a scene keeps.

        :param scene_id: The scene to read.
        :param object_ids: The scene's object instances, in the order its ground truth
            lists them, which is the order its collision flags follow.
        :return: One count per instance.
        :raises GraspLabelsMissingError: If the scene's collision flags are not on this
            machine.
        :raises SceneObjectCountMismatchError: If the collision flags do not hold one
            array per instance.
        """
        path = self.scene_label_path(scene_id)
        if not path.is_file():
            raise GraspLabelsMissingError([path])
        collisions = np.load(path)
        if len(collisions.files) != len(object_ids):
            raise SceneObjectCountMismatchError(
                scene_id, len(object_ids), len(collisions.files)
            )
        return [
            int(
                (
                    self._antipodal(object_id)
                    & ~collisions[f"{NUMPY_UNNAMED_ARRAY_PREFIX}{index}"]
                ).sum()
            )
            for index, object_id in enumerate(object_ids)
        ]


class GraspCountIndexField(StrEnum):
    """
    The keys a written :class:`GraspCountIndex` holds its counts under.
    """

    COUNTS_BY_SCENE = "counts_by_scene"


@dataclass
class GraspCountIndex(SubclassJSONSerializer):
    """
    How many grasps every object instance of every scene read so far keeps.
    """

    counts_by_scene: Dict[str, List[int]] = field(default_factory=dict)
    """
    Per scene id, one count per object instance, in the order the scene's ground truth
    lists them.
    """

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            GraspCountIndexField.COUNTS_BY_SCENE: self.counts_by_scene,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(counts_by_scene=data[GraspCountIndexField.COUNTS_BY_SCENE])

    @classmethod
    def read(cls, path: Path) -> Self:
        """
        :param path: Where the index is kept.
        :return: The index, empty if it was never written.
        """
        if not path.is_file():
            return cls()
        return cls.from_json(json.loads(path.read_text()))

    def write(self, path: Path) -> None:
        """
        :param path: Where to keep the index.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()))

    def counts(
        self, labels: GraspLabels, scene_id: str, object_ids: Sequence[int]
    ) -> List[int]:
        """
        The grasps a scene's object instances keep, read off the labels the first time
        the scene is asked for.

        :param labels: The labels to read from.
        :param scene_id: The scene to count for.
        :param object_ids: The scene's object instances, in ground-truth order.
        :return: One count per instance.
        """
        if scene_id not in self.counts_by_scene:
            self.counts_by_scene[scene_id] = labels.grasp_counts(scene_id, object_ids)
        return self.counts_by_scene[scene_id]
