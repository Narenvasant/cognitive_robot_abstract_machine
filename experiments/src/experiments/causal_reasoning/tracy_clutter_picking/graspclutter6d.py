"""
Reading a GraspClutter6D scene (https://sites.google.com/view/graspclutter6d) into the
layout an attempt runs on.

GraspClutter6D stores its scenes in the BOP format: per scene, ``scene_gt.json`` lists
every object's id and pose in each image's camera frame, and ``scene_camera.json``
lists each image's camera pose in the scene's world frame. Standing the objects on
the table from those poses makes a
:class:`~experiments.causal_reasoning.tracy_clutter_picking.domain.ClutterSceneLayout` of them,
so the same attempt and the same pipelines apply once the objects' own models are
annotated in the semantic digital twin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import Any, ClassVar, Dict, List, Self

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterEnvironment,
    ClutterSceneLayout,
    ObjectCategory,
    PlacedObject,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import UnknownSceneRecordError


class SceneFile(StrEnum):
    """
    The files of one BOP-format scene directory.
    """

    OBJECT_POSES = "scene_gt.json"
    CAMERA_POSES = "scene_camera.json"


class ObjectPoseKey(StrEnum):
    """
    The keys of one object's record in ``scene_gt.json``.
    """

    OBJECT_ID = "obj_id"
    ROTATION = "cam_R_m2c"
    TRANSLATION = "cam_t_m2c"


class CameraPoseKey(StrEnum):
    """
    The keys of one image's record in ``scene_camera.json``.
    """

    ROTATION = "cam_R_w2c"
    TRANSLATION = "cam_t_w2c"


@dataclass(frozen=True)
class RigidTransform:
    """
    A rotation and a translation, as BOP writes them.
    """

    rotation: np.ndarray
    """
    The 3x3 rotation matrix.
    """

    translation: np.ndarray
    """
    The translation, in metres.
    """

    millimetres_per_metre: ClassVar[float] = 1000.0
    """
    BOP poses are in millimetres.
    """

    @classmethod
    def from_json(cls, rotation: List[float], translation: List[float]) -> Self:
        """
        :param rotation: The rotation's nine entries, row by row.
        :param translation: The translation's three entries, in millimetres.
        :return: The transform, in metres.
        """
        return cls(
            rotation=np.array(rotation, dtype=float).reshape(3, 3),
            translation=np.array(translation, dtype=float) / cls.millimetres_per_metre,
        )

    def inverse(self) -> RigidTransform:
        """
        :return: The transform undoing this one.
        """
        return RigidTransform(
            rotation=self.rotation.T, translation=-self.rotation.T @ self.translation
        )

    def compose(self, other: RigidTransform) -> RigidTransform:
        """
        :param other: The transform applied first.
        :return: This transform applied after ``other``.
        """
        return RigidTransform(
            rotation=self.rotation @ other.rotation,
            translation=self.rotation @ other.translation + self.translation,
        )

    @property
    def yaw(self) -> float:
        """
        Rotation about the vertical, in radians.
        """
        return float(math.atan2(self.rotation[1, 0], self.rotation[0, 0]))


@dataclass(frozen=True)
class ObjectPose:
    """
    One object of a scene, in the scene's world frame.
    """

    object_id: int
    """
    GraspClutter6D's id of the object, 1 to 200.
    """

    world_transform: RigidTransform
    """
    The object's pose in the scene's world frame.
    """


@dataclass
class GraspClutter6DScene:
    """
    The objects of one GraspClutter6D scene as seen in one of its images, in the scene's
    world frame.
    """

    objects: List[ObjectPose]
    """
    Every object the image's record lists.
    """

    @classmethod
    def from_json(
        cls, object_poses: Dict[str, Any], camera_poses: Dict[str, Any], image_id: int
    ) -> Self:
        """
        Read the objects of one image.

        :param object_poses: The parsed ``scene_gt.json``.
        :param camera_poses: The parsed ``scene_camera.json``.
        :param image_id: The image whose records to read.
        :return: The scene's objects.
        :raises UnknownSceneRecordError: If the image has no record.
        """
        key = str(image_id)
        if key not in object_poses or key not in camera_poses:
            raise UnknownSceneRecordError(image_id)
        camera_record = camera_poses[key]
        camera_in_world = RigidTransform.from_json(
            camera_record[CameraPoseKey.ROTATION],
            camera_record[CameraPoseKey.TRANSLATION],
        ).inverse()
        return cls(
            objects=[
                ObjectPose(
                    object_id=int(record[ObjectPoseKey.OBJECT_ID]),
                    world_transform=camera_in_world.compose(
                        RigidTransform.from_json(
                            record[ObjectPoseKey.ROTATION],
                            record[ObjectPoseKey.TRANSLATION],
                        )
                    ),
                )
                for record in object_poses[key]
            ]
        )

    @classmethod
    def from_directory(cls, scene_directory: Path, image_id: int) -> Self:
        """
        Read the objects of one image out of a scene directory.

        :param scene_directory: The scene's directory, holding the two pose files.
        :param image_id: The image whose records to read.
        :return: The scene's objects.
        """
        return cls.from_json(
            json.loads((scene_directory / SceneFile.OBJECT_POSES).read_text()),
            json.loads((scene_directory / SceneFile.CAMERA_POSES).read_text()),
            image_id,
        )

    def to_layout(
        self,
        target_index: int,
        environment: ClutterEnvironment,
        friction_coefficient: float,
        grasp_yaw: float,
        category_of: Dict[int, ObjectCategory],
    ) -> ClutterSceneLayout:
        """
        Stand the scene's objects on the table as a layout to attempt.

        :param target_index: Index into :attr:`objects` of the object to pick.
        :param environment: The kind of environment the scene was recorded in.
        :param friction_coefficient: The grasp contact's friction coefficient.
        :param grasp_yaw: The grasp's yaw relative to the target.
        :param category_of: What kind of object each GraspClutter6D object id is.
        :return: The layout.
        """
        return ClutterSceneLayout(
            environment=environment,
            objects=[
                PlacedObject(
                    category=category_of[pose.object_id],
                    x=float(pose.world_transform.translation[0]),
                    y=float(pose.world_transform.translation[1]),
                    yaw=pose.world_transform.yaw,
                )
                for pose in self.objects
            ],
            target_index=target_index,
            friction_coefficient=friction_coefficient,
            grasp_yaw=grasp_yaw,
        )
