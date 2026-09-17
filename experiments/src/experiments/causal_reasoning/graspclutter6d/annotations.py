"""
Reading a GraspClutter6D scene's BOP annotation files, either from an extracted copy of
the dataset or from a dataset server holding one.

:mod:`semantic_digital_twin.adapters.grasp_clutter_6d_dataset.schema` reads the same
files to build a world of bodies and meshes. This reads only the numbers the scene's
aggregation statistics are computed from, as plain arrays, because the experiment fits
over a thousand scenes of fifty-two frames each and never needs a pose as a symbolic
transformation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

import numpy as np
from semantic_digital_twin.adapters.dataset_server import DatasetServer
from semantic_digital_twin.exceptions import DatasetServerError, PathResolutionError
from typing_extensions import Any, Dict, List, Optional, Self, Tuple

from experiments.causal_reasoning.graspclutter6d.exceptions import (
    SceneAnnotationsUnavailableError,
)

MILLIMETERS_PER_METER = 1000.0
"""
GraspClutter6D records every length in millimeters, as the BOP format prescribes; the
rest of this package works in meters.
"""

# %% the payload's own names


class SceneAnnotationFile(StrEnum):
    """
    The annotation files a scene's directory holds beside its images.
    """

    CAMERA = "scene_camera.json"
    """
    Per frame, the camera's intrinsics, depth scale and pose in the scene's world frame.
    """

    GROUND_TRUTH = "scene_gt.json"
    """
    Per frame, the pose of every object instance relative to that camera.
    """

    VISIBILITY = "scene_gt_info.json"
    """
    Per frame, how much of every object instance the camera actually sees.
    """


class BopField(StrEnum):
    """
    The field names the BOP format gives the values inside the annotation files.
    """

    ROTATION_WORLD_TO_CAMERA = "cam_R_w2c"
    TRANSLATION_WORLD_TO_CAMERA = "cam_t_w2c"
    TRANSLATION_MODEL_TO_CAMERA = "cam_t_m2c"
    OBJECT_ID = "obj_id"
    VISIBLE_FRACTION = "visib_fract"


# %% what one frame records


@dataclass(frozen=True)
class CameraPose:
    """
    Where one frame's camera stands in the scene's own world frame.
    """

    rotation_world_to_camera: np.ndarray
    """
    The rotation taking a point from the world frame into the camera frame.
    """

    translation_world_to_camera: np.ndarray
    """
    The translation taking a point from the world frame into the camera frame, in
    meters.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        :param data: One entry of ``scene_camera.json``.
        :return: The pose it records.
        """
        return cls(
            rotation_world_to_camera=np.asarray(
                data[BopField.ROTATION_WORLD_TO_CAMERA], dtype=float
            ).reshape(3, 3),
            translation_world_to_camera=np.asarray(
                data[BopField.TRANSLATION_WORLD_TO_CAMERA], dtype=float
            ).reshape(3)
            / MILLIMETERS_PER_METER,
        )

    @property
    def position(self) -> np.ndarray:
        """
        The camera's own position in the world frame, in meters.
        """
        return -self.rotation_world_to_camera.T @ self.translation_world_to_camera

    def world_positions_of(self, camera_positions: np.ndarray) -> np.ndarray:
        """
        :param camera_positions: Positions in this camera's frame, in meters, one per
            row.
        :return: The same positions in the world frame.
        """
        return (
            camera_positions - self.translation_world_to_camera
        ) @ self.rotation_world_to_camera


@dataclass(frozen=True)
class ObjectPlacement:
    """
    One object instance as one frame sees it.
    """

    object_id: int
    """
    The id of the object's model, matching the ``obj_%06d`` numbering of the dataset's
    model files.
    """

    camera_position: np.ndarray
    """
    The object's position in the camera's frame, in meters.
    """

    visible_fraction: float
    """
    The share of the object's pixels this frame actually sees, the dataset's
    ``visib_fract``.
    """

    @classmethod
    def from_json(cls, pose: Dict[str, Any], visibility: Dict[str, Any]) -> Self:
        """
        :param pose: One entry of a frame's list in ``scene_gt.json``.
        :param visibility: The entry at the same position in ``scene_gt_info.json``.
        :return: The placement the two record together.
        """
        return cls(
            object_id=int(pose[BopField.OBJECT_ID]),
            camera_position=np.asarray(
                pose[BopField.TRANSLATION_MODEL_TO_CAMERA], dtype=float
            ).reshape(3)
            / MILLIMETERS_PER_METER,
            visible_fraction=float(visibility[BopField.VISIBLE_FRACTION]),
        )


@dataclass(frozen=True)
class AnnotatedFrame:
    """
    One frame of a scene: where its camera stood and what it saw.
    """

    image_id: int
    """
    The frame's number within the scene.

    GraspClutter6D records thirteen viewpoints with     four cameras each, numbered ``4
    * viewpoint + camera`` from one.
    """

    camera: CameraPose
    """
    Where the frame's camera stood.
    """

    placements: List[ObjectPlacement] = field(default_factory=list)
    """
    Every object instance the frame records, in the order the ground truth lists them,
    which is the order the scene's collision labels follow.
    """

    @property
    def visible_share(self) -> float:
        """
        The mean share of an object this frame sees.
        """
        return float(
            np.mean([placement.visible_fraction for placement in self.placements])
        )

    @property
    def distance_to_objects(self) -> float:
        """
        Distance from the camera to the objects' centre, in meters.
        """
        positions = np.array(
            [placement.camera_position for placement in self.placements]
        )
        return float(np.linalg.norm(positions.mean(axis=0)))

    def world_positions(self) -> np.ndarray:
        """
        :return: Every object instance's position in the scene's world frame, in meters,
            one per row and in :attr:`placements` order.
        """
        return self.camera.world_positions_of(
            np.array([placement.camera_position for placement in self.placements])
        )


@dataclass(frozen=True)
class SceneAnnotations:
    """
    Everything the annotation files of one scene record.
    """

    scene_id: str
    """
    The scene's id, e.g. ``"000005"``.
    """

    frames: List[AnnotatedFrame] = field(default_factory=list)
    """
    The scene's frames, in ascending frame number.
    """

    @property
    def object_ids(self) -> List[int]:
        """
        The id of every object instance of the scene, in the order the ground truth
        lists them.
        """
        return [placement.object_id for placement in self.frames[0].placements]

    def mean_visible_fractions(self) -> np.ndarray:
        """
        :return: Per object instance, the share of it the cameras see, averaged over
            every frame, in :attr:`object_ids` order.
        """
        return np.array(
            [
                [placement.visible_fraction for placement in frame.placements]
                for frame in self.frames
            ]
        ).mean(axis=0)

    def object_position_spread(self) -> Tuple[float, float]:
        """
        :return: The diagonal of the box bounding the objects' positions and their
            vertical spread, both in meters.

        Read in the world frame each frame records its camera against and averaged over
        the frames: the dataset calibrates that frame per camera, so the frames agree on
        the objects' positions relative to each other but not on where the origin sits.
        """
        spreads = np.array(
            [np.ptp(frame.world_positions(), axis=0) for frame in self.frames]
        ).mean(axis=0)
        return float(np.linalg.norm(spreads)), float(spreads[2])

    @classmethod
    def from_files(cls, scene_id: str, files: Dict[SceneAnnotationFile, str]) -> Self:
        """
        :param scene_id: The scene's id.
        :param files: The contents of the scene's three annotation files.
        :return: The parsed annotations, holding only the frames that carry a camera
            pose, object poses and visibility alike; the dataset also numbers a few
            calibration frames, which its visibility file does not cover.
        """
        cameras = json.loads(files[SceneAnnotationFile.CAMERA])
        ground_truth = json.loads(files[SceneAnnotationFile.GROUND_TRUTH])
        visibility = json.loads(files[SceneAnnotationFile.VISIBILITY])
        return cls(
            scene_id=scene_id,
            frames=[
                AnnotatedFrame(
                    image_id=int(image_id),
                    camera=CameraPose.from_json(cameras[image_id]),
                    placements=[
                        ObjectPlacement.from_json(pose, seen)
                        for pose, seen in zip(
                            ground_truth[image_id], visibility[image_id]
                        )
                    ],
                )
                for image_id in sorted(visibility, key=int)
                if image_id in cameras and image_id in ground_truth
            ],
        )


# %% where the annotation files are read from


@dataclass
class SceneAnnotationSource(ABC):
    """
    Somewhere a scene's annotation files can be read from.
    """

    @abstractmethod
    def read(self, scene_id: str, annotation_file: SceneAnnotationFile) -> str:
        """
        :param scene_id: The scene to read.
        :param annotation_file: Which of its annotation files to read.
        :return: The file's contents.
        :raises SceneAnnotationsUnavailableError: If the file cannot be read.
        """

    def scene(self, scene_id: str) -> SceneAnnotations:
        """
        :param scene_id: The scene to read.
        :return: Everything its annotation files record.
        """
        return SceneAnnotations.from_files(
            scene_id,
            {
                annotation_file: self.read(scene_id, annotation_file)
                for annotation_file in SceneAnnotationFile
            },
        )


@dataclass
class ExtractedScenes(SceneAnnotationSource):
    """
    The scenes of an extracted copy of the dataset on this machine.
    """

    directory: Path
    """
    The extracted ``scenes`` directory, holding one folder per scene.
    """

    def read(self, scene_id: str, annotation_file: SceneAnnotationFile) -> str:
        path = self.directory / scene_id / annotation_file.value
        if not path.is_file():
            raise SceneAnnotationsUnavailableError(scene_id, f"{path} does not exist")
        return path.read_text()


@dataclass
class ServedScenes(SceneAnnotationSource):
    """
    The scenes of a copy of the dataset held on a dataset server, read over http and
    kept in the server's local cache.

    Asking the server for one annotation file brings the scene's other two with it,
    since it answers a reference by copying the files of the directory holding it, and a
    scene's images sit in directories of their own.
    """

    server: DatasetServer
    """
    The server holding the dataset.
    """

    dataset_directory: str = "graspclutter6d-dataset"
    """
    The dataset's own directory on the server, below the server's dataset root.
    """

    scenes_directory: str = "scenes"
    """
    The directory holding one folder per scene, below :attr:`dataset_directory`.
    """

    def read(self, scene_id: str, annotation_file: SceneAnnotationFile) -> str:
        reference = str(
            PurePosixPath(self.server.dataset_root)
            / self.dataset_directory
            / self.scenes_directory
            / scene_id
            / annotation_file.value
        )
        try:
            path = Path(self.server.resolve(reference))
        except (DatasetServerError, PathResolutionError) as error:
            raise SceneAnnotationsUnavailableError(scene_id, str(error)) from error
        if not path.is_file():
            raise SceneAnnotationsUnavailableError(
                scene_id, f"the server holds no {annotation_file.value}"
            )
        return path.read_text()


def scene_annotation_source(
    extracted_scenes: Optional[Path] = None,
) -> SceneAnnotationSource:
    """
    The source the environment offers, preferring an extracted copy on this machine over
    the dataset server.

    :param extracted_scenes: An extracted ``scenes`` directory to read from, if there is
        one.
    :return: The source to read scenes from.
    :raises SceneAnnotationsUnavailableError: If neither is available.
    """
    if extracted_scenes is not None and extracted_scenes.is_dir():
        return ExtractedScenes(directory=extracted_scenes)
    server = DatasetServer.from_environment()
    if server is None:
        raise SceneAnnotationsUnavailableError(
            "any", "no extracted scenes directory and no dataset server configured"
        )
    return ServedScenes(server=server)
