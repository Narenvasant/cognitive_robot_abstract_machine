"""
A kitchen around Tracy's table, for showing the experiment: models of AICOR's object
library placed as fixed bodies with visual geometry only, so nothing about the physics
of a pick changes.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

from semantic_digital_twin.specifications.connections import (
    FixedConnectionSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

# %% one model of the library


@dataclass(frozen=True)
class PlacedModel:
    """
    One model of the object library and where it stands.
    """

    directory: str
    """
    The model's folder, relative to the library's ``objects/articulated``.
    """

    x: float
    """
    Where the model's origin stands along the world's x-axis, in metres.
    """

    y: float
    """
    Where it stands along the world's y-axis, in metres.
    """

    yaw: float
    """
    Its rotation about the vertical, in radians.
    """


@dataclass(frozen=True)
class ModelPart:
    """
    One visual mesh of a model, where the model's own frame holds it.
    """

    name: str
    """
    The link the mesh belongs to.
    """

    mesh: Path
    """
    The mesh's file.
    """

    model_T_part: np.ndarray
    """
    Where the part stands in the model's frame, with every joint at zero.
    """

    @classmethod
    def all_of(cls, urdf: Path) -> list[ModelPart]:
        """
        :param urdf: A model's URDF.
        :return: Every visual mesh of it, placed with every joint at zero.
        """
        root = ElementTree.parse(urdf).getroot()
        parent_of = {
            joint.find("child").get("link"): (
                joint.find("parent").get("link"),
                cls.transformation_of(joint.find("origin")),
            )
            for joint in root.iter("joint")
        }

        def model_T(link: str) -> np.ndarray:
            if link not in parent_of:
                return np.eye(4)
            parent, parent_T_link = parent_of[link]
            return model_T(parent) @ parent_T_link

        return [
            cls(
                link.get("name"),
                urdf.parent / visual.find("geometry/mesh").get("filename"),
                model_T(link.get("name"))
                @ cls.transformation_of(visual.find("origin")),
            )
            for link in root.iter("link")
            for visual in link.iter("visual")
            if visual.find("geometry/mesh") is not None
        ]

    @staticmethod
    def transformation_of(origin: ElementTree.Element | None) -> np.ndarray:
        """
        :param origin: A URDF ``<origin>``, or None.
        :return: The transformation it names.
        """
        if origin is None:
            return np.eye(4)
        x, y, z = (float(value) for value in origin.get("xyz", "0 0 0").split())
        roll, pitch, yaw = (
            float(value) for value in origin.get("rpy", "0 0 0").split()
        )
        return trimesh.transformations.compose_matrix(
            angles=(roll, pitch, yaw), translate=(x, y, z)
        )


# %% the backdrop


@dataclass
class KitchenBackdrop:
    """
    A kitchen run with a fridge beside it, behind Tracy and facing its table.
    """

    library: Path
    """
    The object library's ``objects/articulated`` folder.
    """

    cache: Path = field(
        default_factory=lambda: Path.home() / ".cramera" / "backdrop_meshes"
    )
    """
    Where the converted meshes are kept.
    """

    models: list[PlacedModel] = field(
        default_factory=lambda: [
            PlacedModel(
                "kitchen-appliances/103-kitchen_run_cabinets", -1.4, -0.1, -math.pi / 2
            ),
            PlacedModel(
                "kitchen-appliances/034-fridge_freezer", -1.45, 1.65, -math.pi / 2
            ),
        ]
    )
    """
    The models and where they stand.
    """

    def add_to(self, world: World) -> None:
        """
        Stand every model's parts in the world, fixed and drawn only.

        :param world: The world to furnish.
        """
        with world.modify_world():
            for model in self.models:
                directory = self.library / model.directory
                parts = ModelPart.all_of(directory / "model.urdf")
                upright = self.upright_rotation(parts)
                world_T_model = trimesh.transformations.compose_matrix(
                    angles=(0.0, 0.0, model.yaw), translate=(model.x, model.y, 0.0)
                )
                for part in parts:
                    self.stand(world, part, world_T_model @ part.model_T_part, upright)

    def upright_rotation(self, parts: list[ModelPart]) -> np.ndarray:
        """
        :param parts: A model's parts, its body first.
        :return: The rotation that stands the model's meshes upright: a mesh drawn with
            its height along y is turned onto z, as glTF's own axes are.
        """
        extents = self.loaded(parts[0].mesh).extents
        if extents[1] > extents[2]:
            return trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0))
        return np.eye(4)

    def stand(
        self,
        world: World,
        part: ModelPart,
        world_T_part: np.ndarray,
        upright: np.ndarray,
    ) -> None:
        """
        :param world: The world.
        :param part: One mesh of a model.
        :param world_T_part: Where it stands.
        :param upright: The rotation standing its mesh upright.
        """
        mesh = self.converted(part.mesh, upright)
        BodySpecification(
            name=f"kitchen_{part.name}",
            shapes=ShapeCollection([]),
            visual_shapes=ShapeCollection(
                [Mesh(origin=HomogeneousTransformationMatrix(), filename=str(mesh))]
            ),
            connection_specification=FixedConnectionSpecification(),
        ).spawn(
            world,
            parent_T_self=HomogeneousTransformationMatrix(
                world_T_part, reference_frame=world.root
            ),
        )

    @staticmethod
    def loaded(mesh: Path) -> trimesh.Trimesh:
        """
        :param mesh: A mesh file.
        :return: Its geometry as one mesh.
        """
        return trimesh.load(mesh, force="mesh")

    def converted(self, mesh: Path, upright: np.ndarray) -> Path:
        """
        :param mesh: A mesh file the simulator may not read.
        :param upright: The rotation standing it upright.
        :return: The same mesh as an OBJ file, stood upright, converted once.
        """
        target = self.cache / f"{mesh.parent.parent.name}_{mesh.stem}.obj"
        if not target.is_file():
            self.cache.mkdir(parents=True, exist_ok=True)
            geometry = self.loaded(mesh)
            geometry.apply_transform(upright)
            geometry.export(target)
        return target
