import math

import pytest

from experiments.tiago_robocasa.pick_place import TiagoPickPlaceTask
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the shape a piece of fruit collides as


def test_the_hull_of_a_body_reaches_as_far_as_its_pieces_do():
    """
    A piece turned half a right angle reaches past its own half width along
    both horizontal axes, and a piece set above it reaches higher.
    """
    world = World()
    body = Body(name=PrefixedName("fruit"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
    turned_edge, upright_edge, offset, raised = 0.04, 0.02, 0.1, 0.05
    body.collision = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=offset, yaw=math.pi / 4, reference_frame=body
                ),
                scale=Scale(turned_edge, turned_edge, turned_edge),
            ),
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=raised, reference_frame=body
                ),
                scale=Scale(upright_edge, upright_edge, upright_edge),
            ),
        ],
        reference_frame=body,
    )
    turned_reach = turned_edge / math.sqrt(2)

    hull = TiagoPickPlaceTask.convex_hull_of(body).mesh

    assert hull.is_convex
    assert hull.bounds.tolist() == [
        pytest.approx([-upright_edge / 2, -turned_reach, -turned_edge / 2]),
        pytest.approx([offset + turned_reach, turned_reach, raised + upright_edge / 2]),
    ]
