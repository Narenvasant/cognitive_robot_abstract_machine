"""
Tests for the scene the Tiago milk pick-and-place demo builds and the grasp it chooses,
both of which are settled before any physics is involved.
"""

from __future__ import annotations

import pytest

from experiments.tiago_milk_pick_place.exceptions import NotOverTheTable, OutOfReach
from experiments.tiago_milk_pick_place.pick_place import MilkPickPlaceDemonstration
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world import World


@pytest.fixture(scope="module")
def populated_scene() -> tuple[MilkPickPlaceDemonstration, World]:
    demonstration = MilkPickPlaceDemonstration()
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return demonstration, world


# %% the scene


def test_the_carton_stands_on_the_table_where_the_layout_puts_it(populated_scene):
    """
    The carton is spawned on the table's measured top, not at a height written into the
    layout, so the demo follows the apartment it is pointed at.
    """
    demonstration, world = populated_scene
    carton = world.get_body_by_name(demonstration.carton.name)
    box = carton.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()

    assert float(box.min_z) == pytest.approx(
        demonstration.table_top + demonstration.layout.clearance_above_table, abs=1e-6
    )


def test_the_carton_weighs_what_a_carton_of_its_size_holds(populated_scene):
    """
    The mesh declares no mass, and what the carton weighs is what the fingers have to
    hold by friction.
    """
    demonstration, world = populated_scene
    carton = world.get_body_by_name(demonstration.carton.name)
    litres = float(carton.collision.shapes[0].mesh.volume) * 1000.0

    assert carton.inertial.mass == pytest.approx(
        litres * demonstration.layout.milk_density
    )


def test_a_spot_beside_the_table_is_refused():
    """
    Both spots have to lie over the table, because the carton is put down on it.
    """
    demonstration = MilkPickPlaceDemonstration()
    demonstration.layout.place_y = demonstration.layout.place_y - 10.0

    with pytest.raises(NotOverTheTable):
        demonstration.populate_scene(demonstration.build_simulated_world())


def test_a_spot_the_arm_cannot_reach_is_refused():
    """
    The robot drives between the two spots, so each has to lie within the arm's reach of
    the pose it is reached from.
    """
    demonstration = MilkPickPlaceDemonstration()
    demonstration.layout.maximum_reach = 0.1

    with pytest.raises(OutOfReach):
        demonstration.populate_scene(demonstration.build_simulated_world())


# %% the step along the table


def test_the_carton_is_set_down_one_step_to_the_robot_s_left(populated_scene):
    """
    The robot faces the table along the world's x-axis, which puts its left along the
    world's y-axis, and the step is taken along the table rather than across it.
    """
    layout = populated_scene[0].layout

    assert layout.place_y > layout.pick_y
    assert layout.place_x == layout.pick_x


def test_the_arm_reaches_both_spots_from_where_the_robot_stands(populated_scene):
    """
    The base does not move while the carton is held, so one standing spot has to serve
    the pick-up and the place.
    """
    layout = populated_scene[0].layout

    for x, y in ((layout.pick_x, layout.pick_y), (layout.place_x, layout.place_y)):
        assert layout.distance_from_the_robot(x, y) <= layout.maximum_reach


# %% the grasp


def test_the_grasp_closes_across_the_carton_s_flat_faces(populated_scene):
    """
    The grasps a carton offers are evenly spaced around it, so some close across a pair
    of faces and some corner to corner.

    A corner-to-corner grip is almost the carton's diagonal, and the carton turns out of
    it as soon as it is lifted.
    """
    demonstration, world = populated_scene
    robot = world.get_semantic_annotations_by_type(demonstration.used_robot)[0]
    candidates = demonstration.carton.annotation.grasp_candidates()
    narrowest = min(candidate.width_along_closing_axis for candidate in candidates)

    chosen = demonstration.choose_grasp(demonstration.choose_arm(robot))

    assert chosen.width_along_closing_axis <= (
        narrowest + demonstration.layout.face_width_tolerance
    )


def test_the_grasp_is_approached_from_the_robot_s_side_of_the_carton(populated_scene):
    """
    The hand travels along the grasp frame's x-axis, so a grasp whose axis points back
    at the robot would be reached by driving the hand through the carton.
    """
    demonstration, world = populated_scene
    robot = world.get_semantic_annotations_by_type(demonstration.used_robot)[0]

    chosen = demonstration.choose_grasp(demonstration.choose_arm(robot))

    assert demonstration._alignment_with_the_approach(chosen) > 0.0


def test_the_open_hand_clears_the_carton_it_closes_on(populated_scene):
    """
    The hand reaches around the carton with its fingers open, so the room between the
    pads has to exceed the width it closes on -- otherwise the approach pushes the
    carton over instead of surrounding it.
    """
    demonstration, world = populated_scene
    robot = world.get_semantic_annotations_by_type(demonstration.used_robot)[0]
    arm = demonstration.choose_arm(robot)
    chosen = demonstration.choose_grasp(arm)
    gripper = arm.end_effector
    gripper.get_joint_state_by_type(GripperState.OPEN).apply_to(world)

    assert gripper.fingertips.separation > chosen.width_along_closing_axis
