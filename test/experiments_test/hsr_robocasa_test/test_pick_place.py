"""
The HSR performing RoboCasa's ``PickPlaceCounterToStove`` task in a RoboCasa kitchen
lowered to its height, simulated in MuJoCo with the food held by friction alone.
"""

import os

import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from experiments.hsr_robocasa.pick_place_counter_to_stove import (
    FixtureType,
    HSRBArmPlane,
    LayoutKey,
    PickPlaceCounterToStove,
    RoboCasaLayout,
    RowOffTheFloor,
    TaskObject,
)
from semantic_digital_twin.robots.hsrb import HSRBJoint

runs_in_ci = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "false",
    reason="MuJoCo tests only run in CI",
)

REACHED_POSE_TOLERANCE = 0.005
"""
How far, in metres, the tool frame may end up from a pose the arm plane is asked to
reach: the resolution of its searches.
"""

RESTING_TOLERANCE = 0.02
"""
How far, in metres, the food's underside may end up above the pan's floor and still
count as lying on it.
"""


@pytest.fixture(scope="module")
def experiment_after_the_run() -> PickPlaceCounterToStove:
    experiment = PickPlaceCounterToStove(headless=True)
    experiment.build_world()
    experiment.run()
    return experiment


# %% the lowered kitchen


def test_the_counters_stand_at_the_height_the_kitchen_is_lowered_to():
    experiment = PickPlaceCounterToStove()
    layout = RoboCasaLayout.from_layout_type(experiment.layout)

    lowered = layout.lowered_to(experiment.counter_height)

    counter_heights = [
        fixture[LayoutKey.SIZE][2]
        for group in lowered.values()
        for row in group.values()
        if isinstance(row, list)
        for fixture in row
        if isinstance(fixture, dict)
        and fixture.get(LayoutKey.TYPE) == FixtureType.COUNTER
    ]
    assert counter_heights == pytest.approx(
        [experiment.counter_height] * len(counter_heights)
    )


def test_what_hangs_on_the_walls_keeps_its_height():
    experiment = PickPlaceCounterToStove()
    layout = RoboCasaLayout.from_layout_type(experiment.layout)

    lowered = layout.lowered_to(experiment.counter_height)

    for group_name, group in layout.configuration.items():
        for row_name, row in group.items():
            if row_name in RowOffTheFloor:
                assert lowered[group_name][row_name] == row


# %% the arm plane


def test_the_arm_plane_puts_the_tool_frame_where_a_configuration_of_it_does():
    arm_plane = HSRBArmPlane()
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    for joint, position in {
        HSRBJoint.ARM_LIFT: 0.4,
        HSRBJoint.ARM_FLEX: -0.8,
        HSRBJoint.WRIST_FLEX: -1.2,
    }.items():
        arm_plane.world.get_connection_by_name(joint).position = position
    base_T_tool = arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )

    for joint, position in arm_plane.joint_positions_for(base_T_tool).items():
        arm_plane.world.get_connection_by_name(joint).position = position
    reached = arm_plane.world.compute_forward_kinematics_np(
        arm_plane.robot.root, tool_frame
    )

    assert reached[:3, 3] == pytest.approx(
        base_T_tool[:3, 3], abs=REACHED_POSE_TOLERANCE
    )


# %% the run


@runs_in_ci
def test_the_food_ends_lying_in_the_pan(
    experiment_after_the_run: PickPlaceCounterToStove,
):
    """
    Held by friction alone, the food travels from the plate to the pan and comes to
    rest on the pan's floor.
    """
    experiment = experiment_after_the_run
    floor = experiment.pan_floor
    food = experiment.extent(experiment.world.get_body_by_name(TaskObject.FOOD))
    x, y, _ = food.center.to_np()[:3]

    assert floor.min_x < x < floor.max_x
    assert floor.min_y < y < floor.max_y
    assert food.min_z == pytest.approx(floor.max_z, abs=RESTING_TOLERANCE)
