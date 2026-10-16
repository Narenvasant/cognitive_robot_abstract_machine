"""
The HSR performing RoboCasa's pick-and-place tasks in a RoboCasa kitchen lowered to
its height, simulated in MuJoCo with the object held by friction alone.
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from experiments.hsr_robocasa.pick_place import (
    HSRPickPlaceTask,
    PickPlaceCounterToSink,
    PickPlaceCounterToStove,
    PickPlaceDrawerToCounter,
    PickPlaceSinkToCounter,
    PickPlaceStoveToCounter,
    TaskObject,
)
from experiments.hsr_robocasa.robocasa_task import (
    FixtureType,
    HSRBArmPlane,
    LayoutKey,
    RoboCasaLayout,
    RowOffTheFloor,
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
How far, in metres, the object's underside may end up outside the height of the
receptacle's floor and still count as resting on it.
"""


@dataclass
class TaskOutcome:
    """
    A task performed in MuJoCo, with what its receptacle looked like before the run.
    """

    experiment: HSRPickPlaceTask
    """
    The task, after its run.
    """

    receptacle_outline: np.ndarray
    """
    The lower and upper corner of the box the receptacle's collision geometry
    occupied before the run, in the kitchen's frame.
    """

    release_height: float
    """
    The height the object was to be let go above.
    """


def outline_of(experiment: HSRPickPlaceTask) -> np.ndarray:
    """
    :return: The lower and upper corner of the box the collision geometry of the
        task's receptacle occupies, in the kitchen's frame.
    """
    boxes = [
        experiment.extent(body)
        for body in experiment.colliding_bodies_below(experiment.receptacle)
    ]
    return np.array(
        [
            np.min([[box.min_x, box.min_y, box.min_z] for box in boxes], axis=0),
            np.max([[box.max_x, box.max_y, box.max_z] for box in boxes], axis=0),
        ]
    )


@pytest.fixture(
    scope="module",
    params=[
        PickPlaceCounterToStove,
        PickPlaceStoveToCounter,
        PickPlaceCounterToSink,
        PickPlaceSinkToCounter,
        PickPlaceDrawerToCounter,
    ],
)
def outcome(request) -> TaskOutcome:
    experiment = request.param(headless=True)
    experiment.build_world()
    receptacle_outline = outline_of(experiment)
    release_height = experiment.release_height
    experiment.run()
    return TaskOutcome(experiment, receptacle_outline, release_height)


# %% the lowered kitchen


def test_the_counters_stand_at_the_height_the_kitchen_is_lowered_to():
    experiment = PickPlaceCounterToStove()
    layout = RoboCasaLayout.from_layout_type(experiment.layout)

    lowered = layout.lowered_fixtures(experiment.counter_height)

    counters = [
        fixture[LayoutKey.NAME]
        for group in layout.configuration.values()
        for row in group.values()
        if isinstance(row, list)
        for fixture in row
        if fixture.get(LayoutKey.TYPE) == FixtureType.COUNTER
    ]
    assert [lowered[counter][LayoutKey.SIZE][2] for counter in counters] == (
        pytest.approx([experiment.counter_height] * len(counters))
    )


def test_what_hangs_on_the_walls_keeps_its_height():
    experiment = PickPlaceCounterToStove()
    layout = RoboCasaLayout.from_layout_type(experiment.layout)

    lowered = layout.lowered_fixtures(experiment.counter_height)

    on_the_walls = [
        fixture[LayoutKey.NAME]
        for group in layout.configuration.values()
        for row_name, row in group.items()
        if row_name in RowOffTheFloor and isinstance(row, list)
        for fixture in row
    ]
    assert [name for name in on_the_walls if name in lowered] == []


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


def test_the_arm_plane_turns_the_wrist_to_the_goals_roll_where_it_puts_the_tool_frame():
    arm_plane = HSRBArmPlane()
    tool_frame = arm_plane.robot.all_arms[0].end_effector.tool_frame
    for joint, position in {
        HSRBJoint.ARM_LIFT: 0.4,
        HSRBJoint.ARM_FLEX: -0.8,
        HSRBJoint.WRIST_FLEX: -1.2,
        HSRBJoint.WRIST_ROLL: 0.7,
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

    assert reached == pytest.approx(base_T_tool, abs=REACHED_POSE_TOLERANCE)


# %% the run


@runs_in_ci
def test_the_object_ends_resting_in_the_receptacle(outcome: TaskOutcome):
    """
    Held by friction alone, the object travels to the receptacle and comes to rest in
    it, standing or fallen over: within its outline, and no higher than it was let
    go.
    """
    experiment = outcome.experiment
    food = experiment.extent(experiment.world.get_body_by_name(TaskObject.FOOD))
    x, y, _ = food.center.to_np()[:3]
    lower, upper = outcome.receptacle_outline

    assert lower[0] < x < upper[0]
    assert lower[1] < y < upper[1]
    assert (
        lower[2] - RESTING_TOLERANCE
        <= food.min_z
        <= outcome.release_height + RESTING_TOLERANCE
    )
