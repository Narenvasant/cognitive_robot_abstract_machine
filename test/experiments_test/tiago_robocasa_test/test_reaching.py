"""
How Tiago's arm gets from its park pose to where it waits before taking an object in a
RoboCasa kitchen.

The run needs RoboCasa's kitchen assets, so it is skipped wherever RoboCasa is not
installed.
"""

import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

import numpy as np

from experiments.tiago_robocasa.pick_place import PickPlaceCounterToSink
from semantic_digital_twin.robots.tiago import TiagoMujocoBody

# %% the way to the object


def test_the_hand_stays_below_the_head_on_its_way_to_the_object():
    """
    The arm is sent from its park pose to the joints the reach was solved for, each
    joint moving straight to its own goal, so the hand passes over the head whenever the
    solve lands on the shoulder raised rather than lowered.
    """
    task = PickPlaceCounterToSink()
    task.build_world()
    task.park_the_arms()
    task.grasp_taken = task.grasp()
    task.drive_to_where_the_hand_arrives(task.reaches_the_pre_grasp)
    [reach] = task.arm_motions_to(task.pre_grasp())
    joints = [task.world.get_connection_by_name(name) for name in reach.names]
    parked = np.array([joint.position for joint in joints])
    head_underside = task.extent(
        task.world.get_body_by_name(TiagoMujocoBody.HEAD)
    ).min_z
    tool_frame = task.arm.end_effector.tool_frame

    highest = -np.inf
    for share in np.linspace(0.0, 1.0, 50):
        for joint, position in zip(
            joints, (1 - share) * parked + share * np.array(reach.positions)
        ):
            joint.position = position
        task.world.notify_state_change()
        highest = max(
            highest,
            task.world.compute_forward_kinematics_np(task.world.root, tool_frame)[2, 3],
        )

    assert highest < head_underside
