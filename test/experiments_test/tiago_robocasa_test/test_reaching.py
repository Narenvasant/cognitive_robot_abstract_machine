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


def test_the_hand_never_rises_over_the_head_on_its_way_to_the_object():
    """
    The reach sends each joint straight to its goal, so the hand passes over the head
    whenever the solve leaves the arm turned over from the way the park pose holds it.
    """
    task = PickPlaceCounterToSink()
    task.build_world()
    task.park_the_arms()
    task.grasp_taken = task.grasp()
    task.drive_to_where_the_hand_arrives(task.reaches_the_pre_grasp)
    reach = task.arm_motions_to(task.pre_grasp())
    head_top = task.extent(task.world.get_body_by_name(TiagoMujocoBody.HEAD)).max_z
    tool_frame = task.arm.end_effector.tool_frame

    highest = -np.inf
    for motion in reach:
        joints = [task.world.get_connection_by_name(name) for name in motion.names]
        start = np.array([joint.position for joint in joints])
        for share in np.linspace(0.0, 1.0, 50):
            for joint, position in zip(
                joints, (1 - share) * start + share * np.array(motion.positions)
            ):
                joint.position = position
            task.world.notify_state_change()
            highest = max(
                highest,
                task.world.compute_forward_kinematics_np(task.world.root, tool_frame)[
                    2, 3
                ],
            )

    assert highest < head_top
