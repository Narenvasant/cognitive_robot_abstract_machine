"""
How Tiago's arm gets from its park pose to where it waits before taking an object in a
RoboCasa kitchen.

The run needs RoboCasa's kitchen assets, so it is skipped wherever RoboCasa is not
installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

import numpy as np
from typing_extensions import Iterator

from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from experiments.tiago_robocasa.composite import CupcakeCleanup
from experiments.tiago_robocasa.pick_place import PickPlaceCounterToSink
from experiments.tiago_robocasa.robocasa_task import TiagoRoboCasaTask
from robocasa.models.scenes.scene_registry import LayoutType
from semantic_digital_twin.robots.tiago import TiagoMujocoBody

# %% walking the reach


def reach_to_the_pre_grasp(task: TiagoRoboCasaTask) -> list[MoveJointsMotion]:
    """
    :param task: A task whose kitchen is composed.
    :return: The reach from the park pose to where the hand waits before taking the
        object, Tiago standing where it reaches from.
    """
    task.park_the_arms()
    task.grasp_taken = task.grasp()
    task.drive_to_where_the_hand_arrives(task.reaches_the_pre_grasp)
    return task.arm_motions_to(task.pre_grasp())


def states_along(
    task: TiagoRoboCasaTask, reach: list[MoveJointsMotion]
) -> Iterator[None]:
    """
    Set the world to each of 50 states on the way of every motion, which moves its
    joints straight to their goals from where they are.

    :param task: The task whose world is moved.
    :param reach: The motions, in order.
    """
    for motion in reach:
        joints = [task.world.get_connection_by_name(name) for name in motion.names]
        start = np.array([joint.position for joint in joints])
        for share in np.linspace(0.0, 1.0, 50):
            for joint, position in zip(
                joints, (1 - share) * start + share * np.array(motion.positions)
            ):
                joint.position = position
            task.world.notify_state_change()
            yield


# %% the way to the object


def test_the_hand_never_rises_over_the_head_on_its_way_to_the_object():
    """
    The reach sends each joint straight to its goal, so the hand passes over the head
    whenever the solve leaves the arm turned over from the way the park pose holds it.
    """
    task = PickPlaceCounterToSink()
    task.build_world()
    reach = reach_to_the_pre_grasp(task)
    head_top = task.extent(task.world.get_body_by_name(TiagoMujocoBody.HEAD)).max_z
    tool_frame = task.arm.end_effector.tool_frame

    highest = max(
        task.world.compute_forward_kinematics_np(task.world.root, tool_frame)[2, 3]
        for _ in states_along(task, reach)
    )

    assert highest < head_top


def test_the_arm_never_passes_through_the_kitchen_on_its_way_to_the_object():
    """
    Lowering the upper arm keeps the hand below the head, but the forearm then swings
    down past the counter's edge into the cabinet below it.
    """
    task = CupcakeCleanup()
    task.layout = LayoutType.LAYOUT007
    task.seed = 51
    task.build_world()
    [step, _] = task.steps
    reach = reach_to_the_pre_grasp(step)
    robot = set(step.robot.bodies_with_collision)

    deepest = min(
        min(
            (
                contact.distance
                for contact in task.world.collision_manager.compute_collisions().contacts
                if (contact.body_a in robot) != (contact.body_b in robot)
            ),
            default=np.inf,
        )
        for _ in states_along(step, reach)
    )

    assert deepest > 0.0
