"""
Tests that parking brings the arms home around the robot's own body rather than through
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tiago import TiagoMujoco, TiagoMujocoBody
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    not TiagoMujoco.is_description_installed(),
    reason="The MuJoCo model of the Tiago is not installed",
)


# %% watching the arm


@dataclass
class ClosestApproach(StateChangeCallback):
    """
    The closest the bodies of one part of a robot come to a body of another part, over
    every state the world passes through.
    """

    moving: list[Body] = field(kw_only=True)
    """
    The bodies that move towards the others.
    """

    kept_clear_of: list[Body] = field(kw_only=True)
    """
    The bodies the moving ones are to stay clear of.
    """

    distance: float = field(init=False, default=float("inf"))
    """
    The closest distance seen so far, in metres; negative when the bodies overlapped.
    """

    def on_state_change(self, **kwargs) -> None:
        detector = self._world.collision_manager.collision_detector
        for body in self.moving:
            for other in self.kept_clear_of:
                closest = detector.check_collision_between_bodies(
                    body, other, distance=0.1
                )
                if closest is not None:
                    self.distance = min(self.distance, closest.distance)


# %% the robot


@pytest.fixture
def tiago_with_the_arm_across_its_head() -> TiagoMujoco:
    """
    The MuJoCo Tiago with its right arm stretched out in front of it, from where moving
    each joint straight to its parked position sweeps the wrist through the head.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    robot = RobotSpecification(TiagoMujoco).spawn(world)
    arm = robot.mobile_base.torso.right_arm
    for connection, position in zip(
        arm.active_connections, [0.43, -1.07, 0.1, 1.17, 1.37, 0.09, 1.31]
    ):
        connection.position = position
    world.notify_state_change()
    return robot


def _park_the_arms(robot: TiagoMujoco) -> None:
    """
    Park both arms, avoiding collisions, in a kinematic simulation.
    """
    plan = execute_single(ParkArmsAction(robot.all_arms), Context(robot._world, robot))
    with kinematically_simulated_robot(collision_avoidance=True):
        plan.perform()


# %% parking


def test_parking_keeps_the_arm_off_the_head(tiago_with_the_arm_across_its_head):
    robot = tiago_with_the_arm_across_its_head
    world = robot._world
    arm = robot.mobile_base.torso.right_arm
    approach = ClosestApproach(
        _world=world,
        moving=[
            body
            for body in arm.bodies
            if body.has_collision() and body is not robot.mobile_base.torso.root
        ],
        kept_clear_of=[
            body
            for body in robot.bodies_with_collision
            if body.name.name == TiagoMujocoBody.HEAD
        ],
    )

    _park_the_arms(robot)
    approach.stop()

    assert approach.distance > 0.0


def test_parking_around_the_head_still_parks_the_arm(
    tiago_with_the_arm_across_its_head,
):
    robot = tiago_with_the_arm_across_its_head
    arm = robot.mobile_base.torso.right_arm
    parked = arm.get_joint_state_by_type(StaticJointState.PARK)

    _park_the_arms(robot)

    reached = [connection.position for connection in parked.connections]
    assert reached == pytest.approx(list(parked.target_values), abs=0.05)
