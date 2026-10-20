"""
What Tiago's RoboCasa tasks share where it moves a fixture by its handle: a drawer along
its rails, a door about its hinge.

The handle is held by the grip of the fingers alone and the fixture moves because the
hand draws it along. Nothing commands the fixture's own joint: a joint written into
carries the fixture with it whatever the hand is doing, which is the thing these
experiments are to show happening by contact.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from experiments.tiago_robocasa.robocasa_task import TiagoRoboCasaTask
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% a task that moves a fixture by its handle


@dataclass
class TiagoHandleTask(TiagoRoboCasaTask, ABC):
    """
    One of RoboCasa's tasks in which Tiago takes hold of a fixture's handle and draws
    the fixture along by it.
    """

    handle_reach: float = 0.9
    """
    How far in front of its base's centre Tiago takes hold of the handle, in metres.

    Measured on a RoboCasa drawer, whose handle stands 0.79 m high: the hand gets onto
    it from 0.90 m to 0.95 m in front of the base and from nowhere nearer, because the
    arm hangs from a shoulder at the height of the counter and cannot fold back under
    itself. Nothing is in its way there -- the same holds with collision avoidance
    switched off.
    """

    base_follows_the_arm: bool = True
    """
    The base is driven along with the arm: the stretch of space the arm alone can hold
    the hand in is centimetres deep at a handle's height, while the fixture's travel is
    half a metre, so the hand can only stay on the handle if the base comes too.
    """

    tool_frame_tolerance: float = 0.03
    """
    How close the tool frame has to come to a goal, in metres: a simulated arm trails
    its commands while it moves, and a hand that is held to the millimetre never
    arrives.
    """

    finger_clearance: float = 0.015
    """
    How much room the open fingers are given to pass the handle, in metres, on top of
    how far they reach past the grasp.

    The hand waits that far in front of the handle and backs off that far again.
    """

    stall_minimum_time: float = 0.3
    """
    How long the fingers have to be stalled before the handle counts as held, in
    seconds.

    Closing on a bar stalls the fingers rather than shutting them.
    """

    closing_velocity: float = 0.05
    """
    How fast the fingers close on the handle, in metres per second.
    """

    pull_velocity: float = 0.05
    """
    How fast the hand draws the fixture along, in metres per second.

    Slower than the hand carries an object: the fingers only hold the handle by friction, and the
    fixture's own drag works against them the whole way.
    """

    pull_overshoot: float = 0.06
    """
    How far past where the fixture stops the hand is drawn, in metres: the fingers lose
    a little of the handle to slip the whole way along, and without it the fixture ends
    up short of where the task wants it.
    """

    upright_handle_roll: float = 3 * math.pi / 4
    """
    How far the hand is rolled about its approach to take an upright handle, in radians,
    which closes the fingers across the bar at an angle.

    Measured in front of a RoboCasa microwave: with the hand unrolled, or rolled a
    quarter turn either way, the arm reaches no height between 1.1 m and 1.7 m at all;
    rolled three eighths of a turn it reaches up to 1.5 m level and 1.6 m tilted up.
    """

    door_waypoint_angle: float = math.radians(10)
    """
    How far a door turns between the waypoints the hand is led through, in radians:

    the handle travels on an arc about the hinge, and a hand led straight to where the
    door ends up would pull across the arc rather than along it.
    """

    impedance_ratio: float = 50.0
    """
    Stiffer friction than carrying an object needs: the fingers have to hold a bar
    against a drawer's drag rather than against an object's weight.
    """

    noslip_iterations: int = 30
    """
    More passes than carrying an object needs, for the same reason as
    :attr:`impedance_ratio`.
    """

    open_ratio: float = 0.95
    """
    The share of its travel from which on RoboCasa counts the fixture open.
    """

    closed_ratio: float = 0.05
    """
    The share of its travel up to which RoboCasa counts the fixture closed.
    """

    @property
    @abstractmethod
    def handle(self) -> Handle:
        """
        :return: The handle the fixture is moved by.
        """

    @property
    @abstractmethod
    def fixture(self) -> Body:
        """
        :return: The fixture the task works, which the moving part sits in.
        """

    @property
    @abstractmethod
    def goal_position(self) -> float:
        """
        :return: Where the mechanism's joint is to end up.
        """

    @property
    def pull_goal(self) -> float:
        """
        :return: Where the hand draws the mechanism's joint to, which is
            :attr:`pull_overshoot` past where the fixture stops.
        """
        goal = self.goal_position
        return goal + math.copysign(self.pull_overshoot, goal - self.mechanism.position)

    @property
    def pull_waypoints(self) -> list[float]:
        """
        :return: The positions of the mechanism's joint the hand is led through on its
            way to :attr:`pull_goal`: the goal alone for a drawer, whose handle travels
            in a straight line, and one every :attr:`door_waypoint_angle` for a door.
        """
        start, goal = self.mechanism.position, self.pull_goal
        if not isinstance(self.mechanism, RevoluteConnection):
            return [goal]
        count = max(1, math.ceil(abs(goal - start) / self.door_waypoint_angle))
        return [start + (goal - start) * step / count for step in range(1, count + 1)]

    # %% the fixture

    @property
    def mechanism(self) -> ActiveConnection1DOF:
        """
        :return: The joint the fixture moves on.
        """
        return self.handle.root.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

    @property
    def moving_part(self) -> Body:
        """
        :return: The part of the fixture that moves on its joint, carrying the handle.
        """
        return self.mechanism.child

    @property
    def travel(self) -> float:
        """
        :return: How far along its joint the moving part stands, as a share of the
            travel its limits leave it.
        """
        limits = self.mechanism.dof.limits
        return float(
            (self.mechanism.position - limits.lower.position)
            / (limits.upper.position - limits.lower.position)
        )

    @property
    def handle_box(self) -> tuple[np.ndarray, np.ndarray]:
        """
        :return: The lowest and highest corner of the handle, in the kitchen's frame.
        """
        return self.box_around([self.handle.root])

    @property
    def handle_middle(self) -> np.ndarray:
        """
        :return: The middle of :attr:`handle_box`.
        """
        lowest, highest = self.handle_box
        return (lowest + highest) / 2

    @property
    def handle_is_upright(self) -> bool:
        """
        :return: Whether the handle stands taller than it is wide.
        """
        lowest, highest = self.handle_box
        size = highest - lowest
        return bool(size[2] > max(size[0], size[1]))

    @property
    def handle_grasp_roll(self) -> float:
        """
        :return: How far the hand is rolled about its approach to take the handle, in
            radians: :attr:`upright_handle_roll` on an upright handle, and a quarter
            turn on a level one, which the fingers close across from above and below.
        """
        return self.upright_handle_roll if self.handle_is_upright else math.pi / 2

    @property
    def first_target(self) -> np.ndarray:
        return self.handle_middle

    @property
    def handled_bodies(self) -> list[Body]:
        """
        :return: The whole fixture: not only the part that moves and the handle it
            carries, but the housing around them, which Tiago stands against while it
            works the mechanism.
        """
        return self.colliding_bodies_below(self.fixture)

    @property
    def headings_to_try(self) -> tuple[float, ...]:
        """
        :return: The one way that faces the fixture, read off which way its handle
            stands out from the part that moves.
        """
        outward = self.handle_middle - self.extent(self.moving_part).center.to_np()[:3]
        return (float(np.arctan2(-outward[1], -outward[0])),)

    @property
    def outcome(self) -> str:
        """
        :return: How far along its joint the fixture now stands.
        """
        return f"{self.moving_part.name.name} stands at {self.travel:.0%} of its travel"

    # %% where things stand once the fixture has moved

    def shift_to(self, position: float) -> np.ndarray:
        """
        :param position: Where the mechanism's joint is to be.
        :return: The transform, in the kitchen's frame, that takes what moves with the
            fixture from where it stands now to where it stands then.
        """
        joint = self.mechanism
        travel = position - joint.position
        movement = np.eye(4)
        match joint:
            case RevoluteConnection():
                movement[:3, :3] = Rotation.from_rotvec(
                    joint.axis.to_np()[:3] * travel
                ).as_matrix()
            case PrismaticConnection():
                movement[:3, 3] = joint.axis.to_np()[:3] * travel
        world_T_connection = joint.child.global_pose.to_np() @ np.linalg.inv(
            joint.connection_T_child_expression.to_np()
        )
        return world_T_connection @ movement @ np.linalg.inv(world_T_connection)

    def shifted(self, pose: Pose, position: float) -> Pose:
        """
        :param pose: A pose that moves with the fixture.
        :param position: Where the mechanism's joint is to be.
        :return: Where that pose stands then.
        """
        return HomogeneousTransformationMatrix(
            self.shift_to(position) @ pose.to_np(), reference_frame=self.world.root
        ).pose

    # %% taking hold of the handle

    def handle_grasp(self) -> GraspCandidate:
        """
        The grasp that takes the handle at its middle, approaching it from where Tiago
        stands, the fingers closing across the bar from above and below.

        It is worked out where the fixture stands closed and carried back to where the
        fixture stands now, so that a handle on an open door is taken at the angle the
        door has turned to.

        :return: The grasp.
        """
        handle = self.handle
        closed = self.mechanism.dof.limits.lower.position
        closed_middle = (self.shift_to(closed) @ np.append(self.handle_middle, 1.0))[:3]
        world_T_closed_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            *closed_middle,
            roll=self.handle_grasp_roll,
            pitch=self.handle_grasp_pitch,
            yaw=self.heading,
            reference_frame=self.world.root,
        ).pose
        return GraspCandidate(
            handle,
            self.world.transform(
                self.shifted(world_T_closed_grasp, self.mechanism.position),
                handle.root,
            ),
        )

    @property
    def handle_grasp_pitch(self) -> float:
        """
        :return: How far the hand is tilted from level as it takes the handle, in
            radians, negative upwards: not at all.
        """
        return 0.0

    def reaches_the_handle(self) -> bool:
        """
        :return: Whether the arm gets the hand onto the handle from where Tiago stands,
            worked out afresh because the grasp follows Tiago as Tiago is moved about.
        """
        return self.can_reach(self.handle_grasp().world_T_grasp)

    def taking_hold_of(self, grasp: GraspCandidate) -> GraspingAction:
        """
        :param grasp: The grasp the handle is taken by.
        :return: Reaching onto the handle and closing the fingers on it, which stall on
            the bar rather than shutting.
        """
        return GraspingAction(
            grasp,
            self.arm,
            approach_clearance=self.approach_clearance,
            grasp_closing_velocity=self.closing_velocity,
            grasp_stall_minimum_time=self.stall_minimum_time,
            tolerate_grasp_stall=True,
            position_threshold=self.tool_frame_tolerance,
        )

    @property
    def approach_clearance(self) -> float:
        """
        :return: How far in front of the handle the hand waits, in metres: far enough
            for the open fingers to pass the handle, and no further, because the arm's
            envelope at a handle's height is only centimetres deep.
        """
        return float(
            self.arm.end_effector.fingers_reach_beyond_the_grasp + self.finger_clearance
        )

    def pre_grasp(self) -> Pose:
        """
        :return: Where the hand waits before it moves onto the handle, read afresh so it
            follows Tiago as Tiago is moved about.
        """
        grasp = self.handle_grasp()
        return (
            self.taking_hold_of(grasp)
            .grasp_pose_sequence(grasp.world_T_grasp, self.arm.end_effector, grasp)
            .pre_grasp
        )

    # %% the plan

    def build_plan(self, context: Context) -> Plan:
        """
        Stand where the hand closes on the handle, bring the hand in front of it, take
        hold of it, draw the fixture to where the task wants it and let go.

        The stand has to leave the handle itself within reach of the arm, not merely the
        pose the hand waits at in front of it: those lie a hand's breadth apart and the
        arm's envelope at a handle's height is narrower than that, so a stand chosen for
        the waiting pose brings the arm there and stalls. Covering the rest, and the
        fixture's own travel, is what the base is along for.

        :param context: The context to plan in.
        :return: The plan.
        """
        self.stand_where_the_hand_arrives(
            self.reaches_the_handle, self.handle_middle, self.handle_reach
        )
        grasp = self.handle_grasp()
        pre_grasp = self.pre_grasp()
        goal = self.pull_goal
        return sequential(
            [
                ParkArmsAction(self.robot.all_arms),
                MoveToolCenterPointMotion(
                    pre_grasp,
                    self.arm,
                    allow_gripper_collision=True,
                    position_threshold=self.tool_frame_tolerance,
                ),
                self.taking_hold_of(grasp),
                *[
                    MoveToolCenterPointMotion(
                        self.shifted(grasp.world_T_grasp, waypoint),
                        self.arm,
                        allow_gripper_collision=True,
                        max_linear_velocity=self.pull_velocity,
                        position_threshold=self.tool_frame_tolerance,
                    )
                    for waypoint in self.pull_waypoints
                ],
                MoveGripperMotion(
                    GripperState.OPEN,
                    self.arm.end_effector,
                    allow_gripper_collision=True,
                ),
                MoveToolCenterPointMotion(
                    self.shifted(pre_grasp, goal),
                    self.arm,
                    allow_gripper_collision=True,
                    position_threshold=self.tool_frame_tolerance,
                ),
                ParkArmsAction(self.robot.all_arms),
            ],
            context=context,
        ).plan
