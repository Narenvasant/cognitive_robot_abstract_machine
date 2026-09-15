"""
Runs Giskard's own control loop against the live, physically simulated world.

Every control cycle Giskard writes its command into the world state, the MuJoCo
synchronizer hands that to the joints' servos as their set point, and the physics steps
in between. A motion is therefore reached in the physics, as fast and as hard as the
servos allow, rather than planned kinematically first and played back afterwards.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from typing_extensions import Callable, Dict, List, Optional, Union

from coraplex.datastructures.enums import Arms
from coraplex.exceptions import MotionDidNotFinish
from semantic_digital_twin.adapters.real_time_simulation import RealTimeSimulation
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.tracy import (
    Tracy,
    TracyJoint,
    TracyLeftGripper,
    TracyRightGripper,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


def arm_of(robot: Tracy, arm_side: Arms) -> Arm:
    """
    :param robot: The robot to look the arm up on.
    :param arm_side: Which arm to return; must be :attr:`Arms.LEFT` or
        :attr:`Arms.RIGHT`.
    :return: ``robot``'s own left or right arm.
    """
    return robot.left_arm if arm_side == Arms.LEFT else robot.right_arm


# %% the gripper's own geometry


@dataclass(frozen=True)
class RobotiqGripper:
    """
    One of Tracy's Robotiq 2F-85 grippers, by the arm it hangs off, read directly off
    Tracy's own end-effector and joint description.
    """

    arm_side: Arms
    """
    Which arm the gripper hangs off.
    """

    def end_effector(self, robot: Tracy) -> Union[TracyLeftGripper, TracyRightGripper]:
        """
        :param robot: The robot the gripper belongs to.
        :return: The gripper, as Tracy's own description builds it.
        """
        return arm_of(robot, self.arm_side).end_effector

    def left_fingertip(self, robot: Tracy) -> Body:
        """
        :param robot: The robot the gripper belongs to.
        :return: The left fingertip pad's body.
        """
        return self.end_effector(robot).thumb.tip

    def right_fingertip(self, robot: Tracy) -> Body:
        """
        :param robot: The robot the gripper belongs to.
        :return: The right fingertip pad's body.
        """
        return self.end_effector(robot).finger.tip

    @property
    def knuckle_joint(self) -> TracyJoint:
        """
        Tracy's own name of the joint that actually drives the gripper; every other
        finger joint in the mimic linkage follows it.
        """
        return (
            TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE
            if self.arm_side == Arms.LEFT
            else TracyJoint.RIGHT_GRIPPER_LEFT_KNUCKLE
        )

    def knuckle_raw_dof(self, world: World) -> DegreeOfFreedom:
        """
        :param world: The world the gripper's own connections are described in.
        :return: The raw degree of freedom driving the knuckle.
        """
        return world.get_connection_by_name(self.knuckle_joint).raw_dof

    def closing_raw_angle_for_half_width(
        self,
        world: World,
        target_half_width: float,
        iterations: int = 30,
    ) -> float:
        """
        The knuckle's raw angle at which the fingertip pads' own inner faces first reach
        ``target_half_width`` out from the gripper's own centreline.

        The pad's inner position decreases monotonically as the knuckle closes, so
        bisection against an isolated scratch copy of ``world`` converges reliably.

        :param world: The live world to clone for the search; never itself modified.
        :param target_half_width: The half-width, in metres, to close to.
        :param iterations: Bisection steps; 30 narrows the joint's own ~0.8 rad range to
            well under a micro-radian.
        :return: The raw angle.
        """
        scratch_world = deepcopy(world)
        [scratch_robot] = scratch_world.get_semantic_annotations_by_type(Tracy)
        gripper_root = self.end_effector(scratch_robot).root
        left_fingertip = self.left_fingertip(scratch_robot)
        raw_dof = self.knuckle_raw_dof(scratch_world)

        def inner_x(raw_angle: float) -> float:
            """
            The left pad's innermost point along the closing axis at one raw angle,
            moving the scratch world's own state directly.
            """
            scratch_world.state[raw_dof.id].position = raw_angle
            scratch_world.notify_state_change()
            scratch_world.update_forward_kinematics()
            return (
                left_fingertip.collision.as_bounding_box_collection_in_frame(
                    gripper_root
                )
                .bounding_box()
                .min_x
            )

        lower, upper = raw_dof.limits.lower.position, raw_dof.limits.upper.position
        if target_half_width >= inner_x(lower):
            return lower
        if target_half_width <= inner_x(upper):
            return upper
        for _ in range(iterations):
            midpoint = (lower + upper) / 2
            if inner_x(midpoint) > target_half_width:
                lower = midpoint
            else:
                upper = midpoint
        return upper


# %% running motions live


@dataclass
class MotionRunner:
    """
    Runs one Giskard motion after another against the live, physically simulated world,
    ticking Giskard's own control loop in lockstep with the physics.
    """

    simulation: RealTimeSimulation
    """
    The running simulation of the world the motions are run in.
    """

    target_frequency: int = 50
    """
    Giskard's control rate, in cycles per simulated second; the physics advances one
    control period between cycles.
    """

    prediction_horizon: int = 4
    """
    Giskard's QP prediction horizon, matching the one
    :class:`~coraplex.plans.executables.GiskardExecutable` builds its own controller
    with.
    """

    max_ticks: int = 2000
    """
    Control cycles a single motion gets before giving up, matching
    :attr:`~coraplex.datastructures.dataclasses.Context.ticks_per_motion`'s own default.
    """

    convergence_threshold: float = 0.01
    """
    Largest per-joint distance, in radians, between a joint's simulated position and its
    set point for the motion to count as settled.
    """

    settle_timeout: float = 10.0
    """
    Simulated seconds to wait for the servos to settle once Giskard's own motion has
    ended.
    """

    squeeze_margin: float = 0.001
    """
    How far, in metres, past a grasped object's own half-width the fingers are commanded
    to close, so they press into it firmly rather than merely touching.

    Kept small, since it is commanded penetration into a rigid object.
    """

    grasp_settle_time: float = 0.5
    """
    Simulated seconds the fingers are held at their closing target before the caller
    moves the arm, so the servos build up their holding force against the object.
    """

    @property
    def world(self) -> World:
        """
        The live world the motions are run in.
        """
        return self.simulation.world

    @property
    def tick_period(self) -> float:
        """
        Simulated seconds one control cycle stands for.
        """
        return 1.0 / self.target_frequency

    def run(
        self,
        task: Task,
        avoid_collisions: bool,
        stop_when: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Tick ``task`` against the live world until Giskard's own motion ends, advancing
        the physics one control period per tick.

        :param task: The Giskard task to run.
        :param avoid_collisions: Whether Giskard's own ``ExternalCollisionAvoidance``
            runs alongside. Set False for a goal whose own point is to approach and
            touch an object, since avoidance would otherwise treat that object as an
            obstacle and make the goal unreachable.
        :param stop_when: Checked after every cycle; the motion stops early once it
            returns True.
        :raises MotionDidNotFinish: If the goal was not reached within
            :attr:`max_ticks`.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(task)
        if avoid_collisions:
            motion_statechart.add_node(ExternalCollisionAvoidance())
        end_motion = EndMotion()
        end_motion.start_condition = task.observation_variable
        motion_statechart.add_node(end_motion)

        executor = Executor(
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=self.target_frequency,
                    prediction_horizon=self.prediction_horizon,
                    verbose=False,
                ),
            )
        )
        executor.compile(motion_statechart)
        try:
            for _ in range(self.max_ticks):
                executor.tick()
                self.simulation.advance(self.tick_period)
                if motion_statechart.is_end_motion():
                    return
                if stop_when is not None and stop_when():
                    return
        finally:
            executor.set_velocity_acceleration_jerk_to_zero()
            motion_statechart.cleanup_nodes(context=executor.context)
            executor.context.cleanup()
        raise MotionDidNotFinish(failed_motions=[task])

    def settle(self, joint_names: List[str], timeout: Optional[float] = None) -> None:
        """
        Advance the physics until every one of ``joint_names`` has reached its set
        point, or the timeout passes.

        :param joint_names: The joints to wait for.
        :param timeout: Simulated seconds to wait; :attr:`settle_timeout` if not given.
        """
        if timeout is None:
            timeout = self.settle_timeout
        simulator = self.simulation.mirror.simulator
        set_points = {
            joint_name: self.world.state[
                self.world.get_connection_by_name(joint_name).raw_dof.id
            ].position
            for joint_name in joint_names
        }
        simulated_time = 0.0
        errors: Dict[str, float] = {}
        while simulated_time < timeout:
            self.simulation.advance(self.tick_period)
            simulated_time += self.tick_period
            errors = {
                joint_name: abs(
                    simulator.get_joint_value(joint_name).result - set_point
                )
                for joint_name, set_point in set_points.items()
            }
            if max(errors.values()) < self.convergence_threshold:
                return
        worst_joint = max(errors, key=errors.get)
        logger.warning(
            "Motion did not settle within %.0fs; worst joint %s is %.3f rad off.",
            timeout,
            worst_joint,
            errors[worst_joint],
        )

    def hold(self, duration: float) -> None:
        """
        Advance the physics with every set point held where it is.

        :param duration: Simulated seconds to hold.
        """
        self.simulation.advance(duration)

    def reach(
        self,
        robot: Tracy,
        arm_side: Arms,
        goal_pose: Pose,
        translation_only: bool,
        avoid_collisions: bool = True,
    ) -> None:
        """
        Move an arm's tool frame to a Cartesian goal and wait for the arm to settle.

        :param robot: The robot whose arm moves.
        :param arm_side: Which arm's tool frame should reach ``goal_pose``.
        :param goal_pose: Target pose for the arm's tool frame.
        :param translation_only: If True, only the tool frame's position is constrained;
            otherwise both position and orientation are.
        :param avoid_collisions: See :meth:`run`.
        """
        arm = arm_of(robot, arm_side)
        if translation_only:
            task = CartesianPosition(
                root_link=robot.root,
                tip_link=arm.end_effector.tool_frame,
                goal_point=goal_pose.to_position(),
            )
        else:
            task = CartesianPose(
                root_link=robot.root,
                tip_link=arm.end_effector.tool_frame,
                goal_pose=goal_pose,
            )
        self.run(task, avoid_collisions)
        self.settle(
            [
                connection.raw_dof.name.name
                for connection in arm.active_connections
                if isinstance(connection, ActiveConnection1DOF)
            ]
        )

    def move_joints(
        self,
        targets: Dict[str, float],
        avoid_collisions: bool = True,
        settle_timeout: Optional[float] = None,
        stop_when: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Move joints to target positions with Giskard's own
        :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointPositionList`, which
        keeps every joint within its own velocity limit, and wait for them to settle.

        :param targets: Target position by joint name.
        :param avoid_collisions: See :meth:`run`.
        :param settle_timeout: See :meth:`settle`.
        :param stop_when: See :meth:`run`.
        """
        joint_connections = [
            self.world.get_connection_by_name(name) for name in targets
        ]
        goal_state = JointState.from_mapping(
            dict(zip(joint_connections, targets.values()))
        )
        self.run(JointPositionList(goal_state=goal_state), avoid_collisions, stop_when)
        self.settle(list(targets), settle_timeout)

    def park_arms(
        self, robot: Tracy, arm_sides: List[Arms], avoid_collisions: bool = True
    ) -> None:
        """
        Move ``arm_sides`` to their park configuration.

        :param robot: The robot whose arms are parked.
        :param arm_sides: Which arms to park, e.g. ``[Arms.LEFT, Arms.RIGHT]``.
        :param avoid_collisions: See :meth:`run`.
        """
        targets: Dict[str, float] = {}
        for arm_side in arm_sides:
            park_state = arm_of(robot, arm_side).get_joint_state_by_type(
                StaticJointState.PARK
            )
            for connection, target in zip(
                park_state.connections, park_state.target_values
            ):
                targets[connection.raw_dof.name.name] = target
        self.move_joints(targets, avoid_collisions=avoid_collisions)

    def set_gripper(
        self,
        robot: Tracy,
        arm_side: Arms,
        state: GripperState,
        settle_timeout: float = 3.0,
    ) -> None:
        """
        Open or close an arm's gripper.

        Collision avoidance is always off: the goal is to close the fingers around
        whatever object is between them, which avoidance would treat as an obstacle.

        :param robot: The robot whose gripper is driven.
        :param arm_side: Which arm's gripper to drive.
        :param state: The gripper state to command, e.g. :attr:`GripperState.CLOSE`.
        :param settle_timeout: Simulated seconds to wait for the fingers to settle.
        """
        goal_state = arm_of(robot, arm_side).end_effector.get_joint_state_by_type(state)
        # every connection of the mimic linkage shares one raw degree of freedom, so
        # each connection's own target is converted back to that degree of freedom's
        raw_targets: Dict[str, float] = {}
        for connection, target in zip(goal_state.connections, goal_state.target_values):
            raw_targets[connection.raw_dof.name.name] = (
                target - connection.offset
            ) / connection.multiplier
        self.move_joints(
            raw_targets, avoid_collisions=False, settle_timeout=settle_timeout
        )

    def close_gripper_around(
        self,
        robot: Tracy,
        arm_side: Arms,
        target_body: Body,
        squeeze_margin: Optional[float] = None,
        half_width: Optional[float] = None,
    ) -> None:
        """
        Close an arm's gripper around ``target_body``, sized to the object's own width
        instead of driving to the fully closed position.

        Closing all the way on an object that is not perfectly centred between the
        fingers wedges it sideways rather than gripping it, so the fingers close to the
        object's half-width along the closing axis minus the squeeze margin, and are then
        held there so the squeeze is actually applied: a face contact holds on friction
        alone, but a point or edge contact slips straight back out without it.

        :param robot: The robot whose gripper is driven.
        :param arm_side: Which arm's gripper to drive.
        :param target_body: The body to close around.
        :param squeeze_margin: How far past the half-width the fingers close;
            :attr:`squeeze_margin` if not given.
        :param half_width: Half the width, in metres, the pads are to meet the object
            across, for an object grasped across a known pair of faces. Defaults to
            half the object's own bounding-box width along the closing axis, which
            over-reads the width of a box standing a little turned relative to the
            gripper.
        """
        if squeeze_margin is None:
            squeeze_margin = self.squeeze_margin
        gripper = RobotiqGripper(arm_side)
        if half_width is None:
            bounding_box = target_body.collision.as_bounding_box_collection_in_frame(
                gripper.end_effector(robot).root
            ).bounding_box()
            half_width = (bounding_box.max_x - bounding_box.min_x) / 2
        target_inner_x = max(0.0, half_width - squeeze_margin)
        raw_angle = gripper.closing_raw_angle_for_half_width(self.world, target_inner_x)
        raw_dof = gripper.knuckle_raw_dof(self.world)

        simulator = self.simulation.mirror.simulator
        target_name = target_body.name.name
        left_fingertip_name = gripper.left_fingertip(robot).name.name
        right_fingertip_name = gripper.right_fingertip(robot).name.name

        def both_fingertips_touching() -> bool:
            """
            Whether both pads are in contact with the target right now.
            """
            left_contacts = simulator.get_contact_bodies(
                body_name=left_fingertip_name, including_children=False
            ).result
            right_contacts = simulator.get_contact_bodies(
                body_name=right_fingertip_name, including_children=False
            ).result
            return target_name in left_contacts and target_name in right_contacts

        knuckle = self.world.get_connection_by_name(raw_dof.name.name)
        goal_state = JointState.from_mapping({knuckle: raw_angle})
        self.run(JointPositionList(goal_state=goal_state), avoid_collisions=False)
        # the fingers stop on the object short of their set point, so the servos are
        # given a fixed squeeze time rather than waited for to settle
        self.hold(self.grasp_settle_time)
        logger.info(
            "%s: gripper closed to its own half-width-sized target (%.4fm); "
            "both fingertips %s the object.",
            target_body.name,
            target_inner_x,
            "reached" if both_fingertips_touching() else "never reached",
        )
