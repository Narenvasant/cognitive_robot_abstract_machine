"""
A robot performs RoboCasa's counter to sink pick and place task with an underspecified
coraplex plan.

RoboCasa composes the kitchen, samples the object to move and says where its own robot
starts. The robot is spawned there and given a pick up whose standing pose and grasp are
left open; coraplex grounds them by trying candidates against a copy of the world. It
then stands in front of the sink and lowers the object into the basin along a path a
graph of convex sets plans through the free space around the sink, and lets go of it
just above the basin floor. Whether the task succeeded is decided by RoboCasa's own
check.

The episode is performed either kinematically or in MuJoCo physics, where the robot's
servos and drive move it and its fingers hold the object by friction alone.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.robocasa_task_execution.counter_to_sink
    python -m experiments.robocasa_task_execution.counter_to_sink --robot tiago
    python -m experiments.robocasa_task_execution.counter_to_sink --layout LAYOUT005
    python -m experiments.robocasa_task_execution.counter_to_sink --physics
    python -m experiments.robocasa_task_execution.counter_to_sink --no-collision-avoidance

.. note::
    The kitchen assets must be downloaded once via
    ``python -m robocasa.scripts.download_kitchen_assets`` before an episode can be
    composed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, StrEnum

import mujoco

import numpy as np
from typing_extensions import TYPE_CHECKING, Self

from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType
    from robocasa.utils import object_utils

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import (
    ExecutionEnvironment,
    PhysicallySimulatedRobot,
)
from coraplex.plans.plan_node import PlanNode
from coraplex.plans.factories import sequential
from contextlib import nullcontext

from experiments.robocasa_task_execution.exceptions import NoClearStandingPose
from experiments.scene_recording import SceneRecorder, SceneRecording, ViewerCamera
from coraplex.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from experiments.robocasa_task_execution.placing_into_the_basin import (
    PlacingIntoTheBasin,
    SinkBasin,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    RoboCasaTask,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.grasping.grasp_candidates import HasGraspCandidates
from semantic_digital_twin.reasoning.robot_predicates import is_body_gripped
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.robots.tiago import TiagoMujoco
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactParameters,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)

if TYPE_CHECKING:
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
        PickPlaceCounterToSink,
    )

logger = logging.getLogger(__name__)

DEFAULT_RECORDING_PATH = Path.home() / "Downloads" / "robocasa_counter_to_sink.mp4"
"""
Where a recording is written when ``--record`` is given no path of its own.
"""


# %% names RoboCasa gives


class RoboCasaTaskName(StrEnum):
    """
    The RoboCasa kitchen task environments performed here.
    """

    PICK_PLACE_COUNTER_TO_SINK = "PickPlaceCounterToSink"


class KitchenRobot(Enum):
    """
    The robots that can be sent into a RoboCasa kitchen to perform a task.

    Each one reads its own description and declares its own drive, so spawning it into
    the kitchen is the same call either way.
    """

    PR2 = PR2
    TIAGO = TiagoMujoco


class RoboCasaObjectName(StrEnum):
    """
    The names RoboCasa gives the objects of a pick and place task.
    """

    TARGET = "obj"
    """
    The object the task is about, as opposed to the distractors placed around it.
    """


# %% how an episode ended


@dataclass
class CounterToSinkOutcome:
    """
    How an episode of the counter to sink task ended.
    """

    object_in_sink: bool
    """
    Whether RoboCasa's own check finds the target object inside the sink.
    """

    object_released: bool
    """
    Whether the robot let go of the target object, so that it no longer moves with the
    gripper.
    """

    @property
    def succeeded(self) -> bool:
        """
        Whether the task is accomplished: the target object lies in the sink and the
        robot has let go of it.
        """
        return self.object_in_sink and self.object_released


# %% one episode of the task


@dataclass
class CounterToSinkEpisode:
    """
    One episode of RoboCasa's counter to sink task, with a robot standing where
    RoboCasa starts its own one.
    """

    environment: PickPlaceCounterToSink
    """
    The reset RoboCasa environment the episode was composed in, holding RoboCasa's own
    ground truth such as the inside of the sink.
    """

    task: RoboCasaTask
    """
    The task, bound to the world the robot performs it in.
    """

    robot: AbstractRobot
    """
    The robot performing the task.
    """

    @classmethod
    def compose(
        cls,
        layout: LayoutType,
        style: StyleType,
        seed: int | None = None,
        robot: KitchenRobot = KitchenRobot.PR2,
    ) -> Self:
        """
        Compose a new episode, with the target object and distractors RoboCasa samples
        for it.

        :param layout: The kitchen layout to compose the episode in.
        :param style: The visual style of the kitchen.
        :param seed: The seed RoboCasa samples the objects and their placements with;
            ``None`` samples a different episode every time.
        :param robot: The robot to send into the kitchen.
        :return: The episode, with the robot standing clear of the kitchen at or behind
            RoboCasa's start pose.
        """
        loader = RoboCasaDatasetLoader()
        environment = loader.build_task_environment(
            RoboCasaTaskName.PICK_PLACE_COUNTER_TO_SINK, layout, style, seed=seed
        )
        task = loader.load_task_from_environment(environment)
        spawned = RobotSpecification(
            semantic_annotation_type=robot.value,
            odom_T_robot_start=task.robot_base_pose,
        ).spawn(task.world)
        episode = cls(environment=environment, task=task, robot=spawned)
        episode.stand_clear_of_the_kitchen()
        return episode

    @property
    def carrying_arm(self) -> Arm:
        """
        :return: The arm that picks the target object up and lowers it into the basin.
        """
        return self.robot.mobile_base.torso.right_arm

    def stand_clear_of_the_kitchen(
        self, step: float = 0.05, maximum_distance: float = 1.0
    ) -> None:
        """
        Park the arms of the robot and step it back from RoboCasa's start pose until it
        collides with nothing, while letting its base rest on the floor.

        :param step: How far the robot steps back at a time.
        :param maximum_distance: How far behind the start pose the robot stands at most.
        :raises NoClearStandingPose: If the robot collides wherever it stands up to
            ``maximum_distance`` behind the start pose.
        """
        world = self.task.world
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=list(self.robot.bodies_with_collision),
                    body_group_b=[
                        floor.root
                        for floor in world.get_semantic_annotations_by_type(Floor)
                    ],
                )
            )
        world.collision_manager.update_collision_matrix()
        for arm in self.robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        start_pose = self.task.robot_base_pose.to_np()
        drive = self.robot.root.parent_connection
        for distance in np.arange(0.0, maximum_distance + step / 2, step):
            start_T_standing_pose = np.eye(4)
            start_T_standing_pose[0, 3] = -distance
            drive.origin = start_pose @ start_T_standing_pose
            if not self.robot.is_in_collision:
                return
        raise NoClearStandingPose(
            start_pose=self.task.robot_base_pose, maximum_distance=maximum_distance
        )

    @property
    def target_object(self) -> HasGraspCandidates:
        """
        The object the task asks to be put into the sink.
        """
        world = self.task.world
        body = world.get_body_by_name(
            self.environment.objects[RoboCasaObjectName.TARGET].root_body
        )
        return next(
            annotation
            for annotation in world.get_semantic_annotations_by_type(HasGraspCandidates)
            if annotation.root is body
        )

    @property
    def target_object_released(self) -> bool:
        """
        Whether the target object no longer moves with the robot: it neither hangs off
        the robot nor lies gripped between the fingers of one of its grippers.
        """
        body = self.target_object.root
        hangs_off_the_robot = (
            body
            in self.task.world.get_kinematic_structure_entities_of_branch(
                self.robot.root
            )
        )
        gripped = any(
            is_body_gripped(body, arm.end_effector) for arm in self.robot.all_arms
        )
        return not hangs_off_the_robot and not gripped

    def placement_pose(self, drop_height: float = 0.1) -> Pose:
        """
        :param drop_height: How far above the rim of the sink the lowest point of the
            target object is let go of.
        :return: Where the target object is let go of: centred over the first basin of
            the sink, oriented as it stands now.
        """
        basin = SinkBasin.first_of(self.environment.sink)
        world = self.task.world
        body = self.target_object.root
        object_pose = body.global_pose
        lowest_point = (
            body.collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .min_z
        )
        height_above_lowest_point = object_pose.to_np()[2, 3] - lowest_point
        return Pose(
            position=Point3(
                basin.floor_center[0],
                basin.floor_center[1],
                basin.rim_height + height_above_lowest_point + drop_height,
            ),
            orientation=object_pose.quaternion,
            reference_frame=world.root,
        )

    def pick_up_plan(self, sampling_seed: int) -> PlanNode:
        """
        :param sampling_seed: The seed the standing poses are sampled with.
        :return: The plan picking the target object up from the counter with the right
            arm, standing wherever and grasping however a trial finds to work, and
            parking the arms afterwards.
        """
        context = Context(self.task.world, self.robot, sampling_seed=sampling_seed)
        arms = self.robot.all_arms
        pick_up = MoveAndPickUpAction.from_graspable_by_closest_grasps(
            graspable=self.target_object, arm=self.carrying_arm, context=context
        )
        return sequential(
            [
                MoveTorsoAction(TorsoState.HIGH),
                ParkArmsAction(arms),
                pick_up,
                ParkArmsAction(arms),
            ],
            context,
        )

    def placing_into_the_sink(self) -> PlacingIntoTheBasin:
        """
        :return: The placing of the target object, held by the right arm, into the first
            basin of the sink.
        """
        return PlacingIntoTheBasin(
            basin=SinkBasin.first_of(self.environment.sink),
            held_object=self.target_object.root,
            arm=self.carrying_arm,
            context=Context(self.task.world, self.robot),
        )

    def perform(
        self, sampling_seed: int = 42, collision_avoidance: bool = True
    ) -> CounterToSinkOutcome:
        """
        Have the robot carry the target object into the sink kinematically, and let
        RoboCasa's physics settle it there.

        :param sampling_seed: The seed the standing poses are sampled with.
        :param collision_avoidance: Whether the robot avoids colliding with the kitchen
            and with itself while moving.
        :return: How the episode ended.
        """
        with self.kinematic_environment(collision_avoidance):
            self.perform_the_pick_up(sampling_seed)
            self.perform_the_placing()
        self.let_the_target_object_fall()
        return self.outcome()

    def kinematic_environment(
        self, collision_avoidance: bool = True
    ) -> ExecutionEnvironment:
        """
        :param collision_avoidance: Whether the robot avoids colliding with the kitchen
            and with itself while moving.
        :return: The environment an episode is performed in without physics.
        """
        return ExecutionEnvironment(
            ExecutionType.KINEMATICALLY_SIMULATED,
            collision_avoidance=collision_avoidance,
        )

    def perform_the_pick_up(self, sampling_seed: int = 42) -> None:
        """
        Take the target object off the counter, in whatever environment is open.

        :param sampling_seed: The seed the standing poses are sampled with.
        """
        self.pick_up_plan(sampling_seed).perform()

    def perform_the_placing(self) -> None:
        """
        Lower the held object into the basin and let go of it, in whatever environment
        is open.
        """
        self.placing_into_the_sink().plan().perform()

    def start_physical_simulation(
        self,
        step_size: timedelta = timedelta(milliseconds=2),
        impedance_ratio: float = 10.0,
    ) -> MujocoSim:
        """
        Start simulating the episode's world in MuJoCo, stepped by whatever performs in
        it.

        Friction is resolved in elliptic cones, and the impedance ratio raised above
        MuJoCo's default of one, as MuJoCo's documentation recommends where objects are
        held by friction.

        :param step_size: How much simulated time one physics step advances.
        :param impedance_ratio: How much harder the contacts resist sliding than
            pressing in; MuJoCo's ``impratio``.
        :return: The started simulation; stop it when done.
        """
        self.let_the_fingers_hold_the_target_object()
        simulation = MujocoSim(
            world=self.task.world,
            headless=True,
            step_size=step_size.total_seconds(),
            integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
            impratio=impedance_ratio,
        )
        simulation.start_stepped_simulation()
        return simulation

    def camera_on_the_task(self, distance: float = 3.0) -> ViewerCamera:
        """
        :param distance: How far the camera stands from the object, in metres.
        :return: A camera orbiting the object the task is about, so the recording shows
            the counter, the sink and the robot between them.
        """
        position = self.target_object.root.global_transform.to_np()[:3, 3].ravel()
        return ViewerCamera(
            look_at=(float(position[0]), float(position[1]), float(position[2])),
            distance=distance,
            azimuth=135.0,
            elevation=-25.0,
        )

    def let_the_fingers_hold_the_target_object(self) -> None:
        """
        Give the target object and the fingertips contacts that resist being rolled.

        A contact that resolves sliding alone carries the object's weight but leaves it
        free to turn between the two pads under that weight, so it arrives in the basin
        on its side. The sliding friction stays the default a shape gets, since that is
        what carries the weight.
        """
        held = [self.target_object.root]
        for arm in self.robot.all_arms:
            if not isinstance(arm.end_effector, HasTwoFingers):
                continue
            held.extend(arm.end_effector.fingertips.both)
        ContactParameters.create_for_grasped_object(
            sliding_friction=ContactFriction().sliding
        ).apply_to(held)

    def perform_in_physics(
        self,
        simulation: MujocoSim,
        sampling_seed: int = 42,
        collision_avoidance: bool = True,
        settling_time: timedelta = timedelta(seconds=2),
        recording: SceneRecording | None = None,
    ) -> CounterToSinkOutcome:
        """
        Have the robot carry the target object into the sink in a physical simulation,
        where its fingers hold the object by friction alone and the object drops onto
        the basin floor once they let go.

        :param simulation: The started simulation of the episode's world, see
            :meth:`start_physical_simulation`.
        :param sampling_seed: The seed the standing poses are sampled with.
        :param collision_avoidance: Whether the robot avoids colliding with the kitchen
            and with itself while moving.
        :param settling_time: How long the physics runs on after the plan, for the
            object to come to rest.
        :param recording: Where to write a video of the episode, or nothing to run
            without recording one.
        :return: How the episode ended.
        """
        recorder = (
            nullcontext()
            if recording is None
            else SceneRecorder(
                simulation=simulation,
                camera=self.camera_on_the_task(),
                recording=recording,
            )
        )
        with recorder, PhysicallySimulatedRobot(
            simulation=simulation, collision_avoidance=collision_avoidance
        ):
            self.perform_the_pick_up(sampling_seed)
            self.perform_the_placing()
            simulation.step_simulation(settling_time)
        return self.outcome()

    def let_the_target_object_fall(self, settling_time: float = 2.0) -> None:
        """
        Let RoboCasa's physics move the released target object until it comes to rest,
        and put it where it landed. An object the robot still holds stays where it is.

        :param settling_time: How many seconds of simulated time the object falls for.
        """
        if not self.target_object_released:
            return
        joint = self.environment.objects[RoboCasaObjectName.TARGET].joints[0]
        self.copy_target_object_pose_into_robocasa()
        simulation = self.environment.sim
        simulation.data.set_joint_qvel(joint, np.zeros(6))
        for _ in range(round(settling_time / simulation.model.opt.timestep)):
            simulation.step()
        x, y, z, quaternion_w, quaternion_x, quaternion_y, quaternion_z = (
            simulation.data.get_joint_qpos(joint)
        )
        self.target_object.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                x,
                y,
                z,
                quaternion_x,
                quaternion_y,
                quaternion_z,
                quaternion_w,
                reference_frame=self.task.world.root,
            )
        )

    def copy_target_object_pose_into_robocasa(self) -> None:
        """
        Put the target object in RoboCasa's simulation where it stands in the world.
        """
        world_T_object = self.target_object.root.global_pose
        x, y, z, w = world_T_object.quaternion.to_np()
        self.environment.sim.data.set_joint_qpos(
            self.environment.objects[RoboCasaObjectName.TARGET].joints[0],
            np.concatenate([world_T_object.to_np()[:3, 3], [w, x, y, z]]),
        )
        self.environment.sim.forward()

    def outcome(self) -> CounterToSinkOutcome:
        """
        Judge the current state of the world by RoboCasa's own check, after copying the
        pose of the target object into RoboCasa's simulation.

        :return: How the episode stands.
        """
        self.copy_target_object_pose_into_robocasa()
        return CounterToSinkOutcome(
            object_in_sink=bool(
                object_utils.obj_inside_of(
                    self.environment,
                    RoboCasaObjectName.TARGET,
                    self.environment.sink,
                    partial_check=True,
                )
            ),
            object_released=self.target_object_released,
        )


# %% running an episode from the command line


def _parse_arguments() -> argparse.Namespace:
    """
    Parse the command-line arguments selecting the kitchen and how the episode is
    performed.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        default=LayoutType.LAYOUT001.name,
        choices=[layout.name for layout in LayoutType],
        help="RoboCasa kitchen layout to compose the episode in.",
    )
    parser.add_argument(
        "--style",
        default=StyleType.STYLE001.name,
        choices=[style.name for style in StyleType],
        help="RoboCasa visual style of the kitchen.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed RoboCasa samples the objects and their placements with.",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Perform the episode in MuJoCo physics instead of kinematically.",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=str(DEFAULT_RECORDING_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Write a video of a physical episode, by default to "
            f"{DEFAULT_RECORDING_PATH}."
        ),
    )
    parser.add_argument(
        "--robot",
        default=KitchenRobot.PR2.name,
        choices=[robot.name for robot in KitchenRobot],
        help="The robot that performs the task.",
    )
    parser.add_argument(
        "--collision-avoidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the robot avoids colliding with the kitchen and with itself.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Compose one episode of the counter to sink task, have the chosen robot perform it
    and report how it ended.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    arguments = _parse_arguments()
    episode = CounterToSinkEpisode.compose(
        LayoutType[arguments.layout],
        StyleType[arguments.style],
        seed=arguments.seed,
        robot=KitchenRobot[arguments.robot],
    )
    logger.info("Task: %s", episode.task.instruction)
    if not arguments.physics:
        outcome = episode.perform(collision_avoidance=arguments.collision_avoidance)
        logger.info("Outcome: %s, succeeded: %s", outcome, outcome.succeeded)
        return
    simulation = episode.start_physical_simulation()
    outcome = episode.perform_in_physics(
        simulation,
        collision_avoidance=arguments.collision_avoidance,
        recording=(
            None
            if arguments.record is None
            else SceneRecording(path=Path(arguments.record))
        ),
    )
    simulation.stop_simulation()
    logger.info("Outcome: %s, succeeded: %s", outcome, outcome.succeeded)
    if arguments.record is not None:
        logger.info("Wrote a video to %s", arguments.record)


if __name__ == "__main__":
    main()
