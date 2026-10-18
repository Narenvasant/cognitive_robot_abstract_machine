"""
Tiago performs RoboCasa's pick-and-place tasks: it takes the object RoboCasa sampled off
where it lies and puts it where the task asks for, holding it by the grip of its fingers
alone.

Run one with (the ``experiments`` package must be importable)::

    python -m experiments.tiago_robocasa.pick_place PickPlaceCounterToSink
    python -m experiments.tiago_robocasa.pick_place PickPlaceCounterToSink --record
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from experiments.scene_recording import SceneRecording
from experiments.tiago_robocasa.robocasa_task import TiagoRoboCasaTask
from semantic_digital_twin.adapters.multi_sim import (
    ContactDimensionality,
    MujocoGeom,
)
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sink
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

DEFAULT_RECORDING_DIRECTORY = Path.home() / "Downloads"
"""
Where a recording is written when one is asked for without a path.
"""


# %% the objects a pick and place task names


class TaskObject(StrEnum):
    """
    The names RoboCasa gives the bodies of a pick-and-place task.
    """

    FOOD = "obj_main"
    """
    The object the task moves.
    """

    CONTAINER = "container_main"
    """
    What it is to be put into.
    """


# %% picking something up and putting it somewhere


@dataclass
class TiagoPickPlaceTask(TiagoRoboCasaTask):
    """
    Tiago takes the task's object off where it lies and puts it into the receptacle.
    """

    finger_clearance: float = 0.08
    """
    How far clear of the object the shut fingers wait before the hand moves on to the
    grasp, in metres, on top of how far they reach past it.
    """

    carry_velocity: float = 0.1
    """
    How fast the hand travels while it is not about to touch anything, in metres per
    second.
    """

    approach_velocity: float = 0.03
    """
    How fast the hand creeps the last stretch onto the grasp, in metres per second.
    """

    closing_velocity: float = 0.05
    """
    How fast the fingers close, in metres per second.
    """

    stall_minimum_time: float = 0.3
    """
    How long the fingers have to be stalled before the grasp counts as closed, in
    seconds.
    """

    arrival_tolerance: float = 0.03
    """
    How close the hand has to come to a goal before it counts as reached, in metres.

    Loose enough that a simulated arm, which trails what it is asked for while it
    moves, is taken to have arrived: held to millimetres it never is, and the motion
    gives up with the hand short of the grasp.
    """

    heading_tolerance: float = 0.15
    """
    How closely the hand has to face a goal before it counts as reached, in radians.
    """

    lift: float = 0.15
    """
    How far the object is lifted clear of what it rested on, in metres.
    """

    placing_reach: float = 0.65
    """
    How far in front of its base Tiago lets the object go, in metres.
    """

    release_clearance: float = 0.02
    """
    How far above the receptacle's floor the object's underside is let go, in metres.
    """

    floor_margin: float = 0.03
    """
    How far in from the edge of the receptacle's floor the object is let go, in metres.
    """

    @property
    def first_target(self) -> np.ndarray:
        """
        :return: Where the object is taken hold of: its middle.
        """
        return self.extent(self.food.root).center.to_np()[:3]

    @property
    def grasp_heading(self) -> float:
        """
        :return: The direction the hand approaches in, as a yaw in the kitchen's frame.
        """
        return math.pi / 2

    @property
    def food(self) -> HasGraspCandidates:
        """
        :return: The annotation of the object the task moves.
        """
        body = self.world.get_body_by_name(TaskObject.FOOD)
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(
                HasGraspCandidates
            )
            if annotation.root is body
        ][0]

    @property
    def receptacle(self) -> Body:
        """
        :return: The root of what the object is to be put into.
        """
        return self.world.get_body_by_name(TaskObject.CONTAINER)

    @property
    def receptacle_floor(self) -> VolumetricBoundingBox:
        """
        :return: The floor of the receptacle, as the widest of the convex pieces its
            collision geometry is built from.
        """
        pieces = [
            piece
            for body in self.colliding_bodies_below(self.receptacle)
            for piece in body.collision.as_bounding_box_collection_in_frame(
                self.world.root
            ).bounding_boxes
        ]
        return max(pieces, key=lambda piece: piece.width * piece.depth)

    def grasp(self) -> GraspCandidate:
        """
        :return: The grasp that takes the object at its middle, approaching towards
            :attr:`grasp_heading` and from above at :attr:`grasp_pitch`.
        """
        world_T_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.first_target,
            pitch=self.grasp_pitch,
            yaw=self.grasp_heading,
            reference_frame=self.world.root,
        )
        return GraspCandidate(
            self.food, self.world.transform(world_T_grasp.pose, self.food.root)
        )

    @property
    def release_height(self) -> float:
        """
        :return: The height above which the object is let go, in metres.
        """
        return self.receptacle_floor.max_z

    @property
    def release_point(self) -> np.ndarray:
        """
        :return: Where the centre of the object is let go: over the part of the
            receptacle's floor nearest Tiago, clear of its edge and its floor.
        """
        floor = self.receptacle_floor
        extent = self.extent(self.food.root)
        centre = floor.center.to_np()[:3]
        return np.array(
            [
                centre[0],
                min(centre[1], floor.min_y + extent.width / 2 + self.floor_margin),
                self.release_height + extent.height / 2 + self.release_clearance,
            ]
        )

    def goal(self) -> Pose:
        """
        :return: Where the object is to be put: its centre at :attr:`release_point`,
            turned as it stands now.
        """
        body = self.food.root
        world_T_food = body.global_pose.to_np()
        world_T_food[:3, 3] += self.release_point - self.extent(body).center.to_np()[:3]
        return HomogeneousTransformationMatrix(
            world_T_food, reference_frame=self.world.root
        ).pose

    def pre_grasp(self) -> Pose:
        """
        :return: Where the hand waits before it moves onto the grasp, read afresh so it
            follows the robot as the robot is moved about.
        """
        grasp = self.grasp()
        return (
            self.pick_up_of(grasp)
            .grasp_pose_sequence(grasp.world_T_grasp, self.arm.end_effector, grasp)
            .pre_grasp
        )

    def pick_up_of(self, grasp: GraspCandidate) -> PickUpAction:
        """
        :param grasp: The grasp to take.
        :return: The pick up taking it, with the hand creeping the last stretch onto it
            and the fingers alone carrying what they close on.
        """
        arm = self.arm
        return PickUpAction(
            grasp,
            arm,
            approach_clearance=(
                arm.end_effector.fingers_reach_beyond_the_grasp + self.finger_clearance
            ),
            retreat_distance=self.lift,
            pre_approach_linear_velocity=self.carry_velocity,
            final_approach_linear_velocity=self.approach_velocity,
            lift_linear_velocity=self.carry_velocity,
            grasp_closing_velocity=self.closing_velocity,
            grasp_stall_minimum_time=self.stall_minimum_time,
            tolerate_grasp_stall=True,
            held_by_contact_alone=True,
            position_threshold=self.arrival_tolerance,
            orientation_threshold=self.heading_tolerance,
        )

    def build_plan(self, context: Context) -> Plan:
        """
        Stand where the hand reaches the object, bring the arm there, take the object,
        and unless the task stops at that, carry it to the receptacle and put it in.

        :param context: The context to plan in.
        :return: The plan.
        """
        self.stand_where_the_hand_arrives(self.pre_grasp)
        pick_up = self.pick_up_of(self.grasp())
        taking = [
            ParkArmsAction(self.robot.all_arms),
            *self.arm_motions_to(self.pre_grasp()),
            pick_up,
        ]
        if not self.puts_it_in_the_receptacle:
            return sequential(taking, context=context).plan
        return sequential(
            taking
            + [
                NavigateAction(
                    self.standing_pose(self.release_point, self.placing_reach)
                ),
                PlaceAction(self.food, self.goal(), rises_off_the_object=True),
                ParkArmsAction(self.robot.all_arms),
            ],
            context=context,
        ).plan

    def prepare_contacts(self) -> None:
        """
        Have the object's contacts resolve the twisting and rolling friction it is
        given, so it neither turns in the hand nor rolls on once it is let go.
        """
        for shape in self.food.root.collision.shapes:
            shape.get_simulator_property_of_type(MujocoGeom).contact_dimensionality = (
                ContactDimensionality.SLIDING_TWISTING_AND_ROLLING
            )


# %% the tasks


@dataclass
class PickPlaceCounterToSink(TiagoPickPlaceTask):
    """
    Tiago takes the object off the counter and puts it in the sink.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task with: the object near the counter's edge, right
    of the sink.
    """

    food_category: str | None = None
    """
    Which kind of object RoboCasa is to put on the counter, or nothing to let it sample
    whatever it likes.

    Left open: holding the category fixed pins the object to the same corner of the
    counter every time, and Tiago cannot stand at that corner.
    """

    @property
    def task_parameters(self) -> dict[str, Any]:
        """
        :return: What RoboCasa is told about the task beyond its name.
        """
        if self.food_category is None:
            return {}
        return {"obj_groups": self.food_category}

    @property
    def receptacle(self) -> Body:
        """
        :return: The sink, which is a fixture of the kitchen rather than an object the
            task placed.
        """
        return self.world.get_semantic_annotations_by_type(Sink)[0].root

    @property
    def release_height(self) -> float:
        """
        :return: The sink's rim, which the object is dropped in from just above, read
            off the sink rather than off what the counters were set to.
        """
        return float(self.extent(self.receptacle).max_z)


# %% running one from the command line


def main() -> None:
    """
    Run the task the command line names and report whether the object was moved.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["PickPlaceCounterToSink"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="PATH")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--noslip", type=int, default=10)
    parser.add_argument("--reach", type=float, default=None)
    parser.add_argument("--pyramidal", action="store_true")
    parser.add_argument("--left-arm", action="store_true")
    parser.add_argument("--pitch", type=float, default=None, help="grasp pitch in degrees")
    parser.add_argument("--counter-height", type=float, default=None)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--pick-only", action="store_true")
    parser.add_argument("--no-avoidance", action="store_true")
    arguments = parser.parse_args()

    recording = None
    if arguments.record is not None:
        path = Path(arguments.record) if arguments.record else (
            DEFAULT_RECORDING_DIRECTORY / f"tiago_{arguments.task.lower()}.mp4"
        )
        recording = SceneRecording(path=path)
    task = PickPlaceCounterToSink(
        recording=recording,
        headless=not arguments.viewer,
        noslip_iterations=arguments.noslip,
        works_with_the_left_arm=arguments.left_arm,
        friction_cone=(
            mujoco.mjtCone.mjCONE_PYRAMIDAL
            if arguments.pyramidal
            else mujoco.mjtCone.mjCONE_ELLIPTIC
        ),
    )
    if arguments.reach is not None:
        task.reach = arguments.reach
    if arguments.pitch is not None:
        task.grasp_pitch = math.radians(arguments.pitch)
    if arguments.counter_height is not None:
        task.counter_height = arguments.counter_height
    if arguments.seed is not None:
        task.seed = arguments.seed
    for attempt in range(1, arguments.attempts + 1):
        episode = PickPlaceCounterToSink(
            recording=recording,
            headless=not arguments.viewer,
            noslip_iterations=arguments.noslip,
            works_with_the_left_arm=arguments.left_arm,
            friction_cone=(
                mujoco.mjtCone.mjCONE_PYRAMIDAL
                if arguments.pyramidal
                else mujoco.mjtCone.mjCONE_ELLIPTIC
            ),
        )
        if arguments.seed is not None:
            episode.seed = arguments.seed
        if arguments.reach is not None:
            episode.reach = arguments.reach
        if arguments.pitch is not None:
            episode.grasp_pitch = math.radians(arguments.pitch)
        if arguments.counter_height is not None:
            episode.counter_height = arguments.counter_height
        episode.puts_it_in_the_receptacle = not arguments.pick_only
        episode.collision_avoidance = not arguments.no_avoidance
        try:
            episode.build_world()
            started = episode.extent(episode.food.root).center.to_np()[:3].copy()
            logger.info("episode %d: %s", attempt, episode.task.instruction)
            episode.run()
        except Exception as failure:
            logger.info("episode %d stopped: %s: %s", attempt, type(failure).__name__, failure)
            continue
        ended = episode.extent(episode.food.root).center.to_np()[:3]
        rose = (ended[2] - started[2]) * 100
        logger.info(
            "episode %d finished: the object went from %s to %s, rising %.1f cm",
            attempt, started.round(3), ended.round(3), rose,
        )
        break
    else:
        logger.error("no episode of %d ran to its end", arguments.attempts)
        return

    if recording is not None:
        logger.info("wrote a video to %s", recording.path)


if __name__ == "__main__":
    main()
