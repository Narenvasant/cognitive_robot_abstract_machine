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
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from experiments.scene_recording import SceneRecording
from experiments.tiago_robocasa.exceptions import NoClearStandingPose
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

    def prepare_contacts(self) -> None:
        """
        Have the object's contacts resolve the twisting and rolling friction it is given,
        so it neither turns in the hand nor rolls on once it is let go.
        """
        for shape in self.food.root.collision.shapes:
            shape.get_simulator_property_of_type(MujocoGeom).contact_dimensionality = (
                ContactDimensionality.SLIDING_TWISTING_AND_ROLLING
            )

    def build_plan(self, context: Context) -> Plan:
        """
        Stand in line with the object, bring the hand to where the pick up starts its
        approach, take the object, drive to the receptacle holding it, and put it in.

        :param context: The context to plan in.
        :return: The plan.
        """
        arm = self.arm
        grasp = self.grasp()
        pick_up = PickUpAction(
            grasp,
            arm,
            retreat_distance=self.lift,
            held_by_contact_alone=True,
            tolerate_grasp_stall=True,
        )
        pre_grasp = pick_up.grasp_pose_sequence(
            grasp.world_T_grasp, arm.end_effector, grasp
        ).pre_grasp
        return sequential(
            [
                ParkArmsAction(self.robot.all_arms),
                NavigateAction(self.standing_pose(self.first_target, self.reach)),
                MoveToolCenterPointMotion(
                    pre_grasp, arm, allow_gripper_collision=True
                ),
                pick_up,
                NavigateAction(
                    self.standing_pose(self.release_point, self.placing_reach)
                ),
                PlaceAction(self.food, self.goal(), rises_off_the_object=True),
                ParkArmsAction(self.robot.all_arms),
            ],
            context=context,
        ).plan


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

    food_category: str | None = "canned_food"
    """
    A can, which stands upright and is a shape two flat pads can hold.
    """

    @property
    def task_parameters(self) -> dict[str, Any]:
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
        :return: The sink's rim, level with the counter: the object is dropped in from
            just above it.
        """
        return self.counter_height


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
        try:
            task.build_world()
            break
        except NoClearStandingPose as refused:
            logger.info("episode %d: %s", attempt, refused.error_message())
    else:
        logger.error("no episode in %d put Tiago within reach", arguments.attempts)
        return
    started = task.extent(task.food.root).center.to_np()[:3].copy()
    logger.info("task: %s", task.task.instruction)
    task.run()
    ended = task.extent(task.food.root).center.to_np()[:3]
    logger.info(
        "the object went from %s to %s, rising %.1f cm",
        started.round(3),
        ended.round(3),
        (ended[2] - started[2]) * 100,
    )
    if recording is not None:
        logger.info("wrote a video to %s", recording.path)


if __name__ == "__main__":
    main()
