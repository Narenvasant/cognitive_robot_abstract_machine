"""
The scene and the plan: Tiago takes the milk carton from one spot on the dining table
and sets it down at another, without moving its base.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionLike, PlanNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
)
from experiments.tiago_milk_pick_place.exceptions import (
    NotOverTheTable,
    OutOfReach,
    UnexpectedTableHeight,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World


# %% layout


@dataclass
class MilkPickPlaceLayout:
    """
    Every position and distance of the scene and the plan, in metres unless a field says
    otherwise.

    The base never moves, so both the pick and the place have to lie within the arm's
    reach of where the robot stands.
    """

    table_name: str = "table_area_main"
    """
    Name of the body the carton stands on.
    """

    expected_table_top: float = 0.7226
    """
    How high that top is expected to be, measured from the apartment's collision
    geometry.
    """

    table_top_tolerance: float = 0.02
    """
    How far the measured top may differ from the expected one before the demo stops.
    """

    standing_x: float = 4.05
    """
    Where the robot stands, west of the table, for both halves of the task.
    """

    standing_y_at_pick: float = 4.10
    """
    Where the robot stands along the table to reach the carton.
    """

    standing_y_at_place: float = 3.55
    """
    Where the robot stands along the table to set the carton down, which it drives to
    while holding it.
    """

    standing_yaw: float = 0.0
    """
    Which way the robot faces, in radians; zero looks along the world's x-axis, at the
    table.
    """

    pick_x: float = 4.78
    """
    Where the carton starts across the table.
    """

    pick_y: float = 4.10
    """
    Where the carton starts along the table.
    """

    place_x: float = 4.78
    """
    Where the carton is put down across the table.
    """

    place_y: float = 3.55
    """
    Where the carton is put down along the table, far enough from where it started that
    the robot has to drive there.
    """

    drive_translation_velocity_limit: float = 0.15
    """
    How fast the base may travel, in metres per second.

    The base is written into the simulation once per control cycle rather than driven by
    its wheels, so the physics sees no velocity for those jumps; a slow drive keeps each
    jump small enough not to shake the carton out of the hand.
    """

    drive_rotation_velocity_limit: float = 0.3
    """
    How fast the base may turn, in radians per second, for the same reason.
    """

    clearance_above_table: float = 0.003
    """
    How far above the table the carton is spawned, so the physics settles it down rather
    than pushing it out of a touching contact.
    """

    place_drop_height: float = 0.008
    """
    How far above its resting height the carton is released, so one that slipped in the
    hand is not pressed into the table.
    """

    lift_height: float = 0.12
    """
    How far above the grasp the carton is carried between the two spots.
    """

    maximum_reach: float = 0.85
    """
    How far the arm reaches from where the robot stands.
    """

    maximum_tilt_degrees: float = 15.0
    """
    How far the carton may lean from upright at the end and still count as standing.
    """

    gripper_far_distance: float = 0.20
    """
    How far the hand has to end up from the carton for the task to count as finished.
    """

    place_tolerance: float = 0.08
    """
    How close to the target the carton has to come to rest, in metres.
    """

    def standing_pose(self, world: World, y: float) -> Pose:
        """
        :param world: The world the pose is expressed in.
        :param y: Where along the table the robot stands, in metres.
        :return: Where the robot stands, facing the table.
        """
        return Pose.from_xyz_rpy(
            self.standing_x,
            y,
            0.0,
            yaw=self.standing_yaw,
            reference_frame=world.root,
        )

    def distance_from(self, standing_y: float, x: float, y: float) -> float:
        """
        :param standing_y: Where along the table the robot stands, in metres.
        :param x: A point's x coordinate in the world.
        :param y: A point's y coordinate in the world.
        :return: How far the point lies from the robot, horizontally.
        """
        return float(math.hypot(x - self.standing_x, y - standing_y))


# %% the scene


@dataclass
class MilkOnTheTable:
    """
    The carton once it stands in the world, with what the plan needs to know about it.
    """

    annotation: Milk
    """
    The carton in the world, which the plan picks up and the check reads back.
    """

    width: float
    """
    How wide it is where the fingers close, in metres.
    """

    height: float
    """
    How tall it is, in metres.
    """

    grasp_height_above_bottom: float
    """
    How far above its bottom its body origin sits, in metres.
    """

    @property
    def name(self) -> str:
        """
        :return: The name its body carries.
        """
        return self.annotation.root.name.name


# %% demonstration


@dataclass
class MilkPickPlaceDemonstration(RobotDemonstration):
    """
    Tiago moves the milk carton from one spot on the dining table to another.

    ..note:: The physical run does not call :meth:`RobotDemonstration.run`, because
        MuJoCo has to be built between populating the scene and performing the plan.
    """

    used_robot: Type[AbstractRobot] = Tiago
    """
    The robot that performs the task.
    """

    layout: MilkPickPlaceLayout = field(default_factory=MilkPickPlaceLayout)
    """
    Every position of the scene and the plan.
    """

    apartment_path: str = "coraplex/resources/worlds/apartment.urdf"
    """
    The environment the task is performed in, relative to the repository root.
    """

    carton_mesh_path: str = "coraplex/resources/objects/milk.stl"
    """
    The carton's mesh, relative to the repository root.
    """

    ros_node_name: str = "tiago_milk_pick_place_node"
    """
    What the demo's ROS node is called, when one is needed.
    """

    carton: MilkOnTheTable | None = field(init=False, default=None)
    """
    The carton once it stands in the world.
    """

    table_top: float = field(init=False, default=0.0)
    """
    The height of the table's surface, measured when the scene is populated.
    """

    def build_simulated_world(self) -> World:
        """
        Put Tiago in the apartment, standing at the dining table.

        The drive is limited to a slow walk: the base's pose is written into the
        simulation once per control cycle rather than driven by its wheels, so the
        physics sees no velocity for those jumps, and a slow drive keeps each one small.
        """
        return WorldSpecification.from_urdf(
            self.apartment_path,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(
                        self.layout.standing_x,
                        self.layout.standing_y_at_pick,
                        0.0,
                        yaw=self.layout.standing_yaw,
                    ),
                    drive_translation_velocity_limits=(
                        self.layout.drive_translation_velocity_limit
                    ),
                    drive_rotation_velocity_limits=(
                        self.layout.drive_rotation_velocity_limit
                    ),
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        """
        :param world: The world to look in.
        :return: Whether the carton already stands in it.
        """
        return world.is_kinematic_structure_entity_in_world_by_name("milk.stl")

    def measure_table_top(self, world: World) -> float:
        """
        Read the height of the table's top surface off its collision geometry.

        :param world: The world holding the table.
        :return: The height of the top, in metres.
        :raises UnexpectedTableHeight: If it is not the height the layout was written
            for.
        """
        table = world.get_body_by_name(self.layout.table_name)
        box = table.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        if (
            abs(box.max_z - self.layout.expected_table_top)
            > self.layout.table_top_tolerance
        ):
            raise UnexpectedTableHeight(
                self.layout.table_name,
                float(box.max_z),
                self.layout.expected_table_top,
                self.layout.table_top_tolerance,
            )
        return float(box.max_z)

    def populate_scene(self, world: World) -> None:
        """
        Stand the carton on the table and check the plan can reach both spots.

        :param world: The world to populate.
        :raises UnexpectedTableHeight: If the table is not where the layout expects.
        :raises NotOverTheTable: If a spot does not lie over the table.
        :raises OutOfReach: If a spot lies beyond the arm.
        """
        self.table_top = self.measure_table_top(world)
        self._check_both_spots_are_over_the_table(world)
        self._check_both_spots_are_within_reach()
        self.carton = self._spawn_carton(world)

    def _spawn_carton(self, world: World) -> MilkOnTheTable:
        """
        Put the carton on the table and measure it.

        :param world: The world to add it to.
        :return: The carton, with its measurements.
        """
        carton_world = STLParser(self.carton_mesh_path).parse()
        world.merge_world(carton_world)
        body = world.get_body_by_name("milk.stl")
        own_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            self.layout.pick_x,
            self.layout.pick_y,
            self.table_top - float(own_box.min_z) + self.layout.clearance_above_table,
            reference_frame=world.root,
        )
        world.notify_state_change()
        annotation = Milk(root=body)
        with world.modify_world():
            world.add_semantic_annotation(annotation)
        box = body.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        origin_z = float(body.global_transform.to_np()[2, 3])
        return MilkOnTheTable(
            annotation=annotation,
            width=float(box.max_y - box.min_y),
            height=float(box.max_z - box.min_z),
            grasp_height_above_bottom=origin_z - float(box.min_z),
        )

    def _check_both_spots_are_over_the_table(self, world: World) -> None:
        """
        Both spots have to lie over the table rather than beside it.

        :param world: The world holding the table.
        :raises NotOverTheTable: If one of them lies outside.
        """
        table = world.get_body_by_name(self.layout.table_name)
        box = table.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        for name, x, y in (
            ("the carton", self.layout.pick_x, self.layout.pick_y),
            ("the target", self.layout.place_x, self.layout.place_y),
        ):
            if not (box.min_x <= x <= box.max_x and box.min_y <= y <= box.max_y):
                raise NotOverTheTable(name, x, y)

    def _check_both_spots_are_within_reach(self) -> None:
        """
        The robot drives between the two spots, so each has to be within the arm's
        reach of the pose it is reached from.

        :raises OutOfReach: If one of them lies beyond it.
        """
        for name, standing_y, x, y in (
            (
                "the carton",
                self.layout.standing_y_at_pick,
                self.layout.pick_x,
                self.layout.pick_y,
            ),
            (
                "the target",
                self.layout.standing_y_at_place,
                self.layout.place_x,
                self.layout.place_y,
            ),
        ):
            distance = self.layout.distance_from(standing_y, x, y)
            if distance > self.layout.maximum_reach:
                raise OutOfReach(name, distance, self.layout.maximum_reach)

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around Tiago in ``world``.

        :param world: The world the plan runs in.
        :return: The context the plan is built and performed with.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
            _debug=self.debug,
        )

    def choose_arm(self, robot: AbstractRobot) -> Arm:
        """
        :param robot: The robot that performs the task.
        :return: The arm whose hand rests nearer the carton, which is the one that
            reaches it without crossing the robot's body.
        """
        return min(
            robot.all_arms,
            key=lambda arm: float(
                np.linalg.norm(
                    arm.end_effector.tool_frame.global_transform.to_np()[:2, 3]
                    - np.array([self.layout.pick_x, self.layout.pick_y])
                )
            ),
        )

    def choose_grasp(self, arm: Arm) -> GraspCandidate:
        """
        :param arm: The arm that reaches for the carton.
        :return: The grasp whose approach best matches the line from the robot to the
            carton, so the hand does not reach through it.
        """
        towards = np.array(
            [
                self.layout.pick_x - self.layout.standing_x,
                self.layout.pick_y - self.layout.standing_y_at_pick,
            ]
        )
        towards = towards / np.linalg.norm(towards)
        return max(
            self.carton.annotation.grasp_candidates(),
            key=lambda candidate: float(
                candidate.world_T_grasp.to_np()[:2, 0] @ towards
            ),
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Pick the carton up, lift it, drive along the table, and set it down there.

        The arm is not parked while the carton is held: the park posture can swing a
        held object against the robot's own body, and nothing excludes that contact.

        :param context: The context to plan in.
        :return: The plan.
        """
        world = context.world
        robot = context.robot
        arm = self.choose_arm(robot)
        grasp = self.choose_grasp(arm)
        tool_at_grasp = arm.end_effector.tool_frame_goal_for_grasp(grasp.world_T_grasp)
        lifted = Pose(
            position=Point3(
                x=float(tool_at_grasp.position.to_np()[0]),
                y=float(tool_at_grasp.position.to_np()[1]),
                z=float(tool_at_grasp.position.to_np()[2]) + self.layout.lift_height,
                reference_frame=world.root,
            ),
            orientation=tool_at_grasp.orientation,
            reference_frame=world.root,
        )
        resting_z = (
            self.table_top
            + self.carton.grasp_height_above_bottom
            + self.layout.place_drop_height
        )
        steps: List[ActionLike] = [
            ParkArmsAction(robot.all_arms),
            MoveTorsoAction(TorsoState.HIGH),
            PickUpAction(grasp, arm),
            MoveManipulatorAction(
                target_pose=lifted,
                end_effector=arm.end_effector,
                allow_gripper_collision=True,
            ),
            NavigateAction(
                self.layout.standing_pose(world, self.layout.standing_y_at_place)
            ),
            PlaceAction(
                object_designator=self.carton.annotation,
                target_location=Pose.from_xyz_rpy(
                    self.layout.place_x,
                    self.layout.place_y,
                    resting_z,
                    reference_frame=world.root,
                ),
            ),
            ParkArmsAction(robot.all_arms),
        ]
        return sequential(steps, context=context)
