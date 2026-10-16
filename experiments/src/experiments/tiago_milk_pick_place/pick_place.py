"""
The scene and the plan: Tiago takes the milk carton from one spot on the apartment's
dining table, drives a short way along it, and sets the carton down.
"""

from __future__ import annotations

import math
import os

import numpy as np
from dataclasses import dataclass, field

from typing_extensions import List, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionLike, PlanNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from experiments.tiago_milk_pick_place.exceptions import (
    NotOverTheTable,
    OutOfReach,
    UnexpectedTableHeight,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.robots.tiago import TiagoMujoco
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.world_entity import Body

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

    standing_x: float = 4.25
    """
    Where the robot stands, west of the table, for both halves of the task.

    Close enough that the carton sits where the arm already reaches: the right hand
    rests about 0.37 m in front of the base and 0.49 m to its right, so a target put
    there needs no reach across the robot's own body.
    """

    standing_y_at_pick: float = 4.10
    """
    Where the robot stands along the table to reach the carton.
    """

    standing_y_at_place: float = 3.70
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

    pick_y: float = 3.70
    """
    Where the carton starts along the table.
    """

    place_x: float = 4.78
    """
    Where the carton is put down across the table.
    """

    place_y: float = 3.30
    """
    Where the carton is put down along the table, far enough from where it started that
    the robot has to drive there.
    """

    drive_translation_velocity_limit: float = 0.1
    """
    How fast the base may travel, in metres per second.

    A slow walk: the hand holds the carton by friction alone, so the robot drives no
    faster than the grip can take being accelerated.
    """

    drive_rotation_velocity_limit: float = 0.1
    """
    How fast the base may turn, in radians per second.

    Slower still, because turning is what the grip feels. The base only drives along the
    way it faces, so moving to a spot beside it means turning towards that spot, driving
    and turning back -- and each of those turns swings the carried carton through the
    arc of the outstretched arm.
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

    finger_clearance: float = 0.08
    """
    The gap left between the carton and the fingertips before the hand moves onto the
    grasp.

    The fingers are still shut when the hand waits there, so this is the gap between the
    carton and a closed hand. It is a generous one: the arm lags behind the straight
    line it is given most while it is getting up to speed, and the further back that
    happens the more of it is over with by the time the carton is between the fingers.
    """

    lift_height: float = 0.12
    """
    How far above the grasp the carton is carried between the two spots.
    """

    approach_velocity: float = 0.005
    """
    How fast the hand travels onto the grasp, in metres per second.

    A crawl. The open hand clears the carton by a centimetre on each side, and the arm
    lags behind the line it is given in proportion to how fast it is asked to travel, so
    this is set by how much of that centimetre the lag is allowed to eat.
    """

    arrival_tolerance: float = 0.002
    """
    How close the hand has to come to a goal before the next motion starts, in metres.

    Tighter than the default for the same reason: a motion that ends while the arm is
    still settling leaves the next one starting from a pose that is drifting, and half a
    centimetre of drift is all it takes to knock the carton over.
    """

    heading_tolerance: float = 0.005
    """
    How closely the hand has to face the way a goal states, in radians.

    This hand holds what it grasps a good 24 cm from the wrist the goals are stated at,
    so a heading that is off by the usual two hundredths of a radian puts the fingers
    half a centimetre to the side -- as much as they clear the carton by.
    """

    closing_velocity: float = 0.05
    """
    How fast the fingers close, in radians per second.

    A gripper that snaps shut hits the carton with one finger first and flicks it out
    from between them; closing slowly lets the second finger arrive before the carton
    has moved.
    """

    stall_minimum_time: float = 0.3
    """
    How long the fingers have to stand still against the carton for the grasp to count
    as closed, in seconds.

    The fingers never reach the angle they are asked for, because the carton is between
    them; what ends the motion is that they stop moving.
    """

    carry_velocity: float = 0.1
    """
    How fast the hand travels while it holds the carton, in metres per second.
    """

    arm_side: str = "right"
    """
    Which arm carries the carton.

    The robot faces the table along the world's x-axis, so its left arm points at the
    wall behind the table's north end and its right arm points over the open table. The
    arm is named rather than guessed at, because an arm chosen for being nearest the
    carton is the one that ends up in the wall.
    """

    face_width_tolerance: float = 0.005
    """
    How much wider than the narrowest grasp another may be and still count as closing
    across the carton's flat faces, in metres.

    The grasps a carton offers are evenly spaced around it, so some close across a pair
    of faces and some across the corners between them. A corner-to-corner grip is almost
    the carton's diagonal: the fingers meet two edges instead of two faces, and the
    carton turns out from between them as soon as it is lifted.
    """

    face_width_tolerance: float = 0.005
    """
    How much wider than the narrowest grasp another may be and still count as closing
    across the carton's flat faces, in metres.

    The grasps a carton offers are evenly spaced around it, so some close across a pair
    of faces and some across the corners between them. A corner-to-corner grip is almost
    the carton's diagonal: the fingers meet two edges instead of two faces, and the
    carton turns out from between them as soon as it is lifted.
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

    milk_density: float = 1.03
    """
    What milk weighs, in kilograms per litre.

    The carton's mesh carries no mass of its own, so a simulation would weigh it as
    water. Milk is a little denser, and the difference is what the fingers have to hold.
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

    used_robot: Type[AbstractRobot] = TiagoMujoco
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

    It is resolved to an absolute path before the carton is spawned: MuJoCo looks a mesh
    up relative to the scene file it writes, not relative to where the demo was started.
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

        The drive is limited to a slow walk, because the hand holds the carton by
        friction alone and every acceleration of the base is felt in that grip.
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

    def raise_the_torso(self, robot: AbstractRobot) -> None:
        """
        Stand the torso at the height the arm works from, before the simulation is
        built.

        The torso is set rather than commanded: its joint carries the whole upper body
        against a hard stop, and a servo asked to lift it saturates its force without
        moving, so the demo starts it where it needs to be and leaves it there.

        :param robot: The robot whose torso is raised.
        """
        robot.mobile_base.torso.get_joint_state_by_type(TorsoState.HIGH).apply_to(
            robot.root._world
        )

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
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        self.park_the_arms(robot)
        self.raise_the_torso(robot)
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
        carton_world = STLParser(os.path.abspath(self.carton_mesh_path)).parse()
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
        self._give_the_carton_its_weight(body)
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

    def _give_the_carton_its_weight(self, body: Body) -> None:
        """
        Weigh the carton as a real one of its size, filled with milk.

        Its mesh declares no mass, so a simulation would compute one from the shape at
        the density of water. Milk is denser than water, and what the carton weighs is
        exactly what the fingers have to hold by friction.

        :param body: The carton's body.
        """
        mesh = body.collision.shapes[0].mesh
        litres = float(mesh.volume) * 1000.0
        mass = litres * self.layout.milk_density
        extent = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
        width = float(extent.max_x - extent.min_x)
        depth = float(extent.max_y - extent.min_y)
        height = float(extent.max_z - extent.min_z)
        body.inertial = Inertial(
            mass=mass,
            inertia=InertiaTensor.from_values(
                mass * (depth**2 + height**2) / 12.0,
                mass * (width**2 + height**2) / 12.0,
                mass * (width**2 + depth**2) / 12.0,
                0.0,
                0.0,
                0.0,
            ),
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
        The robot drives between the two spots, so each has to be within the arm's reach
        of the pose it is reached from.

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
        :return: The arm the layout names, which is the one pointing over the table
            rather than at the wall behind it.
        """
        [arm] = [
            arm
            for arm in robot.all_arms
            if self.layout.arm_side in arm.name.name.lower()
        ]
        return arm

    def park_the_arms(self, robot: AbstractRobot) -> None:
        """
        Tuck both arms before the simulation is built.

        An arm left in the posture the description spawns it in reaches about a metre
        out to its side, which at this standing pose puts the left arm inside the wall
        behind the table. The physics would then start by pushing the arm out of the
        wall, and a motion planned from there never reaches its goal.

        :param robot: The robot whose arms are tucked.
        """
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(arm.root._world)

    def choose_grasp(self, arm: Arm) -> GraspCandidate:
        """
        :param arm: The arm that reaches for the carton.
        :return: The grasp that closes across a pair of the carton's flat faces, from
            whichever of those directions the robot faces most squarely.
        """
        candidates = self.carton.annotation.grasp_candidates()
        narrowest = min(candidate.width_along_closing_axis for candidate in candidates)
        across_the_faces = [
            candidate
            for candidate in candidates
            if candidate.width_along_closing_axis
            <= narrowest + self.layout.face_width_tolerance
        ]
        return max(across_the_faces, key=self._alignment_with_the_approach)

    def _alignment_with_the_approach(self, candidate: GraspCandidate) -> float:
        """
        :param candidate: A grasp on the carton.
        :return: How well the way the hand would travel onto it matches the line from
            the robot to the carton, from one for straight along it to minus one for
            straight back through the robot.
        """
        towards = np.array(
            [
                self.layout.pick_x - self.layout.standing_x,
                self.layout.pick_y - self.layout.standing_y_at_pick,
            ]
        )
        towards = towards / np.linalg.norm(towards)
        return float(candidate.world_T_grasp.to_np()[:2, 0] @ towards)

    def approach_clearance(self, arm: Arm) -> float:
        """
        How far back from the grasp the hand waits before it moves onto it, measured
        from the tool frame the goal is stated at.

        :param arm: The arm that reaches for the carton.
        :return: Far enough that the shut fingers stand clear of the carton, in metres.
        """
        return (
            arm.end_effector.fingers_reach_beyond_the_grasp
            + self.layout.finger_clearance
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Pick the carton up, drive along the table and set it down.

        The torso is already at the height the arm works from, set when the scene was
        populated, so the plan does not command it.

        The plan ends with the place, which already withdraws the hand from the carton
        it has just set down. Parking the arms after that sweeps the hand back across
        the table and knocks the carton over, because nothing in this demo keeps the arm
        away from what it is not touching on purpose.

        :param context: The context to plan in.
        :return: The plan.
        """
        return sequential(self.plan_steps(context), context=context)

    def plan_steps(self, context: Context) -> List[ActionLike]:
        """
        The steps of the plan, in order.

        :param context: The context to plan in.
        :return: Park, pick up, drive, place.
        """
        world = context.world
        robot = context.robot
        arm = self.choose_arm(robot)
        resting_z = (
            self.table_top
            + self.carton.grasp_height_above_bottom
            + self.layout.place_drop_height
        )
        return [
            ParkArmsAction(robot.all_arms),
            PickUpAction(
                grasp=self.choose_grasp(arm),
                arm=arm,
                approach_clearance=self.approach_clearance(arm),
                retreat_distance=self.layout.lift_height,
                pre_approach_linear_velocity=self.layout.carry_velocity,
                final_approach_linear_velocity=self.layout.approach_velocity,
                lift_linear_velocity=self.layout.carry_velocity,
                grasp_closing_velocity=self.layout.closing_velocity,
                grasp_stall_minimum_time=self.layout.stall_minimum_time,
                tolerate_grasp_stall=True,
                position_threshold=self.layout.arrival_tolerance,
                orientation_threshold=self.layout.heading_tolerance,
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
                approach_clearance=self.approach_clearance(arm),
                retreat_distance=self.layout.lift_height,
                transport_linear_velocity=self.layout.carry_velocity,
                placing_linear_velocity=self.layout.approach_velocity,
                release_opening_velocity=self.layout.closing_velocity,
                retract_linear_velocity=self.layout.approach_velocity,
                position_threshold=self.layout.arrival_tolerance,
                orientation_threshold=self.layout.heading_tolerance,
            ),
        ]

    def build_plan_prefix(self, context: Context, step_count: int) -> PlanNode:
        """
        The first steps of the plan, for finding which of them a run gets through.

        :param context: The context to plan in.
        :param step_count: How many of the plan's steps to take.
        :return: A plan of just those steps.
        """
        return sequential(self.plan_steps(context)[:step_count], context=context)
