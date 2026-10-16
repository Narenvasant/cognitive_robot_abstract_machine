"""
The Toyota HSR performs RoboCasa's pick-and-place tasks in a RoboCasa kitchen,
simulated in MuJoCo: it picks an object up where the task puts it and sets it into the
receptacle the task names.

RoboCasa composes the kitchen and samples the objects; the HSR replaces the robot
RoboCasa would use. The object is held by nothing but the friction between it and the
fingers.

Run the stove task with::

    python -m experiments.hsr_robocasa.pick_place
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import yaml
from krrood.ormatic.utils import classproperty
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from semantic_digital_twin.adapters.multi_sim import (
    ContactDimensionality,
    MujocoGeom,
    MujocoSim,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    RoboCasaTask,
)
from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Sink
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

from experiments.hsr_kitchen_mujoco.demo import (
    FRICTION_CONE,
    IMPEDANCE_RATIO,
    NO_SLIP_ITERATIONS,
    SETTLING_DURATION,
    STEP_SIZE,
    hand_offset,
)

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import (
        LayoutType,
        StyleType,
        get_layout_path,
    )

# %% the scene


class TaskObject(StrEnum):
    """
    The bodies RoboCasa adds for a pick-and-place task, by the names it gives them.
    """

    FOOD = "obj_main"
    """
    The object to pick up and set down, food in most tasks.
    """

    CONTAINER = "container_main"
    """
    The container the object is to be set into, where the task has one.
    """


class KitchenBody(StrEnum):
    """
    The bodies of the RoboCasa kitchen the experiment refers to, by their names.
    """

    FLOOR = "floor_room_main"
    """
    The floor, which the HSR's base stands on and so never keeps clear of.
    """

    FLOOR_BACKING = "floor_backing_room_main"
    """
    The slab under the floor, which the HSR's base comes as close to as to the floor.
    """


class WristLink(StrEnum):
    """
    The links of the HSR's wrist, by the names its description gives them. Taking an
    object from where it lies and setting it into its receptacle brings them as close
    to these as the hand itself, so they may touch them as the hand may.
    """

    WRIST_FLEX = "wrist_flex_link"
    """
    The link the wrist flexes.
    """

    WRIST_ROLL = "wrist_roll_link"
    """
    The link the wrist rolls, which carries the hand.
    """


@dataclass
class HSRBArmPlane:
    """
    The vertical plane the HSR's arm swings its hand in while its arm and wrist roll
    joints stand at zero, and the joint positions that put the hand at a pose in it.
    In that plane, the hand's tilt depends on the sum of arm and wrist flex alone, and
    the wrist's place on arm flex and lift alone.
    """

    search_steps: int = 600
    """
    How many positions each of the two searches tries within a joint's limits.
    """

    world: World = field(init=False)
    """
    A world holding nothing but an HSR, whose joints the searches set, so that no
    simulated robot moves.
    """

    robot: HSRB = field(init=False)
    """
    The HSR of :attr:`world`.
    """

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
        self.robot = RobotSpecification(HSRB).spawn(self.world)

    def joint_positions_for(self, base_T_tool: np.ndarray) -> dict[HSRBJoint, float]:
        """
        :param base_T_tool: Where the tool frame is to be, relative to the HSR's root;
            it has to lie in the arm's plane.
        :return: The positions of the arm's joints that put the tool frame there, as
            near as the searches get.
        """
        end_effector = self.robot.all_arms[0].end_effector
        wanted_approach = base_T_tool[:3, :3] @ end_effector.approach_axis.to_np()[:3]
        self._set({HSRBJoint.ARM_ROLL: 0.0, HSRBJoint.WRIST_ROLL: 0.0})
        tilt = max(
            self._tilts(),
            key=lambda candidate: self._approach_at(candidate) @ wanted_approach,
        )
        self._set_tilt(tilt)
        wrist_T_tool = self._base_T(end_effector.tool_frame)[:3, 3] - self._wrist()
        wanted_wrist = base_T_tool[:3, 3] - wrist_T_tool
        lift_limits = self._limits(HSRBJoint.ARM_LIFT)
        wrist_limits = self._limits(HSRBJoint.WRIST_FLEX)
        candidates = []
        for flex in np.linspace(*self._limits(HSRBJoint.ARM_FLEX), self.search_steps):
            if not wrist_limits[0] <= tilt - flex <= wrist_limits[1]:
                continue
            self._set({HSRBJoint.ARM_LIFT: 0.0, HSRBJoint.ARM_FLEX: flex})
            wrist = self._wrist()
            lift = wanted_wrist[2] - wrist[2]
            if lift_limits[0] <= lift <= lift_limits[1]:
                candidates.append((abs(wanted_wrist[0] - wrist[0]), flex, lift))
        _, flex, lift = min(candidates)
        return {
            HSRBJoint.ARM_LIFT: float(lift),
            HSRBJoint.ARM_FLEX: float(flex),
            HSRBJoint.ARM_ROLL: 0.0,
            HSRBJoint.WRIST_FLEX: float(tilt - flex),
            HSRBJoint.WRIST_ROLL: 0.0,
        }

    def _tilts(self) -> np.ndarray:
        """
        :return: The sums of arm and wrist flex the joints' limits allow.
        """
        flex = self._limits(HSRBJoint.ARM_FLEX)
        wrist = self._limits(HSRBJoint.WRIST_FLEX)
        return np.linspace(flex[0] + wrist[0], flex[1] + wrist[1], self.search_steps)

    def _set_tilt(self, tilt: float) -> None:
        """
        Tilt the hand by ``tilt``, the sum of arm and wrist flex, with the arm flexed as
        little as the wrist's limits allow.
        """
        flex = float(np.clip(tilt, *self._limits(HSRBJoint.ARM_FLEX)))
        self._set({HSRBJoint.ARM_FLEX: flex, HSRBJoint.WRIST_FLEX: tilt - flex})

    def _approach_at(self, tilt: float) -> np.ndarray:
        """
        :return: The direction the hand approaches along when tilted by ``tilt``,
            relative to the HSR's root.
        """
        self._set_tilt(tilt)
        end_effector = self.robot.all_arms[0].end_effector
        return (
            self._base_T(end_effector.tool_frame)[:3, :3]
            @ end_effector.approach_axis.to_np()[:3]
        )

    def _wrist(self) -> np.ndarray:
        """
        :return: Where the wrist flex joint is, relative to the HSR's root.
        """
        wrist = self.world.get_connection_by_name(HSRBJoint.WRIST_FLEX).child
        return self._base_T(wrist)[:3, 3]

    def _base_T(self, entity: KinematicStructureEntity) -> np.ndarray:
        """
        :return: The pose of ``entity`` relative to the HSR's root.
        """
        return self.world.compute_forward_kinematics_np(self.robot.root, entity)

    def _limits(self, joint: HSRBJoint) -> tuple[float, float]:
        """
        :return: The lower and upper position limit of ``joint``.
        """
        limits = self.world.get_connection_by_name(joint).dof.limits
        return limits.lower.position, limits.upper.position

    def _set(self, positions: dict[HSRBJoint, float]) -> None:
        """
        Set the joints of :attr:`world`'s HSR to ``positions``.
        """
        for joint, position in positions.items():
            self.world.get_connection_by_name(joint).position = position


class LayoutKey(StrEnum):
    """
    The keys of a RoboCasa layout this experiment reads or changes.
    """

    SIZE = "size"
    """
    A fixture's width, depth and height.
    """

    POSITION = "pos"
    """
    Where a fixture stands, along the wall, out from it, and up.
    """

    STACK_HEIGHT = "stack_height"
    """
    How high the fixtures below a stacked fixture reach.
    """

    MAIN_GROUP = "main_group"
    """
    The fixtures along the kitchen's main wall.
    """

    BOTTOM_ROW = "bottom_row"
    """
    The counters and appliances standing on the floor along the main wall.
    """

    TYPE = "type"
    """
    What kind of fixture an entry is.
    """

    NAME = "name"
    """
    The name of a fixture.
    """


class FixtureType(StrEnum):
    """
    The kinds of fixture this experiment looks for in a RoboCasa layout.
    """

    COUNTER = "counter"
    """
    A counter.
    """


class RowOffTheFloor(StrEnum):
    """
    The rows of a RoboCasa layout whose fixtures hang on a wall or make up the room,
    rather than stand on the floor.
    """

    WALLS = "walls"
    """
    The walls themselves.
    """

    FLOOR = "floor"
    """
    The floor itself.
    """

    WALL_ACCESSORIES = "wall_accessories"
    """
    Outlets, switches and the like on the walls.
    """

    TOP_ROW = "top_row"
    """
    The wall cabinets and the appliances hanging between them.
    """

    WALL_STACK = "wall_stack"
    """
    Fixtures stacked on a wall.
    """

    WALL_STACKS = "wall_stacks"
    """
    Fixtures stacked on a wall, as some layouts spell it.
    """


@dataclass
class RoboCasaLayout:
    """
    A RoboCasa kitchen layout, as RoboCasa's layout files describe it.
    """

    configuration: dict[str, Any]
    """
    The layout, as read from its file.
    """

    @classmethod
    def from_layout_type(cls, layout: LayoutType) -> RoboCasaLayout:
        """
        :param layout: One of RoboCasa's layouts.
        :return: That layout, read from RoboCasa's file of it.
        """
        with open(get_layout_path(layout)) as layout_file:
            return cls(yaml.safe_load(layout_file))

    @property
    def counter_height(self) -> float:
        """
        How high the counters along the main wall stand, in metres.
        """
        return next(
            fixture[LayoutKey.SIZE][2]
            for fixture in self.configuration[LayoutKey.MAIN_GROUP][
                LayoutKey.BOTTOM_ROW
            ]
            if fixture[LayoutKey.TYPE] == FixtureType.COUNTER
        )

    def lowered_fixtures(self, counter_height: float) -> dict[str, dict[str, Any]]:
        """
        :param counter_height: How high the counters are to stand, in metres.
        :return: The updates, by fixture name, that scale every height of the fixtures
            standing on the floor by the same factor, so that the counters stand
            ``counter_height`` high; what hangs on the walls keeps its height.
        """
        factor = counter_height / self.counter_height
        return {
            fixture[LayoutKey.NAME]: update
            for group in self.configuration.values()
            for row_name, row in group.items()
            if row_name not in RowOffTheFloor and isinstance(row, list)
            for fixture in row
            if (update := self._lowered(fixture, factor))
        }

    @staticmethod
    def _lowered(fixture: dict[str, Any], factor: float) -> dict[str, Any]:
        """
        :param fixture: A fixture of the layout.
        :param factor: What to multiply each of its heights with.
        :return: The fixture's size, position and stack height with their heights
            scaled, for those it states.
        """
        update = {}
        for key in (LayoutKey.SIZE, LayoutKey.POSITION):
            value = fixture.get(key)
            if (
                isinstance(value, list)
                and len(value) == 3
                and isinstance(value[2], (int, float))
            ):
                update[key] = [value[0], value[1], value[2] * factor]
        if LayoutKey.STACK_HEIGHT in fixture:
            update[LayoutKey.STACK_HEIGHT] = fixture[LayoutKey.STACK_HEIGHT] * factor
        return update


@dataclass
class HSRPickPlaceTask(ABC):
    """
    One of RoboCasa's pick-and-place tasks, performed by the HSR. Each task is a
    subclass named after the RoboCasa task it performs.
    """

    layout: LayoutType = LayoutType.LAYOUT001
    """
    The RoboCasa kitchen layout.
    """

    style: StyleType = StyleType.STYLE001
    """
    The RoboCasa kitchen style.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task's objects and their places with.
    """

    food_category: str | None = None
    """
    The RoboCasa object category the object to pick up is sampled from; ``None``
    leaves the choice to the task.
    """

    counter_height: float = 0.75
    """
    How high the kitchen's counters stand, in metres, lowered from RoboCasa's 0.92 m
    together with everything else standing on the floor.
    """

    reach: float = 0.6
    """
    How far in front of its base's centre the HSR takes the food, in metres, so that
    both the grasp and the pre-grasp before it lie within the 0.43 m to 0.64 m its
    arm reaches with the hand tilted down at :attr:`grasp_pitch`.
    """

    placing_reach: float = 0.6
    """
    How far in front of its base's centre the HSR sets the food down, in metres.
    """

    release_clearance: float = 0.03
    """
    How far above :attr:`release_height` the HSR lets the food go, in metres.
    """

    tool_frame_tolerance: float = 0.015
    """
    How far, in metres, the tool frame may stop from a goal of the pick up or the
    placing: the base stops within about a centimetre of where it is sent, and the
    arm, swinging in its plane, cannot make up for that sideways.
    """

    floor_margin: float = 0.05
    """
    How far in from the edge of the receptacle's floor the food is set down, in
    metres.
    """

    grasp_pitch: float = math.pi / 4
    """
    How steeply the hand comes down onto the food, in radians below the horizontal.
    """

    collision_avoidance: bool = True
    """
    Whether Giskard keeps the HSR clear of the kitchen and of itself.
    """

    headless: bool = False
    """
    Whether the simulation runs without MuJoCo's own viewer.
    """

    task: RoboCasaTask = field(init=False)
    """
    The task as RoboCasa composed it, built by :meth:`build_world`.
    """

    world: World = field(init=False)
    """
    The kitchen with the task's objects and the HSR, built by :meth:`build_world`.
    """

    @classproperty
    def task_name(cls) -> str:
        """
        The name of the RoboCasa task, which the subclass carries.
        """
        return cls.__name__

    def build_world(self) -> World:
        """
        Compose the RoboCasa task scene in the lowered kitchen and put the HSR in front
        of the food.

        :return: The world.
        """
        self.task = RoboCasaDatasetLoader().load_task(
            self.task_name,
            self.layout,
            self.style,
            seed=self.seed,
            task_parameters=(
                {} if self.food_category is None else {"obj_groups": self.food_category}
            ),
            fixture_updates=RoboCasaLayout.from_layout_type(
                self.layout
            ).lowered_fixtures(self.counter_height),
        )
        self.world = self.task.world
        robot = RobotSpecification(HSRB).spawn(self.world)
        robot.mobile_base.full_body_controlled = False
        with self.world.modify_world():
            self.world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=robot.bodies_with_collision,
                    body_group_b=[
                        self.world.get_body_by_name(floor)
                        for floor in (KitchenBody.FLOOR, KitchenBody.FLOOR_BACKING)
                    ],
                )
            )
            self.world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=[
                        self.world.get_body_in_branch_by_name(robot.root, link)
                        for link in WristLink
                    ],
                    body_group_b=[
                        body
                        for root in [*self.task.manipulated_objects, self.receptacle]
                        for body in self.colliding_bodies_below(root)
                    ],
                )
            )
        self.robot.root.parent_connection.origin = self.standing_pose(
            self.extent(self.food.root).center.to_np()[:3], self.reach
        ).homogeneous_matrix
        return self.world

    def colliding_bodies_below(self, root: Body) -> list[Body]:
        """
        :return: The bodies of the object rooted in ``root`` that carry its collision
            geometry: the root itself, or the bodies below it RoboCasa puts it in.
        """
        return [
            body
            for body in [root]
            + list(
                self.world.compute_descendent_child_kinematic_structure_entities(root)
            )
            if isinstance(body, Body) and body.has_collision()
        ]

    @property
    def robot(self) -> HSRB:
        """
        The HSR.
        """
        return self.world.get_semantic_annotations_by_type(HSRB)[0]

    @property
    def food(self) -> HasGraspCandidates:
        """
        The annotation of the food.
        """
        food = self.world.get_body_by_name(TaskObject.FOOD)
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(
                HasGraspCandidates
            )
            if annotation.root is food
        ][0]

    def extent(self, body: Body) -> VolumetricBoundingBox:
        """
        :return: The axis-aligned box the collision geometry of ``body`` occupies in the
            kitchen's frame.
        """
        return body.collision.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_box()

    @property
    def receptacle(self) -> Body:
        """
        The root of what the object is to be set into: the task's container.
        """
        return self.world.get_body_by_name(TaskObject.CONTAINER)

    @property
    def receptacle_floor(self) -> VolumetricBoundingBox:
        """
        The floor of the receptacle, in the kitchen's frame: the widest of the convex
        pieces RoboCasa builds its collision geometry from.
        """
        pieces = [
            piece
            for body in self.colliding_bodies_below(self.receptacle)
            for piece in body.collision.as_bounding_box_collection_in_frame(
                self.world.root
            ).bounding_boxes
        ]
        return max(pieces, key=lambda piece: piece.width * piece.depth)

    def standing_pose(self, target: np.ndarray, reach: float) -> Pose:
        """
        :param target: Where in the kitchen the hand has to reach.
        :param reach: How far in front of the base's centre ``target`` is to be.
        :return: Where the HSR stands to reach it: facing the counter, ``reach`` back
            from ``target``, with its hand rather than its base in line with it.
        """
        return Pose.from_xyz_rpy(
            x=float(target[0]) + hand_offset(self.world),
            y=float(target[1]) - reach,
            yaw=math.pi / 2,
            reference_frame=self.world.root,
        )

    def grasp(self) -> GraspCandidate:
        """
        :return: The grasp that takes the food at its centre, approaching it from the
            HSR's side of the counter and from above at :attr:`grasp_pitch`, the
            fingers closing along the counter.
        """
        food = self.food.root
        world_T_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.extent(food).center.to_np()[:3],
            pitch=self.grasp_pitch,
            yaw=math.pi / 2,
            reference_frame=self.world.root,
        )
        food_T_grasp = self.world.transform(world_T_grasp.pose, food)
        return GraspCandidate(self.food, food_T_grasp)

    def pre_grasp_joint_positions(self, pre_grasp: Pose) -> dict[HSRBJoint, float]:
        """
        :param pre_grasp: Where the tool frame starts its approach onto the food.
        :return: The arm's joint positions that put the tool frame there, once the HSR
            stands where it does to reach the food.
        """
        world_T_base = self.standing_pose(
            self.extent(self.food.root).center.to_np()[:3], self.reach
        ).to_np()
        world_T_pre_grasp = self.world.transform(pre_grasp, self.world.root).to_np()
        return HSRBArmPlane().joint_positions_for(
            np.linalg.inv(world_T_base) @ world_T_pre_grasp
        )

    @property
    def release_point(self) -> np.ndarray:
        """
        Where the centre of the food is to be let go: its underside
        :attr:`release_clearance` above :attr:`release_height`, over the part of the
        receptacle's floor nearest the HSR but at least :attr:`floor_margin` in from
        its edge.
        """
        floor = self.receptacle_floor
        food_extent = self.extent(self.food.root)
        floor_center = floor.center.to_np()[:3]
        return np.array(
            [
                floor_center[0],
                min(
                    floor_center[1],
                    floor.min_y + food_extent.width / 2 + self.floor_margin,
                ),
                self.release_height + food_extent.height / 2 + self.release_clearance,
            ]
        )

    def goal(self) -> Pose:
        """
        :return: Where the food is to be put: its centre at :attr:`release_point`,
            turned as it is now.
        """
        food = self.food.root
        world_T_food = food.global_pose.to_np()
        world_T_food[:3, 3] += self.release_point - self.extent(food).center.to_np()[:3]
        return HomogeneousTransformationMatrix(
            world_T_food, reference_frame=self.world.root
        ).pose

    @property
    def release_height(self) -> float:
        """
        The height above which the food is let go: the top of the receptacle's floor.
        """
        return self.receptacle_floor.max_z

    def build_plan(self, context: Context) -> Plan:
        """
        Stand in line with the food, raise the arm and move it to where the pick up
        starts its approach, pick the food up, drive to the receptacle holding it, and
        put it in. The arm is raised first, so that the hand leaves its park pose above
        the counter rather than in front of the cabinets below it.

        :param context: The context the actions are built in.
        :return: The plan.
        """
        robot = context.robot
        arm = robot.all_arms[0]
        goal = self.goal()
        grasp = self.grasp()
        pick_up = PickUpAction(grasp, arm, position_threshold=self.tool_frame_tolerance)
        pre_grasp = pick_up.grasp_pose_sequence(
            grasp.world_T_grasp, arm.end_effector, grasp
        ).pre_grasp
        pre_grasp_joints = self.pre_grasp_joint_positions(pre_grasp)
        return sequential(
            [
                ParkArmsAction(robot.all_arms),
                NavigateAction(
                    self.standing_pose(
                        self.extent(self.food.root).center.to_np()[:3], self.reach
                    )
                ),
                MoveJointsMotion(
                    names=[HSRBJoint.ARM_LIFT],
                    positions=[pre_grasp_joints[HSRBJoint.ARM_LIFT]],
                ),
                MoveJointsMotion(
                    names=list(pre_grasp_joints),
                    positions=list(pre_grasp_joints.values()),
                ),
                pick_up,
                NavigateAction(
                    self.standing_pose(self.release_point, self.placing_reach)
                ),
                PlaceAction(
                    self.food, goal, position_threshold=self.tool_frame_tolerance
                ),
                ParkArmsAction(robot.all_arms),
            ],
            context=context,
        ).plan

    def run(self) -> None:
        """
        Let the scene settle, then build the plan from where the food came to rest and
        carry it out against MuJoCo physics, the food held by friction alone. The
        food's contacts resolve the twisting and rolling friction RoboCasa gives it, so
        it neither turns in the hand nor rolls on without end.
        """
        context = Context(world=self.world, robot=self.robot, evaluate_conditions=False)
        for shape in self.food.root.collision.shapes:
            shape.get_simulator_property_of_type(MujocoGeom).contact_dimensionality = (
                ContactDimensionality.SLIDING_TWISTING_AND_ROLLING
            )
        simulation = MujocoSim(
            world=self.world,
            headless=self.headless,
            step_size=STEP_SIZE,
            cone=FRICTION_CONE,
            impratio=IMPEDANCE_RATIO,
            noslip_iterations=NO_SLIP_ITERATIONS,
        )
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(SETTLING_DURATION)
            plan = self.build_plan(context)
            with PhysicallySimulatedRobot(
                simulation=simulation, collision_avoidance=self.collision_avoidance
            ):
                plan.perform()
            simulation.step_simulation(SETTLING_DURATION)
        finally:
            simulation.stop_simulation()


# %% the tasks


@dataclass
class PickPlaceCounterToStove(HSRPickPlaceTask):
    """
    Pick the food from the plate on the counter and place it in the pan on the stove.
    """

    seed: int = 2
    """
    The seed RoboCasa samples the task with: the plate on the counter left of the
    stove, near its edge.
    """

    food_category: str | None = "cheese"
    """
    A block of cheese, which lies still on its plate.
    """


@dataclass
class PickPlaceStoveToCounter(HSRPickPlaceTask):
    """
    Pick the food from the pan on the stove and place it on the plate or in the bowl
    on the counter.
    """

    seed: int = 12
    """
    The seed RoboCasa samples the task with: the pan on the front left burner.
    """

    food_category: str | None = "lemon"
    """
    A lemon, small enough for the HSR's hand.
    """


@dataclass
class PickPlaceCounterToSink(HSRPickPlaceTask):
    """
    Pick the object from the counter and place it in the sink.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task with: the object near the counter's edge right
    of the sink.
    """

    food_category: str | None = "canned_food"
    """
    A can, standing upright.
    """

    @property
    def receptacle(self) -> Body:
        return self.world.get_semantic_annotations_by_type(Sink)[0].root

    @property
    def release_height(self) -> float:
        """
        The sink's rim, level with the counter: the object is dropped into the sink
        from just above it.
        """
        return self.counter_height


@dataclass
class PickPlaceSinkToCounter(HSRPickPlaceTask):
    """
    Pick the food from the sink and place it on the plate on the counter.
    """

    seed: int = 11
    """
    The seed RoboCasa samples the task with.
    """

    food_category: str | None = "yogurt"
    """
    A pot of yogurt.
    """

    grasp_pitch: float = math.pi / 3
    """
    Steeper than on the counter, so that the wrist passes above the sink's front wall
    while the hand reaches down to the food below the rim.
    """

    placing_reach: float = 0.52
    """
    Closer than when taking the food from the sink: held at :attr:`grasp_pitch`, the
    food has to be lifted above the plate, and the arm reaches that high only up to
    0.54 m in front of the base.
    """

    tool_frame_tolerance: float = 0.02
    """
    Reaching down into the sink, the arm comes within its collision buffer of the
    counter's edge and the sink's rim 1.6 cm before the grasp, with the food already
    between the fingers.
    """


if __name__ == "__main__":
    experiment = PickPlaceCounterToStove()
    experiment.build_world()
    experiment.run()
