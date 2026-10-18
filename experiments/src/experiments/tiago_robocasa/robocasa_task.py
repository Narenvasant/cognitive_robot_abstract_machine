"""
What Tiago's RoboCasa tasks share: a RoboCasa kitchen lowered to where Tiago reaches,
the robot standing in front of what it is to take hold of, and the whole thing carried
out against MuJoCo physics with nothing attached to the hand.

The kitchen, the objects and where they stand are RoboCasa's. Tiago replaces RoboCasa's
own robot, and every object it moves is held by the grip of its fingers alone.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import mujoco
import numpy as np
from krrood.ormatic.utils import classproperty

from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.execution_environment import PhysicallySimulatedRobot
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from experiments.hsr_robocasa.robocasa_task import KitchenBody, RoboCasaLayout
from experiments.tiago_robocasa.exceptions import NoClearStandingPose
from experiments.scene_recording import SceneRecorder, SceneRecording, ViewerCamera
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    RoboCasaTask,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.tiago import TiagoMujoco
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import Body

# %% how the physics is set up

STEP_SIZE = 1e-3
"""
How much simulated time one physics step covers, in seconds.
"""

SETTLING_DURATION = timedelta(seconds=1)
"""
How long the scene is left to settle before the robot starts and after it is done.
"""

FRICTION_CONE = mujoco.mjtCone.mjCONE_ELLIPTIC
"""
How MuJoCo bounds a contact's friction: the elliptic cone is the exact one; the default
pyramid only approximates it and lets a squeezed object creep out of the hand.
"""

IMPEDANCE_RATIO = 10.0
"""
How much stiffer MuJoCo makes friction than the push along the contact normal.
"""

NO_SLIP_ITERATIONS = 10
"""
How many passes MuJoCo spends removing the slip its soft contacts leave, so a held
object stays where the fingers hold it.
"""


# %% one of RoboCasa's tasks, performed by Tiago


@dataclass
class TiagoRoboCasaTask(ABC):
    """
    One of RoboCasa's tasks, performed by Tiago. Each task is a subclass named after the
    RoboCasa task it performs.
    """

    layout: Any = None
    """
    The RoboCasa kitchen layout; the first one unless a subclass says otherwise.
    """

    style: Any = None
    """
    The RoboCasa kitchen style.
    """

    seed: int = 0
    """
    The seed RoboCasa samples the task's objects and their places with.
    """

    counter_height: float | None = None
    """
    How high the kitchen's counters are made to stand, in metres, or nothing to leave
    RoboCasa's kitchen at the size it ships.

    Tiago reaches RoboCasa's own counters, so the kitchen is not shrunk for it the way
    it has to be for a robot that cannot.
    """

    reach: float = 0.6
    """
    How far in front of its base's centre Tiago takes hold of what it reaches for, in
    metres.
    """

    grasp_pitch: float = math.pi / 4
    """
    How steeply the hand comes down onto what it takes hold of, in radians below the
    horizontal.
    """

    stand_step: float = 0.05
    """
    How far the stand is moved at a time while looking for a clear one, in metres.
    """

    stand_search: float = 0.4
    """
    How far back of, and along from, the ideal stand a clear one may be looked for, in
    metres.
    """

    tool_frame_tolerance: float = 0.015
    """
    How far, in metres, the tool frame may stop from a goal of the task's motions.
    """

    collision_avoidance: bool = True
    """
    Whether Giskard keeps Tiago clear of the kitchen and of itself.
    """

    solver: Any = mujoco.mjtSolver.mjSOL_CG
    """
    Which constraint solver MuJoCo uses.

    The conjugate-gradient one, rather than MuJoCo's default Newton: Newton factorizes
    a Hessian that this kitchen, with the robot's hand among its fixtures, makes
    rank-deficient, and the simulation stops dead when it does.
    """

    friction_cone: Any = FRICTION_CONE
    """
    How MuJoCo bounds a contact's friction.
    """

    noslip_iterations: int = NO_SLIP_ITERATIONS
    """
    How many passes MuJoCo spends removing contact slip.
    """

    headless: bool = True
    """
    Whether the simulation runs without MuJoCo's own viewer.
    """

    recording: SceneRecording | None = None
    """
    Where to write a video of the run, or nothing to run without recording one.
    """

    task: RoboCasaTask = field(init=False)
    """
    The task as RoboCasa composed it, built by :meth:`build_world`.
    """

    world: World = field(init=False)
    """
    The kitchen with the task's objects and Tiago, built by :meth:`build_world`.
    """

    @classproperty
    def task_name(cls) -> str:
        """
        The name of the RoboCasa task, which the subclass carries.
        """
        return cls.__name__

    @property
    def task_parameters(self) -> dict[str, Any]:
        """
        :return: What RoboCasa is told about the task beyond its name.
        """
        return {}

    @property
    @abstractmethod
    def first_target(self) -> np.ndarray:
        """
        :return: Where in the kitchen the hand first has to reach, which is where Tiago
            is stood.
        """

    @abstractmethod
    def build_plan(self, context: Context) -> Plan:
        """
        :param context: The context to plan in.
        :return: The plan carrying the task out.
        """

    @property
    def robot(self) -> TiagoMujoco:
        """
        :return: Tiago, in the built world.
        """
        return self.world.get_semantic_annotations_by_type(TiagoMujoco)[0]

    works_with_the_left_arm: bool = False
    """
    Which of Tiago's two arms does the work.

    The other one is parked at its side, and in a kitchen that leaves it pointing at
    whatever the robot's other shoulder faces, so the arm that works is the one that
    keeps the idle arm out of the wall.
    """

    @property
    def arm(self):
        """
        :return: The arm Tiago works with.
        """
        torso = self.robot.mobile_base.torso
        return torso.left_arm if self.works_with_the_left_arm else torso.right_arm

    def extent(self, body: Body) -> VolumetricBoundingBox:
        """
        :param body: A body of the kitchen.
        :return: Its bounding box in the kitchen's frame.
        """
        return body.collision.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_box()

    def colliding_bodies_below(self, root: Body) -> list[Body]:
        """
        :param root: A body of the kitchen.
        :return: It and everything under it that has collision geometry.
        """
        return [
            body
            for body in self.world.compute_descendent_child_kinematic_structure_entities(
                root
            )
            + [root]
            if isinstance(body, Body) and body.has_collision()
        ]

    def build_world(self) -> World:
        """
        Compose the RoboCasa task scene in the lowered kitchen and stand Tiago in front
        of :attr:`first_target`.

        :return: The world.
        """
        from robocasa.models.scenes.scene_registry import LayoutType, StyleType

        layout = self.layout if self.layout is not None else LayoutType.LAYOUT001
        style = self.style if self.style is not None else StyleType.STYLE001
        self.task = RoboCasaDatasetLoader().load_task(
            self.task_name,
            layout,
            style,
            seed=self.seed,
            task_parameters=self.task_parameters,
            fixture_updates=self.fixture_updates(layout),
        )
        self.world = self.task.world
        robot = RobotSpecification(TiagoMujoco).spawn(self.world)
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
        self.raise_the_torso()
        self.stand_where_nothing_touches()
        return self.world

    def raise_the_torso(self) -> None:
        """
        Stand the torso at its full height before the simulation is built.

        The torso is set rather than commanded: its joint carries the whole upper body,
        and a servo asked to lift that saturates its force without moving, so the robot
        starts where it needs to be. With the torso down, Tiago's hand does not reach a
        RoboCasa counter at all.
        """
        self.robot.mobile_base.torso.get_joint_state_by_type(TorsoState.HIGH).apply_to(
            self.world
        )

    def stand_where_nothing_touches(self) -> None:
        """
        Put Tiago where it reaches :attr:`first_target`, stepping it back and along the
        counter until it touches nothing.

        Tiago is wider than the robot a RoboCasa kitchen is laid out around and carries
        a second arm at its side, so the stand that puts its working hand in front of
        the target often leaves a wheel in a cabinet or an elbow in a wall.

        :raises NoClearStandingPose: If it touches the kitchen at every stand tried.
        """
        self.world.collision_manager.update_collision_matrix()
        drive = self.robot.root.parent_connection
        for back, along in self.stands_to_try():
            drive.origin = self.standing_pose(
                self.first_target + np.array([along, 0.0, 0.0]), self.reach + back
            ).homogeneous_matrix
            if not self.robot.is_in_collision:
                return
        raise NoClearStandingPose(
            target=self.first_target, tried=len(self.stands_to_try())
        )

    def stands_to_try(self) -> list[tuple[float, float]]:
        """
        :return: How much further back and how far along the counter to try standing,
            nearest first.
        """
        steps = np.arange(0.0, self.stand_search + self.stand_step / 2, self.stand_step)
        slides = [0.0] + [
            side * step for step in steps[1:] for side in (1.0, -1.0)
        ]
        return [(float(back), float(along)) for back in steps for along in slides]

    def fixture_updates(self, layout: Any) -> dict[str, Any]:
        """
        :param layout: The kitchen layout being composed.
        :return: What to change about the kitchen's fixtures, which is nothing unless
            the counters are to stand at a height of their own.
        """
        if self.counter_height is None:
            return {}
        return RoboCasaLayout.from_layout_type(layout).lowered_fixtures(
            self.counter_height
        )

    def standing_pose(
        self, target: np.ndarray, reach: float, heading: float = math.pi / 2
    ) -> Pose:
        """
        :param target: Where in the kitchen the hand has to reach.
        :param reach: How far in front of the base's centre ``target`` is to be.
        :param heading: The direction Tiago faces, as a yaw in the kitchen's frame.
        :return: Where Tiago stands to reach it, with the arm it works with rather than
            its base in line with the target.
        """
        facing = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-math.sin(heading), math.cos(heading)])
        x, y = target[:2] - reach * facing - self.shoulder_offset * left
        return Pose.from_xyz_rpy(
            x=float(x), y=float(y), yaw=heading, reference_frame=self.world.root
        )

    @property
    def shoulder_offset(self) -> float:
        """
        :return: How far the working arm's shoulder sits to the robot's left of its
            base's centre, in metres, which is negative for the right arm.
        """
        base = self.robot.root.global_transform.to_np()
        shoulder = self.arm.bodies[1].global_transform.to_np()
        return float((np.linalg.inv(base) @ shoulder)[1, 3])

    def arm_motions_to(self, tool_frame_goal: Pose) -> list[MoveJointsMotion]:
        """
        Bring the arm to a tool frame goal in joint space.

        Where the arm is to go is worked out in a kinematically simulated world, which
        the goal is reachable in, and the joints it ends up in are what the physical
        run is asked for. A tool frame goal given to the physical run directly stalls:
        the arm is still on its way when the controller is satisfied, and the hand ends
        up short of what it is reaching for.

        :param tool_frame_goal: Where the tool frame is to be.
        :return: The motion bringing the arm there.
        """
        context = Context(
            world=self.world, robot=self.robot, evaluate_conditions=False
        )
        joints = [
            connection
            for connection in self.arm.active_connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
        with self.world.reset_state_context():
            with ExecutionEnvironment(
                ExecutionType.KINEMATICALLY_SIMULATED,
                collision_avoidance=self.collision_avoidance,
            ):
                sequential(
                    [
                        MoveToolCenterPointMotion(
                            tool_frame_goal, self.arm, allow_gripper_collision=True
                        )
                    ],
                    context=context,
                ).plan.perform()
            positions = [float(joint.position) for joint in joints]
        return [
            MoveJointsMotion(
                names=[joint.name.name for joint in joints], positions=positions
            )
        ]

    def prepare_contacts(self) -> None:
        """
        Set up how the task's objects make contact in MuJoCo, before the simulation is
        built. Leaves them as RoboCasa gives them.
        """

    def camera_on(self, target: np.ndarray, distance: float = 2.5) -> ViewerCamera:
        """
        :param target: What the recording looks at.
        :param distance: How far the camera stands from it, in metres.
        :return: A camera orbiting that point.
        """
        return ViewerCamera(
            look_at=(float(target[0]), float(target[1]), float(target[2])),
            distance=distance,
            azimuth=140.0,
            elevation=-25.0,
        )

    def run(self) -> None:
        """
        Let the scene settle, then carry the task's plan out against MuJoCo physics,
        recording it if a recording was asked for.
        """
        context = Context(
            world=self.world,
            robot=self.robot,
            evaluate_conditions=False,
            motion_tolerances=MotionToleranceConfig(
                default_tcp_position_threshold=self.tool_frame_tolerance
            ),
        )
        self.prepare_contacts()
        simulation = MujocoSim(
            world=self.world,
            headless=self.headless,
            step_size=STEP_SIZE,
            cone=self.friction_cone,
            impratio=IMPEDANCE_RATIO,
            noslip_iterations=self.noslip_iterations,
            solver=self.solver,
        )
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(SETTLING_DURATION)
            plan = self.build_plan(context)
            with self._recorder(simulation):
                with PhysicallySimulatedRobot(
                    simulation=simulation,
                    collision_avoidance=self.collision_avoidance,
                ):
                    plan.perform()
                simulation.step_simulation(SETTLING_DURATION)
        finally:
            simulation.stop_simulation()

    def _recorder(self, simulation: MujocoSim):
        """
        :param simulation: The running simulation frames are taken from.
        :return: What writes the video, or a context that writes nothing.
        """
        from contextlib import nullcontext

        if self.recording is None:
            return nullcontext()
        return SceneRecorder(
            simulation=simulation,
            camera=self.camera_on(self.first_target),
            recording=self.recording,
        )
