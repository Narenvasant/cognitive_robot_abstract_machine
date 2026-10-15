"""
What Tiago needs before MuJoCo is built so that its hand can hold the carton by
friction, and the run that drives the physics with a viewer open.

The stiffnesses are the ones MuJoCo's own model of this robot uses, in
``mujoco_menagerie/pal_tiago_dual``. That model asks for a damping ratio of one, which
MuJoCo turns into a damping from each joint's inertia; the inertias here were read off
the built model's mass matrix, so the same damping can be given to CRAM's servos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

import mujoco
import numpy as np
from typing_extensions import Dict, List, Tuple

from coraplex.execution_environment import PhysicallySimulatedRobot
from experiments.tiago_milk_pick_place.exceptions import (
    CartonDidNotSettle,
    MissingDrive,
    ServoOnWrongJoint,
)
from semantic_digital_twin.adapters.multi_sim import (
    ContactCategories,
    MujocoGeom,
    MujocoSim,
)
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)


# %% the joints a servo drives


@dataclass(frozen=True)
class TiagoServo:
    """
    What one joint is given so that it can carry the carton without sagging or ringing.
    """

    stiffness: float
    """
    How hard the servo pulls the joint towards its set point, in newton metres per
    radian or newtons per metre. Taken from MuJoCo's own model of this robot.
    """

    inertia: float
    """
    What the joint's own motion weighs, read off the built model's mass matrix.

    The damping follows from this and the stiffness, which is how MuJoCo turns the
    model's damping ratio into a damping of its own.
    """

    force_limit: float
    """
    The largest force or torque the servo may exert.
    """

    armature: float
    """
    Inertia added to the joint, which keeps a light link from ringing.
    """

    def gains(self, damping_ratio: float) -> ServoGains:
        """
        :param damping_ratio: How damped the servo should be, where one is critically
            damped.
        :return: What the servo pulls with.
        """
        return ServoGains(
            stiffness=self.stiffness,
            damping=2.0
            * damping_ratio
            * math.sqrt(self.stiffness * (self.inertia + self.armature)),
            torque_limit=self.force_limit,
        )

    @property
    def dynamics(self) -> JointDynamics:
        """
        :return: The joint's own inertia and friction.
        """
        return JointDynamics(armature=self.armature)


class TiagoServoJoint(StrEnum):
    """
    The joints a position servo drives so that the arm can carry the carton.

    The torso and the seven arm joints carry the load, and the gripper's finger joint is
    the grip. The head and the wheels are written kinematically, as the grasp does not
    depend on them.
    """

    TORSO_LIFT = "torso_lift_joint"
    """
    Raises the whole upper body.
    """

    ARM_1 = "arm_{side}_1_joint"
    """
    Turns the arm at the shoulder.
    """

    ARM_2 = "arm_{side}_2_joint"
    """
    Lifts the arm at the shoulder, under the longest lever.
    """

    ARM_3 = "arm_{side}_3_joint"
    """
    Rolls the upper arm.
    """

    ARM_4 = "arm_{side}_4_joint"
    """
    Bends the elbow.
    """

    ARM_5 = "arm_{side}_5_joint"
    """
    Rolls the forearm.
    """

    ARM_6 = "arm_{side}_6_joint"
    """
    Bends the wrist.
    """

    ARM_7 = "arm_{side}_7_joint"
    """
    Rolls the hand.
    """

    GRIPPER_FINGER = "gripper_{side}_finger_joint"
    """
    Closes the fingers, so its force limit is the grip.
    """

    def named_for(self, side: str) -> str:
        """
        :param side: Which arm, ``left`` or ``right``.
        :return: The joint's name in the robot's description.
        """
        return self.value.format(side=side)


# %% the settings one run uses


@dataclass
class PhysicsSettings:
    """
    How the demo's MuJoCo physics is set up.
    """

    step_size: float = 1e-3
    """
    How long one physics step is, in seconds.
    """

    damping_ratio: float = 1.0
    """
    How damped every servo is, where one is critically damped, as the model this
    robot's stiffnesses come from asks for.
    """

    servos: Dict[TiagoServoJoint, TiagoServo] = field(
        default_factory=lambda: {
            TiagoServoJoint.TORSO_LIFT: TiagoServo(50000.0, 178.383, 2000.0, 0.1),
            TiagoServoJoint.ARM_1: TiagoServo(250.0, 25.761, 200.0, 0.05),
            TiagoServoJoint.ARM_2: TiagoServo(1500.0, 21.583, 200.0, 0.05),
            TiagoServoJoint.ARM_3: TiagoServo(600.0, 12.021, 100.0, 0.05),
            TiagoServoJoint.ARM_4: TiagoServo(1000.0, 14.846, 100.0, 0.05),
            TiagoServoJoint.ARM_5: TiagoServo(80.0, 12.008, 40.0, 0.02),
            TiagoServoJoint.ARM_6: TiagoServo(80.0, 12.221, 40.0, 0.02),
            TiagoServoJoint.ARM_7: TiagoServo(80.0, 12.001, 40.0, 0.02),
            TiagoServoJoint.GRIPPER_FINGER: TiagoServo(500.0, 0.0002, 16.0, 0.01),
        }
    )
    """
    What every servoed joint is given.

    The gripper's force limit is above the 5.197 N the real robot's model states: the
    carton weighs what MuJoCo makes of its mesh at water density, and a pad has to press
    hard enough to hold that by friction alone.
    """

    settle_time: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    """
    How long the physics runs before the plan starts, so the carton comes to rest.
    """

    maximum_settling_drift: float = 0.005
    """
    How far the carton may move while settling, in metres, before the run stops.
    """

    elliptic_friction_cone: bool = False
    """
    Whether to use MuJoCo's elliptic friction cone, which is its recommendation for
    grasping.
    """

    impression_ratio: float = 10.0
    """
    How much more MuJoCo resists sliding than pressing, used only with the elliptic
    cone.
    """

    wheel_root_name: str = "base_footprint"
    """
    The body below which a collision shape has its contacts turned off.

    The base's pose is written into the simulation rather than driven, so a wheel
    touching the floor can only make the free wheel spin and jitter.
    """


# %% declaring all that on the robot


@dataclass
class TiagoGraspPhysics:
    """
    Puts on Tiago what its hand needs to hold the carton: a servo on every joint that
    carries it, no weight of its own to hold up, and no wheels fighting the floor.

    Runs once, before MuJoCo is built, and changes only the world.
    """

    settings: PhysicsSettings = field(default_factory=PhysicsSettings)
    """
    What to declare.
    """

    def prepare(self, world: World, robot: AbstractRobot, arm: Arm) -> List[str]:
        """
        Declare everything the grasp needs.

        :param world: The world the robot stands in.
        :param robot: The robot to prepare.
        :param arm: The arm that carries the carton, which decides which side's joints
            are servoed.
        :return: The names of the bodies whose inertia had to be balanced.
        """
        balanced = self.balance_impossible_inertias(world)
        with world.modify_world():
            self._declare_servos(world, robot, self._side_of(arm))
            self._compensate_gravity(robot)
            self._silence_the_wheels(world, robot)
        return balanced

    def balance_impossible_inertias(self, world: World) -> List[str]:
        """
        Give every body whose inertia no rigid body could have one that it could.

        Four links of this robot's description carry such an inertia: the antennas'
        off-diagonal term exceeds the bound its diagonal allows, and the first arm links
        fail ``A + B >= C``. Both are balanced the way MuJoCo's own ``balanceinertia``
        does it, by making the tensor isotropic with the same trace.

        :param world: The world to look through.
        :return: The names of the bodies that were balanced.
        """
        balanced = []
        for body in world.bodies:
            if body.inertial is None:
                continue
            moments = np.sort(np.linalg.eigvalsh(body.inertial.inertia.data))
            if moments[0] >= 0.0 and moments[0] + moments[1] >= moments[2]:
                continue
            body.inertial.inertia.data[:] = np.eye(3) * float(moments.sum() / 3.0)
            balanced.append(body.name.name)
        return balanced

    def _side_of(self, arm: Arm) -> str:
        """
        :param arm: The arm that carries the carton.
        :return: Which side it is, as its joints are named.
        """
        return "left" if "left" in arm.name.name.lower() else "right"

    def _connections_by_name(
        self, robot: AbstractRobot
    ) -> Dict[str, ActiveConnection1DOF]:
        """
        :param robot: The robot to read.
        :return: Every joint of the robot that can be driven, by name.
        """
        return {
            connection.name.name: connection
            for connection in robot.connections
            if isinstance(connection, ActiveConnection1DOF)
        }

    def _declare_servos(self, world: World, robot: AbstractRobot, side: str) -> None:
        """
        Put one position servo on each joint that carries the carton.

        :param world: The world the servos are added to.
        :param robot: The robot whose joints they drive.
        :param side: Which arm carries it.
        """
        connections = self._connections_by_name(robot)
        for joint, servo in self.settings.servos.items():
            connection = connections[joint.named_for(side)]
            connection.dynamics = servo.dynamics
            actuator = PositionServo(
                name=PrefixedName(
                    f"{connection.raw_dof.name.name}_servo",
                    prefix=connection.raw_dof.name.prefix,
                ),
                gains=servo.gains(self.settings.damping_ratio),
            )
            actuator.add_dof(connection.raw_dof)
            world.add_actuator(actuator)

    def _compensate_gravity(self, robot: AbstractRobot) -> None:
        """
        Let the physics carry the robot's own weight, so the servos carry only the
        carton and the contact forces.

        :param robot: The robot to compensate.
        """
        for body in robot.bodies:
            if body.get_simulator_property_of_type(GravityCompensation) is None:
                body.add_simulator_property(GravityCompensation(fraction=1.0))

    def _silence_the_wheels(self, world: World, robot: AbstractRobot) -> None:
        """
        Turn off the contacts of everything at or below the base.

        :param world: The world holding the robot.
        :param robot: The robot whose wheels are silenced.
        """
        root = world.get_body_by_name(self.settings.wheel_root_name)
        below = set(
            world.compute_descendent_child_kinematic_structure_entities(root)
        ) | {root}
        for entity in below:
            if not isinstance(entity, Body) or not entity.has_collision:
                continue
            for shape in entity.collision.shapes:
                geometry = shape.get_simulator_property_of_type(MujocoGeom)
                if geometry is None:
                    shape.add_simulator_property(
                        MujocoGeom(
                            contact_type=ContactCategories.NONE,
                            contact_affinity=ContactCategories.NONE,
                        )
                    )
                else:
                    geometry.contact_type = ContactCategories.NONE
                    geometry.contact_affinity = ContactCategories.NONE


# %% the base in the simulation


@dataclass(eq=False)
class BaseInMujoco(StateChangeCallback):
    """
    Writes the robot's base pose into MuJoCo whenever the world changes.

    The MuJoCo builder leaves a drive out, so without this the base stays where it was
    built and driving would not move the robot in the viewer. The arm, the hand and the
    carton stay fully physical either way.

    ..note:: The base is placed once per control cycle rather than pushed by its wheels,
        so the physics sees no velocity for those steps. A jump across the finger pads
        presses the carton through contact, which is harmless; a jump along them is not
        resisted by friction, which is why the drive is kept slow.
    """

    simulation: object = field(kw_only=True)
    """
    The running simulation whose model carries the base body.
    """

    drive: object = field(kw_only=True)
    """
    The connection the base is moved by, read for the pose to write.
    """

    base_body: Body = field(kw_only=True)
    """
    The body whose pose is written, which is the drive's child.
    """

    def __post_init__(self):
        """
        Register for state changes and write where the base already stands.
        """
        super().__post_init__()
        self._written_pose: np.ndarray | None = None
        self.on_state_change()

    def on_state_change(self, **kwargs) -> None:
        """
        Write the base's pose into MuJoCo if it moved since the last write.

        :param kwargs: What the world passes its callbacks, which this one ignores.
        """
        odom_T_base = self._world.compute_forward_kinematics_np(
            self.drive.parent, self.drive.child
        )
        if self._written_pose is not None and np.allclose(
            odom_T_base, self._written_pose, atol=1e-9
        ):
            return
        self._written_pose = odom_T_base.copy()
        model = self.simulation.simulator._mj_model
        index = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.base_body.name.name
        )
        model.body_pos[index] = odom_T_base[:3, 3]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, odom_T_base[:3, :3].flatten())
        model.body_quat[index] = quaternion

    @classmethod
    def for_robot(
        cls, world: World, robot: AbstractRobot, simulation: object
    ) -> BaseInMujoco:
        """
        Find the robot's drive and follow it.

        :param world: The world the robot stands in.
        :param robot: The robot whose base is followed.
        :param simulation: The running simulation to write into.
        :return: The callback, already having written where the base stands.
        """
        entity = robot.root
        while entity.parent_connection is not None:
            connection = entity.parent_connection
            if isinstance(connection, (DifferentialDrive, OmniDrive)):
                return cls(
                    _world=world,
                    simulation=simulation,
                    drive=connection,
                    base_body=connection.child,
                )
            entity = connection.parent
        raise MissingDrive(robot.name.name)


# %% where the run is watched from


@dataclass
class ViewerCamera:
    """
    Where the viewer looks from, so the run opens on the table rather than on the
    apartment's origin.
    """

    look_at: Tuple[float, float, float] = (4.6, 3.9, 0.9)
    """
    The point the camera orbits, which is between the robot and the carton.
    """

    distance: float = 2.6
    """
    How far the camera sits from that point, in metres.
    """

    azimuth: float = 150.0
    """
    Where it sits around the point, in degrees.
    """

    elevation: float = -20.0
    """
    How far above the point it sits, in degrees.
    """


# %% driving the physics


@dataclass
class PhysicalMujocoRun:
    """
    Builds MuJoCo for a populated world, settles the carton, and performs a plan against
    the physics with the viewer open.
    """

    world: World
    """
    The populated world to simulate.
    """

    robot: AbstractRobot
    """
    The robot in it.
    """

    carton_body_name: str
    """
    The body whose settling is checked and whose contacts are reported.
    """

    settings: PhysicsSettings = field(default_factory=PhysicsSettings)
    """
    How the physics is set up.
    """

    camera: ViewerCamera = field(default_factory=ViewerCamera)
    """
    Where the viewer looks from.
    """

    headless: bool = False
    """
    Whether to run without a window, which is what a test needs.
    """

    simulation: object | None = field(init=False, default=None)
    """
    The simulation, once it is built.
    """

    base_callback: BaseInMujoco | None = field(init=False, default=None)
    """
    What keeps the base's pose in MuJoCo up to date.
    """

    def open(self) -> None:
        """
        Build MuJoCo, start it stepped, and settle the carton.

        :raises ServoOnWrongJoint: If a servo drives a joint it was not declared for.
        :raises CartonDidNotSettle: If the carton is still moving afterwards.
        """
        self.simulation = MujocoSim(
            world=self.world,
            headless=self.headless,
            step_size=self.settings.step_size,
            **self._friction_configuration(),
        )
        self._check_every_servo_drives_its_own_joint()
        self.base_callback = BaseInMujoco.for_robot(
            self.world, self.robot, self.simulation
        )
        self.simulation.start_stepped_simulation()
        self._aim_the_camera()
        self._settle()

    def _friction_configuration(self) -> Dict[str, object]:
        """
        :return: What to pass MuJoCo about its friction cone, which is empty unless the
            elliptic cone was asked for.
        """
        if not self.settings.elliptic_friction_cone:
            return {}
        return {
            "cone": mujoco.mjtCone.mjCONE_ELLIPTIC,
            "impratio": self.settings.impression_ratio,
        }

    def _check_every_servo_drives_its_own_joint(self) -> None:
        """
        The builder resolves a servo to the first connection carrying its degree of
        freedom, which need not be the joint it was declared for.

        :raises ServoOnWrongJoint: If one of them drives another joint.
        """
        model = self.simulation.simulator._mj_model
        for index in range(model.nu):
            servo_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            driven = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[index, 0]
            )
            declared = servo_name.removesuffix("_servo")
            if driven != declared:
                raise ServoOnWrongJoint(servo_name, declared, driven)

    def _aim_the_camera(self) -> None:
        """
        Point the viewer at the table.
        """
        if self.headless:
            return
        renderer = self.simulation.simulator.renderer
        with renderer.lock():
            renderer.cam.lookat[:] = self.camera.look_at
            renderer.cam.distance = self.camera.distance
            renderer.cam.azimuth = self.camera.azimuth
            renderer.cam.elevation = self.camera.elevation

    def carton_position(self) -> np.ndarray:
        """
        Where the carton stands in the simulation.

        The simulator hands back a live view of its own state, so the reading is copied:
        two readings taken apart would otherwise be the same array, and any movement
        between them would measure as none.

        :return: A snapshot of its position.
        """
        return np.array(
            self.simulation.simulator.get_body_position(self.carton_body_name).result,
            dtype=float,
        ).ravel()

    def carton_tilt_degrees(self) -> float:
        """
        :return: How far the carton leans from upright, in degrees.
        """
        upright = (
            self.world.get_body_by_name(self.carton_body_name)
            .global_transform.to_np()[:3, 2]
            .ravel()
        )
        return float(np.degrees(np.arccos(np.clip(upright[2], -1.0, 1.0))))

    def _settle(self) -> None:
        """
        Let the physics bring the carton to rest before the plan starts.

        The drift is measured over a second window: a simulation that has not stepped
        yet holds no positions to compare against, and the first moments are the carton
        coming to rest on the table.

        :raises CartonDidNotSettle: If it is still moving or leaning afterwards.
        """
        self.simulation.step_simulation(self.settings.settle_time)
        before = self.carton_position()
        self.simulation.step_simulation(self.settings.settle_time)
        drift = float(np.linalg.norm(self.carton_position() - before))
        if drift > self.settings.maximum_settling_drift:
            raise CartonDidNotSettle(drift, self.carton_tilt_degrees())

    def report_rest_contacts(self) -> str:
        """
        What every part of the robot touches once the carton has settled.

        The expected answer is nothing, and this is the first thing to read when the arm
        jitters.

        :return: One line per robot body that touches something.
        """
        lines = []
        for body in self.robot.bodies:
            contacts = self.simulation.simulator.get_contact_bodies(
                body.name.name
            ).result
            if contacts:
                lines.append(f"  {body.name.name} touches {sorted(contacts)}")
        return "\n".join(lines) if lines else "  the robot touches nothing"

    def perform(self, plan: object, collision_avoidance: bool = False) -> None:
        """
        Perform a plan against the physics.

        :param plan: The plan to perform.
        :param collision_avoidance: Whether Giskard avoids collisions while doing it.
        """
        with PhysicallySimulatedRobot(
            simulation=self.simulation, collision_avoidance=collision_avoidance
        ):
            plan.perform()

    def keep_alive(self) -> None:
        """
        Keep stepping while the viewer is open, so the scene stays live after the plan.
        """
        if self.headless:
            return
        while self.simulation.simulator.is_running():
            self.simulation.step_simulation(timedelta(milliseconds=10))

    def close(self) -> None:
        """
        Stop following the base and stop the simulation.
        """
        if self.base_callback is not None:
            self.base_callback.stop()
        if self.simulation is not None:
            self.simulation.stop_simulation()
