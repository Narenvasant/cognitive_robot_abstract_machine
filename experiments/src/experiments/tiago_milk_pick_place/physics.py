"""
The run that performs a plan against MuJoCo physics with a viewer open.

Nothing about the robot's drives is set up here: Tiago's MuJoCo description states what
every joint is driven with, and
:class:`~semantic_digital_twin.robots.tiago.TiagoMujoco` declares those servos itself,
so the arm, the hand and the base are all moved by the physics as soon as it is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import mujoco
import numpy as np
from typing_extensions import Dict, List

from coraplex.execution_environment import PhysicallySimulatedRobot
from experiments.tiago_milk_pick_place.exceptions import (
    CartonDidNotSettle,
    ServoOnWrongJoint,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import WheeledDrive
from semantic_digital_twin.world_description.world_entity import PositionServo

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

    settle_time: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    """
    How long the physics runs before the plan starts, so the carton comes to rest.
    """

    maximum_settling_drift: float = 0.005
    """
    How far the carton may move while settling, in metres, before the run stops.
    """

    elliptic_friction_cone: bool = True
    """
    Whether to use MuJoCo's elliptic friction cone, which it recommends for grasping:
    the hand holds the carton by friction alone, and the default pyramidal cone lets a
    held object creep along the pads under its own weight.
    """

    impression_ratio: float = 10.0
    """
    How much more MuJoCo resists sliding than pressing, used only with the elliptic
    cone.
    """


# %% where the run is watched from


@dataclass
class ViewerCamera:
    """
    Where the viewer looks from, so the window opens on the table rather than on the
    middle of the apartment.
    """

    look_at: tuple = (4.6, 3.9, 0.9)
    """
    The point the camera orbits, in metres.
    """

    distance: float = 2.6
    """
    How far the camera stands from that point, in metres.
    """

    azimuth: float = 150.0
    """
    Which way the camera faces, in degrees.
    """

    elevation: float = -20.0
    """
    How far the camera looks down, in degrees.
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
        expected = self._joint_each_servo_should_drive()
        for index in range(model.nu):
            servo_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            driven = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[index, 0]
            )
            declared = expected[servo_name]
            if driven != declared:
                raise ServoOnWrongJoint(servo_name, declared, driven)

    def _joint_each_servo_should_drive(self) -> Dict[str, str]:
        """
        :return: The name of the simulated joint each servo is meant to drive, by servo
            name: a drive's own axis joint for a servo on one of its degrees of freedom,
            and the joint of the same name as the servo for every other.
        """
        drive_joints = {
            degree_of_freedom.id: joint_name
            for connection in self.world.connections
            if isinstance(connection, WheeledDrive)
            for degree_of_freedom, joint_name in (
                connection.single_axis_joint_names.items()
            )
        }
        return {
            actuator.name.name: drive_joints.get(
                degree_of_freedom.id, degree_of_freedom.name.name
            )
            for actuator in self.world.actuators
            if isinstance(actuator, PositionServo)
            for degree_of_freedom in actuator.dofs
        }

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

        The expected answer is the wheels on the floor, and this is the first thing to
        read when the arm jitters.

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
        Stop the simulation.
        """
        if self.simulation is not None:
            self.simulation.stop_simulation()
