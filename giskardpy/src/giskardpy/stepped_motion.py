"""
Running motions against a physically simulated world, in lockstep with its physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Optional, TYPE_CHECKING

from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import MotionDidNotEndError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.qp.qp_controller_config import QPControllerConfig

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoSim
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SteppedMotion:
    """
    One motion after another run against a physically simulated world, in lockstep with
    its physics.

    Every control cycle's command lands in the world state, the simulation's servos take
    it as their set point, and the physics advances one cycle before the next command is
    worked out.
    """

    simulation: MujocoSim
    """
    The simulation the motions run in; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`
    already.
    """

    target_frequency: int = 50
    """
    Control cycles per simulated second; the physics advances one control period between
    cycles.
    """

    control_cycle_limit: int = 2000
    """
    Control cycles one motion is given before it counts as never having ended.
    """

    settled_threshold: float = 0.01
    """
    How close a joint's simulated position has to come to its set point, in radians or
    metres, to count as having arrived.
    """

    settling_timeout: timedelta = timedelta(seconds=10)
    """
    Simulated time :meth:`settled` waits before giving up on a joint arriving.
    """

    @property
    def world(self) -> World:
        """
        The world the motions are run in, as the simulation holds it.
        """
        return self.simulation.world

    @property
    def control_period(self) -> timedelta:
        """
        Simulated time between two control cycles.
        """
        return timedelta(seconds=1 / self.target_frequency)

    def run(self, task: Task, avoid_collisions: bool = True) -> None:
        """
        Run one task against the simulation until its own end condition is met.

        :param task: What the robot is to do.
        :param avoid_collisions: Whether collision avoidance runs alongside; turn it off
            for a goal that is meant to touch something.
        :raises MotionDidNotEndError: If the task never ended within
            :attr:`control_cycle_limit`.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(task)
        if avoid_collisions:
            motion_statechart.add_node(ExternalCollisionAvoidance())
        motion_statechart.add_node(EndMotion.when_true(task))

        executor = Executor(
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=self.target_frequency, verbose=False
                ),
            ),
            pacer=SteppedSimulationPacer(self.simulation),
        )
        executor.compile(motion_statechart)
        try:
            executor.tick_until_end(timeout=self.control_cycle_limit)
        except TimeoutError as never_ended:
            raise MotionDidNotEndError(
                task=task, control_cycles=self.control_cycle_limit
            ) from never_ended

    def settled(
        self, joint_names: list[str], timeout: Optional[timedelta] = None
    ) -> bool:
        """
        Advance the physics until every one of ``joint_names`` has reached the set point
        the last command left it, or the timeout passes.

        :param joint_names: The joints to wait for.
        :param timeout: Simulated time to wait; :attr:`settling_timeout` if not given.
        :return: Whether every joint arrived; trivially so when none were named.
        """
        if not joint_names:
            return True
        timeout = self.settling_timeout if timeout is None else timeout
        set_points = {
            joint_name: self.world.state[
                self.world.get_connection_by_name(joint_name).raw_dof.id
            ].position
            for joint_name in joint_names
        }
        waited = timedelta()
        while waited < timeout:
            self.simulation.step_simulation(self.control_period)
            waited += self.control_period
            if max(self._distances_to(set_points).values()) < self.settled_threshold:
                return True
        return False

    def close_gripper_around(
        self,
        arm: Arm,
        body: Body,
        width: float,
        squeeze_margin: float = 0.001,
        held_for: timedelta = timedelta(milliseconds=500),
    ) -> bool:
        """
        Close an arm's gripper onto a body and hold it there.

        The pads are driven to ``width`` less the squeeze margin, so they press into the
        body rather than closing past it onto each other.

        :param arm: The arm whose gripper closes.
        :param body: The body to close around.
        :param width: How wide the body stands where the pads meet it, in metres.
        :param squeeze_margin: How far past its surface each pad is sent, in metres.
        :param held_for: Simulated time the pads are held there once they arrive, so the
            servos build up their grip.
        :return: Whether both pads ended up touching the body.
        """
        gripper = arm.end_effector
        pads_apart = max(0.0, width - 2 * squeeze_margin)
        thumb_pad, finger_pad = gripper.pads
        self.run(
            CartesianPosition(
                root_link=finger_pad,
                tip_link=thumb_pad,
                goal_point=gripper.thumb_tip_goal(pads_apart),
            ),
            avoid_collisions=False,
        )
        self.hold(held_for)
        return all(
            body.name.name
            in self.simulation.simulator.get_contact_bodies(
                body_name=pad.name.name, including_children=False
            ).result
            for pad in gripper.pads
        )

    def hold(self, duration: timedelta) -> None:
        """
        Advance the physics with every set point left where it is.

        :param duration: Simulated time to hold for.
        """
        self.simulation.step_simulation(duration)

    def _distances_to(self, set_points: dict[str, float]) -> dict[str, float]:
        """
        :param set_points: The position each joint was last commanded to.
        :return: How far each joint stands from it right now.
        """
        return {
            joint_name: abs(
                self.simulation.simulator.get_joint_value(joint_name).result - set_point
            )
            for joint_name, set_point in set_points.items()
        }
