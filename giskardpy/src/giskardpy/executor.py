from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from typing_extensions import Optional

import numpy as np

from giskardpy.data_types.exceptions import NonPositiveRealTimeFactorError
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    MotionDidNotEndError,
    PlotterNotConfiguredError,
    WorldStateArrayReplacedError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.plotters.debug_expression_trajectory_plotter import (
    DebugExpressionTrajectoryPlotter,
)
from giskardpy.qp.exceptions import EmptyProblemException
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoSim
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.world import World


@dataclass
class Pacer(ABC):
    """
    Decides how long a loop waits between two cycles.
    """

    target_frequency: float = field(init=False)
    """
    Frequency of the loop in hertz, set by whoever runs the loop.
    """

    @abstractmethod
    def sleep(self) -> None:
        """
        Wait until the loop may start its next cycle.
        """


@dataclass
class NoPacing(Pacer):
    """
    Lets a loop run as fast as the hardware allows.
    """

    def sleep(self) -> None:
        pass


@dataclass
class ScheduledPacer(Pacer, ABC):
    """
    Holds a loop at a fixed cycle duration by sleeping until the next slot.

    A cycle that overruns its slot is not compensated by a shorter following one; the
    schedule simply skips to the next slot after the current time.
    """

    _next_target_time: float | None = field(default=None, init=False)
    """
    Point in time the next cycle may start at, None until the first sleep.
    """

    @property
    @abstractmethod
    def cycle_duration(self) -> float:
        """
        How many seconds one cycle should take.
        """

    def sleep(self) -> None:
        cycle_duration = self.cycle_duration
        now = time.monotonic()
        if self._next_target_time is None:
            self._next_target_time = now + cycle_duration
        sleep_time = self._next_target_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
            now = self._next_target_time
        while self._next_target_time <= now:
            self._next_target_time += cycle_duration


@dataclass
class RealTimePacer(ScheduledPacer):
    """
    Holds a loop at its target frequency in wall clock time.
    """

    @property
    def cycle_duration(self) -> float:
        return 1 / self.target_frequency


@dataclass
class SimulationPacer(ScheduledPacer):
    """
    Runs a loop at a multiple of its target frequency to speed up or slow down a
    simulation.
    """

    real_time_factor: float = 1.0
    """
    How much faster than real time the loop runs; ``2.0`` is twice as fast.
    """

    def __post_init__(self):
        if self.real_time_factor <= 0:
            raise NonPositiveRealTimeFactorError(self.real_time_factor)

    @property
    def cycle_duration(self) -> float:
        return 1 / (self.target_frequency * self.real_time_factor)


@dataclass
class SteppedSimulationPacer(Pacer):
    """
    Holds a loop by stepping a physically simulated world one cycle forward between two
    ticks, so a controller ticking against the world runs in lockstep with its physics.

    Every tick's command lands in the world state, the simulation's servos take it as
    their set point, and the physics advances one cycle before the next tick reads the
    world back.
    """

    simulation: MujocoSim
    """
    The simulation to step; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`
    already.
    """

    def sleep(self) -> None:
        self.simulation.step_simulation(timedelta(seconds=1 / self.target_frequency))


@dataclass
class Executor:
    """
    Represents the main execution entity that manages motion statecharts, collision
    scenes, and control cycles for the robot's operations.
    """

    context: MotionStatechartContext

    trajectory_plotter: WorldStateTrajectoryPlotter | None = field(default=None)
    """
    The trajectory plotter used to plot the robot's trajectory.
    """

    debug_expression_plotter: DebugExpressionTrajectoryPlotter | None = field(
        default=None
    )
    """
    Records and plots how the debug expressions evolved during the motion.
    """

    pacer: Pacer = field(default_factory=NoPacing)
    """
    Paces the loop that ticks this executor.
    """

    # %% init False
    motion_statechart: MotionStatechart | None = field(init=False, default=None)
    """
    The motion statechart describing the robot's motion logic, set by :meth:`compile`.
    """

    qp_controller: QPController | None = field(default=None, init=False)
    """
    Optional quadratic programming controller used for motion control.
    """

    _compiled_world_state_data: np.ndarray | None = field(default=None, init=False)
    """
    The world state array the motion statechart was compiled against.

    The compiled updaters read it through a memory view, so it must stay the very same
    array for as long as they are in use.
    """

    @property
    def time(self) -> timedelta:
        """
        Simulated time of the control cycles executed since the last compile.
        """
        return self.control_cycles * self.context.qp_controller_config.control_time_step

    def __post_init__(self):
        self.pacer.target_frequency = self.context.qp_controller_config.target_frequency
        self._create_control_cycles_variable()

    def _create_control_cycles_variable(self):
        self.context.control_cycle_variable = FloatVariable("control_cycles")
        self.context.float_variable_data.register_expression(
            self.context.control_cycle_variable
        )

    @property
    def control_cycles(self) -> float:
        return float(
            self.context.float_variable_data.get_value(
                self.context.control_cycle_variable
            )
        )

    @control_cycles.setter
    def control_cycles(self, value):
        self.context.float_variable_data.set_value(
            self.context.control_cycle_variable, value
        )

    def compile(self, motion_statechart: MotionStatechart):
        self.motion_statechart = motion_statechart
        self.control_cycles = 0
        self.motion_statechart.compile(self.context)
        self._compiled_world_state_data = self.context.world.state._data
        self._compile_qp_controller(self.context.qp_controller_config)
        if self.trajectory_plotter is not None:
            self.trajectory_plotter.reset(
                self.context.world.state, self.time.total_seconds()
            )
        if self.debug_expression_plotter is not None:
            self.debug_expression_plotter.reset(
                self.motion_statechart.collect_debug_expressions()
            )
            self.debug_expression_plotter.debug_expression_trajectory.append(
                self.time.total_seconds()
            )
        self.context.collision_manager.update_collision_matrix()
        # do one tick to immediately active nodes whose start condition is constant true.
        self.motion_statechart.tick(self.context)

    def tick(self):
        self._raise_if_world_state_array_was_replaced()
        self.control_cycles += 1
        if self.context.requires_collision_checking:
            self.context.collision_manager.compute_collisions()
        self.motion_statechart.tick(self.context)
        if self.debug_expression_plotter is not None:
            self.debug_expression_plotter.debug_expression_trajectory.append(
                self.time.total_seconds()
            )
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.compute_command(
            world_state=self.context.world.state._data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            float_variables=self.context.float_variable_data.data,
        )
        self.context.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_time_step.total_seconds(),
            self.qp_controller.config.max_derivative,
        )
        if self.trajectory_plotter is not None:
            self.trajectory_plotter.world_state_trajectory.append(
                self.context.world.state, self.time.total_seconds()
            )

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.

        :param timeout: Max number of ticks to perform.
        """
        try:
            for i in range(timeout):
                self.tick()
                self.pacer.sleep()
                if self.motion_statechart.is_end_motion():
                    return
            raise TimeoutError("Timeout reached while waiting for end of motion.")
        finally:
            self.set_velocity_acceleration_jerk_to_zero()
            self.motion_statechart.cleanup_nodes(context=self.context)
            self.context.cleanup()

    def _raise_if_world_state_array_was_replaced(self):
        """
        Ensures the world still holds the state array the motion statechart compiled
        against.

        :raises WorldStateArrayReplacedError: If the world replaced its state array,
            which leaves the compiled updaters reading a detached copy of the state.
        """
        if self._compiled_world_state_data is None:
            return
        if self.context.world.state._data is self._compiled_world_state_data:
            return
        raise WorldStateArrayReplacedError(
            compiled_degrees_of_freedom=self._compiled_world_state_data.shape[1],
            current_degrees_of_freedom=self.context.world.state._data.shape[1],
        )

    def set_velocity_acceleration_jerk_to_zero(self):
        """
        Clear all commanded derivatives of the world state.
        """
        self.context.world.state.velocities[:] = 0
        self.context.world.state.accelerations[:] = 0
        self.context.world.state.jerks[:] = 0

    def _compile_qp_controller(self, controller_config: QPControllerConfig):
        ordered_dofs = sorted(
            self.context.world.active_degrees_of_freedom,
            key=lambda dof: self.context.world.state._index[dof.id],
        )
        constraint_collection = (
            self.motion_statechart.combine_constraint_collections_of_nodes()
        )
        if len(constraint_collection._constraints) == 0:
            self.qp_controller = None
            # to not build controller, if there are no constraints
            return
        self.qp_controller = QPController(
            config=controller_config,
            degrees_of_freedom=ordered_dofs,
            constraint_collection=constraint_collection,
            world_state_symbols=self.context.world.state.get_variables(),
            life_cycle_variables=self.motion_statechart.life_cycle_state.life_cycle_symbols(),
            float_variables=self.context.float_variable_data.variables,
        )
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException()

    def plot_debug_expressions(self, file_name: str = "./debug_expressions.pdf"):
        """
        Plot the recorded debug expressions to the given PDF file.
        """
        if self.debug_expression_plotter is None:
            raise PlotterNotConfiguredError("debug expression plotter")
        self.debug_expression_plotter.plot(file_name)


# %% motions run in lockstep with the physics


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
                goal=task.unique_name, control_cycles=self.control_cycle_limit
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

    def move_joints(
        self, goal_state: JointState, avoid_collisions: bool = True
    ) -> bool:
        """
        Drive joints to the positions a state names, through the degrees of freedom that
        drive them, and wait for them to arrive.

        :param goal_state: Where the joints are to end up.
        :param avoid_collisions: See :meth:`run`.
        :return: Whether every one of them arrived.
        """
        targets = goal_state.degree_of_freedom_targets
        self.run(
            JointPositionList(goal_state=JointState.from_str_dict(targets, self.world)),
            avoid_collisions=avoid_collisions,
        )
        return self.settled(list(targets))

    def park_arms(self, arms: list[Arm], avoid_collisions: bool = True) -> bool:
        """
        Drive arms to the configuration they are parked in.

        :param arms: The arms to park.
        :param avoid_collisions: See :meth:`run`.
        :return: Whether every joint of every arm arrived.
        """
        parked: dict[ActiveConnection1DOF, float] = {}
        for arm in arms:
            state = arm.get_joint_state_by_type(StaticJointState.PARK)
            parked.update(dict(zip(state.connections, state.target_values)))
        return self.move_joints(
            JointState.from_mapping(parked), avoid_collisions=avoid_collisions
        )

    def reach(
        self,
        arm: Arm,
        goal_pose: Pose,
        hold_orientation: bool = True,
        avoid_collisions: bool = True,
    ) -> bool:
        """
        Move an arm's tool frame to a pose and wait for the arm to come to rest there.

        :param arm: The arm to move.
        :param goal_pose: Where its tool frame is to end up.
        :param hold_orientation: Whether the goal's orientation is held as well as its
            position.
        :param avoid_collisions: See :meth:`run`.
        :return: Whether every joint of the arm arrived.
        """
        tool_frame = arm.end_effector.tool_frame
        if hold_orientation:
            task = CartesianPose(
                root_link=self.world.root, tip_link=tool_frame, goal_pose=goal_pose
            )
        else:
            task = CartesianPosition(
                root_link=self.world.root,
                tip_link=tool_frame,
                goal_point=goal_pose.position,
            )
        self.run(task, avoid_collisions=avoid_collisions)
        return self.settled(
            [
                connection.raw_dof.name.name
                for connection in arm.active_connections
                if isinstance(connection, ActiveConnection1DOF)
            ]
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
