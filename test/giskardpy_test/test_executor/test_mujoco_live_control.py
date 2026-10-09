"""
Giskard's own control loop driving a physically simulated Tracy live: each control
cycle's command becomes the servos' set point through the world state, and the physics
steps in lockstep between cycles.

Skipped where Tracy's description is not installed.
"""

from __future__ import annotations

from datetime import timedelta

import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from giskardpy.executor import Executor, SteppedMotion, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.exceptions import MotionDidNotEndError
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = [
    pytest.mark.skipif(
        not tracy_installed(), reason="iai_tracy_description is not installed"
    ),
    pytest.mark.skipif(
        not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
    ),
]


@pytest.fixture
def parked_tracy() -> Tracy:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    robot = RobotSpecification(Tracy).spawn(world)
    for arm in robot.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    return robot


def test_the_simulated_arm_reaches_the_pose_giskard_commands_live(parked_tracy):
    """
    A goal ticked by Giskard against the world is reached by the arm in the physics, not
    merely in the world's own belief: every cycle's command is handed to the servos as
    their set point and the physics steps in between.
    """
    control_frequency = 50
    tick_limit = 500
    tracking_tolerance = 0.02
    world = parked_tracy._world
    tool_frame = parked_tracy.left_arm.end_effector.tool_frame
    start = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]
    goal_point = Point3(start[0], start[1], start[2] + 0.15, reference_frame=world.root)
    reach = CartesianPosition(
        name="reach", root_link=world.root, tip_link=tool_frame, goal_point=goal_point
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([reach, EndMotion.when_true(reach)])
    controller_config = QPControllerConfig(target_frequency=control_frequency)

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        executor = Executor(
            context=MotionStatechartContext(
                world=world, qp_controller_config=controller_config
            ),
            pacer=SteppedSimulationPacer(simulation),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(timeout=tick_limit)
        simulation.step_simulation(timedelta(seconds=1))
        simulated = numpy.array(
            simulation.simulator.get_body_position(
                body_name=tool_frame.name.name
            ).result
        )
    finally:
        simulation.stop_simulation()

    assert motion_statechart.is_end_motion()
    assert numpy.linalg.norm(simulated - goal_point.to_np()[:3]) <= tracking_tolerance


# %% motions run in lockstep with the physics


@pytest.fixture
def stepped_tracy(parked_tracy) -> SteppedMotion:
    """
    A parked Tracy in a started simulation, with the motions run against it.
    """
    simulation = MujocoSim(world=parked_tracy._world, headless=True)
    simulation.start_stepped_simulation()
    yield SteppedMotion(simulation=simulation)
    simulation.stop_simulation()


def test_a_motion_run_in_lockstep_is_reached_in_the_physics(
    stepped_tracy, parked_tracy
):
    """
    The arm ends up where it was sent in the physics, not merely in the world's own
    belief about where it is.
    """
    world = stepped_tracy.world
    tool_frame = parked_tracy.left_arm.end_effector.tool_frame
    start = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]
    goal_point = Point3(start[0], start[1], start[2] + 0.15, reference_frame=world.root)

    stepped_tracy.run(
        CartesianPosition(
            name="reach",
            root_link=world.root,
            tip_link=tool_frame,
            goal_point=goal_point,
        ),
        avoid_collisions=False,
    )

    simulated = numpy.array(
        stepped_tracy.simulation.simulator.get_body_position(
            body_name=tool_frame.name.name
        ).result
    )
    assert numpy.linalg.norm(simulated - goal_point.to_np()[:3].ravel()) <= 0.02


def test_the_arms_joints_settle_on_what_the_motion_commanded(
    stepped_tracy, parked_tracy
):
    """
    A motion ends as soon as the controller is happy, which is before the servos have
    caught up, so the joints are waited for separately.
    """
    world = stepped_tracy.world
    arm = parked_tracy.left_arm
    tool_frame = arm.end_effector.tool_frame
    start = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]

    stepped_tracy.run(
        CartesianPosition(
            name="reach",
            root_link=world.root,
            tip_link=tool_frame,
            goal_point=Point3(
                start[0], start[1], start[2] + 0.1, reference_frame=world.root
            ),
        ),
        avoid_collisions=False,
    )

    assert stepped_tracy.settled(
        [
            connection.raw_dof.name.name
            for connection in arm.active_connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
    )


def test_a_motion_that_cannot_end_says_so(stepped_tracy, parked_tracy):
    """
    A goal the arm cannot reach is not left ticking forever.
    """
    world = stepped_tracy.world
    stepped_tracy.control_cycle_limit = 5

    with pytest.raises(MotionDidNotEndError):
        stepped_tracy.run(
            CartesianPosition(
                name="unreachable",
                root_link=world.root,
                tip_link=parked_tracy.left_arm.end_effector.tool_frame,
                goal_point=Point3(10.0, 10.0, 10.0, reference_frame=world.root),
            ),
            avoid_collisions=False,
        )


def test_holding_advances_the_physics_without_a_new_command(stepped_tracy):
    before = stepped_tracy.simulation.simulator.current_simulation_time

    stepped_tracy.hold(timedelta(seconds=0.5))

    after = stepped_tracy.simulation.simulator.current_simulation_time
    assert after - before == pytest.approx(0.5, abs=0.05)


def test_waiting_for_no_joints_is_already_settled(stepped_tracy):
    assert stepped_tracy.settled([])


# %% closing a gripper onto something


def test_closing_drives_the_pads_to_the_width_it_was_given(stepped_tracy, parked_tracy):
    """
    The pads end up as far apart as the body is wide, less the squeeze margin on each
    side.
    """
    gripper = parked_tracy.left_arm.end_effector
    nothing_there = Body(name=PrefixedName("nothing_there"))

    stepped_tracy.close_gripper_around(
        parked_tracy.left_arm, nothing_there, width=0.04, squeeze_margin=0.001
    )

    assert gripper.pad_separation == pytest.approx(0.038, abs=0.002)


def test_closing_on_nothing_reports_no_grip(stepped_tracy, parked_tracy):
    """
    Both pads have to end up touching the body for the grip to count, and closing on
    thin air touches nothing.
    """
    assert not stepped_tracy.close_gripper_around(
        parked_tracy.left_arm, Body(name=PrefixedName("nothing_there")), width=0.04
    )


def test_a_wider_body_sends_the_fingertip_frames_further_apart(parked_tracy):
    gripper = parked_tracy.left_arm.end_effector

    assert gripper.fingertip_distance_for(0.06) - gripper.fingertip_distance_for(
        0.03
    ) == pytest.approx(0.03)
