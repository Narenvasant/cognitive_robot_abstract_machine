"""
Run the demo::

    python -m experiments.tiago_milk_pick_place [--kinematic] [--headless]
                                               [--pyramidal-cone]

Tiago takes the milk carton from one spot on the apartment's dining table, drives a
short way along it, and sets the carton down. The hand holds the carton by contact
friction: nothing attaches it to the robot.

``--kinematic`` is a dry run without physics or a viewer, which checks that the robot
reaches both spots in seconds.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass

print = functools.partial(print, flush=True)

from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from experiments.tiago_milk_pick_place.physics import (
    PhysicalMujocoRun,
    PhysicsSettings,
)
from experiments.tiago_milk_pick_place.pick_place import MilkPickPlaceDemonstration

# %% what one run was asked for


@dataclass
class DemoArguments:
    """
    What one run of the demo was asked for.
    """

    kinematic: bool
    """
    Whether to skip the physics and only check that both spots are reachable.
    """

    headless: bool
    """
    Whether to run the physics without a window.
    """

    pyramidal_cone: bool
    """
    Whether to use MuJoCo's default pyramidal friction cone instead of the elliptic one
    it recommends for grasping.
    """

    @classmethod
    def from_command_line(cls) -> DemoArguments:
        """
        :return: What the command line asked for.
        """
        parser = argparse.ArgumentParser(
            prog="python -m experiments.tiago_milk_pick_place",
            description=(
                "Tiago moves a milk carton along the apartment's dining table, holding "
                "it by friction in MuJoCo."
            ),
        )
        parser.add_argument(
            "--kinematic",
            action="store_true",
            help="dry run without physics or a viewer",
        )
        parser.add_argument(
            "--headless", action="store_true", help="run the physics without a window"
        )
        parser.add_argument(
            "--pyramidal-cone",
            action="store_true",
            help="use MuJoCo's default friction cone instead of the elliptic one",
        )
        parsed = parser.parse_args()
        return cls(
            kinematic=parsed.kinematic,
            headless=parsed.headless,
            pyramidal_cone=parsed.pyramidal_cone,
        )


# %% running it


def report_the_scene(demonstration: MilkPickPlaceDemonstration) -> None:
    """
    Print what the scene came out as, which is the first thing to read when a reach or a
    grasp goes wrong.

    :param demonstration: The demonstration whose scene was populated.
    """
    carton = demonstration.carton
    print(
        f"table top {demonstration.table_top:.4f} m, carton {carton.width:.3f} m wide "
        f"and {carton.height:.3f} m tall, grasped "
        f"{carton.grasp_height_above_bottom:.3f} m above its bottom"
    )


def run_kinematically() -> None:
    """
    Perform the plan in a kinematically simulated world, which checks that the robot
    reaches both spots before any physics is involved.
    """
    demonstration = MilkPickPlaceDemonstration(
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    report_the_scene(demonstration)
    context = demonstration.build_context(world)
    plan = demonstration.build_plan(context)
    with ExecutionEnvironment(ExecutionType.KINEMATICALLY_SIMULATED):
        plan.perform()
    carton = demonstration.carton.annotation.root.global_transform.to_np()
    print(
        f"the carton ended at ({carton[0, 3]:.3f}, {carton[1, 3]:.3f}, "
        f"{carton[2, 3]:.3f}), asked for "
        f"({demonstration.layout.place_x:.3f}, {demonstration.layout.place_y:.3f})"
    )


def run_physically(arguments: DemoArguments) -> None:
    """
    Perform the plan against MuJoCo physics, with the viewer open.

    :param arguments: What the run was asked for.
    """
    demonstration = MilkPickPlaceDemonstration()
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    report_the_scene(demonstration)
    robot = world.get_semantic_annotations_by_type(demonstration.used_robot)[0]
    arm = demonstration.choose_arm(robot)
    grasp = demonstration.choose_grasp(arm)
    print(
        f"the hand closes on {grasp.width_along_closing_axis * 100:.1f} cm and stands "
        f"{arm.end_effector.fingertips.separation * 100:.1f} cm wide"
    )
    settings = PhysicsSettings(elliptic_friction_cone=not arguments.pyramidal_cone)
    run = PhysicalMujocoRun(
        world=world,
        robot=robot,
        carton_body_name=demonstration.carton.name,
        settings=settings,
        headless=arguments.headless,
    )
    run.open()
    print("at rest:")
    print(run.report_rest_contacts())
    started = run.carton_position()
    context = demonstration.build_context(world)
    run.perform(demonstration.build_plan(context))
    ended = run.carton_position()
    print(
        f"the carton went from ({started[0]:.3f}, {started[1]:.3f}, {started[2]:.3f}) "
        f"to ({ended[0]:.3f}, {ended[1]:.3f}, {ended[2]:.3f})"
    )
    print(f"it leans {run.carton_tilt_degrees():.1f} degrees from upright")
    print(
        "its parent is still "
        f"{demonstration.carton.annotation.root.parent_connection.parent.name.name}"
    )
    run.keep_alive()
    run.close()


def main() -> None:
    """
    Run the demo as the command line asked for.
    """
    arguments = DemoArguments.from_command_line()
    if arguments.kinematic:
        run_kinematically()
        return
    run_physically(arguments)


if __name__ == "__main__":
    main()
