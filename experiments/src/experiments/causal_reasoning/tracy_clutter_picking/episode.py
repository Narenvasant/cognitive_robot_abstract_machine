"""
One attempt to pick the target carton out of a layout's clutter, with MuJoCo standing in
for Tracy and the target held by contact friction alone: a coraplex pick-up performed by
a :class:`~coraplex.plans.executors.PhysicallySimulatedPlanExecutor`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import mujoco
import numpy as np
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor
from coraplex.plans.failures import PlanFailure
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import Pose, RotationMatrix
from PIL import Image
from typing_extensions import Dict, List, Optional

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickOutcome,
    ClutterSceneLayout,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    EpisodePlanningFailedError,
)
from experiments.causal_reasoning.tracy_clutter_picking.scene import MilkClutterWorld
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

logger = logging.getLogger(__name__)


@dataclass
class BodyPositions:
    """
    Where the cartons stand at one moment of an attempt.
    """

    positions: Dict[str, np.ndarray]
    """
    Each carton's centre in the world root frame, keyed by body name.
    """

    @classmethod
    def read(cls, simulation: MujocoSim, names: List[str]) -> BodyPositions:
        """
        :param simulation: The started simulation to read from.
        :param names: The bodies to read.
        :return: Their current centres, copied out of the simulator's live state so the
            reading stays put once the simulation moves on.
        """
        positions = simulation.simulator.get_bodies_positions(names).result
        return cls({name: np.array(position) for name, position in positions.items()})

    def horizontal_displacement(self, other: BodyPositions, name: str) -> float:
        """
        :param other: An earlier reading.
        :param name: The body to compare.
        :return: How far the body moved in the plane since ``other``, in metres.
        """
        delta = self.positions[name] - other.positions[name]
        return float(math.hypot(delta[0], delta[1]))

    def rise(self, other: BodyPositions, name: str) -> float:
        """
        :param other: An earlier reading.
        :param name: The body to compare.
        :return: How far the body rose since ``other``, in metres.
        """
        return float(self.positions[name][2] - other.positions[name][2])


@dataclass
class PickEpisode:
    """
    Runs one attempt on a layout and records what it did to the scene.
    """

    headless: bool = True
    """
    Whether to run without MuJoCo's viewer window.
    """

    screenshot_directory: Optional[Path] = None
    """
    Where to save a screenshot before and after the pick, or ``None`` for none.
    """

    keep_viewer_open: bool = False
    """
    Whether to keep the viewer window open after the attempt until it is closed.
    """

    lift_threshold: float = 0.1
    """
    How far, in metres, the target has to have risen at the end of the attempt to count
    as lifted: well below the hover height a held carton returns to, well above
    anything a carton merely nudged by the fingers reaches.
    """

    settle_time: timedelta = timedelta(seconds=1)
    """
    Simulated time the scene is left to come to rest before and after the pick.
    """

    screenshot_width: int = 640
    """
    Width, in pixels, of a screenshot of the run: the widest MuJoCo renders offscreen
    without the model declaring a bigger framebuffer.
    """

    screenshot_height: int = 480
    """
    Height, in pixels, of a screenshot of the run.
    """

    lift_distance: float = 0.3
    """
    How far, in metres, the hand rises with the carton after closing on it: as high as
    the hover the attempts were recorded with, well clear of the clutter.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a squeezed carton creep between the
    pads.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: at MuJoCo's default of 1 a held carton
    slides out under its own weight despite a firm squeeze.
    """

    no_slip_iterations: int = 10
    """
    Passes spent removing the slip soft contacts leave, so a held carton stays where the
    pads hold it.
    """

    scene: Optional[MilkClutterWorld] = field(init=False, default=None)
    """
    The scene of the attempt currently running.
    """

    target_milk: Optional[Milk] = field(init=False, default=None)
    """
    The annotation of the carton to pick in the attempt currently running.
    """

    simulation: Optional[MujocoSim] = field(init=False, default=None)
    """
    The simulation of the attempt currently running.
    """

    def run(self, layout: ClutterSceneLayout) -> ClutterPickOutcome:
        """
        Build the layout's clutter, pick its target, and measure the result.

        :param layout: The layout to attempt.
        :return: What the attempt did to the scene.
        :raises EpisodePlanningFailedError: If a motion of the pick could not be
            planned; the attempt is then dropped rather than recorded.
        """
        self.scene = MilkClutterWorld(layout)
        self.target_milk = Milk(root=self.scene.target)
        with self.scene.world.modify_world():
            self.scene.world.add_semantic_annotations([self.target_milk])
        names = [milk.name.name for milk in self.scene.milks]
        simulation = MujocoSim(
            world=self.scene.world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )
        self.simulation = simulation
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(self.settle_time)
            before = BodyPositions.read(simulation, names)
            self._screenshot(simulation, "before_pick")
            self._pick(simulation)
            simulation.step_simulation(self.settle_time)
            after = BodyPositions.read(simulation, names)
            self._screenshot(simulation, "after_pick")
            outcome = self._outcome(before, after)
            logger.info(
                "Pick %s: target rose %.3fm.",
                "lifted" if outcome.lifted else "failed",
                outcome.lift_height,
            )
            if self.keep_viewer_open and not self.headless:
                self._wait_for_viewer(simulation)
        finally:
            simulation.stop_simulation()
        return outcome

    def _pick(self, simulation: MujocoSim) -> None:
        """
        Pick the target with the left arm, coming down on it from above with the
        fingers closing along the layout's grasp yaw.

        :param simulation: The running simulation.
        :raises EpisodePlanningFailedError: If any of the attempt's motions failed, so
            the grasp itself was never tested.
        """
        world = self.scene.world
        grasp = GraspCandidate(
            self.target_milk,
            Pose(
                orientation=RotationMatrix.from_rpy(
                    pitch=np.pi / 2, yaw=self.scene.layout.grasp_yaw
                ).quaternion,
                reference_frame=self.scene.target,
            ),
        )
        executor = PhysicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(world, [RobotAccess(self.scene.robot)]),
            simulation=simulation,
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(
            Sequence(
                nodes=[
                    PickUpAction(
                        grasp=grasp,
                        arm=self.scene.pick_arm,
                        retreat_distance=self.lift_distance,
                    )
                ]
            )
        )
        executor.compile(statechart)
        try:
            executor.execute()
        except PlanFailure as error:
            raise EpisodePlanningFailedError(str(error)) from error
        finally:
            executor.close()

    def _outcome(
        self, before: BodyPositions, after: BodyPositions
    ) -> ClutterPickOutcome:
        """
        :param before: Where the cartons stood before the pick.
        :param after: Where they are after it.
        :return: The attempt's outcome.
        """
        lift_height = after.rise(before, self.scene.target.name.name)
        return ClutterPickOutcome(
            lift_height=lift_height,
            lifted=lift_height > self.lift_threshold,
            neighbour_displacements=[
                after.horizontal_displacement(before, neighbour.name.name)
                for neighbour in self.scene.neighbours
            ],
        )

    def _screenshot(self, simulation: MujocoSim, label: str) -> None:
        """
        Save what the scene camera sees, if a screenshot directory was given.

        :param simulation: The running simulation.
        :param label: What the moment is called in the file name.
        """
        if self.screenshot_directory is None:
            return
        image = simulation.simulator.capture_rgb(
            camera_name=self.scene.camera_name,
            height=self.screenshot_height,
            width=self.screenshot_width,
        ).result
        self.screenshot_directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(self.screenshot_directory / f"{label}.png")

    @staticmethod
    def _wait_for_viewer(simulation: MujocoSim) -> None:
        """
        Keep stepping the physics until the viewer window is closed.

        :param simulation: The running simulation.
        """
        while simulation.simulator.renderer.is_running():
            simulation.step_simulation(timedelta(milliseconds=20))
