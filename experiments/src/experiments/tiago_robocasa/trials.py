"""
Running one of Tiago's RoboCasa tasks over freshly composed kitchens.

A task is tried again in a new kitchen until it comes off: what RoboCasa samples decides
where the object lies and what stands around it, and so whether Tiago can stand where
the task needs it at all.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from experiments.scene_recording import SceneRecording
from experiments.tiago_robocasa.robocasa_task import TiagoRoboCasaTask
from typing_extensions import Type

logger = logging.getLogger(__name__)


# %% how an episode is set up


@dataclass
class EpisodeSettings:
    """
    How each episode of a run is set up. What is left as ``None`` is left as the task
    itself has it.
    """

    recording: SceneRecording | None = None
    """
    Where the episode is recorded to, or nothing to run it unrecorded.
    """

    headless: bool = True
    """
    Whether the simulation runs without a viewer window.
    """

    noslip_iterations: int = 10
    """
    How many passes MuJoCo spends removing the slip its soft contacts leave.
    """

    friction_cone: Any = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    How MuJoCo bounds a contact's friction.
    """

    works_with_the_left_arm: bool = False
    """
    Which of Tiago's arms does the work.
    """

    collision_avoidance: bool = True
    """
    Whether the arm is planned around what stands in its way.
    """

    seed: int | None = None
    """
    The seed RoboCasa samples the kitchen with.
    """

    reach: float | None = None
    """
    How far in front of its base Tiago works.
    """

    grasp_pitch: float | None = None
    """
    How far from level the hand comes down onto what it takes hold of, in radians.
    """

    counter_height: float | None = None
    """
    What the kitchen's counters are lowered to, in metres.
    """

    def episode_of(self, task: Type[TiagoRoboCasaTask]) -> TiagoRoboCasaTask:
        """
        :param task: The task to set an episode of up.
        :return: The episode.
        """
        episode = task(
            recording=self.recording,
            headless=self.headless,
            noslip_iterations=self.noslip_iterations,
            friction_cone=self.friction_cone,
            works_with_the_left_arm=self.works_with_the_left_arm,
            collision_avoidance=self.collision_avoidance,
        )
        if self.seed is not None:
            episode.seed = self.seed
        if self.reach is not None:
            episode.reach = self.reach
        if self.grasp_pitch is not None:
            episode.grasp_pitch = self.grasp_pitch
        if self.counter_height is not None:
            episode.counter_height = self.counter_height
        return episode


# %% running the episodes


@dataclass
class Trials:
    """
    One of Tiago's RoboCasa tasks, tried over one kitchen after another.
    """

    task: Type[TiagoRoboCasaTask]
    """
    The task being tried.
    """

    settings: EpisodeSettings
    """
    How each episode is set up.
    """

    attempts: int = 12
    """
    How many kitchens are tried at most.
    """

    tries_every_kitchen: bool = False
    """
    Whether every kitchen is tried, to count how many the task comes off in, rather
    than stopping at the first it does.
    """

    done: int = field(init=False, default=0)
    """
    In how many kitchens the task came off.
    """

    def run(self) -> None:
        """
        Try the task until it comes off, or until every kitchen asked for has been
        tried, reporting what became of each.
        """
        for attempt in range(1, self.attempts + 1):
            episode = self.settings.episode_of(self.task)
            if self.done:
                episode.recording = None
            self.try_in_a_new_kitchen(attempt, episode)
            if self.done and not self.tries_every_kitchen:
                break
        self.report()

    def try_in_a_new_kitchen(self, attempt: int, episode: TiagoRoboCasaTask) -> None:
        """
        Compose a kitchen and carry the task out in it, counting it where it came off.

        A kitchen Tiago cannot work in is this experiment's subject rather than its
        error, so what the attempt raises is reported, along with how far the scene got,
        and the next kitchen is tried.

        :param attempt: Which attempt this is, for the report.
        :param episode: The episode to carry out.
        """
        try:
            episode.build_world()
        except Exception as failure:
            logger.info(
                "episode %d was not composed: %s: %s",
                attempt,
                type(failure).__name__,
                str(failure)[:120],
            )
            return
        logger.info("episode %d: %s", attempt, episode.task.instruction)
        try:
            episode.run()
        except Exception as failure:
            logger.info(
                "episode %d stopped with %s: %s: %s",
                attempt,
                episode.outcome,
                type(failure).__name__,
                str(failure)[:120],
            )
            return
        logger.info("episode %d finished: %s", attempt, episode.outcome)
        if episode.task_is_done:
            self.done += 1

    def report(self) -> None:
        """
        Say how the run went, and where its recording was written.
        """
        if self.tries_every_kitchen:
            logger.info(
                "%s: the task came off in %d of %d kitchens",
                self.task.task_name,
                self.done,
                self.attempts,
            )
        elif self.done == 0:
            logger.error(
                "%s: the task came off in none of the %d kitchens tried",
                self.task.task_name,
                self.attempts,
            )
        if self.settings.recording is not None and self.done:
            logger.info("wrote a video to %s", self.settings.recording.path)


# %% asking for a run from the command line


def parser_of(description: str, tasks: list[str]) -> argparse.ArgumentParser:
    """
    :param description: What the run does.
    :param tasks: The names of the tasks that can be asked for.
    :return: The command line a run is asked for with.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("task", choices=tasks)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="PATH")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--noslip", type=int, default=10)
    parser.add_argument("--reach", type=float, default=None)
    parser.add_argument("--pitch", type=float, default=None, help="in degrees")
    parser.add_argument("--pyramidal", action="store_true")
    parser.add_argument("--left-arm", action="store_true")
    parser.add_argument("--counter-height", type=float, default=None)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--no-avoidance", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="try every kitchen and count the ones the task comes off in, rather than"
        " stopping at the first",
    )
    return parser


def settings_from(arguments: argparse.Namespace) -> EpisodeSettings:
    """
    :param arguments: What the command line was given.
    :return: How the episodes of that run are set up.
    """
    recording = None
    if arguments.record is not None:
        recording = SceneRecording(
            path=Path(arguments.record)
            if arguments.record
            else Path.home() / "Downloads" / f"tiago_{arguments.task.lower()}.mp4"
        )
    return EpisodeSettings(
        recording=recording,
        headless=not arguments.viewer,
        noslip_iterations=arguments.noslip,
        friction_cone=(
            mujoco.mjtCone.mjCONE_PYRAMIDAL
            if arguments.pyramidal
            else mujoco.mjtCone.mjCONE_ELLIPTIC
        ),
        works_with_the_left_arm=arguments.left_arm,
        collision_avoidance=not arguments.no_avoidance,
        seed=arguments.seed,
        reach=arguments.reach,
        grasp_pitch=None if arguments.pitch is None else math.radians(arguments.pitch),
        counter_height=arguments.counter_height,
    )
