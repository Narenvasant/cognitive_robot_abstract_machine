"""
Run RoboCasa's counter to sink task over many sampled kitchens and report how many of
them the robot completes.

RoboCasa samples the kitchen, the object to move and its placement afresh for every
episode, so one episode says little: an episode the robot cannot reach around is as much
a property of the sample as of the robot. A run over many of them is what a result looks
like.

Run with (the ``experiments`` package must be importable)::

    python -m experiments.robocasa_task_execution.trials --robot TIAGO --episodes 10
    python -m experiments.robocasa_task_execution.trials --robot TIAGO --physics
    python -m experiments.robocasa_task_execution.trials --record-first-pick-up
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from experiments.robocasa_task_execution.counter_to_sink import (
    CounterToSinkEpisode,
    CounterToSinkOutcome,
    KitchenRobot,
)
from contextlib import nullcontext
from datetime import timedelta

from coraplex.execution_environment import PhysicallySimulatedRobot
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from experiments.scene_recording import SceneRecorder, SceneRecording

logger = logging.getLogger(__name__)

DEFAULT_RECORDING_PATH = Path.home() / "Downloads" / "robocasa_counter_to_sink.mp4"
"""
Where the recording of the first episode with a successful pick up is written.
"""

# %% how far an episode got


class TaskStage(IntEnum):
    """
    How far through the task an episode got, each stage including the ones before it.
    """

    NOTHING = 0
    """
    The robot found no way to pick the object up at all.
    """

    PICKED_UP = 1
    """
    The object left the counter and is held by the fingers.
    """

    PLACED = 2
    """
    The object lies in the sink and the robot has let go of it.
    """


# %% how one attempt ended


@dataclass
class Attempt:
    """
    One episode the robot was given, and what became of it.
    """

    instruction: str
    """
    What RoboCasa asked for, in its own words.
    """

    reached: TaskStage = TaskStage.NOTHING
    """
    The furthest stage the robot got to.
    """

    outcome: CounterToSinkOutcome | None = None
    """
    RoboCasa's own judgement of the episode, or nothing if the robot never got that far.
    """

    gave_up_with: str | None = None
    """
    What stopped the episode, named and quoted, or nothing if it ran to its end.
    """

    @property
    def succeeded(self) -> bool:
        """
        :return: Whether the object ended up in the sink and was let go of.
        """
        return self.reached is TaskStage.PLACED


# %% a run of attempts


@dataclass
class TaskTrials:
    """
    A run of the counter to sink task over freshly sampled kitchens.
    """

    robot: KitchenRobot
    """
    The robot that attempts every episode.
    """

    layout: LayoutType = LayoutType.LAYOUT001
    """
    The kitchen layout every episode is composed in.
    """

    style: StyleType = StyleType.STYLE001
    """
    The visual style of the kitchen.
    """

    in_physics: bool = False
    """
    Whether the episodes are performed in MuJoCo physics rather than kinematically.
    """

    recording_path: Path | None = None
    """
    Where to record the first episode the robot completes, or nothing to record none.
    """

    settling_time: timedelta = timedelta(seconds=2)
    """
    How long the physics runs on after a plan, for the object to come to rest.
    """

    attempts: list[Attempt] = field(default_factory=list)
    """
    What became of each episode attempted so far, in order.
    """

    def reached(self, stage: TaskStage) -> int:
        """
        :param stage: The stage to count.
        :return: How many episodes got at least that far.
        """
        return sum(attempt.reached >= stage for attempt in self.attempts)

    def run(self, episodes: int) -> None:
        """
        Attempt that many freshly sampled episodes, recording the first one in which the
        robot gets the object off the counter, if a path was given.

        :param episodes: How many episodes to attempt.
        """
        for number in range(1, episodes + 1):
            record = (
                self.recording_path is not None
                and self.reached(TaskStage.PICKED_UP) == 0
            )
            attempt = self._attempt(record=record)
            self.attempts.append(attempt)
            logger.info(
                "episode %d/%d: %s -- reached %s%s",
                number,
                episodes,
                attempt.instruction,
                attempt.reached.name,
                f", stopped by {attempt.gave_up_with}" if attempt.gave_up_with else "",
            )

    def _attempt(self, record: bool) -> Attempt:
        """
        Compose one episode and have the robot attempt it.

        An episode the robot cannot ground a pick up in, or whose motion fails, is a
        failed attempt rather than the end of the run, which is what lets a run report a
        rate at all.

        :param record: Whether to write a video of this episode.
        :return: What became of it.
        """
        episode = CounterToSinkEpisode.compose(
            self.layout, self.style, robot=self.robot
        )
        attempt = Attempt(instruction=episode.task.instruction)
        try:
            self._perform(episode, attempt, record)
        except Exception as failure:
            attempt.gave_up_with = f"{type(failure).__name__}: {failure}"
        return attempt

    def _perform(
        self, episode: CounterToSinkEpisode, attempt: Attempt, record: bool
    ) -> None:
        """
        Perform the episode stage by stage, marking each as it completes, so an attempt
        stopped half way still says what the robot achieved.

        :param episode: The episode to perform.
        :param attempt: The attempt to record progress in.
        :param record: Whether to write a video of it.
        """
        if not self.in_physics:
            with episode.kinematic_environment():
                self._perform_the_stages(episode, attempt)
            episode.let_the_target_object_fall()
            attempt.outcome = episode.outcome()
            return
        simulation = episode.start_physical_simulation()
        try:
            with self._recorder_for(
                episode, simulation, record
            ), PhysicallySimulatedRobot(simulation=simulation):
                self._perform_the_stages(episode, attempt)
                simulation.step_simulation(self.settling_time)
            attempt.outcome = episode.outcome()
        finally:
            simulation.stop_simulation()

    def _recorder_for(
        self, episode: CounterToSinkEpisode, simulation: MujocoSim, record: bool
    ) -> object:
        """
        :param episode: The episode the camera is aimed at.
        :param simulation: The running simulation the frames are taken from.
        :param record: Whether to record at all.
        :return: What writes the video, or a context that writes nothing.
        """
        if not record:
            return nullcontext()
        return SceneRecorder(
            simulation=simulation,
            camera=episode.camera_on_the_task(),
            recording=SceneRecording(path=self.recording_path),
        )

    @staticmethod
    def _perform_the_stages(episode: CounterToSinkEpisode, attempt: Attempt) -> None:
        """
        :param episode: The episode to perform, in an environment already open.
        :param attempt: The attempt to record progress in.
        """
        episode.perform_the_pick_up()
        attempt.reached = TaskStage.PICKED_UP
        episode.perform_the_placing()
        attempt.reached = TaskStage.PLACED


# %% running the trials from the command line


def _parse_arguments() -> argparse.Namespace:
    """
    :return: What the command line asked for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default=KitchenRobot.TIAGO.name,
        choices=[robot.name for robot in KitchenRobot],
        help="The robot that attempts the episodes.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="How many episodes to attempt."
    )
    parser.add_argument(
        "--layout",
        default=LayoutType.LAYOUT001.name,
        choices=[layout.name for layout in LayoutType],
        help="RoboCasa kitchen layout to compose the episodes in.",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Perform the episodes in MuJoCo physics instead of kinematically.",
    )
    parser.add_argument(
        "--record-first-pick-up",
        nargs="?",
        const=str(DEFAULT_RECORDING_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Record the first episode in which the object leaves the counter, "
            f"by default to {DEFAULT_RECORDING_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the trials the command line asked for and report how far the robot got in
    them.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    arguments = _parse_arguments()
    trials = TaskTrials(
        robot=KitchenRobot[arguments.robot],
        layout=LayoutType[arguments.layout],
        in_physics=arguments.physics,
        recording_path=(
            None
            if arguments.record_first_pick_up is None
            else Path(arguments.record_first_pick_up)
        ),
    )
    trials.run(arguments.episodes)
    attempted = len(trials.attempts)
    for stage in (TaskStage.PICKED_UP, TaskStage.PLACED):
        logger.info(
            "%s reached %s in %d of %d episodes",
            arguments.robot,
            stage.name,
            trials.reached(stage),
            attempted,
        )


if __name__ == "__main__":
    main()
