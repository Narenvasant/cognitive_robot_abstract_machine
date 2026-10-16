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
    python -m experiments.robocasa_task_execution.trials --record-first-success
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
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
from experiments.scene_recording import SceneRecording

logger = logging.getLogger(__name__)

DEFAULT_RECORDING_PATH = Path.home() / "Downloads" / "robocasa_counter_to_sink.mp4"
"""
Where the recording of the first completed episode is written.
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

    outcome: CounterToSinkOutcome | None = None
    """
    How the episode ended, or nothing if the robot never got that far.
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
        return self.outcome is not None and self.outcome.succeeded


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

    attempts: list[Attempt] = field(default_factory=list)
    """
    What became of each episode attempted so far, in order.
    """

    @property
    def completed(self) -> int:
        """
        :return: How many episodes the robot completed.
        """
        return sum(attempt.succeeded for attempt in self.attempts)

    def run(self, episodes: int) -> None:
        """
        Attempt that many freshly sampled episodes, recording the first completed one if
        a path was given.

        :param episodes: How many episodes to attempt.
        """
        for number in range(1, episodes + 1):
            attempt = self._attempt(
                record=self.recording_path is not None and self.completed == 0
            )
            self.attempts.append(attempt)
            logger.info(
                "episode %d/%d: %s -- %s",
                number,
                episodes,
                attempt.instruction,
                (
                    "completed"
                    if attempt.succeeded
                    else (attempt.gave_up_with or "failed")
                ),
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
            attempt.outcome = self._perform(episode, record)
        except Exception as failure:
            attempt.gave_up_with = f"{type(failure).__name__}: {failure}"
        return attempt

    def _perform(
        self, episode: CounterToSinkEpisode, record: bool
    ) -> CounterToSinkOutcome:
        """
        :param episode: The episode to perform.
        :param record: Whether to write a video of it.
        :return: How it ended.
        """
        if not self.in_physics:
            return episode.perform()
        simulation = episode.start_physical_simulation()
        try:
            return episode.perform_in_physics(
                simulation,
                recording=(
                    SceneRecording(path=self.recording_path) if record else None
                ),
            )
        finally:
            simulation.stop_simulation()


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
        "--record-first-success",
        nargs="?",
        const=str(DEFAULT_RECORDING_PATH),
        default=None,
        metavar="PATH",
        help=f"Record the first completed episode, by default to {DEFAULT_RECORDING_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the trials the command line asked for and report how many episodes were
    completed.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    arguments = _parse_arguments()
    trials = TaskTrials(
        robot=KitchenRobot[arguments.robot],
        layout=LayoutType[arguments.layout],
        in_physics=arguments.physics,
        recording_path=(
            None
            if arguments.record_first_success is None
            else Path(arguments.record_first_success)
        ),
    )
    trials.run(arguments.episodes)
    logger.info(
        "%s completed %d of %d episodes",
        arguments.robot,
        trials.completed,
        len(trials.attempts),
    )


if __name__ == "__main__":
    main()
