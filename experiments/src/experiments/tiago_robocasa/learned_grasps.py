"""
Tiago takes a RoboCasa object by a grasp a PR2 learned on that kind of object.

The grasps come from the policy iteration the PR2 ran in RoboCasa kitchens: a model
fitted over the grasps it tried and whether they lifted the object, asked for grasps
that lift. A grasp says where on the object's surface it takes hold and how the gripper
comes in, and says it about the object rather than about the robot, which is what lets
another robot try it.

Run it with::

    python -m experiments.tiago_robocasa.learned_grasps PickPlaceCounterToSink \
        --grasps grasps_milk.json --category milk --record
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from dataclasses import field as dataclass_field, make_dataclass
from pathlib import Path

from typing_extensions import Type

from experiments.tiago_robocasa.exceptions import NoGraspTheArmReaches
from experiments.tiago_robocasa.pick_place import TASKS, TiagoPickPlaceTask
from experiments.tiago_robocasa.trials import Trials, parser_of, settings_from
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.surface_grasp import SurfaceGrasp

logger = logging.getLogger(__name__)


# %% what the PR2 learned


@dataclass
class LearnedGrasps:
    """
    The grasps a model learned from a PR2's trials drew for one kind of object, each
    drawn from what the model says lifts it.
    """

    annotation_type: str
    """
    The kind of object the model was learned for.
    """

    attempts_the_model_was_learned_from: int
    """
    How many grasps the PR2 tried to learn it.
    """

    verified_lift_rate: float | None
    """
    The share of grasps drawn from it that lifted the object for the PR2.
    """

    grasps: list[SurfaceGrasp]
    """
    The grasps drawn from it.
    """

    @classmethod
    def read(cls, path: Path) -> LearnedGrasps:
        """
        :param path: Where the drawn grasps were written.
        :return: Them.
        """
        written = json.loads(path.read_text())
        return cls(
            annotation_type=written["annotation_type"],
            attempts_the_model_was_learned_from=written[
                "attempts_the_model_was_learned_from"
            ],
            verified_lift_rate=written["verified_lift_rate"],
            grasps=[SurfaceGrasp(**row) for row in written["grasps"]],
        )


# %% taking an object by one of them


@dataclass
class PicksWithLearnedGrasps(TiagoPickPlaceTask):
    """
    A pick-and-place task that takes the object by a learned grasp instead of at its
    middle.
    """

    learned: LearnedGrasps = None
    """
    The grasps to try.
    """

    grasps_to_try: int = 4
    """
    How many of them are tried from each stand: a stand none of a few can be taken from
    is the wrong stand, and every grasp costs the arm a solve.
    """

    chosen_grasp: GraspCandidate | None = field(default=None, init=False)
    """
    The grasp the arm reaches from where Tiago stands.
    """

    def grasp(self) -> GraspCandidate:
        """
        :return: The grasp the stand was chosen for.
        :raises NoGraspTheArmReaches: If no stand was found for any grasp.
        """
        if self.chosen_grasp is None:
            raise NoGraspTheArmReaches(
                annotation_type=self.learned.annotation_type,
                tried=self.grasps_to_try,
            )
        return self.chosen_grasp

    def reaches_the_pre_grasp(self) -> bool:
        """
        Try the first few drawn grasps from where Tiago stands and keep the first one
        the arm reaches.

        :return: Whether the arm reaches any of them.
        """
        for surface_grasp in self.learned.grasps[: self.grasps_to_try]:
            if not surface_grasp.reaches_surface_of(self.food):
                continue
            self.chosen_grasp = surface_grasp.grasp_candidate(self.food)
            if self.can_reach(self.pre_grasp()):
                return True
        self.chosen_grasp = None
        return False


def taking_the_learned_grasps(
    task: Type[TiagoPickPlaceTask], learned: LearnedGrasps
) -> Type[TiagoPickPlaceTask]:
    """
    Which pick-and-place task to run is asked for on the command line, while the grasps
    to take the object by belong to the task, so the one is made to carry the other.

    :param task: The task to run.
    :param learned: The grasps to take the object by.
    :return: That task, taking the object by those grasps.
    """
    return make_dataclass(
        task.__name__,
        [
            (
                "learned",
                LearnedGrasps,
                dataclass_field(default_factory=lambda: learned),
            )
        ],
        bases=(PicksWithLearnedGrasps, task),
    )


# %% running one from the command line


def main() -> None:
    """
    Run the task the command line names, taking the object by a learned grasp.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    parser = parser_of(__doc__, list(TASKS))
    parser.add_argument("--grasps", type=Path, required=True)
    arguments = parser.parse_args()
    learned = LearnedGrasps.read(arguments.grasps)
    logger.info(
        "%d grasps for %s, learned from %d of the PR2's attempts, which lifted the"
        " object for it in %.0f%% of the drawn grasps it tried",
        len(learned.grasps),
        learned.annotation_type,
        learned.attempts_the_model_was_learned_from,
        100 * learned.verified_lift_rate,
    )
    task = taking_the_learned_grasps(TASKS[arguments.task], learned)
    Trials(
        task=task,
        settings=settings_from(arguments),
        attempts=arguments.attempts,
        tries_every_kitchen=arguments.benchmark,
    ).run()


if __name__ == "__main__":
    main()
