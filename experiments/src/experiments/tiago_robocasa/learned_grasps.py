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
import math

import numpy as np
from dataclasses import dataclass, field, replace
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

    heading_tolerance: float = 0.35
    """
    How far the hand may be turned from a grasp's own twist and still count as holding
    it, in radians.

    Looser than a grasp of Tiago's own asks for: the model turns the fingers about the
    approach by whatever suited the arm that learned it, and a wrist that has to hold
    that exactly stalls a hand's breadth from the object. Twenty degrees out, two
    fingers still close on the same two faces.
    """

    headings_to_choose_from: int = 4
    """
    How many of the ways the drawn grasps come in are tried as ways for Tiago to face.
    """

    grasps_to_try: int = 4
    """
    How many grasps are tried from each stand.

    Each one costs the arm a solve, of which a stand search does hundreds, so this is
    what a run spends its time on.
    """

    @property
    def headings_to_try(self) -> tuple[float, ...]:
        """
        :return: The ways the drawn grasps come in, which are the ways Tiago faces to
            take them.

        A grasp says which side of the object the hand comes from, so it says where the
        robot has to stand; facing the kitchen's quarter turns instead and hoping a
        grasp fits is what left the arm reaching none of them. The quarter turns follow
        all the same, because the model draws grasps from every side of the object,
        the side the counter is on included, and Tiago cannot stand there at all.
        """
        headings = {}
        for candidate, _ in self.candidates_on_the_object():
            approach = candidate.world_T_grasp.to_np()[:2, 0]
            heading = float(np.arctan2(approach[1], approach[0]))
            headings.setdefault(round(heading, 1), heading)
        return (
            tuple(headings.values())[: self.headings_to_choose_from]
            + self.quarter_turns
        )

    grasps_to_choose_from: int = 60
    """
    How many of the drawn grasps the ones to try are chosen from.
    """

    chosen_grasp: GraspCandidate | None = field(default=None, init=False)
    """
    The grasp the arm reaches from where Tiago stands.
    """

    candidates: list[tuple[GraspCandidate, SurfaceGrasp]] | None = field(
        default=None, init=False
    )
    """
    The drawn grasps as grasps on the object, once they have been worked out, each with
    the grasp it was drawn as.
    """

    judged_by_the_grasp: bool = False
    """
    Whether an episode counts once a learned grasp has taken the object off its
    support, which is what the model is asked for, rather than once the task it serves
    is carried out.
    """

    @property
    def task_is_done(self) -> bool:
        """
        :return: Whether the model did what it is asked, where the run is judged by the
            grasp, and otherwise whether the task was carried out.
        """
        if self.judged_by_the_grasp:
            return self.lifted >= self.minimum_lift
        return super().task_is_done

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
        Try the drawn grasps that come in most squarely from where Tiago stands, and
        keep the first one the arm reaches.

        :return: Whether the arm reaches any of them.
        """
        for candidate, surface_grasp in self.grasps_from_tiagos_side()[
            : self.grasps_to_try
        ]:
            self.chosen_grasp = candidate
            if self.can_reach(self.pre_grasp()):
                logger.info(
                    "taking the %s by a learned grasp: azimuth %.2f, height %.2f,"
                    " depth %.3f m, pitch %.2f, roll %.2f",
                    self.learned.annotation_type.lower(),
                    surface_grasp.azimuth,
                    surface_grasp.height,
                    surface_grasp.depth,
                    surface_grasp.pitch,
                    surface_grasp.roll,
                )
                return True
        self.chosen_grasp = None
        return False

    def grasps_from_tiagos_side(self) -> list[tuple[GraspCandidate, SurfaceGrasp]]:
        """
        A grasp says which way round the object it takes hold, and the object lies in
        the kitchen whichever way RoboCasa turned it, so much of what the model draws
        asks the arm to reach around the object to its far side.

        :return: The drawn grasps whose hand comes in from the side Tiago stands on,
            each with the grasp it was drawn as, the most squarely first.
        """
        towards_tiago = self.robot.root.global_transform.to_np()[:2, 3]
        squareness = []
        for candidate, surface_grasp in self.candidates_on_the_object():
            world_T_grasp = candidate.world_T_grasp.to_np()
            towards = towards_tiago - world_T_grasp[:2, 3]
            along = float(world_T_grasp[:2, 0] @ towards / np.linalg.norm(towards))
            if along >= 0.0:
                continue
            squareness.append((along, candidate, surface_grasp))
        return [
            (candidate, surface_grasp)
            for _, candidate, surface_grasp in sorted(
                squareness, key=lambda row: row[0]
            )
        ]

    def candidates_on_the_object(self) -> list[tuple[GraspCandidate, SurfaceGrasp]]:
        """
        :return: The drawn grasps as grasps on the object, kept once they are worked
            out: a grasp is told about the object's own surface, which does not move
            while a stand is looked for, and casting the rays against it again at every
            stand is what makes the search slow.
        """
        if self.candidates is None:
            self.candidates = [
                (grasp.grasp_candidate(self.food), grasp)
                for surface_grasp in self.learned.grasps[: self.grasps_to_choose_from]
                if surface_grasp.reaches_surface_of(self.food)
                for grasp in self.both_ways_round(surface_grasp)
            ]
        return self.candidates

    def both_ways_round(self, surface_grasp: SurfaceGrasp) -> list[SurfaceGrasp]:
        """
        :param surface_grasp: A drawn grasp.
        :return: It, and the same grasp with its fingers the other way round.

        Two fingers closing along a line close on the same two faces whichever way
        round they are, so the model says as much about the one as about the other,
        while a wrist that cannot turn far enough for one can often manage the other.
        """
        taken = self.as_deep_as_tiago_grips(surface_grasp)
        return [taken, replace(taken, roll=taken.roll + math.pi)]

    def as_deep_as_tiago_grips(self, surface_grasp: SurfaceGrasp) -> SurfaceGrasp:
        """
        :param surface_grasp: A drawn grasp.
        :return: The same grasp, closing where Tiago's fingers have the object between
            them.

        Of what a grasp states, the depth is the one thing about the gripper rather
        than about the object: how far past the surface *that* gripper's centre closes.
        The PR2's couple of centimetres leave Tiago, whose pads sit four centimetres
        either side of its grasp, with one pad on the object's face and the other well
        inside it. Where the hand comes from and how it is turned -- what the model
        actually learned -- are left as drawn.
        """
        extent = self.extent(self.food.root)
        across = np.array([extent.depth, extent.width]) / 2
        sight = np.array(
            [math.cos(surface_grasp.azimuth), math.sin(surface_grasp.azimuth)]
        )
        return replace(surface_grasp, depth=float(np.abs(sight) @ across))


def taking_the_learned_grasps(
    task: Type[TiagoPickPlaceTask], learned: LearnedGrasps, by_the_grasp: bool
) -> Type[TiagoPickPlaceTask]:
    """
    Which pick-and-place task to run is asked for on the command line, while the grasps
    to take the object by belong to the task, so the one is made to carry the other.

    :param task: The task to run.
    :param learned: The grasps to take the object by.
    :param by_the_grasp: Whether an episode counts once one of them lifts the object.
    :return: That task, taking the object by those grasps.
    """
    return make_dataclass(
        task.__name__,
        [
            (
                "learned",
                LearnedGrasps,
                dataclass_field(default_factory=lambda: learned),
            ),
            ("judged_by_the_grasp", bool, dataclass_field(default=by_the_grasp)),
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
    parser.add_argument(
        "--by-the-grasp",
        action="store_true",
        help="count an episode once a learned grasp has lifted the object, rather than"
        " once the task it serves is carried out",
    )
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
    task = taking_the_learned_grasps(
        TASKS[arguments.task], learned, arguments.by_the_grasp
    )
    Trials(
        task=task,
        settings=settings_from(arguments),
        attempts=arguments.attempts,
        tries_every_kitchen=arguments.benchmark,
    ).run()


if __name__ == "__main__":
    main()
