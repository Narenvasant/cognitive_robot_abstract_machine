"""
The causal answers of the clutter picking experiment in words a person reads: what each
answer means in general, what it says about one run before the pick, and why a run came
out the way it did.

Every sentence is built from the numbers of the answer documents
:class:`~experiments.causal_reasoning.tracy_clutter_picking.presentation.CausalAnalysis`
writes, so the words never claim more than the answers hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Callable, Dict, List, Optional

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClosingAxisSide,
    ClutterEnvironment,
    ClutterPickScene,
    ClutterSceneLayout,
    DistanceBand,
    NeighbourThresholds,
)

# %% reading a probability


def percent(probability: Optional[float]) -> str:
    """
    :param probability: A probability, or None where it is unknown.
    :return: It in whole percent, or ``unknown``.
    """
    return "unknown" if probability is None else f"{100 * probability:.0f}%"


def points(difference: float) -> str:
    """
    :param difference: A difference of two probabilities.
    :return: It in percentage points.
    """
    return f"{100 * abs(difference):.0f} percentage points"


class Likelihood(StrEnum):
    """
    How likely a lift is, in the words a person uses.
    """

    LIKELY = "likely"
    """
    At least :attr:`LikelihoodScale.likely_from`.
    """

    UNCERTAIN = "uncertain"
    """
    Between the two bounds of the :class:`LikelihoodScale`.
    """

    UNLIKELY = "unlikely"
    """
    Below :attr:`LikelihoodScale.unlikely_below`.
    """


@dataclass
class LikelihoodScale:
    """
    Where a probability stops being called unlikely and starts being called likely.
    """

    unlikely_below: float = 0.4
    """
    A probability below this is unlikely.
    """

    likely_from: float = 0.7
    """
    A probability from this on is likely.
    """

    def of(self, probability: float) -> Likelihood:
        """
        :param probability: A probability.
        :return: How likely it is in words.
        """
        if probability >= self.likely_from:
            return Likelihood.LIKELY
        if probability < self.unlikely_below:
            return Likelihood.UNLIKELY
        return Likelihood.UNCERTAIN


# %% the causes of one run


class QuestionKind(StrEnum):
    """
    Which of the experiment's questions an answer belongs to.
    """

    FRICTION = "friction"
    CROWDING_ADJUSTED = "crowding_adjusted"
    CROWDING_UNADJUSTED = "crowding_unadjusted"
    CLOSING_AXIS_SIDE = "closing_axis_side"


@dataclass
class RunCauses:
    """
    What a run is given that the causal answers are about, known before the pick.
    """

    environment: ClutterEnvironment
    """
    Where the clutter stands.
    """

    friction: float
    """
    The grasp friction.
    """

    adjacent_neighbours: int
    """
    How many neighbours stand right next to the target.
    """

    neighbour_sides: List[ClosingAxisSide]
    """
    Per neighbour, its side of the fingers' closing axis.
    """

    @classmethod
    def of(
        cls,
        layout: ClutterSceneLayout,
        thresholds: NeighbourThresholds = NeighbourThresholds(),
    ) -> RunCauses:
        """
        :param layout: The run's layout.
        :param thresholds: What turns a neighbour's geometry into its band and side.
        :return: The run's causes.
        """
        target = layout.target
        closing_axis_yaw = target.yaw + layout.grasp_yaw
        return cls(
            environment=layout.environment,
            friction=layout.friction_coefficient,
            adjacent_neighbours=sum(
                thresholds.distance_band(placed.distance_to(target))
                is DistanceBand.ADJACENT
                for placed in layout.neighbours
            ),
            neighbour_sides=[
                thresholds.closing_axis_side(
                    math.atan2(placed.y - target.y, placed.x - target.x)
                    - closing_axis_yaw
                )
                for placed in layout.neighbours
            ],
        )


@dataclass
class SimilarAttempts:
    """
    The recorded attempts that had what a run has.
    """

    count: int
    """
    How many there are.
    """

    lifted: int
    """
    How many of them lifted the target.
    """

    @classmethod
    def among(
        cls,
        scenes: List[ClutterPickScene],
        environment: ClutterEnvironment,
        friction: Optional[float] = None,
        adjacent_neighbours: Optional[int] = None,
    ) -> SimilarAttempts:
        """
        :param scenes: The recorded attempts.
        :param environment: The environment to match.
        :param friction: The friction to match, if any.
        :param adjacent_neighbours: The number of adjacent neighbours to match, if any.
        :return: The attempts that match.
        """
        matching = [
            scene
            for scene in scenes
            if scene.environment is environment
            and (friction is None or scene.friction_coefficient == friction)
            and (
                adjacent_neighbours is None or crowding_of(scene) == adjacent_neighbours
            )
        ]
        return cls(len(matching), sum(scene.lifted for scene in matching))

    def sentence(self, description: str, without_any: str = "") -> str:
        """
        :param description: What the attempts had, after "attempts".
        :param without_any: What follows when there is no such attempt.
        :return: How many there were and how many came up.
        """
        if self.count == 0:
            return f"No recorded attempt {description}{without_any}."
        return (
            f"Of the {self.count} recorded attempts {description}, {self.lifted} "
            f"lifted the carton."
        )


def crowding_of(scene: ClutterPickScene) -> int:
    """
    :param scene: A recorded attempt.
    :return: How many of its neighbours stand adjacent to the target.
    """
    return sum(
        neighbour.distance_band is DistanceBand.ADJACENT
        for neighbour in scene.neighbours
    )


# %% one answer document


@dataclass
class AnswerRegions:
    """
    The regions of one causal answer document, looked up by the cause's value.
    """

    question: Dict[str, Any]
    """
    The answer document.
    """

    def at(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        :param value: A value of the cause.
        :return: The region holding it, or None if the recorded attempts never had it.
        """
        matching = [
            region for region in self.question["regions"] if value in region["holds"]
        ]
        return matching[0] if matching else None

    @property
    def best(self) -> Dict[str, Any]:
        """
        :return: The region under which the effect is most likely when set.
        """
        return max(self.question["regions"], key=lambda region: region["adjusted"])

    @property
    def worst(self) -> Dict[str, Any]:
        """
        :return: The region under which the effect is least likely when set.
        """
        return min(self.question["regions"], key=lambda region: region["adjusted"])

    @staticmethod
    def value_of(region: Dict[str, Any]) -> Any:
        """
        :param region: A region.
        :return: The single value it holds, or its description if it holds several.
        """
        holds = region["holds"]
        return holds[0] if len(holds) == 1 else region["description"]


def friction_text(value: Any) -> str:
    """
    :param value: A friction.
    :return: It in words.
    """
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def crowding_text(value: Any) -> str:
    """
    :param value: A number of adjacent neighbours.
    :return: It in words.
    """
    if value == 0:
        return "no carton right next to it"
    if value == 1:
        return "1 carton right next to it"
    return f"{value} cartons right next to it"


def compared_with_best(
    probability: Optional[float], best_value: str, best_probability: float
) -> str:
    """
    :param probability: How likely the lift is at the run's value.
    :param best_value: The best value of the cause, in words.
    :param best_probability: How likely the lift is at it.
    :return: How the run's value compares with the best one.
    """
    if probability is not None and best_probability - probability < 0.01:
        return "No value of it does better."
    return f"Set to {best_value}, it would be {percent(best_probability)}."


# %% what each answer means in general


@dataclass
class GeneralReading:
    """
    The one-sentence takeaway of every causal answer of the recorded attempts.
    """

    scale: LikelihoodScale = field(default_factory=LikelihoodScale)
    """
    What counts as likely.
    """

    small_shift: float = 0.05
    """
    A shift from adjusting below this barely changes the answer.
    """

    def takeaway(
        self, question: Dict[str, Any], questions: List[Dict[str, Any]]
    ) -> str:
        """
        :param question: One answer document.
        :param questions: Every answer document of the experiment.
        :return: What the answer means, in one or two sentences.
        """
        regions = AnswerRegions(question)
        match QuestionKind(question["kind"]):
            case QuestionKind.FRICTION:
                return self.friction_takeaway(regions)
            case QuestionKind.CROWDING_ADJUSTED:
                return self.crowding_takeaway(regions)
            case QuestionKind.CROWDING_UNADJUSTED:
                adjusted = next(
                    other
                    for other in questions
                    if other["kind"] == QuestionKind.CROWDING_ADJUSTED
                )
                return self.confounding_takeaway(regions, AnswerRegions(adjusted))
            case QuestionKind.CLOSING_AXIS_SIDE:
                return self.side_takeaway(regions)

    def friction_takeaway(self, regions: AnswerRegions) -> str:
        """
        :param regions: The friction answer.
        :return: Its takeaway.
        """
        worst, best = regions.worst, regions.best
        shift = regions.question["largestShiftFromAdjusting"]
        bias = (
            "Comparing bins with tables barely changes this, so it is the friction "
            "itself, not where the clutter stands, that decides."
            if shift < self.small_shift
            else f"Adjusting for the environment moves these numbers by up to "
            f"{points(shift)}: where the clutter stands also matters."
        )
        return (
            f"The grip decides the pick. Set to {friction_text(AnswerRegions.value_of(worst))}, "
            f"the friction lifts the carton {percent(worst['adjusted'])} of the time; set to "
            f"{friction_text(AnswerRegions.value_of(best))}, {percent(best['adjusted'])}. {bias}"
        )

    def crowding_takeaway(self, regions: AnswerRegions) -> str:
        """
        :param regions: The crowding answer, adjusted for the environment.
        :return: Its takeaway.
        """
        ordered = sorted(
            regions.question["regions"], key=lambda region: region["holds"][0]
        )
        alone, crowded = ordered[0], ordered[-1]
        direction = (
            "The more cartons stand right next to the target, the less often it comes up"
            if crowded["adjusted"] < alone["adjusted"]
            else "Cartons right next to the target do not keep it from coming up"
        )
        return (
            f"{direction}: with {crowding_text(AnswerRegions.value_of(alone))} a pick lifts it "
            f"{percent(alone['adjusted'])} of the time, with "
            f"{crowding_text(AnswerRegions.value_of(crowded))} {percent(crowded['adjusted'])}. "
            f"Neighbours that close get in the way of the closing fingers."
        )

    def confounding_takeaway(
        self, unadjusted: AnswerRegions, adjusted: AnswerRegions
    ) -> str:
        """
        :param unadjusted: The crowding answer without adjusting for anything.
        :param adjusted: The same answer adjusted for the environment.
        :return: How much ignoring the environment would mislead.
        """
        shifts = [
            (region, adjusted.at(region["holds"][0]))
            for region in unadjusted.question["regions"]
        ]
        region, counterpart = max(
            shifts,
            key=lambda pair: abs(pair[1]["adjusted"] - pair[0]["adjusted"]),
        )
        difference = counterpart["adjusted"] - region["adjusted"]
        if abs(difference) < self.small_shift:
            return (
                "Ignoring the environment would give nearly the same answer, so the "
                "crowding answer above is not an artefact of where the clutter stands."
            )
        return (
            f"Ignoring the environment would mislead: with "
            f"{crowding_text(AnswerRegions.value_of(region))}, simply counting the recorded "
            f"attempts says a lift comes {percent(region['adjusted'])} of the time, "
            f"while comparing bins with bins and tables with tables says "
            f"{percent(counterpart['adjusted'])}. Bins and tables differ in how crowded "
            f"and how slippery their attempts were, which mixes the two up."
        )

    @staticmethod
    def side_takeaway(regions: AnswerRegions) -> str:
        """
        :param regions: The closing-axis answer.
        :return: Its takeaway.
        """
        along = regions.at(ClosingAxisSide.ALONG.value)
        across = regions.at(ClosingAxisSide.ACROSS.value)
        return (
            f"The fingers push what stands in their way: a neighbour in line with the "
            f"closing fingers is pushed aside {percent(along['adjusted'])} of the time, one "
            f"off to their side only {percent(across['adjusted'])}."
        )


# %% what the answers say about one run


def answer_about(answers: Dict[str, Any], cause: str) -> Dict[str, Any]:
    """
    :param answers: A run's answers document.
    :param cause: The cause an answer is about.
    :return: That answer.
    """
    return next(answer for answer in answers["answers"] if answer["cause"] == cause)


@dataclass
class RunReading:
    """
    The causal answers applied to one run: what its causes make likely before the pick,
    and how that compares with what the pick did.
    """

    questions: Dict[QuestionKind, Dict[str, Any]]
    """
    The experiment's answer documents.
    """

    scenes: List[ClutterPickScene]
    """
    The recorded attempts the answers were fitted on.
    """

    scale: LikelihoodScale = field(default_factory=LikelihoodScale)
    """
    What counts as likely.
    """

    def friction_answer(self, causes: RunCauses) -> Dict[str, Any]:
        """
        :param causes: The run's causes.
        :return: What the friction answer says about the run.
        """
        regions = AnswerRegions(self.questions[QuestionKind.FRICTION])
        here, best = regions.at(causes.friction), regions.best
        similar = SimilarAttempts.among(
            self.scenes, causes.environment, friction=causes.friction
        )
        probability = None if here is None else here["adjusted"]
        return {
            "cause": "Grasp friction",
            "question": "Does the grip of the fingertips let Tracy lift the carton?",
            "runValue": friction_text(causes.friction),
            "probability": probability,
            "bestValue": friction_text(AnswerRegions.value_of(best)),
            "bestProbability": best["adjusted"],
            "likelihood": None if here is None else self.scale.of(probability).value,
            "reading": (
                f"This run grips with a friction of {friction_text(causes.friction)}. "
                f"With that friction, a pick lifts the carton {percent(probability)} of "
                f"the time. "
                + compared_with_best(
                    probability,
                    friction_text(AnswerRegions.value_of(best)),
                    best["adjusted"],
                )
            ),
            "evidence": similar.sentence(
                f"in a {causes.environment.value} with this friction",
                ", so the answer rests on the attempts in the other environment",
            ),
        }

    def crowding_answer(self, causes: RunCauses) -> Dict[str, Any]:
        """
        :param causes: The run's causes.
        :return: What the crowding answer says about the run.
        """
        regions = AnswerRegions(self.questions[QuestionKind.CROWDING_ADJUSTED])
        here, best = regions.at(causes.adjacent_neighbours), regions.best
        similar = SimilarAttempts.among(
            self.scenes,
            causes.environment,
            adjacent_neighbours=causes.adjacent_neighbours,
        )
        probability = None if here is None else here["adjusted"]
        return {
            "cause": "Crowding",
            "question": "Do the cartons right next to the target keep it from coming up?",
            "runValue": crowding_text(causes.adjacent_neighbours),
            "probability": probability,
            "bestValue": crowding_text(AnswerRegions.value_of(best)),
            "bestProbability": best["adjusted"],
            "likelihood": None if here is None else self.scale.of(probability).value,
            "reading": (
                f"The target has {crowding_text(causes.adjacent_neighbours)}. With that "
                f"many, a pick lifts it {percent(probability)} of the time. "
                + compared_with_best(
                    probability,
                    crowding_text(AnswerRegions.value_of(best)),
                    best["adjusted"],
                )
            ),
            "evidence": similar.sentence(
                f"in a {causes.environment.value} with "
                f"{crowding_text(causes.adjacent_neighbours)}",
                ", so the answer rests on the attempts in the other environment",
            ),
        }

    def environment_answer(self, causes: RunCauses) -> Dict[str, Any]:
        """
        :param causes: The run's causes.
        :return: Why the answers compare like with like for the run's environment.
        """
        rates = {
            environment: SimilarAttempts.among(self.scenes, environment)
            for environment in ClutterEnvironment
        }
        crowding = {
            environment: self.mean_over(environment, crowding_of)
            for environment in ClutterEnvironment
        }
        friction = {
            environment: self.mean_over(
                environment, lambda scene: scene.friction_coefficient
            )
            for environment in ClutterEnvironment
        }
        here = rates[causes.environment]
        others = ", ".join(
            f"{percent(attempts.lifted / max(attempts.count, 1))} on a "
            f"{environment.value}"
            for environment, attempts in rates.items()
            if environment is not causes.environment
        )
        return {
            "cause": "Environment",
            "runValue": causes.environment.value,
            "reading": (
                f"This run stands in a {causes.environment.value}. Recorded picks there "
                f"lifted the carton {percent(here.lifted / max(here.count, 1))} of the "
                f"time, against {others}. But the environment also changes the other "
                f"causes: in a bin {crowding[ClutterEnvironment.BIN]:.1f} cartons stood "
                f"right next to the target on average and the friction was "
                f"{friction[ClutterEnvironment.BIN]:.2f}, on a table "
                f"{crowding[ClutterEnvironment.TABLE]:.1f} and "
                f"{friction[ClutterEnvironment.TABLE]:.2f}. So the answers above "
                f"compare bins with bins and tables with tables before they blame the "
                f"friction or the crowding."
            ),
        }

    def mean_over(
        self,
        environment: ClutterEnvironment,
        quantity: Callable[[ClutterPickScene], float],
    ) -> float:
        """
        :param environment: An environment.
        :param quantity: What to average.
        :return: Its mean over the recorded attempts in the environment.
        """
        values = [
            quantity(scene) for scene in self.scenes if scene.environment is environment
        ]
        return sum(values) / max(len(values), 1)

    def neighbour_answer(self, causes: RunCauses) -> Dict[str, Any]:
        """
        :param causes: The run's causes.
        :return: Which neighbours the pick is expected to push, by their side of the
            closing axis.
        """
        regions = AnswerRegions(self.questions[QuestionKind.CLOSING_AXIS_SIDE])
        chance = {side: regions.at(side.value)["adjusted"] for side in ClosingAxisSide}
        along = [
            index
            for index, side in enumerate(causes.neighbour_sides)
            if side is ClosingAxisSide.ALONG
        ]
        expected = sum(chance[side] for side in causes.neighbour_sides)
        named = (
            "none of the neighbours stands"
            if not along
            else f"neighbour{'s' if len(along) > 1 else ''} "
            f"{', '.join(str(index) for index in along)} "
            f"{'stand' if len(along) > 1 else 'stands'}"
        )
        return {
            "cause": "Side of the closing axis",
            "alongClosingAxis": along,
            "expectedPushed": expected,
            "reading": (
                f"In this clutter {named} in line with the closing fingers, where a "
                f"neighbour is pushed {percent(chance[ClosingAxisSide.ALONG])} of the "
                f"time; the other {len(causes.neighbour_sides) - len(along)} stand off "
                f"to the side ({percent(chance[ClosingAxisSide.ACROSS])} each). The pick "
                f"is expected to push about {expected:.1f} neighbours aside."
            ),
        }

    def document(self, run: str, causes: RunCauses) -> Dict[str, Any]:
        """
        :param run: The run's identifier.
        :param causes: The run's causes.
        :return: The run's answers before the pick, with the expected outcome.
        """
        friction = self.friction_answer(causes)
        crowding = self.crowding_answer(causes)
        limiting = min(
            [
                answer
                for answer in (friction, crowding)
                if answer["probability"] is not None
            ],
            key=lambda answer: answer["probability"],
            default=None,
        )
        similar = SimilarAttempts.among(
            self.scenes,
            causes.environment,
            friction=causes.friction,
            adjacent_neighbours=causes.adjacent_neighbours,
        )
        expectation = (
            {
                "likelihood": None,
                "reading": "The recorded attempts never had these causes.",
            }
            if limiting is None
            else {
                "likelihood": self.scale.of(limiting["probability"]).value,
                "limitingCause": limiting["cause"],
                "reading": (
                    f"A lift is {self.scale.of(limiting['probability']).value}: "
                    f"{limiting['cause'].lower()} limits it most, at "
                    f"{percent(limiting['probability'])}. "
                    + similar.sentence(
                        f"with the same environment, friction and crowding"
                    )
                ),
            }
        )
        return {
            "run": run,
            "state": "running",
            "causes": {
                "environment": causes.environment.value,
                "friction": causes.friction,
                "adjacentNeighbours": causes.adjacent_neighbours,
                "neighbours": len(causes.neighbour_sides),
            },
            "answers": [
                friction,
                crowding,
                self.environment_answer(causes),
                self.neighbour_answer(causes),
            ],
            "expectation": expectation,
        }

    def with_outcome(
        self,
        document: Dict[str, Any],
        scene: Optional[ClutterPickScene],
        failure: Optional[str],
    ) -> Dict[str, Any]:
        """
        :param document: The run's answers before the pick.
        :param scene: The run as an attempt, or None if its motion failed.
        :param failure: What stopped the run's motion, if anything did.
        :return: The answers with what the pick did and whether it was expected.
        """
        finished = dict(document, state="finished")
        if scene is None:
            finished["outcome"] = {
                "lifted": False,
                "asExpected": None,
                "reading": f"The motion stopped before the grasp was tried: {failure}",
            }
            return finished
        expected = document["expectation"]["likelihood"]
        as_expected = (
            None
            if expected in (None, Likelihood.UNCERTAIN.value)
            else (expected == Likelihood.LIKELY.value) == scene.lifted
        )
        pushed = [
            index
            for index, neighbour in enumerate(scene.neighbours)
            if neighbour.disturbed
        ]
        result = (
            f"The carton came up {100 * scene.lift_height:.1f} cm"
            if scene.lifted
            else f"The carton stayed down (it rose {100 * scene.lift_height:.1f} cm)"
        )
        verdict = {
            None: "",
            True: ", as the answers expected",
            False: ", against what the answers expected",
        }[as_expected]
        finished["outcome"] = {
            "lifted": scene.lifted,
            "asExpected": as_expected,
            "pushedNeighbours": pushed,
            "reading": (
                f"{result}{verdict}. The pick pushed {len(pushed)} neighbour"
                f"{'' if len(pushed) == 1 else 's'} aside, against about "
                f"{answer_about(document, 'Side of the closing axis')['expectedPushed']:.1f} expected."
            ),
        }
        return finished


# %% why a run came out the way it did


@dataclass
class DiagnosisStory:
    """
    A run's diagnosis told as the steps a person reasons through: what happened, what
    could explain it, what each cause would have changed, and what to do.
    """

    explaining_gap: float = 0.1
    """
    How much changing a cause alone has to raise the chance of the lift for the cause
    to explain a failure.
    """

    def steps(
        self,
        diagnosis: Dict[str, Any],
        answers: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        :param diagnosis: The run's diagnosis document, with its factors ranked.
        :param answers: The run's answers, with its outcome.
        :return: The steps, each with a title and a paragraph.
        """
        outcome = answers["outcome"]
        factors = diagnosis["factors"]
        steps = [{"title": "What happened", "text": outcome["reading"]}]
        if diagnosis.get("motionFailure"):
            steps.append(
                {
                    "title": "Why the grasp was never tested",
                    "text": "The robot could not finish its motion, so friction and "
                    "crowding were never put to the test in this run.",
                }
            )
            return steps
        named = " and ".join(
            f"{factor['factor'].lower()} ({factor['observed']})" for factor in factors
        )
        steps.append(
            {
                "title": "What could explain it",
                "text": f"Two things in this run are known to decide whether the carton "
                f"comes up: {named}. Each is checked by asking what would have happened "
                f"had only that one been different.",
            }
        )
        for factor in factors:
            steps.append(
                {
                    "title": f"If only the {factor['factor'].lower()} had been different",
                    "text": (
                        f"Kept at {factor['observed']}, a pick lifts the carton "
                        f"{percent(factor['probability_at_observed'])} of the time. "
                        + compared_with_best(
                            factor["probability_at_observed"],
                            factor["best_value"],
                            factor["probability_at_best"],
                        )
                    ),
                }
            )
        if diagnosis["lifted"]:
            steps.append(
                {
                    "title": "Why it worked",
                    "text": "Neither cause stood in the way: "
                    + "; ".join(
                        f"{factor['factor'].lower()} at {factor['observed']} lifts "
                        f"{percent(factor['probability_at_observed'])} of the time"
                        for factor in factors
                    )
                    + ".",
                }
            )
            return steps
        primary = factors[0]
        steps.append(
            {
                "title": "Conclusion",
                "text": (
                    f"{primary['factor']} explains the failure best: changing it alone "
                    f"would have raised the chance of the lift the most, by "
                    f"{points(primary['gap'])}."
                    if primary["gap"] >= self.explaining_gap
                    else "Neither cause explains the failure well: changing either "
                    "alone would barely have helped, so this pick most likely failed "
                    "by chance."
                ),
            }
        )
        steps.append({"title": "What to change", "text": diagnosis["correction"]})
        steps.append(
            {
                "title": "Why the comparison is fair",
                "text": answer_about(answers, "Environment")["reading"],
            }
        )
        return steps
