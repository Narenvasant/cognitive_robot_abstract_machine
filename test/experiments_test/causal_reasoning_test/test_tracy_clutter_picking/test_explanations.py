"""
Tests for :mod:`experiments.causal_reasoning.tracy_clutter_picking.explanations`.
"""

from __future__ import annotations

import pytest

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClosingAxisSide,
    ClutterEnvironment,
    ClutterPickScene,
    ClutterSceneLayout,
    ObjectCategory,
    PlacedObject,
)
from experiments.causal_reasoning.tracy_clutter_picking.explanations import (
    AnswerRegions,
    DiagnosisStory,
    Likelihood,
    LikelihoodScale,
    QuestionKind,
    RunCauses,
    RunReading,
)

# %% answer documents as the presentation writes them


def region(value, conditioned: float, adjusted: float) -> dict:
    """
    :return: A region of a cause holding one value.
    """
    return {
        "description": f"[{value}, {value}]",
        "share": 0.5,
        "conditioned": conditioned,
        "adjusted": adjusted,
        "holds": [value],
    }


def question(kind: QuestionKind, regions: list[dict]) -> dict:
    """
    :return: An answer document of the given kind.
    """
    return {
        "kind": kind.value,
        "asked": kind.value,
        "adjustedFor": [],
        "trainingExamples": 10,
        "largestShiftFromAdjusting": 0.0,
        "regions": regions,
    }


@pytest.fixture
def questions() -> dict[QuestionKind, dict]:
    """
    :return: Answers where a slippery grip never lifts and a crowded target seldom does.
    """
    return {
        QuestionKind.FRICTION: question(
            QuestionKind.FRICTION,
            [region(0.125, 0.0, 0.0), region(0.5, 1.0, 1.0)],
        ),
        QuestionKind.CROWDING_ADJUSTED: question(
            QuestionKind.CROWDING_ADJUSTED,
            [region(0, 0.9, 0.9), region(2, 0.5, 0.45)],
        ),
        QuestionKind.CROWDING_UNADJUSTED: question(
            QuestionKind.CROWDING_UNADJUSTED,
            [region(0, 0.9, 0.9), region(2, 0.5, 0.5)],
        ),
        QuestionKind.CLOSING_AXIS_SIDE: question(
            QuestionKind.CLOSING_AXIS_SIDE,
            [region("along", 0.3, 0.3), region("across", 0.05, 0.05)],
        ),
    }


def milk(x: float, y: float) -> PlacedObject:
    """
    :return: A carton standing at the given place.
    """
    return PlacedObject(category=ObjectCategory.MILK, x=x, y=y, yaw=0.0)


@pytest.fixture
def slippery_crowded_layout() -> ClutterSceneLayout:
    """
    :return: A target with two adjacent neighbours, one in line with the closing axis
        and one off to its side, picked with the slipperiest grip.
    """
    return ClutterSceneLayout(
        environment=ClutterEnvironment.BIN,
        objects=[milk(0.0, 0.0), milk(0.07, 0.0), milk(0.0, 0.07)],
        target_index=0,
        friction_coefficient=0.125,
        grasp_yaw=0.0,
    )


def scene_of(layout: ClutterSceneLayout, lifted: bool) -> ClutterPickScene:
    """
    :return: The layout recorded as an attempt that did or did not lift the target.
    """
    return ClutterPickScene(
        environment=layout.environment,
        target_x=0.0,
        target_y=0.0,
        friction_coefficient=layout.friction_coefficient,
        grasp_yaw=layout.grasp_yaw,
        neighbours=[],
        lifted=lifted,
        lift_height=0.2 if lifted else 0.0,
    )


# %% the causes of a run


def test_run_causes_count_the_adjacent_neighbours_and_their_sides(
    slippery_crowded_layout,
):
    causes = RunCauses.of(slippery_crowded_layout)

    assert (causes.adjacent_neighbours, causes.neighbour_sides) == (
        2,
        [ClosingAxisSide.ALONG, ClosingAxisSide.ACROSS],
    )


def test_a_region_is_found_by_the_value_it_holds(questions):
    regions = AnswerRegions(questions[QuestionKind.FRICTION])

    assert (regions.at(0.5), regions.best, regions.at(0.75)) == (
        questions[QuestionKind.FRICTION]["regions"][1],
        questions[QuestionKind.FRICTION]["regions"][1],
        None,
    )


@pytest.mark.parametrize(
    "probability, likelihood",
    [(0.0, Likelihood.UNLIKELY), (0.5, Likelihood.UNCERTAIN), (0.9, Likelihood.LIKELY)],
)
def test_likelihood_scale_names_a_probability(probability, likelihood):
    assert LikelihoodScale().of(probability) is likelihood


# %% what the answers say about a run


def test_expectation_names_the_cause_that_limits_the_lift_most(
    questions, slippery_crowded_layout
):
    document = RunReading(questions, []).document(
        "run_001", RunCauses.of(slippery_crowded_layout)
    )

    assert (
        document["expectation"]["limitingCause"],
        document["expectation"]["likelihood"],
    ) == ("Grasp friction", Likelihood.UNLIKELY.value)


def test_expected_neighbours_pushed_add_up_their_sides(
    questions, slippery_crowded_layout
):
    regions = AnswerRegions(questions[QuestionKind.CLOSING_AXIS_SIDE])
    document = RunReading(questions, []).document(
        "run_001", RunCauses.of(slippery_crowded_layout)
    )
    [neighbours] = [
        answer
        for answer in document["answers"]
        if answer["cause"] == "Side of the closing axis"
    ]

    assert (neighbours["alongClosingAxis"], neighbours["expectedPushed"]) == (
        [0],
        regions.at("along")["adjusted"] + regions.at("across")["adjusted"],
    )


@pytest.mark.parametrize("lifted, as_expected", [(False, True), (True, False)])
def test_outcome_says_whether_the_pick_went_as_expected(
    questions, slippery_crowded_layout, lifted, as_expected
):
    reading = RunReading(questions, [])
    document = reading.document("run_001", RunCauses.of(slippery_crowded_layout))

    finished = reading.with_outcome(
        document, scene_of(slippery_crowded_layout, lifted), None
    )

    assert (finished["state"], finished["outcome"]["asExpected"]) == (
        "finished",
        as_expected,
    )


# %% why a run came out the way it did


def test_a_failed_pick_is_reasoned_through_to_a_correction(
    questions, slippery_crowded_layout
):
    reading = RunReading(questions, [])
    answers = reading.with_outcome(
        reading.document("run_001", RunCauses.of(slippery_crowded_layout)),
        scene_of(slippery_crowded_layout, False),
        None,
    )
    diagnosis = {
        "lifted": False,
        "motionFailure": None,
        "correction": "Grip harder.",
        "factors": [
            {
                "factor": "Grasp friction",
                "observed": "0.125",
                "probability_at_observed": 0.0,
                "best_value": "0.5",
                "probability_at_best": 1.0,
                "gap": 1.0,
            }
        ],
    }

    steps = DiagnosisStory().steps(diagnosis, answers)

    assert [step["title"] for step in steps] == [
        "What happened",
        "What could explain it",
        "If only the grasp friction had been different",
        "Conclusion",
        "What to change",
        "Why the comparison is fair",
    ]
