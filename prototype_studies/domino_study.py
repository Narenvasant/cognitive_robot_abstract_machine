"""
Experiment 2 of the pairwise-relations prototype: domino clutter.

Pieces in a row topple each other where they touch with probability ``q``; a bin raises
how many pairs touch, how near the pieces stand and how easily they fall. Three
questions are answered by the new model (interface attribute + exposures), by the
current relational circuit and by plain conditioning on the training data, and compared
with the truth computed by intervening in the generator.

Usage: python -m prototype_studies.domino_study <output.json> [spillover ...]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import numpy as np

import test.krrood_test.dataset.ormatic_interface  # noqa: F401  # maps the mimic classes
from krrood.entity_query_language.factories import a, variable
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
from krrood.parametrization.pairwise_intervention import (
    PairwiseInterventionResolver,
    PairwiseQuestion,
    PartOfInstance,
)
from probabilistic_model.probabilistic_circuit.relational.part_interface import (
    PartInterface,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)
from test.krrood_test.dataset.domino_scenes import DominoSceneGenerator
from test.krrood_test.dataset.example_classes import (
    DominoBand,
    DominoContact,
    DominoPiece,
    DominoScene,
)

PIECES = 4
TRUTH_SAMPLES = 40000
MONTE_CARLO = 200


def scene_query(touching: list, bands: list) -> object:
    pieces = [a(DominoPiece)(band=band, disturbed=...) for band in bands]
    contacts = [
        a(DominoContact)(
            first_piece=pieces[position],
            second_piece=pieces[position + 1],
            touching=touching[position],
        )
        for position in range(len(pieces) - 1)
    ]
    return a(DominoScene)(
        environment=..., pieces=pieces, contacts=contacts, target_lifted=...
    )


# %% the questions


@dataclass
class Question:
    name: str
    asked: str
    pairwise: PairwiseQuestion
    exposures: list[str]

    def truth(self, generator: DominoSceneGenerator, seed: int) -> list[float]:
        raise NotImplementedError

    def plain_conditioning(self, scenes: list[DominoScene]) -> list[float]:
        raise NotImplementedError


class SeparateTwoPieces(Question):
    def __init__(self):
        super().__init__(
            "separate_two_pieces",
            "Separate piece 0 from piece 1: is piece 0 disturbed?",
            PairwiseQuestion(
                query_with_cause=lambda value: scene_query(
                    [value, ..., ...], [...] * PIECES
                ),
                cause_values=[False, True],
                effect_attribute="disturbed",
                effect_value=True,
                effect_part=PartOfInstance("pieces", 0),
                adjustment_names=["DominoScene.environment"],
            ),
            ["touching_piece_count"],
        )

    def truth(self, generator, seed):
        random_state = np.random.default_rng(seed)
        return [
            float(
                np.mean(
                    [
                        generator.scene(random_state, PIECES, touching={0: value})
                        .pieces[0]
                        .disturbed
                        for _ in range(TRUTH_SAMPLES)
                    ]
                )
            )
            for value in self.pairwise.cause_values
        ]

    def plain_conditioning(self, scenes):
        return [
            float(
                np.mean(
                    [
                        scene.pieces[0].disturbed
                        for scene in scenes
                        if scene.contacts[0].touching is value
                    ]
                )
            )
            for value in self.pairwise.cause_values
        ]


class MovePieceFar(Question):
    def __init__(self):
        super().__init__(
            "move_neighbour_far",
            "Move piece 1, which touches piece 0, to the far band: is piece 0 disturbed?",
            PairwiseQuestion(
                query_with_cause=lambda value: scene_query(
                    [True, ..., ...], [..., value] + [...] * (PIECES - 2)
                ),
                cause_values=[DominoBand.NEAR, DominoBand.FAR],
                effect_attribute="disturbed",
                effect_value=True,
                effect_part=PartOfInstance("pieces", 0),
                adjustment_names=["DominoScene.environment"],
                evidence={"DominoScene.contacts[0].touching": True},
            ),
            ["touching_near_piece_count", "touching_piece_count"],
        )

    def truth(self, generator, seed):
        random_state = np.random.default_rng(seed)
        result = []
        for value in self.pairwise.cause_values:
            kept = []
            while len(kept) < TRUTH_SAMPLES // 2:
                scene = generator.scene(random_state, PIECES, bands={1: value})
                if scene.contacts[0].touching:
                    kept.append(scene.pieces[0].disturbed)
            result.append(float(np.mean(kept)))
        return result

    def plain_conditioning(self, scenes):
        return [
            float(
                np.mean(
                    [
                        scene.pieces[0].disturbed
                        for scene in scenes
                        if scene.contacts[0].touching and scene.pieces[1].band is value
                    ]
                )
            )
            for value in self.pairwise.cause_values
        ]


class NoTouchingPairs(Question):
    def __init__(self):
        super().__init__(
            "no_touching_pairs",
            "Set the touching pairs to none or all three: is the target lifted?",
            PairwiseQuestion(
                query_with_cause=lambda value: scene_query([value] * (PIECES - 1), [...] * PIECES),
                cause_values=[False, True],
                effect_attribute="DominoScene.target_lifted",
                effect_value=True,
                adjustment_names=["DominoScene.environment"],
            ),
            [],
        )

    def truth(self, generator, seed):
        random_state = np.random.default_rng(seed)
        return [
            float(
                np.mean(
                    [
                        generator.scene(
                            random_state,
                            PIECES,
                            touching={position: value for position in range(PIECES - 1)},
                        ).target_lifted
                        for _ in range(TRUTH_SAMPLES)
                    ]
                )
            )
            for value in self.pairwise.cause_values
        ]

    def plain_conditioning(self, scenes):
        return [
            float(
                np.mean(
                    [
                        scene.target_lifted
                        for scene in scenes
                        if len(scene.pieces) == PIECES
                        and all(contact.touching is value for contact in scene.contacts)
                    ]
                    or [np.nan]
                )
            )
            for value in self.pairwise.cause_values
        ]


# %% the models


def fitted(scenes, exposures: list[str], with_interface: bool) -> RelationalProbabilisticCircuit:
    interface = PartInterface(
        {"pieces": ["environment"]} if with_interface else {},
        {"pieces": exposures} if exposures else {},
    )
    return RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=["DominoSceneAggregations.touching_pair_count()"],
            method=JointProbabilityTree(),
        ),
        part_interface=interface,
        monte_carlo_sample_count=MONTE_CARLO,
    ).fit(scenes)


def answer(model, question: Question, seed: int) -> list[float]:
    np.random.seed(seed)
    result = PairwiseInterventionResolver(model).resolve(question.pairwise)
    return [entry.probability for entry in result.answers]


def main():
    output = sys.argv[1]
    spillovers = [float(value) for value in sys.argv[2:]] or [0.0, 0.3, 0.6]
    questions = [SeparateTwoPieces(), MovePieceFar(), NoTouchingPairs()]
    rows = []
    for spillover in spillovers:
        generator = DominoSceneGenerator(spillover=spillover)
        truths = {question.name: question.truth(generator, 12345) for question in questions}
        for scene_count in (400, 1000):
            for repeat in range(3):
                scenes = generator.scenes(scene_count, seed=1000 * repeat + scene_count)
                for question in questions:
                    start = time.time()
                    new = answer(fitted(scenes, question.exposures, True), question, repeat)
                    current = answer(fitted(scenes, [], False), question, repeat)
                    plain = question.plain_conditioning(scenes)
                    row = dict(
                        spillover=spillover,
                        scene_count=scene_count,
                        repeat=repeat,
                        question=question.name,
                        cause_values=[str(value) for value in question.pairwise.cause_values],
                        truth=truths[question.name],
                        new=new,
                        current=current,
                        plain=plain,
                        seconds=time.time() - start,
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    json.dump(rows, open(output, "w"), indent=1)


if __name__ == "__main__":
    main()
