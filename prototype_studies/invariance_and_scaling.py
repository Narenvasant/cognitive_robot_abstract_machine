"""
Order independence and scaling checks of the pairwise-relations prototype.

Usage: python -m prototype_studies.invariance_and_scaling <output.json>
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from random_events.product_algebra import SimpleEvent

import test.krrood_test.dataset.ormatic_interface  # noqa: F401  # maps the mimic classes
from krrood.entity_query_language.factories import a
from krrood.parametrization.pairwise_intervention import (
    PairwiseInterventionResolver,
    PairwiseQuestion,
    PartOfInstance,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
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


def fitted(scenes, exposures):
    return RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=["DominoSceneAggregations.touching_pair_count()"],
            method=JointProbabilityTree(),
        ),
        part_interface=PartInterface(
            {"pieces": ["environment"]}, {"pieces": exposures}
        ),
        monte_carlo_sample_count=200,
    ).fit(scenes)


def row_query(
    piece_count: int,
    order: list[int],
    contact_order: list[int],
    touching: dict[int, object],
    bands: dict[int, object],
):
    """
    :param piece_count: Pieces in a row; piece ``i`` touches piece ``i + 1``.
    :param order: Where each listed piece position takes its piece from.
    :param contact_order: The order the contacts are listed in.
    :param touching: Values of contacts by the row position of their first piece.
    :param bands: Values of bands by row position.
    :return: The unresolved query.
    """
    pieces_in_row = [
        a(DominoPiece)(band=bands.get(position, ...), disturbed=...)
        for position in range(piece_count)
    ]
    contacts = [
        a(DominoContact)(
            first_piece=pieces_in_row[position],
            second_piece=pieces_in_row[position + 1],
            touching=touching.get(position, ...),
        )
        for position in contact_order
    ]
    return a(DominoScene)(
        environment=...,
        pieces=[pieces_in_row[position] for position in order],
        contacts=contacts,
        target_lifted=...,
    )


def answers_in_order(model, order, contact_order, seed):
    listed_position_of_piece_0 = order.index(0)
    separate = PairwiseQuestion(
        query_with_cause=lambda value: row_query(4, order, contact_order, {0: value}, {}),
        cause_values=[False, True],
        effect_attribute="disturbed",
        effect_value=True,
        effect_part=PartOfInstance("pieces", listed_position_of_piece_0),
        adjustment_names=["DominoScene.environment"],
    )
    contact_listed = contact_order.index(0)
    move = PairwiseQuestion(
        query_with_cause=lambda value: row_query(
            4, order, contact_order, {0: True}, {1: value}
        ),
        cause_values=[DominoBand.NEAR, DominoBand.FAR],
        effect_attribute="disturbed",
        effect_value=True,
        effect_part=PartOfInstance("pieces", listed_position_of_piece_0),
        adjustment_names=["DominoScene.environment"],
        evidence={f"DominoScene.contacts[{contact_listed}].touching": True},
    )
    result = []
    for question, (model_of_question) in ((separate, model[0]), (move, model[1])):
        np.random.seed(seed)
        result += [
            entry.probability
            for entry in PairwiseInterventionResolver(model_of_question)
            .resolve(question)
            .answers
        ]
    return result


def scene_probability(model, order, contact_order, scene, seed):
    """
    :return: The probability the circuit grounded for the listed order gives the
        scene's bands, disturbances and contacts, each read at its listed position.
    """
    query = row_query(4, order, contact_order, {}, {})
    query.resolve()
    np.random.seed(seed)
    circuit = model.ground(query)
    variable_of = {variable.name: variable for variable in circuit.variables}
    assignment = {}
    for listed, position in enumerate(order):
        piece = scene.pieces[position]
        assignment[variable_of[f"DominoScene.pieces[{listed}].band"]] = piece.band
        assignment[variable_of[f"DominoScene.pieces[{listed}].disturbed"]] = piece.disturbed
    for listed, position in enumerate(contact_order):
        assignment[variable_of[f"DominoScene.contacts[{listed}].touching"]] = scene.contacts[
            position
        ].touching
    return circuit.probability(SimpleEvent.from_data(assignment).as_composite_set())


def main():
    output = sys.argv[1]
    generator = DominoSceneGenerator(spillover=0.3)
    scenes = generator.scenes(1000, seed=7)
    models = (
        fitted(scenes, ["touching_piece_count"]),
        fitted(scenes, ["touching_near_piece_count", "touching_piece_count"]),
    )
    test_scene = generator.scene(np.random.default_rng(99), 4)
    identity = ([0, 1, 2, 3], [0, 1, 2])
    reference = answers_in_order(models, *identity, seed=0)
    reference_probability = scene_probability(models[0], *identity, test_scene, seed=0)
    random_state = np.random.default_rng(3)
    differences, probability_differences = [], []
    for _ in range(20):
        order = [int(value) for value in random_state.permutation(4)]
        contact_order = [int(value) for value in random_state.permutation(3)]
        answers = answers_in_order(models, order, contact_order, seed=0)
        differences.append(float(np.max(np.abs(np.array(answers) - reference))))
        probability_differences.append(
            abs(
                scene_probability(models[0], order, contact_order, test_scene, seed=0)
                - reference_probability
            )
        )
    result = {
        "order_independence": {
            "reorderings": 20,
            "reference_answers": reference,
            "max_answer_difference": max(differences),
            "reference_scene_probability": reference_probability,
            "max_scene_probability_difference": max(probability_differences),
        }
    }
    print(json.dumps(result), flush=True)
    scaling_generator = DominoSceneGenerator(spillover=0.3, piece_counts=(4, 9, 16, 25))
    scaling_model = fitted(scaling_generator.scenes(1000, seed=11), ["touching_piece_count"])
    result["scaling"] = []
    for piece_count in (4, 9, 16, 25):
        query = row_query(
            piece_count,
            list(range(piece_count)),
            list(range(piece_count - 1)),
            {position: bool(position % 2) for position in range(piece_count - 1)},
            {},
        )
        query.resolve()
        np.random.seed(0)
        start = time.time()
        circuit = scaling_model.ground(query)
        seconds = time.time() - start
        result["scaling"].append(
            {
                "pieces": piece_count,
                "pairs": piece_count - 1,
                "grounding_seconds": seconds,
                "nodes": len(list(circuit.nodes())),
                "variables": len(circuit.variables),
            }
        )
        print(json.dumps(result["scaling"][-1]), flush=True)
    json.dump(result, open(output, "w"), indent=1)


if __name__ == "__main__":
    main()
