"""
Whether the neural estimator's answer about a terminal atom depends on how much capacity
the network has.

The question is whether an atom's element causes it to be terminal. Chemistry fixes the
answer: an atom with a single bond is terminal, so hydrogen and chlorine always are and
carbon and nitrogen never are. The estimator's conditional probabilities match that, and
its adjusted ones do not. This script fits the estimator twice on the same split, once
with the hidden layers the comparison uses and once with wider ones and far more
optimiser steps, and prints what each answers.

Run it as::

    python -m experiments.causal_reasoning.mutagenesis.deep_set_capacity
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from dataclasses import dataclass, replace

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
from experiments.causal_reasoning.comparison.neural_baseline import (
    NeuralAdjustmentBaseline,
)
from experiments.causal_reasoning.mutagenesis.dataset import (
    fetch_mutagenesis_molecules,
    mutagenesis_dataset,
)
from experiments.causal_reasoning.mutagenesis.domain import molecule_domain
from experiments.causal_reasoning.mutagenesis.queries import ElementCausesTerminalAtom
from typing_extensions import Sequence, Tuple

logger = logging.getLogger(__name__)

ELEMENTS_REPORTED = ("c", "h", "o")
"""
The elements to print, carbon and nitrogen being the ones chemistry says are never
terminal and hydrogen the one it says always is.
"""


@dataclass(frozen=True)
class Capacity:
    """
    One configuration of the network.
    """

    label: str
    """
    What to call it when printing.
    """

    hidden_layer_sizes: Tuple[int, ...]
    """
    The widths of the perceptron's hidden layers.
    """

    max_iterations: int
    """
    How many optimiser steps the perceptron may take.
    """


CAPACITIES = (
    Capacity("small", (64, 32), 1000),
    Capacity("large", (128, 64), 20000),
)
"""
The configuration the comparison uses, and a far larger one.
"""


def answers(
    examples: Sequence[object],
    capacity: Capacity,
    dropped_attributes: Sequence[str] = (),
) -> None:
    """
    Fit the estimator with one capacity and print what it answers.

    :param examples: The molecules to fit on.
    :param capacity: The configuration to fit with.
    :param dropped_attributes: Atom attributes to withhold on top of the ones the domain
        already withholds, for checking what the answer rests on.
    """
    domain = molecule_domain()
    if dropped_attributes:
        withheld = dict(domain.outcome_part_attributes)
        withheld["atoms"] = tuple(withheld.get("atoms", ())) + tuple(
            dropped_attributes
        )
        domain = replace(domain, outcome_part_attributes=withheld)
    baseline = NeuralAdjustmentBaseline(
        domain=domain,
        hidden_layer_sizes=capacity.hidden_layer_sizes,
        max_iterations=capacity.max_iterations,
    )
    baseline.fit(examples)
    outcome = baseline.ask(ElementCausesTerminalAtom())
    by_region = {effect.cause_region: effect for effect in outcome.effects}
    reported = ", ".join(
        f"{element} {by_region[element].adjusted_probability:.3f}"
        for element in ELEMENTS_REPORTED
        if element in by_region
    )
    print(
        f"{capacity.label:>5}  layers {capacity.hidden_layer_sizes}  "
        f"max_iter {capacity.max_iterations:>5}  "
        f"iterations run {baseline.last_iteration_count}  "
        f"adjusted {reported}"
    )


def main() -> None:
    """
    Fit both configurations on the first split and print their answers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--drop",
        nargs="+",
        default=[],
        metavar="ATTRIBUTE",
        help="atom attributes to withhold on top of the domain's own exclusions",
    )
    arguments = parser.parse_args()

    dataset = mutagenesis_dataset(fetch_mutagenesis_molecules())
    training, _ = dataset.split(
        arguments.train_fraction, np.random.default_rng(arguments.seed)
    )
    print(f"Fitted on {len(training.examples)} molecules, seed {arguments.seed}")
    if arguments.drop:
        print(f"Withholding on top of the domain's own: {', '.join(arguments.drop)}")
    for capacity in CAPACITIES:
        answers(training.examples, capacity, arguments.drop)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    main()
