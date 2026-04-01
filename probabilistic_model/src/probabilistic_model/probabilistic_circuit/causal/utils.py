from __future__ import annotations

import copy
from typing import Any, Set

from scipy.special import logsumexp

from random_events.variable import Variable

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)


def attach_marginal_circuit(
    marginal_circuit: ProbabilisticCircuit,
    target_product: ProductUnit,
    target_circuit: ProbabilisticCircuit,
) -> None:
    """
    Attach the root of marginal_circuit as a child of target_product,
    constructing fresh nodes owned by target_circuit.

    marginal() and log_truncated_in_place() return flat circuits
    (SumUnit -> leaves, or a single leaf), so one level of recursion suffices.

    :param marginal_circuit: The marginal or truncated circuit whose root to attach.
    :param target_product: The ProductUnit to attach the root as a child of.
    :param target_circuit: The owning circuit for all newly created nodes.
    """
    root = marginal_circuit.root
    if isinstance(root, SumUnit):
        new_sum_unit = SumUnit(probabilistic_circuit=target_circuit)
        for child_log_weight, child_subcircuit in root.log_weighted_subcircuits:
            new_sum_unit.add_subcircuit(
                leaf(copy.deepcopy(child_subcircuit.distribution), target_circuit),
                child_log_weight,
            )
        target_product.add_subcircuit(new_sum_unit)
    else:
        target_product.add_subcircuit(
            leaf(copy.deepcopy(root.distribution), target_circuit)
        )


def variables_of_simple_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a SimpleEvent.

    A SimpleEvent is a VariableMap — directly iterable as a dict via .keys().

    :param support_event: A SimpleEvent whose variable keys to extract.
    :returns: Set of Variable objects keyed in the event.
    """
    return set(support_event.keys())


def variables_of_composite_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a composite Event.

    A composite Event exposes .simple_sets, each of which is a SimpleEvent.
    Delegates per-SimpleEvent extraction to variables_of_simple_event.

    :param support_event: A composite Event whose variable keys to extract.
    :returns: Union of Variable keys across all simple sets.
    """
    variable_set: Set[Variable] = set()
    for simple_set in support_event.simple_sets:
        variable_set.update(variables_of_simple_event(simple_set))
    return variable_set


def variables_of_support_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys constrained by a support event.

    Routes to variables_of_simple_event or variables_of_composite_event
    based on whether the event exposes .simple_sets.

    :param support_event: A SimpleEvent or composite Event.
    :returns: Set of Variable objects constrained by the event.
    """
    if hasattr(support_event, "simple_sets"):
        return variables_of_composite_event(support_event)
    return variables_of_simple_event(support_event)


def sum_unit_is_normalized(sum_unit: SumUnit, tolerance: float = 1e-6) -> bool:
    """
    Return True iff the SumUnit's log-weights sum to log(1) == 0.

    Uses logsumexp for numerical stability, matching SumUnit.normalize().

    :param sum_unit: The SumUnit to check.
    :param tolerance: Maximum absolute deviation from 0.0 permitted.
    :returns: True if the weights are normalised within tolerance.
    """
    log_weights = sum_unit.log_weights
    if len(log_weights) == 0:
        return True
    return abs(float(logsumexp(log_weights))) < tolerance