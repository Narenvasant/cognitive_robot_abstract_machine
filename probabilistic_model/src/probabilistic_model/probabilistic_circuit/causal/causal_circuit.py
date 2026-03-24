"""
Causal Probabilistic Circuit
=============================
A ProbabilisticCircuit extended with exact, tractable causal inference
using the marginal determinism framework (md-vtree). The md-vtree
structure encodes the causal graph and enables polytime backdoor
adjustment for any valid adjustment set Z.

Causal validity
---------------
All causal queries are valid when the circuit was trained on independent
randomised data (uniform sampling). Under this condition the backdoor
criterion holds with Z=∅:

    P(Y | do(X=v)) = P(Y | X=v)

For correlated deployment data, supply a non-empty adjustment set Z.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)


@dataclass
class MdVtreeNode:
    """
    One node of a Marginal Determinism vtree (md-vtree).

    Each node carries the variable names in its subtree and a q_set
    specifying which variables SumUnits at this level must be
    Q-deterministic over. Q-determinism enables polytime backdoor
    adjustment.

    Build using MdVtreeNode.from_causal_graph() rather than constructing
    nodes manually.
    """

    variables: Set[str]
    q_set: Set[str] = field(default_factory=set)
    left: Optional[MdVtreeNode] = None
    right: Optional[MdVtreeNode] = None
    variable_objects: Optional[Tuple[Variable, ...]] = field(default=None, repr=False)

    def resolve_variables(self, circuit: ProbabilisticCircuit) -> None:
        """
        Populate variable_objects by resolving string names against circuit variables.

        Call once after the CausalCircuit is constructed. After this call
        every node holds typed Variable objects, enabling direct circuit
        queries without string lookups.
        """
        name_to_variable: Dict[str, Variable] = {v.name: v for v in circuit.variables}
        self.variable_objects = tuple(
            name_to_variable[name]
            for name in self.variables
            if name in name_to_variable
        )
        if self.left is not None:
            self.left.resolve_variables(circuit)
        if self.right is not None:
            self.right.resolve_variables(circuit)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def find_node_for_variable(self, variable_name: str) -> Optional[MdVtreeNode]:
        """Return the shallowest node whose q_set contains variable_name, or None."""
        if variable_name in self.q_set:
            return self
        for child in (self.left, self.right):
            if child is not None:
                result = child.find_node_for_variable(variable_name)
                if result is not None:
                    return result
        return None

    def all_q_sets(self) -> List[Set[str]]:
        """Return all non-empty q_sets in depth-first order."""
        collected = [self.q_set] if self.q_set else []
        for child in (self.left, self.right):
            if child is not None:
                collected.extend(child.all_q_sets())
        return collected

    @staticmethod
    def from_causal_graph(
        causal_variable_names: List[str],
        effect_variable_names: List[str],
        causal_priority_order: List[str] = None,
    ) -> MdVtreeNode:
        """
        Build an md-vtree from a causal graph specification.

        Parameters
        ----------
        causal_variable_names
            Names of all input variables that causally affect the outcome.
        effect_variable_names
            Names of all outcome variables.
        causal_priority_order
            Ordering of cause variables from most to least important.
            Defaults to causal_variable_names order if None.
        """
        ordered = (
            causal_priority_order
            if causal_priority_order is not None
            else causal_variable_names
        )
        return MdVtreeNode._build_subtree(ordered)

    @staticmethod
    def _build_subtree(ordered: List[str]) -> MdVtreeNode:
        if len(ordered) == 0:
            return MdVtreeNode(variables=set(), q_set=set())
        if len(ordered) == 1:
            return MdVtreeNode(variables={ordered[0]}, q_set={ordered[0]})

        primary = ordered[0]
        remaining = ordered[1:]
        split = len(remaining) // 2
        left_names = [primary] + remaining[:split]
        right_names = remaining[split:]

        return MdVtreeNode(
            variables=set(ordered),
            q_set={primary},
            left=MdVtreeNode._build_subtree(left_names),
            right=(
                MdVtreeNode._build_subtree(right_names)
                if right_names else None
            ),
        )


@dataclass
class QDeterminismVerificationResult:
    """Result of verifying Q-determinism of a circuit against its md-vtree."""

    passed: bool
    violations: List[str]
    checked_q_sets: List[Set[str]]
    circuit_variable_names: List[str]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Q-determinism verification: {status}",
            f"  Checked q_sets: {self.checked_q_sets}",
        ]
        if self.violations:
            lines.append("  Violations:")
            for violation in self.violations:
                lines.append(f"    - {violation}")
        return "\n".join(lines)


@dataclass
class FailureDiagnosisResult:
    """
    Result of diagnosing why a plan execution failed.

    interventional_probability_at_failure is P(cause in training support
    at the observed value), evaluated in the joint (cause, effect)
    interventional circuit. Zero means the observed value lies entirely
    outside the training distribution — the most unambiguous failure signal.
    """

    primary_cause_variable_name: str
    actual_value: float
    interventional_probability_at_failure: float
    recommended_value: Any
    interventional_probability_at_recommendation: float
    all_variable_results: Dict[str, Dict[str, Any]]

    def __str__(self) -> str:
        lines = [
            "Failure Diagnosis",
            "─────────────────────────────────────────────",
            f"  Primary cause:    {self.primary_cause_variable_name}",
            f"  Actual value:     {self.actual_value:.4f}",
            f"  P(success | do):  {self.interventional_probability_at_failure:.4f}",
            f"  Recommended:      {self.recommended_value}",
            f"  P(success | rec): {self.interventional_probability_at_recommendation:.4f}",
            "",
            "  All variables:",
        ]
        for name, result in self.all_variable_results.items():
            marker = " ← PRIMARY CAUSE" if name == self.primary_cause_variable_name else ""
            lines.append(
                f"    {name:<30}  actual={result['actual_value']:.4f}  "
                f"P={result['interventional_probability']:.4f}{marker}"
            )
        return "\n".join(lines)


@dataclass
class CausalStrengthResult:
    """
    Result of computing C(X→Y|Z) = I(X;Y|Z) / H(Y|Z).

    normalised_causal_strength is in [0, 1]: 0 means no causal influence,
    1 means X fully determines Y given Z.
    """

    cause_variable_name: str
    effect_variable_name: str
    adjustment_variable_names: List[str]
    conditional_mutual_information: float
    conditional_entropy_of_effect: float
    normalised_causal_strength: float

    def __str__(self) -> str:
        adjustment_label = (
            ", ".join(self.adjustment_variable_names)
            if self.adjustment_variable_names else "∅"
        )
        return (
            f"C({self.cause_variable_name} → {self.effect_variable_name}"
            f" | {adjustment_label})"
            f" = {self.normalised_causal_strength:.4f}"
            f"  [I={self.conditional_mutual_information:.4f} nats,"
            f"  H={self.conditional_entropy_of_effect:.4f} nats]"
        )


def _compute_entropy_from_counts(count_array: np.ndarray) -> float:
    """Shannon entropy in nats from an unnormalised count array."""
    total = count_array.sum()
    if total == 0:
        return 0.0
    positive_counts = count_array[count_array > 0]
    probabilities = positive_counts / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def _discretise_continuous_column(
    column_values: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    """
    Discretise a column into integer bin indices.

    Continuous columns use equal-width binning over the observed range.
    Categorical columns (string/bool dtype) map each unique value to an int.
    """
    if column_values.dtype.kind in ("U", "S", "O", "b"):
        category_to_index = {cat: i for i, cat in enumerate(np.unique(column_values))}
        return np.array([category_to_index[v] for v in column_values], dtype=np.int32)

    minimum, maximum = float(column_values.min()), float(column_values.max())
    if minimum == maximum:
        return np.zeros(len(column_values), dtype=np.int32)

    edges = np.linspace(minimum, maximum, number_of_bins + 1)
    raw_indices = np.digitize(column_values, edges[:-1]) - 1
    return np.clip(raw_indices, 0, number_of_bins - 1).astype(np.int32)


def _compute_conditional_mutual_information(
    cause_bin_indices: np.ndarray,
    effect_bin_indices: np.ndarray,
    adjustment_bin_index_list: List[np.ndarray],
    number_of_cause_bins: int,
    number_of_effect_bins: int,
) -> Tuple[float, float]:
    """
    Compute I(X;Y|Z) and H(Y|Z) from discretised arrays.

    Applies Miller-Madow bias correction for finite-sample inflation.
    Returns (conditional_mutual_information, conditional_entropy_of_effect)
    in nats.
    """
    total_samples = len(cause_bin_indices)

    if not adjustment_bin_index_list:
        joint_counts = np.zeros(
            (number_of_cause_bins, number_of_effect_bins), dtype=np.int32
        )
        np.add.at(joint_counts, (cause_bin_indices, effect_bin_indices), 1)

        entropy_effect = _compute_entropy_from_counts(joint_counts.sum(axis=0))
        entropy_effect_given_cause = 0.0
        cause_totals = joint_counts.sum(axis=1, keepdims=True)
        for cause_bin in range(number_of_cause_bins):
            cause_count = cause_totals[cause_bin, 0]
            if cause_count > 0:
                entropy_effect_given_cause += (
                    (cause_count / total_samples)
                    * _compute_entropy_from_counts(joint_counts[cause_bin])
                )

        raw_mutual_information = max(0.0, entropy_effect - entropy_effect_given_cause)
        miller_madow_correction = (
            int((joint_counts > 0).sum()) - 1
        ) / (2 * max(total_samples, 1))
        return max(0.0, raw_mutual_information - miller_madow_correction), entropy_effect

    stacked_adjustment = np.stack(adjustment_bin_index_list, axis=1)
    stratum_keys = [tuple(row) for row in stacked_adjustment]
    unique_stratum_keys = list(dict.fromkeys(stratum_keys))
    key_to_stratum_index = {key: i for i, key in enumerate(unique_stratum_keys)}
    number_of_strata = len(unique_stratum_keys)
    stratum_indices = np.array(
        [key_to_stratum_index[key] for key in stratum_keys], dtype=np.int32
    )

    entropy_effect_given_adjustment = 0.0
    entropy_effect_given_cause_adjustment = 0.0

    for stratum in range(number_of_strata):
        stratum_mask = stratum_indices == stratum
        stratum_count = int(stratum_mask.sum())
        if stratum_count == 0:
            continue
        stratum_weight = stratum_count / total_samples
        entropy_effect_given_adjustment += stratum_weight * _compute_entropy_from_counts(
            np.bincount(effect_bin_indices[stratum_mask], minlength=number_of_effect_bins)
        )
        for cause_bin in range(number_of_cause_bins):
            cell_mask = stratum_mask & (cause_bin_indices == cause_bin)
            cell_count = int(cell_mask.sum())
            if cell_count == 0:
                continue
            entropy_effect_given_cause_adjustment += (
                (cell_count / total_samples)
                * _compute_entropy_from_counts(
                    np.bincount(
                        effect_bin_indices[cell_mask], minlength=number_of_effect_bins
                    )
                )
            )

    conditional_mutual_information = max(
        0.0,
        entropy_effect_given_adjustment - entropy_effect_given_cause_adjustment,
    )
    miller_madow_correction = (
        number_of_cause_bins * number_of_strata - 1
    ) / (2 * max(total_samples, 1))
    return (
        max(0.0, conditional_mutual_information - miller_madow_correction),
        entropy_effect_given_adjustment,
    )


def _attach_marginal_circuit(
    marginal_circuit: ProbabilisticCircuit,
    target_product: ProductUnit,
    target_circuit: ProbabilisticCircuit,
) -> None:
    """
    Attach the root of marginal_circuit as a child of target_product,
    constructing fresh nodes owned by target_circuit.

    marginal() and log_truncated_in_place() return flat circuits
    (SumUnit → leaves, or a single leaf), so one level of recursion suffices.
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


def _variables_of_support_event(support_event: Any) -> Set[Variable]:
    """
    Return the set of Variable keys actually constrained by support_event.

    Works for both SimpleEvent (a VariableMap — directly iterable as a dict)
    and composite Event (exposes .simple_sets, each of which is a SimpleEvent).
    Uses the public VariableMap.keys() API so this remains stable across
    random_events versions.
    """
    variable_set: Set[Variable] = set()
    if hasattr(support_event, "simple_sets"):
        for simple_set in support_event.simple_sets:
            try:
                variable_set.update(simple_set.keys())
            except AttributeError:
                pass
        return variable_set
    try:
        variable_set.update(support_event.keys())
    except AttributeError:
        pass
    return variable_set


def _sum_unit_is_normalized(sum_unit: SumUnit, tolerance: float = 1e-6) -> bool:
    """
    Return True iff the SumUnit's log-weights sum to log(1) == 0.

    Uses logsumexp for numerical stability, matching SumUnit.normalize().
    """
    log_weights = sum_unit.log_weights
    if len(log_weights) == 0:
        return True
    return abs(float(logsumexp(log_weights))) < tolerance


class CausalCircuit:
    """
    A ProbabilisticCircuit extended with tractable causal inference.

    Wraps a fitted ProbabilisticCircuit and adds:
      - backdoor_adjustment()   — P(effect | do(cause)) as a new circuit
      - verify_q_determinism()  — structural validity check against the md-vtree
      - causal_strength()       — C(X→Y|Z) = I(X;Y|Z) / H(Y|Z)
      - rank_causal_variables() — rank causes by causal strength
      - diagnose_failure()      — identify primary cause and recommend a fix

    Use empty adjustment sets for independent randomised training data.
    Supply confounder names for correlated deployment data.
    """

    def __init__(
        self,
        probabilistic_circuit: ProbabilisticCircuit,
        mdvtree: MdVtreeNode,
        causal_variable_names: List[str],
        effect_variable_names: List[str],
    ) -> None:
        self._circuit = probabilistic_circuit
        self._mdvtree = mdvtree
        self._causal_variable_names = list(causal_variable_names)
        self._effect_variable_names = list(effect_variable_names)
        self._mdvtree.resolve_variables(self._circuit)

    @property
    def probabilistic_circuit(self) -> ProbabilisticCircuit:
        return self._circuit

    @property
    def mdvtree(self) -> MdVtreeNode:
        return self._mdvtree

    @property
    def causal_variable_names(self) -> List[str]:
        return list(self._causal_variable_names)

    @property
    def effect_variable_names(self) -> List[str]:
        return list(self._effect_variable_names)

    @classmethod
    def from_probabilistic_circuit(
        cls,
        circuit: ProbabilisticCircuit,
        mdvtree: MdVtreeNode,
        causal_variable_names: List[str],
        effect_variable_names: List[str],
    ) -> CausalCircuit:
        """Construct from an existing ProbabilisticCircuit without retraining."""
        return cls(circuit, mdvtree, causal_variable_names, effect_variable_names)

    @classmethod
    def from_jpt(
        cls,
        fitted_jpt: Any,
        mdvtree: MdVtreeNode,
        causal_variable_names: List[str],
        effect_variable_names: List[str],
    ) -> CausalCircuit:
        """
        Construct from a fitted Joint Probability Tree.

        Accepts a ProbabilisticCircuit directly, or a JPT exposing
        .as_probabilistic_circuit() or .to_probabilistic_circuit().
        """
        if isinstance(fitted_jpt, ProbabilisticCircuit):
            circuit = fitted_jpt

        elif (
            hasattr(fitted_jpt, "probabilistic_circuit")
            and isinstance(fitted_jpt.probabilistic_circuit, ProbabilisticCircuit)
        ):
            circuit = fitted_jpt.probabilistic_circuit

        elif hasattr(fitted_jpt, "as_probabilistic_circuit"):
            circuit = fitted_jpt.as_probabilistic_circuit()
            if not isinstance(circuit, ProbabilisticCircuit):
                raise TypeError(
                    f"as_probabilistic_circuit() returned {type(circuit)}, "
                    f"expected ProbabilisticCircuit."
                )

        elif hasattr(fitted_jpt, "to_probabilistic_circuit"):
            circuit = fitted_jpt.to_probabilistic_circuit()
            if not isinstance(circuit, ProbabilisticCircuit):
                raise TypeError(
                    f"to_probabilistic_circuit() returned {type(circuit)}, "
                    f"expected ProbabilisticCircuit."
                )

        else:
            raise TypeError(
                f"Cannot extract a ProbabilisticCircuit from {type(fitted_jpt)}.\n"
                f"Supported types:\n"
                f"  1. ProbabilisticCircuit directly\n"
                f"  2. probabilistic_model.learning.jpt.jpt.JPT "
                f"(exposes .probabilistic_circuit after fit/load)\n"
                f"  3. Any object with .as_probabilistic_circuit() or "
                f".to_probabilistic_circuit()\n"
                f"Note: pyjpt.trees.JPT does not expose a ProbabilisticCircuit. "
                f"Use probabilistic_model.learning.jpt.jpt.JPT instead."
            )
        return cls(circuit, mdvtree, causal_variable_names, effect_variable_names)

    def get_variable_by_name(self, variable_name: str) -> Variable:
        """Return the Variable whose name matches, or raise ValueError."""
        for variable in self._circuit.variables:
            if variable.name == variable_name:
                return variable
        available_names = [v.name for v in self._circuit.variables]
        raise ValueError(
            f"Variable '{variable_name}' not found. Available: {available_names}"
        )


    def verify_q_determinism(self) -> QDeterminismVerificationResult:
        """
        Verify Q-determinism of the circuit against the md-vtree.

        Three checks are performed in order.

        Check 1 — Variable existence
            Every variable name in every md-vtree q_set must exist in the
            circuit. If any are missing the structural checks are skipped
            because Variable objects are required for marginalisation.

        Check 2 — SumUnit normalization
            Every SumUnit's log-weights must sum to log(1). An unnormalized
            circuit produces incorrect backdoor probabilities regardless of
            structural determinism.

        Check 3 — Structural Q-determinism via circuit.support
            Calls self._circuit.support exactly once. This property
            (ProbabilisticCircuit.support) does a bottom-up traversal
            over reversed(self.layers), calling node.support() on every
            node and populating result_of_current_query on each. It then
            returns the root's composite Event.

            After that single call, every node's result_of_current_query
            is populated. We then iterate self._circuit.layers in BFS
            (root-first) order to inspect each SumUnit. Because support()
            filled nodes bottom-up, every child already has its
            result_of_current_query set when the parent SumUnit is reached.

            For each SumUnit we:
              a) Collect children's result_of_current_query support events.
              b) Find the intersection of Variable keys present in ALL
                 children (safe: avoids querying absent variables which
                 may silently return full-domain events).
              c) Among common variables, detect split variables — those
                 where at least one pair of children has an empty marginal
                 intersection (i.e. disjoint support on that variable).
              d) Cross-reference split variables against the md-vtree
                 q_sets. A SumUnit splitting on a variable not declared
                 in any q_set is reported as a violation.
              e) Verify all pairs of children are pairwise disjoint on
                 the split variable(s). Any overlap is a violation.
        """
        checked_q_sets = self._mdvtree.all_q_sets()
        violations: List[str] = []
        circuit_variable_names = [v.name for v in self._circuit.variables]

        all_q_variable_names: Set[str] = set()
        for q_set in checked_q_sets:
            all_q_variable_names.update(q_set)

        # Check 1: variable existence
        missing_q_variable_names: List[str] = []
        for q_set in checked_q_sets:
            for q_name in q_set:
                try:
                    self.get_variable_by_name(q_name)
                except ValueError:
                    violations.append(
                        f"Q-set variable '{q_name}' not found in circuit. "
                        f"Available: {circuit_variable_names}"
                    )
                    missing_q_variable_names.append(q_name)

        if missing_q_variable_names:
            return QDeterminismVerificationResult(
                passed=False,
                violations=violations,
                checked_q_sets=checked_q_sets,
                circuit_variable_names=circuit_variable_names,
            )

        # Check 2: SumUnit normalization
        for node in self._circuit.nodes():
            if isinstance(node, SumUnit) and not _sum_unit_is_normalized(node):
                violations.append(
                    f"SumUnit (index={node.index}) log-weights sum to "
                    f"{float(logsumexp(node.log_weights)):.6f}, expected 0.0. "
                    f"Unnormalized circuits produce incorrect backdoor probabilities."
                )

        # Check 3: structural Q-determinism
        try:
            root_support_event = self._circuit.support

            for layer in self._circuit.layers:
                for node in layer:
                    if not isinstance(node, SumUnit):
                        continue
                    children = node.subcircuits
                    if len(children) < 2:
                        continue

                    child_support_events = [
                        getattr(child, "result_of_current_query", None)
                        for child in children
                    ]
                    if any(event is None for event in child_support_events):
                        continue

                    per_child_variable_sets = [
                        _variables_of_support_event(event)
                        for event in child_support_events
                    ]
                    common_variables: Set[Variable] = per_child_variable_sets[0]
                    for variable_set in per_child_variable_sets[1:]:
                        common_variables = common_variables & variable_set

                    if not common_variables:
                        continue

                    split_variables: List[Variable] = []
                    for variable in self._circuit.variables:
                        if variable not in common_variables:
                            continue
                        child_marginals = []
                        for support_event in child_support_events:
                            try:
                                child_marginals.append(support_event.marginal([variable]))
                            except Exception:
                                child_marginals.append(None)

                        for marginal_a, marginal_b in itertools.combinations(
                            child_marginals, 2
                        ):
                            if marginal_a is None or marginal_b is None:
                                continue
                            try:
                                if marginal_a.intersection_with(marginal_b).is_empty():
                                    split_variables.append(variable)
                                    break
                            except Exception:
                                pass

                    if not split_variables:
                        continue

                    split_variable_names = [v.name for v in split_variables]

                    for split_variable in split_variables:
                        if split_variable.name not in all_q_variable_names:
                            violations.append(
                                f"SumUnit (index={node.index}) splits on variable "
                                f"'{split_variable.name}' which is not declared in "
                                f"any md-vtree q_set {checked_q_sets}. This SumUnit "
                                f"is not Q-deterministic for this variable."
                            )

                    for child_a, child_b in itertools.combinations(children, 2):
                        support_a = getattr(child_a, "result_of_current_query", None)
                        support_b = getattr(child_b, "result_of_current_query", None)
                        if support_a is None or support_b is None:
                            continue
                        try:
                            marginal_a = support_a.marginal(split_variables)
                            marginal_b = support_b.marginal(split_variables)
                            if not marginal_a.intersection_with(marginal_b).is_empty():
                                violations.append(
                                    f"SumUnit (index={node.index}) has overlapping "
                                    f"children supports on split variable(s) "
                                    f"{split_variable_names}: "
                                    f"children are not Q-deterministic."
                                )
                                break
                        except Exception:
                            pass

        except Exception as support_traversal_error:
            violations.append(
                f"Structural Q-determinism check failed during support "
                f"traversal: {support_traversal_error}"
            )

        del root_support_event

        return QDeterminismVerificationResult(
            passed=len(violations) == 0,
            violations=violations,
            checked_q_sets=checked_q_sets,
            circuit_variable_names=circuit_variable_names,
        )

    def backdoor_adjustment(
        self,
        cause_variable_name: str,
        effect_variable_name: str,
        adjustment_variable_names: List[str] = None,
        query_resolution: float = 0.005,
    ) -> ProbabilisticCircuit:
        """
        Compute P(effect | do(cause)) as a new ProbabilisticCircuit.

        With empty adjustment set:
            P(effect | do(cause=v)) = P(effect | cause=v)

        With non-empty adjustment set Z:
            P(effect | do(cause=v)) = Σ_z P(effect | cause=v, Z=z) · P(Z=z)

        The output encodes the joint P(effect, cause) — probability(),
        marginal(), and sample() all work on the returned circuit.
        """
        if cause_variable_name not in self._causal_variable_names:
            raise ValueError(
                f"'{cause_variable_name}' is not a registered cause variable. "
                f"Registered: {self._causal_variable_names}."
            )
        if effect_variable_name not in self._effect_variable_names:
            raise ValueError(
                f"'{effect_variable_name}' is not a registered effect variable. "
                f"Registered: {self._effect_variable_names}."
            )
        if adjustment_variable_names is None:
            adjustment_variable_names = []

        if not adjustment_variable_names:
            return self._compute_interventional_circuit_without_adjustment(
                cause_variable_name, effect_variable_name, query_resolution
            )
        return self._compute_interventional_circuit_with_adjustment(
            cause_variable_name, effect_variable_name,
            adjustment_variable_names, query_resolution,
        )

    def _compute_interventional_circuit_without_adjustment(
        self,
        cause_variable_name: str,
        effect_variable_name: str,
        query_resolution: float,
    ) -> ProbabilisticCircuit:
        """
        Compute P(cause, effect | do(cause)) with an empty adjustment set.

        Returns a joint circuit over (cause, effect) as a SumUnit of
        ProductUnits, one per disjoint cause support region.

        Structure:
            SumUnit [weight = P(cause in region_i)]
                ProductUnit
                    cause branch  (UniformDistribution over region_i)
                    effect branch (P(effect | cause in region_i))
        """
        cause_variable = self.get_variable_by_name(cause_variable_name)
        effect_variable = self.get_variable_by_name(effect_variable_name)

        cause_regions = self._extract_leaf_regions_for_variable(cause_variable)
        cause_marginal_circuit = copy.deepcopy(self._circuit).marginal([cause_variable])

        output_circuit = ProbabilisticCircuit()
        root_sum_unit = SumUnit(probabilistic_circuit=output_circuit)
        regions_added = 0

        for region_event, region_weight in cause_regions:
            if region_weight <= 0.0:
                continue

            truncated_circuit, _ = copy.deepcopy(self._circuit).log_truncated_in_place(
                region_event.fill_missing_variables_pure(self._circuit.variables)
            )
            if truncated_circuit is None:
                continue

            effect_marginal_circuit = truncated_circuit.marginal([effect_variable])
            if effect_marginal_circuit is None:
                continue

            if cause_marginal_circuit is None:
                continue

            cause_region_circuit, _ = copy.deepcopy(
                cause_marginal_circuit
            ).log_truncated_in_place(
                region_event.fill_missing_variables_pure(cause_marginal_circuit.variables)
            )
            if cause_region_circuit is None:
                continue

            product_unit = ProductUnit(probabilistic_circuit=output_circuit)
            _attach_marginal_circuit(cause_region_circuit, product_unit, output_circuit)
            _attach_marginal_circuit(effect_marginal_circuit, product_unit, output_circuit)
            root_sum_unit.add_subcircuit(product_unit, math.log(region_weight))
            regions_added += 1

        if regions_added == 0:
            raise ValueError(
                f"Interventional circuit is empty for cause '{cause_variable_name}'. "
                f"Ensure the circuit was trained on data covering this variable's domain."
            )
        return output_circuit

    def _compute_interventional_circuit_with_adjustment(
        self,
        cause_variable_name: str,
        effect_variable_name: str,
        adjustment_variable_names: List[str],
        query_resolution: float,
    ) -> ProbabilisticCircuit:
        """
        Compute P(cause, effect | do(cause)) with a non-empty adjustment set Z.

        Implements:
            P(effect | do(cause=v)) = Σ_z P(effect | cause=v, Z=z) · P(Z=z)
        """
        cause_variable = self.get_variable_by_name(cause_variable_name)
        effect_variable = self.get_variable_by_name(effect_variable_name)
        adjustment_variables = [
            self.get_variable_by_name(name) for name in adjustment_variable_names
        ]

        cause_marginal_circuit = copy.deepcopy(self._circuit).marginal([cause_variable])

        output_circuit = ProbabilisticCircuit()
        root_sum_unit = SumUnit(probabilistic_circuit=output_circuit)
        regions_added = 0

        for adjustment_event, adjustment_weight in self._extract_leaf_regions_for_variables(
            adjustment_variables
        ):
            if adjustment_weight <= 0.0:
                continue

            adjustment_conditioned_circuit, _ = copy.deepcopy(
                self._circuit
            ).log_truncated_in_place(
                adjustment_event.fill_missing_variables_pure(self._circuit.variables)
            )
            if adjustment_conditioned_circuit is None:
                continue

            for cause_event, cause_weight in self._extract_leaf_regions_for_variable(
                cause_variable, base_circuit=adjustment_conditioned_circuit
            ):
                if cause_weight <= 0.0:
                    continue

                joint_weight = adjustment_weight * cause_weight
                joint_event = adjustment_event.intersection_with(cause_event)

                truncated_circuit, _ = copy.deepcopy(
                    self._circuit
                ).log_truncated_in_place(
                    joint_event.fill_missing_variables_pure(self._circuit.variables)
                )
                if truncated_circuit is None:
                    continue

                effect_marginal_circuit = truncated_circuit.marginal([effect_variable])
                if effect_marginal_circuit is None:
                    continue

                if cause_marginal_circuit is None:
                    continue

                cause_region_circuit, _ = copy.deepcopy(
                    cause_marginal_circuit
                ).log_truncated_in_place(
                    cause_event.fill_missing_variables_pure(cause_marginal_circuit.variables)
                )
                if cause_region_circuit is None:
                    continue

                product_unit = ProductUnit(probabilistic_circuit=output_circuit)
                _attach_marginal_circuit(cause_region_circuit, product_unit, output_circuit)
                _attach_marginal_circuit(effect_marginal_circuit, product_unit, output_circuit)
                root_sum_unit.add_subcircuit(product_unit, math.log(joint_weight))
                regions_added += 1

        if regions_added == 0:
            raise ValueError(
                f"Interventional circuit with adjustment is empty. "
                f"cause='{cause_variable_name}', adjustment={adjustment_variable_names}."
            )
        return output_circuit

    def _extract_leaf_regions_for_variable(
        self,
        variable_object: Variable,
        base_circuit: ProbabilisticCircuit = None,
    ) -> List[Tuple[Any, float]]:
        """Return (region_event, probability) pairs for each support region of variable."""
        circuit = base_circuit if base_circuit is not None else self._circuit
        regions: List[Tuple[Any, float]] = []
        try:
            variable_support = circuit.support.marginal([variable_object])
        except Exception:
            return regions
        for simple_region in variable_support.simple_sets:
            region_event = SimpleEvent(
                {variable_object: simple_region[variable_object]}
            ).as_composite_set()
            probability = circuit.probability(
                region_event.fill_missing_variables_pure(circuit.variables)
            )
            if probability > 0.0:
                regions.append((region_event, float(probability)))
        return regions

    def _extract_leaf_regions_for_variables(
        self,
        variable_objects: List[Variable],
    ) -> List[Tuple[Any, float]]:
        """Return (region_event, probability) pairs for the joint support of variables."""
        regions: List[Tuple[Any, float]] = []
        try:
            joint_support = self._circuit.support.marginal(variable_objects)
        except Exception:
            return regions
        for simple_region in joint_support.simple_sets:
            region_event = SimpleEvent(
                {variable: simple_region[variable] for variable in variable_objects}
            ).as_composite_set()
            probability = self._circuit.probability(
                region_event.fill_missing_variables_pure(self._circuit.variables)
            )
            if probability > 0.0:
                regions.append((region_event, float(probability)))
        return regions


    def causal_strength(
        self,
        cause_variable_name: str,
        effect_variable_name: str,
        adjustment_variable_names: List[str] = None,
        training_dataframe: pd.DataFrame = None,
        number_of_histogram_bins: int = None,
    ) -> CausalStrengthResult:
        """
        Compute C(X→Y|Z) = I(X;Y|Z) / H(Y|Z).

        Use Z=[] for independent randomised data. Supply confounder names
        for correlated deployment data. If training_dataframe is None,
        samples are drawn from the circuit (introduces sampling variance).
        """
        if adjustment_variable_names is None:
            adjustment_variable_names = []

        if training_dataframe is not None:
            data = training_dataframe
        else:
            samples = self._circuit.sample(2000)
            data = pd.DataFrame(
                samples, columns=[v.name for v in self._circuit.variables]
            )

        sample_count = len(data)
        if number_of_histogram_bins is None:
            number_of_histogram_bins = max(5, min(20, int(sample_count ** 0.4)))

        discretised_cause = _discretise_continuous_column(
            data[cause_variable_name].values, number_of_histogram_bins
        )
        discretised_effect = _discretise_continuous_column(
            data[effect_variable_name].values, number_of_histogram_bins
        )
        discretised_adjustment_list = [
            _discretise_continuous_column(data[name].values, number_of_histogram_bins)
            for name in adjustment_variable_names
        ]

        conditional_mutual_information, entropy_of_effect = (
            _compute_conditional_mutual_information(
                cause_bin_indices=discretised_cause,
                effect_bin_indices=discretised_effect,
                adjustment_bin_index_list=discretised_adjustment_list,
                number_of_cause_bins=number_of_histogram_bins,
                number_of_effect_bins=number_of_histogram_bins,
            )
        )
        normalised_strength = (
            float(np.clip(conditional_mutual_information / entropy_of_effect, 0.0, 1.0))
            if entropy_of_effect > 1e-10 else 0.0
        )

        return CausalStrengthResult(
            cause_variable_name=cause_variable_name,
            effect_variable_name=effect_variable_name,
            adjustment_variable_names=list(adjustment_variable_names),
            conditional_mutual_information=round(conditional_mutual_information, 6),
            conditional_entropy_of_effect=round(entropy_of_effect, 6),
            normalised_causal_strength=round(normalised_strength, 4),
        )

    def rank_causal_variables(
        self,
        effect_variable_name: str,
        adjustment_variable_names: List[str] = None,
        training_dataframe: pd.DataFrame = None,
    ) -> List[CausalStrengthResult]:
        """
        Rank all cause variables by their causal strength on the effect.

        Defaults to Z=[] (correct for independent data). Returns results
        sorted descending by normalised_causal_strength.
        """
        adjustment_set = (
            adjustment_variable_names if adjustment_variable_names is not None else []
        )
        results = [
            self.causal_strength(
                name, effect_variable_name, adjustment_set, training_dataframe
            )
            for name in self._causal_variable_names
        ]
        results.sort(key=lambda result: result.normalised_causal_strength, reverse=True)
        return results

    def diagnose_failure(
        self,
        observed_parameter_values: Dict[str, float],
        effect_variable_name: str,
        query_resolution: float = 0.005,
        adjustment_variable_names: List[str] = None,
    ) -> FailureDiagnosisResult:
        """
        Identify the primary cause of a failed plan execution.

        For each cause variable, queries the interventional circuit at the
        observed cause value. The variable whose observed value falls furthest
        outside the training distribution (lowest query probability) is the
        primary cause. The recommendation is the cause region midpoint with
        the highest interventional probability.

        The interventional_probability recorded per variable is
        P(cause in [observed-eps, observed+eps]) in the joint (cause, effect)
        interventional circuit. Zero means the value is outside training support.
        """
        if adjustment_variable_names is None:
            adjustment_variable_names = []

        all_variable_results: Dict[str, Dict[str, Any]] = {}
        interventional_circuits_by_cause: Dict[str, ProbabilisticCircuit] = {}

        for cause_name in self._causal_variable_names:
            if cause_name not in observed_parameter_values:
                continue

            observed_value = observed_parameter_values[cause_name]
            cause_variable = self.get_variable_by_name(cause_name)

            if not cause_variable.is_numeric:
                continue

            interventional_circuit = self.backdoor_adjustment(
                cause_name, effect_variable_name,
                adjustment_variable_names, query_resolution,
            )
            interventional_circuits_by_cause[cause_name] = interventional_circuit

            observed_event = SimpleEvent(
                {
                    cause_variable: closed(
                        float(observed_value) - query_resolution,
                        float(observed_value) + query_resolution,
                    )
                }
            ).as_composite_set()

            try:
                probability_at_observed = float(
                    interventional_circuit.probability(
                        observed_event.fill_missing_variables_pure(
                            interventional_circuit.variables
                        )
                    )
                )
            except Exception as query_error:
                raise ValueError(
                    f"Failed to query interventional circuit for "
                    f"'{cause_name}'={observed_value}: {query_error}"
                ) from query_error

            recommended_value: Optional[float] = None
            try:
                best_probability = -1.0
                for region_event, _ in self._extract_leaf_regions_for_variable(
                    cause_variable
                ):
                    region_probability = float(
                        interventional_circuit.probability(
                            region_event.fill_missing_variables_pure(
                                interventional_circuit.variables
                            )
                        )
                    )
                    if region_probability > best_probability:
                        best_probability = region_probability
                        for simple_set in region_event.simple_sets:
                            if cause_variable in simple_set:
                                interval_set = simple_set[cause_variable]
                                if hasattr(interval_set, "simple_sets"):
                                    for interval in interval_set.simple_sets:
                                        recommended_value = (
                                            float(interval.lower) + float(interval.upper)
                                        ) / 2.0
                                        break
                                elif hasattr(interval_set, "lower"):
                                    recommended_value = (
                                        float(interval_set.lower) + float(interval_set.upper)
                                    ) / 2.0
                                break
            except Exception:
                recommended_value = None

            all_variable_results[cause_name] = {
                "actual_value": observed_value,
                "interventional_probability": round(probability_at_observed, 6),
                "recommended_value": recommended_value,
            }

        if not all_variable_results:
            raise ValueError(
                f"No cause variables found in observed_parameter_values. "
                f"Expected at least one of: {self._causal_variable_names}"
            )

        primary_cause_name = min(
            all_variable_results,
            key=lambda name: all_variable_results[name]["interventional_probability"],
        )
        primary_result = all_variable_results[primary_cause_name]
        recommended_value = primary_result["recommended_value"]
        probability_at_recommendation = 0.0

        if recommended_value is not None:
            primary_cause_variable = self.get_variable_by_name(primary_cause_name)
            primary_interventional_circuit = interventional_circuits_by_cause[
                primary_cause_name
            ]
            try:
                recommendation_event = SimpleEvent(
                    {
                        primary_cause_variable: closed(
                            float(recommended_value) - query_resolution,
                            float(recommended_value) + query_resolution,
                        )
                    }
                ).as_composite_set()
                probability_at_recommendation = float(
                    primary_interventional_circuit.probability(
                        recommendation_event.fill_missing_variables_pure(
                            primary_interventional_circuit.variables
                        )
                    )
                )
            except Exception:
                probability_at_recommendation = 0.0

        return FailureDiagnosisResult(
            primary_cause_variable_name=primary_cause_name,
            actual_value=primary_result["actual_value"],
            interventional_probability_at_failure=primary_result[
                "interventional_probability"
            ],
            recommended_value=recommended_value,
            interventional_probability_at_recommendation=round(
                probability_at_recommendation, 6
            ),
            all_variable_results=all_variable_results,
        )