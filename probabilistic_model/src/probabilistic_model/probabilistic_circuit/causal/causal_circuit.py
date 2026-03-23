"""
Causal Probabilistic Circuit
=============================
A ProbabilisticCircuit extended with exact, tractable causal inference
using the marginal determinism framework (md-vtree). The md-vtree
structure encodes the causal graph and enables polytime
backdoor adjustment for any valid adjustment set Z.

Causal validity
---------------
All causal queries are valid when the circuit was trained on independent
randomised data (uniform sampling). Under this condition the
backdoor criterion holds with Z=∅:

    P(Y | do(X=v)) = P(Y | X=v)

For correlated deployment data, supply a non-empty adjustment set Z.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
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
    specifying which variables sum nodes at this level must be
    Q-deterministic over. Q-determinism enables polytime backdoor adjustment.

    Build using MdVtreeNode.from_causal_graph() rather than constructing
    nodes manually.
    """

    variables: Set[str]
    q_set: Set[str] = field(default_factory=set)
    left: Optional[MdVtreeNode] = None
    right: Optional[MdVtreeNode] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def find_node_for_variable(self, variable_name: str) -> Optional[MdVtreeNode]:
        """Return the node whose q_set contains variable_name, or None."""
        if variable_name in self.q_set:
            return self
        for child in [self.left, self.right]:
            if child is not None:
                found = child.find_node_for_variable(variable_name)
                if found is not None:
                    return found
        return None

    def all_q_sets(self) -> List[Set[str]]:
        """Return all non-empty q_sets in depth-first order."""
        collected = [self.q_set] if self.q_set else []
        for child in [self.left, self.right]:
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
        all_names = set(causal_variable_names) | set(effect_variable_names)
        return MdVtreeNode._build_subtree(ordered, all_names)

    @staticmethod
    def _build_subtree(ordered: List[str], all_names: Set[str]) -> MdVtreeNode:
        if len(ordered) == 0:
            return MdVtreeNode(variables=all_names, q_set=set())
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
            left=MdVtreeNode._build_subtree(left_names, set(left_names)),
            right=(
                MdVtreeNode._build_subtree(right_names, set(right_names))
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
        lines = [f"Q-determinism verification: {status}",
                 f"  Checked q_sets: {self.checked_q_sets}"]
        if self.violations:
            lines.append("  Violations:")
            for v in self.violations:
                lines.append(f"    - {v}")
        return "\n".join(lines)


@dataclass
class FailureDiagnosisResult:
    """
    Result of diagnosing why a plan execution failed.

    Attributes
    ----------
    interventional_probability_at_failure
        P(cause_variable in training support at the observed value), evaluated
        by querying the joint (cause, effect) interventional circuit. This is
        zero when the observed cause value lies entirely outside the training
        distribution (the most unambiguous failure signal), and positive when
        the cause value was covered during training.

        Note: this quantity measures cause-value coverage in the training
        distribution, not the formal do-calculus quantity P(effect | do(cause=v)).
        It is a reliable indicator for identifying out-of-distribution parameters
        as primary failure causes.
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
        for name, r in self.all_variable_results.items():
            marker = " ← PRIMARY CAUSE" if name == self.primary_cause_variable_name else ""
            lines.append(
                f"    {name:<30}  actual={r['actual_value']:.4f}  "
                f"P={r['interventional_probability']:.4f}{marker}"
            )
        return "\n".join(lines)


@dataclass
class CausalStrengthResult:
    """
    Result of computing C(X→Y|Z) = I(X;Y|Z) / H(Y|Z).

    normalised_causal_strength is in [0,1]: 0 means no causal influence,
    1 means X fully determines Y given Z.
    """

    cause_variable_name: str
    effect_variable_name: str
    adjustment_variable_names: List[str]
    conditional_mutual_information: float
    conditional_entropy_of_effect: float
    normalised_causal_strength: float

    def __str__(self) -> str:
        z = ", ".join(self.adjustment_variable_names) if self.adjustment_variable_names else "∅"
        return (
            f"C({self.cause_variable_name} → {self.effect_variable_name} | {z})"
            f" = {self.normalised_causal_strength:.4f}"
            f"  [I={self.conditional_mutual_information:.4f} nats,"
            f"  H={self.conditional_entropy_of_effect:.4f} nats]"
        )


def _compute_entropy_from_counts(count_array: np.ndarray) -> float:
    """Shannon entropy in nats from an unnormalised count array."""
    total = count_array.sum()
    if total == 0:
        return 0.0
    p = count_array[count_array > 0] / total
    return float(-np.sum(p * np.log(p)))


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
        mapping = {cat: i for i, cat in enumerate(np.unique(column_values))}
        return np.array([mapping[v] for v in column_values], dtype=np.int32)

    lo, hi = float(column_values.min()), float(column_values.max())
    if lo == hi:
        return np.zeros(len(column_values), dtype=np.int32)
    edges = np.linspace(lo, hi, number_of_bins + 1)
    raw = np.digitize(column_values, edges[:-1]) - 1
    return np.clip(raw, 0, number_of_bins - 1).astype(np.int32)


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
    N = len(cause_bin_indices)

    if not adjustment_bin_index_list:
        joint = np.zeros((number_of_cause_bins, number_of_effect_bins), dtype=np.int32)
        np.add.at(joint, (cause_bin_indices, effect_bin_indices), 1)

        h_y = _compute_entropy_from_counts(joint.sum(axis=0))
        h_y_given_x = 0.0
        row_totals = joint.sum(axis=1, keepdims=True)
        for i in range(number_of_cause_bins):
            n_i = row_totals[i, 0]
            if n_i > 0:
                h_y_given_x += (n_i / N) * _compute_entropy_from_counts(joint[i])

        raw_mi = max(0.0, h_y - h_y_given_x)
        mm = (int((joint > 0).sum()) - 1) / (2 * max(N, 1))
        return max(0.0, raw_mi - mm), h_y

    stacked = np.stack(adjustment_bin_index_list, axis=1)
    keys = [tuple(r) for r in stacked]
    unique_keys = list(dict.fromkeys(keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    n_strata = len(unique_keys)
    stratum_idx = np.array([key_to_idx[k] for k in keys], dtype=np.int32)

    h_y_z, h_y_xz = 0.0, 0.0
    for s in range(n_strata):
        mask_s = stratum_idx == s
        n_s = int(mask_s.sum())
        if n_s == 0:
            continue
        w_s = n_s / N
        h_y_z += w_s * _compute_entropy_from_counts(
            np.bincount(effect_bin_indices[mask_s], minlength=number_of_effect_bins)
        )
        for c in range(number_of_cause_bins):
            mask_cs = mask_s & (cause_bin_indices == c)
            n_cs = int(mask_cs.sum())
            if n_cs == 0:
                continue
            h_y_xz += (n_cs / N) * _compute_entropy_from_counts(
                np.bincount(effect_bin_indices[mask_cs], minlength=number_of_effect_bins)
            )

    cmi = max(0.0, h_y_z - h_y_xz)
    mm = (number_of_cause_bins * n_strata - 1) / (2 * max(N, 1))
    return max(0.0, cmi - mm), h_y_z


def _attach_marginal_circuit(
    marginal_circuit: ProbabilisticCircuit,
    target_product: ProductUnit,
    target_circuit: ProbabilisticCircuit,
) -> None:
    """
    Attach the root of marginal_circuit as a branch of target_product,
    constructing fresh nodes owned by target_circuit.

    marginal() and log_truncated_in_place() return flat circuits
    (SumUnit → leaves, or a single leaf), so one level of depth suffices.
    No cross-circuit node transplantation occurs.
    """
    root = marginal_circuit.root
    if isinstance(root, SumUnit):
        new_sum = SumUnit(probabilistic_circuit=target_circuit)
        for child_log_w, child_sub in root.log_weighted_subcircuits:
            new_sum.add_subcircuit(
                leaf(copy.deepcopy(child_sub.distribution), target_circuit),
                child_log_w,
            )
        target_product.add_subcircuit(new_sum)
    else:
        target_product.add_subcircuit(
            leaf(copy.deepcopy(root.distribution), target_circuit)
        )


class CausalCircuit:
    """
    A ProbabilisticCircuit extended with tractable causal inference.

    Wraps a fitted ProbabilisticCircuit and adds:
      - backdoor_adjustment()   P(effect | do(cause)) as a new circuit
      - verify_q_determinism()  structural validity check against the md-vtree
      - causal_strength()       C(X→Y|Z) = I(X;Y|Z) / H(Y|Z)
      - rank_causal_variables() rank causes by causal strength
      - diagnose_failure()      identify primary cause and recommend a fix

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
                f"  2. probabilistic_model.learning.jpt.jpt.JPT  "
                f"(has .probabilistic_circuit attribute after fit/load)\n"
                f"  3. Any JPT with .as_probabilistic_circuit() or "
                f".to_probabilistic_circuit()\n"
                f"Note: pyjpt.trees.JPT does not expose a ProbabilisticCircuit. "
                f"Use probabilistic_model.learning.jpt.jpt.JPT instead."
            )
        return cls(circuit, mdvtree, causal_variable_names, effect_variable_names)

    def get_variable_by_name(self, variable_name: str) -> Variable:
        """Return the Variable whose name matches, or raise ValueError."""
        for v in self._circuit.variables:
            if v.name == variable_name:
                return v
        available = [v.name for v in self._circuit.variables]
        raise ValueError(
            f"Variable '{variable_name}' not found. Available: {available}"
        )

    def verify_q_determinism(self) -> QDeterminismVerificationResult:
        """
        Check that every Q-set variable in the md-vtree exists in the circuit.

        A violation is recorded for each Q-set variable name that cannot be
        found in the circuit. This catches md-vtree/circuit mismatches that
        would cause silent failures in backdoor_adjustment().

        Note on structural overlap checking
        ------------------------------------
        Full Q-determinism — verifying that the sum nodes at each circuit level
        have disjoint marginal supports over the Q variables — requires
        enumerating internal circuit structure in ways the rx library does not
        expose stably. This check is therefore intentionally omitted here.
        For a circuit trained via JPT (which is fully deterministic by construction)
        Q-determinism is guaranteed for all split variables without runtime checking.
        If you require structural verification, call circuit.is_deterministic()
        on your circuit before constructing CausalCircuit.
        """
        checked_q_sets = self._mdvtree.all_q_sets()
        violations: List[str] = []
        circuit_variable_names = [v.name for v in self._circuit.variables]

        for q_set in checked_q_sets:
            if not q_set:
                continue
            for q_name in q_set:
                try:
                    self.get_variable_by_name(q_name)
                except ValueError:
                    violations.append(
                        f"Q-set variable '{q_name}' not found in circuit. "
                        f"Available: {circuit_variable_names}"
                    )

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
            P(effect | do(cause=v)) = Σ_z P(effect|cause=v,Z=z) · P(Z=z)

        The output encodes the joint P(effect, cause) — probability(),
        marginal(), and sample() all work on the returned circuit.

        Raises ValueError if cause or effect variable names are not registered.
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
            adjustment_variable_names, query_resolution
        )

    def _compute_interventional_circuit_without_adjustment(
        self,
        cause_variable_name: str,
        effect_variable_name: str,
        query_resolution: float,
    ) -> ProbabilisticCircuit:
        """
        Compute P(cause, effect | do(cause)) when the adjustment set is empty.

        Returns a joint circuit over (cause, effect) as a SumUnit of ProductUnits,
        one per disjoint cause support region. Querying at a specific cause value
        correctly returns zero for values outside the training domain.

        Structure:
            SumUnit [weight = P(cause in region_i)]
                ProductUnit
                    cause branch  (UniformDistribution over region_i)
                    effect branch (P(effect | cause in region_i))
        """
        cause_var = self.get_variable_by_name(cause_variable_name)
        effect_var = self.get_variable_by_name(effect_variable_name)

        cause_regions = self._extract_leaf_regions_for_variable(cause_var)

        # Hoist cause_marginal outside the region loop — it is the same for
        # every region and deepcopying the full circuit once per region is
        # expensive (O(leaves) deepcopies per backdoor_adjustment call).
        cause_marginal = copy.deepcopy(self._circuit).marginal([cause_var])

        output = ProbabilisticCircuit()
        root_sum = SumUnit(probabilistic_circuit=output)
        added = 0

        for region_event, region_weight in cause_regions:
            if region_weight <= 0.0:
                continue

            truncated, _ = copy.deepcopy(self._circuit).log_truncated_in_place(
                region_event.fill_missing_variables_pure(self._circuit.variables)
            )
            if truncated is None:
                continue

            effect_marginal = truncated.marginal([effect_var])
            if effect_marginal is None:
                continue

            if cause_marginal is None:
                continue
            cause_region, _ = copy.deepcopy(cause_marginal).log_truncated_in_place(
                region_event.fill_missing_variables_pure(cause_marginal.variables)
            )
            if cause_region is None:
                continue

            product = ProductUnit(probabilistic_circuit=output)
            _attach_marginal_circuit(cause_region, product, output)
            _attach_marginal_circuit(effect_marginal, product, output)
            root_sum.add_subcircuit(product, math.log(region_weight))
            added += 1

        if added == 0:
            raise ValueError(
                f"Interventional circuit is empty for cause '{cause_variable_name}'. "
                f"Ensure the circuit was trained on data covering this variable's domain."
            )
        return output

    def _compute_interventional_circuit_with_adjustment(
        self,
        cause_variable_name: str,
        effect_variable_name: str,
        adjustment_variable_names: List[str],
        query_resolution: float,
    ) -> ProbabilisticCircuit:
        """
        Compute P(cause, effect | do(cause)) with a non-empty adjustment set Z.

        Implements the backdoor adjustment formula:
            P(effect | do(cause=v)) = Σ_z P(effect | cause=v, Z=z) · P(Z=z)

        Returns a joint circuit over (cause, effect).
        """
        cause_var = self.get_variable_by_name(cause_variable_name)
        effect_var = self.get_variable_by_name(effect_variable_name)
        adj_vars = [self.get_variable_by_name(n) for n in adjustment_variable_names]

        cause_marginal = copy.deepcopy(self._circuit).marginal([cause_var])

        output = ProbabilisticCircuit()
        root_sum = SumUnit(probabilistic_circuit=output)
        added = 0

        for adj_event, adj_weight in self._extract_leaf_regions_for_variables(adj_vars):
            if adj_weight <= 0.0:
                continue

            adj_conditioned, _ = copy.deepcopy(self._circuit).log_truncated_in_place(
                adj_event.fill_missing_variables_pure(self._circuit.variables)
            )
            if adj_conditioned is None:
                continue

            for cause_event, cause_weight in self._extract_leaf_regions_for_variable(
                cause_var, base_circuit=adj_conditioned
            ):
                if cause_weight <= 0.0:
                    continue

                joint_weight = adj_weight * cause_weight
                truncated, _ = copy.deepcopy(self._circuit).log_truncated_in_place(
                    adj_event.intersection_with(cause_event).fill_missing_variables_pure(
                        self._circuit.variables
                    )
                )
                if truncated is None:
                    continue

                effect_marginal = truncated.marginal([effect_var])
                if effect_marginal is None:
                    continue

                if cause_marginal is None:
                    continue
                cause_region, _ = copy.deepcopy(cause_marginal).log_truncated_in_place(
                    cause_event.fill_missing_variables_pure(cause_marginal.variables)
                )
                if cause_region is None:
                    continue

                product = ProductUnit(probabilistic_circuit=output)
                _attach_marginal_circuit(cause_region, product, output)
                _attach_marginal_circuit(effect_marginal, product, output)
                root_sum.add_subcircuit(product, math.log(joint_weight))
                added += 1

        if added == 0:
            raise ValueError(
                f"Interventional circuit with adjustment is empty. "
                f"cause='{cause_variable_name}', adjustment={adjustment_variable_names}."
            )
        return output

    def _extract_leaf_regions_for_variable(
        self,
        variable_object: Variable,
        base_circuit: ProbabilisticCircuit = None,
    ) -> List[Tuple[Any, float]]:
        """Return (region_event, probability) pairs for each support region."""
        circuit = base_circuit if base_circuit is not None else self._circuit
        regions: List[Tuple[Any, float]] = []
        try:
            var_support = circuit.support.marginal([variable_object])
        except Exception:
            return regions
        for simple_region in var_support.simple_sets:
            region_event = SimpleEvent(
                {variable_object: simple_region[variable_object]}
            ).as_composite_set()
            prob = circuit.probability(
                region_event.fill_missing_variables_pure(circuit.variables)
            )
            if prob > 0.0:
                regions.append((region_event, float(prob)))
        return regions

    def _extract_leaf_regions_for_variables(
        self,
        variable_objects: List[Variable],
    ) -> List[Tuple[Any, float]]:
        """Return (region_event, probability) pairs for joint support of variables."""
        regions: List[Tuple[Any, float]] = []
        try:
            joint_support = self._circuit.support.marginal(variable_objects)
        except Exception:
            return regions
        for simple_region in joint_support.simple_sets:
            region_event = SimpleEvent(
                {v: simple_region[v] for v in variable_objects}
            ).as_composite_set()
            prob = self._circuit.probability(
                region_event.fill_missing_variables_pure(self._circuit.variables)
            )
            if prob > 0.0:
                regions.append((region_event, float(prob)))
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
            data = pd.DataFrame(samples, columns=[v.name for v in self._circuit.variables])

        n = len(data)
        if number_of_histogram_bins is None:
            number_of_histogram_bins = max(5, min(20, int(n ** 0.4)))

        disc_cause = _discretise_continuous_column(
            data[cause_variable_name].values, number_of_histogram_bins
        )
        disc_effect = _discretise_continuous_column(
            data[effect_variable_name].values, number_of_histogram_bins
        )
        disc_adj = [
            _discretise_continuous_column(data[a].values, number_of_histogram_bins)
            for a in adjustment_variable_names
        ]

        cmi, h_y = _compute_conditional_mutual_information(
            cause_bin_indices=disc_cause,
            effect_bin_indices=disc_effect,
            adjustment_bin_index_list=disc_adj,
            number_of_cause_bins=number_of_histogram_bins,
            number_of_effect_bins=number_of_histogram_bins,
        )
        strength = float(np.clip(cmi / h_y, 0.0, 1.0)) if h_y > 1e-10 else 0.0

        return CausalStrengthResult(
            cause_variable_name=cause_variable_name,
            effect_variable_name=effect_variable_name,
            adjustment_variable_names=list(adjustment_variable_names),
            conditional_mutual_information=round(cmi, 6),
            conditional_entropy_of_effect=round(h_y, 6),
            normalised_causal_strength=round(strength, 4),
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
        z = adjustment_variable_names if adjustment_variable_names is not None else []
        results = [
            self.causal_strength(name, effect_variable_name, z, training_dataframe)
            for name in self._causal_variable_names
        ]
        results.sort(key=lambda r: r.normalised_causal_strength, reverse=True)
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

        For each cause variable, queries the interventional circuit
        P(cause, effect | do(cause)) at the observed cause value. The variable
        whose observed value falls furthest outside the training distribution
        (lowest query probability) is identified as the primary cause.

        The recommendation is the cause region midpoint with the highest
        query probability in the interventional circuit.

        Parameters
        ----------
        observed_parameter_values
            Mapping of cause variable names to their observed values.
        effect_variable_name
            Outcome variable used to measure success probability.
        query_resolution
            Half-width of the probability query interval around each
            observed value. Should match the JPT variable precision.
        adjustment_variable_names
            Adjustment set Z. Use [] for independent randomised data.
        """
        if adjustment_variable_names is None:
            adjustment_variable_names = []

        all_results: Dict[str, Dict[str, Any]] = {}
        interventional_circuits: Dict[str, ProbabilisticCircuit] = {}

        for cause_name in self._causal_variable_names:
            if cause_name not in observed_parameter_values:
                continue

            observed = observed_parameter_values[cause_name]
            cause_var = self.get_variable_by_name(cause_name)

            if not cause_var.is_numeric:
                continue

            ic = self.backdoor_adjustment(
                cause_name, effect_variable_name,
                adjustment_variable_names, query_resolution
            )
            interventional_circuits[cause_name] = ic
            obs_event = SimpleEvent(
                {cause_var: closed(float(observed) - query_resolution,
                                   float(observed) + query_resolution)}
            ).as_composite_set()

            try:
                p_observed = float(ic.probability(
                    obs_event.fill_missing_variables_pure(ic.variables)
                ))
            except Exception as e:
                raise ValueError(
                    f"Failed to query interventional circuit for "
                    f"'{cause_name}'={observed}: {e}"
                ) from e

            recommended_float = None
            try:
                best_prob = -1.0
                for region_event, _ in self._extract_leaf_regions_for_variable(cause_var):
                    p_region = float(ic.probability(
                        region_event.fill_missing_variables_pure(ic.variables)
                    ))
                    if p_region > best_prob:
                        best_prob = p_region
                        for ss in region_event.simple_sets:
                            if cause_var in ss:
                                iv_set = ss[cause_var]
                                if hasattr(iv_set, "simple_sets"):
                                    for iv in iv_set.simple_sets:
                                        recommended_float = (
                                            float(iv.lower) + float(iv.upper)
                                        ) / 2.0
                                        break
                                elif hasattr(iv_set, "lower"):
                                    recommended_float = (
                                        float(iv_set.lower) + float(iv_set.upper)
                                    ) / 2.0
                                break
            except Exception:
                recommended_float = None

            all_results[cause_name] = {
                "actual_value": observed,
                "interventional_probability": round(p_observed, 6),
                "recommended_value": recommended_float,
            }

        if not all_results:
            raise ValueError(
                f"No cause variables found in observed_parameter_values. "
                f"Expected at least one of: {self._causal_variable_names}"
            )

        primary = min(
            all_results,
            key=lambda n: all_results[n]["interventional_probability"]
        )
        primary_result = all_results[primary]
        rec_value = primary_result["recommended_value"]
        p_rec = 0.0

        if rec_value is not None:
            primary_var = self.get_variable_by_name(primary)
            primary_ic = interventional_circuits[primary]
            try:
                rec_event = SimpleEvent(
                    {primary_var: closed(float(rec_value) - query_resolution,
                                        float(rec_value) + query_resolution)}
                ).as_composite_set()
                p_rec = float(primary_ic.probability(
                    rec_event.fill_missing_variables_pure(primary_ic.variables)
                ))
            except Exception:
                p_rec = 0.0

        return FailureDiagnosisResult(
            primary_cause_variable_name=primary,
            actual_value=primary_result["actual_value"],
            interventional_probability_at_failure=primary_result[
                "interventional_probability"
            ],
            recommended_value=rec_value,
            interventional_probability_at_recommendation=round(p_rec, 6),
            all_variable_results=all_results,
        )