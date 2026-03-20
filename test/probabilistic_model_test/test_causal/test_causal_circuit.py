"""
Tests for causal_circuit.py

The guiding principle: every test must fail if the implementation is
replaced by a stub. Tests that pass for any implementation are removed.
"""

import math
import unittest

import numpy as np
import pandas as pd
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.distributions.uniform import UniformDistribution

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    CausalStrengthResult,
    FailureDiagnosisResult,
    MdVtreeNode,
    QDeterminismVerificationResult,
    _compute_conditional_mutual_information,
    _compute_entropy_from_counts,
    _discretise_continuous_column,
)


# ══════════════════════════════════════════════════════════════════════════════
# Circuit builders
# ══════════════════════════════════════════════════════════════════════════════


def _build_independent_two_variable_circuit() -> tuple:
    """
    ProductUnit over two independent variables x and y.
        SumUnit_x  [x∈[0,1] w=0.6,  x∈[1,2] w=0.4]
        SumUnit_y  [y∈[0,1] w=0.5,  y∈[1,2] w=0.5]
    P(y∈[0,1]) = 0.5 regardless of x.
    Returns (circuit, x_variable, y_variable).
    """
    x = Continuous("x")
    y = Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    sx = SumUnit(probabilistic_circuit=circuit)
    sx.add_subcircuit(leaf(UniformDistribution(x, closed(0, 1).simple_sets[0]), circuit), math.log(0.6))
    sx.add_subcircuit(leaf(UniformDistribution(x, closed(1, 2).simple_sets[0]), circuit), math.log(0.4))

    sy = SumUnit(probabilistic_circuit=circuit)
    sy.add_subcircuit(leaf(UniformDistribution(y, closed(0, 1).simple_sets[0]), circuit), math.log(0.5))
    sy.add_subcircuit(leaf(UniformDistribution(y, closed(1, 2).simple_sets[0]), circuit), math.log(0.5))

    root.add_subcircuit(sx)
    root.add_subcircuit(sy)
    return circuit, x, y


def _build_three_variable_circuit() -> tuple:
    """
    ProductUnit over three independent variables x (cause), y (cause), z (effect).
        SumUnit_x [x∈[0,1] w=0.7, x∈[1,2] w=0.3]
        SumUnit_y [y∈[0,1] w=0.4, y∈[1,2] w=0.6]
        SumUnit_z [z∈[0,1] w=0.8, z∈[1,2] w=0.2]
    Returns (circuit, x, y, z).
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    for var, w0 in [(x, 0.7), (y, 0.4), (z, 0.8)]:
        s = SumUnit(probabilistic_circuit=circuit)
        s.add_subcircuit(leaf(UniformDistribution(var, closed(0, 1).simple_sets[0]), circuit), math.log(w0))
        s.add_subcircuit(leaf(UniformDistribution(var, closed(1, 2).simple_sets[0]), circuit), math.log(1 - w0))
        root.add_subcircuit(s)

    return circuit, x, y, z


def _build_correlated_circuit() -> tuple:
    """
    Mixture circuit where x strongly predicts y and w is independent of y.

    Two equal-weight components:
      Low:  x∈[0,1], w∈[0,1], y∈[0,0.4]
      High: x∈[1,2], w∈[0,1], y∈[9.6,10]

    With 20 bins over [0,10] each y band fits in one bin, so H(Y|X)≈0
    and C(X→Y) = I(X;Y)/H(Y) → 1.0. w has identical marginal in both
    components so C(W→Y) ≈ 0.

    Returns (circuit, x_variable, w_variable, y_variable).
    """
    x, w, y = Continuous("x"), Continuous("w"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root_sum = SumUnit(probabilistic_circuit=circuit)

    for x_range, y_range in [((0, 1), (0, 0.4)), ((1, 2), (9.6, 10))]:
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(leaf(UniformDistribution(x, closed(*x_range).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(w, closed(0, 1).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(y, closed(*y_range).simple_sets[0]), circuit))
        root_sum.add_subcircuit(product, math.log(0.5))

    return circuit, x, w, y


# ══════════════════════════════════════════════════════════════════════════════
# MdVtreeNode
# ══════════════════════════════════════════════════════════════════════════════


class MdVtreeNodeLeafTestCase(unittest.TestCase):

    def test_leaf_with_single_variable_is_leaf(self):
        self.assertTrue(MdVtreeNode(variables={"x"}, q_set={"x"}).is_leaf())

    def test_node_with_children_is_not_leaf(self):
        parent = MdVtreeNode(
            variables={"x", "y"}, q_set={"x"},
            left=MdVtreeNode(variables={"x"}, q_set={"x"}),
            right=MdVtreeNode(variables={"y"}, q_set={"y"}),
        )
        self.assertFalse(parent.is_leaf())

    def test_leaf_with_no_children_is_leaf(self):
        self.assertTrue(MdVtreeNode(variables={"x"}, q_set=set()).is_leaf())


class MdVtreeNodeFindVariableTestCase(unittest.TestCase):

    def setUp(self):
        self.root = MdVtreeNode(
            variables={"x", "y"}, q_set={"x"},
            left=MdVtreeNode(variables={"x"}, q_set={"x"}),
            right=MdVtreeNode(variables={"y"}, q_set={"y"}),
        )

    def test_find_variable_in_root_q_set(self):
        found = self.root.find_node_for_variable("x")
        self.assertIsNotNone(found)
        self.assertIn("x", found.q_set)

    def test_find_variable_in_child_q_set(self):
        found = self.root.find_node_for_variable("y")
        self.assertIsNotNone(found)
        self.assertIn("y", found.q_set)

    def test_find_nonexistent_variable_returns_none(self):
        self.assertIsNone(self.root.find_node_for_variable("z"))

    def test_find_returns_shallowest_matching_node(self):
        found = self.root.find_node_for_variable("x")
        self.assertEqual(found.q_set, {"x"})
        self.assertIsNotNone(found.left)


class MdVtreeNodeAllQSetsTestCase(unittest.TestCase):

    def test_single_node_returns_its_q_set(self):
        all_sets = MdVtreeNode(variables={"x"}, q_set={"x"}).all_q_sets()
        self.assertEqual(len(all_sets), 1)
        self.assertIn({"x"}, all_sets)

    def test_tree_returns_all_non_empty_q_sets(self):
        root = MdVtreeNode(
            variables={"x", "y"}, q_set={"x"},
            left=MdVtreeNode(variables={"x"}, q_set={"x"}),
            right=MdVtreeNode(variables={"y"}, q_set={"y"}),
        )
        self.assertEqual(len(root.all_q_sets()), 3)

    def test_empty_q_set_not_included(self):
        self.assertEqual(len(MdVtreeNode(variables={"x"}, q_set=set()).all_q_sets()), 0)


class MdVtreeNodeFromCausalGraphTestCase(unittest.TestCase):

    def test_root_q_set_is_first_in_priority_order(self):
        root = MdVtreeNode.from_causal_graph(["x", "y", "z"], ["o"], causal_priority_order=["z", "x", "y"])
        self.assertEqual(root.q_set, {"z"})

    def test_without_priority_order_uses_given_order(self):
        root = MdVtreeNode.from_causal_graph(["x", "y"], ["o"])
        self.assertEqual(root.q_set, {"x"})

    def test_all_causal_variables_appear_in_tree(self):
        root = MdVtreeNode.from_causal_graph(["x", "y", "z"], ["o"])
        all_vars = set().union(*root.all_q_sets())
        for name in ["x", "y", "z"]:
            self.assertIn(name, all_vars)

    def test_five_variable_tree_root_is_correct(self):
        root = MdVtreeNode.from_causal_graph(
            ["x1", "x2", "x3", "x4", "x5"], ["y"],
            causal_priority_order=["x1", "x2", "x3", "x4", "x5"],
        )
        self.assertFalse(root.is_leaf())
        self.assertEqual(root.q_set, {"x1"})

    def test_pick_place_five_cause_variables(self):
        root = MdVtreeNode.from_causal_graph(
            causal_variable_names=[
                "pick_approach_x", "pick_approach_y",
                "pick_arm", "place_approach_x", "place_approach_y",
            ],
            effect_variable_names=["milk_end_z"],
            causal_priority_order=[
                "pick_approach_x", "place_approach_x",
                "pick_arm", "pick_approach_y", "place_approach_y",
            ],
        )
        self.assertEqual(root.q_set, {"pick_approach_x"})
        all_vars = set().union(*root.all_q_sets())
        for name in ["pick_approach_x", "pick_approach_y", "pick_arm",
                     "place_approach_x", "place_approach_y"]:
            self.assertIn(name, all_vars)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


class EntropyComputationTestCase(unittest.TestCase):

    def test_uniform_has_maximum_entropy(self):
        self.assertAlmostEqual(
            _compute_entropy_from_counts(np.array([25, 25, 25, 25])), math.log(4), delta=1e-10
        )

    def test_deterministic_has_zero_entropy(self):
        self.assertAlmostEqual(
            _compute_entropy_from_counts(np.array([100, 0, 0, 0])), 0.0, delta=1e-10
        )

    def test_empty_array_returns_zero(self):
        self.assertEqual(_compute_entropy_from_counts(np.array([0, 0, 0])), 0.0)

    def test_entropy_is_non_negative(self):
        self.assertGreaterEqual(_compute_entropy_from_counts(np.array([10, 20, 30, 40])), 0.0)

    def test_two_equal_outcomes(self):
        self.assertAlmostEqual(
            _compute_entropy_from_counts(np.array([50, 50])), math.log(2), delta=1e-10
        )

    def test_does_not_exceed_log_support_size(self):
        self.assertLessEqual(
            _compute_entropy_from_counts(np.array([10, 30, 60])), math.log(3) + 1e-10
        )


class DiscretisationTestCase(unittest.TestCase):

    def test_continuous_maps_to_int_bins(self):
        r = _discretise_continuous_column(np.array([0.0, 0.5, 1.0, 1.5, 2.0]), 4)
        self.assertEqual(r.dtype, np.int32)
        self.assertTrue(np.all(r >= 0) and np.all(r < 4))

    def test_constant_column_maps_to_zero(self):
        self.assertTrue(np.all(_discretise_continuous_column(np.array([3.14] * 4), 5) == 0))

    def test_categorical_string_encoded(self):
        r = _discretise_continuous_column(np.array(["LEFT", "RIGHT", "LEFT"]), 10)
        self.assertEqual(r.dtype, np.int32)
        self.assertEqual(len(np.unique(r)), 2)

    def test_output_length_matches_input(self):
        self.assertEqual(len(_discretise_continuous_column(np.linspace(0, 1, 100), 10)), 100)

    def test_minimum_maps_to_zero_bin(self):
        self.assertEqual(_discretise_continuous_column(np.array([0.0, 0.5, 1.0]), 3)[0], 0)

    def test_maximum_maps_to_last_bin(self):
        self.assertEqual(_discretise_continuous_column(np.array([0.0, 0.5, 1.0]), 3)[-1], 2)

    def test_ordering_preserved(self):
        r = _discretise_continuous_column(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), 4)
        for i in range(len(r) - 1):
            self.assertLessEqual(r[i], r[i + 1])


class ConditionalMutualInformationTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_independent_variables_near_zero(self):
        x, y = np.random.randint(0, 5, 2000), np.random.randint(0, 5, 2000)
        cmi, _ = _compute_conditional_mutual_information(x, y, [], 5, 5)
        self.assertAlmostEqual(cmi, 0.0, delta=0.05)

    def test_identical_variables_high_cmi(self):
        v = np.random.randint(0, 5, 1000)
        cmi, _ = _compute_conditional_mutual_information(v, v, [], 5, 5)
        self.assertGreater(cmi, 1.0)

    def test_cmi_non_negative(self):
        x, y = np.random.randint(0, 4, 500), np.random.randint(0, 4, 500)
        cmi, _ = _compute_conditional_mutual_information(x, y, [], 4, 4)
        self.assertGreaterEqual(cmi, 0.0)

    def test_conditional_cmi_non_negative(self):
        x = np.random.randint(0, 4, 500)
        y = np.random.randint(0, 4, 500)
        z = np.random.randint(0, 3, 500)
        cmi, h_y_z = _compute_conditional_mutual_information(x, y, [z], 4, 4)
        self.assertGreaterEqual(cmi, 0.0)
        self.assertGreaterEqual(h_y_z, 0.0)

    def test_entropy_of_effect_positive_and_bounded(self):
        x, y = np.random.randint(0, 4, 1000), np.random.randint(0, 4, 1000)
        _, h_y = _compute_conditional_mutual_information(x, y, [], 4, 4)
        self.assertGreater(h_y, 0.0)
        self.assertLessEqual(h_y, math.log(4) + 0.05)

    def test_perfectly_correlated_cmi_near_entropy(self):
        v = np.random.randint(0, 5, 2000)
        cmi, h_y = _compute_conditional_mutual_information(v, v, [], 5, 5)
        self.assertGreater(cmi / h_y, 0.9)


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════


class QDeterminismVerificationResultTestCase(unittest.TestCase):

    def test_passed_str_contains_pass(self):
        r = QDeterminismVerificationResult(True, [], [{"x"}], ["x", "y"])
        self.assertIn("PASS", str(r))
        self.assertNotIn("FAIL", str(r))

    def test_failed_str_contains_fail_and_violation(self):
        r = QDeterminismVerificationResult(False, ["overlap on x"], [{"x"}], ["x"])
        self.assertIn("FAIL", str(r))
        self.assertIn("overlap on x", str(r))


class FailureDiagnosisResultTestCase(unittest.TestCase):

    def setUp(self):
        self.result = FailureDiagnosisResult(
            primary_cause_variable_name="x",
            actual_value=1.3,
            interventional_probability_at_failure=0.0,
            recommended_value=1.65,
            interventional_probability_at_recommendation=0.149,
            all_variable_results={
                "x": {"actual_value": 1.3, "interventional_probability": 0.0, "recommended_value": 1.65},
                "y": {"actual_value": 0.1, "interventional_probability": 0.089, "recommended_value": 0.0},
            },
        )

    def test_str_contains_primary_cause_marker(self):
        self.assertIn("PRIMARY CAUSE", str(self.result))

    def test_str_contains_recommended_value(self):
        self.assertIn("1.65", str(self.result))

    def test_primary_cause_has_lowest_probability(self):
        lowest = min(self.result.all_variable_results.values(),
                     key=lambda r: r["interventional_probability"])
        self.assertEqual(lowest["interventional_probability"],
                         self.result.interventional_probability_at_failure)


class CausalStrengthResultTestCase(unittest.TestCase):

    def setUp(self):
        self.result = CausalStrengthResult("x", "y", ["z1", "z2"], 0.412, 0.723, 0.57)

    def test_str_contains_names_and_strength(self):
        s = str(self.result)
        self.assertIn("x", s)
        self.assertIn("y", s)
        self.assertIn("0.5700", s)
        self.assertIn("z1", s)

    def test_empty_adjustment_shows_empty_set_symbol(self):
        r = CausalStrengthResult("x", "y", [], 0.3, 0.6, 0.5)
        self.assertIn("∅", str(r))

    def test_normalised_strength_in_unit_interval(self):
        self.assertGreaterEqual(self.result.normalised_causal_strength, 0.0)
        self.assertLessEqual(self.result.normalised_causal_strength, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# CausalCircuit construction
# ══════════════════════════════════════════════════════════════════════════════


class CausalCircuitConstructionTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, _, _ = _build_independent_two_variable_circuit()
        self.mdvtree = MdVtreeNode.from_causal_graph(["x"], ["y"])

    def test_from_probabilistic_circuit_returns_causal_circuit(self):
        cc = CausalCircuit.from_probabilistic_circuit(self.circuit, self.mdvtree, ["x"], ["y"])
        self.assertIsInstance(cc, CausalCircuit)

    def test_variable_names_stored_and_copied(self):
        cc = CausalCircuit.from_probabilistic_circuit(self.circuit, self.mdvtree, ["x"], ["y"])
        self.assertEqual(cc.causal_variable_names, ["x"])
        self.assertEqual(cc.effect_variable_names, ["y"])
        names = cc.causal_variable_names
        names.append("mutated")
        self.assertNotIn("mutated", cc.causal_variable_names)

    def test_from_jpt_accepts_probabilistic_circuit(self):
        cc = CausalCircuit.from_jpt(self.circuit, self.mdvtree, ["x"], ["y"])
        self.assertIsInstance(cc, CausalCircuit)

    def test_from_jpt_raises_type_error_with_informative_message(self):
        with self.assertRaises(TypeError) as ctx:
            CausalCircuit.from_jpt(42, self.mdvtree, ["x"], ["y"])
        self.assertIn("ProbabilisticCircuit", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# get_variable_by_name
# ══════════════════════════════════════════════════════════════════════════════


class GetVariableByNameTestCase(unittest.TestCase):

    def setUp(self):
        circuit, _, _ = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            circuit, MdVtreeNode.from_causal_graph(["x"], ["y"]), ["x"], ["y"]
        )

    def test_existing_variable_returned(self):
        self.assertEqual(self.cc.get_variable_by_name("x").name, "x")

    def test_nonexistent_raises_value_error_with_available_names(self):
        with self.assertRaises(ValueError) as ctx:
            self.cc.get_variable_by_name("ghost")
        self.assertIn("x", str(ctx.exception))
        self.assertIn("y", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# verify_q_determinism
# ══════════════════════════════════════════════════════════════════════════════


class VerifyQDeterminismTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, _, _ = _build_independent_two_variable_circuit()

    def _make_cc(self, mdvtree):
        return CausalCircuit.from_probabilistic_circuit(
            self.circuit, mdvtree, ["x"], ["y"]
        )

    def test_correct_circuit_passes(self):
        result = self._make_cc(MdVtreeNode.from_causal_graph(["x"], ["y"])).verify_q_determinism()
        self.assertTrue(result.passed)
        self.assertEqual(len(result.violations), 0)

    def test_returns_checked_q_sets_and_variable_names(self):
        result = self._make_cc(MdVtreeNode.from_causal_graph(["x"], ["y"])).verify_q_determinism()
        self.assertGreater(len(result.checked_q_sets), 0)
        self.assertIn("x", result.circuit_variable_names)

    def test_nonexistent_q_variable_is_violation(self):
        mdvtree = MdVtreeNode(variables={"x", "ghost"}, q_set={"ghost"})
        result = self._make_cc(mdvtree).verify_q_determinism()
        self.assertFalse(result.passed)
        self.assertTrue(any("ghost" in v for v in result.violations))


# ══════════════════════════════════════════════════════════════════════════════
# backdoor_adjustment — structural
# ══════════════════════════════════════════════════════════════════════════════


class BackdoorAdjustmentStructuralTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, MdVtreeNode.from_causal_graph(["x"], ["y"]), ["x"], ["y"]
        )

    def test_returns_probabilistic_circuit_with_variables(self):
        ic = self.cc.backdoor_adjustment("x", "y", [])
        self.assertIsInstance(ic, ProbabilisticCircuit)
        self.assertGreater(len(ic.variables), 0)

    def test_non_causal_variable_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cc.backdoor_adjustment("y", "y", [])

    def test_non_effect_variable_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cc.backdoor_adjustment("x", "x", [])

    def test_error_message_mentions_registered_causes(self):
        with self.assertRaises(ValueError) as ctx:
            self.cc.backdoor_adjustment("y", "y", [])
        self.assertIn("x", str(ctx.exception))

    def test_default_and_explicit_empty_adjustment_agree(self):
        ic_explicit = self.cc.backdoor_adjustment("x", "y", [])
        ic_default = self.cc.backdoor_adjustment("x", "y")
        event = (
            SimpleEvent({self.y: closed(0, 1)})
            .as_composite_set()
            .fill_missing_variables_pure(ic_explicit.variables)
        )
        self.assertAlmostEqual(
            ic_explicit.probability(event), ic_default.probability(event), delta=1e-6
        )


# ══════════════════════════════════════════════════════════════════════════════
# backdoor_adjustment — correctness
# ══════════════════════════════════════════════════════════════════════════════


class BackdoorAdjustmentCorrectnessTestCase(unittest.TestCase):
    """
    For P(x,y) = P(x)*P(y), the backdoor criterion requires:
        P(y | do(x=v)) = P(y)  for all v in the training support.
    """

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, MdVtreeNode.from_causal_graph(["x"], ["y"]), ["x"], ["y"]
        )
        self.ic = self.cc.backdoor_adjustment("x", "y", [])

    def _p_y_in_0_1(self) -> float:
        event = (
            SimpleEvent({self.y: closed(0, 1)})
            .as_composite_set()
            .fill_missing_variables_pure(self.ic.variables)
        )
        return float(self.ic.probability(event))

    def test_total_probability_integrates_to_one(self):
        event = (
            SimpleEvent({self.y: closed(0, 2)})
            .as_composite_set()
            .fill_missing_variables_pure(self.ic.variables)
        )
        self.assertAlmostEqual(self.ic.probability(event), 1.0, delta=0.01)

    def test_interventional_prob_matches_marginal_for_independent_circuit(self):
        marginal = float(self.circuit.probability(
            SimpleEvent({self.y: closed(0, 1)})
            .as_composite_set()
            .fill_missing_variables_pure(self.circuit.variables)
        ))
        self.assertAlmostEqual(self._p_y_in_0_1(), marginal, delta=0.05)

    def test_interventional_prob_same_across_cause_regions(self):
        p1, p2 = self._p_y_in_0_1(), self._p_y_in_0_1()
        self.assertAlmostEqual(p1, 0.5, delta=0.05)
        self.assertAlmostEqual(p2, 0.5, delta=0.05)

    def test_outside_cause_domain_has_zero_probability(self):
        event = (
            SimpleEvent({self.x: closed(5, 6)})
            .as_composite_set()
            .fill_missing_variables_pure(self.ic.variables)
        )
        self.assertAlmostEqual(float(self.ic.probability(event)), 0.0, delta=1e-6)

    def test_outside_effect_domain_has_zero_probability(self):
        event = (
            SimpleEvent({self.y: closed(5, 6)})
            .as_composite_set()
            .fill_missing_variables_pure(self.ic.variables)
        )
        self.assertAlmostEqual(float(self.ic.probability(event)), 0.0, delta=1e-6)


class BackdoorAdjustmentWithAdjustmentTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y, self.z = _build_three_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, MdVtreeNode.from_causal_graph(["x", "y"], ["z"]), ["x", "y"], ["z"]
        )

    def test_adjusted_circuit_returns_circuit_integrating_to_one(self):
        ic = self.cc.backdoor_adjustment("x", "z", ["y"])
        self.assertIsInstance(ic, ProbabilisticCircuit)
        event = (
            SimpleEvent({self.z: closed(0, 2)})
            .as_composite_set()
            .fill_missing_variables_pure(ic.variables)
        )
        self.assertAlmostEqual(float(ic.probability(event)), 1.0, delta=0.01)

    def test_adjustment_on_independent_data_matches_no_adjustment(self):
        ic_no = self.cc.backdoor_adjustment("x", "z", [])
        ic_adj = self.cc.backdoor_adjustment("x", "z", ["y"])
        event = SimpleEvent({self.z: closed(0, 1)}).as_composite_set()
        p_no = float(ic_no.probability(event.fill_missing_variables_pure(ic_no.variables)))
        p_adj = float(ic_adj.probability(event.fill_missing_variables_pure(ic_adj.variables)))
        self.assertAlmostEqual(p_no, 0.8, delta=0.05)
        self.assertAlmostEqual(p_adj, 0.8, delta=0.05)
        self.assertAlmostEqual(p_no, p_adj, delta=0.05)


class CausalStrengthStructuralTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        circuit, _, _ = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            circuit, MdVtreeNode.from_causal_graph(["x"], ["y"]), ["x"], ["y"]
        )
        samples = circuit.sample(2000)
        self.df = pd.DataFrame(samples, columns=[v.name for v in circuit.variables])

    def test_returns_result_with_correct_fields(self):
        r = self.cc.causal_strength("x", "y", training_dataframe=self.df)
        self.assertIsInstance(r, CausalStrengthResult)
        self.assertEqual(r.cause_variable_name, "x")
        self.assertEqual(r.effect_variable_name, "y")
        self.assertEqual(r.adjustment_variable_names, [])

    def test_normalised_strength_in_unit_interval(self):
        r = self.cc.causal_strength("x", "y", training_dataframe=self.df)
        self.assertGreaterEqual(r.normalised_causal_strength, 0.0)
        self.assertLessEqual(r.normalised_causal_strength, 1.0)

    def test_entropy_of_effect_is_positive(self):
        r = self.cc.causal_strength("x", "y", training_dataframe=self.df)
        self.assertGreater(r.conditional_entropy_of_effect, 0.0)

    def test_without_dataframe_samples_from_circuit(self):
        r = self.cc.causal_strength("x", "y")
        self.assertIsInstance(r, CausalStrengthResult)
        self.assertGreaterEqual(r.normalised_causal_strength, 0.0)


class CausalStrengthCorrectnessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.circuit, cls.x, cls.w, cls.y = _build_correlated_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MdVtreeNode.from_causal_graph(["x", "w"], ["y"]),
            ["x", "w"], ["y"],
        )
        samples = cls.circuit.sample(5000)
        cls.df = pd.DataFrame(samples, columns=[v.name for v in cls.circuit.variables])

    def test_correlated_variable_has_high_causal_strength(self):
        r = self.cc.causal_strength("x", "y", training_dataframe=self.df)
        self.assertGreater(
            r.normalised_causal_strength, 0.9,
            msg=f"Expected C(x→y) > 0.9, got {r.normalised_causal_strength:.4f}",
        )

    def test_independent_variable_has_low_causal_strength(self):
        r = self.cc.causal_strength("w", "y", training_dataframe=self.df)
        self.assertLess(
            r.normalised_causal_strength, 0.15,
            msg=f"Expected C(w→y) < 0.15, got {r.normalised_causal_strength:.4f}",
        )

    def test_rank_puts_correlated_variable_first(self):
        ranking = self.cc.rank_causal_variables("y", training_dataframe=self.df)
        self.assertEqual(len(ranking), 2)
        self.assertEqual(ranking[0].cause_variable_name, "x")

    def test_correlated_strictly_outranks_independent(self):
        x_r = self.cc.causal_strength("x", "y", training_dataframe=self.df)
        w_r = self.cc.causal_strength("w", "y", training_dataframe=self.df)
        self.assertGreater(x_r.normalised_causal_strength, w_r.normalised_causal_strength)


class RankCausalVariablesTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        circuit, _, _, _ = _build_three_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            circuit, MdVtreeNode.from_causal_graph(["x", "y"], ["z"]), ["x", "y"], ["z"]
        )
        samples = circuit.sample(2000)
        self.df = pd.DataFrame(samples, columns=[v.name for v in circuit.variables])

    def test_one_result_per_cause_sorted_descending(self):
        ranking = self.cc.rank_causal_variables("z", training_dataframe=self.df)
        self.assertEqual(len(ranking), 2)
        for i in range(len(ranking) - 1):
            self.assertGreaterEqual(
                ranking[i].normalised_causal_strength, ranking[i + 1].normalised_causal_strength
            )

    def test_effect_variable_consistent(self):
        for r in self.cc.rank_causal_variables("z", training_dataframe=self.df):
            self.assertEqual(r.effect_variable_name, "z")

    def test_default_z_is_empty_for_independent_data(self):
        for r in self.cc.rank_causal_variables("z", training_dataframe=self.df):
            self.assertEqual(r.adjustment_variable_names, [])



class DiagnoseFailureStructuralTestCase(unittest.TestCase):

    def setUp(self):
        circuit, _, _ = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            circuit, MdVtreeNode.from_causal_graph(["x"], ["y"]), ["x"], ["y"]
        )

    def test_returns_result_with_correct_fields(self):
        r = self.cc.diagnose_failure({"x": 0.5}, "y")
        self.assertIsInstance(r, FailureDiagnosisResult)
        self.assertEqual(r.primary_cause_variable_name, "x")
        self.assertAlmostEqual(r.actual_value, 0.5, delta=1e-6)
        self.assertIn("x", r.all_variable_results)
        for vr in r.all_variable_results.values():
            for key in ("actual_value", "interventional_probability", "recommended_value"):
                self.assertIn(key, vr)

    def test_missing_cause_variables_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cc.diagnose_failure({}, "y")

    def test_primary_cause_has_minimum_interventional_probability(self):
        r = self.cc.diagnose_failure({"x": 0.5}, "y")
        min_p = min(vr["interventional_probability"] for vr in r.all_variable_results.values())
        self.assertAlmostEqual(r.interventional_probability_at_failure, min_p, delta=1e-6)

    def test_recommendation_probability_non_negative(self):
        r = self.cc.diagnose_failure({"x": 0.5}, "y")
        self.assertGreaterEqual(r.interventional_probability_at_recommendation, 0.0)

    def test_recommendation_probability_positive_for_in_domain_query(self):
        r = self.cc.diagnose_failure({"x": 0.5}, "y")
        self.assertGreater(
            r.interventional_probability_at_recommendation, 0.0,
            msg=f"Expected P(rec) > 0 for in-domain x=0.5, got {r.interventional_probability_at_recommendation}",
        )


class DiagnoseFailureCorrectnessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.w, cls.y = _build_correlated_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MdVtreeNode.from_causal_graph(["x", "w"], ["y"]),
            ["x", "w"], ["y"],
        )

    def test_out_of_range_cause_identified_as_primary_with_zero_probability(self):
        r = self.cc.diagnose_failure({"x": 5.0, "w": 0.5}, "y")
        self.assertEqual(
            r.primary_cause_variable_name, "x",
            msg=f"Expected x as primary cause, got {r.primary_cause_variable_name}. "
                f"Results: {r.all_variable_results}",
        )
        self.assertAlmostEqual(r.interventional_probability_at_failure, 0.0, delta=1e-6)

    def test_in_range_variable_has_nonzero_probability(self):
        r = self.cc.diagnose_failure({"x": 5.0, "w": 0.5}, "y")
        w_p = r.all_variable_results["w"]["interventional_probability"]
        self.assertGreater(w_p, 0.0, msg=f"Expected P(y|do(w=0.5)) > 0, got {w_p}")

    def test_both_in_range_gives_nonzero_probabilities(self):
        r = self.cc.diagnose_failure({"x": 0.5, "w": 0.5}, "y")
        for name, vr in r.all_variable_results.items():
            self.assertGreater(vr["interventional_probability"], 0.0,
                               msg=f"Expected positive P for {name}")

    def test_recommendation_improves_over_failure(self):
        r = self.cc.diagnose_failure({"x": 5.0, "w": 0.5}, "y")
        self.assertGreater(
            r.interventional_probability_at_recommendation,
            r.interventional_probability_at_failure,
            msg=(f"P(rec)={r.interventional_probability_at_recommendation:.4f} must exceed "
                 f"P(failure)={r.interventional_probability_at_failure:.4f}"),
        )

    def test_recommendation_is_within_training_domain(self):
        r = self.cc.diagnose_failure({"x": 5.0, "w": 0.5}, "y")
        self.assertGreater(
            r.interventional_probability_at_recommendation, 0.0,
            msg=f"Recommendation probability must be > 0. Got {r.interventional_probability_at_recommendation}",
        )

    def test_str_output_contains_primary_cause_marker(self):
        r = self.cc.diagnose_failure({"x": 5.0, "w": 0.5}, "y")
        output = str(r)
        self.assertGreater(len(output), 0)
        self.assertIn("PRIMARY CAUSE", output)


class EndToEndIntegrationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)

        cls.circuit3, cls.x3, cls.y3, cls.z3 = _build_three_variable_circuit()
        cls.cc3 = CausalCircuit.from_probabilistic_circuit(
            cls.circuit3,
            MdVtreeNode.from_causal_graph(["x", "y"], ["z"], causal_priority_order=["x", "y"]),
            ["x", "y"], ["z"],
        )
        samples3 = cls.circuit3.sample(2000)
        cls.df3 = pd.DataFrame(samples3, columns=[v.name for v in cls.circuit3.variables])

        cls.circuit_corr, cls.xc, cls.wc, cls.yc = _build_correlated_circuit()
        cls.cc_corr = CausalCircuit.from_probabilistic_circuit(
            cls.circuit_corr,
            MdVtreeNode.from_causal_graph(["x", "w"], ["y"]),
            ["x", "w"], ["y"],
        )
        samples_corr = cls.circuit_corr.sample(5000)
        cls.df_corr = pd.DataFrame(samples_corr, columns=[v.name for v in cls.circuit_corr.variables])

    def test_verification_passes(self):
        self.assertTrue(self.cc3.verify_q_determinism().passed)

    def test_backdoor_circuit_has_positive_probability_in_effect_domain(self):
        ic = self.cc3.backdoor_adjustment("x", "z", [])
        event = (
            SimpleEvent({self.z3: closed(0, 2)})
            .as_composite_set()
            .fill_missing_variables_pure(ic.variables)
        )
        self.assertGreater(float(ic.probability(event)), 0.0)

    def test_causal_strength_ranking_length_and_validity(self):
        ranking = self.cc3.rank_causal_variables("z", training_dataframe=self.df3)
        self.assertEqual(len(ranking), 2)
        for r in ranking:
            self.assertGreaterEqual(r.normalised_causal_strength, 0.0)
            self.assertLessEqual(r.normalised_causal_strength, 1.0)
            self.assertGreater(r.conditional_entropy_of_effect, 0.0)

    def test_causal_variable_outranks_independent(self):
        ranking = self.cc_corr.rank_causal_variables("y", training_dataframe=self.df_corr)
        self.assertEqual(ranking[0].cause_variable_name, "x")

    def test_failure_diagnosis_identifies_out_of_range_primary_cause(self):
        r = self.cc3.diagnose_failure({"x": 5.0, "y": 0.5}, "z")
        self.assertEqual(r.primary_cause_variable_name, "x")
        self.assertAlmostEqual(r.interventional_probability_at_failure, 0.0, delta=1e-6)

    def test_failure_diagnosis_in_range_has_nonnegative_probabilities(self):
        r = self.cc3.diagnose_failure({"x": 0.5, "y": 0.5}, "z")
        for vr in r.all_variable_results.values():
            self.assertGreaterEqual(vr["interventional_probability"], 0.0)

    def test_full_pipeline_str_output_contains_marker(self):
        r = self.cc3.diagnose_failure({"x": 5.0, "y": 0.5}, "z")
        self.assertIn("PRIMARY CAUSE", str(r))

    def test_pick_place_mdvtree_covers_all_variables(self):
        mdvtree = MdVtreeNode.from_causal_graph(
            causal_variable_names=[
                "pick_approach_x", "pick_approach_y",
                "pick_arm", "place_approach_x", "place_approach_y",
            ],
            effect_variable_names=["milk_end_z"],
            causal_priority_order=[
                "pick_approach_x", "place_approach_x",
                "pick_arm", "pick_approach_y", "place_approach_y",
            ],
        )
        all_vars = set().union(*mdvtree.all_q_sets())
        for name in ["pick_approach_x", "pick_approach_y", "pick_arm",
                     "place_approach_x", "place_approach_y"]:
            self.assertIn(name, all_vars)


if __name__ == "__main__":
    unittest.main()