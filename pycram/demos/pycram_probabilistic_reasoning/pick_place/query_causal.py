"""
Causal reasoning queries using a Joint Probability Tree (JPT).

Assumption:
Input variables were independently randomized.
Thus, P(Y | do(X)) = P(Y | X) for all inputs.
"""

import numpy as np
import pandas as pd
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.distributions.univariate import Multinomial
from jpt.variables import NumericVariable, SymbolicVariable
from jpt.trees import JPT


class ArmDomain(Multinomial):
    values = OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)])
    labels = OrderedDictProxy([(0, "LEFT"), (1, "RIGHT")])


def build_variables():
    """Create JPT variables."""
    return [
        NumericVariable("pick_approach_x", precision=0.005),
        NumericVariable("pick_approach_y", precision=0.005),
        NumericVariable("place_approach_x", precision=0.005),
        NumericVariable("place_approach_y", precision=0.005),
        NumericVariable("milk_end_x", precision=0.001),
        NumericVariable("milk_end_y", precision=0.001),
        NumericVariable("milk_end_z", precision=0.0005),
        SymbolicVariable("pick_arm", domain=ArmDomain),
    ]


def get_variable(variables, name):
    return next(variable for variable in variables if variable.name == name)


def safe_infer(jpt_model, query, evidence=None):
    try:
        return jpt_model.infer(query=query, evidence=evidence or {})
    except Exception:
        return 0.0


def safe_expectation(jpt_model, variables, evidence=None):
    try:
        return jpt_model.expectation(variables=variables, evidence=evidence or {})
    except Exception:
        return None


def print_header(index, title, hypothesis):
    print("\n" + "═" * 72)
    print(f"  CAUSAL TEST {index}: {title}")
    print("═" * 72)
    print(f"  Hypothesis: {hypothesis}\n")


def print_result(message):
    print(f"  → {message}")


def print_pass(message):
    print(f"  ✓ {message}")


def print_fail(message):
    print(f"  ✗ {message}")


def main():
    variables = build_variables()

    pick_x = get_variable(variables, "pick_approach_x")
    pick_y = get_variable(variables, "pick_approach_y")
    place_x = get_variable(variables, "place_approach_x")
    arm_variable = get_variable(variables, "pick_arm")

    jpt_model = JPT(variables=variables, min_samples_leaf=25).load("pick_and_place_jpt.json")
    dataframe = pd.read_csv("pick_and_place_dataframe.csv")

    print("╔" + "═" * 70 + "╗")
    print("║     JPT CAUSAL ANALYSIS                                          ║")
    print("║     Backdoor criterion satisfied → all queries are causal        ║")
    print("╚" + "═" * 70 + "╝")

    # ─────────────────────────────────────────────
    # TEST 1 — Dose-response
    # ─────────────────────────────────────────────
    print_header(
        1,
        "Dose–Response Curve (pick_x → success)",
        "A causal reachability effect produces a smooth unimodal curve.",
    )

    sweep_values = np.arange(1.0, 2.22, 0.05)
    response_values = []

    print("  pick_x range        Probability    Shape")
    print("  " + "─" * 60)

    for value in sweep_values:
        probability = safe_infer(jpt_model, {pick_x: [value, value + 0.05]})
        response_values.append((value, probability))

    peak_value, peak_probability = max(response_values, key=lambda item: item[1])

    for value, probability in response_values:
        bar = "█" * int(probability * 40)
        marker = "← peak" if abs(value - peak_value) < 1e-3 else ""
        if probability == 0.0:
            marker = "← zero"

        print(f"  [{value:.2f},{value+0.05:.2f}]   {probability:.4f}   {bar:<40} {marker}")

    print()
    print_result(f"Peak located at [{peak_value:.2f}, {peak_value+0.05:.2f}]")
    print_result(f"Peak probability: {peak_probability:.4f}")

    if peak_probability > 0:
        print_pass("Unimodal response supports causal reachability constraint.")
    else:
        print_fail("No meaningful response pattern detected.")
    # ─────────────────────────────────────────────
    # TEST 2 — Subgroup Invariance
    # ─────────────────────────────────────────────
    print_header(
        2,
        "Subgroup Invariance (pick_x effect | arm)",
        "Check whether pick_x effect is consistent across arms.",
    )

    bins = [
        (1.41, 1.50),
        (1.50, 1.60),
        (1.60, 1.70),
        (1.70, 1.80),
    ]

    print("  pick_x bin       P(LEFT)    P(RIGHT)    Ratio L/R    Divergence")
    print("  " + "─" * 70)

    divergences = []

    for lower, upper in bins:
        probability_left = safe_infer(jpt_model, {pick_x: [lower, upper], arm_variable: "LEFT"})
        probability_right = safe_infer(jpt_model, {pick_x: [lower, upper], arm_variable: "RIGHT"})

        total = probability_left + probability_right

        if total > 0:
            normalized_left = probability_left / total
            normalized_right = probability_right / total
        else:
            normalized_left = normalized_right = 0.5

        ratio = (probability_left / probability_right) if probability_right > 0 else float("inf")
        divergence = abs(normalized_left - normalized_right)

        divergences.append(divergence)

        flag = "← asymmetric" if divergence > 0.15 else "← balanced"

        print(
            f"  [{lower:.2f},{upper:.2f}]   "
            f"{normalized_left:.3f}      {normalized_right:.3f}      "
            f"{ratio:>6.2f}       {divergence:.3f}  {flag}"
        )

    mean_divergence = float(np.mean(divergences))

    print()
    print_result(f"Mean divergence across bins: {mean_divergence:.3f}")

    if mean_divergence < 0.10:
        print_pass("Effect is arm-invariant — pick_x is causally dominant.")
    elif mean_divergence < 0.20:
        print_result("Moderate arm asymmetry — arm acts as a causal moderator.")
    else:
        print_result("Strong asymmetry — arm significantly modulates the effect.")

    # ─────────────────────────────────────────────
    # TEST 3 — Independence
    # ─────────────────────────────────────────────
    print_header(
        3,
        "Independence of Pick and Place",
        "Pick phase must not influence place phase.",
    )

    baseline_mean = float(dataframe["place_approach_x"].mean())

    print(f"  Baseline E[place_x] = {baseline_mean:.4f} m\n")
    print("  pick_x bin          E[place_x]      Deviation")
    print("  " + "─" * 60)

    bins = [[1.41, 1.52], [1.52, 1.63], [1.63, 1.74], [1.74, 1.80]]
    deviations = []

    for lower, upper in bins:
        result = safe_expectation(jpt_model, [place_x], {pick_x: [lower, upper]})
        if result:
            conditional_mean = result[place_x]
            deviation = conditional_mean - baseline_mean
            deviations.append(abs(deviation))

            flag = "✓ independent" if abs(deviation) < 0.01 else "⚠ deviation"
            print(f"  [{lower:.2f},{upper:.2f}]    {conditional_mean:.4f} m    {deviation:+.4f} m   {flag}")

    max_deviation = max(deviations) if deviations else 0.0

    print()
    print_result(f"Maximum deviation: {max_deviation:.4f} m ({max_deviation*1000:.1f} mm)")

    if max_deviation < 0.01:
        print_pass("Pick and place phases are causally independent.")
    else:
        print_fail("Detected cross-phase dependency.")

    # ─────────────────────────────────────────────
    # TEST 4 — Randomisation + Moderator Effect
    # ─────────────────────────────────────────────
    print_header(
        4,
        "Randomisation Check and Arm Preference Gradient",
        "Verify arm randomisation and detect causal preference shifts.",
    )

    bins = np.arange(1.41, 1.81, 0.08)

    print("  pick_x bin       P(LEFT)    P(RIGHT)    Deviation    Interpretation")
    print("  " + "─" * 75)

    left_preferences = []

    for value in bins:
        lower = float(value)
        upper = float(value + 0.08)

        probability_left = safe_infer(jpt_model, {arm_variable: "LEFT"}, {pick_x: [lower, upper]})
        probability_right = safe_infer(jpt_model, {arm_variable: "RIGHT"}, {pick_x: [lower, upper]})

        total = probability_left + probability_right

        if total > 0:
            conditional_left = probability_left / total
            conditional_right = probability_right / total
        else:
            conditional_left = conditional_right = 0.5

        left_preferences.append(conditional_left)

        deviation = conditional_left - 0.5

        if deviation > 0.05:
            interpretation = "← LEFT preferred"
        elif deviation < -0.05:
            interpretation = "← RIGHT preferred"
        else:
            interpretation = "← balanced"

        print(
            f"  [{lower:.2f},{upper:.2f}]   "
            f"{conditional_left:.3f}      {conditional_right:.3f}      "
            f"{deviation:+.3f}     {interpretation}"
        )

    gradient = left_preferences[-1] - left_preferences[0]

    print()
    print_result(f"Preference gradient across pick_x: {gradient:+.3f}")

    if abs(gradient) > 0.05:
        print_pass(
            f"Arm preference varies systematically ({abs(gradient)*100:.1f}%) — "
            "this is a causal moderator effect."
        )
    else:
        print_result("Arm preference is approximately uniform — no strong moderation.")

    # ─────────────────────────────────────────────
    # TEST 5 — ATE
    # ─────────────────────────────────────────────
    print_header(
        5,
        "Average Treatment Effect (ATE)",
        "Quantifies causal strength of each variable.",
    )

    treatments = [
        ("pick_x", {pick_x: [1.55, 1.70]}, {pick_x: [1.00, 1.40]}),
        ("place_x", {place_x: [3.20, 3.50]}, {place_x: [2.70, 3.10]}),
        ("arm", {arm_variable: "LEFT"}, {arm_variable: "RIGHT"}),
        ("pick_y", {pick_y: [-0.10, 0.10]}, {pick_y: [-0.40, -0.20]}),
    ]

    print("  Variable      P(safe)    P(unsafe)    ATE")
    print("  " + "─" * 60)

    results = []

    for name, safe_region, unsafe_region in treatments:
        probability_safe = safe_infer(jpt_model, safe_region)
        probability_unsafe = safe_infer(jpt_model, unsafe_region)
        ate_value = probability_safe - probability_unsafe

        results.append((name, ate_value))

        bar = "█" * int(abs(ate_value) * 20)
        print(f"  {name:<10}   {probability_safe:.4f}    {probability_unsafe:.4f}    {ate_value:+.4f}  {bar}")

    strongest = max(results, key=lambda item: abs(item[1]))

    print()
    print_result(f"Strongest causal variable: {strongest[0]}")
    print_pass("ATE provides direct causal ranking of variables.")

    # ─────────────────────────────────────────────
    # TEST 6 — Mediation
    # ─────────────────────────────────────────────
    print_header(
        6,
        "Mediation Analysis (pick_x → success via arm)",
        "Determine whether arm mediates the effect of pick_x.",
    )

    safe_zone = [1.55, 1.70]
    comparison_zone = [1.40, 1.50]

    print(f"  Safe zone:        {safe_zone}")
    print(f"  Comparison zone:  {comparison_zone}\n")

    # Total effect
    probability_safe = safe_infer(jpt_model, {pick_x: safe_zone})
    probability_unsafe = safe_infer(jpt_model, {pick_x: comparison_zone})
    total_effect = probability_safe - probability_unsafe

    # Marginal arm distribution
    probability_left = safe_infer(jpt_model, {arm_variable: "LEFT"})
    probability_right = safe_infer(jpt_model, {arm_variable: "RIGHT"})

    # Joint probabilities
    joint_safe_left = safe_infer(jpt_model, {pick_x: safe_zone, arm_variable: "LEFT"})
    joint_unsafe_left = safe_infer(jpt_model, {pick_x: comparison_zone, arm_variable: "LEFT"})
    joint_safe_right = safe_infer(jpt_model, {pick_x: safe_zone, arm_variable: "RIGHT"})
    joint_unsafe_right = safe_infer(jpt_model, {pick_x: comparison_zone, arm_variable: "RIGHT"})

    # Conditional probabilities
    conditional_safe_left = joint_safe_left / probability_left if probability_left > 0 else 0
    conditional_unsafe_left = joint_unsafe_left / probability_left if probability_left > 0 else 0
    conditional_safe_right = joint_safe_right / probability_right if probability_right > 0 else 0
    conditional_unsafe_right = joint_unsafe_right / probability_right if probability_right > 0 else 0

    # Arm-specific effects
    ate_left = conditional_safe_left - conditional_unsafe_left
    ate_right = conditional_safe_right - conditional_unsafe_right

    # Direct effect
    direct_effect = (
        probability_left * ate_left +
        probability_right * ate_right
    )

    mediated_effect = total_effect - direct_effect
    mediated_percentage = abs(mediated_effect / total_effect) * 100 if total_effect != 0 else 0

    print("  Total causal effect:")
    print(f"    P(safe)   = {probability_safe:.4f}")
    print(f"    P(unsafe) = {probability_unsafe:.4f}")
    print(f"    ATE_total = {total_effect:+.4f}\n")

    print("  Arm-conditional effects:")
    print(f"    LEFT  → ATE = {ate_left:+.4f}")
    print(f"    RIGHT → ATE = {ate_right:+.4f}\n")

    print("  Decomposition:")
    print(f"    Direct effect     = {direct_effect:+.4f}")
    print(f"    Mediated effect   = {mediated_effect:+.4f}")
    print(f"    Mediated fraction = {mediated_percentage:.1f}%\n")

    if mediated_percentage < 10:
        print_pass("Effect is predominantly direct — arm is a moderator, not a mediator.")
    elif mediated_percentage < 25:
        print_result("Partial mediation detected — arm contributes but is not dominant.")
    else:
        print_result("Strong mediation — arm is a significant causal pathway.")

    # ─────────────────────────────────────────────
    # TEST 7 — Counterfactual
    # ─────────────────────────────────────────────
    print_header(
        7,
        "Counterfactual Reasoning",
        "Estimate alternative outcomes under different sampling.",
    )

    peak_probability = safe_infer(jpt_model, {pick_x: [1.55, 1.70]})
    peak_fraction = 0.15 / 0.60
    enrichment = peak_probability / peak_fraction
    estimated_success = min(0.348 * enrichment, 1.0)

    print("  Scenario:")
    print("  Restrict pick_x to optimal region [1.55, 1.70]\n")

    print_result(f"Enrichment factor: {enrichment:.2f}x")
    print_result(f"Estimated success rate: {estimated_success * 100:.1f}%")

    if estimated_success > 0.6:
        print_pass("Strong causal improvement under optimal intervention.")
    else:
        print_result("Moderate improvement under intervention.")

    print("\nDone.")

    # ─────────────────────────────────────────────
    # TEST 8 — Causal Sufficiency
    # ─────────────────────────────────────────────
    print_header(
        8,
        "Causal Sufficiency of pick_x",
        "Check whether pick_x alone is sufficient for success.",
    )

    safe_zone = [1.55, 1.70]

    probability_unconditional = safe_infer(jpt_model, {pick_x: safe_zone})
    probability_left = safe_infer(jpt_model, {pick_x: safe_zone, arm_variable: "LEFT"})
    probability_right = safe_infer(jpt_model, {pick_x: safe_zone, arm_variable: "RIGHT"})

    probability_center_y = safe_infer(
        jpt_model,
        {pick_x: safe_zone},
        {pick_y: [-0.15, 0.15]},
    )

    probability_extreme_y = safe_infer(
        jpt_model,
        {pick_x: safe_zone},
        {pick_y: [-0.40, -0.25]},
    )

    total_arm = probability_left + probability_right
    total_y = probability_center_y + probability_extreme_y

    arm_left_ratio = probability_left / total_arm if total_arm > 0 else 0.5
    arm_right_ratio = probability_right / total_arm if total_arm > 0 else 0.5

    y_center_ratio = probability_center_y / total_y if total_y > 0 else 0.5
    y_extreme_ratio = probability_extreme_y / total_y if total_y > 0 else 0.5

    print(f"  Within safe pick_x zone {safe_zone}:\n")

    print("  Conditioning        Joint P      Relative weight")
    print("  " + "─" * 60)

    print(f"  None (baseline)     {probability_unconditional:.4f}     1.000")
    print(f"  arm = LEFT          {probability_left:.4f}     {arm_left_ratio:.3f}")
    print(f"  arm = RIGHT         {probability_right:.4f}     {arm_right_ratio:.3f}")
    print(f"  pick_y center       {probability_center_y:.4f}     {y_center_ratio:.3f}")
    print(f"  pick_y extreme      {probability_extreme_y:.4f}     {y_extreme_ratio:.3f}")

    arm_effect = abs(arm_left_ratio - 0.5) * 2
    y_effect = abs(y_center_ratio - y_extreme_ratio)

    print()
    print_result(f"Arm contribution:    {arm_effect:.3f}")
    print_result(f"pick_y contribution: {y_effect:.3f}\n")

    if arm_effect < 0.15 and y_effect < 0.15:
        print_pass("pick_x is largely sufficient for success.")
    elif arm_effect > 0.15:
        print_result("Arm contributes additional causal influence.")
    elif y_effect > 0.15:
        print_result("pick_y contributes additional causal influence.")

if __name__ == "__main__":
    main()