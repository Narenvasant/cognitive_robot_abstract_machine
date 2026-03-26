"""
Fit a JPT (Joint Probability Tree) over the pick-and-place DataFrame.

Usage:
    python fit_jpt.py

Input:
    pick_and_place_dataframe.csv   (same directory)

Output:
    pick_and_place_jpt.json        (saved JPT model)
"""

import pandas as pd
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("pick_place/pick_and_place_dataframe.csv")

# Columns used for fitting (exclude plan_id and arm_encoded redundant column)
FIT_COLS = [
    "pick_approach_x",
    "pick_approach_y",
    "place_approach_x",
    "place_approach_y",
    "milk_end_x",
    "milk_end_y",
    "milk_end_z",
    "pick_arm",
]

print(f"Fitting JPT on {len(df)} rows, {len(FIT_COLS)} variables")

# ── Define variables ───────────────────────────────────────────────────────────
variables = [
    NumericVariable("pick_approach_x",  precision=0.005),
    NumericVariable("pick_approach_y",  precision=0.005),
    NumericVariable("place_approach_x", precision=0.005),
    NumericVariable("place_approach_y", precision=0.005),
    NumericVariable("milk_end_x",       precision=0.001),
    NumericVariable("milk_end_y",       precision=0.001),
    NumericVariable("milk_end_z",       precision=0.0005),
    SymbolicVariable("pick_arm",        domain=["LEFT", "RIGHT"]),
]

# ── Fit ────────────────────────────────────────────────────────────────────────
jpt = JPT(variables=variables, min_samples_leaf=25)
jpt.fit(df[FIT_COLS])

print(f"JPT fitted.")
print(f"  Leaves    : {len(jpt.leaves)}")
print(f"  Variables : {[v.name for v in variables]}")

# ── Save ───────────────────────────────────────────────────────────────────────
JPT_PATH = "pick_place/pick_and_place_jpt.json"
jpt.save(JPT_PATH)
print(f"\nSaved -> {JPT_PATH}")

# ── Sanity checks ──────────────────────────────────────────────────────────────
print("\n── Sanity checks ──")

arm_var   = next(v for v in variables if v.name == "pick_arm")
milk_z    = next(v for v in variables if v.name == "milk_end_z")
pick_x    = next(v for v in variables if v.name == "pick_approach_x")

# 1. Marginal P(arm=LEFT)
p_left = jpt.infer(query={arm_var: "LEFT"})
print(f"P(arm=LEFT)  = {p_left:.3f}  (expected ≈ {913/1742:.3f})")

# 2. E[milk_end_z]
e_z = jpt.expectation(targets=[milk_z])[milk_z]
print(f"E[milk_end_z] = {e_z:.4f}  (expected ≈ 0.8053)")

# 3. E[pick_approach_x | arm=LEFT]
e_px_left = jpt.expectation(
    targets=[pick_x],
    evidence={arm_var: "LEFT"}
)[pick_x]
e_px_right = jpt.expectation(
    targets=[pick_x],
    evidence={arm_var: "RIGHT"}
)[pick_x]
print(f"E[pick_approach_x | LEFT]  = {e_px_left:.4f}")
print(f"E[pick_approach_x | RIGHT] = {e_px_right:.4f}")
print("(these should be close to each other ≈ 1.619 — arm is independent of approach position)")

print("\nDone.")