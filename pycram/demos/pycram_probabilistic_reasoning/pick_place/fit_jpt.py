"""
Fit a JPT over the pick-and-place DataFrame.

Usage:
    python fit_jpt.py

Input:
    pick_and_place_dataframe.csv   (same directory)

Output:
    pick_and_place_jpt.json
"""

import pandas as pd
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.distributions.univariate import Multinomial
from jpt.variables import NumericVariable, SymbolicVariable
from jpt.trees import JPT

# ── Symbolic domain for arm ────────────────────────────────────────────────────
ArmDomain = type('ArmDomain', (Multinomial,), {
    'values': OrderedDictProxy([('LEFT', 0), ('RIGHT', 1)]),
    'labels': OrderedDictProxy([(0, 'LEFT'), (1, 'RIGHT')]),
})

# ── Variables ──────────────────────────────────────────────────────────────────
variables = [
    NumericVariable("pick_approach_x",  precision=0.005),
    NumericVariable("pick_approach_y",  precision=0.005),
    NumericVariable("place_approach_x", precision=0.005),
    NumericVariable("place_approach_y", precision=0.005),
    NumericVariable("milk_end_x",       precision=0.001),
    NumericVariable("milk_end_y",       precision=0.001),
    NumericVariable("milk_end_z",       precision=0.0005),
    SymbolicVariable("pick_arm",        domain=ArmDomain),
]

FIT_COLS = [v.name for v in variables]

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("pick_and_place_dataframe.csv")
print(f"Loaded {len(df)} rows")

# ── Fit ────────────────────────────────────────────────────────────────────────
print("Fitting JPT...")
jpt = JPT(variables=variables, min_samples_leaf=25)
jpt.fit(df[FIT_COLS])

print(f"Fitted.  Leaves: {len(jpt.leaves)}")

# ── Save ───────────────────────────────────────────────────────────────────────
jpt.save("pick_and_place_jpt.json")
print("Saved -> pick_and_place_jpt.json")

# ── Sanity checks ──────────────────────────────────────────────────────────────
print("\n── Sanity checks ──")

arm_var = next(v for v in variables if v.name == "pick_arm")
milk_z  = next(v for v in variables if v.name == "milk_end_z")
pick_x  = next(v for v in variables if v.name == "pick_approach_x")

p_left = jpt.infer(query={arm_var: "LEFT"})
print(f"P(arm=LEFT)       = {p_left:.3f}  (expected ≈ {913/1742:.3f})")

e_z = jpt.expectation(variables=[milk_z])[milk_z]
print(f"E[milk_end_z]     = {e_z:.4f}  (expected ≈ 0.8053)")

e_px_left  = jpt.expectation(variables=[pick_x], evidence={arm_var: "LEFT"})[pick_x]
e_px_right = jpt.expectation(variables=[pick_x], evidence={arm_var: "RIGHT"})[pick_x]
print(f"E[pick_x | LEFT]  = {e_px_left:.4f}")
print(f"E[pick_x | RIGHT] = {e_px_right:.4f}")
print("(both ≈ 1.619 — arm independent of approach position)")

print("\nDone.")