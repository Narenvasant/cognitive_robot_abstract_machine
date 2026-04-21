"""
Fit a degraded JPT. Target: 20-35% baseline success rate.

v9: precision=0.15, min_samples_leaf=600, noise σ=0.18 (approach only)
    milk_end untouched — keeps causal structure for circuit diagnosis
"""

import numpy as np
import pandas as pd

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

ArmDomain = type(
    "ArmDomain",
    (Multinomial,),
    {
        "values": OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)]),
        "labels": OrderedDictProxy([(0, "LEFT"), (1, "RIGHT")]),
    },
)

variables = [
    NumericVariable("pick_approach_x",  precision=0.15),
    NumericVariable("pick_approach_y",  precision=0.15),
    NumericVariable("place_approach_x", precision=0.15),
    NumericVariable("place_approach_y", precision=0.15),
    NumericVariable("milk_end_x",       precision=0.001),   # original
    NumericVariable("milk_end_y",       precision=0.001),   # original
    NumericVariable("milk_end_z",       precision=0.0005),  # original
    SymbolicVariable("pick_arm",        domain=ArmDomain),
]

FIT_COLS         = [v.name for v in variables]
MIN_SAMPLES_LEAF = 600
NOISE_SIGMA      = 0.18
NOISE_COLS       = ["pick_approach_x", "pick_approach_y",
                    "place_approach_x", "place_approach_y"]
RANDOM_SEED      = 42

df = pd.read_csv("pick_and_place_dataframe.csv")
print(f"Loaded {len(df)} rows")
print(f"Arm: LEFT={(df['pick_arm']=='LEFT').sum()}  RIGHT={(df['pick_arm']=='RIGHT').sum()}")

rng = np.random.default_rng(RANDOM_SEED)
df[NOISE_COLS] = df[NOISE_COLS] + rng.normal(0, NOISE_SIGMA, size=(len(df), len(NOISE_COLS)))
print(f"Added noise σ={NOISE_SIGMA} to approach coordinates")

print(f"\nFitting JPT  (precision=0.15, min_samples_leaf={MIN_SAMPLES_LEAF}) ...")
jpt = JPT(variables=variables, min_samples_leaf=MIN_SAMPLES_LEAF)
jpt.fit(df[FIT_COLS])
print(f"Fitted.  Leaves: {len(jpt.leaves)}")

jpt.save("pick_and_place_jpt.json")
print("Saved → pick_and_place_jpt.json")

print("\n── Sanity checks ──")
arm_var = next(v for v in variables if v.name == "pick_arm")
milk_z  = next(v for v in variables if v.name == "milk_end_z")
pick_x  = next(v for v in variables if v.name == "pick_approach_x")

p_left     = jpt.infer(query={arm_var: "LEFT"})
e_z        = jpt.expectation(variables=[milk_z])[milk_z]
e_px_left  = jpt.expectation(variables=[pick_x], evidence={arm_var: "LEFT"})[pick_x]
e_px_right = jpt.expectation(variables=[pick_x], evidence={arm_var: "RIGHT"})[pick_x]

print(f"P(arm=LEFT)       = {p_left:.3f}  (original ≈ {913/1742:.3f})")
print(f"E[milk_end_z]     = {e_z:.4f}  (original ≈ 0.8053)")
print(f"E[pick_x | LEFT]  = {e_px_left:.4f}")
print(f"E[pick_x | RIGHT] = {e_px_right:.4f}")
print(f"(arm correlation preserved if both ≈ original values)")

print(f"\n── Model summary ──")
print(
    f"  Leaves           : {len(jpt.leaves)}\n"
    f"  Training rows    : {len(df)}\n"
    f"  approach prec    : 0.15  (30x coarser than v1)\n"
    f"  min_samples_leaf : {MIN_SAMPLES_LEAF}\n"
    f"  noise σ          : {NOISE_SIGMA}  (approach only, milk_end untouched)\n"
    f"  Target baseline  : 20-35%\n"
    f"  Expected lift    : +15-25pp from causal correction\n"
    f"\n"
    f"  Tuning guide if still off:\n"
    f"    Too many successes (>40%): raise noise to 0.22, prec to 0.18\n"
    f"    Too few successes  (<15%): lower noise to 0.14, prec to 0.12"
)
print("\nDone.")