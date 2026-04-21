"""
Degraded JPT Training Script

Fits a deliberately coarsened Joint Probability Tree (JPT) on the
pick-and-place training data to reproduce the degraded planning distribution
condition described in:

    "Causally-Aware Robot Action Verification via Interventional Probabilistic Circuits"
    SPAI @ IJCAI 2026

Purpose
-------
The degraded JPT is used to evaluate the Causal Circuit under a weak planning
distribution, where systematic sampling failures arise from coarse leaf
precision and approach-coordinate noise. This condition tests whether the
Causal Circuit can convert failures into successes (correctness role) rather
than merely accelerating recovery from rare failures (efficiency role).

Degradation Strategy
--------------------
Two modifications are applied to the training data and model hyperparameters:

1. Gaussian noise (sigma=0.18) is added to all four approach coordinates
   (pick_approach_x, pick_approach_y, place_approach_x, place_approach_y)
   before fitting. This spreads approach positions beyond the feasible region,
   causing the sampler to frequently draw values that the simulator rejects.

2. Coarse leaf precision (0.15, 30x coarser than the high-quality model) and
   a high minimum samples per leaf (600) produce a small number of broad
   leaves that cover constraint-violating regions indiscriminately.

The milk_end variables are left untouched so that the causal structure of
the effect variable (milk_end_z) is preserved for Causal Circuit diagnosis.

Output
------
    pick_and_place_jpt.json — overwrites the existing model file.

    Back up the high-quality model before running this script:
        cp pick_and_place_jpt.json pick_and_place_jpt_high_quality.json

Hyperparameter Tuning Guide
---------------------------
If the resulting baseline success rate falls outside the 20-35% target:
    Too many successes (>40%): increase noise sigma to 0.22, precision to 0.18
    Too few successes  (<15%): decrease noise sigma to 0.14, precision to 0.12
"""

import numpy as np
import pandas as pd

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


# =============================================================================
# Configuration
# =============================================================================

TRAINING_CSV_PATH:   str   = "pick_and_place_dataframe.csv"
OUTPUT_MODEL_PATH:   str   = "pick_and_place_jpt.json"

APPROACH_PRECISION:  float = 0.15
EFFECT_PRECISION_XY: float = 0.001
EFFECT_PRECISION_Z:  float = 0.0005
MIN_SAMPLES_PER_LEAF: int  = 600

APPROACH_NOISE_SIGMA: float = 0.18
APPROACH_NOISE_COLUMNS: list = [
    "pick_approach_x",
    "pick_approach_y",
    "place_approach_x",
    "place_approach_y",
]

RANDOM_SEED: int = 42

REFERENCE_P_ARM_LEFT:  float = 913 / 1742
REFERENCE_E_MILK_END_Z: float = 0.8053


# =============================================================================
# Variable Definitions
# =============================================================================

ArmChoiceDomain = type(
    "ArmChoiceDomain",
    (Multinomial,),
    {
        "values": OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)]),
        "labels": OrderedDictProxy([(0, "LEFT"), (1, "RIGHT")]),
    },
)

VARIABLES = [
    NumericVariable("pick_approach_x",  precision=APPROACH_PRECISION),
    NumericVariable("pick_approach_y",  precision=APPROACH_PRECISION),
    NumericVariable("place_approach_x", precision=APPROACH_PRECISION),
    NumericVariable("place_approach_y", precision=APPROACH_PRECISION),
    NumericVariable("milk_end_x",       precision=EFFECT_PRECISION_XY),
    NumericVariable("milk_end_y",       precision=EFFECT_PRECISION_XY),
    NumericVariable("milk_end_z",       precision=EFFECT_PRECISION_Z),
    SymbolicVariable("pick_arm",        domain=ArmChoiceDomain),
]

FIT_COLUMNS = [variable.name for variable in VARIABLES]


# =============================================================================
# Data Loading and Noise Injection
# =============================================================================

print("=" * 64)
print("  Degraded JPT Training")
print(f"  Approach precision   : {APPROACH_PRECISION}  (30x coarser than high-quality)")
print(f"  Min samples per leaf : {MIN_SAMPLES_PER_LEAF}")
print(f"  Approach noise sigma : {APPROACH_NOISE_SIGMA}")
print(f"  Target baseline rate : 20-35%")
print("=" * 64)

print(f"\n[data] Loading training data from {TRAINING_CSV_PATH} ...")
training_data = pd.read_csv(TRAINING_CSV_PATH)
print(f"[data] Loaded {len(training_data)} rows")
print(
    f"[data] Arm distribution: "
    f"LEFT={(training_data['pick_arm'] == 'LEFT').sum()}  "
    f"RIGHT={(training_data['pick_arm'] == 'RIGHT').sum()}"
)

print(f"\n[noise] Injecting Gaussian noise (sigma={APPROACH_NOISE_SIGMA}) "
      f"into approach coordinates ...")
random_generator = np.random.default_rng(RANDOM_SEED)
noise = random_generator.normal(
    loc=0,
    scale=APPROACH_NOISE_SIGMA,
    size=(len(training_data), len(APPROACH_NOISE_COLUMNS)),
)
training_data[APPROACH_NOISE_COLUMNS] = training_data[APPROACH_NOISE_COLUMNS] + noise
print(f"[noise] Noise injected into: {APPROACH_NOISE_COLUMNS}")
print("[noise] milk_end variables left untouched to preserve causal structure.")


# =============================================================================
# Model Fitting
# =============================================================================

print(
    f"\n[fit] Fitting JPT  "
    f"(precision={APPROACH_PRECISION}, "
    f"min_samples_per_leaf={MIN_SAMPLES_PER_LEAF}) ..."
)
degraded_jpt = JPT(variables=VARIABLES, min_samples_leaf=MIN_SAMPLES_PER_LEAF)
degraded_jpt.fit(training_data[FIT_COLUMNS])
print(f"[fit] Fitted. Leaves: {len(degraded_jpt.leaves)}")


# =============================================================================
# Model Saving
# =============================================================================

degraded_jpt.save(OUTPUT_MODEL_PATH)
print(f"\n[save] Model saved to {OUTPUT_MODEL_PATH}")
print("[save] NOTE: This overwrites the existing model file.")
print("[save] Back up the high-quality model first if needed:")
print("[save]     cp pick_and_place_jpt.json pick_and_place_jpt_high_quality.json")


# =============================================================================
# Sanity Checks
# =============================================================================

print("\n" + "-" * 64)
print("  Sanity Checks")
print("-" * 64)

arm_variable    = next(v for v in VARIABLES if v.name == "pick_arm")
milk_z_variable = next(v for v in VARIABLES if v.name == "milk_end_z")
pick_x_variable = next(v for v in VARIABLES if v.name == "pick_approach_x")

probability_arm_left        = degraded_jpt.infer(query={arm_variable: "LEFT"})
expected_milk_end_z         = degraded_jpt.expectation(variables=[milk_z_variable])[milk_z_variable]
expected_pick_x_given_left  = degraded_jpt.expectation(
    variables=[pick_x_variable],
    evidence={arm_variable: "LEFT"},
)[pick_x_variable]
expected_pick_x_given_right = degraded_jpt.expectation(
    variables=[pick_x_variable],
    evidence={arm_variable: "RIGHT"},
)[pick_x_variable]

print(
    f"  P(arm=LEFT)          = {probability_arm_left:.3f}  "
    f"(reference: {REFERENCE_P_ARM_LEFT:.3f})"
)
print(
    f"  E[milk_end_z]        = {expected_milk_end_z:.4f}  "
    f"(reference: {REFERENCE_E_MILK_END_Z:.4f})"
)
print(f"  E[pick_x | LEFT]     = {expected_pick_x_given_left:.4f}")
print(f"  E[pick_x | RIGHT]    = {expected_pick_x_given_right:.4f}")
print(
    "  Arm correlation is preserved if E[pick_x | LEFT] and "
    "E[pick_x | RIGHT] are close to the reference values."
)


# =============================================================================
# Model Summary
# =============================================================================

print("\n" + "-" * 64)
print("  Model Summary")
print("-" * 64)
print(f"  Leaves               : {len(degraded_jpt.leaves)}")
print(f"  Training rows        : {len(training_data)}")
print(f"  Approach precision   : {APPROACH_PRECISION}")
print(f"  Min samples per leaf : {MIN_SAMPLES_PER_LEAF}")
print(f"  Approach noise sigma : {APPROACH_NOISE_SIGMA}  (approach coordinates only)")
print(f"  milk_end variables   : untouched (original precision)")
print(f"  Target baseline rate : 20-35%")
print(f"  Expected causal lift : +15-25 percentage points")
print()
print("  Tuning guide:")
print("    Baseline > 40%  ->  increase noise sigma to 0.22, precision to 0.18")
print("    Baseline < 15%  ->  decrease noise sigma to 0.14, precision to 0.12")
print("-" * 64)
print("\n[done] Degraded JPT training complete.")