"""
Sequential Runner: JPT Baseline and Causal Circuit Pick-and-Place

Runs both apartment pick-and-place experiments back to back within a single
rclpy context, producing directly comparable output for the paper:

    "Causally-Aware Robot Action Verification via Interventional Probabilistic Circuits"
    SPAI @ IJCAI 2026

Execution Order
---------------
1. JPT-only baseline  (pick_and_place_jpt_baseline.py)
2. JPT + Causal Circuit (pick_and_place_causal.py)

Output Files
------------
    jpt_baseline_output.txt   — full terminal output of the JPT baseline run
    causal_output.txt         — full terminal output of the causal circuit run
    run_comparison_summary.txt — side-by-side comparison of both final summaries

Usage
-----
    python run_experiments.py

Database Configuration (optional)
----------------------------------
By default both demos write to the same PostgreSQL database. To use separate
databases for cleaner record separation:

    export SEMANTIC_DIGITAL_TWIN_DATABASE_URI_JPT=postgresql://.../jpt_db
    export SEMANTIC_DIGITAL_TWIN_DATABASE_URI_CAUSAL=postgresql://.../causal_db

If only the base URI is set, both demos use it:

    export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=postgresql://.../shared_db

Implementation Notes
--------------------
rclpy is initialised once in this runner and its init/shutdown/spin methods
are replaced with no-ops before the demo modules are loaded. This prevents
each demo from attempting to re-initialise or shut down the shared ROS2
context. Both demos receive their own ROS2 node for independent TF and
visualisation publishing.

A single shared _robust_raw_dof patch is installed on ActiveConnection1DOF
after both demo modules are loaded, replacing the per-module closures that
each demo installs independently. Without this, the causal demo's closure
(which references its own module-level world variable, initially None) would
overwrite the JPT demo's closure during module loading, causing grasp IK
failures during the JPT run. The runner-level world variable is updated
automatically via hooks on each demo's world construction function.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from unittest.mock import MagicMock

from typing_extensions import Any

_nav2_mock = MagicMock()
sys.modules["nav2_msgs"]                       = _nav2_mock
sys.modules["nav2_msgs.action"]                = _nav2_mock.action
sys.modules["nav2_msgs.action.NavigateToPose"] = _nav2_mock.action.NavigateToPose


# =============================================================================
# Output Tee: Mirror stdout to terminal and log file simultaneously
# =============================================================================

class OutputTee:
    """
    Mirrors all writes to both the original sys.stdout and a log file.

    Used to capture the full terminal output of each demo run to a file
    without suppressing live terminal output during execution.
    """

    def __init__(self, filepath: str) -> None:
        self._log_file      = open(filepath, "w", buffering=1)
        self._original_stdout = sys.__stdout__

    def write(self, text: str) -> None:
        self._original_stdout.write(text)
        self._log_file.write(text)

    def flush(self) -> None:
        self._original_stdout.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        return self._original_stdout.fileno()

    def close(self) -> None:
        self._log_file.close()


# =============================================================================
# Output File Paths
# =============================================================================

OUTPUT_DIRECTORY         = os.path.dirname(os.path.abspath(__file__))
JPT_BASELINE_LOG_PATH    = os.path.join(OUTPUT_DIRECTORY, "jpt_baseline_output.txt")
CAUSAL_LOG_PATH          = os.path.join(OUTPUT_DIRECTORY, "causal_output.txt")
COMPARISON_SUMMARY_PATH  = os.path.join(OUTPUT_DIRECTORY, "run_comparison_summary.txt")

JPT_BASELINE_MODULE_FILE = "pick_and_place_jpt_baseline.py"
CAUSAL_MODULE_FILE       = "pick_and_place_causal.py"

DEFAULT_DATABASE_URI = (
    "postgresql://semantic_digital_twin:naren"
    "@localhost:5432/probabilistic_reasoning"
)


# =============================================================================
# ROS2 Initialisation (shared across both demo runs)
# =============================================================================

import rclpy as _rclpy

_rclpy.init()
_shared_ros_node = _rclpy.create_node("pick_and_place_experiment_runner")
_ros_spin_thread = threading.Thread(
    target=_rclpy.spin, args=(_shared_ros_node,), daemon=True
)
_ros_spin_thread.start()
print("[runner] Shared ROS2 node started.")

_rclpy.init     = lambda *args, **kwargs: None
_rclpy.shutdown = lambda *args, **kwargs: None
_rclpy.spin     = lambda *args, **kwargs: None


# =============================================================================
# Demo Module Loading
# =============================================================================

def _load_demo_module(filename: str) -> Any:
    """
    Load a demo module from the experiment directory by filename.

    The module is registered in sys.modules before execution so that
    dataclass definitions and other module-level constructs resolve
    correctly during import.

    Parameters
    ----------
    filename:
        Python filename of the demo module, relative to OUTPUT_DIRECTORY.

    Returns
    -------
    module
        The loaded and executed demo module.
    """
    module_name = (
        filename
        .replace(".py", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
    module_path = os.path.join(OUTPUT_DIRECTORY, filename)
    spec        = importlib.util.spec_from_file_location(module_name, module_path)
    module      = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


print("\n" + "=" * 64)
print("  Loading experiment modules ...")
print("=" * 64)

jpt_baseline_module = _load_demo_module(JPT_BASELINE_MODULE_FILE)
causal_module       = _load_demo_module(CAUSAL_MODULE_FILE)


causal_module.JPT_VARIABLES            = jpt_baseline_module.JPT_VARIABLES
causal_module.ArmChoiceDomain          = jpt_baseline_module.ArmChoiceDomain
causal_module.JPT_MIN_SAMPLES_PER_LEAF = jpt_baseline_module.JPT_MIN_SAMPLES_PER_LEAF

print("[runner] JPT variable definitions shared between modules.")



_runner_active_world = None

from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF as _ActiveConnection1DOF,
)

def _shared_robust_raw_dof(self) -> Any:
    """Redirect stale _world references to the currently active world."""
    target_world = self._world
    if (
        target_world is None
        or len(target_world._world_entity_hash_table) == 0
        or len(target_world.degrees_of_freedom) == 0
    ):
        if _runner_active_world is not None:
            target_world = _runner_active_world
            self._world  = target_world
    return target_world.get_degree_of_freedom_by_id(self.dof_id)

_ActiveConnection1DOF.raw_dof = property(_shared_robust_raw_dof)
print("[runner] Shared ActiveConnection1DOF.raw_dof patch installed.")


# =============================================================================
# World Construction Hooks
# =============================================================================

_jpt_original_build_world    = jpt_baseline_module._build_apartment_world
_causal_original_build_world = causal_module._build_apartment_world


def _jpt_build_world_hook(apartment_urdf_path: Any) -> tuple:
    global _runner_active_world
    world, robot = _jpt_original_build_world(apartment_urdf_path)
    _runner_active_world = world
    print("[runner] Active world updated for JPT baseline run.")
    return world, robot


def _causal_build_world_hook(apartment_urdf_path: Any) -> tuple:
    global _runner_active_world
    world, robot = _causal_original_build_world(apartment_urdf_path)
    _runner_active_world = world
    print("[runner] Active world updated for causal circuit run.")
    return world, robot


jpt_baseline_module._build_apartment_world = _jpt_build_world_hook
causal_module._build_apartment_world       = _causal_build_world_hook

print("[runner] World construction hooks installed.")


# =============================================================================
# Database URI Configuration
# =============================================================================

jpt_baseline_module.DATABASE_URI = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI_JPT",
    os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI", DEFAULT_DATABASE_URI),
)
causal_module.DATABASE_URI = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI_CAUSAL",
    os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI", DEFAULT_DATABASE_URI),
)

print(f"[runner] JPT baseline database    : {jpt_baseline_module.DATABASE_URI}")
print(f"[runner] Causal circuit database  : {causal_module.DATABASE_URI}")
print(f"\n[runner] Output files:")
print(f"    JPT baseline : {JPT_BASELINE_LOG_PATH}")
print(f"    Causal       : {CAUSAL_LOG_PATH}")
print(f"    Summary      : {COMPARISON_SUMMARY_PATH}")


# =============================================================================
# Experiment 1: JPT Baseline
# =============================================================================

print("\n" + "=" * 64)
print("  EXPERIMENT 1 OF 2: JPT Baseline (blind resampling)")
print("=" * 64)

jpt_tee = OutputTee(JPT_BASELINE_LOG_PATH)
sys.stdout = jpt_tee

jpt_start_time  = time.time()
jpt_statistics  = None

try:
    jpt_baseline_module.run_pick_and_place_jpt_baseline()
    jpt_statistics = getattr(jpt_baseline_module, "_last_run_statistics", None)
except Exception as error:
    import traceback as _traceback
    print(f"\n[runner] JPT baseline raised an exception: {error}")
    _traceback.print_exc()

jpt_elapsed_seconds = time.time() - jpt_start_time

sys.stdout = sys.__stdout__
jpt_tee.close()

print(
    f"\n[runner] JPT baseline complete in "
    f"{jpt_elapsed_seconds / 3600:.2f}h ({jpt_elapsed_seconds:.0f}s)"
)
print(f"[runner] Output saved to {JPT_BASELINE_LOG_PATH}")


# =============================================================================
# Experiment 2: JPT + Causal Circuit
# =============================================================================

print("\n" + "=" * 64)
print("  EXPERIMENT 2 OF 2: JPT + Causal Circuit")
print("=" * 64)

causal_tee = OutputTee(CAUSAL_LOG_PATH)
sys.stdout = causal_tee

causal_start_time = time.time()
causal_statistics = None

try:
    causal_module.run_pick_and_place_with_causal_correction()
    causal_statistics = getattr(causal_module, "_last_run_statistics", None)
except Exception as error:
    import traceback as _traceback
    print(f"\n[runner] Causal circuit run raised an exception: {error}")
    _traceback.print_exc()

causal_elapsed_seconds = time.time() - causal_start_time

sys.stdout = sys.__stdout__
causal_tee.close()

print(
    f"\n[runner] Causal circuit run complete in "
    f"{causal_elapsed_seconds / 3600:.2f}h ({causal_elapsed_seconds:.0f}s)"
)
print(f"[runner] Output saved to {CAUSAL_LOG_PATH}")


# =============================================================================
# ROS2 Shutdown
# =============================================================================

print("\n[runner] Shutting down ROS2 ...")
_shared_ros_node.destroy_node()
_ros_spin_thread.join(timeout=3.0)
try:
    _rclpy.context.get_default_context().try_shutdown()
except Exception:
    pass
print("[runner] ROS2 shut down.")


# =============================================================================
# Comparison Summary
# =============================================================================

def _extract_final_summary_block(log_path: str) -> str:
    """
    Extract the final summary block from a demo log file.

    Locates the last pair of separator lines (lines beginning with 30 or more
    equals signs) and returns all content from the second-to-last separator
    to the end of the file. Returns an error message if the file cannot be
    read or no separator is found.

    Parameters
    ----------
    log_path:
        Path to the demo log file produced by OutputTee.

    Returns
    -------
    str
        The extracted summary block, or an error message string.
    """
    try:
        lines = open(log_path).readlines()
        separator_indices = [
            index for index, line in enumerate(lines)
            if line.strip().startswith("=" * 30)
        ]
        if len(separator_indices) >= 2:
            return "".join(lines[separator_indices[-2]:])
        elif separator_indices:
            return "".join(lines[separator_indices[-1]:])
        return "(summary block not found in log)"
    except Exception as error:
        return f"(could not read log: {error})"


total_elapsed_seconds = jpt_elapsed_seconds + causal_elapsed_seconds

summary_lines = [
    "=" * 64,
    "  EXPERIMENT COMPARISON SUMMARY",
    "  Causally-Aware Robot Action Verification via",
    "  Interventional Probabilistic Circuits — SPAI @ IJCAI 2026",
    "=" * 64,
    "",
    f"  JPT baseline wall-clock time    : "
    f"{jpt_elapsed_seconds / 3600:.2f}h  ({jpt_elapsed_seconds:.0f}s)",
    f"  Causal circuit wall-clock time  : "
    f"{causal_elapsed_seconds / 3600:.2f}h  ({causal_elapsed_seconds:.0f}s)",
    f"  Total experiment time           : "
    f"{total_elapsed_seconds / 3600:.2f}h  ({total_elapsed_seconds:.0f}s)",
    "",
    "-" * 64,
    "  EXPERIMENT 1: JPT BASELINE — FINAL SUMMARY",
    "-" * 64,
    _extract_final_summary_block(JPT_BASELINE_LOG_PATH),
    "",
    "-" * 64,
    "  EXPERIMENT 2: JPT + CAUSAL CIRCUIT — FINAL SUMMARY",
    "-" * 64,
    _extract_final_summary_block(CAUSAL_LOG_PATH),
    "",
    "=" * 64,
]

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(COMPARISON_SUMMARY_PATH, "w") as summary_file:
    summary_file.write(summary_text)

print(f"\n[runner] Comparison summary saved to {COMPARISON_SUMMARY_PATH}")