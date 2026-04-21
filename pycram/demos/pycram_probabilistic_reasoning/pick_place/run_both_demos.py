"""
Sequential runner for both apartment pick-and-place demos.

Runs the JPT-only demo first, then the causal circuit demo, sharing
a single rclpy context. Both demos publish to RViz (different nodes,
same topics — RViz shows whichever is active).

All output is mirrored to:
  - Terminal (live)
  - jpt_demo_output.txt
  - causal_demo_output.txt
  - run_summary.txt  (final comparison table)

Usage:
    python run_both_demos.py

Separate databases (optional):
    SEMANTIC_DIGITAL_TWIN_DATABASE_URI_JPT=postgresql://.../jpt_db
    SEMANTIC_DIGITAL_TWIN_DATABASE_URI_CAUSAL=postgresql://.../causal_db
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Tee: write to both terminal and a file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    """Mirrors writes to sys.stdout and a log file."""
    def __init__(self, filepath: str):
        self._file   = open(filepath, "w", buffering=1)
        self._stdout = sys.__stdout__

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def fileno(self):
        return self._stdout.fileno()

    def close(self):
        self._file.close()


_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
_JPT_LOG    = os.path.join(_OUTPUT_DIR, "jpt_demo_output.txt")
_CAUSAL_LOG = os.path.join(_OUTPUT_DIR, "causal_demo_output.txt")
_SUMMARY    = os.path.join(_OUTPUT_DIR, "run_summary.txt")

# ---------------------------------------------------------------------------
# Mock nav2_msgs before any import touches giskardpy
# ---------------------------------------------------------------------------
_nav2_mock = MagicMock()
sys.modules["nav2_msgs"]                       = _nav2_mock
sys.modules["nav2_msgs.action"]                = _nav2_mock.action
sys.modules["nav2_msgs.action.NavigateToPose"] = _nav2_mock.action.NavigateToPose

# ---------------------------------------------------------------------------
# Init rclpy ONCE, patch its module IN-PLACE so demo files that do
# 'import rclpy' at the top of their file see the patched version.
# Replacing sys.modules["rclpy"] does NOT work — demo modules already
# hold a direct reference to the real module object via their own import.
# Mutating the real module object's attributes affects all those references.
# ---------------------------------------------------------------------------
import rclpy as _real_rclpy

_real_rclpy.init()
_shared_node     = _real_rclpy.create_node("pick_and_place_apartment_runner")
_spin_thread     = threading.Thread(
    target=_real_rclpy.spin, args=(_shared_node,), daemon=True
)
_spin_thread.start()
print("  [ros] Shared ROS node started.")

def _noop(*args, **kwargs):
    pass

def _shared_create_node(name, **kwargs):
    # Return uniquely-named nodes so both demos can publish independently
    return _real_rclpy.create_node.__wrapped__(name, **kwargs) \
        if hasattr(_real_rclpy.create_node, '__wrapped__') \
        else _shared_node

# Patch in-place BEFORE loading demo modules
_real_rclpy.init        = _noop
_real_rclpy.shutdown    = _noop
_real_rclpy.spin        = _noop
# create_node: let each demo get its own node for independent publishing
# We store the real create_node before overwriting
_real_create_node = None  # will be set below after we capture it

# ---------------------------------------------------------------------------
# Load demo modules
# ---------------------------------------------------------------------------

_DEMO_DIR = _OUTPUT_DIR
JPT_DEMO_FILE    = "pick_place_demo_apartment_jpt.py"
CAUSAL_DEMO_FILE = "pick_place_demo_apartment_jpt_and _causal.py"


def _load_demo(filename: str):
    module_name = (
        filename
        .replace(".py", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
    path = os.path.join(_DEMO_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod   # register before exec for @dataclass
    spec.loader.exec_module(mod)
    return mod


print("=" * 64)
print("  Loading demo modules ...")
print("=" * 64)

_jpt_demo    = _load_demo(JPT_DEMO_FILE)
_causal_demo = _load_demo(CAUSAL_DEMO_FILE)

# Share JPT_VARIABLES and ArmChoiceDomain so pyjpt deserialises consistently
# (both demos define ArmChoiceDomain independently; the model was serialised
# with one specific class object — sharing ensures apply() works correctly)
_causal_demo.JPT_VARIABLES           = _jpt_demo.JPT_VARIABLES
_causal_demo.ArmChoiceDomain         = _jpt_demo.ArmChoiceDomain
_causal_demo.JPT_MIN_SAMPLES_PER_LEAF = _jpt_demo.JPT_MIN_SAMPLES_PER_LEAF

# ---------------------------------------------------------------------------
# Critical fix: reinstall a single shared _robust_raw_dof that reads from
# a runner-level variable, not from either demo module's _APARTMENT_WORLD.
#
# Root cause of pickup failure when running via runner:
#   Both demos define _robust_raw_dof as a closure over their own module-level
#   _APARTMENT_WORLD = None. The causal demo is loaded second, so its version
#   of _robust_raw_dof overwrites the JPT demo's version on _AC1DOF.raw_dof.
#   During the JPT demo's execution, _causal_demo._APARTMENT_WORLD is still
#   None (only set inside pick_and_place_demo_apartment_causal()), so the
#   stale world reference is never repaired → grasp IK fails → pickup fails.
#
# Fix: install one shared _robust_raw_dof that reads from _RUNNER_WORLD,
# a runner-level variable updated at the start of each demo before execution.
# ---------------------------------------------------------------------------

_RUNNER_WORLD = None   # updated before each demo runs

from semantic_digital_twin.world_description.connections import ActiveConnection1DOF as _AC1DOF

def _shared_robust_raw_dof(self):
    target_world = self._world
    if (target_world is None or
            len(target_world._world_entity_hash_table) == 0 or
            len(target_world.degrees_of_freedom) == 0):
        if _RUNNER_WORLD is not None:
            target_world = _RUNNER_WORLD
            self._world  = target_world
    return target_world.get_degree_of_freedom_by_id(self.dof_id)

_AC1DOF.raw_dof = property(_shared_robust_raw_dof)
print("  [runner] Installed shared _robust_raw_dof patch.")

# Hook both demos' _build_world to automatically update _RUNNER_WORLD
# when each demo constructs its world, before any robot motion happens.

_jpt_original_build_world    = _jpt_demo._build_world
_causal_original_build_world = _causal_demo._build_world

def _jpt_build_world_hook(apartment_urdf_path):
    global _RUNNER_WORLD
    world, robot = _jpt_original_build_world(apartment_urdf_path)
    _RUNNER_WORLD = world
    print("  [runner] _RUNNER_WORLD updated for JPT demo.")
    return world, robot

def _causal_build_world_hook(apartment_urdf_path):
    global _RUNNER_WORLD
    world, robot = _causal_original_build_world(apartment_urdf_path)
    _RUNNER_WORLD = world
    print("  [runner] _RUNNER_WORLD updated for causal demo.")
    return world, robot

_jpt_demo._build_world    = _jpt_build_world_hook
_causal_demo._build_world = _causal_build_world_hook

print("  [runner] Hooked _build_world for automatic _RUNNER_WORLD updates.")

# ---------------------------------------------------------------------------
# Database URIs
# ---------------------------------------------------------------------------

_DEFAULT_URI = "postgresql://semantic_digital_twin:naren@localhost:5432/probabilistic_reasoning"

_jpt_demo.DATABASE_URI = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI_JPT",
    os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI", _DEFAULT_URI),
)
_causal_demo.DATABASE_URI = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI_CAUSAL",
    os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI", _DEFAULT_URI),
)

# ---------------------------------------------------------------------------
# Run JPT demo — output mirrored to jpt_demo_output.txt
# ---------------------------------------------------------------------------

print(f"\n  Output will be saved to:")
print(f"    JPT demo   : {_JPT_LOG}")
print(f"    Causal demo: {_CAUSAL_LOG}")
print(f"    Summary    : {_SUMMARY}")

print("\n" + "=" * 64)
print("  STARTING DEMO 1: JPT-only")
print("=" * 64)

_jpt_tee = _Tee(_JPT_LOG)
sys.stdout  = _jpt_tee

_jpt_start = time.time()
_jpt_stats = None
try:
    _jpt_demo.pick_and_place_demo_apartment_jpt()
    _jpt_stats = _jpt_demo._last_statistics if hasattr(_jpt_demo, '_last_statistics') else None
except Exception as _e:
    import traceback
    print(f"\n[runner] JPT demo raised an exception: {_e}")
    traceback.print_exc()
_jpt_elapsed = time.time() - _jpt_start

sys.stdout = sys.__stdout__
_jpt_tee.close()
print(f"\n[runner] JPT demo finished in {_jpt_elapsed / 3600:.2f}h ({_jpt_elapsed:.0f}s)")
print(f"[runner] Output saved to {_JPT_LOG}")

# ---------------------------------------------------------------------------
# Run causal demo — output mirrored to causal_demo_output.txt
# ---------------------------------------------------------------------------

print("\n" + "=" * 64)
print("  STARTING DEMO 2: JPT + CausalCircuit")
print("=" * 64)

_causal_tee = _Tee(_CAUSAL_LOG)
sys.stdout  = _causal_tee

_causal_start = time.time()
_causal_stats = None
try:
    _causal_demo.pick_and_place_demo_apartment_causal()
    _causal_stats = _causal_demo._last_statistics if hasattr(_causal_demo, '_last_statistics') else None
except Exception as _e:
    import traceback
    print(f"\n[runner] Causal demo raised an exception: {_e}")
    traceback.print_exc()
_causal_elapsed = time.time() - _causal_start

sys.stdout = sys.__stdout__
_causal_tee.close()
print(f"\n[runner] Causal demo finished in {_causal_elapsed / 3600:.2f}h ({_causal_elapsed:.0f}s)")
print(f"[runner] Output saved to {_CAUSAL_LOG}")

# ---------------------------------------------------------------------------
# Shutdown ROS
# ---------------------------------------------------------------------------

print("\n[runner] Shutting down ROS ...")
_shared_node.destroy_node()
_spin_thread.join(timeout=3.0)
try:
    _real_rclpy.context.get_default_context().try_shutdown()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Write comparison summary
# ---------------------------------------------------------------------------

def _read_last_summary(log_path: str) -> str:
    """Extract the final summary block from a demo log file."""
    try:
        lines = open(log_path).readlines()
        # Find the last occurrence of the summary separator
        sep_indices = [i for i, l in enumerate(lines) if l.strip().startswith("=" * 30)]
        if len(sep_indices) >= 2:
            start = sep_indices[-2]
            return "".join(lines[start:])
        elif sep_indices:
            return "".join(lines[sep_indices[-1]:])
        return "(summary not found in log)"
    except Exception as e:
        return f"(could not read log: {e})"


summary_lines = [
    "=" * 64,
    "  COMPARISON SUMMARY",
    "=" * 64,
    "",
    f"  JPT demo wall-clock time    : {_jpt_elapsed / 3600:.2f}h  ({_jpt_elapsed:.0f}s)",
    f"  Causal demo wall-clock time : {_causal_elapsed / 3600:.2f}h  ({_causal_elapsed:.0f}s)",
    f"  Total                       : {(_jpt_elapsed + _causal_elapsed) / 3600:.2f}h",
    "",
    "─" * 64,
    "  JPT DEMO FINAL SUMMARY",
    "─" * 64,
    _read_last_summary(_JPT_LOG),
    "",
    "─" * 64,
    "  CAUSAL DEMO FINAL SUMMARY",
    "─" * 64,
    _read_last_summary(_CAUSAL_LOG),
    "",
    "=" * 64,
]

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(_SUMMARY, "w") as f:
    f.write(summary_text)

print(f"\n[runner] Comparison summary saved to {_SUMMARY}")