"""
Apartment world: JPT-guided pick-and-place with GCS navigation and
causal failure diagnosis with active parameter correction.

Uses the open-world JPT (pick_and_place_jpt.json), trained on 1742 successful
open-world Batch 1 plans, to guide approach-position and arm sampling in the
apartment world. The open-world JPT transfers directly because pick/place
mechanics are identical between the two worlds — only navigation geometry
differs, which GCS handles transparently.

JPT variable mapping (open-world → apartment):
    pick_approach_x/y   →  counter_approach_x/y
    place_approach_x/y  →  table_approach_x/y
    milk_end_x/y/z      →  unchanged
    pick_arm            →  unchanged

Causal failure diagnosis with active correction
------------------------------------------------
On each failed iteration, CausalCircuit.diagnose_failure() identifies the
primary causal variable and recommends a corrective value. The demo then
applies a CausalSamplingCorrection for the immediately following iteration:
the primary cause variable is clamped to the recommended region, while all
other variables are still drawn freely from the JPT joint distribution.

This closes the feedback loop:
    failure → causal diagnosis → parameter correction → re-attempt

The CausalSamplingCorrection is designed to be reusable for any JPT-based
robot plan that uses a CausalCircuit. It is task-agnostic — it operates on
variable names and recommended values without any knowledge of the specific
action being planned.

Correction strategy
-------------------
- After a failure, one corrected sample is attempted immediately.
- If the corrected attempt succeeds, sampling reverts to the unconstrained JPT.
- If the corrected attempt also fails, sampling reverts to the unconstrained JPT
  rather than recursively correcting — this prevents the system from getting
  stuck following a bad recommendation.
- Corrections are tracked separately in the run statistics so the contribution
  of the causal circuit to the overall success rate can be measured directly.

GCS return navigation
---------------------
After every iteration (success or failure), the robot navigates back to the
fixed start zone using a single GCS call from the robot's actual current
position in the world. This avoids the previous two-leg approach
(table → counter → start) which used phantom planned positions as the GCS
start point, causing path planning failures and incorrect fallback straight-line
routes that crossed the kitchen counter.

CausalCircuit is built once at startup from a ProbModelJPT (probabilistic_model
variant), which exposes the .probabilistic_circuit attribute required by
CausalCircuit.from_jpt(). The pyjpt model is kept separately for sampling
because it is faster at inference time.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rclpy
import sqlalchemy.types as sqlalchemy_types
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base
from pycram.robot_plans import NavigateAction, ParkArmsAction, PickUpAction, PlaceAction

from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT as JointProbabilityTree
from jpt.variables import NumericVariable, SymbolicVariable

from probabilistic_model.learning.jpt.jpt import JPT as ProbModelJPT
from probabilistic_model.learning.jpt.variables import (
    Continuous as ProbContinuous,
    Symbolic as ProbSymbolic,
)
from random_events.set import Set as RESet
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import SumUnit as _SumUnit

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MdVtreeNode,
)

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose, Quaternion
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import BoundingBox, FileMesh
from semantic_digital_twin.world_description.graph_of_convex_sets import GraphOfConvexSets
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Library patch: SumUnit.simplify() version mismatch
#
# This version of probabilistic_model calls self.add_subcircuit(..., mount=False)
# but add_subcircuit() does not accept a mount= keyword argument.
# Patched here to avoid modifying the library source.
# ---------------------------------------------------------------------------

def _patched_sum_simplify(self) -> None:
    import numpy as _np
    if len(self.subcircuits) == 1:
        for parent, _, edge_data in list(self.probabilistic_circuit.in_edges(self)):
            self.probabilistic_circuit.add_edge(parent, self.subcircuits[0], edge_data)
        self.probabilistic_circuit.remove_node(self)
        return
    for log_weight, subcircuit in self.log_weighted_subcircuits:
        if log_weight == -_np.inf:
            self.probabilistic_circuit.remove_edge(self, subcircuit)
        if type(subcircuit) is type(self):
            for child_log_weight, child_subcircuit in subcircuit.log_weighted_subcircuits:
                self.add_subcircuit(child_subcircuit, child_log_weight + log_weight)
            self.probabilistic_circuit.remove_node(subcircuit)

_SumUnit.simplify = _patched_sum_simplify


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_OF_ITERATIONS: int = 5000

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/probabilistic_reasoning",
)

MILK_SPAWN_X: float = 2.4
MILK_SPAWN_Y: float = 2.5
MILK_SPAWN_Z: float = 1.01

COUNTER_APPROACH_X: float = 1.6
COUNTER_APPROACH_Y: float = 2.5

TABLE_APPROACH_X: float = 4.2
TABLE_APPROACH_Y: float = 4.0

PLACE_TARGET_X: float = 5.0
PLACE_TARGET_Y: float = 4.0
PLACE_TARGET_Z: float = 0.80

COUNTER_APPROACH_MIN_X: float = 1.2
COUNTER_APPROACH_MAX_X: float = 1.8
COUNTER_APPROACH_MIN_Y: float = 2.3
COUNTER_APPROACH_MAX_Y: float = 2.7

TABLE_APPROACH_MIN_X: float = 4.1
TABLE_APPROACH_MAX_X: float = 4.5
TABLE_APPROACH_MIN_Y: float = 3.95
TABLE_APPROACH_MAX_Y: float = 4.05

GCS_SEARCH_MIN_X: float = -1.0
GCS_SEARCH_MAX_X: float =  7.0
GCS_SEARCH_MIN_Y: float = -1.0
GCS_SEARCH_MAX_Y: float =  7.0
GCS_SEARCH_MIN_Z: float =  0.0
GCS_SEARCH_MAX_Z: float =  0.1
GCS_OBSTACLE_BLOAT: float = 0.3

# Robot start position — in the open aisle south of the kitchen counter.
# Verified to be inside the GCS free-space graph for the apartment world.
# The kitchen counter occupies roughly y∈[1.5,3.5]; the aisle is at y<1.2.
ROBOT_INIT_X: float = 1.0
ROBOT_INIT_Y: float = 0.5

GRASP_MANIPULATION_OFFSET: float = 0.06

OPEN_WORLD_PLACE_TARGET_X: float = 4.1

_RESOURCE_PATH:       Path = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF_PATH:  Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL_PATH:        Path = _RESOURCE_PATH / "objects" / "milk.stl"

JPT_MODEL_PATH:           str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")
TRAINING_CSV_PATH:        str = os.path.join(os.path.dirname(__file__), "pick_and_place_dataframe.csv")
JPT_MIN_SAMPLES_PER_LEAF: int = 25

CAUSAL_VARIABLE_NAMES: List[str] = [
    "pick_approach_x",
    "pick_approach_y",
    "place_approach_x",
    "place_approach_y",
    "pick_arm",
]

CAUSAL_PRIORITY_ORDER: List[str] = [
    "pick_approach_x",    # rank 1 — ATE_norm 1.714
    "place_approach_x",   # rank 2 — ATE_norm 1.511
    "pick_arm",           # rank 3 — CT4 moderator
    "pick_approach_y",    # rank 4
    "place_approach_y",   # rank 5
]

EFFECT_VARIABLE_NAMES: List[str] = ["milk_end_z"]

CAUSAL_QUERY_RESOLUTION: float = 0.005

# Width of the correction window applied to the primary cause variable.
# Set to one JPT leaf width (10x the variable precision of 0.005).
# Wide enough to draw valid samples, narrow enough to stay near the recommendation.
CAUSAL_CORRECTION_WINDOW: float = 0.05

# Apartment approach variable bounds in JPT coordinate space (after offset remapping).
# Used to clip corrected values to the physically reachable zone.
# Keyed by JPT variable name. To reuse this correction framework for a different
# task, replace this dict with bounds appropriate for that task's JPT coordinate space.
APARTMENT_VARIABLE_BOUNDS_IN_JPT_SPACE: Dict[str, tuple] = {
    "pick_approach_x":  (COUNTER_APPROACH_MIN_X, COUNTER_APPROACH_MAX_X),
    "pick_approach_y":  (
        COUNTER_APPROACH_MIN_Y - MILK_SPAWN_Y,
        COUNTER_APPROACH_MAX_Y - MILK_SPAWN_Y,
    ),
    "place_approach_x": (
        TABLE_APPROACH_MIN_X - (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X),
        TABLE_APPROACH_MAX_X - (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X),
    ),
    "place_approach_y": (
        TABLE_APPROACH_MIN_Y - PLACE_TARGET_Y,
        TABLE_APPROACH_MAX_Y - PLACE_TARGET_Y,
    ),
}


# ---------------------------------------------------------------------------
# Symbolic arm domain for pyjpt sampling
# ---------------------------------------------------------------------------

ArmChoiceDomain = type(
    "ArmChoiceDomain",
    (Multinomial,),
    {
        "values": OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)]),
        "labels": OrderedDictProxy([(0, "LEFT"), (1, "RIGHT")]),
    },
)

JPT_VARIABLES: List = [
    NumericVariable("pick_approach_x",  precision=0.005),
    NumericVariable("pick_approach_y",  precision=0.005),
    NumericVariable("place_approach_x", precision=0.005),
    NumericVariable("place_approach_y", precision=0.005),
    NumericVariable("milk_end_x",       precision=0.001),
    NumericVariable("milk_end_y",       precision=0.001),
    NumericVariable("milk_end_z",       precision=0.0005),
    SymbolicVariable("pick_arm",        domain=ArmChoiceDomain),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlanParameters:
    """Sampled parameters for one pick-and-place iteration."""
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


@dataclass
class CausalSamplingCorrection:
    """
    A one-shot correction applied to the next JPT sample after a causal failure.

    This class is intentionally task-agnostic. It holds the JPT variable name
    and recommended value from a FailureDiagnosisResult, plus the correction
    window and task-specific variable bounds.

    Generalisation note
    -------------------
    To reuse this correction framework for a different robot action (door opening,
    object handover, pouring, etc.), the only task-specific changes required are:

      1. Define a variable_bounds dict mapping JPT variable names to (min, max)
         in that task's JPT coordinate space — analogous to
         APARTMENT_VARIABLE_BOUNDS_IN_JPT_SPACE here.

      2. Implement the coordinate remapping in the task's sample function
         (analogous to _sample_plan_parameters() here) to convert JPT-space
         recommendations back to the task's world coordinates.

      Everything else — the CausalCircuit, MdVtreeNode, diagnose_failure(),
      _diagnose_and_log(), _build_correction_from_diagnosis(), and the main
      loop correction logic — is reused completely unchanged.

    Fields
    ------
    active
        Whether a correction is pending for the next sample.
    jpt_variable_name
        The variable to correct, in JPT coordinate space.
    recommended_value
        Midpoint of the highest-probability cause region from the
        interventional circuit, in JPT coordinate space.
    correction_window
        Half-width of the interval around the recommended value.
        The corrected sample lies in [recommended_value ± correction_window].
    variable_bounds
        Task-specific (min, max) bounds in JPT space for clipping.
    source_iteration
        The iteration that produced this correction, for logging.
    """
    active:            bool  = False
    jpt_variable_name: str   = ""
    recommended_value: float = 0.0
    correction_window: float = CAUSAL_CORRECTION_WINDOW
    variable_bounds:   tuple = (float("-inf"), float("inf"))
    source_iteration:  int   = 0


@dataclass
class RunStatistics:
    """
    Tracks success and failure counts, separating corrected-attempt outcomes
    from baseline outcomes.

    The separation is the key measurement: it lets you directly compare the
    success rate of causally corrected iterations against the baseline JPT
    success rate, quantifying the causal circuit's contribution to improvement.
    """
    successful_count:        int = 0
    failed_count:            int = 0
    corrected_attempt_count: int = 0
    corrected_success_count: int = 0
    corrected_failure_count: int = 0


# ---------------------------------------------------------------------------
# ORM patch: handle None values in numpy TypeDecorator
# ---------------------------------------------------------------------------

def _patch_orm_numpy_array_type() -> None:
    """
    Patch the PyCRAM ORM numpy TypeDecorator so that None values are passed
    through without calling .astype(), which raises AttributeError when a plan
    action has no associated array data.
    """
    target_class = None
    for module_name in list(sys.modules):
        if "pycram" not in module_name and "orm" not in module_name:
            continue
        module = sys.modules[module_name]
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(candidate, sqlalchemy_types.TypeDecorator)
                and hasattr(candidate, "process_bind_param")
                and "astype" in inspect.getsource(candidate.process_bind_param)
            ):
                target_class = candidate
                break
        if target_class is not None:
            break

    if target_class is None:
        print("  [patch] WARNING: ORM numpy TypeDecorator not found — None-guard skipped.")
        return

    original_process_bind_param = target_class.process_bind_param

    def _none_guarded_process_bind_param(self, value, dialect):
        if value is None:
            return None
        return original_process_bind_param(self, value, dialect)

    target_class.process_bind_param = _none_guarded_process_bind_param
    print(f"  [patch] Patched {target_class.__name__}.process_bind_param.")

_patch_orm_numpy_array_type()


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def _build_world(apartment_urdf_path: Path) -> tuple[World, PR2]:
    world = URDFParser.from_file(str(apartment_urdf_path)).parse()
    pr2_world = URDFParser.from_file(
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    robot = PR2.from_world(pr2_world)
    with world.modify_world():
        world.merge_world_at_pose(
            pr2_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(ROBOT_INIT_X, ROBOT_INIT_Y, 0.0, 0, 0, 0),
        )
    with world.modify_world():
        world.add_semantic_annotation(Table(root=world.get_body_by_name("table_area_main")))
    return world, robot


def _add_localization_frames(world: World, robot: PR2) -> None:
    with world.modify_world():
        map_body  = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_body(map_body)
        world.add_body(odom_body)
        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))
        existing_robot_connection = robot.root.parent_connection
        if existing_robot_connection is not None:
            world.remove_connection(existing_robot_connection)
        world.add_connection(
            OmniDrive.create_with_dofs(
                parent=odom_body,
                child=robot.root,
                world=world,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    ROBOT_INIT_X, ROBOT_INIT_Y, 0.0, 0, 0, 0
                ),
            )
        )


def _spawn_milk(world: World, milk_stl_path: Path) -> Body:
    mesh = FileMesh.from_file(str(milk_stl_path))
    milk_body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_SPAWN_X, MILK_SPAWN_Y, MILK_SPAWN_Z, 0, 0, 0
    )
    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=milk_body, world=world
        )
        world.add_connection(milk_connection)
        milk_connection.origin = spawn_pose
        world.add_semantic_annotation(Milk(root=milk_body))
    return milk_body


def _respawn_milk(world: World, milk_body: Body) -> None:
    """
    Reset the milk carton to its original spawn pose, re-attaching it to the
    world root if it was attached to the robot gripper during pick-up.
    """
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_SPAWN_X, MILK_SPAWN_Y, MILK_SPAWN_Z, 0, 0, 0
    )
    with world.modify_world():
        current_connection = milk_body.parent_connection
        if current_connection is not None and current_connection.parent is not world.root:
            world.remove_connection(current_connection)
            free_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(free_connection)
            free_connection.origin = spawn_pose
        elif current_connection is not None:
            current_connection.origin = spawn_pose
        else:
            free_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(free_connection)
            free_connection.origin = spawn_pose
    print(f"  [respawn] Milk reset to ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")


# Robot position is tracked explicitly in the main loop via robot_x / robot_y
# variables rather than reading from the world transform chain. This avoids
# the ambiguity between the odom_combined local frame and the apartment world
# frame, which caused incorrect (0,0) readings when the OmniDrive connection
# origin was read directly.


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _create_database_session(database_uri: str) -> Session:
    print(f"  [db] Connecting to {database_uri} ...")
    engine = create_engine(database_uri)
    if "postgresql" in database_uri or "postgres" in database_uri:
        _apply_postgresql_patches(engine)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("  [db] Tables verified.")
    return Session(engine)


def _apply_postgresql_patches(engine: Any) -> None:
    _disable_postgresql_identifier_length_validation(engine)
    _shorten_long_postgresql_table_names()
    _register_postgresql_numpy_scalar_coercion(engine)


def _disable_postgresql_identifier_length_validation(engine: Any) -> None:
    engine.dialect.validate_identifier = lambda identifier: None


def _shorten_long_postgresql_table_names() -> None:
    def _shorten_to_limit(name: str, character_limit: int = 63) -> str:
        if len(name) <= character_limit:
            return name
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:character_limit - 9]}_{digest}"

    for table in Base.metadata.tables.values():
        shortened_name = _shorten_to_limit(table.name)
        if shortened_name != table.name:
            table.name     = shortened_name
            table.fullname = shortened_name


def _register_postgresql_numpy_scalar_coercion(engine: Any) -> None:
    import numpy

    def _coerce_numpy_scalar(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        return value

    def _coerce_parameter_dict(parameters: Any) -> Any:
        if isinstance(parameters, dict):
            return {key: _coerce_numpy_scalar(value) for key, value in parameters.items()}
        return parameters

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _coerce_before_cursor_execute(
        connection, cursor, statement, parameters, context, executemany
    ):
        if isinstance(parameters, dict):
            parameters = _coerce_parameter_dict(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(
                _coerce_parameter_dict(parameter_set) for parameter_set in parameters
            )
        return statement, parameters


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    print("  [db] Persisting plan ...")
    session.add(to_dao(plan))
    session.commit()
    print("  [db] Plan committed.")


# ---------------------------------------------------------------------------
# GCS navigation
# ---------------------------------------------------------------------------

def _build_navigation_map(world: World) -> GraphOfConvexSets:
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=GCS_SEARCH_MIN_X, max_x=GCS_SEARCH_MAX_X,
                min_y=GCS_SEARCH_MIN_Y, max_y=GCS_SEARCH_MAX_Y,
                min_z=GCS_SEARCH_MIN_Z, max_z=GCS_SEARCH_MAX_Z,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        world.root,
    )
    print("  Building GCS navigation map ...")
    build_start_time = time.time()
    navigation_map = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=GCS_OBSTACLE_BLOAT,
    )
    node_count = len(list(navigation_map.graph.nodes()))
    print(f"  GCS built in {time.time() - build_start_time:.2f}s  ({node_count} nodes)")
    return navigation_map


def _build_gcs_bounds_array(navigation_map: GraphOfConvexSets) -> np.ndarray:
    """
    Build a fast (N, 6) numpy array of world-frame GCS node bounds.

    Uses node.simple_event to extract world-frame interval bounds, which is
    the same coordinate system that path_from_to uses internally. This is
    correct regardless of the BoundingBox origin transform, because
    simple_event accesses the event representation that GCS builds from the
    free-space computation (world coordinates).

    Returns array with columns: [min_x, min_y, min_z, max_x, max_y, max_z].
    """
    from semantic_digital_twin.datastructures.variables import SpatialVariables
    rows = []
    for node in navigation_map.graph.nodes():
        se = node.simple_event
        x_intervals = se[SpatialVariables.x.value].simple_sets
        y_intervals = se[SpatialVariables.y.value].simple_sets
        z_intervals = se[SpatialVariables.z.value].simple_sets
        if not x_intervals or not y_intervals or not z_intervals:
            continue
        xi, yi, zi = x_intervals[0], y_intervals[0], z_intervals[0]
        rows.append([float(xi.lower), float(yi.lower), float(zi.lower),
                     float(xi.upper), float(yi.upper), float(zi.upper)])
    array = np.array(rows, dtype=np.float64)
    print(f"  GCS bounds array: {len(array)} nodes")
    return array


def _point_in_free_space(bounds: np.ndarray, x: float, y: float, z: float) -> bool:
    """Vectorised O(N) free-space check using precomputed world-frame bounds."""
    return bool(
        ((bounds[:, 0] <= x) & (x <= bounds[:, 3]) &
         (bounds[:, 1] <= y) & (y <= bounds[:, 4]) &
         (bounds[:, 2] <= z) & (z <= bounds[:, 5])).any()
    )


def _snap_to_free_space(
    gcs_bounds:    np.ndarray,
    x:             float,
    y:             float,
    z:             float,
    world:         World,
    search_radius: float = 0.8,
    radial_step:   float = 0.05,
    angular_steps: int   = 16,
) -> Optional[Point3]:
    """
    Return a Point3 in GCS free space at or near (x, y, z).

    Uses the precomputed world-frame bounds array for fast O(N) checks
    instead of the O(N) Python loop in node_of_point(). Spirals outward
    until a free cell is found or search_radius is exceeded.
    """
    if _point_in_free_space(gcs_bounds, x, y, z):
        return Point3(x, y, z, reference_frame=world.root)

    print(f"    [GCS] ({x:.3f},{y:.3f}) not in free space — searching nearby ...")
    for radius in np.arange(radial_step, search_radius + radial_step, radial_step):
        angles = np.linspace(0, 2 * np.pi, angular_steps, endpoint=False)
        for theta in angles:
            probe_x = x + radius * np.cos(theta)
            probe_y = y + radius * np.sin(theta)
            if _point_in_free_space(gcs_bounds, probe_x, probe_y, z):
                print(f"    [GCS] Free point found at ({probe_x:.3f},{probe_y:.3f}) r={radius:.2f}")
                return Point3(probe_x, probe_y, z, reference_frame=world.root)

    print(f"    [GCS] No free point found within radius={search_radius}")
    return None


def _make_pose(x: float, y: float, z: float, reference_frame: Any) -> Pose:
    return Pose(
        position=Point3(x=x, y=y, z=z),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        reference_frame=reference_frame,
    )


def _path_to_navigate_actions(
    path:              List[Point3],
    world_frame:       Any,
    keep_joint_states: bool,
) -> List[NavigateAction]:
    return [
        NavigateAction(
            target_location=Pose(
                position=Point3(
                    x=float(waypoint.x), y=float(waypoint.y), z=0.0,
                    reference_frame=world_frame,
                ),
                orientation=Quaternion(x=0, y=0, z=0, w=1, reference_frame=world_frame),
                reference_frame=world_frame,
            ),
        )
        for waypoint in path[1:]
    ]


def _navigate_via_gcs(
    context:           Context,
    navigation_map:    GraphOfConvexSets,
    gcs_bounds:        np.ndarray,
    start_x:           float,
    start_y:           float,
    goal_x:            float,
    goal_y:            float,
    world:             World,
    keep_joint_states: bool = False,
) -> List[NavigateAction]:
    """
    Plan a collision-free path via GCS and return it as a list of NavigateActions.

    Both start and goal are snapped into GCS free space using the precomputed
    world-frame bounds array (fast vectorised O(N) check). path_from_to is
    then called on the snapped points.

    Raises ValueError if either point cannot be placed in free space or if
    path_from_to returns no path. There is no silent straight-line fallback —
    any straight line through the apartment crosses the kitchen counter.
    """
    midpoint_z = (GCS_SEARCH_MIN_Z + GCS_SEARCH_MAX_Z) / 2.0

    snapped_start = _snap_to_free_space(gcs_bounds, start_x, start_y, midpoint_z, world)
    if snapped_start is None:
        raise ValueError(
            f"GCS: cannot place start ({start_x:.3f},{start_y:.3f}) in free space. "
            f"The position may be inside an obstacle or outside the search space."
        )

    snapped_goal = _snap_to_free_space(gcs_bounds, goal_x, goal_y, midpoint_z, world)
    if snapped_goal is None:
        raise ValueError(
            f"GCS: cannot place goal ({goal_x:.3f},{goal_y:.3f}) in free space. "
            f"The target may be inside an obstacle or outside the search space."
        )

    try:
        path = navigation_map.path_from_to(snapped_start, snapped_goal)
    except Exception as path_error:
        raise ValueError(
            f"GCS: path_from_to failed from ({start_x:.3f},{start_y:.3f}) "
            f"to ({goal_x:.3f},{goal_y:.3f}): {path_error}"
        ) from path_error

    if path is None or len(path) < 2:
        raise ValueError(
            f"GCS: no path found from ({start_x:.3f},{start_y:.3f}) "
            f"to ({goal_x:.3f},{goal_y:.3f})."
        )

    navigate_actions = _path_to_navigate_actions(path, world.root, keep_joint_states)
    print(
        f"    [GCS] ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): "
        f"{len(navigate_actions)} waypoint(s)"
    )
    for index, action in enumerate(navigate_actions):
        position = action.target_location.to_position()
        print(f"           waypoint {index + 1}: ({float(position.x):.3f}, {float(position.y):.3f})")
    return navigate_actions


# ---------------------------------------------------------------------------
# JPT loading and sampling
# ---------------------------------------------------------------------------

def _load_joint_probability_tree(model_path: str) -> JointProbabilityTree:
    print(f"  [jpt] Loading model from {model_path} ...")
    joint_probability_tree = JointProbabilityTree(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    ).load(model_path)
    print(f"  [jpt] Loaded — {len(joint_probability_tree.leaves)} leaves")
    return joint_probability_tree


def _sample_plan_parameters(
    joint_probability_tree: JointProbabilityTree,
    correction:             Optional[CausalSamplingCorrection] = None,
) -> PlanParameters:
    """
    Draw one joint sample from the JPT and map it to PlanParameters.

    If a CausalSamplingCorrection is active, the primary cause variable is
    overridden with the recommended value clamped to the correction window and
    the task's variable bounds. All other variables are drawn freely from the
    joint distribution, preserving inter-variable correlations.

    The correction is applied in JPT coordinate space before the coordinate
    remapping to apartment absolute positions. This ordering is essential:
    the CausalCircuit recommendation is in JPT space (open-world offsets),
    so it must be applied before the offset addition that produces apartment
    absolute coordinates.

    To adapt this function for a different robot action, replace the coordinate
    remapping block with the mapping appropriate for that task's world geometry.
    The correction logic above the remapping block is completely task-agnostic.
    """
    sample_row     = joint_probability_tree.sample(1)[0]
    sample_by_name = {variable.name: sample_row[index]
                      for index, variable in enumerate(JPT_VARIABLES)}

    # Apply causal correction in JPT coordinate space if active.
    if correction is not None and correction.active:
        lower_bound = correction.recommended_value - correction.correction_window
        upper_bound = correction.recommended_value + correction.correction_window
        if correction.variable_bounds != (float("-inf"), float("inf")):
            lower_bound = max(lower_bound, correction.variable_bounds[0])
            upper_bound = min(upper_bound, correction.variable_bounds[1])
        corrected_value = float(
            np.clip(correction.recommended_value, lower_bound, upper_bound)
        )
        sample_by_name[correction.jpt_variable_name] = corrected_value
        print(
            f"  [correction] {correction.jpt_variable_name}: "
            f"{corrected_value:.4f}  "
            f"(window [{lower_bound:.4f}, {upper_bound:.4f}], "
            f"from iteration {correction.source_iteration})"
        )

    arm_label = sample_by_name["pick_arm"]
    if isinstance(arm_label, (int, float)):
        arm_label = ArmChoiceDomain.labels[int(arm_label)]
    pick_arm = Arms.LEFT if arm_label == "LEFT" else Arms.RIGHT

    # Coordinate remapping: JPT open-world offsets → apartment absolute positions.
    # The JPT was trained with milk at y=0 and place target at x=OPEN_WORLD_PLACE_TARGET_X.
    # The apartment milk is at y=MILK_SPAWN_Y and place target at x=PLACE_TARGET_X.
    # Adding back the respective offsets recovers absolute apartment coordinates.
    counter_approach_x = float(np.clip(
        sample_by_name["pick_approach_x"],
        COUNTER_APPROACH_MIN_X, COUNTER_APPROACH_MAX_X,
    ))
    counter_approach_y = float(np.clip(
        sample_by_name["pick_approach_y"] + MILK_SPAWN_Y,
        COUNTER_APPROACH_MIN_Y, COUNTER_APPROACH_MAX_Y,
    ))
    table_approach_x = float(np.clip(
        sample_by_name["place_approach_x"] + (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X),
        TABLE_APPROACH_MIN_X, TABLE_APPROACH_MAX_X,
    ))
    table_approach_y = float(np.clip(
        sample_by_name["place_approach_y"] + PLACE_TARGET_Y,
        TABLE_APPROACH_MIN_Y, TABLE_APPROACH_MAX_Y,
    ))

    return PlanParameters(
        counter_approach_x=counter_approach_x,
        counter_approach_y=counter_approach_y,
        table_approach_x=table_approach_x,
        table_approach_y=table_approach_y,
        pick_arm=pick_arm,
    )


# ---------------------------------------------------------------------------
# CausalCircuit construction
# ---------------------------------------------------------------------------

def _build_prob_model_variables(csv_path: str) -> list:
    """
    Build probabilistic_model variable definitions from training CSV statistics.

    ProbModelJPT requires its own Continuous/Symbolic types from random_events,
    which are separate from pyjpt's NumericVariable. Mean and std initialise
    the Continuous variables as required by the fitting procedure.
    """
    dataframe = pd.read_csv(csv_path)

    def column_mean_and_std(column_name: str):
        return float(dataframe[column_name].mean()), float(dataframe[column_name].std())

    return [
        ProbContinuous("pick_approach_x",  *column_mean_and_std("pick_approach_x")),
        ProbContinuous("pick_approach_y",  *column_mean_and_std("pick_approach_y")),
        ProbContinuous("place_approach_x", *column_mean_and_std("place_approach_x")),
        ProbContinuous("place_approach_y", *column_mean_and_std("place_approach_y")),
        ProbContinuous("milk_end_x",       *column_mean_and_std("milk_end_x")),
        ProbContinuous("milk_end_y",       *column_mean_and_std("milk_end_y")),
        ProbContinuous("milk_end_z",       *column_mean_and_std("milk_end_z")),
        ProbSymbolic("pick_arm", RESet.from_iterable(["LEFT", "RIGHT"])),
    ]


def _build_causal_circuit(csv_path: str) -> CausalCircuit:
    """
    Fit a ProbModelJPT and construct a CausalCircuit from it.

    The CausalCircuit is built once at startup and reused across all iterations.
    A separate ProbModelJPT is used rather than pyjpt because CausalCircuit
    requires a ProbabilisticCircuit object exposed via .probabilistic_circuit
    after fit(). The pyjpt model is retained separately for faster sampling.
    """
    print("  [causal] Fitting ProbModelJPT from training CSV ...")
    prob_model_variables = _build_prob_model_variables(csv_path)
    dataframe = pd.read_csv(csv_path)
    prob_model_jpt = ProbModelJPT(
        variables=prob_model_variables,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    )
    prob_model_jpt.fit(dataframe[[variable.name for variable in prob_model_variables]])
    leaf_count = len(list(prob_model_jpt.probabilistic_circuit.leaves))
    print(f"  [causal] ProbModelJPT fitted. Leaves: {leaf_count}")

    md_vtree = MdVtreeNode.from_causal_graph(
        causal_variable_names=CAUSAL_VARIABLE_NAMES,
        effect_variable_names=EFFECT_VARIABLE_NAMES,
        causal_priority_order=CAUSAL_PRIORITY_ORDER,
    )
    causal_circuit = CausalCircuit.from_jpt(
        fitted_jpt=prob_model_jpt,
        mdvtree=md_vtree,
        causal_variable_names=CAUSAL_VARIABLE_NAMES,
        effect_variable_names=EFFECT_VARIABLE_NAMES,
    )
    verification_result = causal_circuit.verify_q_determinism()
    if verification_result.passed:
        print("  [causal] CausalCircuit ready. Q-determinism: PASS")
    else:
        violation_summary = "; ".join(verification_result.violations)
        print(f"  [causal] CausalCircuit ready. Q-determinism: FAIL — {violation_summary}")
    return causal_circuit


# ---------------------------------------------------------------------------
# Failure diagnosis and correction construction
# ---------------------------------------------------------------------------

def _diagnose_and_log(
    causal_circuit:   CausalCircuit,
    plan_parameters:  PlanParameters,
    iteration_number: int,
) -> Optional[FailureDiagnosisResult]:
    """
    Run causal failure diagnosis, print a structured report, and return the
    FailureDiagnosisResult so the caller can construct a CausalSamplingCorrection.

    Apartment approach coordinates are remapped to open-world JPT space before
    calling diagnose_failure(), because the JPT was trained with the milk at
    y=0 and the place target at x=OPEN_WORLD_PLACE_TARGET_X. The remapping
    converts absolute apartment positions to the lateral offsets that the JPT
    learned during Batch 1 training.

    Returns None if diagnosis raises an exception, allowing the caller to skip
    correction safely without crashing.
    """
    observed_parameter_values = {
        "pick_approach_x":  plan_parameters.counter_approach_x,
        "pick_approach_y":  plan_parameters.counter_approach_y - MILK_SPAWN_Y,
        "place_approach_x": plan_parameters.table_approach_x - (
            PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X
        ),
        "place_approach_y": plan_parameters.table_approach_y - PLACE_TARGET_Y,
        "pick_arm":         plan_parameters.pick_arm.name,
    }

    apartment_variable_name_map = {
        "pick_approach_x":  "counter_approach_x",
        "pick_approach_y":  "counter_approach_y",
        "place_approach_x": "table_approach_x",
        "place_approach_y": "table_approach_y",
    }

    try:
        diagnosis = causal_circuit.diagnose_failure(
            observed_parameter_values=observed_parameter_values,
            effect_variable_name="milk_end_z",
            query_resolution=CAUSAL_QUERY_RESOLUTION,
        )

        primary_cause_display_name = apartment_variable_name_map.get(
            diagnosis.primary_cause_variable_name,
            diagnosis.primary_cause_variable_name,
        )
        out_of_support_marker = (
            "  ← OUT OF TRAINING SUPPORT"
            if diagnosis.interventional_probability_at_failure == 0.0
            else ""
        )

        print(f"\n  ┌─ CAUSAL FAILURE DIAGNOSIS  (iteration {iteration_number}) {'─' * 30}")
        print(f"  │  Primary cause:    {primary_cause_display_name}")
        print(f"  │  Observed value:   {diagnosis.actual_value:.4f}")
        print(f"  │  P(success|do):    {diagnosis.interventional_probability_at_failure:.4f}"
              f"{out_of_support_marker}")
        if diagnosis.recommended_value is not None:
            print(f"  │  Recommended:      {diagnosis.recommended_value:.4f}")
            print(f"  │  P(success|rec):   {diagnosis.interventional_probability_at_recommendation:.4f}")
        print(f"  │")
        print(f"  │  All variables:")
        for jpt_variable_name, variable_result in diagnosis.all_variable_results.items():
            display_name = apartment_variable_name_map.get(jpt_variable_name, jpt_variable_name)
            primary_marker = (
                "  ← PRIMARY CAUSE"
                if jpt_variable_name == diagnosis.primary_cause_variable_name
                else ""
            )
            out_of_support = (
                "  [OUT OF SUPPORT]"
                if variable_result["interventional_probability"] == 0.0
                else ""
            )
            print(
                f"  │    {display_name:<24}  "
                f"actual={variable_result['actual_value']:.4f}  "
                f"P={variable_result['interventional_probability']:.4f}"
                f"{out_of_support}{primary_marker}"
            )
        print(f"  └{'─' * 60}")
        return diagnosis

    except Exception as diagnosis_error:
        print(f"  [causal] Diagnosis failed (iteration {iteration_number}): {diagnosis_error}")
        return None


def _build_correction_from_diagnosis(
    diagnosis:        FailureDiagnosisResult,
    iteration_number: int,
) -> Optional[CausalSamplingCorrection]:
    """
    Construct a CausalSamplingCorrection from a FailureDiagnosisResult.

    Returns None if no recommendation is available — for example when the
    primary cause variable has no high-probability region in the interventional
    circuit (the variable is entirely out of training support).

    This function is the only bridge between the task-agnostic CausalCircuit
    output and the task-specific sampling pipeline. To reuse for a different
    task, replace APARTMENT_VARIABLE_BOUNDS_IN_JPT_SPACE with bounds for
    that task's JPT coordinate space.
    """
    if diagnosis.recommended_value is None:
        print("  [correction] No recommendation available — skipping correction.")
        return None

    variable_bounds = APARTMENT_VARIABLE_BOUNDS_IN_JPT_SPACE.get(
        diagnosis.primary_cause_variable_name,
        (float("-inf"), float("inf")),
    )
    correction = CausalSamplingCorrection(
        active=True,
        jpt_variable_name=diagnosis.primary_cause_variable_name,
        recommended_value=diagnosis.recommended_value,
        correction_window=CAUSAL_CORRECTION_WINDOW,
        variable_bounds=variable_bounds,
        source_iteration=iteration_number,
    )
    print(
        f"  [correction] Scheduled: {diagnosis.primary_cause_variable_name} → "
        f"{diagnosis.recommended_value:.4f}  "
        f"(window ±{CAUSAL_CORRECTION_WINDOW},  bounds {variable_bounds})"
    )
    return correction


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_fixed_plan(
    planning_context: Context,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
    navigation_map:   GraphOfConvexSets,
    gcs_bounds:       np.ndarray,
) -> SequentialPlan:
    """
    Deterministic seed plan used for iteration 1.

    Using known-good fixed parameters on the first iteration confirms that
    the world, robot, and database are correctly initialised before
    probabilistic sampling begins.
    """
    seed_arm = Arms.RIGHT
    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, gcs_bounds,
        start_x=ROBOT_INIT_X,      start_y=ROBOT_INIT_Y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        planning_context, navigation_map, gcs_bounds,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X,    goal_y=TABLE_APPROACH_Y,
        world=world, keep_joint_states=True,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)
    print(
        f"  [plan] seed — "
        f"counter:({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})  "
        f"table:({TABLE_APPROACH_X},{TABLE_APPROACH_Y})  "
        f"arm:{seed_arm}"
    )
    return SequentialPlan(
        planning_context,
        ParkArmsAction(arm=Arms.BOTH),
        *navigate_to_counter,
        PickUpAction(
            object_designator=milk_body,
            arm=seed_arm,
            grasp_description=GraspDescription(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                rotate_gripper=False,
                manipulation_offset=GRASP_MANIPULATION_OFFSET,
                manipulator=robot.right_arm.manipulator,
            ),
        ),
        *navigate_to_table,
        PlaceAction(
            object_designator=milk_body,
            target_location=place_pose,
            arm=seed_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _build_sampled_plan(
    planning_context: Context,
    plan_parameters:  PlanParameters,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
    navigation_map:   GraphOfConvexSets,
    gcs_bounds:       np.ndarray,
    robot_start_x:    float,
    robot_start_y:    float,
) -> SequentialPlan:
    """Build a plan from JPT-sampled (or causally corrected) approach positions."""
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )
    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, gcs_bounds,
        start_x=robot_start_x,                    start_y=robot_start_y,
        goal_x=plan_parameters.counter_approach_x, goal_y=plan_parameters.counter_approach_y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        planning_context, navigation_map, gcs_bounds,
        start_x=plan_parameters.counter_approach_x, start_y=plan_parameters.counter_approach_y,
        goal_x=plan_parameters.table_approach_x,    goal_y=plan_parameters.table_approach_y,
        world=world, keep_joint_states=True,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)
    print(
        f"  [plan] sampled — "
        f"counter:({plan_parameters.counter_approach_x:.3f},{plan_parameters.counter_approach_y:.3f})  "
        f"table:({plan_parameters.table_approach_x:.3f},{plan_parameters.table_approach_y:.3f})  "
        f"arm:{plan_parameters.pick_arm}"
    )
    return SequentialPlan(
        planning_context,
        ParkArmsAction(arm=Arms.BOTH),
        *navigate_to_counter,
        PickUpAction(
            object_designator=milk_body,
            arm=plan_parameters.pick_arm,
            grasp_description=GraspDescription(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                rotate_gripper=False,
                manipulation_offset=GRASP_MANIPULATION_OFFSET,
                manipulator=manipulator,
            ),
        ),
        *navigate_to_table,
        PlaceAction(
            object_designator=milk_body,
            target_location=place_pose,
            arm=plan_parameters.pick_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _navigate_back_to_start(
    planning_context: Context,
    navigation_map:   GraphOfConvexSets,
    gcs_bounds:       np.ndarray,
    world:            World,
    robot_x:          float,
    robot_y:          float,
) -> None:
    """
    Return the robot to the fixed start zone from its actual current position.

    Reads the robot ground-truth position from the world state rather than
    using planned approach coordinates. This is correct regardless of where the
    robot stopped — after a complete successful plan it will be near the table
    approach zone; after a mid-plan failure it may be anywhere.

    If GCS cannot plan a path (robot is deeply inside an obstacle or the graph
    is disconnected at that position), the failure is logged and the next
    iteration reads the robot position fresh, so this is non-fatal.
    There is no straight-line fallback — routing through the counter is never
    acceptable and would cause the next iteration to start from a wrong position.
    """
    print(
        f"  [return] robot position ({robot_x:.2f},{robot_y:.2f}) -> "
        f"start ({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})"
    )
    try:
        return_actions = _navigate_via_gcs(
            planning_context, navigation_map, gcs_bounds,
            start_x=robot_x,           start_y=robot_y,
            goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
            world=world,
        )
    except ValueError as gcs_error:
        print(f"  [return] WARNING: GCS path planning failed: {gcs_error}")
        print("  [return] Robot position will be read fresh at the next iteration.")
        return

    return_plan = SequentialPlan(planning_context, *return_actions)
    with simulated_robot:
        try:
            return_plan.perform()
            print("  [return] Robot at start position.")
        except Exception as return_error:
            print(f"  [return] WARNING: return navigation execution failed: {return_error}")

# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _print_run_summary(
    statistics:           RunStatistics,
    number_of_iterations: int,
    database_session:     Session,
) -> None:
    """
    Print the final run summary including the causal correction contribution.

    The summary separates corrected-attempt statistics from baseline statistics
    so the direct impact of the CausalCircuit can be read off immediately.
    The lift metric is the corrected success rate minus the baseline rate.
    """
    overall_success_rate        = 100 * statistics.successful_count // number_of_iterations
    uncorrected_iteration_count = number_of_iterations - statistics.corrected_attempt_count
    uncorrected_success_count   = (
        statistics.successful_count - statistics.corrected_success_count
    )
    baseline_rate = (
        100 * uncorrected_success_count // max(uncorrected_iteration_count, 1)
    )

    print(f"\n{'=' * 64}")
    print(f"  Run complete.")
    print(f"  {'─' * 60}")
    print(f"  Overall results")
    print(f"    Iterations        : {number_of_iterations}")
    print(f"    Successful        : {statistics.successful_count}  ({overall_success_rate}%)")
    print(f"    Failed            : {statistics.failed_count}")
    print(f"  {'─' * 60}")
    print(f"  Causal correction breakdown")
    print(f"    Corrected attempts  : {statistics.corrected_attempt_count}")

    if statistics.corrected_attempt_count > 0:
        corrected_rate = (
            100 * statistics.corrected_success_count
            // statistics.corrected_attempt_count
        )
        lift = corrected_rate - baseline_rate
        print(
            f"    Corrected successes : {statistics.corrected_success_count}  "
            f"({corrected_rate}%)"
        )
        print(f"    Corrected failures  : {statistics.corrected_failure_count}")
        print(f"  {'─' * 60}")
        print(f"  Baseline success rate (uncorrected)  : {baseline_rate}%")
        print(f"  Corrected success rate               : {corrected_rate}%")
        print(f"  Causal correction lift               : {lift:+d}%")
    else:
        print("    No corrected attempts recorded.")

    print(f"{'=' * 64}")

    try:
        row_count = database_session.execute(
            text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
        ).scalar()
        print(f"  DB rows (SequentialPlanDAO): {row_count}")
    except Exception as count_error:
        print(f"  [db] Could not read row count: {count_error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_and_place_demo_apartment_jpt() -> None:
    """
    Apartment world Batch 2: 5000 iterations of JPT-guided pick-and-place
    with causal failure diagnosis and active parameter correction.

    Iteration 1 uses fixed deterministic parameters to confirm the world,
    robot, and database are correctly initialised. Subsequent iterations
    sample jointly from the open-world JPT.

    When a plan fails, CausalCircuit.diagnose_failure() identifies the primary
    causal variable and recommends a corrective value. A CausalSamplingCorrection
    is then applied to the immediately following iteration, clamping the primary
    cause variable near the recommended value while all other variables are still
    drawn freely from the JPT joint distribution.

    If the corrected attempt also fails, sampling reverts to the unconstrained
    JPT on the next iteration — correction chains on bad recommendations are
    prevented by design.

    After every iteration, the robot navigates back to the start zone from its
    actual current position using a single GCS call, avoiding the previously
    broken two-leg return that used phantom planned positions as start points.

    The final run summary reports baseline and corrected success rates separately,
    quantifying the direct contribution of the causal circuit to improvement.
    """
    print("=" * 64)
    print("  pick_and_place_demo_apartment_jpt  (Batch 2 / JPT + Causal)")
    print(f"  Iterations        : {NUMBER_OF_ITERATIONS}")
    print(f"  Place target      : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model         : {JPT_MODEL_PATH}")
    print(f"  Database          : {DATABASE_URI}")
    print(f"  Correction window : ±{CAUSAL_CORRECTION_WINDOW}")
    print("=" * 64)

    print("\n[1/6] Building apartment world ...")
    world, robot = _build_world(APARTMENT_URDF_PATH)
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    gcs_bounds     = _build_gcs_bounds_array(navigation_map)

    print("\n[2/6] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/6] Loading pyjpt model (for sampling) ...")
    joint_probability_tree = _load_joint_probability_tree(JPT_MODEL_PATH)

    print("\n[4/6] Building CausalCircuit (for diagnosis and correction) ...")
    causal_circuit = _build_causal_circuit(TRAINING_CSV_PATH)

    print("\n[5/6] Starting ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_apartment_jpt_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()
    print("  [ros] Node started.")

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context    = Context(world, robot, None)
        statistics          = RunStatistics()
        pending_correction: Optional[CausalSamplingCorrection] = None
        # Explicit robot position tracking in apartment world coordinates.
        # Updated to ROBOT_INIT after each return navigation completes.
        robot_x: float = ROBOT_INIT_X
        robot_y: float = ROBOT_INIT_Y

        print("\n[6/6] Running iterations ...")

        for iteration_number in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration_number}/{NUMBER_OF_ITERATIONS}  "
                f"(success={statistics.successful_count}  "
                f"failed={statistics.failed_count}  "
                f"corrected={statistics.corrected_success_count}/"
                f"{statistics.corrected_attempt_count})"
            )
            print(f"{'=' * 64}")

            this_iteration_is_corrected = (
                pending_correction is not None and pending_correction.active
            )

            plan = None
            if iteration_number == 1:
                print("  Mode: FIXED")
                current_parameters = None
                try:
                    plan = _build_fixed_plan(
                        planning_context, world, robot, milk_body,
                        navigation_map, gcs_bounds,
                    )
                except ValueError as gcs_error:
                    statistics.failed_count += 1
                    print(f"  RESULT: FAILED (GCS plan build) — {gcs_error}")

            elif this_iteration_is_corrected:
                print(
                    f"  Mode: CAUSALLY CORRECTED  "
                    f"({pending_correction.jpt_variable_name} → "
                    f"{pending_correction.recommended_value:.4f}, "
                    f"from iteration {pending_correction.source_iteration})"
                )
                statistics.corrected_attempt_count += 1
                current_parameters = _sample_plan_parameters(
                    joint_probability_tree, correction=pending_correction
                )
                try:
                    plan = _build_sampled_plan(
                        planning_context, current_parameters, world, robot, milk_body,
                        navigation_map, gcs_bounds,
                        robot_start_x=robot_x, robot_start_y=robot_y,
                    )
                except ValueError as gcs_error:
                    statistics.failed_count += 1
                    statistics.corrected_failure_count += 1
                    print(f"  RESULT: FAILED (GCS plan build) — {gcs_error}")

            else:
                print("  Mode: JPT-SAMPLED")
                current_parameters = _sample_plan_parameters(joint_probability_tree)
                try:
                    plan = _build_sampled_plan(
                        planning_context, current_parameters, world, robot, milk_body,
                        navigation_map, gcs_bounds,
                        robot_start_x=robot_x, robot_start_y=robot_y,
                    )
                except ValueError as gcs_error:
                    statistics.failed_count += 1
                    print(f"  RESULT: FAILED (GCS plan build) — {gcs_error}")

            # Consume the pending correction. Corrections are one-shot — the next
            # iteration samples freely from the JPT unless a new failure triggers
            # a new correction.
            pending_correction = None

            if plan is None:
                # GCS failed to build a valid path during plan construction.
                # Skip execution and reset world state for the next iteration.
                print("  Skipping execution — plan could not be built.")
            else:
                print("\n  Executing plan ...")
                execution_succeeded = False
                with simulated_robot:
                    try:
                        plan.perform()
                        execution_succeeded = True
                        print("  Execution complete.")
                        if this_iteration_is_corrected:
                            statistics.corrected_success_count += 1
                            print("  [correction] Corrected attempt SUCCEEDED.")

                    except Exception as execution_error:
                        statistics.failed_count += 1
                        print(
                            f"  RESULT: FAILED — "
                            f"{type(execution_error).__name__}: {execution_error}"
                        )

                        if this_iteration_is_corrected:
                            statistics.corrected_failure_count += 1
                            print(
                                "  [correction] Corrected attempt also failed — "
                                "reverting to unconstrained JPT sampling."
                            )
                        elif current_parameters is not None:
                            # Diagnose the failure and schedule a correction for the
                            # next iteration. Corrected failures are never re-diagnosed
                            # to prevent correction chains on bad recommendations.
                            diagnosis = _diagnose_and_log(
                                causal_circuit, current_parameters, iteration_number
                            )
                            if diagnosis is not None:
                                pending_correction = _build_correction_from_diagnosis(
                                    diagnosis, iteration_number
                                )

                if execution_succeeded:
                    try:
                        _persist_plan(database_session, plan)
                        statistics.successful_count += 1
                        print(
                            f"  RESULT: SUCCESS  "
                            f"({statistics.successful_count}/{iteration_number} stored,  "
                            f"{NUMBER_OF_ITERATIONS - iteration_number} remaining)"
                        )
                    except Exception as database_error:
                        print(f"  [db] ERROR: {database_error}")
                        traceback.print_exc()

            # Update robot position to the plan endpoint before return navigation.
            # After a successful plan the robot is at table_approach.
            # After a failed plan it may be at counter_approach or table_approach
            # depending on where the failure occurred — use table_approach as the
            # conservative estimate (the further point) so GCS plans a path that
            # covers the full return distance even in the worst case.
            # For iteration 1 (fixed plan) use the fixed approach coordinates.
            if current_parameters is not None:
                robot_x = current_parameters.table_approach_x
                robot_y = current_parameters.table_approach_y
            else:
                robot_x = TABLE_APPROACH_X
                robot_y = TABLE_APPROACH_Y

            print("\n  Resetting ...")
            _respawn_milk(world, milk_body)
            _navigate_back_to_start(
                planning_context, navigation_map, gcs_bounds, world,
                robot_x=robot_x, robot_y=robot_y,
            )
            # After return navigation the robot is at the start zone.
            robot_x = ROBOT_INIT_X
            robot_y = ROBOT_INIT_Y

        _print_run_summary(statistics, NUMBER_OF_ITERATIONS, database_session)

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo_apartment_jpt()