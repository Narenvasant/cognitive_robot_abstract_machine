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
primary causal variable and recommends a corrective region. The demo then
applies a CausalSamplingCorrection for the immediately following iteration:
the primary cause variable is clamped to the midpoint of the recommended
region, while all other variables are still drawn freely from the JPT joint
distribution.

This closes the feedback loop:
    failure → causal diagnosis → parameter correction → re-attempt

Attempt and time tracking
--------------------------
Every iteration that required a causal correction attempt tracks:
  - how many correction attempts were needed before success
  - total wall-clock time for the correction loop
This mirrors the resampling tracking in pick_and_place_demo_apartment_jpt.py
so both files produce directly comparable final summaries for the paper.

Correction strategy
-------------------
- After a failure, one corrected sample is attempted immediately.
- If the corrected attempt succeeds, sampling reverts to the unconstrained JPT.
- If the corrected attempt also fails, sampling reverts to the unconstrained JPT
  rather than recursively correcting — this prevents the system from getting
  stuck following a bad recommendation.
- Corrections are tracked separately in the run statistics so the contribution
  of the causal circuit to the overall success rate can be measured directly.

MAX_CORRECTION_ATTEMPTS caps the number of consecutive causal correction
attempts per iteration before the iteration is declared a hard failure,
matching the MAX_RESAMPLE_ATTEMPTS cap in the JPT-only demo.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# nav2_msgs is not installed; mock it before giskardpy tries to use it
# via pycram.orm.ormatic_interface → giskardpy.motion_statechart.ros2_nodes.ros_tasks
_nav2_mock = MagicMock()
sys.modules["nav2_msgs"]                       = _nav2_mock
sys.modules["nav2_msgs.action"]                = _nav2_mock.action
sys.modules["nav2_msgs.action.NavigateToPose"] = _nav2_mock.action.NavigateToPose

import hashlib
import inspect
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rclpy
import sqlalchemy.types as sqlalchemy_types
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.helper import to_dao
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import ExecutesSequentially, SequentialNode
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base

from krrood.ormatic.utils import create_engine

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT as _PyjptJPT
from jpt.variables import NumericVariable, SymbolicVariable

from probabilistic_model.learning.jpt.jpt import JointProbabilityTree as ProbModelJPT
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import SumUnit as _SumUnit
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
    SupportDeterminismVerificationResult,
)

from pycram.robot_plans.actions.core.navigation import NavigateAction as _NavigateActionBase
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction

from random_events.set import Set as RESet
from random_events.variable import Continuous as REContinuous

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
from semantic_digital_twin.world_description.geometry import BoundingBox, Mesh
from semantic_digital_twin.world_description.graph_of_convex_sets import GraphOfConvexSets
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Library patch: SumUnit.simplify() version mismatch
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
# Patch: fix _migrate_nodes_from_plan stale-index bug
# ---------------------------------------------------------------------------

from pycram.plans.plan import Plan as _Plan

def _patched_migrate_nodes_from_plan(self, other):
    """
    Transfer all nodes and edges from another plan into this plan.
    After this call the other plan's graph will be empty.

    Reads are performed before any mutation so rustworkx indices remain
    valid throughout. Nodes are re-registered with fresh indices before
    edges are re-wired, so both single-node and multi-node plans are
    handled correctly.
    """
    root_ref = other.root
    edges    = list(other.edges)

    for node in other.all_nodes:
        node.index = None
        node.plan  = None
        self.add_node(node)

    for source, target in edges:
        self.add_edge(source, target)

    other.plan_graph.clear()
    return root_ref

_Plan._migrate_nodes_from_plan = _patched_migrate_nodes_from_plan

# Re-alias NavigateAction now that the patch is applied
NavigateAction = _NavigateActionBase

# Patch ActiveConnection1DOF.raw_dof to redirect stale _world references.
_APARTMENT_WORLD = None  # set after world construction

from semantic_digital_twin.world_description.connections import ActiveConnection1DOF as _AC1DOF

def _robust_raw_dof(self):
    target_world = self._world
    if (target_world is None or
            len(target_world._world_entity_hash_table) == 0 or
            len(target_world.degrees_of_freedom) == 0):
        if _APARTMENT_WORLD is not None:
            target_world = _APARTMENT_WORLD
            self._world = target_world
    return target_world.get_degree_of_freedom_by_id(self.dof_id)

_AC1DOF.raw_dof = property(_robust_raw_dof)

# Patch add_subplan to ensure context propagates correctly after migration.
from pycram.robot_plans.actions.base import ActionDescription as _ActionDescription

def _patched_add_subplan(self, subplan_root):
    subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
    self.plan.add_edge(self.plan_node, subplan_root)
    for node in self.plan.all_nodes:
        if node.plan is not self.plan:
            node.plan = self.plan
    return subplan_root

_ActionDescription.add_subplan = _patched_add_subplan


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_OF_ITERATIONS:    int = 5000
MAX_CORRECTION_ATTEMPTS: int = 10  # max causal correction attempts per iteration

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

ROBOT_INIT_X: float = 1.0
ROBOT_INIT_Y: float = 0.5

GRASP_MANIPULATION_OFFSET: float = 0.06

OPEN_WORLD_PLACE_TARGET_X: float = 4.1

_RESOURCE_PATH:      Path = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF_PATH: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL_PATH:       Path = _RESOURCE_PATH / "objects" / "milk.stl"

JPT_MODEL_PATH:           str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")
TRAINING_CSV_PATH:        str = os.path.join(os.path.dirname(__file__), "pick_and_place_dataframe.csv")
JPT_MIN_SAMPLES_PER_LEAF: int = 25

CAUSAL_VARIABLES: List[REContinuous] = [
    REContinuous("pick_approach_x"),
    REContinuous("pick_approach_y"),
    REContinuous("place_approach_x"),
    REContinuous("place_approach_y"),
    REContinuous("pick_arm"),
]

CAUSAL_PRIORITY_ORDER: List[REContinuous] = [
    REContinuous("pick_approach_x"),
    REContinuous("place_approach_x"),
    REContinuous("pick_arm"),
    REContinuous("pick_approach_y"),
    REContinuous("place_approach_y"),
]

EFFECT_VARIABLES: List[REContinuous] = [
    REContinuous("milk_end_z"),
]

CAUSAL_QUERY_RESOLUTION: float = 0.005
CAUSAL_CORRECTION_WINDOW: float = 0.05

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

    Task-agnostic: holds a JPT variable name and recommended value from a
    FailureDiagnosisResult. To reuse for a different task, supply
    variable_bounds appropriate for that task's JPT coordinate space.
    """
    active:            bool  = False
    jpt_variable_name: str   = ""
    recommended_value: float = 0.0
    correction_window: float = CAUSAL_CORRECTION_WINDOW
    variable_bounds:   tuple = (float("-inf"), float("inf"))
    source_iteration:  int   = 0


@dataclass
class CorrectionRecord:
    """
    One iteration that required one or more causal correction attempts.

    Mirrors ResamplingRecord in the JPT-only demo for direct comparison.
    """
    iteration:   int
    attempts:    int    # total correction attempts including the successful one
    elapsed_s:   float  # wall-clock seconds for the full correction loop
    succeeded:   bool   # whether the loop ended in success or hard failure


@dataclass
class RunStatistics:
    """
    Tracks success, failure, and causal correction stats across all iterations.

    Separates corrected-attempt outcomes from baseline outcomes so the direct
    contribution of the causal circuit can be measured for the paper.
    """
    successful_count:        int  = 0
    failed_count:            int  = 0
    hard_failure_count:      int  = 0   # iterations that hit MAX_CORRECTION_ATTEMPTS
    corrected_attempt_count: int  = 0   # total correction attempts across all iterations
    corrected_success_count: int  = 0   # iterations where a correction eventually succeeded
    corrected_failure_count: int  = 0   # correction attempts that individually failed
    correction_records:      list = None

    def __post_init__(self):
        if self.correction_records is None:
            self.correction_records = []

    def record_correction(
        self, iteration: int, attempts: int, elapsed_s: float, succeeded: bool
    ):
        self.correction_records.append(
            CorrectionRecord(
                iteration=iteration,
                attempts=attempts,
                elapsed_s=elapsed_s,
                succeeded=succeeded,
            )
        )


# ---------------------------------------------------------------------------
# ORM patch: handle None values in numpy TypeDecorator
# ---------------------------------------------------------------------------

def _patch_orm_numpy_array_type() -> None:
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
    world.merge_world_at_pose(
        pr2_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(ROBOT_INIT_X, ROBOT_INIT_Y, 0.0, 0, 0, 0),
    )
    with world.modify_world():
        world.add_semantic_annotation(Table(root=world.get_body_by_name("table_area_main")))
    _fix_robot_world_refs(robot, world)
    return world, robot


def _fix_robot_world_refs(robot: PR2, world: World) -> None:
    """
    After merge_world_at_pose clears pr2_world, all WorldEntity objects that
    were part of pr2_world have _world pointing to the now-empty cleared world.
    Fix all connections, bodies, DOFs, and semantic annotations.
    """
    from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
    from dataclasses import fields as _fields, is_dataclass

    for connection in world.connections:
        if connection._world is not world:
            connection._world = world
    for body in world.bodies:
        if body._world is not world:
            body._world = world
            world._world_entity_hash_table[hash(body)] = body
    for dof in world.degrees_of_freedom:
        if dof._world is not world:
            dof._world = world
            world._world_entity_hash_table[hash(dof)] = dof

    visited: set = set()
    def _fix_annotation(obj):
        if id(obj) in visited or obj is None:
            return
        visited.add(id(obj))
        if isinstance(obj, SemanticAnnotation):
            obj._world = world
            if is_dataclass(obj):
                for f in _fields(obj):
                    val = getattr(obj, f.name, None)
                    if isinstance(val, SemanticAnnotation):
                        _fix_annotation(val)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, SemanticAnnotation):
                                _fix_annotation(item)
    _fix_annotation(robot)


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
    mesh = Mesh.from_file(str(milk_stl_path))
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
    """Reset the milk carton to its original spawn pose."""
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
    import enum

    def _coerce_numpy_scalar(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        if isinstance(value, enum.Enum):      return value.value
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


def _persist_plan(session: Session, plan: SequentialNode) -> None:
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
    """Build a fast (N, 6) numpy array of world-frame GCS node bounds."""
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


def _navigate_via_gcs(
    context:        Context,
    navigation_map: GraphOfConvexSets,
    gcs_bounds:     np.ndarray,
    start_x:        float,
    start_y:        float,
    goal_x:         float,
    goal_y:         float,
    world:          World,
) -> List[NavigateAction]:
    """
    Return one NavigateAction per GCS waypoint along the collision-free path.
    Multi-waypoint navigation is safe because _patched_migrate_nodes_from_plan
    correctly handles node re-indexing for any number of sequential actions.
    """
    midpoint_z = (GCS_SEARCH_MIN_Z + GCS_SEARCH_MAX_Z) / 2.0

    snapped_start = _snap_to_free_space(gcs_bounds, start_x, start_y, midpoint_z, world)
    if snapped_start is None:
        raise ValueError(
            f"GCS: cannot place start ({start_x:.3f},{start_y:.3f}) in free space."
        )

    snapped_goal = _snap_to_free_space(gcs_bounds, goal_x, goal_y, midpoint_z, world)
    if snapped_goal is None:
        raise ValueError(
            f"GCS: cannot place goal ({goal_x:.3f},{goal_y:.3f}) in free space."
        )

    try:
        path = navigation_map.path_from_to(snapped_start, snapped_goal)
    except Exception as path_error:
        raise ValueError(
            f"GCS: path_from_to failed from ({start_x:.3f},{start_y:.3f}) "
            f"to ({goal_x:.3f},{goal_y:.3f}): {path_error}"
        ) from path_error

    if path is None or len(path) < 1:
        raise ValueError(
            f"GCS: no path found from ({start_x:.3f},{start_y:.3f}) "
            f"to ({goal_x:.3f},{goal_y:.3f})."
        )

    navigate_actions = [
        NavigateAction(
            target_location=_make_pose(float(wp.x), float(wp.y), 0.0, world.root)
        )
        for wp in path[1:]
    ]

    print(
        f"    [GCS] ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): "
        f"{len(path)} nodes, {len(navigate_actions)} NavigateAction(s)"
    )
    return navigate_actions


# ---------------------------------------------------------------------------
# JPT loading and sampling
# ---------------------------------------------------------------------------

def _load_pyjpt_model(model_path: str) -> _PyjptJPT:
    print(f"  [jpt] Loading pyjpt model from {model_path} ...")
    jpt = _PyjptJPT(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    ).load(model_path)
    print(f"  [jpt] Loaded — {len(jpt.leaves)} leaves")
    return jpt


def _sample_plan_parameters(
    jpt:        _PyjptJPT,
    correction: Optional[CausalSamplingCorrection] = None,
) -> PlanParameters:
    """
    Draw one joint sample from the JPT and map it to apartment PlanParameters.

    If a CausalSamplingCorrection is active, the primary cause variable is
    overridden with the recommended value clamped to the correction window and
    the task's variable bounds. Applied in JPT coordinate space before the
    offset remapping to apartment absolute positions.
    """
    sample_row     = jpt.sample(1)[0]
    sample_by_name = {variable.name: sample_row[index]
                      for index, variable in enumerate(JPT_VARIABLES)}

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

def _build_causal_circuit(csv_path: str) -> CausalCircuit:
    """
    Fit a ProbModelJPT and construct a CausalCircuit from it.
    Built once at startup and reused across all iterations.

    Uses infer_variables_from_dataframe to build AnnotatedVariable objects
    directly from the CSV — this correctly sets mean and standard_deviation
    per column, which JointProbabilityTree uses for max_variances in impurity
    calculation. Manual variable construction is not needed.
    """
    from probabilistic_model.learning.jpt.variables import (
        AnnotatedVariable,
        infer_variables_from_dataframe,
    )

    print("  [causal] Fitting ProbModelJPT from training CSV ...")
    dataframe = pd.read_csv(csv_path)

    # Select only the columns relevant to the causal model.
    causal_columns = [
        "pick_approach_x", "pick_approach_y",
        "place_approach_x", "place_approach_y",
        "milk_end_x", "milk_end_y", "milk_end_z",
        "pick_arm",
    ]
    model_data = dataframe[causal_columns]

    # infer_variables_from_dataframe builds AnnotatedVariables with correct
    # mean and std from the data, which is required for max_variances in
    # the JPT impurity calculation.
    annotated_variables = infer_variables_from_dataframe(model_data)

    prob_model_jpt = ProbModelJPT(
        annotated_variables=annotated_variables,
        min_samples_per_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    )
    prob_model_jpt.fit(model_data)
    leaf_count = len(list(prob_model_jpt.probabilistic_circuit.leaves))
    print(f"  [causal] ProbModelJPT fitted. Leaves: {leaf_count}")

    # Resolve Variable objects from the fitted circuit by name so that
    # CausalCircuit receives the exact instances the circuit was built with.
    # Variable equality is identity-based in random_events.
    circuit_variables_by_name = {
        v.name: v for v in prob_model_jpt.probabilistic_circuit.variables
    }

    causal_variables   = [circuit_variables_by_name[v.name] for v in CAUSAL_VARIABLES]
    effect_variables   = [circuit_variables_by_name[v.name] for v in EFFECT_VARIABLES]
    causal_priority    = [circuit_variables_by_name[v.name] for v in CAUSAL_PRIORITY_ORDER]

    marginal_determinism_tree = MarginalDeterminismTreeNode.from_causal_graph(
        causal_variables=causal_variables,
        effect_variables=effect_variables,
        causal_priority_order=causal_priority,
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit=prob_model_jpt.probabilistic_circuit,
        marginal_determinism_tree=marginal_determinism_tree,
        causal_variables=causal_variables,
        effect_variables=effect_variables,
    )

    try:
        causal_circuit.verify_support_determinism()
        print("  [causal] CausalCircuit ready. Support determinism: PASS")
    except SupportDeterminismVerificationResult as result:
        violation_summary = "; ".join(str(v) for v in result.violations)
        print(f"  [causal] CausalCircuit ready. Support determinism: FAIL — {violation_summary}")

    return causal_circuit


# ---------------------------------------------------------------------------
# Failure diagnosis and correction construction
# ---------------------------------------------------------------------------

def _region_midpoint(region: Any, cause_variable: Any) -> float:
    simple_set   = region.simple_sets[0]
    interval_set = simple_set[cause_variable]
    interval = (
        interval_set.simple_sets[0]
        if hasattr(interval_set, "simple_sets")
        else interval_set
    )
    return (float(interval.lower) + float(interval.upper)) / 2.0


def _diagnose_and_log(
    causal_circuit:   CausalCircuit,
    plan_parameters:  PlanParameters,
    iteration_number: int,
) -> Optional[FailureDiagnosisResult]:
    """
    Run causal failure diagnosis, print a structured report, and return the result.
    Apartment coordinates are remapped to JPT space before calling diagnose_failure().
    """
    cause_variable_by_name = {v.name: v for v in causal_circuit.causal_variables}
    effect_variable_by_name = {v.name: v for v in causal_circuit.effect_variables}

    observed_values = {
        cause_variable_by_name["pick_approach_x"]:  plan_parameters.counter_approach_x,
        cause_variable_by_name["pick_approach_y"]:  (
            plan_parameters.counter_approach_y - MILK_SPAWN_Y
        ),
        cause_variable_by_name["place_approach_x"]: (
            plan_parameters.table_approach_x - (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X)
        ),
        cause_variable_by_name["place_approach_y"]: (
            plan_parameters.table_approach_y - PLACE_TARGET_Y
        ),
    }

    apartment_name_map = {
        "pick_approach_x":  "counter_approach_x",
        "pick_approach_y":  "counter_approach_y",
        "place_approach_x": "table_approach_x",
        "place_approach_y": "table_approach_y",
    }

    try:
        diagnosis = causal_circuit.diagnose_failure(
            observed_values=observed_values,
            effect_variable=effect_variable_by_name["milk_end_z"],
            query_resolution=CAUSAL_QUERY_RESOLUTION,
        )

        primary_name         = diagnosis.primary_cause_variable.name
        primary_display_name = apartment_name_map.get(primary_name, primary_name)
        out_of_support       = (
            "  ← OUT OF TRAINING SUPPORT"
            if diagnosis.interventional_probability_at_failure == 0.0 else ""
        )
        recommended_midpoint = (
            _region_midpoint(diagnosis.recommended_region, diagnosis.primary_cause_variable)
            if diagnosis.recommended_region is not None else None
        )

        print(f"\n  ┌─ CAUSAL FAILURE DIAGNOSIS  (iteration {iteration_number}) {'─' * 28}")
        print(f"  │  Primary cause:    {primary_display_name}")
        print(f"  │  Observed value:   {diagnosis.actual_value:.4f}")
        print(f"  │  P(success|do):    {diagnosis.interventional_probability_at_failure:.4f}"
              f"{out_of_support}")
        if recommended_midpoint is not None:
            print(f"  │  Recommended:      {recommended_midpoint:.4f}  (region midpoint)")
            print(f"  │  P(success|rec):   "
                  f"{diagnosis.interventional_probability_at_recommendation:.4f}")
        print(f"  │")
        print(f"  │  All variables:")
        for cause_var, var_result in diagnosis.all_variable_results.items():
            display_name   = apartment_name_map.get(cause_var.name, cause_var.name)
            primary_marker = "  ← PRIMARY CAUSE" if cause_var == diagnosis.primary_cause_variable else ""
            oos_marker     = "  [OUT OF SUPPORT]" if var_result["interventional_probability"] == 0.0 else ""
            print(
                f"  │    {display_name:<24}  "
                f"actual={var_result['actual_value']:.4f}  "
                f"P={var_result['interventional_probability']:.4f}"
                f"{oos_marker}{primary_marker}"
            )
        print(f"  └{'─' * 58}")
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
    Returns None if no recommendation is available.
    """
    if diagnosis.recommended_region is None:
        print("  [correction] No recommendation available — skipping correction.")
        return None

    recommended_midpoint = _region_midpoint(
        diagnosis.recommended_region, diagnosis.primary_cause_variable
    )
    primary_name    = diagnosis.primary_cause_variable.name
    variable_bounds = APARTMENT_VARIABLE_BOUNDS_IN_JPT_SPACE.get(
        primary_name, (float("-inf"), float("inf"))
    )
    correction = CausalSamplingCorrection(
        active=True,
        jpt_variable_name=primary_name,
        recommended_value=recommended_midpoint,
        correction_window=CAUSAL_CORRECTION_WINDOW,
        variable_bounds=variable_bounds,
        source_iteration=iteration_number,
    )
    print(
        f"  [correction] Scheduled: {primary_name} → {recommended_midpoint:.4f}  "
        f"(window ±{CAUSAL_CORRECTION_WINDOW},  bounds {variable_bounds})"
    )
    return correction


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _actions_to_sequential_plan(
    planning_context: Context,
    actions:          List[Any],
) -> SequentialNode:
    """
    Build a SequentialNode from an arbitrary-length action list using the
    framework factory, which correctly wires all plan_node references.
    """
    from pycram.plans.factories import sequential
    return sequential(actions, context=planning_context)


def _build_fixed_plan(
    planning_context: Context,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
    navigation_map:   GraphOfConvexSets,
    gcs_bounds:       np.ndarray,
) -> SequentialNode:
    """Deterministic seed plan used for iteration 1."""
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
        world=world,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)
    print(
        f"  [plan] seed — "
        f"counter:({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})  "
        f"table:({TABLE_APPROACH_X},{TABLE_APPROACH_Y})  "
        f"arm:{seed_arm}"
    )
    all_actions = [
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
    ]
    return _actions_to_sequential_plan(planning_context, all_actions)


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
) -> SequentialNode:
    """Build a plan from JPT-sampled or causally corrected approach positions."""
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
        world=world,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)
    print(
        f"  [plan] sampled — "
        f"counter:({plan_parameters.counter_approach_x:.3f},{plan_parameters.counter_approach_y:.3f})  "
        f"table:({plan_parameters.table_approach_x:.3f},{plan_parameters.table_approach_y:.3f})  "
        f"arm:{plan_parameters.pick_arm}"
    )
    all_actions = [
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
    ]
    return _actions_to_sequential_plan(planning_context, all_actions)


def _navigate_back_to_start(
    planning_context: Context,
    navigation_map:   GraphOfConvexSets,
    gcs_bounds:       np.ndarray,
    world:            World,
    robot_x:          float,
    robot_y:          float,
) -> None:
    """Return the robot to the fixed start zone via GCS from its actual position."""
    print(
        f"  [return] ({robot_x:.2f},{robot_y:.2f}) -> "
        f"start ({ROBOT_INIT_X},{ROBOT_INIT_Y})"
    )
    try:
        return_actions = _navigate_via_gcs(
            planning_context, navigation_map, gcs_bounds,
            start_x=robot_x,      start_y=robot_y,
            goal_x=ROBOT_INIT_X,  goal_y=ROBOT_INIT_Y,
            world=world,
        )
    except ValueError as gcs_error:
        print(f"  [return] WARNING: GCS path planning failed: {gcs_error}")
        return

    from pycram.plans.factories import sequential as _sequential
    return_plan = _sequential(return_actions, context=planning_context)
    with simulated_robot:
        try:
            return_plan.perform()
            print("  [return] Robot at start position.")
        except Exception as return_error:
            print(f"  [return] WARNING: return navigation failed: {return_error}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def _print_run_summary(
    statistics:           RunStatistics,
    number_of_iterations: int,
    database_session:     Session,
) -> None:
    """
    Print the final run summary.

    Mirrors the JPT-only demo summary layout for direct paper comparison.
    Adds the causal circuit breakdown section showing corrected vs baseline
    success rates and the lift metric.
    """
    recs = statistics.correction_records

    overall_success_rate = 100 * statistics.successful_count // number_of_iterations
    uncorrected_iter     = number_of_iterations - statistics.corrected_attempt_count
    uncorrected_success  = statistics.successful_count - statistics.corrected_success_count
    baseline_rate        = 100 * uncorrected_success // max(uncorrected_iter, 1)

    print(f"\n{'=' * 64}")
    print(f"  Run complete.")
    print(f"  Total iterations       : {number_of_iterations}")
    print(f"  Successful plans       : {statistics.successful_count}  ({overall_success_rate}%)")
    print(f"  Failed attempts        : {statistics.failed_count}")
    print(f"  Hard failures (>{MAX_CORRECTION_ATTEMPTS} attempts) : {statistics.hard_failure_count}")

    # ── Causal correction breakdown ────────────────────────────────────────
    print(f"")
    print(f"  ── Causal Correction Stats ────────────────────────────────")
    print(f"  Iterations with correction attempts : {len(recs)}")
    print(f"  Total correction attempts           : {statistics.corrected_attempt_count}")
    print(f"  Corrected successes                 : {statistics.corrected_success_count}")
    print(f"  Corrected failures                  : {statistics.corrected_failure_count}")

    if statistics.corrected_attempt_count > 0:
        corrected_rate = (
            100 * statistics.corrected_success_count
            // statistics.corrected_attempt_count
        )
        lift = corrected_rate - baseline_rate
        print(f"  Baseline success rate (uncorrected) : {baseline_rate}%")
        print(f"  Corrected success rate              : {corrected_rate}%")
        print(f"  Causal correction lift              : {lift:+d}%")

    # ── Per-iteration correction detail (mirrors JPT demo resampling block) ─
    successful_recs = [r for r in recs if r.succeeded]
    failed_recs     = [r for r in recs if not r.succeeded]

    if successful_recs:
        avg_attempts = sum(r.attempts for r in successful_recs) / len(successful_recs)
        avg_time     = sum(r.elapsed_s for r in successful_recs) / len(successful_recs)
        max_attempts = max(r.attempts for r in successful_recs)
        max_time     = max(r.elapsed_s for r in successful_recs)
        print(f"")
        print(f"  ── Correction Attempt Stats (successful corrections only) ──")
        print(f"  Iterations that eventually succeeded : {len(successful_recs)}")
        print(f"  Avg attempts until success           : {avg_attempts:.2f}")
        print(f"  Avg time until success               : {avg_time:.1f}s")
        print(f"  Max attempts in one iteration        : {max_attempts}")
        print(f"  Max time in one iteration            : {max_time:.1f}s")
        print(f"")
        print(f"  Per-iteration correction detail (succeeded):")
        for r in successful_recs:
            print(
                f"    iter {r.iteration:>4d}:  "
                f"{r.attempts} attempt(s),  {r.elapsed_s:.1f}s  ✓"
            )

    if failed_recs:
        print(f"")
        print(f"  Per-iteration correction detail (hard failures):")
        for r in failed_recs:
            print(
                f"    iter {r.iteration:>4d}:  "
                f"{r.attempts} attempt(s),  {r.elapsed_s:.1f}s  ✗ HARD FAILURE"
            )

    if not recs:
        print(f"  (No iterations required causal correction — first attempt always succeeded)")

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

def pick_and_place_demo_apartment_causal() -> None:
    """
    Apartment world: 5000 iterations of JPT-guided pick-and-place with
    causal failure diagnosis and active parameter correction.

    Iteration 1 uses fixed deterministic parameters to confirm the world,
    robot, and database are correctly initialised.

    For subsequent iterations:
      - Sample from the JPT baseline
      - On failure: diagnose with CausalCircuit, schedule a correction
      - Correction loop: apply CausalSamplingCorrection and retry, up to
        MAX_CORRECTION_ATTEMPTS times per iteration
      - Track attempts and time per iteration for paper comparison

    The final summary mirrors pick_and_place_demo_apartment_jpt.py so both
    outputs can be placed side-by-side in the paper.
    """
    print("=" * 64)
    print("  pick_and_place_demo_apartment_causal  (JPT + CausalCircuit)")
    print(f"  Iterations         : {NUMBER_OF_ITERATIONS}")
    print(f"  Max attempts/iter  : {MAX_CORRECTION_ATTEMPTS}")
    print(f"  Place target       : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model          : {JPT_MODEL_PATH}")
    print(f"  Training CSV       : {TRAINING_CSV_PATH}")
    print(f"  Database           : {DATABASE_URI}")
    print(f"  Correction window  : ±{CAUSAL_CORRECTION_WINDOW}")
    print("=" * 64)

    print("\n[1/6] Building apartment world ...")
    world, robot = _build_world(APARTMENT_URDF_PATH)
    global _APARTMENT_WORLD
    _APARTMENT_WORLD = world
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    gcs_bounds     = _build_gcs_bounds_array(navigation_map)

    print("\n[2/6] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/6] Loading pyjpt model (for sampling) ...")
    jpt = _load_pyjpt_model(JPT_MODEL_PATH)

    print("\n[4/6] Building CausalCircuit (for diagnosis and correction) ...")
    causal_circuit = _build_causal_circuit(TRAINING_CSV_PATH)

    print("\n[5/6] Starting ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_apartment_causal_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()
    print("  [ros] Node started.")

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context = Context(world, robot, None, evaluate_conditions=False)
        statistics       = RunStatistics()
        robot_x: float   = ROBOT_INIT_X
        robot_y: float   = ROBOT_INIT_Y

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

            plan               = None
            current_parameters = None

            # ── Iteration 1: fixed seed ────────────────────────────────────
            if iteration_number == 1:
                print("  Mode: FIXED")
                try:
                    plan = _build_fixed_plan(
                        planning_context, world, robot, milk_body,
                        navigation_map, gcs_bounds,
                    )
                except ValueError as gcs_error:
                    statistics.failed_count += 1
                    print(f"  RESULT: FAILED (GCS plan build) — {gcs_error}")

                if plan is None:
                    print("  Skipping execution — plan could not be built.")
                else:
                    print("\n  Executing plan ...")
                    execution_succeeded = False
                    with simulated_robot:
                        try:
                            plan.perform()
                            execution_succeeded = True
                            print("  Execution complete.")
                        except Exception as execution_error:
                            statistics.failed_count += 1
                            print(
                                f"  RESULT: FAILED — "
                                f"{type(execution_error).__name__}: {execution_error}"
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
                            database_session.rollback()

            # ── Iterations 2+: JPT baseline + causal correction loop ───────
            else:
                print("  Mode: JPT-SAMPLED + CAUSAL CORRECTION on failure")

                # First attempt: unconstrained JPT sample
                current_parameters = _sample_plan_parameters(jpt)
                try:
                    plan = _build_sampled_plan(
                        planning_context, current_parameters, world, robot, milk_body,
                        navigation_map, gcs_bounds,
                        robot_start_x=robot_x, robot_start_y=robot_y,
                    )
                except ValueError as gcs_error:
                    statistics.failed_count += 1
                    print(f"  [attempt 1] FAILED (GCS plan build) — {gcs_error}")
                    plan = None

                execution_succeeded = False
                if plan is not None:
                    print(f"\n  [attempt 1] Executing plan ...")
                    with simulated_robot:
                        try:
                            plan.perform()
                            execution_succeeded = True
                            print("  [attempt 1] Execution complete.")
                        except Exception as execution_error:
                            statistics.failed_count += 1
                            print(
                                f"  [attempt 1] FAILED — "
                                f"{type(execution_error).__name__}: {execution_error}"
                            )

                # ── Causal correction loop (only if first attempt failed) ──
                if not execution_succeeded:
                    # Diagnose failure and build the first correction
                    diagnosis = _diagnose_and_log(
                        causal_circuit, current_parameters, iteration_number
                    )
                    pending_correction = (
                        _build_correction_from_diagnosis(diagnosis, iteration_number)
                        if diagnosis is not None else None
                    )

                    if pending_correction is not None:
                        # Reset world before entering the correction loop
                        _respawn_milk(world, milk_body)
                        _navigate_back_to_start(
                            planning_context, navigation_map, gcs_bounds, world,
                            robot_x=current_parameters.table_approach_x,
                            robot_y=current_parameters.table_approach_y,
                        )

                        correction_attempt  = 0
                        correction_start    = time.time()

                        while (not execution_succeeded
                               and pending_correction is not None
                               and correction_attempt < MAX_CORRECTION_ATTEMPTS):

                            correction_attempt          += 1
                            statistics.corrected_attempt_count += 1

                            current_parameters = _sample_plan_parameters(
                                jpt, correction=pending_correction
                            )
                            # Consume correction — next loop uses plain JPT
                            # unless a new failure triggers another diagnosis
                            pending_correction = None

                            try:
                                plan = _build_sampled_plan(
                                    planning_context, current_parameters, world, robot,
                                    milk_body, navigation_map, gcs_bounds,
                                    robot_start_x=ROBOT_INIT_X, robot_start_y=ROBOT_INIT_Y,
                                )
                            except ValueError as gcs_error:
                                statistics.failed_count += 1
                                statistics.corrected_failure_count += 1
                                print(
                                    f"  [correction {correction_attempt}] "
                                    f"FAILED (GCS plan build) — {gcs_error}"
                                )
                                continue

                            print(f"\n  [correction {correction_attempt}] Executing plan ...")
                            with simulated_robot:
                                try:
                                    plan.perform()
                                    execution_succeeded = True
                                    elapsed = time.time() - correction_start
                                    print(
                                        f"  [correction {correction_attempt}] "
                                        f"Execution complete. ({elapsed:.1f}s elapsed)"
                                    )
                                except Exception as execution_error:
                                    statistics.failed_count += 1
                                    statistics.corrected_failure_count += 1
                                    print(
                                        f"  [correction {correction_attempt}] FAILED — "
                                        f"{type(execution_error).__name__}: {execution_error}"
                                    )
                                    # Diagnose again and schedule next correction
                                    # if attempts remain
                                    if correction_attempt < MAX_CORRECTION_ATTEMPTS:
                                        _respawn_milk(world, milk_body)
                                        _navigate_back_to_start(
                                            planning_context, navigation_map, gcs_bounds, world,
                                            robot_x=current_parameters.table_approach_x,
                                            robot_y=current_parameters.table_approach_y,
                                        )
                                        new_diagnosis = _diagnose_and_log(
                                            causal_circuit, current_parameters, iteration_number
                                        )
                                        if new_diagnosis is not None:
                                            pending_correction = _build_correction_from_diagnosis(
                                                new_diagnosis, iteration_number
                                            )

                        # Record correction loop outcome
                        elapsed = time.time() - correction_start
                        total_attempts = correction_attempt  # correction attempts only

                        if execution_succeeded:
                            statistics.corrected_success_count += 1
                            print(
                                f"  [causal] Succeeded after {total_attempts} correction "
                                f"attempt(s) in {elapsed:.1f}s"
                            )
                            statistics.record_correction(
                                iteration_number, total_attempts, elapsed, succeeded=True
                            )
                        else:
                            statistics.hard_failure_count += 1
                            statistics.failed_count += 1
                            print(
                                f"  RESULT: HARD FAILURE — gave up after "
                                f"{total_attempts} correction attempt(s) ({elapsed:.1f}s)"
                            )
                            statistics.record_correction(
                                iteration_number, total_attempts, elapsed, succeeded=False
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
                        database_session.rollback()

            # ── Reset world for next iteration ─────────────────────────────
            if iteration_number == 1:
                end_x, end_y = TABLE_APPROACH_X, TABLE_APPROACH_Y
            elif current_parameters is not None:
                end_x = current_parameters.table_approach_x
                end_y = current_parameters.table_approach_y
            else:
                end_x, end_y = TABLE_APPROACH_X, TABLE_APPROACH_Y

            print("\n  Resetting ...")
            _respawn_milk(world, milk_body)
            _navigate_back_to_start(
                planning_context, navigation_map, gcs_bounds, world,
                robot_x=end_x, robot_y=end_y,
            )
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
    pick_and_place_demo_apartment_causal()