"""
Apartment World: JPT-Guided Pick-and-Place with Causal Failure Diagnosis

Implements the closed-loop framework described in:
    "Causally-Aware Robot Action Verification via Interventional Probabilistic Circuits"
    SPAI @ IJCAI 2026

Overview
--------
A PR2 robot performs a pick-and-place task in an apartment simulation:
grasping a milk carton from a kitchen counter and placing it on a dining
table. Candidate action parameters (approach positions, arm selection) are
sampled from a Joint Probability Tree (JPT) trained on 1,742 successful
open-world executions and transferred to the apartment world via coordinate
remapping, without retraining.

When a plan fails execution, the Causal Circuit diagnoses the failure by
computing the interventional distribution P(Y | do(x_i = v)) for each cause
variable and identifies the primary cause: the variable whose observed value
minimises the interventional success probability. A one-shot corrective
re-sample is then issued, constraining the primary cause variable to the
recommended region while all other variables are drawn freely from the joint
planning distribution. If the corrected attempt also fails, the system
reverts to unconstrained JPT sampling without chaining further corrections.

Plans whose parameters lie entirely outside the training support are detected
automatically (interventional probability = 0 for all cause variables) and
excluded from the correction loop, ensuring no correction is issued without
grounding in observed data.

Variable Mapping (open-world JPT space -> apartment world)
----------------------------------------------------------
    pick_approach_x / pick_approach_y   ->  counter_approach_x / counter_approach_y
    place_approach_x / place_approach_y ->  table_approach_x   / table_approach_y
    milk_end_x / milk_end_y / milk_end_z                       ->  unchanged
    pick_arm                                                    ->  unchanged

Experiment Configuration
------------------------
    Iterations per run     : 5,000
    Max correction attempts: 10 per iteration (matches JPT-only baseline cap)
    JPT model              : pick_and_place_jpt.json (high-quality or degraded)
    Training data          : pick_and_place_dataframe.csv (1,742 rows)
    Effect variable        : milk_end_z (task-success proxy, threshold tau)
    Causal priority order  : pick_approach_x, place_approach_x, pick_arm,
                             pick_approach_y, place_approach_y (by ATE_norm rank)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rclpy
import sqlalchemy.types as sqlalchemy_types
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT as NativeJPT
from jpt.variables import NumericVariable, SymbolicVariable

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
    SupportDeterminismVerificationResult,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit as ProbabilisticSumUnit,
)

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialNode
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base
from pycram.plans.plan import Plan as PycramPlan
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction as BaseNavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction

from random_events.variable import Continuous as ContinuousVariable

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
    ActiveConnection1DOF,
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


# =============================================================================
# Library Compatibility Patches
# =============================================================================

def _patch_sum_unit_simplify() -> None:
    """
    Replace SumUnit.simplify() to resolve a version mismatch between the
    installed probabilistic_model and the circuit structure produced by the
    JPT conversion. The patched version correctly handles zero-weight edges
    and nested SumUnit merging without mutating the graph during iteration.
    """
    def _simplify(self) -> None:
        import numpy as numpy_local

        if len(self.subcircuits) == 1:
            for parent, _, edge_data in list(
                self.probabilistic_circuit.in_edges(self)
            ):
                self.probabilistic_circuit.add_edge(
                    parent, self.subcircuits[0], edge_data
                )
            self.probabilistic_circuit.remove_node(self)
            return

        for log_weight, subcircuit in self.log_weighted_subcircuits:
            if log_weight == -numpy_local.inf:
                self.probabilistic_circuit.remove_edge(self, subcircuit)
            if type(subcircuit) is type(self):
                for child_log_weight, child_subcircuit in (
                    subcircuit.log_weighted_subcircuits
                ):
                    self.add_subcircuit(
                        child_subcircuit, child_log_weight + log_weight
                    )
                self.probabilistic_circuit.remove_node(subcircuit)

    ProbabilisticSumUnit.simplify = _simplify


def _patch_plan_migrate_nodes() -> None:
    """
    Replace Plan._migrate_nodes_from_plan() to fix a stale-index bug in
    rustworkx when transferring nodes from multi-action plans. All reads
    are performed before any mutation so indices remain valid throughout
    the transfer, and nodes are re-registered before edges are re-wired.
    """
    def _migrate(self, other: PycramPlan) -> Any:
        root_reference = other.root
        edges = list(other.edges)

        for node in other.all_nodes:
            node.index = None
            node.plan = None
            self.add_node(node)

        for source, target in edges:
            self.add_edge(source, target)

        other.plan_graph.clear()
        return root_reference

    PycramPlan._migrate_nodes_from_plan = _migrate


def _patch_action_description_add_subplan() -> None:
    """
    Replace ActionDescription.add_subplan() to ensure that plan node
    references are propagated correctly to all nodes after migration.
    Without this patch, nodes migrated from sub-plans retain a stale
    plan reference that causes graph corruption on multi-step execution.
    """
    def _add_subplan(self, subplan_root: Any) -> Any:
        subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
        self.plan.add_edge(self.plan_node, subplan_root)
        for node in self.plan.all_nodes:
            if node.plan is not self.plan:
                node.plan = self.plan
        return subplan_root

    ActionDescription.add_subplan = _add_subplan


def _patch_active_connection_raw_dof(apartment_world_ref: list) -> None:
    """
    Replace ActiveConnection1DOF.raw_dof with a version that redirects
    stale _world references to the current apartment world. After
    merge_world_at_pose clears the PR2 sub-world, any connection whose
    _world still points to the cleared world will fail on DOF lookup.
    This patch repairs the reference lazily at access time.

    Parameters
    ----------
    apartment_world_ref:
        A one-element list whose single entry is set to the apartment
        World instance after construction. Using a list rather than a
        module-level variable avoids a circular reference at patch time.
    """
    def _raw_dof(self) -> Any:
        target_world = self._world
        if (
            target_world is None
            or len(target_world._world_entity_hash_table) == 0
            or len(target_world.degrees_of_freedom) == 0
        ):
            if apartment_world_ref[0] is not None:
                target_world = apartment_world_ref[0]
                self._world = target_world
        return target_world.get_degree_of_freedom_by_id(self.dof_id)

    ActiveConnection1DOF.raw_dof = property(_raw_dof)


def _patch_orm_numpy_type_decorator() -> None:
    """
    Add a None-guard to the ORM TypeDecorator that serialises numpy arrays
    to PostgreSQL. Without this patch, persisting a plan that contains a
    None-valued field raises an AttributeError inside process_bind_param
    because the original implementation calls .astype() unconditionally.
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
        print(
            "[patch] WARNING: ORM numpy TypeDecorator not found "
            "— None guard skipped."
        )
        return

    original = target_class.process_bind_param

    def _guarded(self, value, dialect):
        if value is None:
            return None
        return original(self, value, dialect)

    target_class.process_bind_param = _guarded
    print(f"[patch] Patched {target_class.__name__}.process_bind_param with None guard.")


_apartment_world_reference: list = [None]

_patch_sum_unit_simplify()
_patch_plan_migrate_nodes()
_patch_action_description_add_subplan()
_patch_active_connection_raw_dof(_apartment_world_reference)
_patch_orm_numpy_type_decorator()

NavigateAction = BaseNavigateAction


# =============================================================================
# Experiment Constants
# =============================================================================

NUMBER_OF_ITERATIONS:    int = 5000
MAX_CORRECTION_ATTEMPTS: int = 10

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/probabilistic_reasoning_causal",
)

# World geometry: milk spawn position
MILK_SPAWN_X: float = 2.4
MILK_SPAWN_Y: float = 2.5
MILK_SPAWN_Z: float = 1.01

# Default approach positions (used for the deterministic seed plan)
COUNTER_APPROACH_X: float = 1.6
COUNTER_APPROACH_Y: float = 2.5
TABLE_APPROACH_X:   float = 4.2
TABLE_APPROACH_Y:   float = 4.0

# Place target in apartment world absolute coordinates
PLACE_TARGET_X: float = 5.0
PLACE_TARGET_Y: float = 4.0
PLACE_TARGET_Z: float = 0.80

# Feasible approach region bounds (apartment world absolute coordinates)
COUNTER_APPROACH_MIN_X: float = 1.2
COUNTER_APPROACH_MAX_X: float = 1.8
COUNTER_APPROACH_MIN_Y: float = 2.3
COUNTER_APPROACH_MAX_Y: float = 2.7
TABLE_APPROACH_MIN_X:   float = 4.1
TABLE_APPROACH_MAX_X:   float = 4.5
TABLE_APPROACH_MIN_Y:   float = 3.95
TABLE_APPROACH_MAX_Y:   float = 4.05

# Graph of Convex Sets (GCS) navigation search space
GCS_SEARCH_MIN_X:   float = -1.0
GCS_SEARCH_MAX_X:   float =  7.0
GCS_SEARCH_MIN_Y:   float = -1.0
GCS_SEARCH_MAX_Y:   float =  7.0
GCS_SEARCH_MIN_Z:   float =  0.0
GCS_SEARCH_MAX_Z:   float =  0.1
GCS_OBSTACLE_BLOAT: float =  0.3

# Robot initial pose
ROBOT_INIT_X: float = 1.0
ROBOT_INIT_Y: float = 0.5

GRASP_MANIPULATION_OFFSET: float = 0.06

# Place target x in the open-world JPT coordinate space (used for remapping)
OPEN_WORLD_PLACE_TARGET_X: float = 4.1

# File paths
_RESOURCE_PATH:      Path = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF_PATH: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL_PATH:       Path = _RESOURCE_PATH / "objects" / "milk.stl"
JPT_MODEL_PATH:      str  = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")
TRAINING_CSV_PATH:   str  = os.path.join(os.path.dirname(__file__), "pick_and_place_dataframe.csv")

# JPT hyperparameters (must match the fitted model)
JPT_MIN_SAMPLES_PER_LEAF: int = 25

# Causal circuit configuration
CAUSAL_VARIABLES: List[ContinuousVariable] = [
    ContinuousVariable("pick_approach_x"),
    ContinuousVariable("pick_approach_y"),
    ContinuousVariable("place_approach_x"),
    ContinuousVariable("place_approach_y"),
    ContinuousVariable("pick_arm"),
]

CAUSAL_PRIORITY_ORDER: List[ContinuousVariable] = [
    ContinuousVariable("pick_approach_x"),
    ContinuousVariable("place_approach_x"),
    ContinuousVariable("pick_arm"),
    ContinuousVariable("pick_approach_y"),
    ContinuousVariable("place_approach_y"),
]

EFFECT_VARIABLES: List[ContinuousVariable] = [
    ContinuousVariable("milk_end_z"),
]

CAUSAL_QUERY_RESOLUTION: float = 0.005
CAUSAL_CORRECTION_WINDOW: float = 0.05

# Feasible bounds in JPT coordinate space (before apartment offset remapping)
VARIABLE_BOUNDS_IN_JPT_SPACE: Dict[str, tuple] = {
    "pick_approach_x": (COUNTER_APPROACH_MIN_X, COUNTER_APPROACH_MAX_X),
    "pick_approach_y": (
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

# Human-readable name mapping from JPT variable names to apartment variable names
JPT_TO_APARTMENT_VARIABLE_NAME: Dict[str, str] = {
    "pick_approach_x":  "counter_approach_x",
    "pick_approach_y":  "counter_approach_y",
    "place_approach_x": "table_approach_x",
    "place_approach_y": "table_approach_y",
}


# =============================================================================
# JPT Variable Definitions
# =============================================================================

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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlanParameters:
    """Sampled or corrected parameters for one pick-and-place iteration."""
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


@dataclass
class CausalSamplingCorrection:
    """
    A one-shot correction applied to the next JPT sample after a failed plan.

    The primary cause variable identified by the Causal Circuit is constrained
    to a narrow window around the recommended value derived from the
    interventional distribution. All other variables are sampled freely from
    the joint JPT distribution, preserving learned correlations.

    This data class is task-agnostic: it operates in JPT coordinate space and
    can be reused for any task by supplying appropriate variable_bounds.

    Attributes
    ----------
    active:
        Whether this correction should be applied on the next sample.
    jpt_variable_name:
        Name of the cause variable to constrain, in JPT coordinate space.
    recommended_value:
        Midpoint of the recommended region from the interventional distribution.
    correction_window:
        Half-width of the constraint interval around recommended_value.
    variable_bounds:
        Hard lower and upper bounds for the corrected value, derived from the
        feasible region of the deployment environment.
    source_iteration:
        Iteration number at which this correction was generated, for logging.
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
    Record of one iteration that required one or more causal correction attempts.

    Mirrors ResamplingRecord in the JPT-only baseline for direct comparison.

    Attributes
    ----------
    iteration:
        Iteration number within the 5,000-iteration run.
    attempts:
        Total number of correction attempts, including the successful one.
    elapsed_seconds:
        Wall-clock time for the full correction loop.
    succeeded:
        True if the correction loop ended in success; False for hard failures.
    """
    iteration:       int
    attempts:        int
    elapsed_seconds: float
    succeeded:       bool


@dataclass
class RunStatistics:
    """
    Tracks success, failure, and causal correction statistics across all iterations.

    Separates corrected-attempt outcomes from uncorrected baseline outcomes so
    the direct contribution of the Causal Circuit can be measured and reported.

    Attributes
    ----------
    successful_count:
        Total iterations that ended in a successful plan execution.
    failed_iterations:
        Total iterations that ended in failure (including hard failures).
    failed_attempts:
        Total individual attempt failures across all iterations.
    hard_failure_count:
        Iterations that exhausted MAX_CORRECTION_ATTEMPTS without success.
    corrected_attempt_count:
        Total correction attempts issued by the Causal Circuit.
    corrected_success_count:
        Iterations where a causal correction eventually led to success.
    corrected_failure_count:
        Individual correction attempts that failed.
    correction_records:
        Per-iteration correction detail for summary reporting.
    """
    successful_count:        int  = 0
    failed_iterations:       int  = 0
    failed_attempts:         int  = 0
    hard_failure_count:      int  = 0
    corrected_attempt_count: int  = 0
    corrected_success_count: int  = 0
    corrected_failure_count: int  = 0
    correction_records:      list = None

    def __post_init__(self) -> None:
        if self.correction_records is None:
            self.correction_records = []

    def record_correction(
        self,
        iteration:       int,
        attempts:        int,
        elapsed_seconds: float,
        succeeded:       bool,
    ) -> None:
        self.correction_records.append(
            CorrectionRecord(
                iteration=iteration,
                attempts=attempts,
                elapsed_seconds=elapsed_seconds,
                succeeded=succeeded,
            )
        )


# =============================================================================
# World Construction
# =============================================================================

def _build_apartment_world(apartment_urdf_path: Path) -> tuple[World, PR2]:
    """
    Parse the apartment URDF and merge the PR2 robot into the world at the
    initial robot pose. A Table semantic annotation is added to the dining
    table body, and stale world references on the robot are repaired after
    the merge clears the PR2 sub-world.
    """
    apartment_world = URDFParser.from_file(str(apartment_urdf_path)).parse()
    pr2_world       = URDFParser.from_file(
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    robot = PR2.from_world(pr2_world)
    apartment_world.merge_world_at_pose(
        pr2_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            ROBOT_INIT_X, ROBOT_INIT_Y, 0.0, 0, 0, 0
        ),
    )
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(
            Table(root=apartment_world.get_body_by_name("table_area_main"))
        )
    _repair_robot_world_references(robot, apartment_world)
    return apartment_world, robot


def _repair_robot_world_references(robot: PR2, world: World) -> None:
    """
    Repair stale _world references on all robot world entities after merge.

    merge_world_at_pose clears the PR2 sub-world, leaving all connections,
    bodies, degrees of freedom, and semantic annotations pointing to the now-
    empty cleared world. This function walks all such objects and redirects
    their _world attribute to the merged apartment world.
    """
    from dataclasses import fields as dataclass_fields, is_dataclass
    from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

    for connection in world.connections:
        if connection._world is not world:
            connection._world = world

    for body in world.bodies:
        if body._world is not world:
            body._world = world
            world._world_entity_hash_table[hash(body)] = body

    for degree_of_freedom in world.degrees_of_freedom:
        if degree_of_freedom._world is not world:
            degree_of_freedom._world = world
            world._world_entity_hash_table[hash(degree_of_freedom)] = degree_of_freedom

    visited_ids: set = set()

    def repair_annotation(obj: Any) -> None:
        if id(obj) in visited_ids or obj is None:
            return
        visited_ids.add(id(obj))
        if isinstance(obj, SemanticAnnotation):
            obj._world = world
            if is_dataclass(obj):
                for field in dataclass_fields(obj):
                    value = getattr(obj, field.name, None)
                    if isinstance(value, SemanticAnnotation):
                        repair_annotation(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, SemanticAnnotation):
                                repair_annotation(item)

    repair_annotation(robot)


def _add_localization_frames(world: World, robot: PR2) -> None:
    """
    Add map and odom_combined localisation frames and connect the robot via
    an OmniDrive joint. This is required for ROS2 TF publishing and navigation.
    """
    with world.modify_world():
        map_body  = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_body(map_body)
        world.add_body(odom_body)
        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))

        existing_connection = robot.root.parent_connection
        if existing_connection is not None:
            world.remove_connection(existing_connection)

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
    """
    Add a milk carton body to the world at the fixed spawn position and
    attach a Milk semantic annotation. Returns the milk Body for later
    use in PickUpAction and respawning.
    """
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
    """
    Reset the milk carton to its original spawn pose.

    If the milk is currently attached to a gripper (non-root parent), the
    existing connection is removed and a new free Connection6DoF to the world
    root is created. If the milk already has a root connection, the origin is
    simply reset in place.
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

    print(
        f"[reset] Milk respawned at "
        f"({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})"
    )


# =============================================================================
# Database
# =============================================================================

def _create_database_session(database_uri: str) -> Session:
    """
    Connect to the PostgreSQL database, create missing tables, and return an
    ORM session. PostgreSQL-specific patches (identifier length validation,
    table name shortening, numpy scalar coercion) are applied automatically.
    """
    print(f"[database] Connecting to {database_uri} ...")
    engine = create_engine(database_uri)

    if "postgresql" in database_uri or "postgres" in database_uri:
        _apply_postgresql_patches(engine)

    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("[database] Schema verified.")
    return Session(engine)


def _apply_postgresql_patches(engine: Any) -> None:
    """Apply all PostgreSQL-specific engine patches."""
    _disable_identifier_length_validation(engine)
    _shorten_long_table_names()
    _register_numpy_scalar_coercion(engine)


def _disable_identifier_length_validation(engine: Any) -> None:
    """
    Disable PostgreSQL's 63-character identifier length validation.
    SQLAlchemy ORM table names generated by pycram sometimes exceed this
    limit; the names are shortened separately in _shorten_long_table_names.
    """
    engine.dialect.validate_identifier = lambda identifier: None


def _shorten_long_table_names(character_limit: int = 63) -> None:
    """
    Truncate any ORM table name that exceeds the PostgreSQL identifier limit.
    A short SHA-256 digest is appended to ensure uniqueness after truncation.
    """
    def shorten(name: str) -> str:
        if len(name) <= character_limit:
            return name
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:character_limit - 9]}_{digest}"

    for table in Base.metadata.tables.values():
        shortened = shorten(table.name)
        if shortened != table.name:
            table.name     = shortened
            table.fullname = shortened


def _register_numpy_scalar_coercion(engine: Any) -> None:
    """
    Register a before_cursor_execute listener that converts numpy scalar
    types and Python enums to native Python types before parameter binding.
    PostgreSQL's psycopg2 driver does not accept numpy scalars directly.
    """
    import numpy
    import enum

    def coerce_scalar(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        if isinstance(value, enum.Enum):      return value.value
        return value

    def coerce_parameters(parameters: Any) -> Any:
        if isinstance(parameters, dict):
            return {key: coerce_scalar(value) for key, value in parameters.items()}
        return parameters

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def before_execute(connection, cursor, statement, parameters, context, executemany):
        if isinstance(parameters, dict):
            parameters = coerce_parameters(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(
                coerce_parameters(parameter_set) for parameter_set in parameters
            )
        return statement, parameters


def _persist_plan(session: Session, plan: SequentialNode) -> None:
    """Persist a completed plan to the database and commit the transaction."""
    print("[database] Persisting plan ...")
    session.add(to_dao(plan))
    session.commit()
    print("[database] Plan committed.")


# =============================================================================
# GCS Navigation
# =============================================================================

def _build_navigation_map(world: World) -> GraphOfConvexSets:
    """
    Build a Graph of Convex Sets (GCS) navigation map from the apartment world
    geometry within the defined search space. Obstacles are bloated by
    GCS_OBSTACLE_BLOAT metres to provide a safety margin around walls and
    furniture. Building time is reported for reference.
    """
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
    print("[navigation] Building GCS navigation map ...")
    build_start = time.time()
    navigation_map = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=GCS_OBSTACLE_BLOAT,
    )
    node_count = len(list(navigation_map.graph.nodes()))
    print(
        f"[navigation] GCS map built in {time.time() - build_start:.2f}s "
        f"({node_count} nodes)"
    )
    return navigation_map


def _build_gcs_bounds_array(navigation_map: GraphOfConvexSets) -> np.ndarray:
    """
    Build a (N, 6) numpy array of world-frame axis-aligned bounding boxes for
    all GCS nodes. Used for fast vectorised free-space queries without iterating
    the graph structure at runtime.

    Returns
    -------
    np.ndarray
        Array of shape (N, 6) with columns [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    from semantic_digital_twin.datastructures.variables import SpatialVariables

    rows = []
    for node in navigation_map.graph.nodes():
        simple_event = node.simple_event
        x_intervals  = simple_event[SpatialVariables.x.value].simple_sets
        y_intervals  = simple_event[SpatialVariables.y.value].simple_sets
        z_intervals  = simple_event[SpatialVariables.z.value].simple_sets
        if not x_intervals or not y_intervals or not z_intervals:
            continue
        x_interval, y_interval, z_interval = x_intervals[0], y_intervals[0], z_intervals[0]
        rows.append([
            float(x_interval.lower), float(y_interval.lower), float(z_interval.lower),
            float(x_interval.upper), float(y_interval.upper), float(z_interval.upper),
        ])

    bounds_array = np.array(rows, dtype=np.float64)
    print(f"[navigation] GCS bounds array: {len(bounds_array)} nodes")
    return bounds_array


def _point_in_free_space(
    bounds_array: np.ndarray, x: float, y: float, z: float
) -> bool:
    """Return True if (x, y, z) lies inside any GCS node bounding box."""
    return bool(
        (
            (bounds_array[:, 0] <= x) & (x <= bounds_array[:, 3]) &
            (bounds_array[:, 1] <= y) & (y <= bounds_array[:, 4]) &
            (bounds_array[:, 2] <= z) & (z <= bounds_array[:, 5])
        ).any()
    )


def _snap_to_free_space(
    bounds_array:  np.ndarray,
    x:             float,
    y:             float,
    z:             float,
    world:         World,
    search_radius: float = 0.8,
    radial_step:   float = 0.05,
    angular_steps: int   = 16,
) -> Optional[Point3]:
    """
    Return the nearest free-space point to (x, y, z), searching radially
    outward if the query point itself is not in free space.

    Returns None if no free point is found within search_radius metres.
    """
    if _point_in_free_space(bounds_array, x, y, z):
        return Point3(x, y, z, reference_frame=world.root)

    print(f"[navigation] ({x:.3f}, {y:.3f}) not in free space — searching nearby ...")
    for radius in np.arange(radial_step, search_radius + radial_step, radial_step):
        for theta in np.linspace(0, 2 * np.pi, angular_steps, endpoint=False):
            candidate_x = x + radius * np.cos(theta)
            candidate_y = y + radius * np.sin(theta)
            if _point_in_free_space(bounds_array, candidate_x, candidate_y, z):
                print(
                    f"[navigation] Free point found at "
                    f"({candidate_x:.3f}, {candidate_y:.3f}), radius={radius:.2f}"
                )
                return Point3(candidate_x, candidate_y, z, reference_frame=world.root)

    print(f"[navigation] No free point found within radius={search_radius}")
    return None


def _make_pose(x: float, y: float, z: float, reference_frame: Any) -> Pose:
    """Construct a Pose at (x, y, z) with identity orientation."""
    return Pose(
        position=Point3(x=x, y=y, z=z),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        reference_frame=reference_frame,
    )


def _navigate_via_gcs(
    context:        Context,
    navigation_map: GraphOfConvexSets,
    bounds_array:   np.ndarray,
    start_x:        float,
    start_y:        float,
    goal_x:         float,
    goal_y:         float,
    world:          World,
) -> List[NavigateAction]:
    """
    Plan a collision-free path from (start_x, start_y) to (goal_x, goal_y)
    via GCS and return one NavigateAction per waypoint.

    Start and goal positions are snapped to the nearest free-space point if
    they do not lie directly inside a GCS node. Multi-waypoint navigation is
    safe because _patch_plan_migrate_nodes handles re-indexing correctly for
    any number of sequential actions.

    Raises
    ------
    ValueError
        If either the start or goal cannot be placed in free space, or if
        GCS finds no collision-free path between them.
    """
    midpoint_z = (GCS_SEARCH_MIN_Z + GCS_SEARCH_MAX_Z) / 2.0

    snapped_start = _snap_to_free_space(bounds_array, start_x, start_y, midpoint_z, world)
    if snapped_start is None:
        raise ValueError(
            f"GCS: cannot place start ({start_x:.3f}, {start_y:.3f}) in free space."
        )

    snapped_goal = _snap_to_free_space(bounds_array, goal_x, goal_y, midpoint_z, world)
    if snapped_goal is None:
        raise ValueError(
            f"GCS: cannot place goal ({goal_x:.3f}, {goal_y:.3f}) in free space."
        )

    try:
        path = navigation_map.path_from_to(snapped_start, snapped_goal)
    except Exception as error:
        raise ValueError(
            f"GCS: path_from_to failed from ({start_x:.3f}, {start_y:.3f}) "
            f"to ({goal_x:.3f}, {goal_y:.3f}): {error}"
        ) from error

    if path is None or len(path) < 1:
        raise ValueError(
            f"GCS: no path found from ({start_x:.3f}, {start_y:.3f}) "
            f"to ({goal_x:.3f}, {goal_y:.3f})."
        )

    navigate_actions = [
        NavigateAction(
            target_location=_make_pose(float(waypoint.x), float(waypoint.y), 0.0, world.root)
        )
        for waypoint in path[1:]
    ]

    print(
        f"[navigation] ({start_x:.2f}, {start_y:.2f}) -> "
        f"({goal_x:.2f}, {goal_y:.2f}): "
        f"{len(path)} nodes, {len(navigate_actions)} waypoints"
    )
    return navigate_actions


# =============================================================================
# JPT Loading and Sampling
# =============================================================================

def _load_jpt_model(model_path: str) -> NativeJPT:
    """Load a pre-fitted JPT model from disk and report the leaf count."""
    print(f"[jpt] Loading model from {model_path} ...")
    model = NativeJPT(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    ).load(model_path)
    print(f"[jpt] Model loaded — {len(model.leaves)} leaves")
    return model


def _sample_plan_parameters(
    jpt_model:  NativeJPT,
    correction: Optional[CausalSamplingCorrection] = None,
) -> PlanParameters:
    """
    Draw one joint sample from the JPT and map it to apartment PlanParameters.

    All sampling is performed in JPT coordinate space. The apartment world
    uses absolute positions, so coordinate offsets are applied after sampling:
        counter_approach_y  = pick_approach_y  + MILK_SPAWN_Y
        table_approach_x    = place_approach_x + (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X)
        table_approach_y    = place_approach_y + PLACE_TARGET_Y

    If a CausalSamplingCorrection is active, the primary cause variable is
    overridden with the recommended value clamped to the correction window
    and feasibility bounds before the coordinate offset is applied.

    Parameters
    ----------
    jpt_model:
        Fitted JPT model from which to draw the joint sample.
    correction:
        Optional one-shot correction from the Causal Circuit. If None or
        inactive, unconstrained JPT sampling is used.

    Returns
    -------
    PlanParameters
        Approach positions and arm selection in apartment absolute coordinates.
    """
    sample_row     = jpt_model.sample(1)[0]
    sample_by_name = {
        variable.name: sample_row[index]
        for index, variable in enumerate(JPT_VARIABLES)
    }

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
            f"[correction] {correction.jpt_variable_name}: {corrected_value:.4f}  "
            f"(window [{lower_bound:.4f}, {upper_bound:.4f}], "
            f"source iteration {correction.source_iteration})"
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


# =============================================================================
# Causal Circuit Construction
# =============================================================================

def _build_causal_circuit(training_csv_path: str) -> CausalCircuit:
    """
    Construct a CausalCircuit from the pre-fitted JPT model.

    The construction proceeds in three steps:

    1. The fitted pyjpt model is used to partition the training data into its
       leaf regions via apply(). This recovers the exact axis-aligned partition
       that pyjpt found during training without requiring a second fit.

    2. A ProbabilisticCircuit is built from these partitions: each partition
       becomes a ProductUnit of per-variable univariate distributions, fitted
       with NygaInduction for continuous variables and SymbolicDistribution for
       symbolic variables. The root is a SumUnit weighted by partition sizes.

    3. An MdVtree is constructed from the ATE_norm causal priority ordering
       over CAUSAL_VARIABLES, and CausalCircuit.from_probabilistic_circuit()
       imposes marginal determinism on the circuit.

    A support determinism verification step is run immediately after
    construction. If the circuit passes, all interventional queries are
    guaranteed to be tractable and exact at runtime. Violations are reported
    but do not prevent the circuit from being returned.

    Parameters
    ----------
    training_csv_path:
        Path to the CSV file used to train the JPT model, containing all
        columns listed in JPT_VARIABLES.

    Returns
    -------
    CausalCircuit
        A circuit ready for interventional failure diagnosis at runtime.
    """
    from probabilistic_model.distributions.distributions import SymbolicDistribution
    from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
    from probabilistic_model.learning.nyga_induction import NygaInduction
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
        ProbabilisticCircuit,
        ProductUnit,
        SumUnit,
        UnivariateDiscreteLeaf,
    )
    from probabilistic_model.utils import MissingDict
    from random_events.variable import Continuous as ContinuousVariableLocal
    import math

    print("[causal] Loading pyjpt model for leaf partitioning ...")
    pyjpt_model = NativeJPT(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    ).load(JPT_MODEL_PATH)

    print("[causal] Building ProbabilisticCircuit from training data ...")
    training_data   = pd.read_csv(training_csv_path)
    causal_columns  = [variable.name for variable in JPT_VARIABLES]
    model_data      = training_data[causal_columns]
    annotated_vars  = infer_variables_from_dataframe(model_data)

    try:
        leaf_assignments = pyjpt_model.apply(model_data)
        leaf_groups: dict = {}
        for row_index, leaf_object in enumerate(leaf_assignments):
            key = id(leaf_object)
            if key not in leaf_groups:
                leaf_groups[key] = []
            leaf_groups[key].append(row_index)

        partitions = [
            (indices, len(indices) / len(model_data))
            for indices in leaf_groups.values()
            if len(indices) > 0
        ]
        print(f"[causal] Leaf partitioning produced {len(partitions)} partitions")

    except Exception as error:
        print(f"[causal] Leaf partitioning failed ({error}) — using single-partition fallback")
        partitions = [(list(range(len(model_data))), 1.0)]

    probabilistic_circuit = ProbabilisticCircuit()
    root_sum_unit         = SumUnit(probabilistic_circuit=probabilistic_circuit)
    leaves_added          = 0

    for indices, weight in partitions:
        if weight <= 0:
            continue

        partition_data = model_data.iloc[indices].reset_index(drop=True)
        if len(partition_data) == 0:
            continue

        product_node   = ProductUnit(probabilistic_circuit=probabilistic_circuit)
        build_succeeded = True

        for annotated_variable in annotated_vars:
            column = partition_data[annotated_variable.variable.name]
            try:
                if isinstance(annotated_variable.variable, ContinuousVariableLocal):
                    column_values = column.values.astype(float)
                    nyga_induction = NygaInduction(
                        annotated_variable.variable,
                        min_likelihood_improvement=annotated_variable.min_likelihood_improvement,
                        min_samples_per_quantile=annotated_variable.min_samples_per_quantile,
                    )
                    distribution_circuit = nyga_induction.fit(column_values)
                    nyga_root = distribution_circuit.root
                    mounted_nodes = probabilistic_circuit.mount(nyga_root)
                    product_node.add_subcircuit(mounted_nodes[nyga_root.index])

                else:
                    all_elements = {
                        element: index
                        for index, element in enumerate(
                            annotated_variable.variable.domain.all_elements
                        )
                    }
                    symbolic_distribution = SymbolicDistribution(
                        variable=annotated_variable.variable,
                        probabilities=MissingDict(float),
                    )
                    encoded_values = column.apply(
                        lambda value: all_elements.get(
                            value, all_elements.get(str(value), 0)
                        )
                    ).values.astype(int)
                    symbolic_distribution.fit_from_indices(encoded_values)
                    leaf_node = UnivariateDiscreteLeaf(
                        symbolic_distribution,
                        probabilistic_circuit=probabilistic_circuit,
                    )
                    product_node.add_subcircuit(leaf_node)

            except Exception as distribution_error:
                print(
                    f"[causal] WARNING: distribution fit failed for "
                    f"{annotated_variable.variable.name}: {distribution_error}"
                )
                build_succeeded = False
                break

        if build_succeeded and len(product_node.subcircuits) == len(annotated_vars):
            root_sum_unit.add_subcircuit(product_node, math.log(weight))
            leaves_added += 1
        else:
            probabilistic_circuit.remove_node(product_node)

    if leaves_added == 0:
        raise RuntimeError(
            "ProbabilisticCircuit construction failed: no leaves were added. "
            "Verify that the training CSV columns match the expected variable names."
        )

    print(f"[causal] ProbabilisticCircuit built from {leaves_added} leaf partitions.")

    circuit_variables_by_name = {
        variable.name: variable for variable in probabilistic_circuit.variables
    }

    resolved_causal_variables = [
        circuit_variables_by_name[variable.name]
        for variable in CAUSAL_VARIABLES
        if variable.name in circuit_variables_by_name
    ]
    resolved_effect_variables = [
        circuit_variables_by_name[variable.name]
        for variable in EFFECT_VARIABLES
        if variable.name in circuit_variables_by_name
    ]
    resolved_priority_order = [
        circuit_variables_by_name[variable.name]
        for variable in CAUSAL_PRIORITY_ORDER
        if variable.name in circuit_variables_by_name
    ]

    marginal_determinism_tree = MarginalDeterminismTreeNode.from_causal_graph(
        causal_variables=resolved_causal_variables,
        effect_variables=resolved_effect_variables,
        causal_priority_order=resolved_priority_order,
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit=probabilistic_circuit,
        marginal_determinism_tree=marginal_determinism_tree,
        causal_variables=resolved_causal_variables,
        effect_variables=resolved_effect_variables,
    )

    try:
        causal_circuit.verify_support_determinism()
        print("[causal] Support determinism verification: PASS")
    except SupportDeterminismVerificationResult as verification_result:
        violation_summary = "; ".join(
            str(violation) for violation in verification_result.violations
        )
        print(f"[causal] Support determinism verification: FAIL — {violation_summary}")

    return causal_circuit


# =============================================================================
# Failure Diagnosis and Correction
# =============================================================================

def _extract_region_midpoint(region: Any, cause_variable: Any) -> float:
    """
    Extract the midpoint of the first interval in a recommended cause region.

    The recommended region is returned by the Causal Circuit as an event over
    the cause variable's domain. This function navigates the event structure
    to retrieve the interval bounds and returns their midpoint.
    """
    simple_set   = region.simple_sets[0]
    interval_set = simple_set[cause_variable]
    interval     = (
        interval_set.simple_sets[0]
        if hasattr(interval_set, "simple_sets")
        else interval_set
    )
    return (float(interval.lower) + float(interval.upper)) / 2.0


def _diagnose_failure_and_log(
    causal_circuit:   CausalCircuit,
    plan_parameters:  PlanParameters,
    iteration_number: int,
) -> Optional[FailureDiagnosisResult]:
    """
    Run causal failure diagnosis on a rejected plan and print a structured report.

    Apartment coordinates are remapped to JPT coordinate space before calling
    diagnose_failure(), because the Causal Circuit was fitted in JPT space.
    The diagnosis report uses apartment-world variable names for readability.

    For each cause variable, the circuit computes the interventional success
    probability at the observed value. The variable with the lowest probability
    is identified as the primary cause. Plans for which all probabilities are
    zero are flagged as out-of-support and excluded from correction downstream.

    Parameters
    ----------
    causal_circuit:
        The fitted Causal Circuit.
    plan_parameters:
        The parameters of the failed plan, in apartment absolute coordinates.
    iteration_number:
        Current iteration number, used for the report header.

    Returns
    -------
    FailureDiagnosisResult or None
        Diagnosis result, or None if the circuit raised an exception.
    """
    cause_variable_by_name  = {v.name: v for v in causal_circuit.causal_variables}
    effect_variable_by_name = {v.name: v for v in causal_circuit.effect_variables}

    observed_values_in_jpt_space = {
        cause_variable_by_name["pick_approach_x"]:  plan_parameters.counter_approach_x,
        cause_variable_by_name["pick_approach_y"]:  (
            plan_parameters.counter_approach_y - MILK_SPAWN_Y
        ),
        cause_variable_by_name["place_approach_x"]: (
            plan_parameters.table_approach_x
            - (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X)
        ),
        cause_variable_by_name["place_approach_y"]: (
            plan_parameters.table_approach_y - PLACE_TARGET_Y
        ),
    }

    try:
        diagnosis = causal_circuit.diagnose_failure(
            observed_values=observed_values_in_jpt_space,
            effect_variable=effect_variable_by_name["milk_end_z"],
            query_resolution=CAUSAL_QUERY_RESOLUTION,
        )

        primary_variable_name = diagnosis.primary_cause_variable.name
        primary_display_name  = JPT_TO_APARTMENT_VARIABLE_NAME.get(
            primary_variable_name, primary_variable_name
        )
        out_of_support_marker = (
            "  <- OUT OF TRAINING SUPPORT"
            if diagnosis.interventional_probability_at_failure == 0.0
            else ""
        )
        recommended_midpoint = (
            _extract_region_midpoint(
                diagnosis.recommended_region, diagnosis.primary_cause_variable
            )
            if diagnosis.recommended_region is not None
            else None
        )

        separator = "-" * 58
        print(f"\n  +{separator}")
        print(f"  | CAUSAL FAILURE DIAGNOSIS  (iteration {iteration_number})")
        print(f"  +{separator}")
        print(f"  | Primary cause   : {primary_display_name}")
        print(f"  | Observed value  : {diagnosis.actual_value:.4f}")
        print(
            f"  | P(Y>=tau|do)   : "
            f"{diagnosis.interventional_probability_at_failure:.4f}"
            f"{out_of_support_marker}"
        )
        if recommended_midpoint is not None:
            print(f"  | Recommended     : {recommended_midpoint:.4f}  (region midpoint)")
            print(
                f"  | P(Y>=tau|rec)  : "
                f"{diagnosis.interventional_probability_at_recommendation:.4f}"
            )
        print(f"  |")
        print(f"  | All cause variables:")
        for cause_variable, variable_result in diagnosis.all_variable_results.items():
            display_name    = JPT_TO_APARTMENT_VARIABLE_NAME.get(
                cause_variable.name, cause_variable.name
            )
            primary_marker  = (
                "  <- PRIMARY CAUSE"
                if cause_variable == diagnosis.primary_cause_variable
                else ""
            )
            support_marker  = (
                "  [OUT OF SUPPORT]"
                if variable_result["interventional_probability"] == 0.0
                else ""
            )
            print(
                f"  |   {display_name:<26} "
                f"observed={variable_result['actual_value']:.4f}  "
                f"P={variable_result['interventional_probability']:.4f}"
                f"{support_marker}{primary_marker}"
            )
        print(f"  +{separator}\n")

        return diagnosis

    except Exception as error:
        print(
            f"[causal] Diagnosis failed at iteration {iteration_number}: {error}"
        )
        return None


def _build_correction_from_diagnosis(
    diagnosis:        FailureDiagnosisResult,
    iteration_number: int,
) -> Optional[CausalSamplingCorrection]:
    """
    Construct a CausalSamplingCorrection from a FailureDiagnosisResult.

    Returns None if the diagnosis provides no recommended region, which
    occurs when the primary cause variable has zero interventional probability
    (out-of-support plan) or when no high-probability region can be identified.

    Parameters
    ----------
    diagnosis:
        The FailureDiagnosisResult produced by the Causal Circuit.
    iteration_number:
        Current iteration number, used for logging and the correction record.

    Returns
    -------
    CausalSamplingCorrection or None
    """
    if diagnosis.recommended_region is None:
        print("[correction] No recommended region available — correction skipped.")
        return None

    recommended_midpoint = _extract_region_midpoint(
        diagnosis.recommended_region, diagnosis.primary_cause_variable
    )
    primary_variable_name = diagnosis.primary_cause_variable.name
    variable_bounds       = VARIABLE_BOUNDS_IN_JPT_SPACE.get(
        primary_variable_name, (float("-inf"), float("inf"))
    )

    correction = CausalSamplingCorrection(
        active=True,
        jpt_variable_name=primary_variable_name,
        recommended_value=recommended_midpoint,
        correction_window=CAUSAL_CORRECTION_WINDOW,
        variable_bounds=variable_bounds,
        source_iteration=iteration_number,
    )

    print(
        f"[correction] Scheduled: {primary_variable_name} -> {recommended_midpoint:.4f}  "
        f"(window +/-{CAUSAL_CORRECTION_WINDOW}, bounds {variable_bounds})"
    )
    return correction


# =============================================================================
# Plan Construction
# =============================================================================

def _build_sequential_plan(
    planning_context: Context,
    actions:          List[Any],
) -> SequentialNode:
    """Build a SequentialNode from a list of actions using the framework factory."""
    from pycram.plans.factories import sequential
    return sequential(actions, context=planning_context)


def _build_seed_plan(
    planning_context: Context,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
    navigation_map:   GraphOfConvexSets,
    bounds_array:     np.ndarray,
) -> SequentialNode:
    """
    Build a deterministic seed plan using fixed approach positions and the
    right arm. Used for iteration 1 to confirm that the world, robot, and
    database are correctly initialised before JPT sampling begins.
    """
    seed_arm = Arms.RIGHT

    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, bounds_array,
        start_x=ROBOT_INIT_X,      start_y=ROBOT_INIT_Y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        planning_context, navigation_map, bounds_array,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X,    goal_y=TABLE_APPROACH_Y,
        world=world,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)

    print(
        f"[plan] seed — "
        f"counter: ({COUNTER_APPROACH_X}, {COUNTER_APPROACH_Y})  "
        f"table: ({TABLE_APPROACH_X}, {TABLE_APPROACH_Y})  "
        f"arm: {seed_arm}"
    )

    return _build_sequential_plan(planning_context, [
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
    ])


def _build_sampled_plan(
    planning_context: Context,
    plan_parameters:  PlanParameters,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
    navigation_map:   GraphOfConvexSets,
    bounds_array:     np.ndarray,
    robot_start_x:    float,
    robot_start_y:    float,
) -> SequentialNode:
    """
    Build a plan from JPT-sampled or causally corrected approach parameters.

    The GCS planner routes the robot from its current position to the counter
    approach position, then from the counter to the table approach position.
    """
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )

    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, bounds_array,
        start_x=robot_start_x,
        start_y=robot_start_y,
        goal_x=plan_parameters.counter_approach_x,
        goal_y=plan_parameters.counter_approach_y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        planning_context, navigation_map, bounds_array,
        start_x=plan_parameters.counter_approach_x,
        start_y=plan_parameters.counter_approach_y,
        goal_x=plan_parameters.table_approach_x,
        goal_y=plan_parameters.table_approach_y,
        world=world,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)

    print(
        f"[plan] sampled — "
        f"counter: ({plan_parameters.counter_approach_x:.3f}, "
        f"{plan_parameters.counter_approach_y:.3f})  "
        f"table: ({plan_parameters.table_approach_x:.3f}, "
        f"{plan_parameters.table_approach_y:.3f})  "
        f"arm: {plan_parameters.pick_arm}"
    )

    return _build_sequential_plan(planning_context, [
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
    ])


def _navigate_robot_to_start(
    planning_context: Context,
    navigation_map:   GraphOfConvexSets,
    bounds_array:     np.ndarray,
    world:            World,
    robot_x:          float,
    robot_y:          float,
) -> None:
    """
    Return the robot to the fixed initial position via GCS navigation.
    Called at the end of each iteration to reset the robot pose for the next
    iteration. Navigation failures are logged but do not terminate the run.
    """
    print(
        f"[reset] Navigating robot from ({robot_x:.2f}, {robot_y:.2f}) "
        f"to start ({ROBOT_INIT_X}, {ROBOT_INIT_Y})"
    )
    try:
        return_actions = _navigate_via_gcs(
            planning_context, navigation_map, bounds_array,
            start_x=robot_x,     start_y=robot_y,
            goal_x=ROBOT_INIT_X, goal_y=ROBOT_INIT_Y,
            world=world,
        )
    except ValueError as error:
        print(f"[reset] WARNING: GCS path planning failed: {error}")
        return

    from pycram.plans.factories import sequential
    return_plan = sequential(return_actions, context=planning_context)
    with simulated_robot:
        try:
            return_plan.perform()
            print("[reset] Robot at initial position.")
        except Exception as error:
            print(f"[reset] WARNING: Return navigation failed: {error}")


# =============================================================================
# Run Summary
# =============================================================================

def _print_run_summary(
    statistics:           RunStatistics,
    number_of_iterations: int,
    database_session:     Session,
) -> None:
    """
    Print the final run summary in a format that mirrors the JPT-only baseline
    for direct side-by-side comparison in the paper.

    Reports overall success rate, failure counts, causal correction statistics
    (total attempts, corrected successes, baseline vs corrected success rate,
    and lift), and per-iteration correction detail for all iterations that
    triggered the correction loop.
    """
    correction_records    = statistics.correction_records
    overall_success_rate  = 100 * statistics.successful_count // number_of_iterations
    uncorrected_iterations = number_of_iterations - len(correction_records)
    uncorrected_successes  = statistics.successful_count - statistics.corrected_success_count
    baseline_success_rate  = 100 * uncorrected_successes // max(uncorrected_iterations, 1)

    divider = "=" * 64
    print(f"\n{divider}")
    print(f"  Run complete.")
    print(f"  Total iterations         : {number_of_iterations}")
    print(f"  Successful plans         : {statistics.successful_count}  ({overall_success_rate}%)")
    print(f"  Failed iterations        : {statistics.failed_iterations}")
    print(f"  Total failed attempts    : {statistics.failed_attempts}")
    print(f"  Hard failures            : {statistics.hard_failure_count}  "
          f"(>{MAX_CORRECTION_ATTEMPTS} attempts exhausted)")

    print(f"")
    print(f"  -- Causal Correction Summary {'─' * 35}")
    print(f"  Iterations with corrections : {len(correction_records)}")
    print(f"  Total correction attempts   : {statistics.corrected_attempt_count}")
    print(f"  Corrections that succeeded  : {statistics.corrected_success_count}")
    print(f"  Corrections that failed     : {statistics.corrected_failure_count}")

    if statistics.corrected_attempt_count > 0:
        corrected_success_rate = (
            100 * statistics.corrected_success_count
            // statistics.corrected_attempt_count
        )
        lift = corrected_success_rate - baseline_success_rate
        print(f"  Baseline success rate       : {baseline_success_rate}%")
        print(f"  Corrected success rate      : {corrected_success_rate}%")
        print(f"  Causal correction lift      : {lift:+d}%")

    successful_records = [record for record in correction_records if record.succeeded]
    failed_records     = [record for record in correction_records if not record.succeeded]

    if successful_records:
        average_attempts = (
            sum(record.attempts for record in successful_records) / len(successful_records)
        )
        average_time = (
            sum(record.elapsed_seconds for record in successful_records)
            / len(successful_records)
        )
        maximum_attempts = max(record.attempts for record in successful_records)
        maximum_time     = max(record.elapsed_seconds for record in successful_records)

        print(f"")
        print(f"  -- Correction Attempt Statistics (successful corrections) {'─' * 6}")
        print(f"  Iterations eventually succeeded : {len(successful_records)}")
        print(f"  Average attempts until success  : {average_attempts:.2f}")
        print(f"  Average time until success      : {average_time:.1f}s")
        print(f"  Maximum attempts (one iteration): {maximum_attempts}")
        print(f"  Maximum time     (one iteration): {maximum_time:.1f}s")
        print(f"")
        print(f"  Per-iteration detail (succeeded):")
        for record in successful_records:
            print(
                f"    iteration {record.iteration:>4d}: "
                f"{record.attempts} attempt(s),  "
                f"{record.elapsed_seconds:.1f}s  SUCCESS"
            )

    if failed_records:
        print(f"")
        print(f"  Per-iteration detail (hard failures):")
        for record in failed_records:
            print(
                f"    iteration {record.iteration:>4d}: "
                f"{record.attempts} attempt(s),  "
                f"{record.elapsed_seconds:.1f}s  HARD FAILURE"
            )

    if not correction_records:
        print(f"  (No corrections issued — all plans succeeded on first attempt)")

    print(f"{divider}")

    try:
        row_count = database_session.execute(
            text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
        ).scalar()
        print(f"  Database rows (SequentialPlanDAO): {row_count}")
    except Exception as error:
        print(f"[database] Could not read row count: {error}")


# =============================================================================
# Main Experiment Entry Point
# =============================================================================

def run_pick_and_place_with_causal_correction() -> None:
    """
    Run 5,000 iterations of JPT-guided pick-and-place with causal failure
    diagnosis and active parameter correction in the apartment simulation.

    Iteration 1 uses fixed deterministic parameters (seed plan) to verify
    that the world, robot, and database are correctly initialised.

    For iterations 2 through 5,000:
      - Draw candidate parameters from the JPT (unconstrained)
      - Attempt plan execution
      - On failure: diagnose with the Causal Circuit, apply a one-shot
        CausalSamplingCorrection, and retry up to MAX_CORRECTION_ATTEMPTS times
      - If the corrected attempt fails, revert to unconstrained JPT sampling
        without chaining further corrections
      - Track attempts and wall-clock time per iteration for paper comparison

    Successful plans are persisted to the PostgreSQL database as SequentialPlanDAO
    records. The final summary is printed in a format that mirrors the JPT-only
    baseline for direct comparison.
    """
    print("=" * 64)
    print("  Causally-Aware Pick-and-Place — Apartment World")
    print(f"  Iterations          : {NUMBER_OF_ITERATIONS}")
    print(f"  Max correction cap  : {MAX_CORRECTION_ATTEMPTS} per iteration")
    print(f"  Place target        : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model           : {JPT_MODEL_PATH}")
    print(f"  Training data       : {TRAINING_CSV_PATH}")
    print(f"  Database            : {DATABASE_URI}")
    print(f"  Correction window   : +/-{CAUSAL_CORRECTION_WINDOW}")
    print("=" * 64)

    print("\n[1/6] Building apartment world ...")
    world, robot = _build_apartment_world(APARTMENT_URDF_PATH)
    _apartment_world_reference[0] = world
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk spawned at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    bounds_array   = _build_gcs_bounds_array(navigation_map)

    print("\n[2/6] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/6] Loading JPT model ...")
    jpt_model = _load_jpt_model(JPT_MODEL_PATH)

    print("\n[4/6] Building Causal Circuit ...")
    causal_circuit = _build_causal_circuit(TRAINING_CSV_PATH)

    print("\n[5/6] Starting ROS2 ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_causal_node")
    ros_spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True
    )
    ros_spin_thread.start()
    print("  ROS2 node started.")

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context    = Context(world, robot, None, evaluate_conditions=False)
        statistics          = RunStatistics()
        robot_x: float      = ROBOT_INIT_X
        robot_y: float      = ROBOT_INIT_Y

        print("\n[6/6] Running iterations ...")

        for iteration_number in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration_number}/{NUMBER_OF_ITERATIONS}  |  "
                f"success={statistics.successful_count}  "
                f"failed_iterations={statistics.failed_iterations}  "
                f"total_attempts={statistics.failed_attempts}  "
                f"corrections={statistics.corrected_success_count}/"
                f"{statistics.corrected_attempt_count}"
            )
            print(f"{'=' * 64}")

            plan               = None
            current_parameters = None

            # ------------------------------------------------------------------
            # Iteration 1: deterministic seed plan
            # ------------------------------------------------------------------
            if iteration_number == 1:
                print("  Mode: SEED (deterministic)")
                try:
                    plan = _build_seed_plan(
                        planning_context, world, robot, milk_body,
                        navigation_map, bounds_array,
                    )
                except ValueError as error:
                    statistics.failed_iterations += 1
                    print(f"  RESULT: FAILED (plan build) — {error}")

                if plan is not None:
                    print("  Executing seed plan ...")
                    execution_succeeded = False
                    with simulated_robot:
                        try:
                            plan.perform()
                            execution_succeeded = True
                        except Exception as error:
                            statistics.failed_iterations += 1
                            print(
                                f"  RESULT: FAILED — "
                                f"{type(error).__name__}: {error}"
                            )

                    if execution_succeeded:
                        statistics.successful_count += 1
                        print(
                            f"  RESULT: SUCCESS  "
                            f"({statistics.successful_count}/{iteration_number} stored)"
                        )
                        try:
                            _persist_plan(database_session, plan)
                        except Exception as error:
                            print(f"[database] ERROR: {error}")
                            traceback.print_exc()
                            database_session.rollback()

            # ------------------------------------------------------------------
            # Iterations 2+: JPT sampling with causal correction on failure
            # ------------------------------------------------------------------
            else:
                print("  Mode: JPT-SAMPLED + CAUSAL CORRECTION on failure")

                current_parameters = _sample_plan_parameters(jpt_model)
                try:
                    plan = _build_sampled_plan(
                        planning_context, current_parameters, world, robot, milk_body,
                        navigation_map, bounds_array,
                        robot_start_x=robot_x, robot_start_y=robot_y,
                    )
                except ValueError as error:
                    statistics.failed_attempts += 1
                    print(f"  [attempt 1] FAILED (plan build) — {error}")
                    plan = None

                execution_succeeded = False
                if plan is not None:
                    print("  [attempt 1] Executing plan ...")
                    with simulated_robot:
                        try:
                            plan.perform()
                            execution_succeeded = True
                            print("  [attempt 1] SUCCESS")
                        except Exception as error:
                            statistics.failed_attempts += 1
                            print(
                                f"  [attempt 1] FAILED — "
                                f"{type(error).__name__}: {error}"
                            )

                # --------------------------------------------------------------
                # Causal correction loop
                # --------------------------------------------------------------
                if not execution_succeeded:
                    diagnosis = _diagnose_failure_and_log(
                        causal_circuit, current_parameters, iteration_number
                    )
                    pending_correction = (
                        _build_correction_from_diagnosis(diagnosis, iteration_number)
                        if diagnosis is not None
                        else None
                    )

                    if pending_correction is not None:
                        _respawn_milk(world, milk_body)
                        _navigate_robot_to_start(
                            planning_context, navigation_map, bounds_array, world,
                            robot_x=current_parameters.table_approach_x,
                            robot_y=current_parameters.table_approach_y,
                        )

                        correction_attempt_count = 0
                        correction_start_time    = time.time()

                        while (
                            not execution_succeeded
                            and pending_correction is not None
                            and correction_attempt_count < MAX_CORRECTION_ATTEMPTS
                        ):
                            correction_attempt_count          += 1
                            statistics.corrected_attempt_count += 1

                            current_parameters = _sample_plan_parameters(
                                jpt_model, correction=pending_correction
                            )
                            pending_correction = None

                            try:
                                plan = _build_sampled_plan(
                                    planning_context, current_parameters, world, robot,
                                    milk_body, navigation_map, bounds_array,
                                    robot_start_x=ROBOT_INIT_X, robot_start_y=ROBOT_INIT_Y,
                                )
                            except ValueError as error:
                                statistics.failed_attempts       += 1
                                statistics.corrected_failure_count += 1
                                print(
                                    f"  [correction {correction_attempt_count}] "
                                    f"FAILED (plan build) — {error}"
                                )
                                continue

                            print(
                                f"  [correction {correction_attempt_count}] "
                                f"Executing corrected plan ..."
                            )
                            with simulated_robot:
                                try:
                                    plan.perform()
                                    execution_succeeded = True
                                    elapsed = time.time() - correction_start_time
                                    print(
                                        f"  [correction {correction_attempt_count}] "
                                        f"SUCCESS  ({elapsed:.1f}s)"
                                    )
                                except Exception as error:
                                    statistics.failed_attempts       += 1
                                    statistics.corrected_failure_count += 1
                                    print(
                                        f"  [correction {correction_attempt_count}] "
                                        f"FAILED — {type(error).__name__}: {error}"
                                    )
                                    if correction_attempt_count < MAX_CORRECTION_ATTEMPTS:
                                        _respawn_milk(world, milk_body)
                                        _navigate_robot_to_start(
                                            planning_context, navigation_map, bounds_array,
                                            world,
                                            robot_x=current_parameters.table_approach_x,
                                            robot_y=current_parameters.table_approach_y,
                                        )
                                        new_diagnosis = _diagnose_failure_and_log(
                                            causal_circuit, current_parameters,
                                            iteration_number,
                                        )
                                        if new_diagnosis is not None:
                                            pending_correction = _build_correction_from_diagnosis(
                                                new_diagnosis, iteration_number
                                            )

                        elapsed_total = time.time() - correction_start_time

                        if execution_succeeded:
                            statistics.corrected_success_count += 1
                            print(
                                f"  [causal] Succeeded after "
                                f"{correction_attempt_count} correction attempt(s) "
                                f"in {elapsed_total:.1f}s"
                            )
                            statistics.record_correction(
                                iteration_number,
                                correction_attempt_count,
                                elapsed_total,
                                succeeded=True,
                            )
                        else:
                            statistics.hard_failure_count  += 1
                            statistics.failed_iterations   += 1
                            print(
                                f"  RESULT: HARD FAILURE — "
                                f"{correction_attempt_count} correction attempt(s) "
                                f"exhausted ({elapsed_total:.1f}s)"
                            )
                            statistics.record_correction(
                                iteration_number,
                                correction_attempt_count,
                                elapsed_total,
                                succeeded=False,
                            )

                if execution_succeeded:
                    statistics.successful_count += 1
                    print(
                        f"  RESULT: SUCCESS  "
                        f"({statistics.successful_count}/{iteration_number} stored,  "
                        f"{NUMBER_OF_ITERATIONS - iteration_number} remaining)"
                    )
                    try:
                        _persist_plan(database_session, plan)
                    except Exception as error:
                        print(f"[database] ERROR: {error}")
                        traceback.print_exc()
                        database_session.rollback()

            # ------------------------------------------------------------------
            # Reset world state for next iteration
            # ------------------------------------------------------------------
            if iteration_number == 1:
                end_x, end_y = TABLE_APPROACH_X, TABLE_APPROACH_Y
            elif current_parameters is not None:
                end_x = current_parameters.table_approach_x
                end_y = current_parameters.table_approach_y
            else:
                end_x, end_y = TABLE_APPROACH_X, TABLE_APPROACH_Y

            print("  Resetting world ...")
            _respawn_milk(world, milk_body)
            _navigate_robot_to_start(
                planning_context, navigation_map, bounds_array, world,
                robot_x=end_x, robot_y=end_y,
            )
            robot_x = ROBOT_INIT_X
            robot_y = ROBOT_INIT_Y

        _print_run_summary(statistics, NUMBER_OF_ITERATIONS, database_session)
        sys.modules[__name__]._last_run_statistics = statistics

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    run_pick_and_place_with_causal_correction()