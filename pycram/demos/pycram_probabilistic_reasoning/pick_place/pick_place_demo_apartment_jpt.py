"""
Apartment World: JPT-Guided Pick-and-Place (Baseline)

Implements the JPT-only baseline described in:
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

When a plan fails execution, the system resamples all parameters jointly
from the JPT and retries immediately. No causal diagnosis or targeted
correction is applied. This is the blind resampling baseline against which
the Causal Circuit variant (pick_and_place_causal.py) is compared.

Variable Mapping (open-world JPT space -> apartment world)
----------------------------------------------------------
    pick_approach_x / pick_approach_y   ->  counter_approach_x / counter_approach_y
    place_approach_x / place_approach_y ->  table_approach_x   / table_approach_y
    milk_end_x / milk_end_y / milk_end_z                       ->  unchanged
    pick_arm                                                    ->  unchanged

Experiment Configuration
------------------------
    Iterations per run     : 5,000
    Max resample attempts  : 10 per iteration (matches causal correction cap)
    JPT model              : pick_and_place_jpt.json (high-quality or degraded)
    Training data          : pick_and_place_dataframe.csv (1,742 rows)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# nav2_msgs is not installed in this environment; mock before any import
# that transitively references it via pycram -> giskardpy -> ros_tasks.
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
from typing import Any, List, Optional

import numpy as np
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


# Apply all patches at import time, before any framework objects are created.
_apartment_world_reference: list = [None]

_patch_plan_migrate_nodes()
_patch_action_description_add_subplan()
_patch_active_connection_raw_dof(_apartment_world_reference)
_patch_orm_numpy_type_decorator()

NavigateAction = BaseNavigateAction


# =============================================================================
# Experiment Constants
# =============================================================================

NUMBER_OF_ITERATIONS:  int = 5000
MAX_RESAMPLE_ATTEMPTS: int = 10

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

# JPT hyperparameters (must match the fitted model)
JPT_MIN_SAMPLES_PER_LEAF: int = 25


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
    """Sampled parameters for one pick-and-place iteration."""
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


@dataclass
class ResamplingRecord:
    """
    Record of one iteration that required more than one JPT sample before success.

    Attributes
    ----------
    iteration:
        Iteration number within the 5,000-iteration run.
    attempts:
        Total number of attempts, including the successful one.
    elapsed_seconds:
        Wall-clock time for the full resampling loop.
    """
    iteration:       int
    attempts:        int
    elapsed_seconds: float


@dataclass
class RunStatistics:
    """
    Tracks success, failure, and resampling statistics across all iterations.

    Attributes
    ----------
    successful_count:
        Total iterations that ended in a successful plan execution.
    failed_iterations:
        Total iterations that ended in failure (hard failures only).
    failed_attempts:
        Total individual attempt failures across all iterations.
    hard_failure_count:
        Iterations that exhausted MAX_RESAMPLE_ATTEMPTS without success.
    resampling_records:
        Per-iteration resampling detail for iterations requiring more than
        one attempt.
    """
    successful_count:   int  = 0
    failed_iterations:  int  = 0
    failed_attempts:    int  = 0
    hard_failure_count: int  = 0
    resampling_records: list = None

    def __post_init__(self) -> None:
        if self.resampling_records is None:
            self.resampling_records = []

    def record_resampling(
        self, iteration: int, attempts: int, elapsed_seconds: float
    ) -> None:
        self.resampling_records.append(
            ResamplingRecord(
                iteration=iteration,
                attempts=attempts,
                elapsed_seconds=elapsed_seconds,
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


def _sample_plan_parameters(jpt_model: NativeJPT) -> PlanParameters:
    """
    Draw one joint sample from the JPT and map it to apartment PlanParameters.

    All sampling is performed in JPT coordinate space. The apartment world
    uses absolute positions, so coordinate offsets are applied after sampling:
        counter_approach_y  = pick_approach_y  + MILK_SPAWN_Y
        table_approach_x    = place_approach_x + (PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X)
        table_approach_y    = place_approach_y + PLACE_TARGET_Y

    Parameters
    ----------
    jpt_model:
        Fitted JPT model from which to draw the joint sample.

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
    Build a plan from JPT-sampled approach parameters.

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
    Called between resampling attempts and at the end of each iteration to
    reset the robot pose. Navigation failures are logged but do not terminate
    the run.
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
    Print the final run summary in a format that mirrors the Causal Circuit
    variant for direct side-by-side comparison in the paper.

    Reports overall success rate, failure counts, and per-iteration resampling
    detail for all iterations that required more than one attempt.
    """
    success_rate       = 100 * statistics.successful_count // number_of_iterations
    resampling_records = statistics.resampling_records

    divider = "=" * 64
    print(f"\n{divider}")
    print(f"  Run complete.")
    print(f"  Total iterations         : {number_of_iterations}")
    print(f"  Successful plans         : {statistics.successful_count}  ({success_rate}%)")
    print(f"  Failed iterations        : {statistics.failed_iterations}")
    print(f"  Total failed attempts    : {statistics.failed_attempts}")
    print(f"  Hard failures            : {statistics.hard_failure_count}  "
          f"(>{MAX_RESAMPLE_ATTEMPTS} attempts exhausted)")
    print(f"  Database                 : {DATABASE_URI}")

    if resampling_records:
        average_attempts = (
            sum(record.attempts for record in resampling_records)
            / len(resampling_records)
        )
        average_time = (
            sum(record.elapsed_seconds for record in resampling_records)
            / len(resampling_records)
        )
        maximum_attempts = max(record.attempts for record in resampling_records)
        maximum_time     = max(record.elapsed_seconds for record in resampling_records)

        print(f"")
        print(f"  -- JPT Resampling Statistics (iterations requiring >1 attempt) {'─' * 3}")
        print(f"  Iterations requiring resampling : {len(resampling_records)}")
        print(f"  Average attempts until success  : {average_attempts:.2f}")
        print(f"  Average time until success      : {average_time:.1f}s")
        print(f"  Maximum attempts (one iteration): {maximum_attempts}")
        print(f"  Maximum time     (one iteration): {maximum_time:.1f}s")
        print(f"")
        print(f"  Per-iteration resampling detail:")
        for record in resampling_records:
            print(
                f"    iteration {record.iteration:>4d}: "
                f"{record.attempts} attempt(s),  "
                f"{record.elapsed_seconds:.1f}s"
            )
    else:
        print(f"  (No resampling required — all plans succeeded on first attempt)")

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

def run_pick_and_place_jpt_baseline() -> None:
    """
    Run 5,000 iterations of JPT-guided pick-and-place in the apartment
    simulation without causal failure diagnosis.

    Iteration 1 uses fixed deterministic parameters (seed plan) to verify
    that the world, robot, and database are correctly initialised.

    For iterations 2 through 5,000:
      - Draw candidate parameters jointly from the JPT
      - Attempt plan execution
      - On failure: respawn milk, return robot to start, resample from JPT,
        and retry — up to MAX_RESAMPLE_ATTEMPTS times per iteration
      - If all attempts fail, declare a hard failure for that iteration

    No causal diagnosis or targeted correction is applied. This is the blind
    resampling baseline against which the Causal Circuit variant is compared.
    Successful plans are persisted to the PostgreSQL database.
    """
    print("=" * 64)
    print("  JPT Pick-and-Place Baseline — Apartment World")
    print(f"  Iterations         : {NUMBER_OF_ITERATIONS}")
    print(f"  Max resample cap   : {MAX_RESAMPLE_ATTEMPTS} per iteration")
    print(f"  Place target       : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model          : {JPT_MODEL_PATH}")
    print(f"  Database           : {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/5] Building apartment world ...")
    world, robot = _build_apartment_world(APARTMENT_URDF_PATH)
    _apartment_world_reference[0] = world
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk spawned at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    bounds_array   = _build_gcs_bounds_array(navigation_map)

    print("\n[2/5] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/5] Loading JPT model ...")
    jpt_model = _load_jpt_model(JPT_MODEL_PATH)

    print("\n[4/5] Starting ROS2 ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_jpt_baseline_node")
    ros_spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True
    )
    ros_spin_thread.start()
    print("  ROS2 node started.")

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context  = Context(world, robot, None, evaluate_conditions=False)
        statistics        = RunStatistics()
        robot_x: float    = ROBOT_INIT_X
        robot_y: float    = ROBOT_INIT_Y

        print("\n[5/5] Running iterations ...")

        for iteration_number in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration_number}/{NUMBER_OF_ITERATIONS}  |  "
                f"success={statistics.successful_count}  "
                f"failed_iterations={statistics.failed_iterations}  "
                f"total_attempts={statistics.failed_attempts}"
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
                    with simulated_robot:
                        try:
                            plan.perform()
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
                        except Exception as error:
                            statistics.failed_iterations += 1
                            print(
                                f"  RESULT: FAILED — "
                                f"{type(error).__name__}: {error}"
                            )

            # ------------------------------------------------------------------
            # Iterations 2+: JPT sampling with blind resampling on failure
            # ------------------------------------------------------------------
            else:
                print("  Mode: JPT-SAMPLED (blind resampling on failure)")

                attempt_count       = 0
                execution_succeeded = False
                resampling_start    = time.time()

                while not execution_succeeded and attempt_count < MAX_RESAMPLE_ATTEMPTS:
                    attempt_count     += 1
                    current_parameters = _sample_plan_parameters(jpt_model)

                    try:
                        plan = _build_sampled_plan(
                            planning_context, current_parameters, world, robot, milk_body,
                            navigation_map, bounds_array,
                            robot_start_x=robot_x, robot_start_y=robot_y,
                        )
                    except ValueError as error:
                        statistics.failed_attempts += 1
                        print(
                            f"  [attempt {attempt_count}] FAILED (plan build) — {error}  "
                            f"-> resampling ..."
                        )
                        continue

                    print(f"  [attempt {attempt_count}] Executing plan ...")
                    with simulated_robot:
                        try:
                            plan.perform()
                            execution_succeeded = True
                            elapsed = time.time() - resampling_start
                            print(
                                f"  [attempt {attempt_count}] SUCCESS  ({elapsed:.1f}s)"
                            )
                        except Exception as error:
                            statistics.failed_attempts += 1
                            print(
                                f"  [attempt {attempt_count}] FAILED — "
                                f"{type(error).__name__}: {error}  "
                                f"-> resampling ..."
                            )
                            _respawn_milk(world, milk_body)
                            _navigate_robot_to_start(
                                planning_context, navigation_map, bounds_array, world,
                                robot_x=current_parameters.table_approach_x,
                                robot_y=current_parameters.table_approach_y,
                            )

                elapsed_total = time.time() - resampling_start

                if not execution_succeeded:
                    statistics.failed_iterations += 1
                    statistics.hard_failure_count += 1
                    print(
                        f"  RESULT: HARD FAILURE — "
                        f"{MAX_RESAMPLE_ATTEMPTS} attempts exhausted "
                        f"({elapsed_total:.1f}s)"
                    )
                else:
                    if attempt_count > 1:
                        statistics.record_resampling(
                            iteration_number, attempt_count, elapsed_total
                        )
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
    run_pick_and_place_jpt_baseline()