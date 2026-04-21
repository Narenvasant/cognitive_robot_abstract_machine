"""
Apartment world: JPT-guided pick-and-place with GCS navigation.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
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
from jpt.trees import JPT as JointProbabilityTree
from jpt.variables import NumericVariable, SymbolicVariable

from pycram.robot_plans.actions.core.navigation import NavigateAction as _NavigateActionBase
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
# Patch: fix _migrate_nodes_from_plan stale-index bug
#
# Plan._migrate_nodes_from_plan() clears other.plan_graph after snapshotting
# edges, but does not update node.index for single-node plans (no edges to
# re-add). This leaves the migrated root with a stale index, causing IndexError
# in rustworkx on the next graph access (e.g. NavigateAction.execute()).
# ---------------------------------------------------------------------------

def _patched_migrate_nodes_from_plan(self, other):
    # Snapshot everything before touching the graph
    all_other_nodes = list(other.all_nodes)
    other_edges = list(other.edges)
    # Find root by in-degree (avoids calling .parent which uses stale indices)
    root_ref = None
    if all_other_nodes:
        for candidate in all_other_nodes:
            preds = other.plan_graph.predecessors(candidate.index)
            if len(preds) == 0:
                root_ref = candidate
                break
    # Now clear and re-register all nodes in self with fresh indices
    other.plan_graph.clear()
    for node in all_other_nodes:
        node.index = None
        node.plan = None
    for node in all_other_nodes:
        self.add_node(node)
    for edge in other_edges:
        self.add_edge(edge[0], edge[1])
    return root_ref


from pycram.plans.plan import Plan as _Plan
_Plan._migrate_nodes_from_plan = _patched_migrate_nodes_from_plan

# Re-alias NavigateAction now that the patch is applied
NavigateAction = _NavigateActionBase

# Patch ActiveConnection1DOF.raw_dof to redirect stale _world references.
# After merge_world_at_pose clears pr2_world, RevoluteConnection objects stored
# in JointState.connections still have self._world = pr2_world (empty/cleared).
# raw_dof calls self._world.get_degree_of_freedom_by_id — which fails on the
# empty pr2_world. We keep a module-level reference to the apartment world and
# redirect the lookup when self._world has an empty hash table.
_APARTMENT_WORLD = None  # set after world construction

from semantic_digital_twin.world_description.connections import ActiveConnection1DOF as _AC1DOF
_orig_raw_dof = _AC1DOF.raw_dof.fget

def _robust_raw_dof(self):
    # If self._world has no DOFs registered (cleared world), redirect to apartment world
    target_world = self._world
    if (target_world is None or
            len(target_world._world_entity_hash_table) == 0 or
            len(target_world.degrees_of_freedom) == 0):
        if _APARTMENT_WORLD is not None:
            target_world = _APARTMENT_WORLD
            self._world = target_world  # repair in place
    return target_world.get_degree_of_freedom_by_id(self.dof_id)

_AC1DOF.raw_dof = property(_robust_raw_dof)



# Patch add_subplan to ensure context propagates correctly after migration.
# execute_single() creates Plan(context=None); after _migrate_nodes_from_plan
# the nodes move into self.plan (which has context), but if any node's plan
# attribute still points to the old None-context plan, world/robot access fails.
from pycram.robot_plans.actions.base import ActionDescription as _ActionDescription
_original_add_subplan = _ActionDescription.add_subplan

def _patched_add_subplan(self, subplan_root):
    subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
    self.plan.add_edge(self.plan_node, subplan_root)
    # Ensure all migrated nodes have the correct plan reference with context
    for node in self.plan.all_nodes:
        if node.plan is not self.plan:
            node.plan = self.plan
    return subplan_root

_ActionDescription.add_subplan = _patched_add_subplan


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_OF_ITERATIONS:    int = 5000
MAX_RESAMPLE_ATTEMPTS:   int = 10   # max JPT resamples per iteration before hard failure

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
JPT_MIN_SAMPLES_PER_LEAF: int = 25


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
class ResamplingRecord:
    """One iteration that required JPT resampling before success."""
    iteration:  int
    attempts:   int    # total attempts including the one that succeeded
    elapsed_s:  float  # wall-clock seconds for the whole resampling loop


@dataclass
class RunStatistics:
    """Tracks success, failure, and JPT-resampling stats across all iterations."""
    successful_count:   int = 0
    failed_iterations:  int = 0   # iterations that ended in failure (hard failures)
    failed_attempts:    int = 0   # total individual attempt failures across all iterations
    hard_failure_count: int = 0   # iterations that exhausted all MAX_RESAMPLE_ATTEMPTS
    resampling_records: list = None

    def __post_init__(self):
        if self.resampling_records is None:
            self.resampling_records = []

    def record_resampling(self, iteration: int, attempts: int, elapsed_s: float):
        self.resampling_records.append(
            ResamplingRecord(iteration=iteration, attempts=attempts, elapsed_s=elapsed_s)
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
    # Parse the apartment world — it already has a root body, so
    # merge_world_at_pose can safely use self.root.
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
    # After merge_world_at_pose, pr2_world.clear() sets _world=None on all
    # semantic annotations that were in pr2_world. Fix them so that
    # robot.manipulator._world, robot.right_arm._world, etc. all point to world.
    _fix_robot_world_refs(robot, world)
    return world, robot


def _fix_robot_world_refs(robot: PR2, world: World) -> None:
    """
    After merge_world_at_pose clears pr2_world, all WorldEntity objects
    (connections, DOFs, bodies, semantic annotations) that were part of
    pr2_world still have _world pointing to the now-empty cleared world.
    Fix ALL connections in the apartment world that have a stale _world
    reference, plus all semantic annotations reachable from the robot.
    """
    from semantic_digital_twin.world_description.world_entity import (
        SemanticAnnotation, WorldEntity,
    )
    from dataclasses import fields as _fields, is_dataclass

    # Fix all connections and their DOFs in the apartment world
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

    # Also fix semantic annotations reachable from the robot
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

    def _coerce_numpy_scalar(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        # Coerce Python enums (e.g. DefaultWeights) to their underlying value
        import enum
        if isinstance(value, enum.Enum):       return value.value
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

    The plan graph patches (_patched_migrate_nodes_from_plan, _robust_raw_dof)
    resolve the rustworkx stale-index and stale-_world issues that previously
    forced single-waypoint navigation. Restoring multi-waypoint navigation
    gives RViz smooth step-by-step base movement matching the original demo.
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

    # Build one NavigateAction per waypoint (skip the start point at index 0).
    # Each waypoint is a collision-free intermediate position along the GCS path.
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
) -> PlanParameters:
    """
    Draw one joint sample from the JPT and map it to apartment PlanParameters.

    The JPT was trained with milk at y=0 and place target at x=OPEN_WORLD_PLACE_TARGET_X.
    The apartment milk is at y=MILK_SPAWN_Y and place target at x=PLACE_TARGET_X.
    Adding back the respective offsets recovers absolute apartment coordinates.
    """
    sample_row     = joint_probability_tree.sample(1)[0]
    sample_by_name = {variable.name: sample_row[index]
                      for index, variable in enumerate(JPT_VARIABLES)}

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
# Plan construction
# ---------------------------------------------------------------------------

def _actions_to_sequential_plan(
    planning_context: Context,
    actions: List[Any],
) -> SequentialNode:
    """
    Build a SequentialNode from an arbitrary-length action list.

    Uses the framework factory sequential(children, context) which:
      - creates a Plan(context) graph container
      - wraps each child via make_node() (ActionDescription → ActionNode, etc.)
      - calls mount_subplan() for nested language nodes
      - calls plan.simplify() to flatten redundant nesting

    This is the only correct construction path — building the graph manually
    does not correctly wire plan_node references needed by ActionDescription
    methods such as add_subplan(), which NavigateAction calls in execute().
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
    """Build a plan from JPT-sampled approach positions."""
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )
    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, gcs_bounds,
        start_x=robot_start_x,                     start_y=robot_start_y,
        goal_x=plan_parameters.counter_approach_x,  goal_y=plan_parameters.counter_approach_y,
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
    """Return the robot to the fixed start zone from its actual current position."""
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

    from pycram.plans.factories import sequential
    return_plan = sequential(return_actions, context=planning_context)
    with simulated_robot:
        try:
            return_plan.perform()
            print("  [return] Robot at start position.")
        except Exception as return_error:
            print(f"  [return] WARNING: return navigation failed: {return_error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_and_place_demo_apartment_jpt() -> None:
    """
    Apartment world: 5000 iterations of JPT-guided pick-and-place with GCS navigation.

    Iteration 1 uses fixed deterministic parameters to confirm the world, robot,
    and database are correctly initialised. Subsequent iterations sample jointly
    from the open-world JPT. On failure the next iteration simply draws a fresh
    sample from the JPT — no causal correction is applied.
    """
    print("=" * 64)
    print("  pick_and_place_demo_apartment_jpt  (JPT)")
    print(f"  Iterations    : {NUMBER_OF_ITERATIONS}")
    print(f"  Place target  : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model     : {JPT_MODEL_PATH}")
    print(f"  Database      : {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/5] Building apartment world ...")
    world, robot = _build_world(APARTMENT_URDF_PATH)
    global _APARTMENT_WORLD
    _APARTMENT_WORLD = world
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    gcs_bounds     = _build_gcs_bounds_array(navigation_map)

    print("\n[2/5] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/5] Loading JPT model ...")
    joint_probability_tree = _load_joint_probability_tree(JPT_MODEL_PATH)

    print("\n[4/5] Starting ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_apartment_jpt_node")
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

        print("\n[5/5] Running iterations ...")

        for iteration_number in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration_number}/{NUMBER_OF_ITERATIONS}  "
                f"(success={statistics.successful_count}  failed_iter={statistics.failed_iterations}  attempts={statistics.failed_attempts})"
            )
            print(f"{'=' * 64}")

            plan = None
            current_parameters = None

            if iteration_number == 1:
                print("  Mode: FIXED")
                try:
                    plan = _build_fixed_plan(
                        planning_context, world, robot, milk_body,
                        navigation_map, gcs_bounds,
                    )
                except ValueError as gcs_error:
                    statistics.failed_iterations += 1
                    plan = None
                    print(f"  RESULT: FAILED (GCS plan build) — {gcs_error}")

                if plan is not None:
                    print("\n  Executing plan ...")
                    with simulated_robot:
                        try:
                            plan.perform()
                            print("  Execution complete.")
                            statistics.successful_count += 1
                            print(
                                f"  RESULT: SUCCESS  "
                                f"({statistics.successful_count}/{iteration_number} stored,  "
                                f"{NUMBER_OF_ITERATIONS - iteration_number} remaining)"
                            )
                            try:
                                _persist_plan(database_session, plan)
                            except Exception as database_error:
                                print(f"  [db] ERROR: {database_error}")
                                traceback.print_exc()
                                database_session.rollback()
                        except Exception as execution_error:
                            statistics.failed_iterations += 1
                            print(
                                f"  RESULT: FAILED — "
                                f"{type(execution_error).__name__}: {execution_error}"
                            )
            else:
                print("  Mode: JPT-SAMPLED  (with resampling on failure)")
                # ── JPT resampling loop ────────────────────────────────────
                # If execution fails we resample ALL plan parameters jointly
                # from the JPT and retry immediately, without resetting the
                # world between attempts (the milk is already in its spawn
                # pose and the robot is already at the start).
                attempt          = 0
                execution_succeeded = False
                resample_start   = time.time()

                while not execution_succeeded and attempt < MAX_RESAMPLE_ATTEMPTS:
                    attempt += 1
                    current_parameters = _sample_plan_parameters(joint_probability_tree)

                    try:
                        plan = _build_sampled_plan(
                            planning_context, current_parameters, world, robot, milk_body,
                            navigation_map, gcs_bounds,
                            robot_start_x=robot_x, robot_start_y=robot_y,
                        )
                    except ValueError as gcs_error:
                        print(
                            f"  [attempt {attempt}] FAILED (GCS plan build) — {gcs_error}"
                            f"  → resampling ..."
                        )
                        statistics.failed_attempts += 1
                        continue   # resample immediately

                    print(f"\n  [attempt {attempt}] Executing plan ...")
                    with simulated_robot:
                        try:
                            plan.perform()
                            execution_succeeded = True
                            elapsed = time.time() - resample_start
                            print(
                                f"  [attempt {attempt}] Execution complete. "
                                f"({elapsed:.1f}s elapsed)"
                            )
                        except Exception as execution_error:
                            statistics.failed_attempts += 1
                            print(
                                f"  [attempt {attempt}] FAILED — "
                                f"{type(execution_error).__name__}: {execution_error}"
                                f"  → resampling ..."
                            )
                            # Reset milk before the next attempt so each sample
                            # starts from the canonical world state.
                            _respawn_milk(world, milk_body)
                            _navigate_back_to_start(
                                planning_context, navigation_map, gcs_bounds, world,
                                robot_x=current_parameters.table_approach_x,
                                robot_y=current_parameters.table_approach_y,
                            )

                elapsed = time.time() - resample_start
                if not execution_succeeded:
                    # All MAX_RESAMPLE_ATTEMPTS exhausted without success
                    statistics.failed_iterations += 1
                    statistics.hard_failure_count += 1
                    print(
                        f"  RESULT: HARD FAILURE — gave up after "
                        f"{MAX_RESAMPLE_ATTEMPTS} attempts ({elapsed:.1f}s)"
                    )
                elif attempt > 1:
                    print(
                        f"  [resampling] Succeeded after {attempt} attempt(s) "
                        f"in {elapsed:.1f}s"
                    )
                    statistics.record_resampling(iteration_number, attempt, elapsed)

                if execution_succeeded:
                    statistics.successful_count += 1
                    print(
                        f"  RESULT: SUCCESS  "
                        f"({statistics.successful_count}/{iteration_number} stored,  "
                        f"{NUMBER_OF_ITERATIONS - iteration_number} remaining)"
                    )
                    try:
                        _persist_plan(database_session, plan)
                    except Exception as database_error:
                        print(f"  [db] ERROR: {database_error}")
                        traceback.print_exc()
                        database_session.rollback()

            # For JPT-sampled iterations the resampling loop handles its own
            # milk respawning and return navigation on each failed attempt.
            # After a successful execution (fixed or sampled) we still need one
            # final respawn + return from the table position.
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

        # ── Final summary ──────────────────────────────────────────────────
        success_rate = 100 * statistics.successful_count // NUMBER_OF_ITERATIONS
        recs = statistics.resampling_records
        print(f"\n{'=' * 64}")
        print(f"  Run complete.")
        print(f"  Total iterations     : {NUMBER_OF_ITERATIONS}")
        print(f"  Successful plans     : {statistics.successful_count}  ({success_rate}%)")
        print(f"  Failed iterations    : {statistics.failed_iterations}")
        print(f"  Total failed attempts: {statistics.failed_attempts}")
        print(f"  Hard failures (>{MAX_RESAMPLE_ATTEMPTS} attempts) : {statistics.hard_failure_count}")
        print(f"  Database             : {DATABASE_URI}")

        if recs:
            avg_attempts = sum(r.attempts for r in recs) / len(recs)
            avg_time     = sum(r.elapsed_s for r in recs) / len(recs)
            max_attempts = max(r.attempts for r in recs)
            max_time     = max(r.elapsed_s for r in recs)
            print(f"")
            print(f"  ── JPT Resampling Stats (iterations that needed >1 attempt) ──")
            print(f"  Iterations requiring resampling : {len(recs)}")
            print(f"  Avg attempts until success      : {avg_attempts:.2f}")
            print(f"  Avg time until success          : {avg_time:.1f}s")
            print(f"  Max attempts in one iteration   : {max_attempts}")
            print(f"  Max time in one iteration       : {max_time:.1f}s")
            print(f"")
            print(f"  Per-iteration resampling detail:")
            for r in recs:
                print(
                    f"    iter {r.iteration:>4d}:  "
                    f"{r.attempts} attempt(s),  {r.elapsed_s:.1f}s"
                )
        else:
            print(f"  (No iterations required resampling — first attempt always succeeded)")
        print(f"{'=' * 64}")

        try:
            row_count = database_session.execute(
                text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
            ).scalar()
            print(f"  DB rows (SequentialPlanDAO): {row_count}")
        except Exception as count_error:
            print(f"  [db] Could not read row count: {count_error}")

        # Expose statistics on the module for the runner to read
        import sys as _sys
        _sys.modules[__name__]._last_statistics = statistics

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo_apartment_jpt()