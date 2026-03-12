"""
pick_and_place_demo_apartment_jpt.py
=====================================
Batch 2 for the apartment pick-and-place world.

Uses the open-world JPT (pick_and_place_jpt.json, trained on 1742 successful
Batch 1 open-world plans) to guide approach-position and arm sampling in the
apartment world.

The apartment Batch 1 produced 0% success with uniform sampling, so there is
no apartment-specific training data.  The open-world JPT is valid here because
the pick/place mechanics — approach position, arm choice, grasp — are identical
between the two worlds.  Only the navigation geometry differs (GCS in the
apartment vs direct navigation in the open world), and GCS handles that
transparently after sampling.

Variable mapping between the two worlds:
    open-world JPT variable     apartment demo name
    ─────────────────────────────────────────────────
    pick_approach_x/y       →   counter_approach_x/y
    place_approach_x/y      →   table_approach_x/y
    milk_end_x/y/z          →   milk_end_x/y/z  (unchanged)
    pick_arm                →   pick_arm         (unchanged)

PLACE_X, PLACE_Y, PLACE_Z are fixed constants identical to the open-world
demo — the same table surface height applies.
"""

from __future__ import annotations

import copy
import datetime
import inspect
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
import rclpy
import sqlalchemy.types as sqlalchemy_types
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import Header, PoseStamped, PyCramPose, PyCramQuaternion, PyCramVector3
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base
from pycram.robot_plans import NavigateAction, ParkArmsAction, PickUpAction, PlaceAction

from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine

from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.distributions.univariate import Multinomial
from jpt.variables import NumericVariable, SymbolicVariable
from jpt.trees import JPT as JointProbabilityTree

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import FileMesh, BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import GraphOfConvexSets
from semantic_digital_twin.world_description.shape_collection import ShapeCollection, BoundingBoxCollection
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_OF_ITERATIONS: int = 5000

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/demo_robot_plans",
)

MILK_X: float = 2.4
MILK_Y: float = 2.5
MILK_Z: float = 1.01

# Fixed seed approach positions used on iteration 1.
# These match the known-good values from the original apartment demo.
COUNTER_APPROACH_X: float = 1.6
COUNTER_APPROACH_Y: float = 2.5
TABLE_APPROACH_X:   float = 4.2
TABLE_APPROACH_Y:   float = 4.0

# Place target on the apartment table surface — fixed, never sampled.
# Must match the table geometry in the apartment URDF.
PLACE_X: float = 5.0
PLACE_Y: float = 4.0
PLACE_Z: float = 0.80

# Safety clip bounds applied after JPT sampling.
# These match the apartment approach zones from the original demo
# (_COUNTER_APPROACH_BOUNDS and _TABLE_APPROACH_BOUNDS) so that the
# robot always navigates to a reachable position inside the apartment.
COUNTER_APPROACH_SAMPLING_BOUNDS: tuple[float, float, float, float] = (1.2, 1.8, 2.3, 2.7)
TABLE_APPROACH_SAMPLING_BOUNDS:   tuple[float, float, float, float] = (4.1, 4.5, 3.95, 4.05)

GRAPH_OF_CONVEX_SETS_MIN_X: float = -1.0
GRAPH_OF_CONVEX_SETS_MAX_X: float = 7.0
GRAPH_OF_CONVEX_SETS_MIN_Y: float = -1.0
GRAPH_OF_CONVEX_SETS_MAX_Y: float = 7.0
GRAPH_OF_CONVEX_SETS_MIN_Z: float = 0.0
GRAPH_OF_CONVEX_SETS_MAX_Z: float = 0.1
GRAPH_OF_CONVEX_SETS_BLOAT_OBSTACLES: float = 0.3

_RESOURCE_PATH = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL:       Path = _RESOURCE_PATH / "objects" / "milk.stl"

# Reuse the open-world JPT directly — no new training data required.
JPT_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")
JPT_MIN_SAMPLES_PER_LEAF: int = 25


# ---------------------------------------------------------------------------
# JPT variable definitions
# Must match fit_jpt.py (open-world) exactly — same variable names as the
# JSON model file.  The mapping to apartment terminology is done in
# _sample_plan_parameters_from_jpt when building PlanParameters.
# ---------------------------------------------------------------------------

ArmChoiceDomain = type(
    "ArmChoiceDomain",
    (Multinomial,),
    {
        "values": OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)]),
        "labels": OrderedDictProxy([(0, "LEFT"), (1, "RIGHT")]),
    },
)

JPT_VARIABLES: list = [
    NumericVariable("pick_approach_x",  precision=0.005),  # → counter_approach_x
    NumericVariable("pick_approach_y",  precision=0.005),  # → counter_approach_y
    NumericVariable("place_approach_x", precision=0.005),  # → table_approach_x
    NumericVariable("place_approach_y", precision=0.005),  # → table_approach_y
    NumericVariable("milk_end_x",       precision=0.001),
    NumericVariable("milk_end_y",       precision=0.001),
    NumericVariable("milk_end_z",       precision=0.0005),
    SymbolicVariable("pick_arm",        domain=ArmChoiceDomain),
]


# ---------------------------------------------------------------------------
# Plan parameters
# ---------------------------------------------------------------------------

@dataclass
class PlanParameters:
    """
    One complete set of plan parameters sampled from the open-world JPT,
    expressed in apartment terminology.

    counter_approach_x/y : robot navigation goal before picking up milk
                           (maps from JPT pick_approach_x/y).
    table_approach_x/y   : robot navigation goal before placing milk
                           (maps from JPT place_approach_x/y).
    pick_arm             : arm used to pick and place the milk.
    """
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------

def _header_deepcopy(self, memo: Any) -> Header:
    if isinstance(self, type):
        return self
    timestamp = getattr(self, "stamp", None) or datetime.datetime.now()
    return Header(
        frame_id=getattr(self, "frame_id", None),
        stamp=copy.deepcopy(timestamp, memo),
        sequence=getattr(self, "sequence", 0),
    )


def _pose_stamped_deepcopy(self, memo: Any) -> PoseStamped:
    if isinstance(self, type):
        return self
    return PoseStamped(
        copy.deepcopy(getattr(self, "pose", None), memo),
        copy.deepcopy(getattr(self, "header", None), memo),
    )


def _header_getattr(self, attribute_name: str) -> Any:
    defaults = {
        "stamp":    lambda: datetime.datetime.now(),
        "sequence": lambda: 0,
        "frame_id": lambda: None,
    }
    if attribute_name in defaults:
        value = defaults[attribute_name]()
        object.__setattr__(self, attribute_name, value)
        return value
    raise AttributeError(attribute_name)


Header.__deepcopy__      = _header_deepcopy
Header.__getattr__       = _header_getattr
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy


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
        print("  [patch] WARNING: ORM numpy TypeDecorator not found; None-guard patch skipped.")
        return
    original = target_class.process_bind_param
    def _none_guarded(self, value, dialect):
        if value is None:
            return None
        return original(self, value, dialect)
    target_class.process_bind_param = _none_guarded
    print(f"  [patch] Patched {target_class.__name__}.process_bind_param to handle None.")

_patch_orm_numpy_array_type()


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def _initialize_world(apartment_urdf: Path) -> tuple[World, PR2]:
    world = URDFParser.from_file(str(apartment_urdf)).parse()
    pr2_world = URDFParser.from_file(
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    robot = PR2.from_world(pr2_world)
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 0.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(pr2_world, robot_pose)
    table_body = world.get_body_by_name("table_area_main")
    with world.modify_world():
        world.add_semantic_annotation(Table(root=table_body))
    return world, robot


def _add_localization_frames(world: World, robot: PR2) -> None:
    with world.modify_world():
        map_body  = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_body(map_body)
        world.add_body(odom_body)
        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))
        existing = robot.root.parent_connection
        if existing is not None:
            world.remove_connection(existing)
        world.add_connection(
            OmniDrive.create_with_dofs(
                parent=odom_body,
                child=robot.root,
                world=world,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.4, 1.5, 0.0, 0, 0, 0
                ),
            )
        )


def _add_milk_object(world: World, stl_path: Path) -> Body:
    mesh = FileMesh.from_file(str(stl_path))
    body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(MILK_X, MILK_Y, MILK_Z, 0, 0, 0)
    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=body, world=world
        )
        world.add_connection(connection)
        connection.origin = pose
        world.add_semantic_annotation(Milk(root=body))
    return body


def _respawn_milk_object(world: World, milk_body: Body) -> None:
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_X, MILK_Y, MILK_Z, 0, 0, 0
    )
    with world.modify_world():
        connection = milk_body.parent_connection
        if connection is not None:
            if connection.parent is not world.root:
                world.remove_connection(connection)
                new_connection = Connection6DoF.create_with_dofs(
                    parent=world.root, child=milk_body, world=world
                )
                world.add_connection(new_connection)
                new_connection.origin = spawn_pose
            else:
                connection.origin = spawn_pose
        else:
            new_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(new_connection)
            new_connection.origin = spawn_pose
    print(f"  [respawn] Milk reset to ({MILK_X}, {MILK_Y}, {MILK_Z})")


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
    import hashlib
    def shorten(name: str, limit: int = 63) -> str:
        if len(name) <= limit:
            return name
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:limit - 9]}_{digest}"
    for table in Base.metadata.tables.values():
        short = shorten(table.name)
        if short != table.name:
            table.name     = short
            table.fullname = short


def _register_postgresql_numpy_scalar_coercion(engine: Any) -> None:
    import numpy
    from sqlalchemy import event

    def _coerce(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        return value

    def _coerce_params(params: Any) -> Any:
        if isinstance(params, dict):
            return {k: _coerce(v) for k, v in params.items()}
        return params

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _before_execute(connection, cursor, statement, parameters, context, executemany):
        if isinstance(parameters, dict):
            parameters = _coerce_params(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(_coerce_params(p) for p in parameters)
        return statement, parameters


def _persist_plan_to_database(session: Session, plan: SequentialPlan) -> None:
    print("  [db] Persisting plan ...")
    session.add(to_dao(plan))
    session.commit()
    print("  [db] Plan committed OK.")


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _build_navigation_map(world: World) -> GraphOfConvexSets:
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=GRAPH_OF_CONVEX_SETS_MIN_X,
                max_x=GRAPH_OF_CONVEX_SETS_MAX_X,
                min_y=GRAPH_OF_CONVEX_SETS_MIN_Y,
                max_y=GRAPH_OF_CONVEX_SETS_MAX_Y,
                min_z=GRAPH_OF_CONVEX_SETS_MIN_Z,
                max_z=GRAPH_OF_CONVEX_SETS_MAX_Z,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        world.root,
    )
    print("  Building GraphOfConvexSets navigation map ...")
    start_time = time.time()
    gcs = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=GRAPH_OF_CONVEX_SETS_BLOAT_OBSTACLES,
    )
    elapsed = time.time() - start_time
    n_nodes = len(list(gcs.graph.nodes()))
    print(f"  GraphOfConvexSets built in {elapsed:.2f} s  ({n_nodes} nodes)")
    return gcs


def _build_gcs_index(gcs: GraphOfConvexSets) -> np.ndarray:
    """Build the node bbox array immediately after GCS construction."""
    arr = _build_gcs_node_array(gcs)
    print(f"  GCS node index built: {len(arr)} bounding boxes")
    return arr


def _build_gcs_node_array(gcs: GraphOfConvexSets) -> np.ndarray:
    """
    Build a (N, 6) numpy array of [min_x, min_y, min_z, max_x, max_y, max_z]
    for all GCS nodes. Called once after GCS is constructed so that
    _point_in_free_space can use fast vectorised bbox checks instead of
    the O(N) Python loop inside gcs.node_of_point.
    """
    nodes = list(gcs.graph.nodes())
    arr = np.array(
        [[n.min_x, n.min_y, n.min_z, n.max_x, n.max_y, n.max_z] for n in nodes],
        dtype=np.float64,
    )
    return arr


def _point_in_free_space(node_array: np.ndarray, x: float, y: float, z: float) -> bool:
    """
    Return True if (x, y, z) is inside at least one GCS bounding box.
    Uses vectorised numpy comparisons — O(N) but fully in C, ~100x faster
    than the Python loop in gcs.node_of_point for large node counts.
    """
    inside = (
        (node_array[:, 0] <= x) & (x <= node_array[:, 3]) &
        (node_array[:, 1] <= y) & (y <= node_array[:, 4]) &
        (node_array[:, 2] <= z) & (z <= node_array[:, 5])
    )
    return bool(inside.any())


def _find_nearest_free_point(
    gcs: GraphOfConvexSets,
    node_array: np.ndarray,
    x: float,
    y: float,
    z: float,
    world: World,
    search_radius: float = 0.6,
    step: float = 0.05,
    steps: int = 16,
) -> Optional[Point3]:
    """
    If (x, y, z) is inside an obstacle, spiral outward to find the nearest
    free point. Uses the pre-built node_array for fast vectorised checks.
    """
    if _point_in_free_space(node_array, x, y, z):
        return Point3(x, y, z, reference_frame=world.root)

    print(f"    [GCS] ({x:.3f},{y:.3f}) occupied — finding nearest free point ...")
    for radius in np.arange(step, search_radius + step, step):
        angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        cxs = x + radius * np.cos(angles)
        cys = y + radius * np.sin(angles)
        for cx, cy in zip(cxs, cys):
            if _point_in_free_space(node_array, cx, cy, z):
                print(f"    [GCS] free point at ({cx:.3f},{cy:.3f}) r={radius:.2f}")
                return Point3(cx, cy, z, reference_frame=world.root)
    print(f"    [GCS] no free point within r={search_radius}")
    return None


def _gcs_collision_free_path(
    gcs: GraphOfConvexSets,
    node_array: np.ndarray,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    world: World,
    snap_to_free: bool = False,
) -> Optional[List[Point3]]:
    z_nav = (GRAPH_OF_CONVEX_SETS_MIN_Z + GRAPH_OF_CONVEX_SETS_MAX_Z) / 2.0

    if snap_to_free:
        start = _find_nearest_free_point(gcs, node_array, start_x, start_y, z_nav, world)
        goal  = _find_nearest_free_point(gcs, node_array, goal_x,  goal_y,  z_nav, world)
        if start is None:
            print(f"    [GCS] start ({start_x:.3f},{start_y:.3f}) has no free neighbour")
            return None
        if goal is None:
            print(f"    [GCS] goal ({goal_x:.3f},{goal_y:.3f}) has no free neighbour")
            return None
    else:
        start = Point3(start_x, start_y, z_nav, reference_frame=world.root)
        goal  = Point3(goal_x,  goal_y,  z_nav, reference_frame=world.root)

    try:
        path = gcs.path_from_to(start, goal)
        print(f"    [GCS] path found: {len(path)} points")
        return path
    except Exception as exc:
        print(f"    [GCS] path_from_to raised: {exc}")
        return None


def _gcs_path_to_pose_stamped_list(
    path: List[Point3],
    world_frame: Any,
) -> List[PoseStamped]:
    poses = []
    for point in path[1:]:
        poses.append(
            PoseStamped(
                pose=PyCramPose(
                    position=PyCramVector3(x=float(point.x), y=float(point.y), z=0.0),
                    orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
                ),
                header=Header(frame_id=world_frame),
            )
        )
    return poses


def _create_pose_stamped(x: float, y: float, z: float, frame_id: Any) -> PoseStamped:
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame_id, stamp=datetime.datetime.now(), sequence=0),
    )


def _navigate_via_gcs(
    context: Context,
    gcs: GraphOfConvexSets,
    node_array: np.ndarray,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    world: World,
    keep_joint_states: bool = False,
    snap_to_free: bool = False,
) -> List[NavigateAction]:
    """
    Produce NavigateActions following a GCS collision-free path.
    Falls back to direct navigation if no path is found.
    snap_to_free=True only for the return trip.
    node_array is the pre-built numpy bbox array for fast free-space checks.
    """
    path = _gcs_collision_free_path(gcs, node_array, start_x, start_y, goal_x, goal_y, world, snap_to_free=snap_to_free)
    if path is None or len(path) < 2:
        print(
            f"    [GCS] No path ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}). "
            f"Falling back to direct navigation."
        )
        return [
            NavigateAction(
                target_location=_create_pose_stamped(goal_x, goal_y, 0.0, world.root),
                keep_joint_states=keep_joint_states,
            )
        ]
    waypoints = _gcs_path_to_pose_stamped_list(path, world.root)
    print(
        f"    [GCS] Path ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): "
        f"{len(waypoints)} waypoint(s)"
    )
    for i, wp in enumerate(waypoints):
        pos = wp.pose.position
        print(f"           waypoint {i+1}: ({pos.x:.3f}, {pos.y:.3f})")
    return [
        NavigateAction(target_location=wp, keep_joint_states=keep_joint_states)
        for wp in waypoints
    ]



def _get_robot_current_position(
    robot: PR2,
    world: World,
    fallback_x: float = 1.4,
    fallback_y: float = 1.5,
) -> tuple[float, float]:
    """
    Read the robot base position by reading the OmniDrive connection origin.

    _add_localization_frames creates an OmniDrive connecting odom_body -> robot.root.
    NavigateAction updates this connection's origin as the robot moves, exactly the
    same way _respawn_milk_object updates the milk connection's origin.

    A HomogeneousTransformationMatrix is a 4x4 matrix; to_np()[0,3] = x, [1,3] = y.
    """
    try:
        conn = robot.root.parent_connection
        if conn is not None and hasattr(conn, "origin") and conn.origin is not None:
            mat = conn.origin.to_np()          # 4x4 numpy array
            x, y = float(mat[0, 3]), float(mat[1, 3])
            print(f"  [robot_pos] OmniDrive origin.to_np() -> ({x:.3f}, {y:.3f})")
            return x, y
    except Exception as e:
        print(f"  [robot_pos] OmniDrive origin read failed: {e}")

    print(f"  [robot_pos] using fallback ({fallback_x}, {fallback_y})")
    return fallback_x, fallback_y


# ---------------------------------------------------------------------------
# JPT loading and sampling
# ---------------------------------------------------------------------------

def _load_joint_probability_tree(model_path: str) -> JointProbabilityTree:
    """
    Load the open-world JPT from disk.

    Variable names in the JSON (pick_approach_x/y, place_approach_x/y) are
    mapped to apartment names (counter_approach_x/y, table_approach_x/y)
    inside _sample_plan_parameters_from_jpt.
    """
    print(f"  [jpt] Loading JPT from {model_path} ...")
    joint_probability_tree = JointProbabilityTree(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    )
    joint_probability_tree = joint_probability_tree.load(model_path)
    print(f"  [jpt] JPT loaded.  Leaves: {len(joint_probability_tree.leaves)}")
    return joint_probability_tree


def _sample_plan_parameters_from_jpt(
    joint_probability_tree: JointProbabilityTree,
) -> PlanParameters:
    """
    Draw one joint sample from the open-world JPT and return it as
    apartment PlanParameters.

    Variable mapping:
        pick_approach_x/y  →  counter_approach_x/y
        place_approach_x/y →  table_approach_x/y

    Safety clipping uses the apartment approach zone bounds.  The open-world
    JPT was trained with Y in (-0.4, 0.4), so Y samples will frequently be
    clipped to the apartment zones (2.3–2.7 for counter, 3.8–4.2 for table).
    This is intentional — it constrains the robot to reachable apartment positions.
    """
    sample_array   = joint_probability_tree.sample(1)
    sample_row     = sample_array[0]
    variable_names = [v.name for v in JPT_VARIABLES]
    sample_by_name = dict(zip(variable_names, sample_row))

    arm_label = sample_by_name["pick_arm"]
    if isinstance(arm_label, (int, float)):
        arm_label = ArmChoiceDomain.labels[int(arm_label)]
    pick_arm = Arms.LEFT if arm_label == "LEFT" else Arms.RIGHT

    cx_min, cx_max, cy_min, cy_max = COUNTER_APPROACH_SAMPLING_BOUNDS
    tx_min, tx_max, ty_min, ty_max = TABLE_APPROACH_SAMPLING_BOUNDS

    return PlanParameters(
        counter_approach_x = float(max(cx_min, min(cx_max, sample_by_name["pick_approach_x"]))),
        counter_approach_y = float(max(cy_min, min(cy_max, sample_by_name["pick_approach_y"]))),
        table_approach_x   = float(max(tx_min, min(tx_max, sample_by_name["place_approach_x"]))),
        table_approach_y   = float(max(ty_min, min(ty_max, sample_by_name["place_approach_y"]))),
        pick_arm           = pick_arm,
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_fixed_plan(
    context:       Context,
    world:         World,
    robot:         PR2,
    milk_body:     Body,
    gcs:           GraphOfConvexSets,
    gcs_node_array: np.ndarray,
    robot_start_x: float = 1.4,
    robot_start_y: float = 1.5,
) -> SequentialPlan:
    """Build the deterministic seed plan used on iteration 1."""
    seed_arm = Arms.RIGHT

    nav_to_counter = _navigate_via_gcs(
        context, gcs, gcs_node_array,
        start_x=robot_start_x, start_y=robot_start_y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    nav_to_table = _navigate_via_gcs(
        context, gcs, gcs_node_array,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X, goal_y=TABLE_APPROACH_Y,
        world=world,
        keep_joint_states=True,
    )
    place_target_pose = _create_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world.root)

    print(f"  [plan] Fixed seed parameters:")
    print(f"         counter approach : ({COUNTER_APPROACH_X}, {COUNTER_APPROACH_Y})")
    print(f"         table   approach : ({TABLE_APPROACH_X}, {TABLE_APPROACH_Y})")
    print(f"         place   target   : ({PLACE_X}, {PLACE_Y}, {PLACE_Z})  [fixed]")
    print(f"         arm              : {seed_arm}")

    actions = (
        [ParkArmsAction(arm=Arms.BOTH)]
        + nav_to_counter
        + [
            PickUpAction(
                object_designator=milk_body,
                arm=seed_arm,
                grasp_description=GraspDescription(
                    approach_direction=ApproachDirection.FRONT,
                    vertical_alignment=VerticalAlignment.NoAlignment,
                    rotate_gripper=False,
                    manipulation_offset=0.06,
                    manipulator=robot.right_arm.manipulator,
                ),
            )
        ]
        + nav_to_table
        + [
            PlaceAction(
                object_designator=milk_body,
                target_location=place_target_pose,
                arm=seed_arm,
            ),
            ParkArmsAction(arm=Arms.BOTH),
        ]
    )
    return SequentialPlan(context, *actions)


def _build_jpt_sampled_plan(
    context:         Context,
    plan_parameters: PlanParameters,
    world:           World,
    robot:           PR2,
    milk_body:       Body,
    gcs:             GraphOfConvexSets,
    gcs_node_array:  np.ndarray,
    robot_start_x:   float,
    robot_start_y:   float,
) -> SequentialPlan:
    """
    Build a plan from JPT-sampled approach positions.

    Place target is always fixed to PLACE_X/Y/Z.
    GCS handles collision-free navigation regardless of sampled positions.
    """
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )

    nav_to_counter = _navigate_via_gcs(
        context, gcs, gcs_node_array,
        start_x=robot_start_x,
        start_y=robot_start_y,
        goal_x=plan_parameters.counter_approach_x,
        goal_y=plan_parameters.counter_approach_y,
        world=world,
    )
    nav_to_table = _navigate_via_gcs(
        context, gcs, gcs_node_array,
        start_x=plan_parameters.counter_approach_x,
        start_y=plan_parameters.counter_approach_y,
        goal_x=plan_parameters.table_approach_x,
        goal_y=plan_parameters.table_approach_y,
        world=world,
        keep_joint_states=True,
    )
    # Place Y is fixed to PLACE_Y (= TABLE_APPROACH_Y = 4.0), matching the
    # fixed plan exactly.  The fixed plan works with RIGHT arm because the
    # robot navigates to TABLE_APPROACH_Y=4.0 and places at PLACE_Y=4.0 —
    # the target is directly ahead.  Sampled iterations must reproduce the
    # same geometry, so we clip table_approach_y to TABLE_APPROACH_Y via the
    # sampling bounds and use PLACE_Y here.  DO NOT use plan_parameters.table_approach_y
    # as the place Y — even small Y deviations rotate the target sideways out
    # of the right arm's reachable workspace.
    place_target_pose = _create_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world.root)

    print(f"  [plan] JPT-sampled parameters:")
    print(f"         counter approach : ({plan_parameters.counter_approach_x:.3f}, {plan_parameters.counter_approach_y:.3f})")
    print(f"         table   approach : ({plan_parameters.table_approach_x:.3f}, {plan_parameters.table_approach_y:.3f})")
    print(f"         place   target   : ({PLACE_X}, {PLACE_Y}, {PLACE_Z})  [fixed — matches fixed plan geometry]")
    print(f"         arm              : {plan_parameters.pick_arm}")

    actions = (
        [ParkArmsAction(arm=Arms.BOTH)]
        + nav_to_counter
        + [
            PickUpAction(
                object_designator=milk_body,
                arm=plan_parameters.pick_arm,
                grasp_description=GraspDescription(
                    approach_direction=ApproachDirection.FRONT,
                    vertical_alignment=VerticalAlignment.NoAlignment,
                    rotate_gripper=False,
                    manipulation_offset=0.06,
                    manipulator=manipulator,
                ),
            )
        ]
        + nav_to_table
        + [
            PlaceAction(
                object_designator=milk_body,
                target_location=place_target_pose,
                arm=plan_parameters.pick_arm,
            ),
            ParkArmsAction(arm=Arms.BOTH),
        ]
    )
    return SequentialPlan(context, *actions)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_and_place_demo_apartment_jpt() -> None:
    """
    Apartment world Batch 2: JPT-guided sampling with GCS navigation.

    Reuses the open-world JPT (pick_and_place_jpt.json) because the apartment
    Batch 1 produced 0% success — no apartment-specific training data exists.
    The open-world JPT is valid because pick/place mechanics are identical;
    GCS handles the apartment geometry transparently.
    """
    print("=" * 64)
    print("  pick_and_place_demo_apartment_jpt  (Batch 2)")
    print(f"  NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}")
    print(f"  PLACE_Z (fixed)      = {PLACE_Z}")
    print(f"  DATABASE_URI         = {DATABASE_URI}")
    print(f"  JPT_MODEL_PATH       = {JPT_MODEL_PATH}")
    print("=" * 64)

    print("\n[1/5] Building world ...")
    world, robot = _initialize_world(APARTMENT_URDF)
    _add_localization_frames(world, robot)
    milk_body = _add_milk_object(world, MILK_STL)
    print(f"  Milk spawned at  x={MILK_X:.3f},  y={MILK_Y:.3f},  z={MILK_Z:.3f}")
    print(f"  Place target at  x={PLACE_X:.3f}, y={PLACE_Y:.3f}, z={PLACE_Z:.3f}  [fixed]")

    gcs = _build_navigation_map(world)
    gcs_node_array = _build_gcs_index(gcs)

    print("\n[2/5] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/5] Loading JPT model ...")
    joint_probability_tree = _load_joint_probability_tree(JPT_MODEL_PATH)

    print("\n[4/5] Initialising ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_apartment_jpt_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()
    print("  [ros] Node started.")

    robot_init_x: float = 1.4
    robot_init_y: float = 1.5

    # Snap robot_init to free space once — (1.4, 1.5) may be inside a
    # bloated obstacle region in GCS. All return trips navigate to this
    # snapped position so GCS can always route to it.
    z_nav = (GRAPH_OF_CONVEX_SETS_MIN_Z + GRAPH_OF_CONVEX_SETS_MAX_Z) / 2.0
    _snapped = _find_nearest_free_point(
        gcs, gcs_node_array, robot_init_x, robot_init_y, z_nav, world
    )
    if _snapped is not None:
        robot_return_x = float(_snapped.x)
        robot_return_y = float(_snapped.y)
        print(f"  [init] return goal snapped to free space: ({robot_return_x:.3f}, {robot_return_y:.3f})")
    else:
        robot_return_x = robot_init_x
        robot_return_y = robot_init_y
        print(f"  [init] WARNING: could not snap return goal — using ({robot_return_x}, {robot_return_y})")


    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context = Context(world, robot, None)

        print("\n[5/5] Running iterations ...")
        successful_count = 0
        failed_count     = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration} / {NUMBER_OF_ITERATIONS}  "
                f"(success={successful_count}  failed={failed_count})"
            )
            print(f"{'=' * 64}")

            if iteration == 1:
                print("  Mode: FIXED (deterministic seed)")
                plan = _build_fixed_plan(
                    planning_context, world, robot, milk_body, gcs, gcs_node_array,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )
            else:
                print("  Mode: JPT-SAMPLED")
                plan_parameters = _sample_plan_parameters_from_jpt(joint_probability_tree)
                plan = _build_jpt_sampled_plan(
                    planning_context, plan_parameters, world, robot, milk_body, gcs, gcs_node_array,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )

            print("\n  Executing plan ...")
            execution_succeeded = False
            with simulated_robot:
                try:
                    plan.perform()
                    execution_succeeded = True
                    print("  plan.perform() completed without exception.")

                except Exception as execution_error:
                    failed_count += 1
                    print(f"\n  RESULT: FAILED  (iteration {iteration})")
                    print(f"  Exception type : {type(execution_error).__name__}")
                    print(f"  Exception msg  : {execution_error}")
                    print("  Traceback (last 3 lines):")
                    for line in traceback.format_exc().strip().splitlines()[-3:]:
                        print(f"    {line}")

            if execution_succeeded:
                try:
                    _persist_plan_to_database(database_session, plan)
                    successful_count += 1
                    print(
                        f"  RESULT: SUCCESS  "
                        f"({successful_count} stored / "
                        f"{iteration} attempted / "
                        f"{NUMBER_OF_ITERATIONS - iteration} remaining)"
                    )
                except Exception as db_error:
                    print(f"  [db] ERROR persisting plan: {db_error}")
                    traceback.print_exc()

            print("\n  Resetting world state ...")
            _respawn_milk_object(world, milk_body)

            # Return robot via the same waypoints as the forward path, reversed:
            #   table_approach -> counter_approach -> start
            # Both intermediate positions are known-free GCS nodes (used in
            # forward navigation). No snapping needed.
            if iteration == 1:
                table_x,   table_y   = TABLE_APPROACH_X,   TABLE_APPROACH_Y
                counter_x, counter_y = COUNTER_APPROACH_X, COUNTER_APPROACH_Y
            else:
                table_x,   table_y   = plan_parameters.table_approach_x,   plan_parameters.table_approach_y
                counter_x, counter_y = plan_parameters.counter_approach_x, plan_parameters.counter_approach_y

            print(f"  [reset] return: table({table_x:.2f},{table_y:.2f}) -> counter({counter_x:.2f},{counter_y:.2f}) -> start({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})")

            leg1 = _navigate_via_gcs(
                planning_context, gcs, gcs_node_array,
                start_x=table_x,   start_y=table_y,
                goal_x=counter_x,  goal_y=counter_y,
                world=world,
            )
            leg2 = _navigate_via_gcs(
                planning_context, gcs, gcs_node_array,
                start_x=counter_x,        start_y=counter_y,
                goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
                world=world,
            )
            return_plan = SequentialPlan(planning_context, *(leg1 + leg2))
            with simulated_robot:
                try:
                    return_plan.perform()
                    print("  [reset] Robot returned to start position.")
                except Exception as return_error:
                    print(f"  [reset] WARNING: return navigation failed: {return_error}")


        batch_two_rate = 100 * successful_count // NUMBER_OF_ITERATIONS
        print(f"\n{'=' * 64}")
        print(f"  Run complete.")
        print(f"  Total iterations         : {NUMBER_OF_ITERATIONS}")
        print(f"  Successful plans         : {successful_count}  ({batch_two_rate}%)")
        print(f"  Failed                   : {failed_count}")
        print(f"  Open-world Batch 1       : 1742 / 5000 = 34%")
        print(f"  Open-world Batch 2 (JPT) : 4462 / 5000 = 89%")
        print(f"  Database                 : {DATABASE_URI}")
        print(f"{'=' * 64}")

        try:
            from sqlalchemy import text
            db_count = database_session.execute(
                text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
            ).scalar()
            print(f"  DB row count (SequentialPlanDAO) : {db_count}")
        except Exception as count_error:
            print(f"  [db] Could not verify row count: {count_error}")

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo_apartment_jpt()