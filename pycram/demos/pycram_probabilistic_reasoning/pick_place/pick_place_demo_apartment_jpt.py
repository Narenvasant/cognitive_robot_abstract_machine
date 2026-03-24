"""
Apartment world: JPT-guided pick-and-place with GCS navigation and causal failure diagnosis.

Uses the open-world JPT (pick_and_place_jpt.json), trained on 1742 successful
open-world Batch 1 plans, to guide approach-position and arm sampling in the
apartment world. The open-world JPT transfers directly because pick/place
mechanics are identical between the two worlds; only navigation geometry
differs, which GCS handles transparently.

JPT variable mapping (open-world → apartment):
    pick_approach_x/y   →  counter_approach_x/y
    place_approach_x/y  →  table_approach_x/y
    milk_end_x/y/z      →  unchanged
    pick_arm            →  unchanged

Causal failure diagnosis
------------------------
On each failed iteration, CausalCircuit.diagnose_failure() is called to
identify the primary causal variable and recommend a corrective value.
The diagnosis uses the backdoor adjustment formula (Pearl 2009, Thm 3.2.2)
with Z=∅, valid for the independent Batch 1 training data.

CausalCircuit is built once at startup from a ProbModelJPT (probabilistic_model
variant), which exposes the .probabilistic_circuit attribute required by
CausalCircuit.from_jpt(). The pyjpt model is kept separately for sampling
because it is faster.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import inspect
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import rclpy
import sqlalchemy.types as sqlalchemy_types
from sqlalchemy import event, text
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

# pyjpt — for sampling (faster, already proven)
from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT as JointProbabilityTree
from jpt.variables import NumericVariable, SymbolicVariable

# probabilistic_model JPT — exposes .probabilistic_circuit for CausalCircuit
from probabilistic_model.learning.jpt.jpt import JPT as ProbModelJPT
from probabilistic_model.learning.jpt.variables import (
    Continuous as ProbContinuous,
    Symbolic as ProbSymbolic,
)
from random_events.set import Set as RESet
from random_events.interval import closed as re_closed
from random_events.product_algebra import SimpleEvent
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import SumUnit as _SumUnit

# CausalCircuit
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MdVtreeNode,
)

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF, FixedConnection, OmniDrive
from semantic_digital_twin.world_description.geometry import BoundingBox, FileMesh
from semantic_digital_twin.world_description.graph_of_convex_sets import GraphOfConvexSets
from semantic_digital_twin.world_description.shape_collection import BoundingBoxCollection, ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# ── Library patch: SumUnit.simplify() version mismatch ────────────────────────
# SumUnit.simplify() in this version of probabilistic_model calls
# self.add_subcircuit(..., mount=False) but add_subcircuit() does not accept
# a mount= keyword. Patch it here rather than modifying the library source.
_original_sum_simplify = _SumUnit.simplify

def _patched_sum_simplify(self) -> None:
    import numpy as _np
    if len(self.subcircuits) == 1:
        for parent, _, data in list(self.probabilistic_circuit.in_edges(self)):
            self.probabilistic_circuit.add_edge(parent, self.subcircuits[0], data)
        self.probabilistic_circuit.remove_node(self)
        return
    for weight, subcircuit in self.log_weighted_subcircuits:
        if weight == -_np.inf:
            self.probabilistic_circuit.remove_edge(self, subcircuit)
        if type(subcircuit) is type(self):
            for sub_weight, sub_subcircuit in subcircuit.log_weighted_subcircuits:
                self.add_subcircuit(sub_subcircuit, sub_weight + weight)
            self.probabilistic_circuit.remove_node(subcircuit)

_SumUnit.simplify = _patched_sum_simplify


# ── Constants ──────────────────────────────────────────────────────────────────

NUMBER_OF_ITERATIONS: int = 5000

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/demo_robot_plans",
)

MILK_SPAWN_X: float = 2.4
MILK_SPAWN_Y: float = 2.5
MILK_SPAWN_Z: float = 1.01

COUNTER_APPROACH_X: float = 1.6
COUNTER_APPROACH_Y: float = 2.5

TABLE_APPROACH_X: float = 4.2
TABLE_APPROACH_Y: float = 4.0

PLACE_X: float = 5.0
PLACE_Y: float = 4.0
PLACE_Z: float = 0.80

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

ROBOT_INIT_X: float = 1.4
ROBOT_INIT_Y: float = 1.5

GRASP_MANIPULATION_OFFSET: float = 0.06

_RESOURCE_PATH:       Path = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF_PATH:  Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL_PATH:        Path = _RESOURCE_PATH / "objects" / "milk.stl"

JPT_MODEL_PATH:           str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")
TRAINING_CSV_PATH:        str = os.path.join(os.path.dirname(__file__), "pick_and_place_dataframe.csv")
JPT_MIN_SAMPLES_PER_LEAF: int = 25

# Causal priority order from CT1-CT8 ATE analysis
CAUSAL_VARIABLE_NAMES: list = [
    "pick_approach_x", "pick_approach_y",
    "place_approach_x", "place_approach_y",
    "pick_arm",
]
CAUSAL_PRIORITY_ORDER: list = [
    "pick_approach_x",   # rank 1 — ATE_norm 1.714
    "place_approach_x",  # rank 2 — ATE_norm 1.511
    "pick_arm",          # rank 3 — CT4 moderator
    "pick_approach_y",   # rank 4
    "place_approach_y",  # rank 5
]
EFFECT_VARIABLE_NAMES: list = ["milk_end_z"]


# ── pyjpt variable definitions (for sampling) ──────────────────────────────────

ArmChoiceDomain = type(
    "ArmChoiceDomain",
    (Multinomial,),
    {
        "values": OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)]),
        "labels": OrderedDictProxy([(0, "LEFT"), (1, "RIGHT")]),
    },
)

JPT_VARIABLES: list = [
    NumericVariable("pick_approach_x",  precision=0.005),
    NumericVariable("pick_approach_y",  precision=0.005),
    NumericVariable("place_approach_x", precision=0.005),
    NumericVariable("place_approach_y", precision=0.005),
    NumericVariable("milk_end_x",       precision=0.001),
    NumericVariable("milk_end_y",       precision=0.001),
    NumericVariable("milk_end_z",       precision=0.0005),
    SymbolicVariable("pick_arm",        domain=ArmChoiceDomain),
]


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class PlanParameters:
    """Sampled parameters for one pick-and-place iteration."""
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


# ── Monkey-patches ─────────────────────────────────────────────────────────────

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
    """Patch the PyCRAM ORM numpy TypeDecorator to handle None values gracefully."""
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
        print("  [patch] WARNING: ORM numpy TypeDecorator not found; None-guard skipped.")
        return
    original_bind = target_class.process_bind_param
    def _none_guarded_bind(self, value, dialect):
        if value is None:
            return None
        return original_bind(self, value, dialect)
    target_class.process_bind_param = _none_guarded_bind
    print(f"  [patch] Patched {target_class.__name__}.process_bind_param.")

_patch_orm_numpy_array_type()


# ── World construction ─────────────────────────────────────────────────────────

def _build_world(apartment_urdf_path: Path) -> tuple[World, PR2]:
    world     = URDFParser.from_file(str(apartment_urdf_path)).parse()
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
    mesh      = FileMesh.from_file(str(milk_stl_path))
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
        connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
        world.add_connection(connection)
        connection.origin = spawn_pose
        world.add_semantic_annotation(Milk(root=milk_body))
    return milk_body


def _respawn_milk(world: World, milk_body: Body) -> None:
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_SPAWN_X, MILK_SPAWN_Y, MILK_SPAWN_Z, 0, 0, 0
    )
    with world.modify_world():
        connection = milk_body.parent_connection
        if connection is not None and connection.parent is not world.root:
            world.remove_connection(connection)
            new_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(new_connection)
            new_connection.origin = spawn_pose
        elif connection is not None:
            connection.origin = spawn_pose
        else:
            new_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(new_connection)
            new_connection.origin = spawn_pose
    print(f"  [respawn] Milk reset to ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")


# ── Database ───────────────────────────────────────────────────────────────────

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
    def _shorten(name: str, limit: int = 63) -> str:
        if len(name) <= limit:
            return name
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:limit - 9]}_{digest}"
    for table in Base.metadata.tables.values():
        shortened = _shorten(table.name)
        if shortened != table.name:
            table.name     = shortened
            table.fullname = shortened


def _register_postgresql_numpy_scalar_coercion(engine: Any) -> None:
    import numpy

    def _coerce_scalar(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        return value

    def _coerce_parameters(parameters: Any) -> Any:
        if isinstance(parameters, dict):
            return {key: _coerce_scalar(value) for key, value in parameters.items()}
        return parameters

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _before_cursor_execute(connection, cursor, statement, parameters, context, executemany):
        if isinstance(parameters, dict):
            parameters = _coerce_parameters(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(_coerce_parameters(p) for p in parameters)
        return statement, parameters


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    print("  [db] Persisting plan ...")
    session.add(to_dao(plan))
    session.commit()
    print("  [db] Plan committed.")


# ── GCS navigation ─────────────────────────────────────────────────────────────

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
    build_start    = time.time()
    navigation_map = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=GCS_OBSTACLE_BLOAT,
    )
    print(f"  GCS built in {time.time() - build_start:.2f}s  ({len(list(navigation_map.graph.nodes()))} nodes)")
    return navigation_map


def _build_gcs_node_array(navigation_map: GraphOfConvexSets) -> np.ndarray:
    """
    Pre-compute a (N, 6) array of [min_x, min_y, min_z, max_x, max_y, max_z]
    for all GCS nodes. Enables vectorised free-space checks ~100x faster than
    the Python loop in GraphOfConvexSets.node_of_point.
    """
    nodes      = list(navigation_map.graph.nodes())
    node_array = np.array(
        [[n.min_x, n.min_y, n.min_z, n.max_x, n.max_y, n.max_z] for n in nodes],
        dtype=np.float64,
    )
    print(f"  GCS node index: {len(node_array)} bounding boxes")
    return node_array


def _is_in_free_space(node_array: np.ndarray, x: float, y: float, z: float) -> bool:
    inside = (
        (node_array[:, 0] <= x) & (x <= node_array[:, 3]) &
        (node_array[:, 1] <= y) & (y <= node_array[:, 4]) &
        (node_array[:, 2] <= z) & (z <= node_array[:, 5])
    )
    return bool(inside.any())


def _find_nearest_free_point(
    navigation_map: GraphOfConvexSets,
    node_array:     np.ndarray,
    x:              float,
    y:              float,
    z:              float,
    world:          World,
    search_radius:  float = 0.6,
    radial_step:    float = 0.05,
    angular_steps:  int   = 16,
) -> Optional[Point3]:
    """Spiral outward from (x, y, z) until a GCS free-space cell is found."""
    if _is_in_free_space(node_array, x, y, z):
        return Point3(x, y, z, reference_frame=world.root)
    print(f"    [GCS] ({x:.3f},{y:.3f}) is occupied — searching for free point ...")
    for radius in np.arange(radial_step, search_radius + radial_step, radial_step):
        angles = np.linspace(0, 2 * np.pi, angular_steps, endpoint=False)
        for cx, cy in zip(x + radius * np.cos(angles), y + radius * np.sin(angles)):
            if _is_in_free_space(node_array, cx, cy, z):
                print(f"    [GCS] Free point at ({cx:.3f},{cy:.3f}) r={radius:.2f}")
                return Point3(cx, cy, z, reference_frame=world.root)
    print(f"    [GCS] No free point found within r={search_radius}")
    return None


def _make_pose_stamped(x: float, y: float, z: float, frame_id: Any) -> PoseStamped:
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame_id, stamp=datetime.datetime.now(), sequence=0),
    )


def _path_to_navigate_actions(
    path:              List[Point3],
    world_frame:       Any,
    keep_joint_states: bool,
) -> List[NavigateAction]:
    return [
        NavigateAction(
            target_location=PoseStamped(
                pose=PyCramPose(
                    position=PyCramVector3(x=float(p.x), y=float(p.y), z=0.0),
                    orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
                ),
                header=Header(frame_id=world_frame),
            ),
            keep_joint_states=keep_joint_states,
        )
        for p in path[1:]
    ]


def _navigate_via_gcs(
    context:           Context,
    navigation_map:    GraphOfConvexSets,
    node_array:        np.ndarray,
    start_x:           float,
    start_y:           float,
    goal_x:            float,
    goal_y:            float,
    world:             World,
    keep_joint_states: bool = False,
) -> List[NavigateAction]:
    """
    Plan a collision-free path via GCS and return it as NavigateActions.
    Falls back to a single direct NavigateAction if GCS cannot find a path.
    """
    z = (GCS_SEARCH_MIN_Z + GCS_SEARCH_MAX_Z) / 2.0
    start_point     = Point3(start_x, start_y, z, reference_frame=world.root)
    goal_point      = Point3(goal_x,  goal_y,  z, reference_frame=world.root)
    direct_fallback = [NavigateAction(
        target_location=_make_pose_stamped(goal_x, goal_y, 0.0, world.root),
        keep_joint_states=keep_joint_states,
    )]

    try:
        path = navigation_map.path_from_to(start_point, goal_point)
    except Exception as path_error:
        print(f"    [GCS] path_from_to failed: {path_error} — direct navigation.")
        return direct_fallback

    if path is None or len(path) < 2:
        print(f"    [GCS] No path found — direct navigation.")
        return direct_fallback

    navigate_actions = _path_to_navigate_actions(path, world.root, keep_joint_states)
    print(f"    [GCS] ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): {len(navigate_actions)} waypoint(s)")
    for i, action in enumerate(navigate_actions):
        pos = action.target_location.pose.position
        print(f"           waypoint {i + 1}: ({pos.x:.3f}, {pos.y:.3f})")
    return navigate_actions


# ── JPT loading and sampling ───────────────────────────────────────────────────

def _load_joint_probability_tree(model_path: str) -> JointProbabilityTree:
    print(f"  [jpt] Loading model from {model_path} ...")
    jpt = JointProbabilityTree(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    ).load(model_path)
    print(f"  [jpt] Loaded — {len(jpt.leaves)} leaves")
    return jpt


def _sample_plan_parameters(jpt: JointProbabilityTree) -> PlanParameters:
    """
    Draw one joint sample from the open-world JPT and map it to apartment
    PlanParameters, clipping both approach positions to the apartment zones.

    pyjpt sample() returns a numpy array (1, n_variables) with columns in
    the same order as JPT_VARIABLES.
    """
    sample_row     = jpt.sample(1)[0]
    sample_by_name = {v.name: sample_row[i] for i, v in enumerate(JPT_VARIABLES)}

    arm_label = sample_by_name["pick_arm"]
    if isinstance(arm_label, (int, float)):
        arm_label = ArmChoiceDomain.labels[int(arm_label)]
    pick_arm = Arms.LEFT if arm_label == "LEFT" else Arms.RIGHT

    return PlanParameters(
        counter_approach_x = float(np.clip(sample_by_name["pick_approach_x"],  COUNTER_APPROACH_MIN_X, COUNTER_APPROACH_MAX_X)),
        counter_approach_y = float(np.clip(sample_by_name["pick_approach_y"],  COUNTER_APPROACH_MIN_Y, COUNTER_APPROACH_MAX_Y)),
        table_approach_x   = float(np.clip(sample_by_name["place_approach_x"], TABLE_APPROACH_MIN_X,   TABLE_APPROACH_MAX_X)),
        table_approach_y   = float(np.clip(sample_by_name["place_approach_y"], TABLE_APPROACH_MIN_Y,   TABLE_APPROACH_MAX_Y)),
        pick_arm           = pick_arm,
    )


# ── CausalCircuit setup ────────────────────────────────────────────────────────

def _build_prob_model_variables(csv_path: str) -> list:
    """
    Build probabilistic_model variable definitions from training CSV statistics.
    ProbModelJPT requires its own Continuous/Symbolic types (sortable
    random_events subclasses), which are separate from pyjpt's NumericVariable.
    """
    df = pd.read_csv(csv_path)
    def _stats(col):
        return float(df[col].mean()), float(df[col].std())
    return [
        ProbContinuous("pick_approach_x",  *_stats("pick_approach_x")),
        ProbContinuous("pick_approach_y",  *_stats("pick_approach_y")),
        ProbContinuous("place_approach_x", *_stats("place_approach_x")),
        ProbContinuous("place_approach_y", *_stats("place_approach_y")),
        ProbContinuous("milk_end_x",       *_stats("milk_end_x")),
        ProbContinuous("milk_end_y",       *_stats("milk_end_y")),
        ProbContinuous("milk_end_z",       *_stats("milk_end_z")),
        ProbSymbolic("pick_arm", RESet.from_iterable(["LEFT", "RIGHT"])),
    ]


def _build_causal_circuit(csv_path: str) -> CausalCircuit:
    """
    Fit a ProbModelJPT and construct a CausalCircuit from it.

    A separate ProbModelJPT is used (rather than pyjpt) because CausalCircuit
    requires access to a ProbabilisticCircuit object, which only the
    probabilistic_model JPT exposes via .probabilistic_circuit after fit().
    pyjpt is kept separately for sampling because it is faster.

    The CausalCircuit is built once at startup and reused for all failure
    diagnosis calls — backdoor_adjustment() is cached per cause variable
    inside diagnose_failure().
    """
    print("  [causal] Fitting ProbModelJPT from training CSV ...")
    prob_vars = _build_prob_model_variables(csv_path)
    df = pd.read_csv(csv_path)
    prob_jpt = ProbModelJPT(variables=prob_vars, min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF)
    prob_jpt.fit(df[[v.name for v in prob_vars]])
    print(f"  [causal] ProbModelJPT fitted. Leaves: {len(list(prob_jpt.probabilistic_circuit.leaves))}")

    mdvtree = MdVtreeNode.from_causal_graph(
        causal_variable_names=CAUSAL_VARIABLE_NAMES,
        effect_variable_names=EFFECT_VARIABLE_NAMES,
        causal_priority_order=CAUSAL_PRIORITY_ORDER,
    )
    causal_circuit = CausalCircuit.from_jpt(
        fitted_jpt=prob_jpt,
        mdvtree=mdvtree,
        causal_variable_names=CAUSAL_VARIABLE_NAMES,
        effect_variable_names=EFFECT_VARIABLE_NAMES,
    )
    q_check = causal_circuit.verify_q_determinism()
    print(f"  [causal] CausalCircuit ready. Q-determinism: {'PASS' if q_check.passed else 'FAIL — ' + '; '.join(q_check.violations)}")
    return causal_circuit


def _diagnose_and_log(
    causal_circuit: CausalCircuit,
    parameters:     PlanParameters,
    iteration:      int,
) -> None:
    """
    Run causal failure diagnosis and print a structured report.

    Maps apartment PlanParameters back to the JPT variable names
    (counter_approach → pick_approach, table_approach → place_approach)
    before calling diagnose_failure().
    """
    # Open-world constants (from pick_and_place_demo.py)
    _OW_PLACE_TARGET_X = 4.1  # PLACE_TARGET_X in the open world

    observed = {
        # x axis coincides in both worlds (milk at x=2.4 in both)
        "pick_approach_x": parameters.counter_approach_x,

        # absolute y → lateral offset from milk centreline
        # open-world milk at y=0, apartment milk at y=MILK_SPAWN_Y=2.5
        "pick_approach_y": parameters.counter_approach_y - MILK_SPAWN_Y,

        # apartment place target is 0.9m further in x than open-world target
        # shift back: 4.1 - 0.9 = 3.2, 4.5 - 0.9 = 3.6 → within JPT [3.2, 3.79]
        "place_approach_x": parameters.table_approach_x - (PLACE_X - _OW_PLACE_TARGET_X),

        # absolute y → lateral offset from place target centreline
        # open-world target at y=0, apartment target at y=PLACE_Y=4.0
        "place_approach_y": parameters.table_approach_y - PLACE_Y,

        "pick_arm": parameters.pick_arm.name,
    }

    try:
        diagnosis = causal_circuit.diagnose_failure(
            observed_parameter_values=observed,
            effect_variable_name="milk_end_z",
            query_resolution=0.005,
        )

        # Map JPT variable names back to apartment names for display
        name_map = {
            "pick_approach_x":  "counter_approach_x",
            "pick_approach_y":  "counter_approach_y",
            "place_approach_x": "table_approach_x",
            "place_approach_y": "table_approach_y",
        }

        primary_display = name_map.get(
            diagnosis.primary_cause_variable_name,
            diagnosis.primary_cause_variable_name,
        )

        print(f"\n  ┌─ CAUSAL FAILURE DIAGNOSIS  (iteration {iteration}) {'─' * 30}")
        print(f"  │  Primary cause:    {primary_display}")
        print(f"  │  Observed value:   {diagnosis.actual_value:.4f}")
        print(f"  │  P(success|do):    {diagnosis.interventional_probability_at_failure:.4f}"
              + ("  ← OUT OF TRAINING SUPPORT" if diagnosis.interventional_probability_at_failure == 0.0 else ""))
        if diagnosis.recommended_value is not None:
            print(f"  │  Recommended:      {diagnosis.recommended_value:.4f}")
            print(f"  │  P(success|rec):   {diagnosis.interventional_probability_at_recommendation:.4f}")
        print(f"  │")
        print(f"  │  All variables:")
        for jpt_name, result in diagnosis.all_variable_results.items():
            display_name = name_map.get(jpt_name, jpt_name)
            marker = "  ← PRIMARY CAUSE" if jpt_name == diagnosis.primary_cause_variable_name else ""
            ood = "  [OUT OF SUPPORT]" if result["interventional_probability"] == 0.0 else ""
            print(f"  │    {display_name:<24}  actual={result['actual_value']:.4f}  "
                  f"P={result['interventional_probability']:.4f}{ood}{marker}")
        print(f"  └{'─' * 60}")

    except Exception as diag_error:
        print(f"  [causal] Diagnosis failed (iteration {iteration}): {diag_error}")


# ── Plan construction ──────────────────────────────────────────────────────────

def _build_fixed_plan(
    context:        Context,
    world:          World,
    robot:          PR2,
    milk_body:      Body,
    navigation_map: GraphOfConvexSets,
    node_array:     np.ndarray,
    robot_start_x:  float,
    robot_start_y:  float,
) -> SequentialPlan:
    """Deterministic seed plan for iteration 1."""
    arm = Arms.RIGHT
    navigate_to_counter = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=robot_start_x,     start_y=robot_start_y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X,    goal_y=TABLE_APPROACH_Y,
        world=world, keep_joint_states=True,
    )
    place_pose = _make_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world.root)
    print(f"  [plan] seed — counter:({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})  "
          f"table:({TABLE_APPROACH_X},{TABLE_APPROACH_Y})  arm:{arm}")
    return SequentialPlan(context,
        ParkArmsAction(arm=Arms.BOTH),
        *navigate_to_counter,
        PickUpAction(
            object_designator=milk_body,
            arm=arm,
            grasp_description=GraspDescription(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                rotate_gripper=False,
                manipulation_offset=GRASP_MANIPULATION_OFFSET,
                manipulator=robot.right_arm.manipulator,
            ),
        ),
        *navigate_to_table,
        PlaceAction(object_designator=milk_body, target_location=place_pose, arm=arm),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _build_sampled_plan(
    context:        Context,
    parameters:     PlanParameters,
    world:          World,
    robot:          PR2,
    milk_body:      Body,
    navigation_map: GraphOfConvexSets,
    node_array:     np.ndarray,
    robot_start_x:  float,
    robot_start_y:  float,
) -> SequentialPlan:
    """Build a plan from JPT-sampled approach positions."""
    manipulator = (
        robot.right_arm.manipulator if parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )
    navigate_to_counter = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=robot_start_x,               start_y=robot_start_y,
        goal_x=parameters.counter_approach_x, goal_y=parameters.counter_approach_y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=parameters.counter_approach_x, start_y=parameters.counter_approach_y,
        goal_x=parameters.table_approach_x,    goal_y=parameters.table_approach_y,
        world=world, keep_joint_states=True,
    )
    place_pose = _make_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world.root)
    print(f"  [plan] sampled — counter:({parameters.counter_approach_x:.3f},{parameters.counter_approach_y:.3f})  "
          f"table:({parameters.table_approach_x:.3f},{parameters.table_approach_y:.3f})  arm:{parameters.pick_arm}")
    return SequentialPlan(context,
        ParkArmsAction(arm=Arms.BOTH),
        *navigate_to_counter,
        PickUpAction(
            object_designator=milk_body,
            arm=parameters.pick_arm,
            grasp_description=GraspDescription(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                rotate_gripper=False,
                manipulation_offset=GRASP_MANIPULATION_OFFSET,
                manipulator=manipulator,
            ),
        ),
        *navigate_to_table,
        PlaceAction(object_designator=milk_body, target_location=place_pose, arm=parameters.pick_arm),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _navigate_back_to_start(
    context:            Context,
    navigation_map:     GraphOfConvexSets,
    node_array:         np.ndarray,
    world:              World,
    table_approach_x:   float,
    table_approach_y:   float,
    counter_approach_x: float,
    counter_approach_y: float,
) -> None:
    """
    Return the robot to the start zone by retracing the forward path in reverse:
        table_approach -> counter_approach -> COUNTER_APPROACH (fixed start zone)
    """
    print(f"  [return] table({table_approach_x:.2f},{table_approach_y:.2f}) -> "
          f"counter({counter_approach_x:.2f},{counter_approach_y:.2f}) -> "
          f"start({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})")
    leg_table_to_counter = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=table_approach_x,  start_y=table_approach_y,
        goal_x=counter_approach_x, goal_y=counter_approach_y,
        world=world,
    )
    leg_counter_to_start = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=counter_approach_x, start_y=counter_approach_y,
        goal_x=COUNTER_APPROACH_X,  goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    return_plan = SequentialPlan(context, *(leg_table_to_counter + leg_counter_to_start))
    with simulated_robot:
        try:
            return_plan.perform()
            print("  [return] Robot at start position.")
        except Exception as return_error:
            print(f"  [return] WARNING: return navigation failed: {return_error}")


# ── Entry point ────────────────────────────────────────────────────────────────

def pick_and_place_demo_apartment_jpt() -> None:
    """Apartment world Batch 2: 5000 iterations of JPT-guided pick-and-place with causal diagnosis."""
    print("=" * 64)
    print("  pick_and_place_demo_apartment_jpt  (Batch 2 / JPT + Causal)")
    print(f"  Iterations   : {NUMBER_OF_ITERATIONS}")
    print(f"  Place target : ({PLACE_X}, {PLACE_Y}, {PLACE_Z})")
    print(f"  JPT model    : {JPT_MODEL_PATH}")
    print(f"  Database     : {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/6] Building apartment world ...")
    world, robot = _build_world(APARTMENT_URDF_PATH)
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    node_array     = _build_gcs_node_array(navigation_map)

    print("\n[2/6] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/6] Loading pyjpt model (for sampling) ...")
    jpt = _load_joint_probability_tree(JPT_MODEL_PATH)

    print("\n[4/6] Building CausalCircuit (for failure diagnosis) ...")
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

        planning_context = Context(world, robot, None)

        print("\n[6/6] Running iterations ...")
        successful_count = 0
        failed_count     = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(f"  Iteration {iteration}/{NUMBER_OF_ITERATIONS}  (success={successful_count}  failed={failed_count})")
            print(f"{'=' * 64}")

            if iteration == 1:
                print("  Mode: FIXED")
                plan = _build_fixed_plan(
                    planning_context, world, robot, milk_body, navigation_map, node_array,
                    robot_start_x=ROBOT_INIT_X, robot_start_y=ROBOT_INIT_Y,
                )
                return_table_approach_x,   return_table_approach_y   = TABLE_APPROACH_X,   TABLE_APPROACH_Y
                return_counter_approach_x, return_counter_approach_y = COUNTER_APPROACH_X, COUNTER_APPROACH_Y
                current_parameters = None
            else:
                print("  Mode: JPT-SAMPLED")
                current_parameters = _sample_plan_parameters(jpt)
                plan = _build_sampled_plan(
                    planning_context, current_parameters, world, robot, milk_body,
                    navigation_map, node_array,
                    robot_start_x=ROBOT_INIT_X, robot_start_y=ROBOT_INIT_Y,
                )
                return_table_approach_x,   return_table_approach_y   = current_parameters.table_approach_x,   current_parameters.table_approach_y
                return_counter_approach_x, return_counter_approach_y = current_parameters.counter_approach_x, current_parameters.counter_approach_y

            print("\n  Executing plan ...")
            execution_succeeded = False
            with simulated_robot:
                try:
                    plan.perform()
                    execution_succeeded = True
                    print("  Execution complete.")
                except Exception as execution_error:
                    failed_count += 1
                    print(f"  RESULT: FAILED — {type(execution_error).__name__}: {execution_error}")
                    for line in traceback.format_exc().strip().splitlines()[-3:]:
                        print(f"    {line}")

                    # ── Causal failure diagnosis ───────────────────────────────
                    # Only meaningful for sampled iterations where we have
                    # JPT-drawn parameters to diagnose.
                    if current_parameters is not None:
                        _diagnose_and_log(causal_circuit, current_parameters, iteration)

            if execution_succeeded:
                try:
                    _persist_plan(database_session, plan)
                    successful_count += 1
                    print(f"  RESULT: SUCCESS  ({successful_count}/{iteration} stored,  "
                          f"{NUMBER_OF_ITERATIONS - iteration} remaining)")
                except Exception as database_error:
                    print(f"  [db] ERROR: {database_error}")
                    traceback.print_exc()

            print("\n  Resetting ...")
            _respawn_milk(world, milk_body)
            _navigate_back_to_start(
                planning_context, navigation_map, node_array, world,
                table_approach_x=return_table_approach_x,     table_approach_y=return_table_approach_y,
                counter_approach_x=return_counter_approach_x, counter_approach_y=return_counter_approach_y,
            )

        success_rate = 100 * successful_count // NUMBER_OF_ITERATIONS
        print(f"\n{'=' * 64}")
        print(f"  Run complete.")
        print(f"  Successful : {successful_count} / {NUMBER_OF_ITERATIONS}  ({success_rate}%)")
        print(f"  Failed     : {failed_count}")
        print(f"{'=' * 64}")

        try:
            row_count = database_session.execute(
                text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
            ).scalar()
            print(f"  DB rows (SequentialPlanDAO): {row_count}")
        except Exception as count_error:
            print(f"  [db] Could not read row count: {count_error}")

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo_apartment_jpt()