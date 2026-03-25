"""
Apartment world: JPT-guided pick-and-place with GCS navigation and causal failure diagnosis.

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

Causal failure diagnosis
------------------------
On each failed iteration, CausalCircuit.diagnose_failure() identifies the
primary causal variable and recommends a corrective value. The diagnosis
uses the backdoor adjustment formula (Pearl 2009, Thm 3.2.2) with Z=∅,
valid for the independent Batch 1 training data.

The CausalCircuit is built once at startup from a ProbModelJPT
(probabilistic_model variant), which exposes the .probabilistic_circuit
attribute required by CausalCircuit.from_jpt(). The pyjpt model is kept
separately for sampling because it is faster at inference time.
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
from random_events.product_algebra import SimpleEvent
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import SumUnit as _SumUnit

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

ROBOT_INIT_X: float = 1.4
ROBOT_INIT_Y: float = 1.5

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
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PlanParameters:
    """Sampled parameters for one pick-and-place iteration."""
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


# ---------------------------------------------------------------------------
# ORM patch: handle None values in numpy TypeDecorator
# ---------------------------------------------------------------------------

def _patch_orm_numpy_array_type() -> None:
    """
    Patch the PyCRAM ORM numpy TypeDecorator so that None values are passed
    through without calling .astype(), which would raise AttributeError when
    a plan action has no associated array data.
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


def _build_gcs_node_array(navigation_map: GraphOfConvexSets) -> np.ndarray:
    """
    Pre-compute a (N, 6) array of [min_x, min_y, min_z, max_x, max_y, max_z]
    for all GCS nodes. Enables vectorised free-space checks roughly 100x faster
    than the Python loop in GraphOfConvexSets.node_of_point.
    """
    nodes = list(navigation_map.graph.nodes())
    node_bounding_boxes = np.array(
        [[node.min_x, node.min_y, node.min_z, node.max_x, node.max_y, node.max_z]
         for node in nodes],
        dtype=np.float64,
    )
    print(f"  GCS node index: {len(node_bounding_boxes)} bounding boxes")
    return node_bounding_boxes


def _is_in_free_space(
    node_bounding_boxes: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> bool:
    inside = (
        (node_bounding_boxes[:, 0] <= x) & (x <= node_bounding_boxes[:, 3]) &
        (node_bounding_boxes[:, 1] <= y) & (y <= node_bounding_boxes[:, 4]) &
        (node_bounding_boxes[:, 2] <= z) & (z <= node_bounding_boxes[:, 5])
    )
    return bool(inside.any())


def _find_nearest_free_point(
    navigation_map:      GraphOfConvexSets,
    node_bounding_boxes: np.ndarray,
    x:                   float,
    y:                   float,
    z:                   float,
    world:               World,
    search_radius:       float = 0.6,
    radial_step:         float = 0.05,
    angular_steps:       int   = 16,
) -> Optional[Point3]:
    """Spiral outward from (x, y, z) until a GCS free-space cell is found."""
    if _is_in_free_space(node_bounding_boxes, x, y, z):
        return Point3(x, y, z, reference_frame=world.root)
    print(f"    [GCS] ({x:.3f},{y:.3f}) is occupied — searching for nearest free point ...")
    for radius in np.arange(radial_step, search_radius + radial_step, radial_step):
        angles = np.linspace(0, 2 * np.pi, angular_steps, endpoint=False)
        for candidate_x, candidate_y in zip(
            x + radius * np.cos(angles),
            y + radius * np.sin(angles),
        ):
            if _is_in_free_space(node_bounding_boxes, candidate_x, candidate_y, z):
                print(f"    [GCS] Free point found at ({candidate_x:.3f},{candidate_y:.3f}) r={radius:.2f}")
                return Point3(candidate_x, candidate_y, z, reference_frame=world.root)
    print(f"    [GCS] No free point found within radius={search_radius}")
    return None


def _make_pose(x: float, y: float, z: float, reference_frame: Any) -> Pose:
    return Pose(
        position=Point3(x=x, y=y, z=z),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        reference_frame=reference_frame,
    )


def _path_to_navigate_actions(
    path:            List[Point3],
    world_frame:     Any,
    keep_joint_states: bool,
) -> List[NavigateAction]:
    return [
        NavigateAction(
            target_location=Pose(
                position=Point3(x=float(waypoint.x), y=float(waypoint.y), z=0.0,
                                reference_frame=world_frame),
                orientation=Quaternion(x=0, y=0, z=0, w=1, reference_frame=world_frame),
                reference_frame=world_frame,
            ),
        )
        for waypoint in path[1:]
    ]


def _navigate_via_gcs(
    context:             Context,
    navigation_map:      GraphOfConvexSets,
    node_bounding_boxes: np.ndarray,
    start_x:             float,
    start_y:             float,
    goal_x:              float,
    goal_y:              float,
    world:               World,
    keep_joint_states:   bool = False,
) -> List[NavigateAction]:
    """
    Plan a collision-free path via GCS and return it as NavigateActions.
    Falls back to a single direct NavigateAction if GCS cannot find a path.
    """
    midpoint_z  = (GCS_SEARCH_MIN_Z + GCS_SEARCH_MAX_Z) / 2.0
    start_point = Point3(start_x, start_y, midpoint_z, reference_frame=world.root)
    goal_point  = Point3(goal_x,  goal_y,  midpoint_z, reference_frame=world.root)

    direct_fallback = [NavigateAction(
        target_location=_make_pose(goal_x, goal_y, 0.0, world.root),
        keep_joint_states=keep_joint_states,
    )]

    try:
        path = navigation_map.path_from_to(start_point, goal_point)
    except Exception as path_error:
        print(f"    [GCS] path_from_to failed: {path_error} — falling back to direct navigation.")
        return direct_fallback

    if path is None or len(path) < 2:
        print("    [GCS] No path found — falling back to direct navigation.")
        return direct_fallback

    navigate_actions = _path_to_navigate_actions(path, world.root, keep_joint_states)
    print(f"    [GCS] ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): "
          f"{len(navigate_actions)} waypoint(s)")
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
) -> PlanParameters:
    """
    Draw one joint sample from the open-world JPT and map it to apartment
    PlanParameters, clipping approach positions to the apartment zones.

    All five plan variables are drawn in a single call so their values reflect
    the learned joint distribution of successful executions rather than being
    drawn independently. The clipping ensures positions stay within the
    apartment's physically reachable zones.
    """
    sample_row     = joint_probability_tree.sample(1)[0]
    sample_by_name = {variable.name: sample_row[index]
                      for index, variable in enumerate(JPT_VARIABLES)}

    arm_label = sample_by_name["pick_arm"]
    if isinstance(arm_label, (int, float)):
        arm_label = ArmChoiceDomain.labels[int(arm_label)]
    pick_arm = Arms.LEFT if arm_label == "LEFT" else Arms.RIGHT

    return PlanParameters(
        counter_approach_x=float(np.clip(
            sample_by_name["pick_approach_x"],
            COUNTER_APPROACH_MIN_X, COUNTER_APPROACH_MAX_X,
        )),
        counter_approach_y=float(np.clip(
            sample_by_name["pick_approach_y"],
            COUNTER_APPROACH_MIN_Y, COUNTER_APPROACH_MAX_Y,
        )),
        table_approach_x=float(np.clip(
            sample_by_name["place_approach_x"],
            TABLE_APPROACH_MIN_X, TABLE_APPROACH_MAX_X,
        )),
        table_approach_y=float(np.clip(
            sample_by_name["place_approach_y"],
            TABLE_APPROACH_MIN_Y, TABLE_APPROACH_MAX_Y,
        )),
        pick_arm=pick_arm,
    )


# ---------------------------------------------------------------------------
# CausalCircuit construction
# ---------------------------------------------------------------------------

def _build_prob_model_variables(csv_path: str) -> list:
    """
    Build probabilistic_model variable definitions from training CSV statistics.

    ProbModelJPT requires its own Continuous/Symbolic types from random_events,
    which are separate from pyjpt's NumericVariable. Mean and std are used to
    initialise the Continuous variables, matching the convention expected by
    the probabilistic_model JPT fitting procedure.
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

    A separate ProbModelJPT is used rather than pyjpt because CausalCircuit
    requires access to a ProbabilisticCircuit object, which only the
    probabilistic_model JPT exposes via .probabilistic_circuit after fit().
    The pyjpt model is kept separately for sampling because it is faster.

    The CausalCircuit is built once at startup and reused across all iterations.
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
# Failure diagnosis
# ---------------------------------------------------------------------------

def _diagnose_and_log(
    causal_circuit:    CausalCircuit,
    plan_parameters:   PlanParameters,
    iteration_number:  int,
) -> None:
    """
    Run causal failure diagnosis and print a structured report.

    Apartment approach coordinates are remapped to open-world JPT coordinates
    before calling diagnose_failure(). This is necessary because the JPT was
    trained in the open world where:
      - The milk is at y=0 (apartment: y=MILK_SPAWN_Y=2.5)
      - The place target is at x=4.1 (apartment: x=PLACE_TARGET_X=5.0)

    The remapping shifts absolute apartment positions to the lateral offsets
    that the JPT learned during Batch 1 training.
    """
    observed_parameter_values = {
        "pick_approach_x": plan_parameters.counter_approach_x,
        "pick_approach_y": plan_parameters.counter_approach_y - MILK_SPAWN_Y,
        "place_approach_x": plan_parameters.table_approach_x - (
            PLACE_TARGET_X - OPEN_WORLD_PLACE_TARGET_X
        ),
        "place_approach_y": plan_parameters.table_approach_y - PLACE_TARGET_Y,
        "pick_arm": plan_parameters.pick_arm.name,
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

    except Exception as diagnosis_error:
        print(f"  [causal] Diagnosis failed (iteration {iteration_number}): {diagnosis_error}")


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_fixed_plan(
    planning_context:    Context,
    world:               World,
    robot:               PR2,
    milk_body:           Body,
    navigation_map:      GraphOfConvexSets,
    node_bounding_boxes: np.ndarray,
    robot_start_x:       float,
    robot_start_y:       float,
) -> SequentialPlan:
    """
    Deterministic seed plan used for iteration 1.

    Using known-good fixed parameters on the first iteration confirms that
    the world, robot, and database are correctly initialised before
    probabilistic sampling begins.
    """
    seed_arm = Arms.RIGHT
    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, node_bounding_boxes,
        start_x=robot_start_x,     start_y=robot_start_y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        planning_context, navigation_map, node_bounding_boxes,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X,    goal_y=TABLE_APPROACH_Y,
        world=world, keep_joint_states=True,
    )
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)
    print(f"  [plan] seed — counter:({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})  "
          f"table:({TABLE_APPROACH_X},{TABLE_APPROACH_Y})  arm:{seed_arm}")
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
    planning_context:    Context,
    plan_parameters:     PlanParameters,
    world:               World,
    robot:               PR2,
    milk_body:           Body,
    navigation_map:      GraphOfConvexSets,
    node_bounding_boxes: np.ndarray,
    robot_start_x:       float,
    robot_start_y:       float,
) -> SequentialPlan:
    """Build a plan from JPT-sampled approach positions and arm choice."""
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )
    navigate_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, node_bounding_boxes,
        start_x=robot_start_x,                    start_y=robot_start_y,
        goal_x=plan_parameters.counter_approach_x, goal_y=plan_parameters.counter_approach_y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        planning_context, navigation_map, node_bounding_boxes,
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
    planning_context:    Context,
    navigation_map:      GraphOfConvexSets,
    node_bounding_boxes: np.ndarray,
    world:               World,
    table_approach_x:    float,
    table_approach_y:    float,
    counter_approach_x:  float,
    counter_approach_y:  float,
) -> None:
    """
    Return the robot to the start zone by retracing the forward path in reverse:
        table_approach → counter_approach → fixed start zone
    """
    print(
        f"  [return] table({table_approach_x:.2f},{table_approach_y:.2f}) -> "
        f"counter({counter_approach_x:.2f},{counter_approach_y:.2f}) -> "
        f"start({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})"
    )
    leg_table_to_counter = _navigate_via_gcs(
        planning_context, navigation_map, node_bounding_boxes,
        start_x=table_approach_x,   start_y=table_approach_y,
        goal_x=counter_approach_x,  goal_y=counter_approach_y,
        world=world,
    )
    leg_counter_to_start = _navigate_via_gcs(
        planning_context, navigation_map, node_bounding_boxes,
        start_x=counter_approach_x, start_y=counter_approach_y,
        goal_x=COUNTER_APPROACH_X,  goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    return_plan = SequentialPlan(
        planning_context,
        *(leg_table_to_counter + leg_counter_to_start),
    )
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
    Apartment world Batch 2: 5000 iterations of JPT-guided pick-and-place
    with causal failure diagnosis.

    Iteration 1 uses fixed deterministic parameters to confirm the world,
    robot, and database are correctly initialised. Subsequent iterations
    sample jointly from the open-world JPT, concentrating parameter choices
    in the region of the space that historically led to successful executions.

    On each failure, CausalCircuit.diagnose_failure() identifies the primary
    causal variable and recommends a corrective value based on the backdoor
    adjustment formula with Z=∅.
    """
    print("=" * 64)
    print("  pick_and_place_demo_apartment_jpt  (Batch 2 / JPT + Causal)")
    print(f"  Iterations   : {NUMBER_OF_ITERATIONS}")
    print(f"  Place target : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model    : {JPT_MODEL_PATH}")
    print(f"  Database     : {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/6] Building apartment world ...")
    world, robot = _build_world(APARTMENT_URDF_PATH)
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map      = _build_navigation_map(world)
    node_bounding_boxes = _build_gcs_node_array(navigation_map)

    print("\n[2/6] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/6] Loading pyjpt model (for sampling) ...")
    joint_probability_tree = _load_joint_probability_tree(JPT_MODEL_PATH)

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

        for iteration_number in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration_number}/{NUMBER_OF_ITERATIONS}  "
                f"(success={successful_count}  failed={failed_count})"
            )
            print(f"{'=' * 64}")

            if iteration_number == 1:
                print("  Mode: FIXED")
                plan = _build_fixed_plan(
                    planning_context, world, robot, milk_body,
                    navigation_map, node_bounding_boxes,
                    robot_start_x=ROBOT_INIT_X, robot_start_y=ROBOT_INIT_Y,
                )
                return_table_approach_x   = TABLE_APPROACH_X
                return_table_approach_y   = TABLE_APPROACH_Y
                return_counter_approach_x = COUNTER_APPROACH_X
                return_counter_approach_y = COUNTER_APPROACH_Y
                current_parameters        = None
            else:
                print("  Mode: JPT-SAMPLED")
                current_parameters = _sample_plan_parameters(joint_probability_tree)
                plan = _build_sampled_plan(
                    planning_context, current_parameters, world, robot, milk_body,
                    navigation_map, node_bounding_boxes,
                    robot_start_x=ROBOT_INIT_X, robot_start_y=ROBOT_INIT_Y,
                )
                return_table_approach_x   = current_parameters.table_approach_x
                return_table_approach_y   = current_parameters.table_approach_y
                return_counter_approach_x = current_parameters.counter_approach_x
                return_counter_approach_y = current_parameters.counter_approach_y

            print("\n  Executing plan ...")
            execution_succeeded = False
            with simulated_robot:
                try:
                    plan.perform()
                    execution_succeeded = True
                    print("  Execution complete.")
                except Exception as execution_error:
                    failed_count += 1
                    print(
                        f"  RESULT: FAILED — "
                        f"{type(execution_error).__name__}: {execution_error}"
                    )
                    if current_parameters is not None:
                        _diagnose_and_log(causal_circuit, current_parameters, iteration_number)

            if execution_succeeded:
                try:
                    _persist_plan(database_session, plan)
                    successful_count += 1
                    print(
                        f"  RESULT: SUCCESS  "
                        f"({successful_count}/{iteration_number} stored,  "
                        f"{NUMBER_OF_ITERATIONS - iteration_number} remaining)"
                    )
                except Exception as database_error:
                    print(f"  [db] ERROR: {database_error}")
                    traceback.print_exc()

            print("\n  Resetting ...")
            _respawn_milk(world, milk_body)
            _navigate_back_to_start(
                planning_context, navigation_map, node_bounding_boxes, world,
                table_approach_x=return_table_approach_x,
                table_approach_y=return_table_approach_y,
                counter_approach_x=return_counter_approach_x,
                counter_approach_y=return_counter_approach_y,
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