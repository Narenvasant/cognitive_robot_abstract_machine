"""
Apartment world: JPT-guided pick-and-place with GCS navigation.

Uses the open-world JPT (pick_and_place_jpt.json), trained on 1742 successful
open-world Batch 1 plans, to guide approach-position and arm sampling in the
apartment world. The open-world JPT transfers directly because pick/place mechanics are identical between the two worlds;
only navigation geometry differs, which GCS handles transparently.

JPT variable mapping (open-world → apartment):
    pick_approach_x/y   →  counter_approach_x/y
    place_approach_x/y  →  table_approach_x/y
    milk_end_x/y/z      →  unchanged
    pick_arm            →  unchanged
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

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT as JointProbabilityTree
from jpt.variables import NumericVariable, SymbolicVariable

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

_RESOURCE_PATH: Path   = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF_PATH: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL_PATH:       Path = _RESOURCE_PATH / "objects" / "milk.stl"

JPT_MODEL_PATH:          str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")
JPT_MIN_SAMPLES_PER_LEAF: int = 25


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


@dataclass
class PlanParameters:
    """Sampled parameters for one pick-and-place iteration."""
    counter_approach_x: float
    counter_approach_y: float
    table_approach_x:   float
    table_approach_y:   float
    pick_arm:           Arms


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
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(MILK_SPAWN_X, MILK_SPAWN_Y, MILK_SPAWN_Z, 0, 0, 0)
    with world.modify_world():
        world.add_body(milk_body)
        connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
        world.add_connection(connection)
        connection.origin = spawn_pose
        world.add_semantic_annotation(Milk(root=milk_body))
    return milk_body


def _respawn_milk(world: World, milk_body: Body) -> None:
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(MILK_SPAWN_X, MILK_SPAWN_Y, MILK_SPAWN_Z, 0, 0, 0)
    with world.modify_world():
        connection = milk_body.parent_connection
        if connection is not None and connection.parent is not world.root:
            world.remove_connection(connection)
            new_connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
            world.add_connection(new_connection)
            new_connection.origin = spawn_pose
        elif connection is not None:
            connection.origin = spawn_pose
        else:
            new_connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
            world.add_connection(new_connection)
            new_connection.origin = spawn_pose
    print(f"  [respawn] Milk reset to ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")



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
    for all GCS nodes. Enables vectorised free-space checks that are ~100x
    faster than the O(N) Python loop in GraphOfConvexSets.node_of_point.
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
        for candidate_x, candidate_y in zip(x + radius * np.cos(angles), y + radius * np.sin(angles)):
            if _is_in_free_space(node_array, candidate_x, candidate_y, z):
                print(f"    [GCS] Free point at ({candidate_x:.3f},{candidate_y:.3f}) r={radius:.2f}")
                return Point3(candidate_x, candidate_y, z, reference_frame=world.root)
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
                    position=PyCramVector3(x=float(point.x), y=float(point.y), z=0.0),
                    orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
                ),
                header=Header(frame_id=world_frame),
            ),
            keep_joint_states=keep_joint_states,
        )
        for point in path[1:]
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
    Plan a collision-free path from (start_x, start_y) to (goal_x, goal_y)
    via GCS and return it as NavigateActions. Falls back to a single direct
    NavigateAction if GCS cannot find a path.
    """
    z_navigation    = (GCS_SEARCH_MIN_Z + GCS_SEARCH_MAX_Z) / 2.0
    start_point     = Point3(start_x, start_y, z_navigation, reference_frame=world.root)
    goal_point      = Point3(goal_x,  goal_y,  z_navigation, reference_frame=world.root)
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
    for index, action in enumerate(navigate_actions):
        position = action.target_location.pose.position
        print(f"           waypoint {index + 1}: ({position.x:.3f}, {position.y:.3f})")
    return navigate_actions


def _load_joint_probability_tree(model_path: str) -> JointProbabilityTree:
    print(f"  [jpt] Loading model from {model_path} ...")
    joint_probability_tree = JointProbabilityTree(
        variables=JPT_VARIABLES,
        min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF,
    ).load(model_path)
    print(f"  [jpt] Loaded — {len(joint_probability_tree.leaves)} leaves")
    return joint_probability_tree


def _sample_plan_parameters(joint_probability_tree: JointProbabilityTree) -> PlanParameters:
    """
    Draw one joint sample from the open-world JPT and map it to apartment
    PlanParameters, clipping both approach positions to the apartment zones.
    """
    sample_row     = joint_probability_tree.sample(1)[0]
    sample_by_name = dict(zip([variable.name for variable in JPT_VARIABLES], sample_row))

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
    """Deterministic seed plan for iteration 1 using the right arm and fixed approach positions."""
    arm = Arms.RIGHT

    navigate_to_counter = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=robot_start_x,    start_y=robot_start_y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    navigate_to_table = _navigate_via_gcs(
        context, navigation_map, node_array,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X,    goal_y=TABLE_APPROACH_Y,
        world=world,
        keep_joint_states=True,
    )
    place_pose = _make_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world.root)

    print(f"  [plan] seed — counter:({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})  table:({TABLE_APPROACH_X},{TABLE_APPROACH_Y})  place:({PLACE_X},{PLACE_Y},{PLACE_Z})  arm:{arm}")

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
        world=world,
        keep_joint_states=True,
    )
    place_pose = _make_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world.root)

    print(f"  [plan] sampled — counter:({parameters.counter_approach_x:.3f},{parameters.counter_approach_y:.3f})  table:({parameters.table_approach_x:.3f},{parameters.table_approach_y:.3f})  place:({PLACE_X},{PLACE_Y},{PLACE_Z})  arm:{parameters.pick_arm}")

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

    Both waypoints are GCS-free positions already validated on the forward legs.
    """
    print(f"  [return] table({table_approach_x:.2f},{table_approach_y:.2f}) -> counter({counter_approach_x:.2f},{counter_approach_y:.2f}) -> start({COUNTER_APPROACH_X},{COUNTER_APPROACH_Y})")

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



def pick_and_place_demo_apartment_jpt() -> None:
    """Apartment world Batch 2: 5000 iterations of JPT-guided pick-and-place."""
    print("=" * 64)
    print("  pick_and_place_demo_apartment_jpt  (Batch 2 / JPT-guided)")
    print(f"  Iterations   : {NUMBER_OF_ITERATIONS}")
    print(f"  Place target : ({PLACE_X}, {PLACE_Y}, {PLACE_Z})")
    print(f"  JPT model    : {JPT_MODEL_PATH}")
    print(f"  Database     : {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/5] Building world ...")
    world, robot = _build_world(APARTMENT_URDF_PATH)
    _add_localization_frames(world, robot)
    milk_body = _spawn_milk(world, MILK_STL_PATH)
    print(f"  Milk at ({MILK_SPAWN_X}, {MILK_SPAWN_Y}, {MILK_SPAWN_Z})")
    navigation_map = _build_navigation_map(world)
    node_array     = _build_gcs_node_array(navigation_map)

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

        planning_context = Context(world, robot, None)

        print("\n[5/5] Running iterations ...")
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
                    robot_start_x=ROBOT_INIT_X,
                    robot_start_y=ROBOT_INIT_Y,
                )
                return_table_approach_x,   return_table_approach_y   = TABLE_APPROACH_X,   TABLE_APPROACH_Y
                return_counter_approach_x, return_counter_approach_y = COUNTER_APPROACH_X, COUNTER_APPROACH_Y
            else:
                print("  Mode: JPT-SAMPLED")
                parameters = _sample_plan_parameters(joint_probability_tree)
                plan = _build_sampled_plan(
                    planning_context, parameters, world, robot, milk_body, navigation_map, node_array,
                    robot_start_x=ROBOT_INIT_X,
                    robot_start_y=ROBOT_INIT_Y,
                )
                return_table_approach_x,   return_table_approach_y   = parameters.table_approach_x,   parameters.table_approach_y
                return_counter_approach_x, return_counter_approach_y = parameters.counter_approach_x, parameters.counter_approach_y

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

            if execution_succeeded:
                try:
                    _persist_plan(database_session, plan)
                    successful_count += 1
                    print(f"  RESULT: SUCCESS  ({successful_count}/{iteration} stored,  {NUMBER_OF_ITERATIONS - iteration} remaining)")
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
            row_count = database_session.execute(text('SELECT COUNT(*) FROM "SequentialPlanDAO"')).scalar()
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