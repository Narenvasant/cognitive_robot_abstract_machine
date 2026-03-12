import os
import time
import threading
import copy
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

import rclpy
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped, PyCramPose, PyCramVector3, PyCramQuaternion, Header

# ---- Monkey-patch for Header and PoseStamped deepcopy ------------------------
def _header_deepcopy(self, memo: Any) -> Header:
    """
    Custom deepcopy for Header to handle missing 'stamp' attribute gracefully.

    :param self: The Header instance.
    :param memo: The deepcopy memo dictionary.
    :return: A new Header instance.
    """
    if isinstance(self, type):
        return self
    return Header(
        frame_id=getattr(self, "frame_id", None),
        stamp=copy.deepcopy(getattr(self, "stamp", None), memo),
        sequence=getattr(self, "sequence", 0),
    )


def _pose_stamped_deepcopy(self, memo: Any) -> PoseStamped:
    """
    Custom deepcopy for PoseStamped to use the custom Header deepcopy.

    :param self: The PoseStamped instance.
    :param memo: The deepcopy memo dictionary.
    :return: A new PoseStamped instance.
    """
    if isinstance(self, type):
        return self
    return PoseStamped(
        copy.deepcopy(getattr(self, "pose", None), memo),
        copy.deepcopy(getattr(self, "header", None), memo),
    )

Header.__deepcopy__ = _header_deepcopy
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy
# ------------------------------------------------------------------------------

from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import ParkArmsAction, NavigateAction, PickUpAction, PlaceAction
from pycram.orm.ormatic_interface import *

from krrood.entity_query_language.factories import (
    underspecified,
    variable,
    variable_from,
)
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF, OmniDrive
from semantic_digital_twin.world_description.geometry import FileMesh, BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import GraphOfConvexSets
from semantic_digital_twin.world_description.shape_collection import ShapeCollection, BoundingBoxCollection
from semantic_digital_twin.world_description.world_entity import Body


NUMBER_OF_ITERATIONS: int = 5000
"""
Total number of plan iterations to run.
Iteration 1 uses fixed/known-good values (original behaviour).
Iterations 2..NUMBER_OF_ITERATIONS use probabilistic sampling.
"""

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///:memory:",
)

MILK_X: float = 2.4
"""Initial milk x-coordinate."""

MILK_Y: float = 2.5
"""Initial milk y-coordinate."""

MILK_Z: float = 1.01
"""Initial milk z-coordinate."""

COUNTER_APPROACH_X: float = 1.6
"""Target robot x-coordinate when approaching the counter."""

COUNTER_APPROACH_Y: float = 2.5
"""Target robot y-coordinate when approaching the counter."""

TABLE_APPROACH_X: float = 4.2
"""Target robot x-coordinate when approaching the table."""

TABLE_APPROACH_Y: float = 4.0
"""Target robot y-coordinate when approaching the table."""

PLACE_X: float = 5.0
"""Target x-coordinate for placing the milk."""

PLACE_Y: float = 4.0
"""Target y-coordinate for placing the milk."""

PLACE_Z: float = 0.80
"""Target z-coordinate for placing the milk."""


GRAPH_OF_CONVEX_SETS_MIN_X: float = -1.0
GRAPH_OF_CONVEX_SETS_MAX_X: float = 7.0
GRAPH_OF_CONVEX_SETS_MIN_Y: float = -1.0
GRAPH_OF_CONVEX_SETS_MAX_Y: float = 7.0

GRAPH_OF_CONVEX_SETS_MIN_Z: float = 0.0
GRAPH_OF_CONVEX_SETS_MAX_Z: float = 0.1

GRAPH_OF_CONVEX_SETS_BLOAT_OBSTACLES: float = 0.3
GRAPH_OF_CONVEX_SETS_BLOAT_WALLS: float = 0.05

_RESOURCE_PATH = Path(__file__).resolve().parents[3] / "resources"
APARTMENT_URDF: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL:       Path = _RESOURCE_PATH / "objects" / "milk.stl"

_ACTION_LABELS = [
    "ParkArms (pre)",
    "Navigate -> counter",
    "PickUp milk",
    "Navigate -> table",
    "Place milk",
    "ParkArms (post)",
]


@dataclass
class ActionEntry:
    """
    Groups an underspecified action description with its UnderspecifiedParameters
    and the pre-built ProbabilisticCircuit so that all three travel together.
    """
    description: Any
    parameters: UnderspecifiedParameters
    distribution: ProbabilisticCircuit


def _build_navigation_map(world: World) -> GraphOfConvexSets:
    """
    Construct the GCS navigation map from the world obstacles.

    :param world: The world instance.
    :return: The generated GraphOfConvexSets.
    """
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
    graph_of_convex_sets = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=GRAPH_OF_CONVEX_SETS_BLOAT_OBSTACLES,
    )
    elapsed = time.time() - start_time
    print(
        f"  GraphOfConvexSets built in {elapsed:.2f} s  "
        f"({len(list(graph_of_convex_sets.graph.nodes()))} nodes)"
    )
    return graph_of_convex_sets


def _gcs_collision_free_path(
    graph_of_convex_sets: GraphOfConvexSets,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    world: World,
) -> Optional[List[Point3]]:
    """
    Find a collision-free path between two points using GraphOfConvexSets.

    :param graph_of_convex_sets: The navigation map.
    :param start_x: Starting x coordinate.
    :param start_y: Starting y coordinate.
    :param goal_x: Goal x coordinate.
    :param goal_y: Goal y coordinate.
    :param world: The world instance.
    :return: A list of points forming the path, or None if no path found.
    """
    z_navigation = (GRAPH_OF_CONVEX_SETS_MIN_Z + GRAPH_OF_CONVEX_SETS_MAX_Z) / 2.0
    start = Point3(start_x, start_y, z_navigation, reference_frame=world.root)
    goal = Point3(goal_x, goal_y, z_navigation, reference_frame=world.root)
    try:
        path = graph_of_convex_sets.path_from_to(start, goal)
    except Exception as exception:
        print(f"    [GraphOfConvexSets] path_from_to raised: {exception}")
        return None
    return path


def _gcs_path_to_pose_stamped_list(
    path: List[Point3],
    world_frame: Any,
) -> List[PoseStamped]:
    """
    Convert a list of Point3 path waypoints to a list of PoseStamped.

    :param path: The list of points in the path.
    :param world_frame: The reference frame for the poses.
    :return: A list of PoseStamped objects.
    """
    poses = []
    for point in path[1:]:
        pose_stamped = PoseStamped(
            pose=PyCramPose(
                position=PyCramVector3(x=float(point.x), y=float(point.y), z=0.0),
                orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
            ),
            header=Header(frame_id=world_frame),
        )
        poses.append(pose_stamped)
    return poses


def _initialize_world(apartment_unified_robot_description_format: Path) -> tuple:
    """
    Initialize the simulation world with the apartment and PR2 robot.

    :param apartment_unified_robot_description_format: Path to the apartment URDF.
    :return: A tuple of (World, PR2).
    """
    world = URDFParser.from_file(str(apartment_unified_robot_description_format)).parse()
    pr2_unified_robot_description_format = (
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    )
    pr2_world = URDFParser.from_file(pr2_unified_robot_description_format).parse()
    robot = PR2.from_world(pr2_world)
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 1.5, 0.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(pr2_world, robot_pose)
    table_body = world.get_body_by_name("table_area_main")
    with world.modify_world():
        table = Table(root=table_body)
        world.add_semantic_annotation(table)
    return world, robot


def _add_localization_frames(world: World, robot: PR2) -> None:
    """
    Add necessary localization frames (map, odom) to the world.

    :param world: The world instance.
    :param robot: The PR2 robot instance.
    """
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_body(map_body)
        world.add_body(odom_body)
        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))
        old_connection = robot.root.parent_connection
        if old_connection is not None:
            world.remove_connection(old_connection)
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


def _add_milk_object(
    world: World, stl_path: Path
) -> tuple[Body, HomogeneousTransformationMatrix]:
    """
    Add the milk object to the world at its initial location.

    :param world: The world instance.
    :param stl_path: Path to the milk object's STL file.
    :return: A tuple containing the milk body and its initial pose.
    """
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
    return body, pose


def _create_pose_stamped(
    x_coordinate: float, y_coordinate: float, z_coordinate: float, frame_id: Any
) -> PoseStamped:
    """
    Create a PoseStamped object with the given coordinates.

    :param x_coordinate: The x coordinate.
    :param y_coordinate: The y coordinate.
    :param z_coordinate: The z coordinate.
    :param frame_id: The reference frame id.
    :return: A PoseStamped instance.
    """
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x_coordinate, y=y_coordinate, z=z_coordinate),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame_id),
    )


def _respawn_milk_object(world: World, milk_body: Body) -> None:
    """
    Reset the milk object to its initial spawn location.

    :param world: The world instance.
    :param milk_body: The milk body instance.
    """
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
    print(f"  Milk respawned at  x={MILK_X}, y={MILK_Y}, z={MILK_Z}")


def _create_database_session(database_uri: str) -> Session:
    """
    Create a SQLAlchemy session and initialize the database.

    :param database_uri: The database connection URI.
    :return: A SQLAlchemy session.
    """
    engine = create_engine(database_uri)
    if "postgresql" in database_uri or "postgres" in database_uri:
        _apply_postgresql_patches(engine)
    session = Session(engine)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    return session


def _apply_postgresql_patches(engine) -> None:
    _patch_identifier_validation(engine)
    _shorten_metadata_table_names()
    _register_numpy_coercion(engine)


def _patch_identifier_validation(engine) -> None:
    engine.dialect.validate_identifier = lambda _ident: None


def _shorten_metadata_table_names() -> None:
    import hashlib
    def _shorten(name: str, max_len: int = 63) -> str:
        if len(name) <= max_len:
            return name
        suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:max_len - 9]}_{suffix}"
    for table in Base.metadata.tables.values():
        short = _shorten(table.name)
        if short != table.name:
            print(f"  [db] identifier shortened: '{table.name}' -> '{short}'")
            table.name = short
            table.fullname = short


def _register_numpy_coercion(engine) -> None:
    import numpy as np
    from sqlalchemy import event

    def _coerce(value):
        if isinstance(value, np.floating):  return float(value)
        if isinstance(value, np.integer):   return int(value)
        if isinstance(value, np.bool_):     return bool(value)
        return value

    def _coerce_params(params):
        if isinstance(params, dict):
            return {k: _coerce(v) for k, v in params.items()}
        return params

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _coerce_numpy_params(conn, cursor, statement, parameters, context, executemany):
        if isinstance(parameters, dict):
            parameters = _coerce_params(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(_coerce_params(p) for p in parameters)
        return statement, parameters


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    plan_dao = to_dao(plan)
    session.add(plan_dao)
    session.commit()


def _navigate_via_graph_of_convex_sets(
    context: Context,
    graph_of_convex_sets: GraphOfConvexSets,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    world: World,
    keep_joint_states: bool = False,
) -> List[NavigateAction]:
    """
    Generate a sequence of NavigateActions following a GCS path.

    :param context: The planning context.
    :param graph_of_convex_sets: The GCS navigation map.
    :param start_x: Start x.
    :param start_y: Start y.
    :param goal_x: Goal x.
    :param goal_y: Goal y.
    :param world: The world instance.
    :param keep_joint_states: Whether to keep joint states during navigation.
    :return: A list of NavigateAction instances.
    """
    path = _gcs_collision_free_path(
        graph_of_convex_sets, start_x, start_y, goal_x, goal_y, world
    )
    if path is None or len(path) < 2:
        print(
            f"    [GraphOfConvexSets] No path found "
            f"({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}). "
            f"Falling back to direct navigation."
        )
        return [
            NavigateAction(
                target_location=_create_pose_stamped(goal_x, goal_y, 0.0, world.root),
                keep_joint_states=keep_joint_states,
            )
        ]
    waypoint_poses = _gcs_path_to_pose_stamped_list(path, world.root)
    print(
        f"    [GraphOfConvexSets] Path ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): "
        f"{len(waypoint_poses)} waypoint(s)"
    )
    for i, pose in enumerate(waypoint_poses):
        position = pose.pose.position
        print(f"           waypoint {i+1}: ({position.x:.3f}, {position.y:.3f})")
    return [
        NavigateAction(target_location=pose_stamped, keep_joint_states=keep_joint_states)
        for pose_stamped in waypoint_poses
    ]


# Fixed plan (iteration 1)

def _build_fixed_plan(
    context: Context,
    world: World,
    robot: PR2,
    milk_body: Body,
    graph_of_convex_sets: GraphOfConvexSets,
    robot_start_x: float = 1.4,
    robot_start_y: float = 1.5,
) -> SequentialPlan:
    """
    Construct a fixed SequentialPlan for the first iteration.

    :param context: The planning context.
    :param world: The world instance.
    :param robot: The PR2 robot instance.
    :param milk_body: The milk body instance.
    :param graph_of_convex_sets: The GCS navigation map.
    :param robot_start_x: Initial robot x.
    :param robot_start_y: Initial robot y.
    :return: A SequentialPlan instance.
    """
    world_frame = world.root
    nav_to_counter = _navigate_via_graph_of_convex_sets(
        context,
        graph_of_convex_sets,
        start_x=robot_start_x,
        start_y=robot_start_y,
        goal_x=COUNTER_APPROACH_X,
        goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    nav_to_table = _navigate_via_graph_of_convex_sets(
        context,
        graph_of_convex_sets,
        start_x=COUNTER_APPROACH_X,
        start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X,
        goal_y=TABLE_APPROACH_Y,
        world=world,
    )
    place_target_pose = _create_pose_stamped(
        PLACE_X, PLACE_Y, PLACE_Z, world_frame
    )
    actions = (
        [ParkArmsAction(arm=Arms.BOTH)]
        + nav_to_counter
        + [
            PickUpAction(
                object_designator=milk_body,
                arm=Arms.RIGHT,
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
                arm=Arms.RIGHT,
            ),
            ParkArmsAction(arm=Arms.BOTH),
        ]
    )
    return SequentialPlan(context, *actions)


# Probabilistic plan (iterations 2+)

def _navigable_pose_description(robot: PR2) -> Any:
    """
    Build an underspecified PoseStamped description for a navigation target.

    :param robot: The PR2 robot instance.
    :return: An underspecified PoseStamped description.
    """
    return underspecified(PoseStamped)(
        pose=underspecified(PyCramPose)(
            position=underspecified(PyCramVector3)(x=..., y=..., z=0),
            orientation=underspecified(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=underspecified(Header)(frame_id=variable_from([robot._world.root])),
    )


def _place_pose_description(robot: PR2) -> Any:
    """
    Build an underspecified PoseStamped for the place target.

    :param robot: The PR2 robot instance.
    :return: An underspecified PoseStamped description.
    """
    return underspecified(PoseStamped)(
        pose=underspecified(PyCramPose)(
            position=underspecified(PyCramVector3)(x=..., y=..., z=PLACE_Z),
            orientation=underspecified(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=underspecified(Header)(frame_id=variable_from([robot._world.root])),
    )


def _build_action_descriptions(
    world: World, robot: PR2, milk_variable: Any
) -> list:
    """
    Construct the sequence of underspecified action descriptions used for
    probabilistic reasoning.

    :param world: The world instance.
    :param robot: The PR2 robot instance.
    :param milk_variable: The variable representing the milk object.
    :return: A list of underspecified action descriptions.
    """
    manipulators = world.get_semantic_annotations_by_type(Manipulator)
    return [
        underspecified(ParkArmsAction)(
            arm=variable(Arms, [Arms.BOTH]),
        ),
        underspecified(NavigateAction)(
            target_location=_navigable_pose_description(robot),
            keep_joint_states=False,
        ),
        underspecified(PickUpAction)(
            object_designator=milk_variable,
            arm=variable(Arms, [Arms.LEFT, Arms.RIGHT]),
            grasp_description=underspecified(GraspDescription)(
                approach_direction=variable(
                    ApproachDirection,
                    [ApproachDirection.FRONT],
                ),
                vertical_alignment=variable(
                    VerticalAlignment,
                    [VerticalAlignment.NoAlignment],
                ),
                rotate_gripper=variable(bool, [False]),
                manipulation_offset=0.06,
                manipulator=variable(Manipulator, manipulators),
            ),
        ),
        underspecified(NavigateAction)(
            target_location=_navigable_pose_description(robot),
            keep_joint_states=False,
        ),
        underspecified(PlaceAction)(
            object_designator=milk_variable,
            target_location=_place_pose_description(robot),
            arm=variable(Arms, [Arms.LEFT, Arms.RIGHT]),
        ),
        underspecified(ParkArmsAction)(
            arm=variable(Arms, [Arms.BOTH]),
        ),
    ]


_COUNTER_APPROACH_BOUNDS = (1.2, 1.8, 2.3, 2.7)
"""Sampling bounds for robot approach to the kitchen counter."""

_TABLE_APPROACH_BOUNDS = (4.1, 4.5, 3.8, 4.2)
"""Sampling bounds for robot approach to the table."""


def _truncate_navigate_distribution(
    distribution: ProbabilisticCircuit,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> ProbabilisticCircuit:
    """
    Truncate a NavigateAction distribution within the given x and y bounds.

    :param distribution: The fully-factorised circuit for a NavigateAction.
    :param x_min: World-frame x minimum bound.
    :param x_max: World-frame x maximum bound.
    :param y_min: World-frame y minimum bound.
    :param y_max: World-frame y maximum bound.
    :return: The truncated circuit or the original if truncation fails.
    """
    from random_events.interval import closed
    from random_events.product_algebra import SimpleEvent
    from random_events.variable import Continuous

    all_names = [variable.name for variable in distribution.variables]
    print(f"    [truncate] distribution variables: {all_names}")

    x_variable = None
    y_variable = None
    for variable in distribution.variables:
        if isinstance(variable, Continuous):
            if variable.name.endswith(".position.x") or variable.name.endswith(".x"):
                x_variable = variable
            elif variable.name.endswith(".position.y") or variable.name.endswith(".y"):
                y_variable = variable

    if x_variable is None or y_variable is None:
        print(
            f"    [truncate] WARNING: could not find position x/y variables in "
            f"distribution (vars={[variable.name for variable in distribution.variables]}). "
            f"Skipping truncation."
        )
        return distribution

    position_event = SimpleEvent(
        {
            x_variable: closed(x_min, x_max),
            y_variable: closed(y_min, y_max),
        }
    ).as_composite_set()

    full_event = position_event.fill_missing_variables_pure(distribution.variables)

    truncated, log_probability = distribution.log_truncated_in_place(full_event)
    if truncated is None:
        print(
            f"    [truncate] WARNING: zero-probability region "
            f"x=[{x_min},{x_max}] y=[{y_min},{y_max}]. "
            f"Keeping untruncated distribution."
        )
        return distribution

    print(
        f"    [truncate] x=[{x_min},{x_max}] y=[{y_min},{y_max}]  "
        f"log_p={log_probability:.3f}"
    )
    return truncated


def _build_action_entry(
    description: Any, approach_bounds: tuple = None
) -> ActionEntry:
    """
    Translate an underspecified description into a concrete ActionEntry.

    Replaces the old MatchParameterizer / MatchToInstanceTranslator pipeline
    with the new UnderspecifiedParameters API.

    :param description: The underspecified action description.
    :param approach_bounds: Optional (x_min, x_max, y_min, y_max) tuple.
    :return: An ActionEntry instance.
    """
    description.resolve()
    parameters = UnderspecifiedParameters(description)
    distribution = fully_factorized(parameters.variables.values())

    if approach_bounds is not None:
        x_min, x_max, y_min, y_max = approach_bounds
        distribution = _truncate_navigate_distribution(
            distribution, x_min, x_max, y_min, y_max
        )

    return ActionEntry(description, parameters, distribution)


def _apply_sample(entry: ActionEntry) -> Any:
    """
    Sample parameters from the entry's distribution and construct a concrete
    action instance using the new UnderspecifiedParameters API.

    Replaces the old _apply_sample / parameterize_object_with_sample pattern.

    :param entry: The action entry to sample from.
    :return: A concrete action instance with sampled parameter values applied.
    """
    raw_sample = entry.distribution.sample(1)[0]
    return entry.parameters.create_instance_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )


def _build_sampled_plan_with_graph_of_convex_sets(
    context: Context,
    entries: list[ActionEntry],
    graph_of_convex_sets: GraphOfConvexSets,
    world: World,
    robot_start_x: float,
    robot_start_y: float,
) -> SequentialPlan:
    """
    Sample fresh parameters for every action entry and rewrite navigation paths.

    Action entry layout (matches _build_action_descriptions):
      0  ParkArmsAction  (pre)
      1  NavigateAction  -> counter area  [truncated to counter approach zone]
      2  PickUpAction
      3  NavigateAction  -> table area    [truncated to table approach zone]
      4  PlaceAction
      5  ParkArmsAction  (post)

    :param context: The planning context.
    :param entries: The list of action entries.
    :param graph_of_convex_sets: The GCS navigation map.
    :param world: The world instance.
    :param robot_start_x: Start x.
    :param robot_start_y: Start y.
    :return: A SequentialPlan instance.
    """
    sampled_instances = []
    for label, entry in zip(_ACTION_LABELS, entries):
        instance = _apply_sample(entry)
        sampled_instances.append(instance)
        print(f"    Sampled  {label}")

    nav_to_counter_instance: NavigateAction = sampled_instances[1]
    nav_to_table_instance: NavigateAction = sampled_instances[3]

    sampled_counter_pos = nav_to_counter_instance.target_location.pose.position
    sampled_table_pos = nav_to_table_instance.target_location.pose.position

    counter_goal_x = float(sampled_counter_pos.x)
    counter_goal_y = float(sampled_counter_pos.y)
    table_goal_x = float(sampled_table_pos.x)
    table_goal_y = float(sampled_table_pos.y)

    print(
        f"    Sampled counter goal: ({counter_goal_x:.3f}, {counter_goal_y:.3f})"
        f"  |  table goal: ({table_goal_x:.3f}, {table_goal_y:.3f})"
    )

    nav_to_counter_actions = _navigate_via_graph_of_convex_sets(
        context,
        graph_of_convex_sets,
        start_x=robot_start_x,
        start_y=robot_start_y,
        goal_x=counter_goal_x,
        goal_y=counter_goal_y,
        world=world,
    )
    nav_to_table_actions = _navigate_via_graph_of_convex_sets(
        context,
        graph_of_convex_sets,
        start_x=counter_goal_x,
        start_y=counter_goal_y,
        goal_x=table_goal_x,
        goal_y=table_goal_y,
        world=world,
    )

    actions = (
        [sampled_instances[0]]
        + nav_to_counter_actions
        + [sampled_instances[2]]
        + nav_to_table_actions
        + [sampled_instances[4]]
        + [sampled_instances[5]]
    )
    return SequentialPlan(context, *actions)


def sequential_plan_with_apartment() -> None:
    """
    Main entry point for the apartment pick-and-place probabilistic reasoning demo.
    """
    print("Building world ...")
    world, robot = _initialize_world(APARTMENT_URDF)
    _add_localization_frames(world, robot)
    milk_body, milk_pose = _add_milk_object(world, MILK_STL)

    print(f"  Milk spawned at  x={MILK_X:.3f},  y={MILK_Y:.3f},  z={MILK_Z:.3f}")
    print(f"  Place target at  x={PLACE_X:.3f}, y={PLACE_Y:.3f}, z={PLACE_Z:.3f}")

    graph_of_convex_sets = _build_navigation_map(world)

    session = _create_database_session(DATABASE_URI)
    print(f"  Database: {DATABASE_URI}")

    rclpy.init()
    ros_node = rclpy.create_node("pycram_graph_of_convex_sets_plan_demo")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    robot_init_x: float = 1.4
    robot_init_y: float = 1.5

    try:
        _transformation_publisher = TFPublisher(_world=world, node=ros_node)
        _visualization_publisher = VizMarkerPublisher(_world=world, node=ros_node)

        context = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        print("\nBuilding per-action probabilistic distributions ...")
        descriptions = _build_action_descriptions(world, robot, milk_variable)

        entries: list[ActionEntry] = [
            _build_action_entry(descriptions[0]),
            _build_action_entry(descriptions[1], _COUNTER_APPROACH_BOUNDS),
            _build_action_entry(descriptions[2]),
            _build_action_entry(descriptions[3], _TABLE_APPROACH_BOUNDS),
            _build_action_entry(descriptions[4]),
            _build_action_entry(descriptions[5]),
        ]

        for label, entry in zip(_ACTION_LABELS, entries):
            n_vars = len(entry.parameters.variables)
            n_dist = len(entry.distribution.variables)
            print(
                f"  {label:<25}  {n_vars:>2} param vars  /  {n_dist:>2} distribution vars"
            )

        successful_count = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            separator = "=" * 64
            print(f"\n{separator}")

            if iteration == 1:
                print(
                    f"  Iteration {iteration:>3} / {NUMBER_OF_ITERATIONS}  [FIXED + GCS navigation]"
                )
                print(separator)
                plan = _build_fixed_plan(
                    context,
                    world,
                    robot,
                    milk_body,
                    graph_of_convex_sets,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )
            else:
                print(
                    f"  Iteration {iteration:>3} / {NUMBER_OF_ITERATIONS}  [SAMPLED + GCS navigation]"
                )
                print(separator)
                plan = _build_sampled_plan_with_graph_of_convex_sets(
                    context,
                    entries,
                    graph_of_convex_sets,
                    world,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )

            with simulated_robot:
                try:
                    plan.perform()
                    _persist_plan(session, plan)
                    successful_count += 1
                    print(
                        f"\n  v  Iteration {iteration} succeeded -- plan persisted to database."
                    )
                except Exception as exception:
                    import traceback
                    traceback.print_exc()
                    print(
                        f"\n  x  Iteration {iteration} failed -- plan not stored.  "
                        f"({type(exception).__name__}: {exception})"
                    )
                finally:
                    _respawn_milk_object(world, milk_body)

        print(f"\n{'=' * 64}")
        print(
            f"Done.  {successful_count} / {NUMBER_OF_ITERATIONS} plans stored in '{DATABASE_URI}'."
        )

    finally:
        session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    sequential_plan_with_apartment()