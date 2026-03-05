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
def _header_deepcopy(self, memo):
    if isinstance(self, type):
        return self
    return Header(
        frame_id=getattr(self, "frame_id", None),
        stamp=copy.deepcopy(getattr(self, "stamp", None), memo),
        sequence=getattr(self, "sequence", 0),
    )

def _pose_stamped_deepcopy(self, memo):
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
    probable,
    probable_variable,
    variable,
    variable_from,
)
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine
from krrood.probabilistic_knowledge.parameterizer import MatchParameterizer, Parameterization
from krrood.probabilistic_knowledge.probable_variable import MatchToInstanceTranslator
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit

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
MILK_Y: float = 2.5
MILK_Z: float = 1.01

COUNTER_APPROACH_X: float = 1.6
COUNTER_APPROACH_Y: float = 2.5

TABLE_APPROACH_X: float = 4.2
TABLE_APPROACH_Y: float = 4.0

PLACE_X: float = 5.0
PLACE_Y: float = 4.0
PLACE_Z: float = 0.80

# Search space bounds for the navigation map (covers the apartment floor area).
# Adjust these to match your apartment URDF extents if needed.
GRAPH_OF_CONVEX_SETS_MIN_X: float = -1.0
GRAPH_OF_CONVEX_SETS_MAX_X: float = 7.0
GRAPH_OF_CONVEX_SETS_MIN_Y: float = -1.0
GRAPH_OF_CONVEX_SETS_MAX_Y: float = 7.0
# Navigation is 2-D: z-slice at floor level.  A thin slab (0.0 -> 0.1) is
# enough to build a floor-plan navigation map.
GRAPH_OF_CONVEX_SETS_MIN_Z: float = 0.0
GRAPH_OF_CONVEX_SETS_MAX_Z: float = 0.1

# Robot footprint bloat applied to obstacles so the robot body is accounted for.
GRAPH_OF_CONVEX_SETS_BLOAT_OBSTACLES: float = 0.3
GRAPH_OF_CONVEX_SETS_BLOAT_WALLS: float = 0.05

_RESOURCE_PATH = Path(__file__).resolve().parents[2] / "resources"
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
    Groups a concrete action instance with its Parameterization and the
    pre-built ProbabilisticCircuit so that all three travel together.
    """
    instance: Any
    parameterization: Parameterization
    distribution: ProbabilisticCircuit



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

    print("  Building GCS navigation map ...")
    t0 = time.time()
    gcs = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=GRAPH_OF_CONVEX_SETS_BLOAT_OBSTACLES,
    )
    elapsed = time.time() - t0
    print(f"  GCS built in {elapsed:.2f} s  ({len(list(gcs.graph.nodes()))} nodes)")
    return gcs


def _gcs_collision_free_path(
    gcs: GraphOfConvexSets,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    world: World,
) -> Optional[List[Point3]]:
    z_nav = (GRAPH_OF_CONVEX_SETS_MIN_Z + GRAPH_OF_CONVEX_SETS_MAX_Z) / 2.0
    start = Point3(start_x, start_y, z_nav, reference_frame=world.root)
    goal  = Point3(goal_x,  goal_y,  z_nav, reference_frame=world.root)
    try:
        path = gcs.path_from_to(start, goal)
    except Exception as exc:
        print(f"    [GCS] path_from_to raised: {exc}")
        return None
    return path


def _gcs_path_to_pose_stamped_list(
    path: List[Point3],
    world_frame,
) -> List[PoseStamped]:
    poses = []
    for point in path[1:]:
        ps = PoseStamped(
            pose=PyCramPose(
                position=PyCramVector3(x=float(point.x), y=float(point.y), z=0.0),
                orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
            ),
            header=Header(frame_id=world_frame),
        )
        poses.append(ps)
    return poses


def _build_world(apartment_urdf: Path) -> tuple:
    world = URDFParser.from_file(str(apartment_urdf)).parse()
    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    pr2_world = URDFParser.from_file(pr2_urdf).parse()
    robot = PR2.from_world(pr2_world)
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 1.5, 0.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(pr2_world, robot_pose)
    table_body = world.get_body_by_name("table_area_main")
    with world.modify_world():
        table = Table(root=table_body)
        world.add_semantic_annotation(table)
    return world, robot


def _add_localization_frame(world, robot) -> None:
    with world.modify_world():
        map_body  = Body(name=PrefixedName("map"))
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


def _add_milk(world: World, stl_path: Path) -> tuple[Body, HomogeneousTransformationMatrix]:
    mesh = FileMesh.from_file(str(stl_path))
    body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(MILK_X, MILK_Y, MILK_Z, 0, 0, 0)
    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(parent=world.root, child=body, world=world)
        world.add_connection(connection)
        connection.origin = pose
        world.add_semantic_annotation(Milk(root=body))
    return body, pose


def _create_pose_stamped(x: float, y: float, z: float, frame) -> PoseStamped:
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame),
    )


def _respawn_milk(world, milk_body: Body) -> None:
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_X, MILK_Y, MILK_Z, 0, 0, 0
    )
    with world.modify_world():
        connection = milk_body.parent_connection
        if connection is not None:
            if connection.parent is not world.root:
                world.remove_connection(connection)
                new_conn = Connection6DoF.create_with_dofs(
                    parent=world.root, child=milk_body, world=world
                )
                world.add_connection(new_conn)
                new_conn.origin = spawn_pose
            else:
                connection.origin = spawn_pose
        else:
            new_conn = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(new_conn)
            new_conn.origin = spawn_pose
    print(f"  Milk respawned at  x={MILK_X}, y={MILK_Y}, z={MILK_Z}")



def _create_session(database_uri: str) -> Session:
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



def _navigate_via_gcs(
    context: Context,
    gcs: GraphOfConvexSets,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    world: World,
    keep_joint_states: bool = False,
) -> List[NavigateAction]:
    path = _gcs_collision_free_path(gcs, start_x, start_y, goal_x, goal_y, world)
    if path is None or len(path) < 2:
        print(
            f"    [GCS] No path found "
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
        f"    [GCS] Path ({start_x:.2f},{start_y:.2f}) -> ({goal_x:.2f},{goal_y:.2f}): "
        f"{len(waypoint_poses)} waypoint(s)"
    )
    for i, pose in enumerate(waypoint_poses):
        p = pose.pose.position
        print(f"           waypoint {i+1}: ({p.x:.3f}, {p.y:.3f})")
    return [
        NavigateAction(target_location=ps, keep_joint_states=keep_joint_states)
        for ps in waypoint_poses
    ]


# Fixed plan (iteration 1)

def _build_fixed_plan(
    context: Context,
    world: World,
    robot,
    milk_body: Body,
    gcs: GraphOfConvexSets,
    robot_start_x: float = 1.4,
    robot_start_y: float = 1.5,
) -> SequentialPlan:
    world_frame = world.root
    nav_to_counter = _navigate_via_gcs(
        context, gcs,
        start_x=robot_start_x, start_y=robot_start_y,
        goal_x=COUNTER_APPROACH_X, goal_y=COUNTER_APPROACH_Y,
        world=world,
    )
    nav_to_table = _navigate_via_gcs(
        context, gcs,
        start_x=COUNTER_APPROACH_X, start_y=COUNTER_APPROACH_Y,
        goal_x=TABLE_APPROACH_X, goal_y=TABLE_APPROACH_Y,
        world=world,
    )
    place_target_pose = _create_pose_stamped(PLACE_X, PLACE_Y, PLACE_Z, world_frame)
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
    Build a probable PoseStamped description for a navigation target with
    free x/y position components for probabilistic sampling.
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=0),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot._world.root])),
    )


def _place_pose_description(robot) -> Any:
    """
    Build a probable PoseStamped for the place target with free x/y and
    fixed z (table surface height).
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=PLACE_Z),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot._world.root])),
    )


def _build_action_descriptions(world, robot, milk_variable) -> list:
    manipulators = world.get_semantic_annotations_by_type(Manipulator)
    return [
        probable_variable(ParkArmsAction)(
            arm=variable(Arms, [Arms.BOTH]),
        ),
        probable_variable(NavigateAction)(
            target_location=_navigable_pose_description(robot),
            keep_joint_states=False,
        ),
        probable_variable(PickUpAction)(
            object_designator=milk_variable,
            arm=variable(Arms, [Arms.LEFT, Arms.RIGHT]),
            grasp_description=probable(GraspDescription)(
                approach_direction=variable(
                    ApproachDirection,
                    [
                        ApproachDirection.FRONT,
                    ],
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
        probable_variable(NavigateAction)(
            target_location=_navigable_pose_description(robot),
            keep_joint_states=False
        ),
        probable_variable(PlaceAction)(
            object_designator=milk_variable,
            target_location=_place_pose_description(robot),
            arm=variable(Arms, [Arms.LEFT, Arms.RIGHT]),
        ),
        probable_variable(ParkArmsAction)(
            arm=variable(Arms, [Arms.BOTH]),
        ),
    ]


# ---- Approach bounds (world frame, derived from apartment URDF) --------------
#
# island_countertop world XY: apartment_root -> kitchen_root(-0.03,0.75)
#   -> side_B(rpy=pi, xyz=(2.91,3.7)) -> island_countertop(0.13,1.786)
#   => centre ~(2.747, 2.664), floor footprint x=[2.295,3.200], y=[1.113,4.215]
#
# The robot approaches the island from the LEFT (x < 2.295).
# Known-good approach: (1.6, 2.5). Apartment walls: x=[0,7], y=[0,7] approx.
#
# Counter approach zone: x=[0.5, 2.2], y=[1.0, 4.3]  (open floor left of island)
# Table approach zone:   x=[3.0, 4.8], y=[2.5, 5.5]  (open floor in front of table)

# (x_min, x_max, y_min, y_max) — world frame, floor level
_COUNTER_APPROACH_BOUNDS = (1.2, 1.8, 2.3, 2.7)
_TABLE_APPROACH_BOUNDS   = (4.1, 4.5, 3.8, 4.2)


def _truncate_navigate_distribution(
    distribution: ProbabilisticCircuit,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> ProbabilisticCircuit:
    """
    Truncate a NavigateAction distribution so that the sampled (x, y) of the
    target_location stays within [x_min, x_max] x [y_min, y_max].

    Previous attempts built a ``random_events.Event`` using ``SpatialVariables``
    objects, which are different Python objects from the ``Continuous`` variables
    inside the circuit (named e.g. ``NavigateAction.target_location.pose.position.x``).
    The mismatch caused truncation to silently have zero effect.

    This function avoids the mismatch by:
      1. Walking the distribution leaves to find the *actual* x and y variable
         objects by inspecting their ``.name`` suffix.
      2. Building the ``SimpleEvent`` with those exact objects so the variable
         equality check inside ``log_truncated_of_simple_event_in_place`` succeeds.
      3. Extending the event to all distribution variables so non-spatial leaves
         (keep_joint_states, orientation, frame_id ...) are left unconstrained.

    :param distribution: The fully-factorised circuit for a NavigateAction.
    :param x_min, x_max: World-frame x bounds for the navigation goal.
    :param y_min, y_max: World-frame y bounds for the navigation goal.
    :return: The truncated circuit (same object, modified in-place), or the
             original if truncation fails.
    """
    from random_events.interval import closed
    from random_events.product_algebra import SimpleEvent
    from random_events.variable import Continuous


    all_names = [v.name for v in distribution.variables]
    print(f"    [truncate] distribution variables: {all_names}")

    x_var = None
    y_var = None
    for var in distribution.variables:
        if isinstance(var, Continuous):
            if var.name.endswith(".position.x") or var.name.endswith(".x"):
                x_var = var
            elif var.name.endswith(".position.y") or var.name.endswith(".y"):
                y_var = var

    if x_var is None or y_var is None:
        print(
            f"    [truncate] WARNING: could not find position x/y variables in "
            f"distribution (vars={[v.name for v in distribution.variables]}). "
            f"Skipping truncation."
        )
        return distribution


    position_event = SimpleEvent({
        x_var: closed(x_min, x_max),
        y_var: closed(y_min, y_max),
    }).as_composite_set()

    full_event = position_event.fill_missing_variables_pure(distribution.variables)

    truncated, log_prob = distribution.log_truncated_in_place(full_event)
    if truncated is None:
        print(
            f"    [truncate] WARNING: zero-probability region "
            f"x=[{x_min},{x_max}] y=[{y_min},{y_max}]. "
            f"Keeping untruncated distribution."
        )
        return distribution

    print(
        f"    [truncate] x=[{x_min},{x_max}] y=[{y_min},{y_max}]  "
        f"log_p={log_prob:.3f}"
    )
    return truncated


def _build_action_entry(description, approach_bounds: tuple = None) -> ActionEntry:
    """
    Translate a probable_variable match description into a concrete action
    instance, derive its Parameterization, and pre-build the
    fully-factorized ProbabilisticCircuit.

    :param description:     The probable_variable match description.
    :param approach_bounds: Optional (x_min, x_max, y_min, y_max) tuple in
                            world-frame coordinates.  When provided the
                            distribution's position x/y leaves are truncated to
                            this rectangle so sampled navigation goals stay
                            within the specified floor area.  Pass
                            ``_COUNTER_APPROACH_BOUNDS`` for the counter
                            NavigateAction and ``_TABLE_APPROACH_BOUNDS`` for
                            the table NavigateAction.
    """
    instance         = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution     = parameterization.create_fully_factorized_distribution()

    if approach_bounds is not None:
        x_min, x_max, y_min, y_max = approach_bounds
        distribution = _truncate_navigate_distribution(
            distribution, x_min, x_max, y_min, y_max
        )

    return ActionEntry(instance, parameterization, distribution)


def _apply_sample(entry: ActionEntry) -> None:
    raw_sample   = entry.distribution.sample(1)[0]
    named_sample = entry.parameterization.create_assignment_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )
    entry.parameterization.parameterize_object_with_sample(entry.instance, named_sample)


def _build_sampled_plan_with_gcs(
    context: Context,
    entries: list[ActionEntry],
    gcs: GraphOfConvexSets,
    world: World,
    robot_start_x: float,
    robot_start_y: float,
) -> SequentialPlan:
    """
    Sample fresh parameters for every action entry, then rewrite the two
    NavigateAction targets so the robot follows a collision-free path derived
    from the GCS instead of the raw sampled pose.

    Action entry layout (matches _build_action_descriptions):
      0  ParkArmsAction  (pre)
      1  NavigateAction  -> counter area  [truncated to counter approach zone]
      2  PickUpAction
      3  NavigateAction  -> table area    [truncated to table approach zone]
      4  PlaceAction
      5  ParkArmsAction  (post)
    """
    for label, entry in zip(_ACTION_LABELS, entries):
        _apply_sample(entry)
        print(f"    Sampled  {label}")

    nav_to_counter_entry: NavigateAction = entries[1].instance
    nav_to_table_entry:   NavigateAction = entries[3].instance

    sampled_counter_pos = nav_to_counter_entry.target_location.pose.position
    sampled_table_pos   = nav_to_table_entry.target_location.pose.position

    counter_goal_x = float(sampled_counter_pos.x)
    counter_goal_y = float(sampled_counter_pos.y)
    table_goal_x   = float(sampled_table_pos.x)
    table_goal_y   = float(sampled_table_pos.y)

    print(
        f"    Sampled counter goal: ({counter_goal_x:.3f}, {counter_goal_y:.3f})"
        f"  |  table goal: ({table_goal_x:.3f}, {table_goal_y:.3f})"
    )

    nav_to_counter_actions = _navigate_via_gcs(
        context, gcs,
        start_x=robot_start_x, start_y=robot_start_y,
        goal_x=counter_goal_x, goal_y=counter_goal_y,
        world=world,
    )
    nav_to_table_actions = _navigate_via_gcs(
        context, gcs,
        start_x=counter_goal_x, start_y=counter_goal_y,
        goal_x=table_goal_x,   goal_y=table_goal_y,
        world=world,
    )

    actions = (
        [entries[0].instance]
        + nav_to_counter_actions
        + [entries[2].instance]
        + nav_to_table_actions
        + [entries[4].instance]
        + [entries[5].instance]
    )
    return SequentialPlan(context, *actions)


# ---- Main entry point -------------------------------------------------------

def sequential_plan_with_apartment() -> None:
    print("Building world ...")
    world, robot = _build_world(APARTMENT_URDF)
    _add_localization_frame(world, robot)
    milk_body, milk_pose = _add_milk(world, MILK_STL)

    print(f"  Milk spawned at  x={MILK_X:.3f},  y={MILK_Y:.3f},  z={MILK_Z:.3f}")
    print(f"  Place target at  x={PLACE_X:.3f}, y={PLACE_Y:.3f}, z={PLACE_Z:.3f}")

    gcs = _build_navigation_map(world)

    session = _create_session(DATABASE_URI)
    print(f"  Database: {DATABASE_URI}")

    rclpy.init()
    ros_node    = rclpy.create_node("pycram_gcs_plan_demo")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    robot_init_x: float = 1.4
    robot_init_y: float = 1.5

    try:
        _transformation_publisher = TFPublisher(_world=world, node=ros_node)
        _visualization_publisher = VizMarkerPublisher(_world=world, node=ros_node)

        context       = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        print("\nBuilding per-action probabilistic distributions ...")
        descriptions = _build_action_descriptions(world, robot, milk_variable)

        entries: list[ActionEntry] = [
            _build_action_entry(descriptions[0]),                                        # ParkArms (pre)
            _build_action_entry(descriptions[1], _COUNTER_APPROACH_BOUNDS),              # Navigate -> counter
            _build_action_entry(descriptions[2]),                                        # PickUp
            _build_action_entry(descriptions[3], _TABLE_APPROACH_BOUNDS),                # Navigate -> table
            _build_action_entry(descriptions[4]),                                        # Place
            _build_action_entry(descriptions[5]),                                        # ParkArms (post)
        ]

        for label, entry in zip(_ACTION_LABELS, entries):
            n_vars = len(entry.parameterization.variables)
            n_dist = len(entry.distribution.variables)
            print(f"  {label:<25}  {n_vars:>2} param vars  /  {n_dist:>2} distribution vars")

        successful_count = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            separator = "=" * 64
            print(f"\n{separator}")

            if iteration == 1:
                print(f"  Iteration {iteration:>3} / {NUMBER_OF_ITERATIONS}  [FIXED + GCS navigation]")
                print(separator)
                plan = _build_fixed_plan(
                    context, world, robot, milk_body, gcs,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )
            else:
                print(f"  Iteration {iteration:>3} / {NUMBER_OF_ITERATIONS}  [SAMPLED + GCS navigation]")
                print(separator)
                plan = _build_sampled_plan_with_gcs(
                    context, entries, gcs, world,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )

            with simulated_robot:
                try:
                    plan.perform()
                    _persist_plan(session, plan)
                    successful_count += 1
                    print(f"\n  v  Iteration {iteration} succeeded -- plan persisted to database.")
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"\n  x  Iteration {iteration} failed -- plan not stored.  ({type(exc).__name__}: {exc})")
                finally:
                    _respawn_milk(world, milk_body)

        print(f"\n{'=' * 64}")
        print(f"Done.  {successful_count} / {NUMBER_OF_ITERATIONS} plans stored in '{DATABASE_URI}'.")

    finally:
        session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    sequential_plan_with_apartment()