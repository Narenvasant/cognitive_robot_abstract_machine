import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

import rclpy
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped, PyCramPose, PyCramVector3, PyCramQuaternion, Header
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


NUM_ITERATIONS: int = 100
"""
Total number of plan iterations to run.
Iteration 1 uses fixed/known-good values (original demo2.py behaviour).
Iterations 2..NUM_ITERATIONS use probabilistic sampling (demo4.py behaviour).
"""

_DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///:memory:",
)

_MILK_X: float = 2.4
_MILK_Y: float = 2.5
_MILK_Z: float = 1.01

_COUNTER_APPROACH_X: float = 1.6
_COUNTER_APPROACH_Y: float = 2.5

_TABLE_APPROACH_X: float = 4.2
_TABLE_APPROACH_Y: float = 4.0

_PLACE_X: float = 5.0
_PLACE_Y: float = 4.0
_PLACE_Z: float = 0.80

# Search space bounds for the navigation map (covers the apartment floor area).
# Adjust these to match your apartment URDF extents if needed.
_GCS_MIN_X: float = -1.0
_GCS_MAX_X: float = 7.0
_GCS_MIN_Y: float = -1.0
_GCS_MAX_Y: float = 7.0
# Navigation is 2-D: z-slice at floor level.  A thin slab (0.0 → 0.1) is
# enough to build a floor-plan navigation map.
_GCS_MIN_Z: float = 0.0
_GCS_MAX_Z: float = 0.1

# Robot footprint bloat applied to obstacles so the robot body is accounted for.
_GCS_BLOAT_OBSTACLES: float = 0.3
_GCS_BLOAT_WALLS: float = 0.05

_RESOURCE_PATH = Path(__file__).resolve().parents[2] / "resources"
APARTMENT_URDF: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
MILK_STL:       Path = _RESOURCE_PATH / "objects" / "milk.stl"

_ACTION_LABELS = [
    "ParkArms (pre)",
    "Navigate → counter",
    "PickUp milk",
    "Navigate → table",
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


# ── Graph of Convex Sets helpers ──────────────────────────────────────────────

def _build_navigation_map(world: World) -> GraphOfConvexSets:
    """
    Build a floor-level Graph of Convex Sets (GCS) navigation map for the world.

    The map is restricted to a thin z-slab at floor level so that the planner
    reasons about 2-D navigation.  Obstacles are bloated by ``_GCS_BLOAT_OBSTACLES``
    to account for the robot footprint.

    :param world: The semantic digital twin world.
    :return: A GCS whose nodes are collision-free convex regions and whose edges
             encode adjacency between them.
    """
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=_GCS_MIN_X,
                max_x=_GCS_MAX_X,
                min_y=_GCS_MIN_Y,
                max_y=_GCS_MAX_Y,
                min_z=_GCS_MIN_Z,
                max_z=_GCS_MAX_Z,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        world.root,
    )

    print("  Building GCS navigation map …")
    t0 = time.time()
    gcs = GraphOfConvexSets.navigation_map_from_world(
        world=world,
        search_space=search_space,
        bloat_obstacles=_GCS_BLOAT_OBSTACLES,
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
    """
    Query the GCS for a collision-free path between two floor-level (x, y) positions.

    The z-coordinate is fixed to the centre of the navigation slab
    ``(_GCS_MIN_Z + _GCS_MAX_Z) / 2`` for both start and goal.

    :param gcs:    The navigation map.
    :param start_x: Start x position in world frame.
    :param start_y: Start y position in world frame.
    :param goal_x:  Goal x position in world frame.
    :param goal_y:  Goal y position in world frame.
    :param world:  The world (needed to attach the reference frame).
    :return: Ordered list of waypoints or ``None`` if no path exists.
    """
    z_nav = (_GCS_MIN_Z + _GCS_MAX_Z) / 2.0

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
    """
    Convert a GCS waypoint list into a list of PoseStamped objects suitable
    for chaining NavigateAction calls.

    :param path:        Waypoints returned by ``gcs.path_from_to``.
    :param world_frame: The world root body used as the header frame.
    :return: One PoseStamped per waypoint (excluding the start).
    """
    poses = []
    for point in path[1:]:  # skip the start point
        ps = PoseStamped(
            pose=PyCramPose(
                position=PyCramVector3(
                    x=float(point.x),
                    y=float(point.y),
                    z=0.0,
                ),
                orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
            ),
            header=Header(frame_id=world_frame),
        )
        poses.append(ps)
    return poses



def _build_world(apartment_urdf: Path) -> tuple:
    """Parse URDF files and return (world, robot)."""
    world = URDFParser.from_file(str(apartment_urdf)).parse()
    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    pr2_world = URDFParser.from_file(pr2_urdf).parse()
    robot = PR2.from_world(pr2_world)
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 0.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(pr2_world, robot_pose)

    table_body = world.get_body_by_name("table_area_main")
    with world.modify_world():
        table = Table(root=table_body)
        world.add_semantic_annotation(table)


    return world, robot


def _add_localization_frame(world, robot) -> None:
    """Insert map → odom_combined → robot base chain for OmniDrive navigation."""
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
    """Add a milk object to the world and return (body, pose)."""
    mesh = FileMesh.from_file(str(stl_path))
    body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(_MILK_X, _MILK_Y, _MILK_Z, 0, 0, 0)
    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(parent=world.root, child=body, world=world)
        world.add_connection(connection)
        connection.origin = pose
        world.add_semantic_annotation(Milk(root=body))
    return body, pose


def _create_pose_stamped(x: float, y: float, z: float, frame) -> PoseStamped:
    """Create a PoseStamped at the given world coordinates."""
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame),
    )


def _respawn_milk(world, milk_body: Body) -> None:
    """
    Teleport the milk body back to its original spawn pose on the counter.

    After PlaceAction, the milk is detached from the robot and re-parented
    under world.root with a Connection6DoF (see PlaceAction.execute). We
    locate that connection and overwrite its origin with the initial pose,
    which is equivalent to teleporting the object without removing/re-adding it.

    This keeps the same Body reference alive so that all existing designators,
    EQL variables, and probabilistic descriptions that point to milk_body
    remain valid across iterations.
    """
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        _MILK_X, _MILK_Y, _MILK_Z, 0, 0, 0
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
    print(f"  ↺  Milk respawned at  x={_MILK_X}, y={_MILK_Y}, z={_MILK_Z}")


# ── Database ──────────────────────────────────────────────────────────────────

def _create_session(database_uri: str) -> Session:
    """
    Create a SQLAlchemy Session with all ORM tables ensured to exist.

    For PostgreSQL, three compatibility patches are applied:

    1.  Identifier shortening — PostgreSQL enforces a 63-character limit on
        identifiers.  ORMatic-generated association table names can exceed this.
        Long names are deterministically shortened using a SHA-256 suffix.

    2.  Numpy scalar coercion — plan execution populates DAO fields with
        numpy scalars (np.float64, np.int64, etc.).  psycopg2 has no adapter
        for these types.  A before_cursor_execute listener converts all numpy
        scalars to their native Python equivalents.

    3.  validate_identifier no-op — SQLAlchemy's dialect runs a Python-side
        length check before DDL is compiled.  With shortened names already
        applied to Base.metadata, this check is redundant and is silenced.
    """
    engine = create_engine(database_uri)

    if "postgresql" in database_uri or "postgres" in database_uri:
        _apply_postgresql_patches(engine)

    session = Session(engine)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    return session


def _apply_postgresql_patches(engine) -> None:
    """Apply all PostgreSQL compatibility patches to *engine* and Base.metadata."""
    _patch_identifier_validation(engine)
    _shorten_metadata_table_names()
    _register_numpy_coercion(engine)


def _patch_identifier_validation(engine) -> None:
    """Silence the Python-side identifier length check in the PostgreSQL dialect."""
    engine.dialect.validate_identifier = lambda _ident: None


def _shorten_metadata_table_names() -> None:
    """
    Rename any Base.metadata table whose name exceeds PostgreSQL's 63-character
    identifier limit to a deterministic shortened form.
    """
    import hashlib

    def _shorten(name: str, max_len: int = 63) -> str:
        if len(name) <= max_len:
            return name
        suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:max_len - 9]}_{suffix}"

    for table in Base.metadata.tables.values():
        short = _shorten(table.name)
        if short != table.name:
            print(f"  [db] identifier shortened: '{table.name}' → '{short}'")
            table.name = short
            table.fullname = short


def _register_numpy_coercion(engine) -> None:
    """
    Register a before_cursor_execute listener that converts numpy scalar types
    to native Python equivalents before psycopg2 serializes them.
    """
    import numpy as np
    from sqlalchemy import event

    def _coerce(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
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


# ── Collision-free navigation helpers ────────────────────────────────────────

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
    """
    Plan a collision-free path from (start_x, start_y) to (goal_x, goal_y)
    using the pre-built GCS navigation map and return the resulting sequence
    of NavigateAction objects.

    If the GCS cannot find a path (e.g. because one of the endpoints falls
    inside an obstacle) a single direct NavigateAction to the goal is returned
    as a fallback, with a warning printed to stdout.

    :param context:           The current plan context (unused here, kept for
                              API symmetry with other helpers).
    :param gcs:               The floor-level GraphOfConvexSets navigation map.
    :param start_x:           Current robot x in world frame.
    :param start_y:           Current robot y in world frame.
    :param goal_x:            Target x in world frame.
    :param goal_y:            Target y in world frame.
    :param world:             The semantic digital twin world.
    :param keep_joint_states: Forwarded verbatim to every NavigateAction.
    :return: List of NavigateAction objects (one per GCS waypoint).
    """
    path = _gcs_collision_free_path(gcs, start_x, start_y, goal_x, goal_y, world)

    if path is None or len(path) < 2:
        print(
            f"    [GCS] ⚠  No collision-free path found "
            f"({start_x:.2f},{start_y:.2f}) → ({goal_x:.2f},{goal_y:.2f}). "
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
        f"    [GCS] Path ({start_x:.2f},{start_y:.2f}) → ({goal_x:.2f},{goal_y:.2f}): "
        f"{len(waypoint_poses)} waypoint(s)"
    )
    for i, pose in enumerate(waypoint_poses):
        p = pose.pose.position
        print(f"           waypoint {i+1}: ({p.x:.3f}, {p.y:.3f})")

    return [
        NavigateAction(target_location=ps, keep_joint_states=keep_joint_states)
        for ps in waypoint_poses
    ]


# ── Fixed plan (iteration 1) ──────────────────────────────────────────────────

def _build_fixed_plan(
    context: Context,
    world: World,
    robot,
    milk_body: Body,
    gcs: GraphOfConvexSets,
    robot_start_x: float = 1.4,
    robot_start_y: float = 1.5,
) -> SequentialPlan:
    """
    Build a SequentialPlan using fixed, known-good coordinates (demo2 behaviour)
    but with collision-free navigation waypoints derived from the GCS.

    The robot start position defaults to the initial spawn pose. After each
    NavigateAction the position is updated so the next GCS query is relative
    to the correct location.

    :param context:        The plan context.
    :param world:          The semantic digital twin world.
    :param robot:          The PR2 robot instance.
    :param milk_body:      The milk body designator.
    :param gcs:            The floor-level GraphOfConvexSets navigation map.
    :param robot_start_x:  Robot x at the start of the plan (world frame).
    :param robot_start_y:  Robot y at the start of the plan (world frame).
    :return: A SequentialPlan ready to be executed.
    """
    world_frame = world.root

    # ── leg 1: spawn → counter ─────────────────────────────────────────────
    nav_to_counter = _navigate_via_gcs(
        context,
        gcs,
        start_x=robot_start_x,
        start_y=robot_start_y,
        goal_x=_COUNTER_APPROACH_X,
        goal_y=_COUNTER_APPROACH_Y,
        world=world,
    )

    # ── leg 2: counter → table ─────────────────────────────────────────────
    nav_to_table = _navigate_via_gcs(
        context,
        gcs,
        start_x=_COUNTER_APPROACH_X,
        start_y=_COUNTER_APPROACH_Y,
        goal_x=_TABLE_APPROACH_X,
        goal_y=_TABLE_APPROACH_Y,
        world=world,
    )

    place_target_pose = _create_pose_stamped(_PLACE_X, _PLACE_Y, _PLACE_Z, world_frame)

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
                    manipulation_offset=0.05,
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


# ── Probabilistic plan (iterations 2+) ───────────────────────────────────────

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
        header=probable(Header)(frame_id=variable_from([robot._world.get_semantic_annotations_by_type(Milk)[0].root])),
    )


def _place_pose_description(robot) -> Any:
    """
    Build a probable PoseStamped for the place target with free x/y and
    fixed z (table surface height).
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=_PLACE_Z),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot._world.get_semantic_annotations_by_type(Table)[0].root])),
    )


def _build_action_descriptions(world, robot, milk_variable) -> list:
    """
    Construct the six probable_variable descriptions for the pick-and-place
    plan. Mirrors the fixed plan structure but with free parameters for the
    sampler to fill in.
    """
    manipulators = world.get_semantic_annotations_by_type(Manipulator)

    return [
        probable_variable(ParkArmsAction)(
            arm=...,
        ),
        probable_variable(NavigateAction)(
            target_location=_navigable_pose_description(robot),
            keep_joint_states=...,
        ),
        probable_variable(PickUpAction)(
            object_designator=milk_variable,
            arm=...,
            grasp_description=probable(GraspDescription)(
                approach_direction=...,
                vertical_alignment=...,
                rotate_gripper=...,
                manipulation_offset=0.05,
                manipulator=variable(Manipulator, manipulators),
            ),
        ),
        probable_variable(NavigateAction)(
            target_location=_navigable_pose_description(robot),
            keep_joint_states=...,
        ),
        probable_variable(PlaceAction)(
            object_designator=milk_variable,
            target_location=_place_pose_description(robot),
            arm=...,
        ),
        probable_variable(ParkArmsAction)(
            arm=...,
        ),
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


def _build_action_entry(description) -> ActionEntry:
    """
    Translate a probable_variable match description into a concrete action
    instance, derive its Parameterization, and pre-build the
    fully-factorized ProbabilisticCircuit.
    """
    instance         = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution     = parameterization.create_fully_factorized_distribution()
    # distribution.log_truncated_in_place()
    return ActionEntry(instance, parameterization, distribution)


def _apply_sample(entry: ActionEntry) -> None:
    """
    Draw one sample from an action's circuit and apply it to the instance
    in-place so the SequentialPlan receives freshly sampled parameters.
    """
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

    The sampled (x, y) is used as the *goal* of each navigation leg.  The
    GCS then finds a collision-free path from the robot's current estimated
    position to that goal.  If the sampled goal lands inside an obstacle, the
    GCS falls back to direct navigation (with a warning).

    Action entry layout assumed (matches _build_action_descriptions):
      0  ParkArmsAction  (pre)
      1  NavigateAction  → counter area
      2  PickUpAction
      3  NavigateAction  → table area
      4  PlaceAction
      5  ParkArmsAction  (post)

    :param context:        The plan context.
    :param entries:        List of ActionEntry objects (one per action).
    :param gcs:            The floor-level GraphOfConvexSets navigation map.
    :param world:          The semantic digital twin world.
    :param robot_start_x:  Robot x at the start of this iteration.
    :param robot_start_y:  Robot y at the start of this iteration.
    :return: A SequentialPlan ready to be executed.
    """
    # Step 1: draw fresh samples for all actions
    for label, entry in zip(_ACTION_LABELS, entries):
        _apply_sample(entry)
        print(f"    Sampled  {label}")

    # Step 2: extract sampled navigation goals
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

    # Step 3: replace sampled NavigateActions with GCS-planned waypoint sequences
    nav_to_counter_actions = _navigate_via_gcs(
        context, gcs,
        start_x=robot_start_x, start_y=robot_start_y,
        goal_x=counter_goal_x,  goal_y=counter_goal_y,
        world=world,
    )
    nav_to_table_actions = _navigate_via_gcs(
        context, gcs,
        start_x=counter_goal_x, start_y=counter_goal_y,
        goal_x=table_goal_x,    goal_y=table_goal_y,
        world=world,
    )

    # Step 4: assemble final action list
    actions = (
        [entries[0].instance]          # ParkArms (pre)
        + nav_to_counter_actions       # collision-free leg 1
        + [entries[2].instance]        # PickUpAction
        + nav_to_table_actions         # collision-free leg 2
        + [entries[4].instance]        # PlaceAction
        + [entries[5].instance]        # ParkArms (post)
    )

    return SequentialPlan(context, *actions)


# ── Main entry point ──────────────────────────────────────────────────────────

def sequential_plan_with_apartment() -> None:
    """
    Execute a pick-and-place plan for NUM_ITERATIONS iterations using a
    Graph of Convex Sets (GCS) navigation map for collision-free path planning:

      Iteration 1   — fixed known-good coordinates, GCS waypoints.
      Iterations 2+ — probabilistically sampled parameters, GCS waypoints.

    The GCS is built once after world construction and reused for all
    iterations.  Each successful execution is persisted to the configured
    database.

    Set the database URI via the environment variable before running:
        export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=\\
            postgresql://semantic_digital_twin:naren@localhost:5432/demo_robot_plans
    """
    print("Building world …")
    world, robot = _build_world(APARTMENT_URDF)
    _add_localization_frame(world, robot)
    milk_body, milk_pose = _add_milk(world, MILK_STL)

    print(f"  Milk spawned at  x={_MILK_X:.3f},  y={_MILK_Y:.3f},  z={_MILK_Z:.3f}")
    print(f"  Place target at  x={_PLACE_X:.3f}, y={_PLACE_Y:.3f}, z={_PLACE_Z:.3f}")


    gcs = _build_navigation_map(world)

    session = _create_session(_DATABASE_URI)
    print(f"  Database: {_DATABASE_URI}")

    rclpy.init()
    ros_node    = rclpy.create_node("pycram_gcs_plan_demo")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    robot_init_x: float = 1.4
    robot_init_y: float = 1.5

    try:
        _tf_publisher  = TFPublisher(_world=world, node=ros_node)
        _viz_publisher = VizMarkerPublisher(_world=world, node=ros_node)

        context       = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        print("\nBuilding per-action probabilistic distributions …")
        descriptions = _build_action_descriptions(world, robot, milk_variable)
        entries: list[ActionEntry] = [_build_action_entry(d) for d in descriptions]

        for label, entry in zip(_ACTION_LABELS, entries):
            n_vars = len(entry.parameterization.variables)
            n_dist = len(entry.distribution.variables)
            print(f"  {label:<25}  {n_vars:>2} param vars  /  {n_dist:>2} distribution vars")

        successful_count = 0

        for iteration in range(1, NUM_ITERATIONS + 1):
            separator = "=" * 64
            print(f"\n{separator}")

            if iteration == 1:
                # ── Iteration 1: fixed known-good coordinates + GCS paths ──
                print(f"  Iteration {iteration:>3} / {NUM_ITERATIONS}  [FIXED + GCS navigation]")
                print(separator)
                plan = _build_fixed_plan(
                    context, world, robot, milk_body, gcs,
                    robot_start_x=robot_init_x,
                    robot_start_y=robot_init_y,
                )

            else:
                # ── Iterations 2+: probabilistic sampling + GCS paths ─────
                print(f"  Iteration {iteration:>3} / {NUM_ITERATIONS}  [SAMPLED + GCS navigation]")
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
                    print(f"\n  ✓  Iteration {iteration} succeeded — plan persisted to database.")
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"\n  ✗  Iteration {iteration} failed — plan not stored.  ({type(exc).__name__}: {exc})")
                finally:
                    _respawn_milk(world, milk_body)

        print(f"\n{'=' * 64}")
        print(f"Done.  {successful_count} / {NUM_ITERATIONS} plans stored in '{_DATABASE_URI}'.")

    finally:
        session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    sequential_plan_with_apartment()