import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF, OmniDrive
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# ── Iteration config ──────────────────────────────────────────────────────────
NUM_ITERATIONS: int = 20
"""
Total number of plan iterations to run.
Iteration 1 uses fixed/known-good values (original demo2.py behaviour).
Iterations 2..NUM_ITERATIONS use probabilistic sampling (demo4.py behaviour).
"""

_DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///:memory:",
)

# ── World coordinates ─────────────────────────────────────────────────────────
_MILK_X: float = 2.4
_MILK_Y: float = 2.5
_MILK_Z: float = 1.01

_COUNTER_APPROACH_X: float = 1.6
_COUNTER_APPROACH_Y: float = 2.5

_TABLE_APPROACH_X: float = 4.2
_TABLE_APPROACH_Y: float = 4.0

_PLACE_X: float = 5.0
_PLACE_Y: float = 4.0
_PLACE_Z: float = 0.80   # slightly above table surface to account for milk height

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


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class ActionEntry:
    """
    Groups a concrete action instance with its Parameterization and the
    pre-built ProbabilisticCircuit so that all three travel together.
    """
    instance: Any
    parameterization: Parameterization
    distribution: ProbabilisticCircuit


# ── World / robot setup ───────────────────────────────────────────────────────

def _build_world(apartment_urdf: Path) -> tuple:
    """Parse URDF files and return (world, robot)."""
    world = URDFParser.from_file(str(apartment_urdf)).parse()
    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    pr2_world = URDFParser.from_file(pr2_urdf).parse()
    robot = PR2.from_world(pr2_world)
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 0.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(pr2_world, robot_pose)
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


def _add_milk(world, stl_path: Path) -> tuple[Body, HomogeneousTransformationMatrix]:
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
        # After PlaceAction the milk is always a direct child of world.root.
        connection = milk_body.parent_connection
        if connection is not None:
            # Re-parent to world.root if it somehow ended up elsewhere.
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
            # Safety fallback: create a fresh connection.
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
        identifiers.  ORMatic-generated association table names can exceed this
        (e.g. 'WorldEntityWithSimulatorPropertiesDAO_simulator_additional_
        properties_association' is 80 chars).  Long names are deterministically
        shortened using a SHA-256 suffix so they remain unique and stable across
        runs.

    2.  Numpy scalar coercion — plan execution populates DAO fields with
        numpy scalars (np.float64, np.int64, etc.).  psycopg2 has no adapter
        for these types and serializes them as their repr ('np.float64(0.0)'),
        which PostgreSQL rejects.  A before_cursor_execute listener converts
        all numpy scalars to their native Python equivalents.

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

    The shortened name is: ``original[:54] + "_" + sha256(original)[:8]``
    This guarantees uniqueness and is stable across runs.
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
    (float64, int64, bool_) to native Python equivalents before psycopg2
    serializes them into the SQL statement.
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


# Iteration 1: fixed plan

def _build_fixed_plan(context: Context, world, robot, milk_body: Body) -> SequentialPlan:
    """
    Build a SequentialPlan using the fixed, known-good coordinates from the
    original demo2.py. Used exclusively for iteration 1.

    All poses use world.root as the frame so coordinates stay correct in world
    space regardless of where the robot navigates to.
    """
    world_frame = world.root

    counter_pose      = _create_pose_stamped(_COUNTER_APPROACH_X, _COUNTER_APPROACH_Y, 0.0,      world_frame)
    table_pose        = _create_pose_stamped(_TABLE_APPROACH_X,   _TABLE_APPROACH_Y,   0.0,      world_frame)
    place_target_pose = _create_pose_stamped(_PLACE_X,            _PLACE_Y,            _PLACE_Z, world_frame)

    return SequentialPlan(
        context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(
            target_location=counter_pose,
            keep_joint_states=False,
        ),
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
        ),
        NavigateAction(
            target_location=table_pose,
            keep_joint_states=False,
        ),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=Arms.RIGHT,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


# Iterations 2+: probabilistic plan

def _navigable_pose_description(robot) -> Any:
    """
    Build a probable PoseStamped description for a navigation target with
    free x/y position components for probabilistic sampling.
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=0),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot.root])),
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
        header=probable(Header)(frame_id=variable_from([robot.root])),
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


def _build_action_entry(description) -> ActionEntry:
    """
    Translate a probable_variable match description into a concrete action
    instance, derive its Parameterization, and pre-build the
    fully-factorized ProbabilisticCircuit.
    """
    instance         = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution     = parameterization.create_fully_factorized_distribution()
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


def _build_sampled_plan(context: Context, entries: list[ActionEntry]) -> SequentialPlan:
    """
    Apply one fresh sample to every ActionEntry and build a SequentialPlan
    from the updated instances. Used for iterations 2 onwards.
    """
    for label, entry in zip(_ACTION_LABELS, entries):
        _apply_sample(entry)
        print(f"    Sampled  {label}")
    return SequentialPlan(context, *[e.instance for e in entries])



def sequential_plan_with_apartment() -> None:
    """
    Execute a pick-and-place plan for NUM_ITERATIONS iterations:

      Iteration 1   — fixed known-good coordinates (original demo2.py behaviour).
      Iterations 2+ — probabilistically sampled parameters (demo4.py behaviour).

    Each successful execution is persisted to the configured database.

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

    session = _create_session(_DATABASE_URI)
    print(f"  Database: {_DATABASE_URI}")

    rclpy.init()
    ros_node    = rclpy.create_node("pycram_fixed_plan_demo")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    try:
        _tf_publisher  = TFPublisher(_world=world, node=ros_node)
        _viz_publisher = VizMarkerPublisher(_world=world, node=ros_node)

        context       = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        # Build probabilistic entries once — instances are reused and updated
        # in-place each sampled iteration, avoiding repeated construction cost.
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

                print(f"  Iteration {iteration:>3} / {NUM_ITERATIONS}  [FIXED — known-good coordinates]")
                print(separator)
                plan = _build_fixed_plan(context, world, robot, milk_body)

            else:
                # ── Iterations 2+: probabilistic sampling ────────────────────
                print(f"  Iteration {iteration:>3} / {NUM_ITERATIONS}  [SAMPLED — probabilistic parameters]")
                print(separator)
                plan = _build_sampled_plan(context, entries)

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
                    # Always respawn milk at its original counter position so the
                    # next iteration starts from a clean, repeatable world state.
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