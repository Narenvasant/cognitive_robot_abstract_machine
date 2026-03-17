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
from typing import Any

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

from krrood.entity_query_language.factories import underspecified, variable, variable_from
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF, FixedConnection, OmniDrive
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_OF_ITERATIONS: int = 5000

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "postgresql://semantic_digital_twin:naren@localhost:5432/demo_robot_plans",
)

ROBOT_INITIAL_X: float = 0.0
ROBOT_INITIAL_Y: float = 0.0

MILK_INITIAL_X: float = 2.4
MILK_INITIAL_Y: float = 0.0
MILK_INITIAL_Z: float = 1.01

PICK_APPROACH_X: float = 1.6
PICK_APPROACH_Y: float = 0.0

PLACE_APPROACH_X: float = 3.5
PLACE_APPROACH_Y: float = 0.0

PLACE_TARGET_X: float = 4.1
PLACE_TARGET_Y: float = 0.0
PLACE_TARGET_Z: float = 0.80

PICK_APPROACH_SAMPLING_BOUNDS:  tuple[float, float, float, float] = (1.2, 1.8, -0.4, 0.4)
PLACE_APPROACH_SAMPLING_BOUNDS: tuple[float, float, float, float] = (3.2, 3.8, -0.4, 0.4)

PR2_URDF_PATH: str  = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
MILK_STL_PATH: Path = Path(__file__).resolve().parents[3] / "resources" / "objects" / "milk.stl"


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------

def _header_deepcopy(self, memo: Any) -> Header:
    if isinstance(self, type):
        return self
    stamp = getattr(self, "stamp", None) or datetime.datetime.now()
    return Header(
        frame_id=getattr(self, "frame_id", None),
        stamp=copy.deepcopy(stamp, memo),
        sequence=getattr(self, "sequence", 0),
    )


def _pose_stamped_deepcopy(self, memo: Any) -> PoseStamped:
    if isinstance(self, type):
        return self
    return PoseStamped(
        copy.deepcopy(getattr(self, "pose", None), memo),
        copy.deepcopy(getattr(self, "header", None), memo),
    )


def _header_getattr(self, name: str) -> Any:
    defaults = {
        "stamp":    lambda: datetime.datetime.now(),
        "sequence": lambda: 0,
        "frame_id": lambda: None,
    }
    if name in defaults:
        value = defaults[name]()
        object.__setattr__(self, name, value)
        return value
    raise AttributeError(name)


Header.__deepcopy__      = _header_deepcopy
Header.__getattr__       = _header_getattr
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy


def _patch_orm_numpy_array_type() -> None:
    """Patch PyCRAM ORM numpy TypeDecorator to tolerate None values."""
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

    def _guarded(self, value, dialect):
        if value is None:
            return None
        return original(self, value, dialect)

    target_class.process_bind_param = _guarded
    print(f"  [patch] Patched {target_class.__name__}.process_bind_param to handle None.")


_patch_orm_numpy_array_type()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionEntry:
    description:       Any
    parameters:        UnderspecifiedParameters
    distribution:      ProbabilisticCircuit
    base_distribution: ProbabilisticCircuit = None


@dataclass
class SampledParameters:
    pick_approach_x:  float
    pick_approach_y:  float
    place_approach_x: float
    place_approach_y: float
    pick_arm:         Arms


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def _build_world_with_robot() -> tuple[World, PR2]:
    print("  [world] Creating scene root ...")
    scene_world     = World(name="pick_and_place_scene")
    scene_root_body = Body(name=PrefixedName("scene"))
    with scene_world.modify_world():
        scene_world.add_kinematic_structure_entity(scene_root_body)

    print("  [world] Loading PR2 URDF ...")
    pr2_world = URDFParser.from_file(PR2_URDF_PATH).parse()
    robot     = PR2.from_world(pr2_world)

    robot_initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with scene_world.modify_world():
        scene_world.merge_world_at_pose(pr2_world, robot_initial_pose)

    print("  [world] PR2 merged into scene.")
    return scene_world, robot


def _add_localization_frames(world: World, robot: PR2) -> None:
    print("  [world] Adding localization frames ...")
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
                    ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
                ),
            )
        )
    print("  [world] Localization frames OK.")


def _add_milk_to_world(world: World) -> Body:
    print(f"  [world] Adding milk at ({MILK_INITIAL_X}, {MILK_INITIAL_Y}, {MILK_INITIAL_Z}) ...")
    mesh      = FileMesh.from_file(str(MILK_STL_PATH))
    milk_body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_INITIAL_X, MILK_INITIAL_Y, MILK_INITIAL_Z, 0, 0, 0
    )
    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=milk_body, world=world
        )
        world.add_connection(milk_connection)
        milk_connection.origin = initial_pose
        world.add_semantic_annotation(Milk(root=milk_body))

    print("  [world] Milk added.")
    return milk_body


def _respawn_milk(world: World, milk_body: Body) -> None:
    initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_INITIAL_X, MILK_INITIAL_Y, MILK_INITIAL_Z, 0, 0, 0
    )
    with world.modify_world():
        current_connection = milk_body.parent_connection
        if current_connection is not None and current_connection.parent is not world.root:
            world.remove_connection(current_connection)
            free_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(free_connection)
            free_connection.origin = initial_pose
        elif current_connection is not None:
            current_connection.origin = initial_pose
        else:
            free_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(free_connection)
            free_connection.origin = initial_pose
    print("  [respawn] Milk reset.")


def _respawn_robot(world: World, robot: PR2) -> None:
    initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with world.modify_world():
        base_connection = robot.root.parent_connection
        if base_connection is not None:
            base_connection.origin = initial_pose
    print("  [respawn] Robot reset.")


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


def _apply_postgresql_patches(engine) -> None:
    _disable_postgresql_identifier_length_validation(engine)
    _shorten_long_postgresql_table_names()
    _register_postgresql_numpy_scalar_coercion(engine)


def _disable_postgresql_identifier_length_validation(engine) -> None:
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


def _register_postgresql_numpy_scalar_coercion(engine) -> None:
    import numpy
    from sqlalchemy import event

    def _coerce(value):
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        return value

    def _coerce_params(params):
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


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    print("  [db] Persisting plan ...")
    session.add(to_dao(plan))
    session.commit()
    print("  [db] Plan committed OK.")


# ---------------------------------------------------------------------------
# Pose construction
# ---------------------------------------------------------------------------

def _create_pose_stamped(x: float, y: float, z: float, frame_id: Any) -> PoseStamped:
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame_id, stamp=datetime.datetime.now(), sequence=0),
    )


# ---------------------------------------------------------------------------
# Probabilistic distribution setup
# ---------------------------------------------------------------------------

def _build_navigable_pose_description(robot: PR2) -> Any:
    return underspecified(PoseStamped)(
        pose=underspecified(PyCramPose)(
            position=underspecified(PyCramVector3)(x=..., y=..., z=0),
            orientation=underspecified(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=robot._world.root, sequence=0),
    )


def _build_navigate_entry(
    robot: PR2,
    keep_joint_states: bool,
    sampling_bounds: tuple[float, float, float, float],
) -> ActionEntry:
    description = underspecified(NavigateAction)(
        target_location=_build_navigable_pose_description(robot),
        keep_joint_states=keep_joint_states,
    )
    description.resolve()
    parameters   = UnderspecifiedParameters(description)
    distribution = fully_factorized(parameters.variables.values())

    x_min, x_max, y_min, y_max = sampling_bounds
    distribution = _truncate_distribution_to_position_bounds(
        distribution, x_min, x_max, y_min, y_max
    )

    return ActionEntry(
        description=description,
        parameters=parameters,
        distribution=distribution,
        base_distribution=copy.deepcopy(distribution),
    )


def _build_pickup_entry(robot: PR2, milk_variable: Any) -> ActionEntry:
    available_manipulators = robot._world.get_semantic_annotations_by_type(Manipulator)

    description = underspecified(PickUpAction)(
        object_designator=milk_variable,
        arm=variable(Arms, [Arms.LEFT, Arms.RIGHT]),
        grasp_description=underspecified(GraspDescription)(
            approach_direction=variable(ApproachDirection, [ApproachDirection.FRONT]),
            vertical_alignment=variable(VerticalAlignment, [VerticalAlignment.NoAlignment]),
            rotate_gripper=variable(bool, [False]),
            manipulation_offset=0.06,
            manipulator=variable(Manipulator, available_manipulators),
        ),
    )
    description.resolve()
    parameters   = UnderspecifiedParameters(description)
    distribution = fully_factorized(parameters.variables.values())

    return ActionEntry(
        description=description,
        parameters=parameters,
        distribution=distribution,
        base_distribution=copy.deepcopy(distribution),
    )


def _truncate_distribution_to_position_bounds(
    distribution: ProbabilisticCircuit,
    x_minimum: float,
    x_maximum: float,
    y_minimum: float,
    y_maximum: float,
) -> ProbabilisticCircuit:
    from random_events.interval import closed
    from random_events.product_algebra import SimpleEvent
    from random_events.variable import Continuous

    x_var = y_var = None
    for rv in distribution.variables:
        if isinstance(rv, Continuous):
            if rv.name.endswith(".position.x") or rv.name.endswith(".x"):
                x_var = rv
            elif rv.name.endswith(".position.y") or rv.name.endswith(".y"):
                y_var = rv

    if x_var is None or y_var is None:
        print(
            f"  [truncate] WARNING: x/y variables not found; skipping. "
            f"Variables: {[v.name for v in distribution.variables]}"
        )
        return distribution

    bounding_event = SimpleEvent(
        {x_var: closed(x_minimum, x_maximum), y_var: closed(y_minimum, y_maximum)}
    ).as_composite_set()
    full_event = bounding_event.fill_missing_variables_pure(distribution.variables)

    candidate    = copy.deepcopy(distribution)
    truncated, _ = candidate.log_truncated_in_place(full_event)

    if truncated is None:
        print(
            f"  [truncate] WARNING: zero-probability region "
            f"x=[{x_minimum},{x_maximum}] y=[{y_minimum},{y_maximum}]; returning untruncated."
        )
        return distribution

    print(f"  [truncate] OK: x=[{x_minimum},{x_maximum}] y=[{y_minimum},{y_maximum}]")
    return truncated


def _sample_from_entry(entry: ActionEntry) -> Any:
    """
    Sample parameters from the entry's distribution and return a concrete
    action instance using the new UnderspecifiedParameters API.

    Replaces the old _sample_parameters_into_action / parameterize_object_with_sample
    in-place mutation pattern with create_instance_from_variables_and_sample.
    """
    raw_sample = entry.distribution.sample(1)[0]
    return entry.parameters.create_instance_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_fixed_plan(
    context:   Context,
    world:     World,
    robot:     PR2,
    milk_body: Body,
) -> SequentialPlan:
    fixed_arm = Arms.RIGHT

    pick_approach_pose  = _create_pose_stamped(PICK_APPROACH_X,  PICK_APPROACH_Y,  0.0,            world.root)
    place_approach_pose = _create_pose_stamped(PLACE_APPROACH_X, PLACE_APPROACH_Y, 0.0,            world.root)
    place_target_pose   = _create_pose_stamped(PLACE_TARGET_X,   PLACE_TARGET_Y,   PLACE_TARGET_Z, world.root)

    print(f"  [plan] Fixed parameters:")
    print(f"         pick  approach : ({PICK_APPROACH_X}, {PICK_APPROACH_Y})")
    print(f"         place approach : ({PLACE_APPROACH_X}, {PLACE_APPROACH_Y})")
    print(f"         place target   : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"         arm            : {fixed_arm}")

    return SequentialPlan(
        context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_approach_pose,  keep_joint_states=False),
        PickUpAction(
            object_designator=milk_body,
            arm=fixed_arm,
            grasp_description=GraspDescription(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                rotate_gripper=False,
                manipulation_offset=0.06,
                manipulator=robot.right_arm.manipulator,
            ),
        ),
        NavigateAction(target_location=place_approach_pose, keep_joint_states=True),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=fixed_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _collect_sampled_parameters(
    pick_navigate_entry:  ActionEntry,
    place_navigate_entry: ActionEntry,
    pickup_entry:         ActionEntry,
) -> SampledParameters:
    pick_nav_instance  = _sample_from_entry(pick_navigate_entry)
    place_nav_instance = _sample_from_entry(place_navigate_entry)
    pickup_instance    = _sample_from_entry(pickup_entry)

    pick_pos  = pick_nav_instance.target_location.pose.position
    place_pos = place_nav_instance.target_location.pose.position

    return SampledParameters(
        pick_approach_x=pick_pos.x,
        pick_approach_y=pick_pos.y,
        place_approach_x=place_pos.x,
        place_approach_y=place_pos.y,
        pick_arm=pickup_instance.arm,
    )


def _build_sampled_plan(
    context:   Context,
    params:    SampledParameters,
    world:     World,
    robot:     PR2,
    milk_body: Body,
) -> SequentialPlan:
    manipulator = (
        robot.right_arm.manipulator if params.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )

    pick_approach_pose  = _create_pose_stamped(params.pick_approach_x,  params.pick_approach_y,  0.0,            world.root)
    place_approach_pose = _create_pose_stamped(params.place_approach_x, params.place_approach_y, 0.0,            world.root)
    place_target_pose   = _create_pose_stamped(PLACE_TARGET_X,          PLACE_TARGET_Y,          PLACE_TARGET_Z, world.root)

    print(f"  [plan] Sampled parameters:")
    print(f"         pick  approach : ({params.pick_approach_x:.3f}, {params.pick_approach_y:.3f})")
    print(f"         place approach : ({params.place_approach_x:.3f}, {params.place_approach_y:.3f})")
    print(f"         place target   : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"         arm            : {params.pick_arm}")

    return SequentialPlan(
        context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_approach_pose,  keep_joint_states=False),
        PickUpAction(
            object_designator=milk_body,
            arm=params.pick_arm,
            grasp_description=GraspDescription(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                rotate_gripper=False,
                manipulation_offset=0.06,
                manipulator=manipulator,
            ),
        ),
        NavigateAction(target_location=place_approach_pose, keep_joint_states=True),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=params.pick_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_and_place_demo() -> None:
    print("=" * 64)
    print("  pick_and_place_demo")
    print(f"  NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}")
    print(f"  PLACE_TARGET_Z       = {PLACE_TARGET_Z}")
    print(f"  DATABASE_URI         = {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/5] Building world ...")
    world, robot = _build_world_with_robot()
    _add_localization_frames(world, robot)
    milk_body = _add_milk_to_world(world)

    print("\n[2/5] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/5] Initialising ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_demo_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()
    print("  [ros] Node started.")

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context = Context(world, robot, None)
        milk_variable    = variable_from([milk_body])

        print("\n[4/5] Building probabilistic distributions ...")
        pick_navigate_entry  = _build_navigate_entry(
            robot, keep_joint_states=False, sampling_bounds=PICK_APPROACH_SAMPLING_BOUNDS
        )
        place_navigate_entry = _build_navigate_entry(
            robot, keep_joint_states=True, sampling_bounds=PLACE_APPROACH_SAMPLING_BOUNDS
        )
        pickup_entry = _build_pickup_entry(robot, milk_variable)
        print("  [distributions] All entries built OK.")

        print("\n[5/5] Running iterations ...")
        successful_count = 0
        failed_count     = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(f"  Iteration {iteration} / {NUMBER_OF_ITERATIONS}  "
                  f"(success={successful_count}  failed={failed_count})")
            print(f"{'=' * 64}")

            if iteration == 1:
                print("  Mode: FIXED (deterministic parameters)")
                plan = _build_fixed_plan(planning_context, world, robot, milk_body)
            else:
                print("  Mode: SAMPLED")
                params = _collect_sampled_parameters(
                    pick_navigate_entry,
                    place_navigate_entry,
                    pickup_entry,
                )
                plan = _build_sampled_plan(planning_context, params, world, robot, milk_body)

            print("\n  Executing plan ...")
            with simulated_robot:
                success = False
                try:
                    plan.perform()
                    success = True
                    print("  plan.perform() completed without exception.")

                except Exception as exc:
                    failed_count += 1
                    print(f"\n  RESULT: FAILED  (iteration {iteration})")
                    print(f"  Exception type : {type(exc).__name__}")
                    print(f"  Exception msg  : {exc}")
                    print("  Traceback (last 3 lines):")
                    lines = traceback.format_exc().strip().splitlines()
                    for line in lines[-3:]:
                        print(f"    {line}")

                finally:
                    if success:
                        try:
                            _persist_plan(database_session, plan)
                            successful_count += 1
                            print(f"  RESULT: SUCCESS  "
                                  f"({successful_count} stored / "
                                  f"{iteration} attempted / "
                                  f"{NUMBER_OF_ITERATIONS - iteration} remaining)")
                        except Exception as db_exc:
                            print(f"  [db] ERROR persisting plan: {db_exc}")
                            traceback.print_exc()

                    print("\n  Resetting world state ...")
                    _respawn_milk(world, milk_body)
                    _respawn_robot(world, robot)

        print(f"\n{'=' * 64}")
        print(f"  Run complete.")
        print(f"  Total iterations : {NUMBER_OF_ITERATIONS}")
        print(f"  Successful plans : {successful_count}  "
              f"({100 * successful_count // NUMBER_OF_ITERATIONS}%)")
        print(f"  Failed           : {failed_count}")
        print(f"  Database         : {DATABASE_URI}")
        print(f"{'=' * 64}")

        try:
            from sqlalchemy import text
            result = database_session.execute(
                text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
            ).scalar()
            print(f"  DB row count (SequentialPlanDAO) : {result}")
            if result == successful_count:
                print("  Counter matches DB — data is clean.")
            else:
                print(f"  WARNING: counter ({successful_count}) != DB rows ({result}).")
                print("  This means the ORM wrote some rows outside _persist_plan.")
                print("  Run TRUNCATE + re-run if you need a clean dataset.")
        except Exception as count_exc:
            print(f"  [db] Could not verify row count: {count_exc}")

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo()