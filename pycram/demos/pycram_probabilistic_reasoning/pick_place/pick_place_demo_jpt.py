from __future__ import annotations

import sys
from unittest.mock import MagicMock

# nav2_msgs is not installed; mock it before giskardpy tries to use it
# via pycram.orm.ormatic_interface → giskardpy.motion_statechart.ros2_nodes.ros_tasks
_nav2_mock = MagicMock()
sys.modules["nav2_msgs"]                         = _nav2_mock
sys.modules["nav2_msgs.action"]                  = _nav2_mock.action
sys.modules["nav2_msgs.action.NavigateToPose"]   = _nav2_mock.action.NavigateToPose

import hashlib
import inspect
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import sqlalchemy.types as sqlalchemy_types
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.helper import to_dao
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import ExecutesSequentially
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base

from krrood.ormatic.utils import create_engine

from jpt.distributions.univariate import Multinomial
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.trees import JPT as JointProbabilityTree
from jpt.variables import NumericVariable, SymbolicVariable

from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose, Quaternion
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Mesh
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

GRASP_MANIPULATION_OFFSET: float = 0.06

MILK_STL_PATH: Path = Path(__file__).resolve().parents[3] / "resources" / "objects" / "milk.stl"

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlanParameters:
    """
    One complete set of plan parameters sampled from the JPT.

    All values are drawn in a single JPT sample call so their joint
    distribution reflects the correlations learned from the Batch 1 data.
    This is the key difference from Batch 1, where each variable was sampled
    independently from a uniform distribution within fixed bounds.
    """
    pick_approach_x:  float
    pick_approach_y:  float
    place_approach_x: float
    place_approach_y: float
    pick_arm:         Arms


# ---------------------------------------------------------------------------
# ORM patch: handle None values in numpy TypeDecorator
# ---------------------------------------------------------------------------

def _patch_orm_numpy_array_type() -> None:
    """
    Patch the PyCRAM ORM numpy TypeDecorator so that None values are passed
    through without calling .astype(), which raises AttributeError when a plan
    action has no associated array data.
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

def _build_world_with_robot() -> tuple[World, PR2]:
    pr2_world = URDFParser.from_file(
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    robot = PR2.from_world(pr2_world)

    # Use pr2_world as the scene directly — no empty World needed.
    # Renaming it and then calling merge_world_at_pose on itself would fail
    # because merge_world_at_pose requires self.root to exist.
    pr2_world.name = "pick_and_place_scene"

    # Set the robot's initial pose via its existing root connection
    initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with pr2_world.modify_world():
        root_connection = robot.root.parent_connection
        if root_connection is not None:
            root_connection.origin = initial_pose

    print("  [world] PR2 world initialised as scene.")
    return pr2_world, robot


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
                    ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
                ),
            )
        )


def _add_milk_to_world(world: World) -> Body:
    mesh = Mesh.from_file(str(MILK_STL_PATH))
    milk_body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_INITIAL_X, MILK_INITIAL_Y, MILK_INITIAL_Z, 0, 0, 0
    )
    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=milk_body, world=world
        )
        world.add_connection(milk_connection)
        milk_connection.origin = spawn_pose
        world.add_semantic_annotation(Milk(root=milk_body))
    print(f"  [world] Milk at ({MILK_INITIAL_X}, {MILK_INITIAL_Y}, {MILK_INITIAL_Z})")
    return milk_body


def _respawn_milk(world: World, milk_body: Body) -> None:
    """Reset the milk carton to its original pose, re-attaching it to the world root
    if it was previously attached to the robot gripper during a pick-up."""
    spawn_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
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
            free_connection.origin = spawn_pose
        elif current_connection is not None:
            current_connection.origin = spawn_pose
        else:
            free_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(free_connection)
            free_connection.origin = spawn_pose
    print(f"  [respawn] Milk reset to ({MILK_INITIAL_X}, {MILK_INITIAL_Y}, {MILK_INITIAL_Z})")


def _respawn_robot(world: World, robot: PR2) -> None:
    """Reset the robot base to its original pose."""
    initial_robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with world.modify_world():
        base_connection = robot.root.parent_connection
        if base_connection is not None:
            base_connection.origin = initial_robot_pose
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


def _persist_plan(session: Session, plan: ExecutesSequentially) -> None:
    print("  [db] Persisting plan ...")
    session.add(to_dao(plan))
    session.commit()
    print("  [db] Plan committed.")


# ---------------------------------------------------------------------------
# Pose helper
# ---------------------------------------------------------------------------

def _make_pose(x: float, y: float, z: float, reference_frame: Any) -> Pose:
    return Pose(
        position=Point3(x=x, y=y, z=z),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
        reference_frame=reference_frame,
    )


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
    Draw one joint sample from the JPT and map it to PlanParameters.

    All five plan variables are drawn in a single call, preserving the
    inter-variable correlations learned from successful Batch 1 executions.
    Safety clipping is applied after sampling to keep positions within the
    original Batch 1 bounds if the JPT extrapolates slightly outside the
    training range.
    """
    sample_row     = joint_probability_tree.sample(1)[0]
    sample_by_name = {variable.name: sample_row[index]
                      for index, variable in enumerate(JPT_VARIABLES)}

    arm_label = sample_by_name["pick_arm"]
    if isinstance(arm_label, (int, float)):
        arm_label = ArmChoiceDomain.labels[int(arm_label)]
    pick_arm = Arms.LEFT if arm_label == "LEFT" else Arms.RIGHT

    pick_x_min,  pick_x_max,  pick_y_min,  pick_y_max  = PICK_APPROACH_SAMPLING_BOUNDS
    place_x_min, place_x_max, place_y_min, place_y_max = PLACE_APPROACH_SAMPLING_BOUNDS

    return PlanParameters(
        pick_approach_x  = float(np.clip(sample_by_name["pick_approach_x"],  pick_x_min,  pick_x_max)),
        pick_approach_y  = float(np.clip(sample_by_name["pick_approach_y"],  pick_y_min,  pick_y_max)),
        place_approach_x = float(np.clip(sample_by_name["place_approach_x"], place_x_min, place_x_max)),
        place_approach_y = float(np.clip(sample_by_name["place_approach_y"], place_y_min, place_y_max)),
        pick_arm         = pick_arm,
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_fixed_plan(
    planning_context: Context,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
) -> ExecutesSequentially:
    """
    Deterministic seed plan used for iteration 1.

    Using known-good fixed parameters on the first iteration confirms that
    the world, robot, and database are correctly initialised before
    probabilistic sampling begins.
    """
    seed_arm   = Arms.RIGHT
    place_pose = _make_pose(PLACE_TARGET_X,   PLACE_TARGET_Y,   PLACE_TARGET_Z, world.root)
    pick_pose  = _make_pose(PICK_APPROACH_X,  PICK_APPROACH_Y,  0.0,            world.root)
    place_app  = _make_pose(PLACE_APPROACH_X, PLACE_APPROACH_Y, 0.0,            world.root)

    print(
        f"  [plan] seed — "
        f"pick:({PICK_APPROACH_X},{PICK_APPROACH_Y})  "
        f"place:({PLACE_APPROACH_X},{PLACE_APPROACH_Y})  "
        f"arm:{seed_arm}"
    )
    return ExecutesSequentially(
        planning_context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_pose),
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
        NavigateAction(target_location=place_app),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_pose,
            arm=seed_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _build_sampled_plan(
    planning_context: Context,
    plan_parameters:  PlanParameters,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
) -> ExecutesSequentially:
    """Build a plan from JPT-sampled approach positions."""
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )
    pick_pose  = _make_pose(plan_parameters.pick_approach_x,  plan_parameters.pick_approach_y,  0.0, world.root)
    place_app  = _make_pose(plan_parameters.place_approach_x, plan_parameters.place_approach_y, 0.0, world.root)
    place_pose = _make_pose(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)

    print(
        f"  [plan] sampled — "
        f"pick:({plan_parameters.pick_approach_x:.3f},{plan_parameters.pick_approach_y:.3f})  "
        f"place:({plan_parameters.place_approach_x:.3f},{plan_parameters.place_approach_y:.3f})  "
        f"arm:{plan_parameters.pick_arm}"
    )
    return ExecutesSequentially(
        planning_context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_pose),
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
        NavigateAction(target_location=place_app),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_pose,
            arm=plan_parameters.pick_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_and_place_demo_jpt() -> None:
    """
    Batch 2: pick-and-place demo with JPT-informed parameter sampling.

    Identical world setup to Batch 1 (pick_and_place_demo.py) except that
    approach positions and arm choice are sampled from the pre-fitted JPT
    rather than independently from uniform distributions.

    The JPT was fitted on the 1742 successful plans from Batch 1. Sampling
    from it concentrates the robot's parameter choices in the region of the
    parameter space that historically led to successful executions.

    Iteration 1 uses fixed deterministic parameters to confirm the world,
    robot, and database are correctly initialised before probabilistic sampling
    begins on subsequent iterations.

    Expected outcome: meaningfully higher success rate than the Batch 1
    baseline of 34.8%.
    """
    print("=" * 64)
    print("  pick_and_place_demo_jpt  (Batch 2 / JPT)")
    print(f"  Iterations     : {NUMBER_OF_ITERATIONS}")
    print(f"  Place target   : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"  JPT model      : {JPT_MODEL_PATH}")
    print(f"  Database       : {DATABASE_URI}")
    print("=" * 64)

    print("\n[1/4] Building world ...")
    world, robot = _build_world_with_robot()
    _add_localization_frames(world, robot)
    milk_body = _add_milk_to_world(world)

    print("\n[2/4] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/4] Loading JPT model ...")
    joint_probability_tree = _load_joint_probability_tree(JPT_MODEL_PATH)

    print("\n[4/4] Starting ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_demo_jpt_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()
    print("  [ros] Node started.")

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context  = Context(world, robot, None)
        successful_count  = 0
        failed_count      = 0

        print("\n[Running iterations ...]")

        for iteration_number in range(1, NUMBER_OF_ITERATIONS + 1):
            print(f"\n{'=' * 64}")
            print(
                f"  Iteration {iteration_number}/{NUMBER_OF_ITERATIONS}  "
                f"(success={successful_count}  failed={failed_count})"
            )
            print(f"{'=' * 64}")

            if iteration_number == 1:
                print("  Mode: FIXED")
                plan = _build_fixed_plan(planning_context, world, robot, milk_body)
            else:
                print("  Mode: JPT-SAMPLED")
                current_parameters = _sample_plan_parameters(joint_probability_tree)
                plan = _build_sampled_plan(
                    planning_context, current_parameters, world, robot, milk_body
                )

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

            print("\n  Resetting world state ...")
            _respawn_milk(world, milk_body)
            _respawn_robot(world, robot)

        # ── Final summary ──────────────────────────────────────────────────
        success_rate = 100 * successful_count // NUMBER_OF_ITERATIONS
        print(f"\n{'=' * 64}")
        print(f"  Run complete.")
        print(f"  Total iterations  : {NUMBER_OF_ITERATIONS}")
        print(f"  Successful plans  : {successful_count}  ({success_rate}%)")
        print(f"  Failed            : {failed_count}")
        print(f"  Batch 1 baseline  : 1742 / 5000 = 34%")
        print(f"  Database          : {DATABASE_URI}")
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
    pick_and_place_demo_jpt()