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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body



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

JPT_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")

JPT_MIN_SAMPLES_PER_LEAF: int = 25


# JPT variable definitions
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
    attribute_defaults = {
        "stamp":    lambda: datetime.datetime.now(),
        "sequence": lambda: 0,
        "frame_id": lambda: None,
    }
    if attribute_name in attribute_defaults:
        value = attribute_defaults[attribute_name]()
        object.__setattr__(self, attribute_name, value)
        return value
    raise AttributeError(attribute_name)


Header.__deepcopy__      = _header_deepcopy
Header.__getattr__       = _header_getattr
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy


def _patch_orm_numpy_array_type() -> None:
    """
    Patch the PyCRAM ORM numpy TypeDecorator so that None values are passed
    through without calling astype(), which would otherwise raise an
    AttributeError when a plan action has no associated array data.
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
        print("  [patch] WARNING: ORM numpy TypeDecorator not found; None-guard patch skipped.")
        return

    original_process_bind_param = target_class.process_bind_param

    def _none_guarded_process_bind_param(self, value, dialect):
        if value is None:
            return None
        return original_process_bind_param(self, value, dialect)

    target_class.process_bind_param = _none_guarded_process_bind_param
    print(f"  [patch] Patched {target_class.__name__}.process_bind_param to handle None.")


_patch_orm_numpy_array_type()



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
    print("  [world] Localization frames OK.")


def _add_milk_to_world(world: World) -> Body:
    print(f"  [world] Adding milk at ({MILK_INITIAL_X}, {MILK_INITIAL_Y}, {MILK_INITIAL_Z}) ...")
    mesh      = FileMesh.from_file(str(MILK_STL_PATH))
    milk_body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    initial_milk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_INITIAL_X, MILK_INITIAL_Y, MILK_INITIAL_Z, 0, 0, 0
    )
    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=milk_body, world=world
        )
        world.add_connection(milk_connection)
        milk_connection.origin = initial_milk_pose
        world.add_semantic_annotation(Milk(root=milk_body))

    print("  [world] Milk added.")
    return milk_body


def _respawn_milk(world: World, milk_body: Body) -> None:
    """Reset the milk carton to its original pose, re-attaching it to the world root
    if it was previously attached to the robot gripper during a pick-up."""
    initial_milk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
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
            free_connection.origin = initial_milk_pose
        elif current_connection is not None:
            current_connection.origin = initial_milk_pose
        else:
            free_connection = Connection6DoF.create_with_dofs(
                parent=world.root, child=milk_body, world=world
            )
            world.add_connection(free_connection)
            free_connection.origin = initial_milk_pose
    print("  [respawn] Milk reset.")


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

    def shorten_to_postgres_limit(name: str, character_limit: int = 63) -> str:
        if len(name) <= character_limit:
            return name
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:character_limit - 9]}_{digest}"

    for table in Base.metadata.tables.values():
        shortened_name = shorten_to_postgres_limit(table.name)
        if shortened_name != table.name:
            table.name     = shortened_name
            table.fullname = shortened_name


def _register_postgresql_numpy_scalar_coercion(engine: Any) -> None:
    import numpy
    from sqlalchemy import event

    def _coerce_numpy_scalar_to_python(value: Any) -> Any:
        if isinstance(value, numpy.floating): return float(value)
        if isinstance(value, numpy.integer):  return int(value)
        if isinstance(value, numpy.bool_):    return bool(value)
        return value

    def _coerce_parameter_dict_or_list(parameters: Any) -> Any:
        if isinstance(parameters, dict):
            return {
                key: _coerce_numpy_scalar_to_python(value)
                for key, value in parameters.items()
            }
        return parameters

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _coerce_before_cursor_execute(
        connection, cursor, statement, parameters, context, executemany
    ):
        if isinstance(parameters, dict):
            parameters = _coerce_parameter_dict_or_list(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(
                _coerce_parameter_dict_or_list(parameter_set)
                for parameter_set in parameters
            )
        return statement, parameters


def _persist_plan_to_database(database_session: Session, plan: SequentialPlan) -> None:
    print("  [db] Persisting plan ...")
    database_session.add(to_dao(plan))
    database_session.commit()
    print("  [db] Plan committed OK.")


def _create_pose_stamped(
    x: float,
    y: float,
    z: float,
    reference_frame: Any,
) -> PoseStamped:
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(
            frame_id=reference_frame,
            stamp=datetime.datetime.now(),
            sequence=0,
        ),
    )


def _load_joint_probability_tree(model_path: str) -> JointProbabilityTree:
    """
    Load the pre-fitted JPT from disk.

    The model was fitted by fit_jpt.py over the 1742 successful Batch 1 plans.
    It encodes the joint distribution over the following variables:
        pick_approach_x, pick_approach_y,
        place_approach_x, place_approach_y,
        milk_end_x, milk_end_y, milk_end_z,
        pick_arm
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
    world: World,
) -> PlanParameters:
    """
    Draw one sample from the JPT and return it as a PlanParameters record.

    All five plan variables — pick_approach_x/y, place_approach_x/y, and
    pick_arm — are drawn in a single call, so their values reflect the learned
    joint distribution of successful executions rather than being drawn
    independently from uniform bounds as in Batch 1.

    Safety clipping is applied after sampling to guarantee the positions stay
    within the original Batch 1 bounds in case the JPT extrapolates slightly
    outside the training data range.
    """
    # sample() returns a numpy array shaped (n_samples, n_variables),
    # with columns ordered identically to JPT_VARIABLES.
    sample_array = joint_probability_tree.sample(1)
    sample_row   = sample_array[0]

    variable_names = [variable.name for variable in JPT_VARIABLES]
    sample_by_name = dict(zip(variable_names, sample_row))

    # Map the arm label to the Arms enum. Some JPT versions return the integer
    # domain index rather than the string label for symbolic variables.
    arm_label = sample_by_name["pick_arm"]
    if isinstance(arm_label, (int, float)):
        arm_label = ArmChoiceDomain.labels[int(arm_label)]
    pick_arm = Arms.LEFT if arm_label == "LEFT" else Arms.RIGHT

    pick_x_min,  pick_x_max,  pick_y_min,  pick_y_max  = PICK_APPROACH_SAMPLING_BOUNDS
    place_x_min, place_x_max, place_y_min, place_y_max = PLACE_APPROACH_SAMPLING_BOUNDS

    return PlanParameters(
        pick_approach_x  = float(max(pick_x_min,  min(pick_x_max,  sample_by_name["pick_approach_x"]))),
        pick_approach_y  = float(max(pick_y_min,  min(pick_y_max,  sample_by_name["pick_approach_y"]))),
        place_approach_x = float(max(place_x_min, min(place_x_max, sample_by_name["place_approach_x"]))),
        place_approach_y = float(max(place_y_min, min(place_y_max, sample_by_name["place_approach_y"]))),
        pick_arm         = pick_arm,
    )



def _build_fixed_plan(
    planning_context: Context,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
) -> SequentialPlan:
    """
    Build the fixed, deterministic plan used for iteration 1.

    Using a known-good seed on the first iteration confirms that the world,
    robot, and database are correctly initialised before probabilistic sampling
    begins on subsequent iterations.
    """
    seed_arm = Arms.RIGHT

    pick_approach_pose  = _create_pose_stamped(PICK_APPROACH_X,  PICK_APPROACH_Y,  0.0,            world.root)
    place_approach_pose = _create_pose_stamped(PLACE_APPROACH_X, PLACE_APPROACH_Y, 0.0,            world.root)
    place_target_pose   = _create_pose_stamped(PLACE_TARGET_X,   PLACE_TARGET_Y,   PLACE_TARGET_Z, world.root)

    print(f"  [plan] Fixed seed parameters:")
    print(f"         pick  approach : ({PICK_APPROACH_X}, {PICK_APPROACH_Y})")
    print(f"         place approach : ({PLACE_APPROACH_X}, {PLACE_APPROACH_Y})")
    print(f"         place target   : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"         arm            : {seed_arm}")

    return SequentialPlan(
        planning_context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_approach_pose,  keep_joint_states=False),
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
        ),
        NavigateAction(target_location=place_approach_pose, keep_joint_states=True),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=seed_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _build_jpt_sampled_plan(
    planning_context: Context,
    plan_parameters:  PlanParameters,
    world:            World,
    robot:            PR2,
    milk_body:        Body,
) -> SequentialPlan:
    """
    Assemble a pick-and-place plan from parameters drawn from the JPT.

    The plan structure is identical to the Batch 1 sampled plan. The only
    difference is that the parameters originate from the JPT joint distribution
    rather than from independent uniform sampling within fixed bounds.
    """
    manipulator = (
        robot.right_arm.manipulator
        if plan_parameters.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )

    pick_approach_pose = _create_pose_stamped(
        plan_parameters.pick_approach_x,
        plan_parameters.pick_approach_y,
        0.0,
        world.root,
    )
    place_approach_pose = _create_pose_stamped(
        plan_parameters.place_approach_x,
        plan_parameters.place_approach_y,
        0.0,
        world.root,
    )
    place_target_pose = _create_pose_stamped(
        PLACE_TARGET_X,
        PLACE_TARGET_Y,
        PLACE_TARGET_Z,
        world.root,
    )

    print(f"  [plan] JPT-sampled parameters:")
    print(f"         pick  approach : ({plan_parameters.pick_approach_x:.3f}, {plan_parameters.pick_approach_y:.3f})")
    print(f"         place approach : ({plan_parameters.place_approach_x:.3f}, {plan_parameters.place_approach_y:.3f})")
    print(f"         place target   : ({PLACE_TARGET_X}, {PLACE_TARGET_Y}, {PLACE_TARGET_Z})")
    print(f"         arm            : {plan_parameters.pick_arm}")

    return SequentialPlan(
        planning_context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_approach_pose,  keep_joint_states=False),
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
        ),
        NavigateAction(target_location=place_approach_pose, keep_joint_states=True),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=plan_parameters.pick_arm,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def pick_and_place_demo_jpt() -> None:
    """
    Batch 2: pick-and-place demo with JPT-informed parameter sampling.

    Identical world setup to Batch 1 (pick_and_place_demo.py) except that
    approach positions and arm choice are sampled from the pre-fitted JPT
    rather than independently from uniform distributions.

    The JPT was fitted on the 1742 successful plans from Batch 1. Sampling
    from it concentrates the robot's parameter choices in the region of the
    parameter space that historically led to successful executions.

    Results are stored in the same database as Batch 1. Batch 2 plans can be
    distinguished from Batch 1 plans by their higher plan_id values.

    Expected outcome: meaningfully higher success rate than the Batch 1
    baseline of 34.8%.
    """
    print("=" * 64)
    print("  pick_and_place_demo_batch_two")
    print(f"  NUMBER_OF_ITERATIONS = {NUMBER_OF_ITERATIONS}")
    print(f"  PLACE_TARGET_Z       = {PLACE_TARGET_Z}")
    print(f"  DATABASE_URI         = {DATABASE_URI}")
    print(f"  JPT_MODEL_PATH       = {JPT_MODEL_PATH}")
    print("=" * 64)

    print("\n[1/5] Building world ...")
    world, robot = _build_world_with_robot()
    _add_localization_frames(world, robot)
    milk_body = _add_milk_to_world(world)

    print("\n[2/5] Connecting to database ...")
    database_session = _create_database_session(DATABASE_URI)

    print("\n[3/5] Loading JPT model ...")
    joint_probability_tree = _load_joint_probability_tree(JPT_MODEL_PATH)

    print("\n[4/5] Initialising ROS ...")
    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_demo_batch_two_node")
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
            print(
                f"  Iteration {iteration} / {NUMBER_OF_ITERATIONS}  "
                f"(success={successful_count}  failed={failed_count})"
            )
            print(f"{'=' * 64}")

            if iteration == 1:
                print("  Mode: FIXED (deterministic seed)")
                plan = _build_fixed_plan(planning_context, world, robot, milk_body)
            else:
                print("  Mode: JPT-SAMPLED")
                plan_parameters = _sample_plan_parameters_from_jpt(joint_probability_tree, world)
                plan = _build_jpt_sampled_plan(planning_context, plan_parameters, world, robot, milk_body)

            print("\n  Executing plan ...")
            with simulated_robot:
                execution_succeeded = False
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
                    traceback_lines = traceback.format_exc().strip().splitlines()
                    for traceback_line in traceback_lines[-3:]:
                        print(f"    {traceback_line}")

                finally:
                    # Only persist fully completed plans — never store partial executions.
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
                        except Exception as database_error:
                            print(f"  [db] ERROR persisting plan: {database_error}")
                            traceback.print_exc()

                    print("\n  Resetting world state ...")
                    _respawn_milk(world, milk_body)
                    _respawn_robot(world, robot)

        # ── Final summary ──────────────────────────────────────────────────
        batch_two_success_rate = 100 * successful_count // NUMBER_OF_ITERATIONS
        print(f"\n{'=' * 64}")
        print(f"  Run complete.")
        print(f"  Total iterations    : {NUMBER_OF_ITERATIONS}")
        print(f"  Successful plans    : {successful_count}  ({batch_two_success_rate}%)")
        print(f"  Failed              : {failed_count}")
        print(f"  Batch 1 baseline    : 1742 / 5000 = 34%")
        print(f"  Database            : {DATABASE_URI}")
        print(f"{'=' * 64}")

        # Cross-check the in-memory counter against the actual database row count.
        try:
            from sqlalchemy import text
            database_row_count = database_session.execute(
                text('SELECT COUNT(*) FROM "SequentialPlanDAO"')
            ).scalar()
            print(f"  DB row count (SequentialPlanDAO) : {database_row_count}")
            if database_row_count >= successful_count:
                print("  Counter is consistent with the database.")
            else:
                print(
                    f"  WARNING: in-memory counter ({successful_count}) "
                    f"exceeds database row count ({database_row_count})."
                )
        except Exception as count_error:
            print(f"  [db] Could not verify row count: {count_error}")

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo_jpt()