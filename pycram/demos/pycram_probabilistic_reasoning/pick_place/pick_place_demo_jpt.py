from __future__ import annotations

import copy
import datetime
import inspect
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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

# ── JPT imports ───────────────────────────────────────────────────────────────
from jpt.distributions.univariate.multinomial import OrderedDictProxy
from jpt.distributions.univariate import Multinomial
from jpt.variables import NumericVariable, SymbolicVariable
from jpt.trees import JPT as JPTModel

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF, FixedConnection, OmniDrive
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants  (identical to pick_and_place_demo.py — do not change)
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

# Kept for reference / clipping safety checks — JPT samples within these by
# construction because it was fitted on data that already satisfied them.
PICK_APPROACH_SAMPLING_BOUNDS:  tuple[float, float, float, float] = (1.2, 1.8, -0.4, 0.4)
PLACE_APPROACH_SAMPLING_BOUNDS: tuple[float, float, float, float] = (3.2, 3.8, -0.4, 0.4)

PR2_URDF_PATH: str = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
MILK_STL_PATH: Path = Path(__file__).resolve().parents[2] / "resources" / "objects" / "milk.stl"

# Path to the fitted JPT model produced by fit_jpt.py
JPT_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "pick_and_place_jpt.json")

# ---------------------------------------------------------------------------
# JPT variable definitions  (must match fit_jpt.py exactly)
# ---------------------------------------------------------------------------

ArmDomain = type("ArmDomain", (Multinomial,), {
    "values": OrderedDictProxy([("LEFT", 0), ("RIGHT", 1)]),
    "labels": OrderedDictProxy([(0, "LEFT"),  (1, "RIGHT")]),
})

JPT_VARIABLES = [
    NumericVariable("pick_approach_x",  precision=0.005),
    NumericVariable("pick_approach_y",  precision=0.005),
    NumericVariable("place_approach_x", precision=0.005),
    NumericVariable("place_approach_y", precision=0.005),
    NumericVariable("milk_end_x",       precision=0.001),
    NumericVariable("milk_end_y",       precision=0.001),
    NumericVariable("milk_end_z",       precision=0.0005),
    SymbolicVariable("pick_arm",        domain=ArmDomain),
]

# ---------------------------------------------------------------------------
# Monkey-patches  (identical to pick_and_place_demo.py)
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


Header.__deepcopy__ = _header_deepcopy
Header.__getattr__  = _header_getattr
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy


def _patch_orm_numpy_array_type() -> None:
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
        logger.warning("ORM numpy array TypeDecorator not found; None-guard patch skipped.")
        return
    original = target_class.process_bind_param
    def _guarded(self, value, dialect):
        if value is None:
            return None
        return original(self, value, dialect)
    target_class.process_bind_param = _guarded


_patch_orm_numpy_array_type()

# ---------------------------------------------------------------------------
# Sampled parameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class SampledParameters:
    """
    One complete set of plan parameters sampled from the JPT.

    All five values are drawn in a single JPT sample call, so their joint
    distribution reflects the correlations learned from Batch 1.
    """
    pick_approach_x:  float
    pick_approach_y:  float
    place_approach_x: float
    place_approach_y: float
    pick_arm:         Arms

# ---------------------------------------------------------------------------
# World construction  (identical to pick_and_place_demo.py)
# ---------------------------------------------------------------------------

def _build_world_with_robot() -> tuple[World, PR2]:
    scene_world = World(name="pick_and_place_scene")
    scene_root_body = Body(name=PrefixedName("scene"))
    with scene_world.modify_world():
        scene_world.add_kinematic_structure_entity(scene_root_body)
    pr2_world = URDFParser.from_file(PR2_URDF_PATH).parse()
    robot = PR2.from_world(pr2_world)
    robot_initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with scene_world.modify_world():
        scene_world.merge_world_at_pose(pr2_world, robot_initial_pose)
    return scene_world, robot


def _add_localization_frames(world: World, robot: PR2) -> None:
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


def _add_milk_to_world(world: World) -> Body:
    mesh = FileMesh.from_file(str(MILK_STL_PATH))
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


def _respawn_robot(world: World, robot: PR2) -> None:
    initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with world.modify_world():
        base_connection = robot.root.parent_connection
        if base_connection is not None:
            base_connection.origin = initial_pose

# ---------------------------------------------------------------------------
# Database  (identical to pick_and_place_demo.py)
# ---------------------------------------------------------------------------

def _create_database_session(database_uri: str) -> Session:
    engine = create_engine(database_uri)
    if "postgresql" in database_uri or "postgres" in database_uri:
        _apply_postgresql_patches(engine)
    Base.metadata.create_all(bind=engine, checkfirst=True)
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
        if isinstance(value, numpy.floating):  return float(value)
        if isinstance(value, numpy.integer):   return int(value)
        if isinstance(value, numpy.bool_):     return bool(value)
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
    session.add(to_dao(plan))
    session.commit()

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
# JPT loading and sampling  ← THE KEY NEW PART
# ---------------------------------------------------------------------------

def _load_jpt(path: str) -> JPTModel:
    """
    Load the pre-fitted JPT from disk.

    The model was fitted by fit_jpt.py over 1742 successful Batch 1 plans.
    It encodes the joint distribution over:
        pick_approach_x/y, place_approach_x/y, milk_end_x/y/z, pick_arm
    """
    logger.info("Loading JPT from %s", path)
    jpt = JPTModel(variables=JPT_VARIABLES, min_samples_leaf=25)
    jpt = jpt.load(path)
    logger.info("JPT loaded. Leaves: %d", len(jpt.leaves))
    return jpt


def _sample_parameters_from_jpt(jpt: JPTModel, world: World) -> SampledParameters:
    """
    Draw one sample from the JPT and return it as a SampledParameters record.

    The JPT samples all five plan variables jointly — pick_approach_x/y,
    place_approach_x/y, and pick_arm — in a single call. Their values are
    therefore drawn from the learned joint distribution, not independently
    from uniform bounds.

    The arm string from the JPT ("LEFT" or "RIGHT") is mapped to the
    Arms enum expected by PyCRAM.

    Safety clipping is applied after sampling to guarantee the positions
    stay within the original bounds in case the JPT extrapolates slightly
    outside the training data range.
    """
    # Sample one row from the JPT (returns a dict: variable -> value)
    sample = jpt.sample(1, columns=["pick_approach_x", "pick_approach_y",
                                     "place_approach_x", "place_approach_y",
                                     "pick_arm"]).iloc[0]

    # Map arm string to Arms enum
    arm_str = sample["pick_arm"]
    pick_arm = Arms.LEFT if arm_str == "LEFT" else Arms.RIGHT

    # Safety clip to original bounds
    px_min, px_max, py_min, py_max = PICK_APPROACH_SAMPLING_BOUNDS
    lx_min, lx_max, ly_min, ly_max = PLACE_APPROACH_SAMPLING_BOUNDS

    return SampledParameters(
        pick_approach_x  = float(max(px_min, min(px_max, sample["pick_approach_x"]))),
        pick_approach_y  = float(max(py_min, min(py_max, sample["pick_approach_y"]))),
        place_approach_x = float(max(lx_min, min(lx_max, sample["place_approach_x"]))),
        place_approach_y = float(max(ly_min, min(ly_max, sample["place_approach_y"]))),
        pick_arm         = pick_arm,
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
    """Deterministic seed plan for iteration 1."""
    fixed_arm = Arms.RIGHT
    pick_approach_pose  = _create_pose_stamped(PICK_APPROACH_X,  PICK_APPROACH_Y,  0.0,            world.root)
    place_approach_pose = _create_pose_stamped(PLACE_APPROACH_X, PLACE_APPROACH_Y, 0.0,            world.root)
    place_target_pose   = _create_pose_stamped(PLACE_TARGET_X,   PLACE_TARGET_Y,   PLACE_TARGET_Z, world.root)
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


def _build_jpt_plan(
    context:   Context,
    params:    SampledParameters,
    world:     World,
    robot:     PR2,
    milk_body: Body,
) -> SequentialPlan:
    """
    Assemble a plan from parameters drawn from the JPT.

    Structurally identical to _build_sampled_plan in pick_and_place_demo.py.
    The only difference is that params comes from the JPT rather than from
    independent uniform distributions.
    """
    manipulator = (
        robot.right_arm.manipulator if params.pick_arm == Arms.RIGHT
        else robot.left_arm.manipulator
    )
    pick_approach_pose  = _create_pose_stamped(params.pick_approach_x,  params.pick_approach_y,  0.0,            world.root)
    place_approach_pose = _create_pose_stamped(params.place_approach_x, params.place_approach_y, 0.0,            world.root)
    place_target_pose   = _create_pose_stamped(PLACE_TARGET_X,          PLACE_TARGET_Y,          PLACE_TARGET_Z, world.root)
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

def pick_and_place_demo_jpt() -> None:
    """
    Batch 2: pick-and-place demo with JPT-informed parameter sampling.

    Identical setup to pick_and_place_demo.py (Batch 1) except that approach
    positions and arm choice are sampled from the fitted JPT rather than from
    independent uniform distributions.

    The JPT was fitted on the 1742 successful plans from Batch 1. Sampling
    from it means the robot draws parameters from the joint posterior of
    successful executions — i.e. it concentrates its samples in the region
    of parameter space that historically led to success.

    Results are stored in the same database as Batch 1. Batch 2 plans can be
    distinguished by their higher plan_id values.

    Expected outcome: higher success rate than Batch 1 (34.8%) because the
    sampler avoids regions of the parameter space that were rarely successful.
    """
    world, robot = _build_world_with_robot()
    _add_localization_frames(world, robot)
    milk_body = _add_milk_to_world(world)

    database_session = _create_database_session(DATABASE_URI)

    # Load the JPT once at startup — reused for all iterations
    jpt = _load_jpt(JPT_MODEL_PATH)

    rclpy.init()
    ros_node        = rclpy.create_node("pick_and_place_demo_jpt_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context = Context(world, robot, None)

        successful_count = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):

            if iteration == 1:
                logger.info(
                    "[%d/%d] Fixed iteration (deterministic seed)",
                    iteration, NUMBER_OF_ITERATIONS,
                )
                plan = _build_fixed_plan(planning_context, world, robot, milk_body)

            else:
                # Sample all parameters jointly from the JPT
                params = _sample_parameters_from_jpt(jpt, world)
                logger.info(
                    "[%d/%d] JPT sample — pick=(%.3f, %.3f)  place=(%.3f, %.3f)  arm=%s",
                    iteration, NUMBER_OF_ITERATIONS,
                    params.pick_approach_x, params.pick_approach_y,
                    params.place_approach_x, params.place_approach_y,
                    params.pick_arm,
                )
                plan = _build_jpt_plan(
                    planning_context, params, world, robot, milk_body
                )

            with simulated_robot:
                try:
                    plan.perform()
                    _persist_plan(database_session, plan)
                    successful_count += 1
                    logger.info("  -> Success (%d stored so far)", successful_count)
                except Exception as exc:
                    traceback.print_exc()
                    logger.error(
                        "Iteration %d failed: %s: %s",
                        iteration, type(exc).__name__, exc,
                    )
                finally:
                    _respawn_milk(world, milk_body)
                    _respawn_robot(world, robot)

        logger.info(
            "Batch 2 complete. %d/%d plans stored (%.1f%% success rate).",
            successful_count, NUMBER_OF_ITERATIONS,
            100.0 * successful_count / NUMBER_OF_ITERATIONS,
        )
        logger.info(
            "Batch 1 reference: 1742/5000 = 34.8%%"
        )

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo_jpt()