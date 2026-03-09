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

from krrood.entity_query_language.factories import probable, probable_variable, variable, variable_from
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


NUMBER_OF_ITERATIONS: int = 30000
"""Total number of plan iterations to run. Iteration 1 uses fixed parameters; all others sample."""

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///pick_and_place_results.db",
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

PICK_APPROACH_SAMPLING_BOUNDS: tuple[float, float, float, float] = (1.2, 1.8, -0.4, 0.4)
PLACE_APPROACH_SAMPLING_BOUNDS: tuple[float, float, float, float] = (3.2, 3.8, -0.4, 0.4)

PR2_URDF_PATH: str = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
MILK_STL_PATH: Path = Path(__file__).resolve().parents[2] / "resources" / "objects" / "milk.stl"

PICK_PLACE_ACTION_LABELS: list[str] = [
    "ParkArms (pre)",
    "Navigate to pick approach",
    "PickUp milk",
    "Navigate to place approach",
    "ParkArms (post)",
]

def _header_deepcopy(self, memo: Any) -> Header:
    """
    Custom deepcopy for Header that substitutes a safe default for the stamp
    field when it was never explicitly assigned.
    """
    if isinstance(self, type):
        return self
    stamp = getattr(self, "stamp", None) or datetime.datetime.now()
    return Header(
        frame_id=getattr(self, "frame_id", None),
        stamp=copy.deepcopy(stamp, memo),
        sequence=getattr(self, "sequence", 0),
    )


def _pose_stamped_deepcopy(self, memo: Any) -> PoseStamped:
    """
    Custom deepcopy for PoseStamped that delegates field copying to the patched
    Header deepcopy so stamp defaults are applied consistently.
    """
    if isinstance(self, type):
        return self
    return PoseStamped(
        copy.deepcopy(getattr(self, "pose", None), memo),
        copy.deepcopy(getattr(self, "header", None), memo),
    )


def _header_getattr(self, name: str) -> Any:
    """
    Fallback attribute accessor for Header fields that were never explicitly set.

    PyCRAM creates Header objects in many places without always initialising every
    field. The DAO serialiser reads stamp unconditionally on every persist call, so
    returning a safe default here prevents AttributeError from propagating through
    plan persistence for any partially-constructed Header.
    """
    defaults = {
        "stamp": lambda: datetime.datetime.now(),
        "sequence": lambda: 0,
        "frame_id": lambda: None,
    }
    if name in defaults:
        value = defaults[name]()
        object.__setattr__(self, name, value)
        return value
    raise AttributeError(name)


Header.__deepcopy__ = _header_deepcopy
Header.__getattr__ = _header_getattr
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy


def _patch_orm_numpy_array_type() -> None:
    """
    Patch the PyCRAM ORM numpy array TypeDecorator to tolerate None values.

    Without this patch, persisting any action whose execution_end_world_state was
    never assigned raises AttributeError because the TypeDecorator calls
    value.astype() unconditionally. The patch is found dynamically by inspecting
    loaded modules so that it survives refactors of the pycram.orm package.
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
        logger.warning("ORM numpy array TypeDecorator not found; None-guard patch skipped.")
        return

    original_process_bind_param = target_class.process_bind_param

    def process_bind_param_with_none_guard(self, value, dialect):
        if value is None:
            return None
        return original_process_bind_param(self, value, dialect)

    target_class.process_bind_param = process_bind_param_with_none_guard


_patch_orm_numpy_array_type()



@dataclass
class ActionEntry:
    """
    Pairs a concrete action instance with its probabilistic parameterization.

    The working distribution may be replaced by a spatially truncated version at
    the start of each iteration. The base distribution is a pristine deepcopy
    preserved at construction time so that truncation always begins from a
    structurally valid circuit, even after a previous iteration produced a
    zero-probability truncation result that would otherwise have corrupted the graph.
    """

    instance: Any
    parameterization: Parameterization
    distribution: ProbabilisticCircuit
    base_distribution: ProbabilisticCircuit = None


def _build_world_with_robot() -> tuple[World, PR2]:
    """
    Construct a minimal simulation world containing a scene root body and the PR2 robot.

    The scene root is an empty anchor body that guarantees the world graph has
    exactly one node with in-degree zero. The PR2 is loaded from its URDF and
    merged into the world at the configured initial pose.

    :return: The constructed World and the PR2 robot instance.
    """
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
    """
    Insert the map and odom_combined coordinate frames and connect the robot base via OmniDrive.

    After this call the kinematic hierarchy is:
      scene (root)
        └── map  (FixedConnection)
              └── odom_combined  (Connection6DoF)
                    └── base_footprint  (OmniDrive)
                          └── PR2 kinematic tree

    :param world: The simulation world.
    :param robot: The PR2 robot instance.
    """
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))

        world.add_body(map_body)
        world.add_body(odom_body)
        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))

        existing_base_connection = robot.root.parent_connection
        if existing_base_connection is not None:
            world.remove_connection(existing_base_connection)

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
    """
    Add the milk object at its configured initial position and attach a Milk semantic annotation.

    :param world: The simulation world.
    :return: The Body representing the milk object.
    """
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
    """
    Reset the milk object to its configured initial position.

    When the milk was attached to the robot gripper during pick, its parent
    connection must first be replaced with a free Connection6DoF rooted at the
    world root before the pose can be restored.

    :param world: The simulation world.
    :param milk_body: The milk Body to reset.
    """
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
    """
    Reset the robot base to its configured initial position.

    :param world: The simulation world.
    :param robot: The PR2 robot instance.
    """
    initial_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INITIAL_X, ROBOT_INITIAL_Y, 0.0, 0, 0, 0
    )
    with world.modify_world():
        base_connection = robot.root.parent_connection
        if base_connection is not None:
            base_connection.origin = initial_pose



def _create_database_session(database_uri: str) -> Session:
    """
    Open a SQLAlchemy session against the given URI and ensure all ORM tables exist.

    For PostgreSQL targets, additional dialect patches are applied to handle
    identifiers longer than 63 characters and numpy scalar parameter types.

    :param database_uri: A SQLAlchemy-compatible database connection URI.
    :return: An open Session ready for use.
    """
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

    def shorten_to_postgresql_limit(name: str, maximum_length: int = 63) -> str:
        if len(name) <= maximum_length:
            return name
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        return f"{name[:maximum_length - 9]}_{digest}"

    for table in Base.metadata.tables.values():
        shortened_name = shorten_to_postgresql_limit(table.name)
        if shortened_name != table.name:
            table.name = shortened_name
            table.fullname = shortened_name


def _register_postgresql_numpy_scalar_coercion(engine) -> None:
    import numpy
    from sqlalchemy import event

    def coerce_numpy_scalar_to_python(value):
        if isinstance(value, numpy.floating):
            return float(value)
        if isinstance(value, numpy.integer):
            return int(value)
        if isinstance(value, numpy.bool_):
            return bool(value)
        return value

    def coerce_parameter_values(parameters):
        if isinstance(parameters, dict):
            return {key: coerce_numpy_scalar_to_python(value) for key, value in parameters.items()}
        return parameters

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def coerce_numpy_parameters_before_execute(
        connection, cursor, statement, parameters, context, executemany
    ):
        if isinstance(parameters, dict):
            parameters = coerce_parameter_values(parameters)
        elif isinstance(parameters, (list, tuple)):
            parameters = type(parameters)(coerce_parameter_values(entry) for entry in parameters)
        return statement, parameters


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    """
    Persist a completed SequentialPlan to the database via the DAO layer.

    :param session: The active SQLAlchemy session.
    :param plan: The plan whose execution record should be stored.
    """
    session.add(to_dao(plan))
    session.commit()


# ---------------------------------------------------------------------------
# Pose construction
# ---------------------------------------------------------------------------

def _create_pose_stamped(
    x_coordinate: float,
    y_coordinate: float,
    z_coordinate: float,
    frame_id: Any,
) -> PoseStamped:
    """
    Build a PoseStamped with identity orientation at the specified position.

    :param x_coordinate: Position along the world x-axis in metres.
    :param y_coordinate: Position along the world y-axis in metres.
    :param z_coordinate: Position along the world z-axis in metres.
    :param frame_id: The reference frame body or identifier string.
    :return: A fully initialised PoseStamped.
    """
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x_coordinate, y=y_coordinate, z=z_coordinate),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame_id, stamp=datetime.datetime.now(), sequence=0),
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_fixed_plan(
    context: Context,
    world: World,
    robot: PR2,
    milk_body: Body,
) -> SequentialPlan:
    """
    Construct a fully deterministic pick-and-place plan using known-good parameters.

    Used for the first iteration to seed the database with at least one guaranteed
    successful execution before probabilistic sampling begins.

    :param context: The active planning context.
    :param world: The simulation world.
    :param robot: The PR2 robot instance.
    :param milk_body: The milk object body.
    :return: A SequentialPlan ready to be performed.
    """
    pick_approach_pose = _create_pose_stamped(PICK_APPROACH_X, PICK_APPROACH_Y, 0.0, world.root)
    place_approach_pose = _create_pose_stamped(PLACE_APPROACH_X, PLACE_APPROACH_Y, 0.0, world.root)
    place_target_pose = _create_pose_stamped(PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root)

    return SequentialPlan(
        context,
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=pick_approach_pose, keep_joint_states=False),
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
        ),
        NavigateAction(target_location=place_approach_pose, keep_joint_states=True),
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=Arms.RIGHT,
        ),
        ParkArmsAction(arm=Arms.BOTH),
    )


def _build_navigable_pose_description(robot: PR2) -> Any:
    """
    Construct a probable PoseStamped description with free x and y axes for sampling.

    The z axis is fixed to zero because the robot navigates on a flat floor.

    :param robot: The PR2 robot instance, used to bind the coordinate frame reference.
    :return: A probable PoseStamped description suitable for use in NavigateAction.
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=0),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot._world.root]), sequence=0),
    )


def _build_action_descriptions(world: World, robot: PR2, milk_variable: Any) -> list:
    """
    Build the five probabilistic action descriptions used for sampled iterations.

    PlaceAction is intentionally absent. Its arm and target pose are derived
    deterministically from the PickUpAction that was just executed, so sampling it
    independently would introduce arm mismatches and unreachable target poses.

    The five descriptions correspond to:
      0  ParkArmsAction  pre-grasp arm parking
      1  NavigateAction  to pick approach area
      2  PickUpAction    with sampled arm and grasp description
      3  NavigateAction  to place approach area
      4  ParkArmsAction  post-place arm parking

    :param world: The simulation world.
    :param robot: The PR2 robot instance.
    :param milk_variable: A krrood variable referencing the milk body.
    :return: A list of five probable action descriptions.
    """
    available_manipulators = world.get_semantic_annotations_by_type(Manipulator)
    return [
        probable_variable(ParkArmsAction)(
            arm=variable(Arms, [Arms.BOTH]),
        ),
        probable_variable(NavigateAction)(
            target_location=_build_navigable_pose_description(robot),
            keep_joint_states=False,
        ),
        probable_variable(PickUpAction)(
            object_designator=milk_variable,
            arm=variable(Arms, [Arms.LEFT, Arms.RIGHT]),
            grasp_description=probable(GraspDescription)(
                approach_direction=variable(ApproachDirection, [ApproachDirection.FRONT]),
                vertical_alignment=variable(VerticalAlignment, [VerticalAlignment.NoAlignment]),
                rotate_gripper=variable(bool, [False]),
                manipulation_offset=0.06,
                manipulator=variable(Manipulator, available_manipulators),
            ),
        ),
        probable_variable(NavigateAction)(
            target_location=_build_navigable_pose_description(robot),
            keep_joint_states=True,
        ),
        probable_variable(ParkArmsAction)(
            arm=variable(Arms, [Arms.BOTH]),
        ),
    ]


def _truncate_distribution_to_position_bounds(
    distribution: ProbabilisticCircuit,
    x_minimum: float,
    x_maximum: float,
    y_minimum: float,
    y_maximum: float,
) -> ProbabilisticCircuit:
    """
    Return a spatially truncated copy of the distribution bounded to the given x/y rectangle.

    The underlying log_truncated_in_place method destroys the circuit graph even
    when the queried region carries zero probability. To protect the caller this
    function always operates on a deepcopy, leaving the source distribution intact.
    When the bounded region has zero probability the original distribution is
    returned unchanged.

    :param distribution: The source circuit. Never mutated.
    :param x_minimum: Lower bound of the x interval in metres.
    :param x_maximum: Upper bound of the x interval in metres.
    :param y_minimum: Lower bound of the y interval in metres.
    :param y_maximum: Upper bound of the y interval in metres.
    :return: A new truncated circuit, or the original distribution if truncation failed.
    """
    from random_events.interval import closed
    from random_events.product_algebra import SimpleEvent
    from random_events.variable import Continuous

    x_variable = None
    y_variable = None
    for random_variable in distribution.variables:
        if isinstance(random_variable, Continuous):
            if random_variable.name.endswith(".position.x") or random_variable.name.endswith(".x"):
                x_variable = random_variable
            elif random_variable.name.endswith(".position.y") or random_variable.name.endswith(".y"):
                y_variable = random_variable

    if x_variable is None or y_variable is None:
        logger.warning(
            "Position x/y variables not found in distribution; truncation skipped. "
            "Variables present: %s",
            [v.name for v in distribution.variables],
        )
        return distribution

    bounding_event = SimpleEvent(
        {
            x_variable: closed(x_minimum, x_maximum),
            y_variable: closed(y_minimum, y_maximum),
        }
    ).as_composite_set()
    full_event = bounding_event.fill_missing_variables_pure(distribution.variables)

    candidate = copy.deepcopy(distribution)
    truncated_distribution, _ = candidate.log_truncated_in_place(full_event)

    if truncated_distribution is None:
        logger.warning(
            "Bounding region x=[%s, %s] y=[%s, %s] has zero probability; "
            "returning untruncated distribution.",
            x_minimum, x_maximum, y_minimum, y_maximum,
        )
        return distribution

    return truncated_distribution


def _build_action_entry(
    description: Any,
    sampling_bounds: Optional[tuple[float, float, float, float]] = None,
) -> ActionEntry:
    """
    Translate a probable action description into a concrete ActionEntry.

    When sampling_bounds are given the distribution is immediately truncated to
    that x/y rectangle. A pristine deepcopy is stored as the base distribution
    so re-truncation in subsequent iterations always starts from a valid circuit.

    :param description: A probable_variable description produced by _build_action_descriptions.
    :param sampling_bounds: Optional (x_min, x_max, y_min, y_max) spatial constraint.
    :return: A fully initialised ActionEntry.
    """
    instance = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution = parameterization.create_fully_factorized_distribution()

    if sampling_bounds is not None:
        x_minimum, x_maximum, y_minimum, y_maximum = sampling_bounds
        distribution = _truncate_distribution_to_position_bounds(
            distribution, x_minimum, x_maximum, y_minimum, y_maximum
        )

    return ActionEntry(
        instance=instance,
        parameterization=parameterization,
        distribution=distribution,
        base_distribution=copy.deepcopy(distribution),
    )


def _sample_parameters_into_action(entry: ActionEntry) -> None:
    """
    Draw one sample from the entry's distribution and apply the values to its action instance.

    :param entry: The ActionEntry whose action instance will be parameterised.
    """
    raw_sample = entry.distribution.sample(1)[0]
    named_sample = entry.parameterization.create_assignment_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )
    entry.parameterization.parameterize_object_with_sample(entry.instance, named_sample)


def _build_sampled_plan(
    context: Context,
    action_entries: list[ActionEntry],
    world: World,
    milk_body: Body,
) -> SequentialPlan:
    """
    Sample all action entries and assemble a complete six-step pick-and-place plan.

    The five sampled entries cover:
      0  ParkArmsAction  pre-grasp
      1  NavigateAction  to pick approach
      2  PickUpAction    arm and grasp description sampled
      3  NavigateAction  to place approach
      4  ParkArmsAction  post-place

    PlaceAction is built deterministically after sampling by reading the arm that
    was selected for PickUpAction and pairing it with the fixed place target pose.
    This ensures the arm holding the milk is always the arm used to place it, and
    that the place target is always within reach.

    :param context: The active planning context.
    :param action_entries: The five ActionEntry objects from _build_action_entry.
    :param world: The simulation world, used to resolve the place target frame.
    :param milk_body: The milk object body used as the PlaceAction object designator.
    :return: A SequentialPlan ready to be performed.
    """
    for label, entry in zip(PICK_PLACE_ACTION_LABELS, action_entries):
        _sample_parameters_into_action(entry)
        logger.debug("Sampled: %s", label)

    pickup_action: PickUpAction = action_entries[2].instance
    place_target_pose = _create_pose_stamped(
        PLACE_TARGET_X, PLACE_TARGET_Y, PLACE_TARGET_Z, world.root
    )

    return SequentialPlan(
        context,
        action_entries[0].instance,
        action_entries[1].instance,
        action_entries[2].instance,
        action_entries[3].instance,
        PlaceAction(
            object_designator=milk_body,
            target_location=place_target_pose,
            arm=pickup_action.arm,
        ),
        action_entries[4].instance,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_and_place_demo() -> None:
    """
    Run the pick-and-place probabilistic reasoning demonstration.

    Executes NUMBER_OF_ITERATIONS plans in a minimal world containing only the
    PR2 robot and a milk carton. The first iteration uses fixed deterministic
    parameters to seed the database. All subsequent iterations sample navigation
    approach positions and pick-up arm assignments from probabilistic distributions.
    The place target position and place arm are always fixed to guarantee reliable
    placement regardless of the sampled pick configuration.

    Successful executions are persisted to DATABASE_URI. Both the milk object and
    the robot are respawned to their initial positions after every iteration.

    World kinematic structure:
      scene (root)
        └── map  (FixedConnection)
              └── odom_combined  (Connection6DoF)
                    └── base_footprint  (OmniDrive)
                          └── PR2 kinematic tree
        └── milk_1  (Connection6DoF)
    """
    world, robot = _build_world_with_robot()
    _add_localization_frames(world, robot)
    milk_body = _add_milk_to_world(world)

    database_session = _create_database_session(DATABASE_URI)

    rclpy.init()
    ros_node = rclpy.create_node("pick_and_place_demo_node")
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()

    try:
        TFPublisher(_world=world, node=ros_node)
        VizMarkerPublisher(_world=world, node=ros_node)

        planning_context = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        action_descriptions = _build_action_descriptions(world, robot, milk_variable)
        action_entries: list[ActionEntry] = [
            _build_action_entry(action_descriptions[0]),
            _build_action_entry(action_descriptions[1], PICK_APPROACH_SAMPLING_BOUNDS),
            _build_action_entry(action_descriptions[2]),
            _build_action_entry(action_descriptions[3], PLACE_APPROACH_SAMPLING_BOUNDS),
            _build_action_entry(action_descriptions[4]),
        ]

        successful_count = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            if iteration == 1:
                print(f"\n[{iteration}/{NUMBER_OF_ITERATIONS}] Fixed iteration (deterministic parameters)")
                plan = _build_fixed_plan(planning_context, world, robot, milk_body)
            else:
                print(f"\n[{iteration}/{NUMBER_OF_ITERATIONS}] Probabilistic iteration (sampled parameters)")
                plan = _build_sampled_plan(planning_context, action_entries, world, milk_body)

            with simulated_robot:
                try:
                    plan.perform()
                    _persist_plan(database_session, plan)
                    successful_count += 1
                    print(f"  -> Success  ({successful_count}/{NUMBER_OF_ITERATIONS} stored)")
                except Exception as exception:
                    traceback.print_exc()
                    logger.error(
                        "Iteration %d failed: %s: %s",
                        iteration, type(exception).__name__, exception,
                    )
                finally:
                    _respawn_milk(world, milk_body)
                    _respawn_robot(world, robot)

        print(f"\nCompleted. {successful_count}/{NUMBER_OF_ITERATIONS} plans stored in {DATABASE_URI}.")

    finally:
        database_session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        ros_spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_and_place_demo()