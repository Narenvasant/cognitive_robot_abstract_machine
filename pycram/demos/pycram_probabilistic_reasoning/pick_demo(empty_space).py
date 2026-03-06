import datetime
import enum
import os
import sys
import time
import threading
import copy
from dataclasses import dataclass
from typing import Any, List, Optional

import rclpy
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import (
    PoseStamped,
    PyCramPose,
    PyCramVector3,
    PyCramQuaternion,
    Header,
)

# ---- Monkey-patch for Header and PoseStamped deepcopy ------------------------
def _header_deepcopy(self, memo: Any) -> Header:
    if isinstance(self, type):
        return self
    stamp = getattr(self, "stamp", None)
    if stamp is None:
        stamp = datetime.datetime.now()
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

Header.__deepcopy__ = _header_deepcopy
PoseStamped.__deepcopy__ = _pose_stamped_deepcopy


def _header_getattr(self, name: str) -> Any:
    # Called only when normal attribute lookup fails (attribute was never set).
    # PyCRAM creates Header objects in many places without always setting every
    # field. The DAO serialiser calls getattr(header, 'stamp') unconditionally,
    # so returning a safe default here makes it structurally impossible for the
    # database persist step to raise AttributeError on any known Header field.
    if name == "stamp":
        value = datetime.datetime.now()
        object.__setattr__(self, "stamp", value)
        return value
    if name == "sequence":
        object.__setattr__(self, "sequence", 0)
        return 0
    if name == "frame_id":
        object.__setattr__(self, "frame_id", None)
        return None
    raise AttributeError(name)


Header.__getattr__ = _header_getattr
# ------------------------------------------------------------------------------

from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import ParkArmsAction, NavigateAction, PickUpAction
from pycram.orm.ormatic_interface import *

# ---- Patch numpy-array type's process_bind_param to handle None -----------
# pycram's ORM defines a custom SQLAlchemy type that serialises numpy arrays
# to BYTEA.  Its process_bind_param calls value.astype(np.float64) without a
# None guard.  When an action's execution_end_world_state is None (e.g. the
# action recorded a start but not an end state), this raises AttributeError
# and blocks the entire commit.  We wrap the method to short-circuit on None.
def _patch_numpy_array_type() -> None:
    import numpy as np
    import importlib, inspect, sqlalchemy.types as sa_types

    target = None
    # Walk every class imported via ormatic_interface / pycram.orm looking for
    # the one whose process_bind_param calls .astype
    for module_name in list(sys.modules):
        if "pycram" not in module_name and "orm" not in module_name:
            continue
        mod = sys.modules[module_name]
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, sa_types.TypeDecorator)
                and hasattr(obj, "process_bind_param")
                and "astype" in inspect.getsource(obj.process_bind_param)
            ):
                target = obj
                break
        if target is not None:
            break

    if target is None:
        print("  [orm patch] WARNING: could not find numpy array type to patch.")
        return

    original = target.process_bind_param

    def _safe_process_bind_param(self, value, dialect):
        if value is None:
            return None
        return original(self, value, dialect)

    target.process_bind_param = _safe_process_bind_param
    print(f"  [orm patch] Patched {target.__name__}.process_bind_param to handle None.")

_patch_numpy_array_type()
# ---------------------------------------------------------------------------


from krrood.entity_query_language.factories import (
    probable,
    probable_variable,
    variable,
    variable_from,
)
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import create_engine
from krrood.probabilistic_knowledge.parameterizer import (
    MatchParameterizer,
    Parameterization,
)
from krrood.probabilistic_knowledge.probable_variable import MatchToInstanceTranslator
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


NUMBER_OF_ITERATIONS: int = 50
"""Total number of plan iterations. Iteration 1 uses fixed values; 2+ use sampling."""

DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///pick_only_results.db",
)

ROBOT_INIT_X: float = 0.0
ROBOT_INIT_Y: float = 0.0

MILK_X: float = 2.4
MILK_Y: float = 0.0
MILK_Z: float = 1.01

APPROACH_X: float = 1.6
APPROACH_Y: float = 0.0

_APPROACH_BOUNDS = (1.2, 1.8, -0.4, 0.4)

_ACTION_LABELS = [
    "ParkArms (pre)",
    "Navigate -> milk",
    "PickUp milk",
]


@dataclass
class ActionEntry:
    """Bundles a concrete action instance with its Parameterization and ProbabilisticCircuit."""
    instance: Any
    parameterization: Parameterization
    distribution: ProbabilisticCircuit


def _initialize_world() -> tuple[World, PR2]:
    """
    Build a minimal world with a single scene-root body and the PR2 robot.

    The scene root (an empty body named 'scene') acts as the unique tree root so
    that the world always has exactly one node with in-degree 0.  The PR2 is then
    merged into this world at ROBOT_INIT_X/Y.

    :return: A tuple of (World, PR2).
    """
    scene_world = World(name="pick_only_scene")
    scene_root = Body(name=PrefixedName("scene"))
    with scene_world.modify_world():
        scene_world.add_kinematic_structure_entity(scene_root)

    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    pr2_world = URDFParser.from_file(pr2_urdf).parse()
    robot = PR2.from_world(pr2_world)

    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        ROBOT_INIT_X, ROBOT_INIT_Y, 0.0, 0, 0, 0
    )
    with scene_world.modify_world():
        scene_world.merge_world_at_pose(pr2_world, robot_pose)

    return scene_world, robot


def _add_localization_frames(world: World, robot: PR2) -> None:
    """
    Insert map and odom_combined frames and wire the robot base via OmniDrive.

    Hierarchy after this call:
      scene (root, in-degree 0)
        └── map            (FixedConnection)
              └── odom_combined  (Connection6DoF)
                    └── <pr2 base>  (OmniDrive)

    The Connection6DoF that merge_world_at_pose created between scene and the
    PR2 base is replaced by the OmniDrive.

    :param world: The world instance.
    :param robot: The PR2 robot instance.
    """
    with world.modify_world():
        map_body  = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))

        world.add_body(map_body)
        world.add_body(odom_body)

        world.add_connection(FixedConnection(parent=world.root, child=map_body))

        world.add_connection(
            Connection6DoF.create_with_dofs(world, map_body, odom_body)
        )


        old_connection = robot.root.parent_connection
        if old_connection is not None:
            world.remove_connection(old_connection)

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


def _add_milk_object(world: World, stl_path: str) -> tuple[Body, HomogeneousTransformationMatrix]:
    """
    Add the milk object at MILK_X/Y/Z into the world.

    :param world: The world instance.
    :param stl_path: Absolute path to the milk STL file.
    :return: (milk Body, initial pose).
    """
    mesh = FileMesh.from_file(stl_path)
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


def _respawn_milk(world: World, milk_body: Body) -> None:
    """Reset the milk object to its initial spawn location after each iteration."""
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



def _create_pose_stamped(x: float, y: float, z: float, frame_id: Any) -> PoseStamped:
    return PoseStamped(
        pose=PyCramPose(
            position=PyCramVector3(x=x, y=y, z=z),
            orientation=PyCramQuaternion(x=0, y=0, z=0, w=1),
        ),
        header=Header(frame_id=frame_id, stamp=datetime.datetime.now(), sequence=0),
    )


def _create_database_session(database_uri: str) -> Session:
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
    """
    Persist a SequentialPlan to the database.

    Header.stamp is guaranteed to be non-None by the Header.__getattr__ patch
    applied at module load time, so no pre-processing is needed here.

    :param session: The SQLAlchemy session.
    :param plan: The plan to persist.
    """
    plan_dao = to_dao(plan)
    session.add(plan_dao)
    session.commit()


def _build_fixed_plan(
    context: Context,
    world: World,
    robot: PR2,
    milk_body: Body,
) -> SequentialPlan:
    """
    Build a deterministic ParkArms -> Navigate -> PickUp plan.

    :param context: The planning context.
    :param world: The world instance.
    :param robot: The PR2 robot instance.
    :param milk_body: The milk body.
    :return: A SequentialPlan.
    """
    approach_pose = _create_pose_stamped(APPROACH_X, APPROACH_Y, 0.0, world.root)
    actions = [
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(target_location=approach_pose, keep_joint_states=False),
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
    ]
    return SequentialPlan(context, *actions)


def _navigable_pose_description(robot: PR2) -> Any:
    """
    Build a probable PoseStamped with free x/y for probabilistic sampling.

    :param robot: The PR2 robot instance.
    :return: A probable PoseStamped description.
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
    Construct the three action descriptions for the pick-only plan.

    :param world: The world instance.
    :param robot: The PR2 robot instance.
    :param milk_variable: Variable representing the milk object.
    :return: List of action descriptions.
    """
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
                    ApproachDirection, [ApproachDirection.FRONT]
                ),
                vertical_alignment=variable(
                    VerticalAlignment, [VerticalAlignment.NoAlignment]
                ),
                rotate_gripper=variable(bool, [False]),
                manipulation_offset=0.06,
                manipulator=variable(Manipulator, manipulators),
            ),
        ),
    ]


def _truncate_navigate_distribution(
    distribution: ProbabilisticCircuit,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> ProbabilisticCircuit:
    """
    Restrict a NavigateAction distribution to the given x/y bounding box.

    :param distribution: The fully-factorised circuit for a NavigateAction.
    :param x_min: Minimum x bound.
    :param x_max: Maximum x bound.
    :param y_min: Minimum y bound.
    :param y_max: Maximum y bound.
    :return: Truncated distribution or original if truncation fails.
    """
    from random_events.interval import closed
    from random_events.product_algebra import SimpleEvent
    from random_events.variable import Continuous

    all_names = [v.name for v in distribution.variables]
    print(f"    [truncate] distribution variables: {all_names}")

    x_var = None
    y_var = None
    for v in distribution.variables:
        if isinstance(v, Continuous):
            if v.name.endswith(".position.x") or v.name.endswith(".x"):
                x_var = v
            elif v.name.endswith(".position.y") or v.name.endswith(".y"):
                y_var = v

    if x_var is None or y_var is None:
        print(
            f"    [truncate] WARNING: could not find position x/y variables "
            f"(vars={[v.name for v in distribution.variables]}). Skipping truncation."
        )
        return distribution

    event = SimpleEvent(
        {x_var: closed(x_min, x_max), y_var: closed(y_min, y_max)}
    ).as_composite_set()
    full_event = event.fill_missing_variables_pure(distribution.variables)
    truncated, log_prob = distribution.log_truncated_in_place(full_event)

    if truncated is None:
        print(
            f"    [truncate] WARNING: zero-probability region "
            f"x=[{x_min},{x_max}] y=[{y_min},{y_max}]. Keeping untruncated."
        )
        return distribution

    print(f"    [truncate] x=[{x_min},{x_max}] y=[{y_min},{y_max}]  log_p={log_prob:.3f}")
    return truncated


def _build_action_entry(description: Any, approach_bounds: tuple = None) -> ActionEntry:
    """
    Translate a description into a concrete ActionEntry with distribution.

    :param description: The probable_variable match description.
    :param approach_bounds: Optional (x_min, x_max, y_min, y_max).
    :return: An ActionEntry.
    """
    instance = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution = parameterization.create_fully_factorized_distribution()
    if approach_bounds is not None:
        x_min, x_max, y_min, y_max = approach_bounds
        distribution = _truncate_navigate_distribution(
            distribution, x_min, x_max, y_min, y_max
        )
    return ActionEntry(instance, parameterization, distribution)


def _apply_sample(entry: ActionEntry) -> None:
    """Sample from the entry's distribution and write parameters into its instance."""
    raw_sample = entry.distribution.sample(1)[0]
    named_sample = entry.parameterization.create_assignment_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )
    entry.parameterization.parameterize_object_with_sample(entry.instance, named_sample)


def _build_sampled_plan(
    context: Context,
    entries: list[ActionEntry],
) -> SequentialPlan:
    """
    Sample fresh parameters for every entry and assemble the plan.

    Entry layout (matches _build_action_descriptions):
      0  ParkArmsAction
      1  NavigateAction  (truncated to _APPROACH_BOUNDS)
      2  PickUpAction

    :param context: The planning context.
    :param entries: List of ActionEntry objects.
    :return: A SequentialPlan.
    """
    for label, entry in zip(_ACTION_LABELS, entries):
        _apply_sample(entry)
        print(f"    Sampled  {label}")

    nav_pos = entries[1].instance.target_location.pose.position
    print(
        f"    Sampled approach goal: ({float(nav_pos.x):.3f}, {float(nav_pos.y):.3f})"
    )

    actions = [entry.instance for entry in entries]
    return SequentialPlan(context, *actions)


def pick_only_demo() -> None:
    """
    Minimal pick-only probabilistic reasoning demo.

    Runs NUMBER_OF_ITERATIONS plans (ParkArms -> Navigate -> PickUp) in an
    empty world containing only a scene-root body, the PR2 robot, and a milk
    object.  Iteration 1 uses fixed, known-good parameters; subsequent
    iterations sample via the Parameterizer / ProbabilisticCircuit pipeline.
    Successful plans are persisted to DATABASE_URI.

    World structure
    ---------------
    scene (root)
      ├── map  (FixedConnection)
      │     └── odom_combined  (Connection6DoF)
      │           └── <pr2 base>  (OmniDrive)
      │                 └── ... rest of PR2 kinematic tree
      └── milk_1  (Connection6DoF)
    """
    print("=" * 64)
    print("  pick_only_demo  -  empty world, ParkArms + Navigate + PickUp")
    print("=" * 64)

    print("\nBuilding world ...")
    world, robot = _initialize_world()
    _add_localization_frames(world, robot)

    from pathlib import Path
    _resource_path = Path(__file__).resolve().parents[2] / "resources"
    milk_stl = str(_resource_path / "objects" / "milk.stl")
    milk_body, _ = _add_milk_object(world, milk_stl)

    print(f"  World root:      '{world.root.name}'")
    print(f"  Milk at          x={MILK_X:.3f},  y={MILK_Y:.3f},  z={MILK_Z:.3f}")
    print(f"  Approach target  x={APPROACH_X:.3f}, y={APPROACH_Y:.3f}")

    session = _create_database_session(DATABASE_URI)
    print(f"  Database:        {DATABASE_URI}")

    rclpy.init()
    ros_node = rclpy.create_node("pick_only_demo_node")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    try:
        _tf_pub  = TFPublisher(_world=world, node=ros_node)
        _viz_pub = VizMarkerPublisher(_world=world, node=ros_node)

        context = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        print("\nBuilding per-action probabilistic distributions ...")
        descriptions = _build_action_descriptions(world, robot, milk_variable)

        entries: list[ActionEntry] = [
            _build_action_entry(descriptions[0]),                    # ParkArms
            _build_action_entry(descriptions[1], _APPROACH_BOUNDS),  # Navigate (truncated)
            _build_action_entry(descriptions[2]),                    # PickUp
        ]

        for label, entry in zip(_ACTION_LABELS, entries):
            n_vars = len(entry.parameterization.variables)
            n_dist = len(entry.distribution.variables)
            print(
                f"  {label:<25}  {n_vars:>2} param vars  /  {n_dist:>2} distribution vars"
            )

        successful_count = 0

        for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
            sep = "=" * 64
            print(f"\n{sep}")

            if iteration == 1:
                print(
                    f"  Iteration {iteration:>3} / {NUMBER_OF_ITERATIONS}  [FIXED parameters]"
                )
                print(sep)
                plan = _build_fixed_plan(context, world, robot, milk_body)
            else:
                print(
                    f"  Iteration {iteration:>3} / {NUMBER_OF_ITERATIONS}  [SAMPLED parameters]"
                )
                print(sep)
                plan = _build_sampled_plan(context, entries)

            with simulated_robot:
                try:
                    plan.perform()
                    _persist_plan(session, plan)
                    successful_count += 1
                    print(
                        f"\n  v  Iteration {iteration} succeeded -- plan persisted to database."
                    )
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(
                        f"\n  x  Iteration {iteration} failed -- plan not stored.  "
                        f"({type(exc).__name__}: {exc})"
                    )
                finally:
                    _respawn_milk(world, milk_body)

        print(f"\n{'=' * 64}")
        print(
            f"Done.  {successful_count} / {NUMBER_OF_ITERATIONS} plans stored "
            f"in '{DATABASE_URI}'."
        )

    finally:
        session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    pick_only_demo()