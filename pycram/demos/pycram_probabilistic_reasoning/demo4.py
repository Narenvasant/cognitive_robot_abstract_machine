"""
demo: Probabilistic pick-and-place with database persistence
========
Probabilistic pick-and-place demo: milk from kitchen counter → dining table
in the PR2 apartment world, using the krrood parameterizer.

Workflow
--------
1.  Parse the apartment URDF and build the semantic world + PR2 robot.
2.  Describe every action in the plan with ``probable_variable`` / ``probable``
    factories that mark free parameters with ``...`` (Ellipsis).
3.  Translate each description into a concrete instance via
    ``MatchToInstanceTranslator``, then derive a ``Parameterization`` from it
    via ``MatchParameterizer``.
4.  Build one **per-action** fully-factorized ``ProbabilisticCircuit`` (a single
    joint distribution is impossible because different actions share field names,
    which violates the uniqueness requirement of ``get_variable_by_name``).
5.  For ``NUM_ITERATIONS`` iterations:
      a.  Draw one sample per action and apply it in-place to the action instance.
      b.  Execute the ``SequentialPlan`` inside ``simulated_robot``.
      c.  On success, persist the whole plan graph to the database via ``to_dao``.

Action sequence
---------------
    ParkArms  →  Navigate-to-counter  →  PickUp(milk)
              →  Navigate-to-table    →  Place(milk)
              →  ParkArms

Database setup
--------------
Set the environment variable before running::

    export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=\\
        postgresql://semantic_digital_twin:naren@localhost:5432/demo_robot_plans

An in-memory SQLite database is used as a silent fallback so the demo can be
run without PostgreSQL for quick local testing (data is lost on exit).

See also:
    https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/
    examples/persistence_of_annotated_worlds.html
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from sqlalchemy.orm import Session

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import (
    Header,
    PoseStamped,
    PyCramPose,
    PyCramQuaternion,
    PyCramVector3,
)
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import *
from pycram.robot_plans import NavigateAction, ParkArmsAction, PickUpAction, PlaceAction

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
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


NUM_ITERATIONS: int = 20

_DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///:memory:",
)


_COUNTER_APPROACH_X: float = 1.0
_COUNTER_APPROACH_Y: float = 3.28
_TABLE_APPROACH_X:   float = 4.2
_TABLE_APPROACH_Y:   float = 4.0

#shelf
# _MILK_X: float = 0.312
# _MILK_Y: float = 3.281
# _MILK_Z: float = 0.928

#counter
_MILK_X: float = 2.4
_MILK_Y: float = 2.5
_MILK_Z: float = 1.01

#table
_PLACE_X: float = 5.0
_PLACE_Y: float = 4.0
_PLACE_Z: float = 0.724


_RESOURCE_PATH = Path(__file__).resolve().parents[2] / "resources"

APARTMENT_URDF: Path = _RESOURCE_PATH / "worlds"  / "apartment.urdf"
MILK_STL:       Path = _RESOURCE_PATH / "objects" / "milk.stl"



@dataclass
class ActionEntry:
    """
    Groups a concrete action instance with its ``Parameterization`` and the
    pre-built ``ProbabilisticCircuit`` so that all three travel together.
    """

    instance: Any
    """Concrete action object whose fields will be updated in-place each iteration."""

    parameterization: Parameterization
    """Mapping from ``ObjectAccessVariable`` → field accessor used by the sampler."""

    distribution: ProbabilisticCircuit
    """Fully-factorized circuit from which one sample is drawn per iteration."""


def _create_session(database_uri: str) -> Session:
    """
    Create an SQLAlchemy ``Session`` and ensure all ORM tables exist.

    Uses the krrood ``create_engine`` wrapper, which registers the project's
    custom JSON (de)serializer so that complex types survive the round-trip.

    :param database_uri: SQLAlchemy-compatible connection string.
    :return: A ready-to-use ``Session``.
    """
    engine = create_engine(database_uri)
    Base.metadata.create_all(bind=engine)
    return Session(engine)


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    """
    Persist a successfully executed ``SequentialPlan`` to the database.

    ``to_dao`` traverses the entire plan graph — nodes, edges, action
    designators, motion designators, execution data, and world-state snapshots —
    and produces the corresponding SQLAlchemy DAO tree.  ``session.add``
    registers the root DAO; SQLAlchemy cascade rules then persist every related
    object on ``commit``.

    :param session: The active SQLAlchemy session.
    :param plan: The plan that completed without error.
    """
    plan_dao = to_dao(plan)
    session.add(plan_dao)
    session.commit()


def _build_world(apartment_urdf: Path) -> tuple[Any, Any]:
    """
    Parse the apartment and PR2 URDF files, merge the robot into the world at
    its home position, and return ``(world, robot)``.

    :param apartment_urdf: Path to ``apartment.urdf``.
    :param robot_urdf: Path to ``pr2.urdf``.
    :return: Tuple of ``(world, PR2)``.
    """
    world = URDFParser.from_file(str(apartment_urdf)).parse()

    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    robot_world = URDFParser.from_file(pr2_urdf).parse()

    robot = PR2.from_world(robot_world)

    robot_home = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 0.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(robot_world, robot_home)

    return world, robot


def _add_localization_frame(world: Any, robot: Any) -> None:
    """
    Insert the ``map → odom_combined → robot-base`` chain required for
    ``OmniDrive`` navigation commands.

    This mirrors the ROS TF tree that a real PR2 would publish so that
    ``NavigateAction`` can operate in the map frame.

    :param world: The semantic world.
    :param robot: The PR2 robot view.
    """
    with world.modify_world():
        map_body  = Body(name=PrefixedName("map"))
        odom_body = Body(name=PrefixedName("odom_combined"))
        world.add_body(map_body)
        world.add_body(odom_body)

        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, odom_body))

        # Re-parent the robot base under odom_combined.
        old_connection = robot.root.parent_connection
        if old_connection is not None:
            world.remove_connection(old_connection)

        world.add_connection(
            OmniDrive.create_with_dofs(
                parent=odom_body,
                child=robot.root,
                world=world,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.4, 1.5, 0.0, 0, 0, 0  # robot home: open area between counter and table
                ),
            )
        )


def _add_milk(world: Any, stl_path: Path) -> tuple[Body, HomogeneousTransformationMatrix]:
    """
    Add a milk object to the semantic world on top of the kitchen counter.

    :param world: The semantic world.
    :param stl_path: Path to ``milk.stl``.
    :return: Tuple of ``(milk_body, initial_pose)``.
    """
    mesh = FileMesh.from_file(str(stl_path))
    milk_body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        _MILK_X, _MILK_Y, _MILK_Z, 0, 0, 0
    )

    with world.modify_world():
        world.add_body(milk_body)
        connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=milk_body, world=world
        )
        world.add_connection(connection)
        connection.origin = pose

    return milk_body, pose


def _build_action_entry(description: Any) -> ActionEntry:
    """
    Translate a ``probable_variable`` Match description into a concrete
    action instance, derive its ``Parameterization``, and pre-build the
    fully-factorized ``ProbabilisticCircuit``.

    The instance is reused across all iterations; ``_apply_sample`` updates it
    in-place each time rather than re-constructing it.

    :param description: A ``probable_variable(...)`` match expression.
    :return: An ``ActionEntry`` ready for repeated sampling.
    """
    instance       = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution   = parameterization.create_fully_factorized_distribution()
    return ActionEntry(instance, parameterization, distribution)


def _apply_sample(entry: ActionEntry) -> None:
    """
    Draw one sample from an action's circuit and apply it to the instance
    in-place so that the ``SequentialPlan`` constructed in the outer loop
    receives freshly sampled parameters.

    :param entry: The ``ActionEntry`` whose instance should be updated.
    """
    raw_sample   = entry.distribution.sample(1)[0]
    named_sample = entry.parameterization.create_assignment_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )
    entry.parameterization.parameterize_object_with_sample(entry.instance, named_sample)



def _navigable_pose(robot: Any, mean_x: float, mean_y: float) -> Any:
    """
    Build a ``probable`` ``PoseStamped`` description for a navigation target.

    The x / y position components are treated as free (``...``) so that the
    probabilistic model can perturb them; z is fixed to 0 (floor plane).
    The quaternion represents a yaw-only rotation that is also kept free.

    The ``frame_id`` is bound to the robot's root body so that the frame
    reference is always consistent with the live world tree.

    .. note::
        Mean values for x / y are **not** encoded here; the fully-factorized
        distribution uses N(0, 1) priors.  If you want to centre the
        distribution on a specific pose, condition the circuit on those values
        before sampling (or shift the sample post-hoc).  For this demo we rely
        on the robot's inverse-kinematics solver tolerating nearby poses.

    :param robot: PR2 robot view (provides the root frame).
    :param mean_x: Nominal x coordinate — documented but not yet wired to the
                   distribution mean (left as future work for conditioning).
    :param mean_y: Nominal y coordinate — same caveat as ``mean_x``.
    :return: A ``probable(PoseStamped)(...)`` expression.
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=0),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot.root])),
    )


def _place_pose(robot: Any) -> Any:
    """
    Build a ``probable`` ``PoseStamped`` for the place target on the dining
    table.  z is fixed to the table surface height; x / y are free.

    :param robot: PR2 robot view.
    :return: A ``probable(PoseStamped)(...)`` expression.
    """
    return probable(PoseStamped)(
        pose=probable(PyCramPose)(
            position=probable(PyCramVector3)(x=..., y=..., z=_PLACE_Z),
            orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
        ),
        header=probable(Header)(frame_id=variable_from([robot.root])),
    )



def _build_action_descriptions(
    world: Any,
    robot: Any,
    milk_variable: Any,
) -> list[Any]:
    """
    Construct the six ``probable_variable`` descriptions for the pick-and-place
    plan.  These are pure data expressions — no side effects occur here.

    Plan sequence
    ~~~~~~~~~~~~~
    1. **ParkArms** — tuck arms before navigating.
    2. **Navigate** → kitchen counter approach pose.
    3. **PickUp**   — grasp milk with a fully-sampled ``GraspDescription``.
    4. **Navigate** → dining table approach pose.
    5. **Place**    — set milk down on the table surface.
    6. **ParkArms** — tuck arms after placing.

    :param world: The semantic world (used to look up ``Manipulator`` annotations).
    :param robot: The PR2 robot view (provides root frame and manipulators).
    :param milk_variable: An EQL ``variable_from`` expression bound to the milk body.
    :return: Ordered list of six ``probable_variable`` match expressions.
    """
    manipulators = world.get_semantic_annotations_by_type(Manipulator)

    return [
        probable_variable(ParkArmsAction)(
            arm=...,
        ),

        probable_variable(NavigateAction)(
            target_location=_navigable_pose(robot, _COUNTER_APPROACH_X, _COUNTER_APPROACH_Y),
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
            target_location=_navigable_pose(robot, _TABLE_APPROACH_X, _TABLE_APPROACH_Y),
            keep_joint_states=...,
        ),

        probable_variable(PlaceAction)(
            object_designator=milk_variable,
            target_location=_place_pose(robot),
            arm=...,
        ),

        # 6 ── Park arms again
        probable_variable(ParkArmsAction)(
            arm=...,
        ),
    ]


def _start_ros_publishers(world: Any, ros_node: Any) -> tuple[Any, Any]:
    """
    Create and return the TF and RViz marker publishers for visualisation.

    Both publishers run their own internal timers / callbacks on the ROS node
    and do not require explicit ``publish()`` calls from the demo loop.

    :param world: The semantic world.
    :param ros_node: The active ``rclpy`` node.
    :return: Tuple of ``(TFPublisher, VizMarkerPublisher)``.
    """
    tf_publisher  = TFPublisher(_world=world, node=ros_node)
    viz_publisher = VizMarkerPublisher(_world=world, node=ros_node)
    return tf_publisher, viz_publisher


def sequential_plan_with_apartment() -> None:
    """
    Run a probabilistically parameterised pick-and-place ``SequentialPlan``
    for ``NUM_ITERATIONS`` iterations and persist each successful execution
    to the configured database.

    The high-level loop is::

        for _ in range(NUM_ITERATIONS):
            sample all per-action distributions
            build SequentialPlan from sampled instances
            with simulated_robot:
                try:
                    plan.perform()
                    persist to DB
                except PlanFailure:
                    log and continue

    Persistence uses the ormatic ``to_dao`` → ``session.add`` → ``session.commit``
    pattern, which stores the full plan graph — actions, motions, parameters and
    world-state snapshots — as a relational record.
    """
    print("Building world …")
    world, robot = _build_world(APARTMENT_URDF)
    _add_localization_frame(world, robot)
    milk_body, milk_pose = _add_milk(world, MILK_STL)

    print(f"  Milk spawned at  x={_MILK_X:.3f},  y={_MILK_Y:.3f},  z={_MILK_Z:.3f}  (Side-A countertop surface)")
    print(f"  Place target at  x={_PLACE_X:.3f}, y={_PLACE_Y:.3f}, z={_PLACE_Z:.3f}  (dining table surface)")

    session = _create_session(_DATABASE_URI)
    print(f"Database: {_DATABASE_URI}")

    rclpy.init()
    ros_node    = rclpy.create_node("sequential_plan_demo")
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True, name="rclpy-spin"
    )
    spin_thread.start()

    try:
        _tf_pub, _viz_pub = _start_ros_publishers(world, ros_node)

        # ── Context + EQL milk variable ───────────────────────────────────────
        context       = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        print("\nBuilding per-action probabilistic distributions …")
        descriptions = _build_action_descriptions(world, robot, milk_variable)
        entries: list[ActionEntry] = [_build_action_entry(d) for d in descriptions]

        _action_labels = [
            "ParkArms (pre)",
            "Navigate → counter",
            "PickUp milk",
            "Navigate → table",
            "Place milk",
            "ParkArms (post)",
        ]
        for label, entry in zip(_action_labels, entries):
            n_vars = len(entry.parameterization.variables)
            n_dist = len(entry.distribution.variables)
            print(f"  {label:<25}  {n_vars:>2} param vars  /  {n_dist:>2} distribution vars")

        successful_count = 0

        for iteration in range(1, NUM_ITERATIONS + 1):
            separator = "=" * 62
            print(f"\n{separator}")
            print(f"  Iteration {iteration:>3} / {NUM_ITERATIONS}")
            print(separator)

            for label, entry in zip(_action_labels, entries):
                _apply_sample(entry)
                print(f"  Sampled  {label}")

            sp = SequentialPlan(context, *[e.instance for e in entries])

            with simulated_robot:
                try:
                    sp.perform()

                    _persist_plan(session, sp)

                    successful_count += 1
                    print(
                        f"\n  ✓  Iteration {iteration} succeeded — "
                        f"plan persisted to database."
                    )

                except Exception as exc:
                    print(
                        f"\n  ✗  Iteration {iteration} failed — "
                        f"plan not stored.  ({type(exc).__name__}: {exc})"
                    )

        print(f"\n{'=' * 62}")
        print(
            f"Done.  {successful_count} / {NUM_ITERATIONS} plans stored "
            f"in '{_DATABASE_URI}'."
        )

    finally:
        session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    sequential_plan_with_apartment()

