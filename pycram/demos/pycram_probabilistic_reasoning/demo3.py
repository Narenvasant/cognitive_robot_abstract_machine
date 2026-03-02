"""
demo3.py
========
Demonstrates probabilistic parameterization of a full SequentialPlan in an
apartment world using the krrood parameterizer.

Workflow
--------
1.  Build the world, robot and milk object.
2.  Define every action as a ``probable_variable`` description.
3.  Translate each description into a concrete instance + Parameterization.
4.  Build one fully-factorized distribution *per action* (a single joint
    distribution is not possible because different actions share field names,
    which violates the uniqueness requirement of ``get_variable_by_name``).
5.  Repeat for ``NUM_ITERATIONS``:
      a.  Sample all distributions and resolve the action instances in-place.
      b.  Execute the plan inside ``simulated_robot``.
      c.  On success, persist the executed SequentialPlan to the database via
          ``to_dao`` so that the full plan graph — actions, motions, parameters
          and world state — is stored as a relational record.

Database setup
--------------
The demo reads the connection string from the environment variable
``SEMANTIC_DIGITAL_TWIN_DATABASE_URI``.  For a local PostgreSQL instance:

    export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=\\
        postgresql://semantic_digital_twin:a_very_strong_password_here@localhost:5432/semantic_digital_twin

For quick local testing without PostgreSQL an in-memory SQLite database is used
as a fallback (data is lost when the process exits).

See also:
    https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/
    examples/persistence_of_annotated_worlds.html
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rclpy
from sqlalchemy import select
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
from pycram.orm.ormatic_interface import *  # noqa: F401,F403  — registers DAO mappings
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_ITERATIONS: int = 20

# Fall back to an in-memory SQLite DB if no environment variable is set.
# For permanent storage set SEMANTIC_DIGITAL_TWIN_DATABASE_URI, e.g.:
#   postgresql://semantic_digital_twin:password@localhost:5432/semantic_digital_twin
_DATABASE_URI: str = os.environ.get(
    "SEMANTIC_DIGITAL_TWIN_DATABASE_URI",
    "sqlite:///:memory:",
)

# ---------------------------------------------------------------------------
# Resource paths (resolved relative to this file)
# ---------------------------------------------------------------------------

_RESOURCE_PATH = Path(__file__).resolve().parents[2] / "resources"

APARTMENT_URDF: Path = _RESOURCE_PATH / "worlds" / "apartment.urdf"
ROBOT_URDF:     Path = _RESOURCE_PATH / "robots" / "pr2.urdf"
MILK_STL:       Path = _RESOURCE_PATH / "objects" / "milk.stl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionEntry:
    """Pairs a concrete action instance with its Parameterization and distribution."""

    instance: Any
    parameterization: Parameterization
    distribution: ProbabilisticCircuit


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _create_session(database_uri: str) -> Session:
    """
    Create an SQLAlchemy session and ensure all ORM tables exist.

    Uses the krrood ``create_engine`` wrapper which registers the project's
    custom JSON (de)serializer so that complex types survive the round-trip.
    """
    engine = create_engine(database_uri)
    Base.metadata.create_all(bind=engine)
    return Session(engine)


def _persist_plan(session: Session, plan: SequentialPlan) -> None:
    """
    Convert a successfully executed SequentialPlan to its DAO representation
    and commit it to the database.

    The ``to_dao`` call traverses the entire plan graph — nodes, edges, action
    designators, motion designators and execution data — and produces the
    corresponding SQLAlchemy model tree.  ``session.add`` registers the root
    DAO; SQLAlchemy's cascade rules persist all related objects automatically
    on ``commit``.

    :param session: The active SQLAlchemy session.
    :param plan: The plan that completed without error.
    """
    plan_dao = to_dao(plan)
    session.add(plan_dao)
    session.commit()


# ---------------------------------------------------------------------------
# World construction helpers
# ---------------------------------------------------------------------------

def _build_world(apartment_urdf: Path, robot_urdf: Path) -> tuple:
    """Parse URDF files and return (world, robot)."""
    # Load apartment world
    apartment_world = URDFParser.from_file(str(apartment_urdf)).parse()

    # Load PR2 using package URL with proper path resolution (same as conftest.py)
    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    pr2_world = URDFParser.from_file(pr2_urdf).parse()

    # Apply robot semantic annotations
    robot = PR2.from_world(pr2_world)

    # Merge worlds at the specified pose
    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 1.0, 0, 0, 0)
    with apartment_world.modify_world():
        apartment_world.merge_world_at_pose(pr2_world, robot_pose)

    return apartment_world, robot


def _add_localization_frame(world, robot) -> None:
    """
    Insert map → odom_combined → robot base chain so that the robot can be
    commanded via an OmniDrive connection.
    """
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
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
                    1.2, 1.5, 0.0, 0, 0, 0
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
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.5, 1.01, 0, 0, 0)

    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(parent=world.root, child=body, world=world)
        world.add_connection(connection)
        connection.origin = pose

    return body, pose


# ---------------------------------------------------------------------------
# Parameterization helpers
# ---------------------------------------------------------------------------

def _build_action_entry(description) -> ActionEntry:
    """
    Translate a ``probable_variable`` Match description into a concrete instance,
    derive its Parameterization and pre-build the fully-factorized distribution.

    The instance is kept alive across iterations so that
    ``parameterize_object_with_sample`` can update it in-place on every draw.
    """
    instance = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution = parameterization.create_fully_factorized_distribution()
    return ActionEntry(instance, parameterization, distribution)


def _apply_sample(entry: ActionEntry) -> None:
    """
    Draw one sample from an action's distribution and apply it to the
    instance in-place.

    :param entry: The ActionEntry whose instance should be updated.
    """
    raw_sample = entry.distribution.sample(1)[0]
    named_sample = entry.parameterization.create_assignment_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )
    entry.parameterization.parameterize_object_with_sample(entry.instance, named_sample)


# ---------------------------------------------------------------------------
# Action description factories
# ---------------------------------------------------------------------------

def _build_action_descriptions(world, robot, milk_variable) -> list:
    """
    Return a list of ``probable_variable`` descriptions for all six actions in
    the plan.  Descriptions are pure data — no side effects occur here.
    """
    def navigable_pose():
        return probable(PoseStamped)(
            pose=probable(PyCramPose)(
                position=probable(PyCramVector3)(x=..., y=..., z=0),
                orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
            ),
            header=probable(Header)(frame_id=variable_from([robot.root])),
        )

    return [
        probable_variable(ParkArmsAction)(
            arm=...,
        ),
        probable_variable(NavigateAction)(
            target_location=navigable_pose(),
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
                manipulator=variable(
                    Manipulator,
                    world.get_semantic_annotations_by_type(Manipulator),
                ),
            ),
        ),
        probable_variable(NavigateAction)(
            target_location=navigable_pose(),
            keep_joint_states=...,
        ),
        probable_variable(PlaceAction)(
            object_designator=milk_variable,
            target_location=navigable_pose(),
            arm=...,
        ),
        probable_variable(ParkArmsAction)(
            arm=...,
        ),
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sequential_plan_with_apartment() -> None:
    """
    Run a probabilistically parameterized SequentialPlan for NUM_ITERATIONS,
    and persist each successful plan execution directly to the database using
    the ormatic ORM (``to_dao`` → ``session.add`` → ``session.commit``).
    """
    world, robot = _build_world(APARTMENT_URDF, ROBOT_URDF)
    _add_localization_frame(world, robot)
    milk_body, _ = _add_milk(world, MILK_STL)

    session = _create_session(_DATABASE_URI)

    rclpy.init()
    ros_node = rclpy.create_node("sequential_plan")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    try:
        _tf_publisher  = TFPublisher(_world=world, node=ros_node)
        _viz_publisher = VizMarkerPublisher(_world=world, node=ros_node)

        context       = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        # Build descriptions, translate to instances and pre-compute distributions
        descriptions = _build_action_descriptions(world, robot, milk_variable)
        entries: list[ActionEntry] = [_build_action_entry(d) for d in descriptions]

        print(f"\nBuilt {len(entries)} per-action distributions:")
        for i, entry in enumerate(entries):
            print(
                f"  Action {i + 1}: {len(entry.parameterization.variables)} variables, "
                f"{len(entry.distribution.variables)} in distribution"
            )

        successful_count = 0

        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration} / {NUM_ITERATIONS}")
            print(f"{'=' * 60}")

            for entry in entries:
                _apply_sample(entry)

            sp = SequentialPlan(context, *[e.instance for e in entries])

            with simulated_robot:
                try:
                    sp.perform()

                    # Plan succeeded — persist the full plan graph to the database.
                    # to_dao traverses the entire SequentialPlan tree (nodes, edges,
                    # action designators, motion designators, execution data) and
                    # produces the corresponding SQLAlchemy DAO tree.  SQLAlchemy's
                    # cascade rules then persist every related object on commit.
                    _persist_plan(session, sp)

                    successful_count += 1
                    print(f"Iteration {iteration}: succeeded — plan persisted to database.")

                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"Iteration {iteration}: TimeoutError — plan not stored. ({exc})")

        print(f"\nDone. {successful_count}/{NUM_ITERATIONS} plans stored in '{_DATABASE_URI}'.")

    finally:
        session.close()
        time.sleep(0.1)
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    sequential_plan_with_apartment()