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
      a.  Sample all distributions.
      b.  Scale position samples (x, y) to be near the relevant world position
          so the robot has a realistic chance of reaching the target.
      c.  Apply the scaled sample to the action instance in-place.
      d.  Execute the plan inside ``simulated_robot``.
      e.  On success, persist the executed SequentialPlan to the database via
          ``to_dao`` so that the full plan graph — actions, motions, parameters
          and world state — is stored as a relational record.

Position scaling
----------------
The fully-factorized distribution samples x and y from N(0, 1) by default.
This produces positions that are far from the milk and the robot cannot reach
them.  After sampling, x and y are linearly scaled into a small window around
a given anchor point (e.g. the milk position or a place position) using:

    scaled = anchor + sample * POSITION_NOISE_STD

where POSITION_NOISE_STD controls how much variation is allowed around the
anchor.  This keeps the parameterization probabilistic while constraining the
robot to positions it can actually execute.

Database setup
--------------
The demo reads the connection string from the environment variable
``SEMANTIC_DIGITAL_TWIN_DATABASE_URI``.  For a local PostgreSQL instance:

    export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=\\
        postgresql://semantic_digital_twin:password@localhost:5432/demo_robot_plans

For quick local testing without PostgreSQL an in-memory SQLite database is used
as a fallback (data is lost when the process exits).

See also:
    https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/
    examples/persistence_of_annotated_worlds.html
"""

from __future__ import annotations

import datetime
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
from krrood.probabilistic_knowledge.object_access_variable import ObjectAccessVariable
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

# Standard deviation (in metres) of position noise added around the anchor.
# Smaller = robot stays closer to the anchor, higher success rate.
# Larger = more diverse dataset but more failures.
POSITION_NOISE_STD: float = 0.15

# Fall back to an in-memory SQLite DB if no environment variable is set.
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
# Known world positions used as scaling anchors
# (derived from apartment URDF — see scripts/inspect_apartment_poses.py)
#
#   island_countertop : x=2.747, y=2.664, z=0.922  <- milk sits here
#   table_area_main   : x=5.000, y=4.000, z=0.000  <- place target
#   robot start pose  : x=1.400, y=1.500            <- inside apartment
# ---------------------------------------------------------------------------

# Milk is placed on the island countertop at (2.4, 2.5).
# The robot should stand ~0.8 m in front of the counter (in the -x direction).
MILK_X: float = 2.4
MILK_Y: float = 2.5
NAVIGATE_TO_MILK_ANCHOR: tuple[float, float] = (MILK_X - 0.8, MILK_Y)

# Place target is the table_area_main at (5.0, 4.0).
# The robot stands at NAVIGATE_TO_PLACE_ANCHOR and places the milk at arm's
# reach in front of it (~0.5 m forward). PLACE_ANCHOR must be close to
# NAVIGATE_TO_PLACE_ANCHOR — if the place target is further than arm reach
# (~0.7 m) the ReachAction times out.
NAVIGATE_TO_PLACE_ANCHOR: tuple[float, float] = (4.2, 4.0)
PLACE_ANCHOR: tuple[float, float] = (NAVIGATE_TO_PLACE_ANCHOR[0] + 0.5, NAVIGATE_TO_PLACE_ANCHOR[1])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionEntry:
    """
    Pairs a concrete action instance with its Parameterization, distribution
    and an optional position anchor used to scale sampled x/y values.

    anchor: (x, y) world coordinates around which position samples are scaled.
            None means no scaling is applied (e.g. ParkArmsAction has no pose).
    """

    instance: Any
    parameterization: Parameterization
    distribution: ProbabilisticCircuit
    anchor: tuple[float, float] | None = None


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

    ``to_dao`` traverses the entire plan graph — nodes, edges, action
    designators, motion designators and execution data — and produces the
    corresponding SQLAlchemy model tree.  ``session.add`` registers the root
    DAO; SQLAlchemy's cascade rules persist all related objects on ``commit``.

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
    world = URDFParser.from_file(str(apartment_urdf)).parse()
    robot_world = URDFParser.from_file(str(robot_urdf)).parse()

    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 1.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(robot_world, robot_pose)

    return world, PR2.from_world(world)


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


def _add_milk(world, stl_path: Path) -> tuple[Body, HomogeneousTransformationMatrix, Connection6DoF]:
    """Add a milk object to the world and return (body, initial_pose, connection)."""
    mesh = FileMesh.from_file(str(stl_path))
    body = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([mesh]),
        collision=ShapeCollection([mesh]),
    )
    pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        MILK_X, MILK_Y, 1.01, 0, 0, 0
    )

    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(
            parent=world.root, child=body, world=world
        )
        world.add_connection(connection)
        connection.origin = pose

    return body, pose, connection


def _reset_milk(world, connection: Connection6DoF, initial_pose: HomogeneousTransformationMatrix) -> None:
    """
    Teleport the milk back to its original pose on the countertop.

    Must be called **inside** the ``simulated_robot`` context, before
    ``sp.perform()``.  The ``connection._world`` reference is only valid while
    the simulation is active — calling this after the context exits causes
    ``AttributeError: NoneType has no attribute state``.

    ``world.modify_world()`` is required because the ``connection.origin``
    setter accesses ``world.state`` which needs an active modification block.
    """
    with world.modify_world():
        connection.origin = initial_pose


# ---------------------------------------------------------------------------
# Parameterization helpers
# ---------------------------------------------------------------------------

def _build_action_entry(
    description, anchor: tuple[float, float] | None = None
) -> ActionEntry:
    """
    Translate a ``probable_variable`` Match description into a concrete instance,
    derive its Parameterization and pre-build the fully-factorized distribution.

    :param description: The ``probable_variable`` Match description.
    :param anchor: Optional (x, y) anchor for position scaling.  Pass the
                   world-frame position the robot should stand near so that
                   sampled x/y values are scaled around it.
    """
    instance = MatchToInstanceTranslator(description).translate()
    parameterization = MatchParameterizer(instance).parameterize()
    distribution = parameterization.create_fully_factorized_distribution()
    return ActionEntry(instance, parameterization, distribution, anchor)


# The exact field names on action instances that hold a PoseStamped.
# Only these fields are checked to avoid traversing the world object graph
# (Body -> Connection -> World -> ...) which causes infinite recursion.
_POSE_STAMPED_FIELDS = ("target_location",)


def _patch_header(instance: Any, header: Header) -> None:
    """
    Set ``header`` on every ``PoseStamped`` held by a known action field.

    Rather than recursing generically through the object graph (which causes
    infinite recursion via Body <-> Connection <-> World cycles), we only
    inspect the specific fields known to hold a ``PoseStamped``.

    :param instance: The action instance to patch (NavigateAction, PlaceAction).
    :param header: The fixed world-frame ``Header`` to assign.
    """
    if instance is None:
        return
    for field_name in _POSE_STAMPED_FIELDS:
        child = getattr(instance, field_name, None)
        if isinstance(child, PoseStamped):
            child.header = header


def _scale_position_sample(
    named_sample: dict[ObjectAccessVariable, Any],
    anchor: tuple[float, float],
) -> dict[ObjectAccessVariable, Any]:
    """
    Rescale the x and y position values in a named sample around an anchor.

    The distribution produces x and y from N(0, 1).  This function maps those
    unit-normal samples into a small window around ``anchor``:

        scaled_x = anchor_x + raw_x * POSITION_NOISE_STD
        scaled_y = anchor_y + raw_y * POSITION_NOISE_STD

    All other variables (arm, grasp direction, etc.) are left unchanged.

    :param named_sample: Dict mapping ObjectAccessVariable -> sampled value.
    :param anchor: (x, y) world-frame anchor coordinates.
    :return: The same dict with x and y values replaced by scaled values.
    """
    anchor_x, anchor_y = anchor

    # Actual suffixes as revealed by inspecting the named_sample keys:
    #   Variable(NavigateAction, ...).target_location.pose.position.x
    #   Variable(NavigateAction, ...).target_location.pose.position.y
    #   Variable(NavigateAction, ...).target_location.position.x
    #   Variable(NavigateAction, ...).target_location.position.y
    x_suffixes = (".pose.position.x", ".position.x")
    y_suffixes = (".pose.position.y", ".position.y")

    scaled = {}
    for var, value in named_sample.items():
        var_name = str(var.variable.name)
        if any(var_name.endswith(s) for s in x_suffixes):
            scaled[var] = anchor_x + float(value) * POSITION_NOISE_STD
        elif any(var_name.endswith(s) for s in y_suffixes):
            scaled[var] = anchor_y + float(value) * POSITION_NOISE_STD
        else:
            scaled[var] = value
    return scaled


def _apply_sample(entry: ActionEntry, verbose: bool = False) -> None:
    """
    Draw one sample from an action's distribution, optionally scale the
    position values around the entry's anchor, then apply the sample to
    the action instance in-place.

    Sampling always comes from the pre-built fully-factorized distribution
    (built once from the parameterizer, reused every iteration).  Scaling is
    applied after sampling so the distribution itself is never modified —
    only the concrete values written to the instance change.

    :param entry: The ActionEntry whose instance should be updated.
    :param verbose: If True, print each sampled variable name and value.
    """
    # Step 1: draw a raw sample from the distribution (N(0,1) for continuous vars)
    raw_sample = entry.distribution.sample(1)[0]

    # Step 2: map raw sample vector -> {ObjectAccessVariable: value} dict
    named_sample = entry.parameterization.create_assignment_from_variables_and_sample(
        entry.distribution.variables, raw_sample
    )

    # Step 3: scale position variables around the anchor into world coordinates
    if entry.anchor is not None:
        named_sample = _scale_position_sample(named_sample, entry.anchor)

    # Step 4: write the scaled values into the action instance in-place
    entry.parameterization.parameterize_object_with_sample(
        entry.instance, named_sample
    )

    if verbose:
        for var, value in named_sample.items():
            print(f"    {str(var.variable.name):<70} = {value}")


# ---------------------------------------------------------------------------
# Action description factories
# ---------------------------------------------------------------------------

def _build_action_descriptions(world, robot, milk_variable) -> list:
    """
    Return a list of ``probable_variable`` descriptions for all six actions in
    the plan.  Descriptions are pure data — no side effects occur here.
    """
    def navigable_pose():
        """
        Pose template with free x, y to be scaled at sample time.

        header is intentionally excluded from the probable() tree — the
        parameterizer cannot handle Header fields (Body, datetime, int).
        It is patched onto every PoseStamped instance after parameterization
        via _patch_header in sequential_plan_with_apartment.
        """
        return probable(PoseStamped)(
            pose=probable(PyCramPose)(
                position=probable(PyCramVector3)(x=..., y=..., z=0),
                orientation=probable(PyCramQuaternion)(x=0, y=0, z=0, w=1),
            ),
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
    milk_body, milk_initial_pose, milk_connection = _add_milk(world, MILK_STL)

    session = _create_session(_DATABASE_URI)

    rclpy.init()
    ros_node = rclpy.create_node("sequential_plan")
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    try:
        _tf_publisher  = TFPublisher(world=world, node=ros_node)
        _viz_publisher = VizMarkerPublisher(world=world, node=ros_node)

        context       = Context(world, robot, None)
        milk_variable = variable_from([milk_body])

        # Build descriptions, translate to instances and pre-compute distributions.
        # Each ActionEntry receives an anchor that constrains its sampled position
        # to a reachable neighbourhood in the world:
        #   - park_arms_start : no pose, no anchor
        #   - navigate_to_milk: anchor near the milk object
        #   - pick_up         : no pose, no anchor
        #   - navigate_to_place: anchor at the place position
        #   - place           : anchor at the place position
        #   - park_arms_end   : no pose, no anchor
        descriptions = _build_action_descriptions(world, robot, milk_variable)
        anchors = [
            None,                        # ParkArmsAction (start)
            NAVIGATE_TO_MILK_ANCHOR,     # NavigateAction → milk
            None,                        # PickUpAction
            NAVIGATE_TO_PLACE_ANCHOR,    # NavigateAction → place
            PLACE_ANCHOR,                # PlaceAction
            None,                        # ParkArmsAction (end)
        ]

        entries: list[ActionEntry] = [
            _build_action_entry(desc, anchor)
            for desc, anchor in zip(descriptions, anchors)
        ]

        # Patch the world-frame header onto every PoseStamped in every action.
        # Header is excluded from the probable() tree (the parameterizer cannot
        # handle its field types), so we restore it here after parameterization.
        world_header = Header(
            frame_id=world.root,
            stamp=datetime.datetime.now(),
            sequence=0,
        )
        for entry in entries:
            _patch_header(entry.instance, world_header)

        print(f"\nBuilt {len(entries)} per-action distributions:")
        for i, entry in enumerate(entries):
            anchor_str = f"anchor={entry.anchor}" if entry.anchor else "no anchor"
            print(
                f"  Action {i + 1}: {len(entry.parameterization.variables)} variables, "
                f"{len(entry.distribution.variables)} in distribution, {anchor_str}"
            )

        successful_count = 0

        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration} / {NUM_ITERATIONS}")
            print(f"{'=' * 60}")

            # Draw a fresh sample from each action's distribution, scale the
            # position variables around the relevant anchor, and write the
            # resulting concrete values into the action instances in-place.
            print("  Sampled parameters:")
            for entry in entries:
                _apply_sample(entry, verbose=True)
                # Restore header after each sample — parameterize_object_with_sample
                # may reconstruct PoseStamped and clear the header field.
                _patch_header(entry.instance, world_header)

            sp = SequentialPlan(context, *[e.instance for e in entries])

            with simulated_robot:
                # Reset milk inside the simulated_robot context. The connection's
                # internal _world reference is only valid while the simulation is
                # active — calling _reset_milk outside (after the context exits)
                # results in AttributeError: 'NoneType' has no attribute 'state'.
                _reset_milk(world, milk_connection, milk_initial_pose)
                try:
                    sp.perform()

                    # Plan succeeded — persist the full plan graph to the database.
                    _persist_plan(session, sp)
                    successful_count += 1
                    print(f"Iteration {iteration}: succeeded — plan persisted to database.")

                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"Iteration {iteration}: failed — plan not stored. ({exc})")

        print(
            f"\nDone. {successful_count}/{NUM_ITERATIONS} plans stored "
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