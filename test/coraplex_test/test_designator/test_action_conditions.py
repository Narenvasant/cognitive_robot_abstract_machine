from unittest.mock import patch

import pytest

from krrood.entity_query_language.factories import (
    get_false_statements,
    evaluate_condition,
    ConditionType,
)
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import (
    ConditionNotSatisfied,
    GraspVerificationFailed,
    MotionDidNotFinish,
)
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def _construct_and_evaluate_condition(action, action_condition):

    condition = action_condition(
        action.bound_variables,
        action.context,
        action.designator_parameter,
    )
    evaluation = evaluate_condition(condition)
    if evaluation:
        return True
    raise ConditionNotSatisfied(
        pre_condition=True, action=action.__class__, condition=condition
    )


def test_get_bound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )

    bound_variables = pick_action._create_variables()

    assert len(bound_variables) == 12
    assert list(bound_variables.keys()) == [
        "grasp_detection_threshold",
        "pre_approach_linear_velocity",
        "final_approach_linear_velocity",
        "grasp_closing_velocity",
        "lift_linear_velocity",
        "grasp_stall_minimum_time",
        "object_friction",
        "object_designator",
        "arm",
        "grasp_description",
        "tolerate_grasp_stall",
        "max_grasp_attempts",
    ]
    assert list(bound_variables["arm"]._domain_) == [Arms.LEFT]
    assert bound_variables["arm"]._type_ == Arms
    assert list(bound_variables["object_designator"]._domain_) == [
        world.get_body_by_name("milk.stl")
    ]
    assert bound_variables["object_designator"]._type_ == Body


def test_pick_up_pre_conditions(mutable_model_world):
    world, view, context = mutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )

    plan = sequential([pick_action], context)

    with pytest.raises(ConditionNotSatisfied):
        _construct_and_evaluate_condition(
            pick_action,
            pick_action.pre_condition,
        )

    pre_condition = pick_action.pre_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )

    false_statements = get_false_statements(pre_condition)

    assert len(false_statements) == 1
    assert false_statements[0]._name_ == "IsObjectReachableBy"

    with pytest.raises(ConditionNotSatisfied):
        _construct_and_evaluate_condition(pick_action, pick_action.pre_condition)

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pre_condition = pick_action.pre_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )

    assert evaluate_condition(pre_condition) == True

    with simulated_robot:
        plan.perform()

    assert evaluate_condition(pre_condition) == False
    _construct_and_evaluate_condition(pick_action, pick_action.post_condition)
    assert _construct_and_evaluate_condition(pick_action, pick_action.post_condition)


def test_pick_up_post_condition(mutable_model_world):
    world, view, context = mutable_model_world
    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.8, 2, 0
    )

    plan = sequential([pick_action], context)

    assert _construct_and_evaluate_condition(pick_action, pick_action.pre_condition)

    with simulated_robot:
        plan.perform()

    assert world.get_body_by_name(
        "milk.stl"
    ) in world.get_kinematic_structure_entities_of_branch(
        view.left_arm.end_effector.tool_frame
    )

    assert _construct_and_evaluate_condition(pick_action, pick_action.post_condition)


# %% grasp/release retry behaviour


def test_pick_up_retries_after_a_failed_verification(immutable_model_world):
    """
    A grasp that fails verification (:meth:`PickUpAction._grasp_succeeded`) on its
    first attempt is retried, and succeeds without raising once a later attempt
    verifies.
    """
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )
    sequential([pick_action], context)

    with patch.object(
        PickUpAction, "_grasp_succeeded", side_effect=[False, True, True]
    ) as grasp_succeeded, patch.object(PickUpAction, "add_subplan") as add_subplan:
        add_subplan.return_value.perform.return_value = None
        pick_action.execute()

    assert grasp_succeeded.call_count == 3
    # one grasp attempt + one re-open (failed attempt), then one grasp attempt + one
    # lift (successful attempt)
    assert add_subplan.call_count == 4


def test_pick_up_raises_after_exhausting_grasp_attempts(immutable_model_world):
    """
    A grasp that never verifies raises :class:`GraspVerificationFailed` after
    :attr:`PickUpAction.max_grasp_attempts` attempts, without ever lifting.
    """
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )
    sequential([pick_action], context)

    with patch.object(
        PickUpAction, "_grasp_succeeded", return_value=False
    ) as grasp_succeeded, patch.object(PickUpAction, "add_subplan") as add_subplan:
        add_subplan.return_value.perform.return_value = None
        with pytest.raises(GraspVerificationFailed) as excinfo:
            pick_action.execute()

    assert grasp_succeeded.call_count == pick_action.max_grasp_attempts
    assert excinfo.value.object_designator == pick_action.object_designator
    assert excinfo.value.attempts == pick_action.max_grasp_attempts


def test_place_retries_release_until_confirmed(immutable_model_world):
    """
    A release that is not yet confirmed (:meth:`PlaceAction._object_released`) is
    retried before retracting.
    """
    world, view, context = immutable_model_world

    place_action = PlaceAction(
        world.get_body_by_name("milk.stl"),
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        Arms.LEFT,
    )
    sequential([place_action], context)

    with patch.object(
        PlaceAction, "_object_held", return_value=True
    ), patch.object(
        PlaceAction, "_object_released", side_effect=[False, True]
    ) as object_released, patch.object(PlaceAction, "add_subplan") as add_subplan:
        add_subplan.return_value.perform.return_value = None
        place_action.execute()

    assert object_released.call_count == 2
    # transport+descend, two OPEN attempts, then retract
    assert add_subplan.call_count == 4


def test_place_retracts_after_exhausting_release_attempts(immutable_model_world):
    """
    A release that never confirms still retracts once
    :attr:`PlaceAction.max_release_attempts` is exhausted, rather than holding the
    end effector at the placing pose indefinitely.
    """
    world, view, context = immutable_model_world

    place_action = PlaceAction(
        world.get_body_by_name("milk.stl"),
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        Arms.LEFT,
    )
    sequential([place_action], context)

    with patch.object(
        PlaceAction, "_object_held", return_value=True
    ), patch.object(
        PlaceAction, "_object_released", return_value=False
    ) as object_released, patch.object(PlaceAction, "add_subplan") as add_subplan:
        add_subplan.return_value.perform.return_value = None
        place_action.execute()

    assert object_released.call_count == place_action.max_release_attempts
    # transport+descend, max_release_attempts OPEN attempts, then retract
    assert add_subplan.call_count == place_action.max_release_attempts + 2


def test_place_raises_when_object_not_held_before_transport(immutable_model_world):
    """
    :meth:`PlaceAction.execute` raises :class:`GraspVerificationFailed` immediately,
    without transporting, if :attr:`PlaceAction.object_designator` is not held when
    placing starts (e.g. it slipped out during the carrying park beforehand).
    """
    world, view, context = immutable_model_world

    place_action = PlaceAction(
        world.get_body_by_name("milk.stl"),
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        Arms.LEFT,
    )
    sequential([place_action], context)

    with patch.object(
        PlaceAction, "_object_held", return_value=False
    ), patch.object(PlaceAction, "add_subplan") as add_subplan:
        add_subplan.return_value.perform.return_value = None
        with pytest.raises(GraspVerificationFailed) as excinfo:
            place_action.execute()

    add_subplan.assert_not_called()
    assert excinfo.value.object_designator == place_action.object_designator


def test_place_raises_when_object_slips_during_transport(immutable_model_world):
    """
    :meth:`PlaceAction.execute` raises :class:`GraspVerificationFailed` right after
    transporting, without attempting any release, if
    :attr:`PlaceAction.object_designator` is no longer held once the descent
    finishes -- it slipped out during this action's own transport/descent rather than
    actually arriving at the target.
    """
    world, view, context = immutable_model_world

    place_action = PlaceAction(
        world.get_body_by_name("milk.stl"),
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        Arms.LEFT,
    )
    sequential([place_action], context)

    with patch.object(
        PlaceAction, "_object_held", side_effect=[True, False]
    ), patch.object(PlaceAction, "add_subplan") as add_subplan:
        add_subplan.return_value.perform.return_value = None
        with pytest.raises(GraspVerificationFailed) as excinfo:
            place_action.execute()

    # only the transport+descend subplan was ever added -- no release, no retract
    assert add_subplan.call_count == 1
    assert excinfo.value.object_designator == place_action.object_designator
