from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict

from coraplex.plans.plan_node import PlanNode
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    or_,
    not_,
    and_,
    variable_from,
    ConditionType,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import GraspVerificationFailed
from coraplex.plans.factories import sequential, execute_single
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.mixins import (
    HasGraspDetectionThreshold,
    PlaceTuningParameters,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import (
    is_body_gripped,
    is_body_in_gripper,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceAction(ActionDescription, PlaceTuningParameters, HasGraspDetectionThreshold):
    """
    Places an Object at a position using an arm.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be place
    """
    target_location: Pose
    """
    Pose in the world at which the object should be placed.
    """

    arm: Arms
    """
    Arm that is currently holding the object
    """

    grasp_release_threshold: float = field(default=0.1, kw_only=True)
    """
    Maximum fraction of sampled rays between the gripper's fingers that may still hit
    :attr:`object_designator` for it to count as released (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """

    max_release_attempts: int = field(default=3, kw_only=True)
    """
    How many times to (re-)issue the OPEN motion, checking afterwards via
    :meth:`_object_released` whether :attr:`object_designator` is actually still
    detected between the fingers, before retracting regardless.

    Guards against retracting while still gripping -- which drags or drops the object
    from height instead of setting it down -- by only retracting once the object is
    confirmed clear of the gripper (or attempts run out).
    """

    def _previous_grasp_description(self) -> GraspDescription:
        """
        :return: The grasp description used by the preceding :class:`PickUpAction` on
            :attr:`object_designator`, or a default front/no-alignment grasp if none is
            found.
        """
        end_effector = ViewManager.get_arm_view(self.arm, self.robot).end_effector
        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        if previous_pick:
            return previous_pick.designator.grasp_description
        return GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
        )

    def _pose_sequence(self) -> tuple[Pose, Pose, Pose]:
        """
        :return: The (transport, placing, retract) poses for this placement.
        """
        return self._previous_grasp_description().pose_sequence(
            self.target_location, self.object_designator, reverse=True
        )

    def _transport_and_descend_plan(
        self, transport_pose: Pose, placing_pose: Pose
    ) -> PlanNode:
        """
        :return: The plan that carries the held object above the target location and
            then descends onto it.
        """
        return sequential(
            [
                MoveToolCenterPointMotion(
                    transport_pose,
                    self.arm,
                    allow_gripper_collision=False,
                    max_linear_velocity=self.transport_linear_velocity,
                ),
                MoveToolCenterPointMotion(
                    placing_pose,
                    self.arm,
                    allow_gripper_collision=False,
                    max_linear_velocity=self.placing_linear_velocity,
                ),
            ],
        )

    def _retract_plan(self, retract_pose: Pose) -> PlanNode:
        """
        :return: The plan that retracts the end effector away from the placed object.
        """
        # A DetachNode(body=self.object_designator, new_parent=self.world.root)
        # would normally go here, but is unnecessary: the object is held only by
        # real contact/friction (AttachNode is likewise unnecessary in
        # PickUpAction, for the same reason -- see its own comment), so nothing
        # needs kinematically re-parenting once released.
        return MoveToolCenterPointMotion(
            retract_pose,
            self.arm,
            max_linear_velocity=self.retract_linear_velocity,
        )

    def _object_released(self) -> bool:
        """
        :return: Whether :attr:`object_designator` is no longer detected between the
            gripper's fingers.
        """
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)
        return (
            is_body_in_gripper(self.object_designator, end_effector)
            < self.grasp_release_threshold
        )

    def _object_held(self) -> bool:
        """
        :return: Whether :attr:`object_designator` is currently detected between the
            gripper's fingers.
        """
        end_effector = ViewManager.get_end_effector_view(self.arm, self.robot)
        return (
            is_body_in_gripper(self.object_designator, end_effector)
            > self.grasp_detection_threshold
        )

    @property
    def _action_plan(self) -> PlanNode:
        transport_pose, placing_pose, retract_pose = self._pose_sequence()
        return sequential(
            [
                self._transport_and_descend_plan(transport_pose, placing_pose),
                MoveGripperMotion(
                    GripperState.OPEN,
                    self.arm,
                    finger_velocity=self.release_opening_velocity,
                ),
                self._retract_plan(retract_pose),
            ],
            self.context,
        )

    def execute(self) -> Any:
        """
        Transport and descend onto :attr:`target_location`, then retry the OPEN
        motion up to :attr:`max_release_attempts` times -- verifying via
        :meth:`_object_released` after each -- before retracting.

        :raises GraspVerificationFailed: If :attr:`object_designator` is not detected
            in the gripper before transport starts, or is no longer detected right
            after descending onto :attr:`target_location` -- either way it slipped out
            (before this action started, e.g. during the park move that carries it
            here, or during this action's own transport/descent) rather than actually
            arriving at the target still held, so "releasing" it there would silently
            do nothing and still report success.
        """
        if not self._object_held():
            raise GraspVerificationFailed(
                object_designator=self.object_designator, attempts=0
            )
        transport_pose, placing_pose, retract_pose = self._pose_sequence()
        self.add_subplan(
            self._transport_and_descend_plan(transport_pose, placing_pose)
        ).perform()
        if not self._object_held():
            raise GraspVerificationFailed(
                object_designator=self.object_designator, attempts=0
            )
        for _ in range(self.max_release_attempts):
            self.add_subplan(
                execute_single(
                    MoveGripperMotion(
                        GripperState.OPEN,
                        self.arm,
                        finger_velocity=self.release_opening_velocity,
                    )
                )
            ).perform()
            if self._object_released():
                break
        self.add_subplan(self._retract_plan(retract_pose)).perform()

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The object needs to be in the gripper frame.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return or_(
            not_(GripperIsFree(end_effector)),
            is_body_gripped(
                variable_from(kwargs["object_designator"]),
                end_effector,
                threshold=kwargs["grasp_detection_threshold"],
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The gripper must be free again and the object needs to be at the target
        location.
        """
        end_effector = ViewManager.get_end_effector_view(
            variables["arm"], context.robot
        )
        return and_(
            GripperIsFree(end_effector),
            not_(
                is_body_gripped(
                    variable_from(kwargs["object_designator"]),
                    end_effector,
                    threshold=kwargs["grasp_release_threshold"],
                )
            ),
            allclose(
                variable_from(kwargs["object_designator"]).global_pose,
                kwargs["target_location"],
                atol=0.03,
            ),
        )
