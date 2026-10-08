from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import List

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.exceptions import (
    GripperAxesNotPerpendicular,
    MoreThanOneBodyHeld,
    NothingHeld,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import Camera, EndEffector
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% fixtures

POSITION_TOLERANCE = 0.005
"""
How close two distances have to be, in meters, to count as the same distance here.
"""

HELD_BODY_OFFSET = HomogeneousTransformationMatrix.from_xyz_rpy(0.01, 0.02, 0.03)
"""
Where the body a gripper holds sits relative to the tool frame.
"""


@pytest.fixture
def pr2_gripper(pr2_world_copy) -> EndEffector:
    """
    The left gripper of a PR2, in a world built from the robot's description alone.
    """
    return pr2_world_copy.get_semantic_annotations_by_type(PR2)[0].left_arm.end_effector


@pytest.fixture
def graspable_box(pr2_world_copy) -> Milk:
    """
    A box-shaped milk carton within the PR2's reach, offering the default ring of
    grasps.
    """
    body = Body(
        name=PrefixedName("graspable_box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.2))]),
    )
    annotation = Milk(root=body)
    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            FixedConnection(
                parent=pr2_world_copy.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.6, z=0.9
                ),
            )
        )
        pr2_world_copy.add_semantic_annotation(annotation)
    return annotation


def hold_body(end_effector: EndEffector, name: str = "held_body") -> Body:
    """
    Attach a body below ``end_effector``'s tool frame, as grasping one does.

    :param end_effector: The gripper that should hold the body.
    :param name: The name of the body it should hold.
    :return: The body it now holds.
    """
    world = end_effector._world
    body = Body(name=PrefixedName(name))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=end_effector.tool_frame,
                child=body,
                parent_T_connection_expression=HELD_BODY_OFFSET,
            )
        )
    return body


# %% the axes a gripper states in its tool frame


@dataclass(eq=False)
class EndEffectorWithSkewedAxes(EndEffector):
    """
    An end effector whose closing axis is not perpendicular to its approach axis.
    """

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3(x=1, y=1, z=0, reference_frame=self.tool_frame)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


def test_the_approach_axis_is_the_grasp_frames_x_axis(pr2_gripper):
    np.testing.assert_allclose(
        (pr2_gripper.tool_R_grasp @ Vector3.X()).to_np()[:3],
        pr2_gripper.approach_axis.to_np()[:3],
        atol=1e-9,
    )


def test_the_closing_axis_is_the_grasp_frames_y_axis(pr2_gripper):
    np.testing.assert_allclose(
        (pr2_gripper.tool_R_grasp @ Vector3.Y()).to_np()[:3],
        pr2_gripper.closing_axis.to_np()[:3],
        atol=1e-9,
    )


def test_the_approach_axis_follows_the_grippers_own_convention(
    pr2_gripper, tracy_world
):
    """
    A PR2 gripper points along its tool frame's x-axis and Tracy's along its z-axis, so
    the very same grasp frame is approached along a different local axis.
    """
    tracy_gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector

    np.testing.assert_allclose(
        pr2_gripper.approach_axis.to_np()[:3], Vector3.X().to_np()[:3], atol=1e-9
    )
    np.testing.assert_allclose(
        tracy_gripper.approach_axis.to_np()[:3], Vector3.Z().to_np()[:3], atol=1e-9
    )


def test_a_gripper_whose_axes_are_not_perpendicular_is_refused(pr2_gripper):
    with pytest.raises(GripperAxesNotPerpendicular):
        EndEffectorWithSkewedAxes(
            name=PrefixedName("skewed_gripper"),
            root=pr2_gripper.root,
            tool_frame=pr2_gripper.tool_frame,
        )


def test_every_robot_states_its_axes_in_the_frame_they_belong_to(
    supported_abstract_robots,
):
    """
    A gripper's axes are read in its tool frame and a camera's in its root, for every
    robot there is.
    """
    for abstract_robot in supported_abstract_robots:
        world = URDFParser.from_file(abstract_robot.get_ros_file_path()).parse()
        abstract_robot.from_world(world)

        for end_effector in world.get_semantic_annotations_by_type(EndEffector):
            assert end_effector.approach_axis.reference_frame is end_effector.tool_frame
            assert end_effector.closing_axis.reference_frame is end_effector.tool_frame
        for camera in world.get_semantic_annotations_by_type(Camera):
            assert camera.forward_facing_axis.reference_frame is camera.root


# %% tool frame goals


def test_tool_frame_goal_keeps_the_grasp_position(pr2_gripper, graspable_box):
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    goal = pr2_gripper.tool_frame_goal(grasp)

    np.testing.assert_allclose(goal.to_np()[:3, 3], grasp.to_np()[:3, 3], atol=1e-9)


def test_tool_frame_goal_applies_the_end_effectors_own_orientation(
    pr2_gripper, graspable_box
):
    """
    Two grippers pointing different ways must be sent different orientations for one
    and the same grasp.
    """
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    goal = pr2_gripper.tool_frame_goal(grasp)

    expected = grasp.rotation_matrix @ pr2_gripper.tool_R_grasp.inverse()
    np.testing.assert_allclose(
        goal.rotation_matrix.to_np(), expected.to_np(), atol=1e-9
    )


# %% the body a gripper holds


def test_an_empty_gripper_holds_nothing(pr2_gripper):
    assert pr2_gripper.held_body is None


def test_the_held_body_is_the_one_below_the_tool_frame(pr2_gripper):
    body = hold_body(pr2_gripper)

    assert pr2_gripper.held_body is body


def test_a_gripper_with_two_bodies_attached_holds_no_single_one(pr2_gripper):
    """
    Which of them the grasp is on cannot be answered, so it is not guessed at.
    """
    hold_body(pr2_gripper, name="first_body")
    hold_body(pr2_gripper, name="second_body")

    with pytest.raises(MoreThanOneBodyHeld):
        pr2_gripper.held_body


def test_the_grasp_of_an_empty_gripper_cannot_be_read(pr2_gripper):
    with pytest.raises(NothingHeld):
        pr2_gripper.held_body_T_grasp


def test_the_held_grasp_is_turned_the_way_the_gripper_faces(pr2_gripper):
    """
    The grasp is what the tool frame reached, so applying the gripper's own orientation
    to it has to lead back to the way the tool frame points.
    """
    body = hold_body(pr2_gripper)

    body_R_grasp = pr2_gripper.held_body_T_grasp.rotation_matrix

    grasp_R_tool = pr2_gripper.tool_R_grasp.inverse()
    body_T_tool = pr2_gripper._world.transform(
        pr2_gripper.tool_frame.global_transform, body
    )
    np.testing.assert_allclose(
        (body_R_grasp @ grasp_R_tool).to_np(),
        body_T_tool.rotation_matrix.to_np(),
        atol=1e-9,
    )


def test_the_held_grasp_is_the_offset_the_body_hangs_at(pr2_gripper):
    body = hold_body(pr2_gripper)

    np.testing.assert_allclose(
        pr2_gripper.held_body_T_grasp.to_np()[:3, 3],
        -HELD_BODY_OFFSET.to_np()[:3, 3],
        atol=1e-9,
    )
    assert pr2_gripper.held_body_T_grasp.reference_frame is body


# %% where a two-finger gripper actually grips


def test_the_pads_are_the_tips_of_the_thumb_and_the_finger(pr2_gripper):
    assert pr2_gripper.pads == (pr2_gripper.thumb.tip, pr2_gripper.finger.tip)


def test_the_grasp_centre_lies_midway_between_the_pads(pr2_gripper):
    """
    The point a gripper takes hold at is between its pads, wherever its tool frame
    happens to sit.
    """
    centres = [
        pad.collision.as_bounding_box_collection_in_frame(pr2_gripper.tool_frame)
        .bounding_box()
        .center.to_np()[:3]
        .ravel()
        for pad in pr2_gripper.pads
    ]

    np.testing.assert_allclose(
        pr2_gripper.grasp_centre.to_np()[:3].ravel(),
        (centres[0] + centres[1]) / 2,
        atol=1e-9,
    )


def test_the_grasp_centre_is_read_in_the_frame_it_belongs_to(pr2_gripper):
    assert pr2_gripper.grasp_centre.reference_frame is pr2_gripper.tool_frame


def test_a_goal_for_the_grasp_centre_puts_the_pads_on_the_grasp(
    pr2_gripper, graspable_box
):
    """
    Reaching the goal places the tool frame, so the pads only meet the grasp if the
    goal was moved back by however far they sit from the tool frame.
    """
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    goal = pr2_gripper.tool_frame_goal_for_grasp_centre(grasp)

    reached = goal.to_np()
    centre_on_the_grasp = (
        reached[:3, :3] @ pr2_gripper.grasp_centre.to_np()[:3].ravel() + reached[:3, 3]
    )
    np.testing.assert_allclose(centre_on_the_grasp, grasp.to_np()[:3, 3], atol=1e-9)


def test_a_goal_for_the_grasp_centre_turns_the_gripper_the_same_way(
    pr2_gripper, graspable_box
):
    """
    Only where the gripper goes changes, not how it is turned.
    """
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    centred = pr2_gripper.tool_frame_goal_for_grasp_centre(grasp)

    np.testing.assert_allclose(
        centred.rotation_matrix.to_np(),
        pr2_gripper.tool_frame_goal(grasp).rotation_matrix.to_np(),
        atol=1e-9,
    )


def test_a_gripper_whose_pads_straddle_its_tool_frame_is_sent_somewhere_else(
    tracy_world, graspable_box
):
    """
    Tracy's pads sit a couple of centimetres from its tool frame, so the two goals for
    one grasp are that far apart; a gripper whose pads meet at its tool frame would get
    the same goal twice.
    """
    gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    centred = gripper.tool_frame_goal_for_grasp_centre(grasp).to_np()[:3, 3]

    offset = np.linalg.norm(gripper.grasp_centre.to_np()[:3])
    assert offset > 0.0
    assert np.linalg.norm(centred - gripper.tool_frame_goal(grasp).to_np()[:3, 3]) == (
        pytest.approx(offset)
    )


# %% closing a gripper to a width


def test_the_fingertip_frames_stand_a_pad_deep_outside_each_surface(pr2_gripper):
    """
    A controller is given a goal on the tip frames while an object is met by the
    surfaces, and the two differ by one pad depth on either side.
    """
    assert pr2_gripper._fingertip_distance == pytest.approx(
        pr2_gripper.pad_separation + 2 * pr2_gripper.pad_depth
    )


def test_asking_for_the_separation_it_already_has_gives_the_distance_it_already_has(
    pr2_gripper,
):
    assert pr2_gripper.fingertip_distance_for(pr2_gripper.pad_separation) == (
        pytest.approx(pr2_gripper._fingertip_distance)
    )


def test_a_wider_object_needs_the_fingertip_frames_further_apart(pr2_gripper):
    narrow = pr2_gripper.fingertip_distance_for(0.02)
    wide = pr2_gripper.fingertip_distance_for(0.05)

    assert wide - narrow == pytest.approx(0.03)


def test_every_two_finger_gripper_there_is_can_measure_its_own_pads(
    supported_abstract_robots,
):
    """
    The pad geometry is read off the gripper's own bodies and its own closing axis, so
    every robot that has a two-finger gripper has it.
    """
    for abstract_robot in supported_abstract_robots:
        world = URDFParser.from_file(abstract_robot.get_ros_file_path()).parse()
        abstract_robot.from_world(world)

        for gripper in world.get_semantic_annotations_by_type(HasTwoFingers):
            assert gripper.grasp_centre.reference_frame is gripper.tool_frame
            assert gripper.pad_depth > 0.0
            assert gripper.fingertip_distance_for(0.0) == pytest.approx(
                2 * gripper.pad_depth
            )
