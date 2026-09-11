"""
The ten-milk clutter on Tracy's own table, built for MuJoCo from a
:class:`~experiments.causal_reasoning.tracy_rspn.domain.ClutterSceneLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from coraplex.datastructures.enums import Arms
from typing_extensions import Dict, List

from experiments.causal_reasoning.tracy_rspn.domain import ClutterSceneLayout
from experiments.tracy_experiments.equipment import (
    add_box,
    apply_gravity_compensation,
    equip_arms_with_servos,
    equip_grippers_with_servos,
    exclude_self_collision,
    joint_state_of_type,
    mount_stationary_robot,
    parse_tracy,
    tracy_table_mount_position,
)
from experiments.tracy_experiments.grasp_contact import (
    GRASP_FRICTION,
    SURFACE_FRICTION,
    apply_contact_friction,
    apply_grasp_contact_parameters,
)
from semantic_digital_twin.adapters.multi_sim import MujocoCamera, MujocoLight
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Actuator, Body

MILK_SIZE = Scale(0.06, 0.06, 0.15)
"""
Edge lengths of a milk carton, in metres: a small carton, narrow enough to leave the
Robotiq 2F-85's 85mm opening room to close around it even when it stands a little
turned.
"""

MILK_COLOR = Color(0.95, 0.95, 0.9, 1.0)
"""
Colour of every carton but the target.
"""

TARGET_COLOR = Color(0.2, 0.5, 0.9, 1.0)
"""
Colour of the carton to pick, so a viewer can tell it apart.
"""

PICK_ARM = Arms.LEFT
"""
The arm that picks; the clutter stands in front of it.
"""

TRACY_MOUNT_X = 0.0
TRACY_MOUNT_Y = 0.0
"""
Where Tracy's own root is bolted, in the scene's root frame.
"""

SCENE_CAMERA_NAME = "clutter_overview_camera"
"""
Name of the fixed camera framing the clutter, for screenshots of a run.
"""

CAMERA_FRAME_MARGIN = 0.15
"""
How far, in metres, the camera's framed box extends beyond the cartons on every side, so
the gripper reaching in stays in view.
"""

CAMERA_DISTANCE_FACTOR = 1.1
"""
How far the camera stands back from the framed box, as a multiple of the box's diagonal.
"""

SCENE_LIGHT_NAME = "clutter_light"
"""
Name of the directional light over the table.
"""


def milk_name(index: int) -> str:
    """
    :param index: The carton's position in the layout's object list.
    :return: The name its body gets.
    """
    return f"milk_{index}"


@dataclass
class MilkClutterWorld:
    """
    Tracy, its table, and one layout's cartons standing on it, equipped to be driven by
    direct MuJoCo actuator control.
    """

    layout: ClutterSceneLayout
    """
    The layout the cartons stand in.
    """

    world: World = field(init=False)
    """
    The assembled world.
    """

    robot: Tracy = field(init=False)
    """
    The mounted robot.
    """

    table_top_z: float = field(init=False)
    """
    Height of the table's top surface above the world root, in metres.
    """

    milks: List[Body] = field(init=False, default_factory=list)
    """
    The cartons, in the order of :attr:`ClutterSceneLayout.objects`.
    """

    actuators: Dict[str, Actuator] = field(init=False, default_factory=dict)
    """
    Every arm and gripper joint's own actuator, keyed by joint name.
    """

    def __post_init__(self):
        tracy_world = parse_tracy()
        mount_position, self.table_top_z = tracy_table_mount_position(
            tracy_world, x=TRACY_MOUNT_X, y=TRACY_MOUNT_Y
        )
        self.world = World()
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(
                Body(name=PrefixedName(name="root", prefix="clutter"))
            )
        self.robot = mount_stationary_robot(
            self.world, Tracy, tracy_world, mount_position
        )
        self._add_milks()
        self._add_camera()
        self._equip_robot()

    @property
    def target(self) -> Body:
        """
        The carton to pick.
        """
        return self.milks[self.layout.target_index]

    @property
    def neighbours(self) -> List[Body]:
        """
        Every carton but the target, in the order of
        :attr:`ClutterSceneLayout.neighbours`.
        """
        return [
            milk
            for index, milk in enumerate(self.milks)
            if index != self.layout.target_index
        ]

    def _add_milks(self) -> None:
        """
        Stand every carton of the layout on the table, with the layout's own friction.
        """
        for index, placed in enumerate(self.layout.objects):
            self.milks.append(
                add_box(
                    self.world,
                    milk_name(index),
                    Point3(placed.x, placed.y, self.table_top_z + MILK_SIZE.z / 2),
                    MILK_SIZE,
                    TARGET_COLOR if index == self.layout.target_index else MILK_COLOR,
                    yaw=placed.yaw,
                )
            )
        friction = [self.layout.friction_coefficient] + list(GRASP_FRICTION[1:])
        apply_grasp_contact_parameters(self.milks, friction)
        apply_contact_friction(self._fingertip_pads(), friction)
        apply_contact_friction([self.robot.root], SURFACE_FRICTION)

    def _fingertip_pads(self) -> List[Body]:
        """
        The picking arm's two fingertip pads.

        MuJoCo gives a contact the larger of its two geoms' friction, so a carton's
        friction only governs the grasp if the pads closing on it carry no more than
        that themselves.
        """
        prefix = "left_" if PICK_ARM == Arms.LEFT else "right_"
        return [
            self.world.get_body_by_name(f"{prefix}robotiq_85_{side}_finger_tip_link")
            for side in ("left", "right")
        ]

    def _add_camera(self) -> None:
        """
        Attach a fixed camera to the world root that frames the cartons.
        """
        xs = [placed.x for placed in self.layout.objects]
        ys = [placed.y for placed in self.layout.objects]
        bounds = np.array(
            [
                [
                    min(xs) - CAMERA_FRAME_MARGIN,
                    min(ys) - CAMERA_FRAME_MARGIN,
                    self.table_top_z,
                ],
                [
                    max(xs) + CAMERA_FRAME_MARGIN,
                    max(ys) + CAMERA_FRAME_MARGIN,
                    self.table_top_z + MILK_SIZE.z + CAMERA_FRAME_MARGIN,
                ],
            ]
        )
        pose = MujocoCamera.overview_pose(
            bounds, distance_factor=CAMERA_DISTANCE_FACTOR
        )
        quaternion_xyzw = pose.to_quaternion().to_np().tolist()
        self.world.root.simulator_additional_properties.append(
            MujocoCamera(
                name=SCENE_CAMERA_NAME,
                body=self.world.root,
                position=pose.to_position().to_np()[:3].tolist(),
                quaternion=[quaternion_xyzw[3]] + quaternion_xyzw[:3],
            )
        )
        self.world.root.simulator_additional_properties.append(
            MujocoLight(
                name=SCENE_LIGHT_NAME,
                body=self.world.root,
                directional=True,
                position=[float(bounds[1][0]), float(bounds[0][1]), 3.0],
                direction=[-0.3, 0.3, -1.0],
                ambient=[0.3, 0.3, 0.3],
                diffuse=[0.6, 0.6, 0.6],
            )
        )

    def _equip_robot(self) -> None:
        """
        Park both arms, open the left gripper, and give every joint a position servo.
        """
        for arm in (self.robot.left_arm, self.robot.right_arm):
            joint_state_of_type(arm, StaticJointState.PARK).apply_to(self.world)
        joint_state_of_type(
            self.robot.right_arm.end_effector, GripperState.CLOSE
        ).apply_to(self.world)
        joint_state_of_type(
            self.robot.left_arm.end_effector, GripperState.OPEN
        ).apply_to(self.world)
        self.world.notify_state_change()
        apply_gravity_compensation(self.world, self.robot)
        exclude_self_collision(self.world, self.robot)
        self.actuators = {
            **equip_arms_with_servos(self.world, self.robot),
            **equip_grippers_with_servos(self.world, self.robot),
        }
