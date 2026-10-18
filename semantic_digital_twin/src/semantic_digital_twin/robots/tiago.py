from __future__ import annotations

import math
import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Dict, Self, List

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.sensors.lidar import Lidar, LidarSource
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLidar,
    HasLeftRightArm,
    HasMobileBase,
    HasNeck,
    HasTorso,
    HasTwoFingers,
    TGenericLeftFinger,
    TGenericRightFinger,
    HasEndEffector,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    AbstractRobotPart,
    Arm,
    Camera,
    Finger,
    MobileBase,
    Neck,
    Torso,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    DriveServos,
    JointServo,
    JointDynamics,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class TiagoJoint(StrEnum):
    """
    Names of the Tiago's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Connections that no controller commands, such as the base wheels and the
        gripper's coupled knuckle and inner finger joints, are left out.
    """

    TORSO_LIFT = "torso_lift_joint"
    HEAD_1 = "head_1_joint"
    HEAD_2 = "head_2_joint"

    LEFT_ARM_1 = "arm_left_1_joint"
    LEFT_ARM_2 = "arm_left_2_joint"
    LEFT_ARM_3 = "arm_left_3_joint"
    LEFT_ARM_4 = "arm_left_4_joint"
    LEFT_ARM_5 = "arm_left_5_joint"
    LEFT_ARM_6 = "arm_left_6_joint"
    LEFT_ARM_7 = "arm_left_7_joint"
    LEFT_GRIPPER_FINGER = "gripper_left_finger_joint"

    RIGHT_ARM_1 = "arm_right_1_joint"
    RIGHT_ARM_2 = "arm_right_2_joint"
    RIGHT_ARM_3 = "arm_right_3_joint"
    RIGHT_ARM_4 = "arm_right_4_joint"
    RIGHT_ARM_5 = "arm_right_5_joint"
    RIGHT_ARM_6 = "arm_right_6_joint"
    RIGHT_ARM_7 = "arm_right_7_joint"
    RIGHT_GRIPPER_FINGER = "gripper_right_finger_joint"


@dataclass(eq=False)
class TiagoLeftThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoLeftIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoRightThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoRightIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoLeftGripper(
    EndEffector, HasTwoFingers[TiagoLeftThumb, TiagoLeftIndexFinger]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.LEFT_GRIPPER_FINGER,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.045, 0.045])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_grasping_frame"
            ),
        )


@dataclass(eq=False)
class TiagoRightGripper(
    EndEffector, HasTwoFingers[TiagoRightThumb, TiagoRightIndexFinger]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.RIGHT_GRIPPER_FINGER,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.045, 0.045])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_grasping_frame"
            ),
        )


@dataclass(eq=False)
class TiagoLeftArm(Arm[TiagoLeftGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.LEFT_ARM_1,
            TiagoJoint.LEFT_ARM_2,
            TiagoJoint.LEFT_ARM_3,
            TiagoJoint.LEFT_ARM_4,
            TiagoJoint.LEFT_ARM_5,
            TiagoJoint.LEFT_ARM_6,
            TiagoJoint.LEFT_ARM_7,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0, -0.8, 1.57, 1.57, -2.0, 1.1, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_tool_link"
            ),
        )


@dataclass(eq=False)
class TiagoRightArm(Arm[TiagoRightGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.RIGHT_ARM_1,
            TiagoJoint.RIGHT_ARM_2,
            TiagoJoint.RIGHT_ARM_3,
            TiagoJoint.RIGHT_ARM_4,
            TiagoJoint.RIGHT_ARM_5,
            TiagoJoint.RIGHT_ARM_6,
            TiagoJoint.RIGHT_ARM_7,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0, -0.8, 1.57, 1.57, -2.0, 1.1, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_tool_link"
            ),
        )


@dataclass(eq=False)
class TiagoCamera(Camera):

    @property
    def forward_facing_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.root)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_front_camera_optical_frame"
            ),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.0665,
            maximal_height=1.4165,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class TiagoNeck(Neck[TiagoCamera]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.HEAD_1,
            TiagoJoint.HEAD_2,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )


@dataclass(eq=False)
class TiagoTorso(
    Torso, HasLeftRightArm[TiagoLeftArm, TiagoRightArm], HasNeck[TiagoNeck]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.TORSO_LIFT,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.175])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.34])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_fixed_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class TiagoBaseLidar(Lidar):
    """
    The SICK TIM551 scanner sweeping the floor in front of the Tiago's base.

    ..note:: The description's own beam count is not a whole number, so the scanner's
        angular resolution of a third of a degree gives the angle between two beams.
    """

    @classmethod
    def with_source(
        cls, robot_root: KinematicStructureEntity, source: LidarSource
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "base_laser_link"
            ),
            scan_pattern=ScanPattern(
                minimum_angle=-1.658133,
                maximum_angle=1.66347956,
                angle_increment=0.00581718,
                minimum_range=0.05,
                maximum_range=10.0,
            ),
            source=source,
        )


@dataclass(eq=False)
class TiagoMobileBase(
    MobileBase[DifferentialDrive],
    HasTorso[TiagoTorso],
    HasLidar[TiagoBaseLidar],
):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    full_body_controlled: bool = field(default=True, kw_only=True)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @property
    def drive_servos(self) -> DriveServos:
        """
        Servos stiff enough for the base to follow its commanded pose while carrying the
        whole robot.

        They stand in for the wheels, which a simulation turns into neither traction nor
        steering, so they are not taken from the real robot; each is critically damped
        for the mass it moves.
        """
        mass = sum(
            body.inertial.mass
            for body in self._robot.bodies
            if body.inertial is not None
        )
        translation_stiffness = 300000.0
        rotation_stiffness = 100000.0
        return DriveServos(
            translation_gains=ServoGains(
                stiffness=translation_stiffness,
                damping=2.0 * math.sqrt(translation_stiffness * mass),
                torque_limit=10000.0,
            ),
            rotation_gains=ServoGains(
                stiffness=rotation_stiffness,
                damping=2.0 * math.sqrt(rotation_stiffness * mass),
                torque_limit=5000.0,
            ),
        )

    def _setup_servos(self) -> None:
        """
        Declare the drive's servos, see :meth:`MobileBase._setup_servos`, and let a
        physical simulation carry the weight of every body standing on the base.
        """
        super()._setup_servos()
        self._compensate_gravity()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )


@dataclass(eq=False)
class Tiago(AbstractRobot, HasMobileBase[TiagoMobileBase]):
    """
    The Tiago++ robot by PAL Robotics with updated Robotiq grippers.

    https://pal-robotics.com/blog/tiago-bi-manual-robot-research/
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "tiago_from_our_robot.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03,
                    violated_distance=0.0,
                    robot=self,
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)


# %% what MuJoCo's own model of the robot drives its joints with


@dataclass(frozen=True)
class TiagoMujocoServo:
    """
    What one joint of this robot is driven with in MuJoCo's own model of it.

    The stiffness and the force limit are the model's own. The damping is what MuJoCo
    works out from the joint's inertia for the critical damping the model asks for, so
    the two arms differ a little even where their stiffnesses match.
    """

    stiffness: float
    """
    Restoring torque per radian, or newton per metre, away from the commanded position.
    """

    damping: float
    """
    Opposing torque per radian per second.
    """

    force_limit: float
    """
    The largest torque, or force, the servo may exert.
    """

    armature: float = 0.1
    """
    The inertia of what spins behind the joint, in kilogram metres squared.

    The description states none, and a joint with none is only as heavy as the links
    beyond it, which for a wrist is almost nothing. Held by a servo this stiff, such a
    joint oscillates apart within a second of simulated time and takes the arm with it.
    A geared robot joint carries the inertia of its rotor, and stating it is what makes
    the arm steady.
    """

    @property
    def servo(self) -> JointServo:
        """
        The servo these values describe, leaving the joint's own inertia and friction as
        the description states them.
        """
        return JointServo(
            gains=ServoGains(
                stiffness=self.stiffness,
                damping=self.damping,
                torque_limit=self.force_limit,
            ),
            dynamics=JointDynamics(armature=self.armature),
        )


@dataclass(eq=False)
class ServoedByItsMujocoModel(AbstractRobotPart, ABC):
    """
    A robot part whose joints are driven by the servos MuJoCo's own model of the robot
    states, so that a physical simulation moves it instead of a kinematic one placing
    it.
    """

    @classproperty
    def servo_of_each_joint(cls) -> Dict[str, TiagoMujocoServo]:
        """
        What drives every servoed joint of the MuJoCo description, by the name that
        description gives it.
        """
        return {
            "torso_lift_joint": TiagoMujocoServo(
                stiffness=50000.0, damping=2361.08, force_limit=500.0
            ),
            "head_1_joint": TiagoMujocoServo(
                stiffness=20.0, damping=0.83, force_limit=5.197
            ),
            "head_2_joint": TiagoMujocoServo(
                stiffness=20.0, damping=0.87, force_limit=5.197
            ),
            "arm_left_1_joint": TiagoMujocoServo(
                stiffness=250.0, damping=48.53, force_limit=43.0
            ),
            "arm_left_2_joint": TiagoMujocoServo(
                stiffness=1500.0, damping=94.28, force_limit=43.0
            ),
            "arm_left_3_joint": TiagoMujocoServo(
                stiffness=600.0, damping=4.36, force_limit=26.0
            ),
            "arm_left_4_joint": TiagoMujocoServo(
                stiffness=1000.0, damping=35.38, force_limit=26.0
            ),
            "arm_left_5_joint": TiagoMujocoServo(
                stiffness=80.0, damping=0.97, force_limit=3.0
            ),
            "arm_left_6_joint": TiagoMujocoServo(
                stiffness=80.0, damping=2.57, force_limit=6.6
            ),
            "arm_left_7_joint": TiagoMujocoServo(
                stiffness=80.0, damping=0.54, force_limit=6.6
            ),
            "arm_right_1_joint": TiagoMujocoServo(
                stiffness=250.0, damping=45.14, force_limit=43.0
            ),
            "arm_right_2_joint": TiagoMujocoServo(
                stiffness=1500.0, damping=87.68, force_limit=43.0
            ),
            "arm_right_3_joint": TiagoMujocoServo(
                stiffness=600.0, damping=3.90, force_limit=26.0
            ),
            "arm_right_4_joint": TiagoMujocoServo(
                stiffness=1000.0, damping=33.51, force_limit=26.0
            ),
            "arm_right_5_joint": TiagoMujocoServo(
                stiffness=80.0, damping=0.66, force_limit=3.0
            ),
            "arm_right_6_joint": TiagoMujocoServo(
                stiffness=80.0, damping=2.56, force_limit=6.6
            ),
            "arm_right_7_joint": TiagoMujocoServo(
                stiffness=80.0, damping=0.53, force_limit=6.6
            ),
            "gripper_left_left_finger_joint": TiagoMujocoServo(
                stiffness=500.0, damping=14.63, force_limit=5.197
            ),
            "gripper_left_right_finger_joint": TiagoMujocoServo(
                stiffness=500.0, damping=14.63, force_limit=5.197
            ),
            "gripper_right_left_finger_joint": TiagoMujocoServo(
                stiffness=500.0, damping=14.63, force_limit=5.197
            ),
            "gripper_right_right_finger_joint": TiagoMujocoServo(
                stiffness=500.0, damping=14.63, force_limit=5.197
            ),
        }

    def _setup_servos(self) -> None:
        for connection in self.active_connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            self._declare_servo(
                connection, self.servo_of_each_joint[connection.name.name].servo
            )
        self._compensate_gravity()


@dataclass(eq=False)
class TiagoMujocoLeftThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftGripper(
    EndEffector,
    HasTwoFingers[TiagoMujocoLeftThumb, TiagoMujocoLeftIndexFinger],
    ServoedByItsMujocoModel,
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        """
        The hand stands at the ends of its fingers' own travel: wide open leaves the
        most room to reach around an object, and shut all the way has whatever is
        between the pads stop them, so the servos hold it with the force they are rated
        for.
        """
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.upper.position
                for connection in gripper_joints
            },
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.lower.position
                for connection in gripper_joints
            },
            state_type=GripperState.CLOSE,
        )
        return [gripper_close, gripper_open]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_grasping_frame"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightGripper(
    EndEffector,
    HasTwoFingers[TiagoMujocoRightThumb, TiagoMujocoRightIndexFinger],
    ServoedByItsMujocoModel,
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        """
        The hand stands at the ends of its fingers' own travel: wide open leaves the
        most room to reach around an object, and shut all the way has whatever is
        between the pads stop them, so the servos hold it with the force they are rated
        for.
        """
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.upper.position
                for connection in gripper_joints
            },
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.lower.position
                for connection in gripper_joints
            },
            state_type=GripperState.CLOSE,
        )
        return [gripper_close, gripper_open]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_grasping_frame"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftArm(Arm[TiagoMujocoLeftGripper], ServoedByItsMujocoModel):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightArm(Arm[TiagoMujocoRightGripper], ServoedByItsMujocoModel):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoCamera(Camera):
    """
    Tiago's head camera, rooted in the head itself.

    The MuJoCo model draws no separate optical frame, so the head's own frame stands in
    for it: the field of view is the camera's, while its origin sits a few centimetres
    behind and below where the lens really is.
    """

    @property
    def forward_facing_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.root)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_2_link"
            ),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.0665,
            maximal_height=1.4165,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class TiagoMujocoNeck(Neck[TiagoMujocoCamera], ServoedByItsMujocoModel):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )


@dataclass(eq=False)
class TiagoMujocoTorso(
    Torso,
    HasLeftRightArm[TiagoMujocoLeftArm, TiagoMujocoRightArm],
    HasNeck[TiagoMujocoNeck],
    ServoedByItsMujocoModel,
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )
        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.15])),
            state_type=TorsoState.MID,
        )
        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.35])),
            state_type=TorsoState.HIGH,
        )
        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoMobileBase(MobileBase[DifferentialDrive], HasTorso[TiagoMujocoTorso]):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @property
    def drive_servos(self) -> DriveServos:
        """
        See :attr:`TiagoMobileBase.drive_servos`.
        """
        return TiagoMobileBase.drive_servos.fget(self)

    def _setup_servos(self) -> None:
        """
        Declare the drive's servos, see :meth:`MobileBase._setup_servos`, and let a
        physical simulation carry the weight of every body standing on the base.
        """
        super()._setup_servos()
        self._compensate_gravity()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            full_body_controlled=False,
        )


@dataclass(eq=False)
class TiagoMujoco(AbstractRobot, HasMobileBase[TiagoMujocoMobileBase]):
    """
    Class that describes the Take It And Go Robot (TIAGo).

    Described by MuJoCo's own model of the robot rather than by the URDF, which is
    what makes its hand able to hold something: the model's gripper is two sliding
    fingers with flat faces, where the URDF's is a four-bar linkage whose fingers follow
    each other through a mimic, and a simulation turns that mimic into a constraint
    whose pads sweep forwards as they close and push an object out of the hand.

    The model carries fewer bodies than the URDF, and leaves out some the URDF has,
    the camera among them.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return str(
            Path(files("semantic_digital_twin")).parent.parent
            / "resources"
            / "mjcf"
            / "tiago_dual"
            / "tiago_dual.xml"
        )

    @classmethod
    def parse_description(cls) -> World:
        """
        Read the MuJoCo description, dropping the body that stands for the scene the
        robot was described in.

        MJCF describes every body inside a world body, so parsing the description leaves
        that world body above the robot's own root. A robot is attached to the rest of
        the world by its root, so a frame above that root would sit between the robot
        and the drive that moves it.

        :return: The world holding nothing but this robot.
        """
        world = MJCFParser(cls.get_ros_file_path()).parse()
        with world.modify_world():
            world.remove_kinematic_structure_entity(world.root)
        return world

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def _setup_collision_rules(self):
        """
        The description states which of its own bodies may touch, so only the rules
        about touching anything else are added.
        """
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

    def _setup_velocity_limits(self):
        self.tighten_dof_velocity_limits_of_1dof_connections(
            new_limits=defaultdict(lambda: 1.0)
        )
