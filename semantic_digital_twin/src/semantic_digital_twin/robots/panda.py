from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import Self, List

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasTwoFingers
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Finger,
    EndEffector,
)
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class PandaLeftFinger(Finger):
    """
    The Panda's fingers have no separate fingertip link, so root and tip are the same body.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        finger = robot_root._world.get_body_in_branch_by_name(
            robot_root, "/left_finger"
        )
        return cls(root=finger, tip=finger)


@dataclass(eq=False)
class PandaRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        finger = robot_root._world.get_body_in_branch_by_name(
            robot_root, "/right_finger"
        )
        return cls(root=finger, tip=finger)


@dataclass(eq=False)
class PandaGripper(EndEffector, HasTwoFingers[PandaLeftFinger, PandaRightFinger]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        finger_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(finger_joints, [0.04, 0.04])),
            state_type=GripperState.OPEN,
        )

        # CLOSE was previously [0.0, 0.0] -- fully closed, as if nothing is
        # between the fingers. But this demo's cubes are 4cm wide (half-width
        # 0.02m), so each finger can only physically travel to ~0.02 before
        # hitting the cube. Commanding 0.0 anyway means the controller keeps
        # pushing at whatever force it's allowed, indefinitely, for as long
        # as contact holds -- a sustained push against an impossible target,
        # not a fixed grip. If that push lands even slightly asymmetrically
        # between the two fingers (plausible given real contact timing),
        # that's a well-motivated explanation for the variable, sometimes-
        # large spin seen at grasp closure (a couple degrees some runs, 30-50
        # degrees others) -- it's not a fixed disturbance, it's sensitive to
        # exactly how the impossible-target push plays out each run.
        # UPDATE (this round): 0.018 was tuned to avoid the indefinite-push
        # spin described above, and it does that -- but it was tuned back
        # when AttachNode's kinematic welding was still doing the actual work
        # of "holding" the cube, so CLOSE's own contact force never had to be
        # strong enough to survive anything on its own. Now that AttachNode
        # is removed and the grasp is held purely by friction, this showed up
        # directly: CLOSE itself converges cleanly (fingers land right at
        # ~0.017), but the cube tumbles wildly during the very next step,
        # lift. 0.018 sits only ~2mm past the cube's actual surface (half-
        # width 0.02) -- MuJoCo's contact force comes from how far a body
        # penetrates the compliant contact model (via solimp/solref), so a
        # target that barely grazes the surface generates correspondingly
        # little normal force, and therefore (via friction = coefficient *
        # normal force) very little actual grip -- enough to look closed, not
        # enough to survive being lifted. Tightened to 0.014 (~6mm past the
        # surface) for genuinely more squeeze; the actuator's own forcerange
        # (+-10N, stacking_scene.xml's /actuator8) still caps how hard it can
        # actually push regardless of how far past the surface this target
        # asks for, so this shouldn't reintroduce the indefinite-max-force
        # spin an unreachable 0.0 target caused before -- but this exact
        # number is a first guess, not a validated one.
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(finger_joints, [0.014, 0.014])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        # Unlike the PR2 (see PR2RightGripper/PR2LeftGripper in pr2.py), which
        # ships a dedicated r_gripper_tool_frame/l_gripper_tool_frame link
        # separate from the palm, the Panda's "/hand" link sits at the wrist
        # mount, not the point where the fingers actually meet. Its MJCF
        # (stacking_scene.xml) previously only marked that pinch point with a
        # <site>, which the world parser doesn't expose as a referenceable
        # KinematicStructureEntity. We've since turned it into a proper (fixed,
        # jointless) <body name="/pinch_site"> in the MJCF, so it parses into a
        # real body here and can serve as tool_frame directly -- fixing the bug
        # where tool_frame=root made PickUpAction/GraspDescription plan grasps
        # relative to the wrist mount instead of the actual pinch point 0.1034m
        # further along local +Z, causing failed/colliding picks.
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "/hand"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "/pinch_site"
            ),
            # Rotates the tool frame's forward-facing axis onto the hand's local
            # +Z, which is where the fingers reach out to (the "pinch_site" in
            # the MJCF sits 0.1034m further along that same axis).
            front_facing_orientation=Quaternion(
                0, 0.7071067811865476, 0, 0.7071067811865476
            ),
        )


@dataclass(eq=False)
class PandaArm(Arm[PandaGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = self.active_connections
        # Franka's standard "ready" pose.
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    connections,
                    [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397],
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link0"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "link7"),
        )


@dataclass(eq=False)
class Panda(AbstractRobot, HasOneArm[PandaArm]):
    """
    The Franka Emika Panda arm, as used in the stacking demo. https://franka.de/
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError("We dont have the ROS Package yet")

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "link0"

    def _setup_collision_rules(self):
        # No SRDF-based self-collision matrix exists for the Panda yet (see
        # resources/collision_configs), so unlike Stretch/Tiago/HSRB this only
        # skips self-collisions instead of ignoring a curated adjacency list.
        #
        # Previously this method did nothing else -- Panda had NO explicit
        # external-collision-avoidance tuning of its own, unlike PR2 (see
        # pr2.py's _setup_collision_rules), which configures buffer_zone_distance
        # per body group. Every ExternalCollisionAvoidance task failing at once
        # for every link during a plain tabletop reach points to Panda running
        # on some generic framework-default buffer that's too large for a task
        # that legitimately needs the gripper within centimeters of a small
        # object -- normal for a pick, not something to avoid. Add Panda-scaled
        # rules following PR2's same pattern, but sized for a tabletop task
        # instead of a room-scale mobile robot: a modest default for the whole
        # robot, tighter for the arm links, tightest for the end effector
        # itself (the part that actually needs to get closest).
        end_effector_bodies = set(self.arm.end_effector.bodies_with_collision)
        arm_bodies = set(self.arm.bodies_with_collision) - end_effector_bodies
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.02,
                    violated_distance=0.0,
                    robot=self,
                    body_subset=arm_bodies,
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.01,
                    violated_distance=0.0,
                    robot=self,
                    body_subset=end_effector_bodies,
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.02, violated_distance=0.0, robot=self
                ),
            ]
        )

    def _setup_velocity_limits(self):
        # HISTORY: arm joints were previously reverted all the way to Franka's
        # near-max 2.175/2.61 rad/s because ParkArmsAction appeared to barely
        # converge at reduced speeds (joint2 target -0.7854, actual only
        # -0.0212, etc.) -- reverting to full speed was the test to confirm
        # park could converge at all. It's since been confirmed (across THREE
        # separate velocity settings -- 0.5/0.6, 1.2/1.4, and this file's
        # then-current 2.175/2.61 -- all producing the byte-for-byte identical
        # partial-convergence signature) that joint velocity was never actually
        # the cause of that convergence failure. The real cause, found and
        # fixed afterward: the demo's own main loop was calling a bare
        # settle()/time.sleep() between steps with nothing holding the arm's
        # position, letting it droop under gravity right after park reported
        # near-perfect convergence -- not a speed problem at all. That fix
        # (removing settle() from the loop) was never revisited against a
        # slower velocity limit afterward, so the arm has been running at
        # Franka's near-max speed ever since -- well past what a pick-and-
        # place demo needs, and the likely direct cause of the "moves too fast
        # during pickup" complaint (CartesianPose/CartesianPosition's own
        # reference_linear_velocity/reference_angular_velocity, and
        # JointPositionList's max_velocity, are confirmed from their own
        # source/docstrings to be used for QP normalization only, not a real
        # speed cap -- these DOF limits are the actual, always-enforced
        # ceiling). Lowering these now that the actual convergence blocker is
        # gone.
        # UPDATE (this round): confirmed via a real run's log -- ParkArmsAction
        # converged cleanly at 1.0/1.2 (every attempt but one converged on the
        # first try; the one exception converged on the second). That's
        # direct evidence velocity isn't the convergence risk it once looked
        # like, so lowering further for a genuinely slower motion as
        # requested. If convergence ever regresses at this speed, that's new
        # information worth its own diagnostic, not a reason to jump back to
        # 2.175/2.61.
        #
        # Finger joints: left at the slower 0.02 -- never implicated in either
        # the convergence finding above or the speed complaint (fingers only
        # need to travel ~2cm during CLOSE/OPEN either way).
        vel_limits = defaultdict(lambda: 0.5)
        vel_limits[self._world.get_connection_by_name("joint5")] = 0.6
        vel_limits[self._world.get_connection_by_name("joint6")] = 0.6
        vel_limits[self._world.get_connection_by_name("joint7")] = 0.6
        vel_limits[self._world.get_connection_by_name("/finger_joint1")] = 0.02
        vel_limits[self._world.get_connection_by_name("/finger_joint2")] = 0.02
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)