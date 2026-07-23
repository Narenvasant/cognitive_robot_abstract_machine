import faulthandler
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot, ExecutionEnvironment
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import sequential, execute_single
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion, MoveToolCenterPointMotion
from coraplex.view_manager import ViewManager

import mujoco
import numpy as np

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose, CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState as GiskardJointState
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.adapters.mjcf import MJCFParser
import semantic_digital_twin.adapters.multi_sim as multi_sim_module
from semantic_digital_twin.adapters.multi_sim import MujocoActuator, MujocoSim
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.world_entity import Actuator

# WORKAROUND: MujocoBuilder._end_build (semantic_digital_twin/adapters/multi_sim.py)
# auto-generates a "home" keyframe by iterating `self.world.bodies`, but the model
# was actually built (and MuJoCo's real internal qpos/DOF layout was determined) by
# iterating `world.bodies_topologically_sorted` instead -- a different order. That
# mismatch scrambles the keyframe's qpos (confirmed: a joint4 ctrl-range value,
# -0.0698, was found sitting in cube3's free-joint quaternion slot). If anything
# resets to this keyframe, free-jointed bodies (the cubes) can end up wildly
# misplaced, which matches the violent instability we've been seeing -- while the
# raw scene file (no keyframe applied) was completely stable for 15s on its own.
#
# Rather than edit the shared multi_sim.py, wrap _end_build here: let it do
# everything it normally does, then as a final corrective pass, overwrite the
# broken keyframe's qpos with the model's own correctly-computed default (qpos0,
# derived straight from each body/joint's own pos/ref attributes -- the same
# values already proven stable).
_original_mujoco_builder_end_build = multi_sim_module.MujocoBuilder._end_build


def _patched_end_build(self, file_path: str):
    _original_mujoco_builder_end_build(self, file_path)

    import xml.etree.ElementTree as ET

    model = mujoco.MjModel.from_xml_path(file_path)
    correct_qpos = model.qpos0

    tree = ET.parse(file_path)
    root = tree.getroot()
    fixed = False
    for key in root.iter("key"):
        if key.get("name") == "home":
            key.set("qpos", " ".join(map(str, correct_qpos.tolist())))
            fixed = True
    if fixed:
        tree.write(file_path, encoding="utf-8", xml_declaration=True)


multi_sim_module.MujocoBuilder._end_build = _patched_end_build

# WORKAROUND #3: MoveToolCenterPointMotion._motion_chart (coraplex/robot_plans/
# motions/gripper.py) builds its CartesianPose task with NO explicit threshold:
#   CartesianPose(root_link=root, tip_link=tip, goal_pose=self.target,
#                 name="MoveTCP", weight=DefaultWeights.WEIGHT_BELOW_CA)
# ReachMotion's own CartesianPose task, by contrast, explicitly sets
# threshold=0.005. That's a real, confirmed difference between the two source
# files (not a guess) -- and it lines up exactly with what we observed: reach
# stays clean (tight 0.005 threshold), lift tumbles (falls back to whatever
# CartesianPose's own class default is, evidently loose enough to report
# "done" well before the wrist has actually finished rotating into place).
# MoveToolCenterPointMotion's dataclass doesn't expose a threshold field to
# set from outside, so patch _motion_chart itself, adding threshold=0.005 to
# match ReachMotion -- otherwise identical to the original.
#
# WORKAROUND #3b: the ORIGINAL _motion_chart (and this patch, until now)
# accepts allow_gripper_collision as a dataclass field but never once reads
# self.allow_gripper_collision inside _motion_chart -- it's dead. Confirmed by
# diffing against MoveTCPWaypointsAlignedMotion in the same file, which DOES
# use it. Wired it through the same way, below.
#
# CORRECTION (base.py now seen): _only_allow_gripper_collision_rules is
# broader than assumed last round. Its actual body:
#   manipulator_bodies = ViewManager().get_end_effector_view(arm, self.robot).bodies_with_collision
#   return [UpdateTemporaryCollisionRules(temporary_rules=[
#       AllowCollisionBetweenGroups(self.world.bodies_with_collision, manipulator_bodies)
#   ])]
# self.world.bodies_with_collision is EVERY collidable body in the world, not
# just the object currently being grasped. So allow_gripper_collision=True
# doesn't scope collision relaxation to "gripper vs the held cube" -- it makes
# the gripper (hand+fingers specifically, not the rest of the arm) immune to
# ALL collisions, cube1/cube2/cube3 included. Wiring it through as originally
# planned would NOT have protected the bystanders during lift -- lift already
# passes allow_gripper_collision=True, and once wired that call would
# explicitly ALLOW the gripper to hit cube1, the opposite of what's needed.
# Left wired below anyway (it's what the framework provides, and matches how
# FollowToolCenterPointPathAction/MoveTCPWaypointsMotion already use it
# elsewhere for genuine "let the gripper brush things generally" cases) --
# but lift itself now uses the narrower path below instead, NOT this one.
#
# Narrow path: allow collision only between the manipulator and one specific
# body (the cube actually being carried), leaving every other body -- floor,
# table, cube1/2/3 -- fully protected by AvoidExternalCollisions. Callers opt
# in by setting an ad-hoc `_narrow_collision_target` attribute on the
# MoveToolCenterPointMotion instance before calling .perform() (see lift's
# call sites below) rather than allow_gripper_collision=True.
import coraplex.robot_plans.motions.gripper as gripper_module


def _narrow_collision_relaxation_nodes(relax_arm, targets):
    """
    Shared by every patched _motion_chart below (MoveToolCenterPointMotion,
    MoveJointsMotion, MoveGripperMotion). Builds the node(s) needed to let
    the robot touch exactly `targets` -- a single body, a list of bodies, or
    None/empty -- while AvoidExternalCollisions stays fully active against
    everything else. No-op (returns []) whenever collision_avoidance isn't
    actually on, since there's nothing to relax against.

    WIDENED this round: id()-matching diagnostics confirmed the relaxation
    mechanism itself was working correctly (right object, right target,
    right code path) -- CLOSE still tumbled ~70deg and place still failed
    identically regardless. That ruled out "not applying" and pointed to
    "not wide enough": this used to relax only the END EFFECTOR's own bodies
    (hand + fingers), but panda.py's _setup_collision_rules adds THREE
    separate AvoidExternalCollisions rules (whole-robot buffer=0.05,
    arm_bodies buffer=0.02, end_effector_bodies buffer=0.01) -- relaxing only
    the narrowest one left link6/link7 (the wrist, mechanically right next
    to any grasp point) still fighting to stay clear of the very cube the
    fingers are closing onto, a direct tug-of-war within the same rigid
    chain. Relaxing the WHOLE ROBOT against `targets` instead removes that
    conflict everywhere at once; it does not widen which bodies count as
    `targets` (still exactly the cube/surface passed in), so cube1/cube2/
    cube3/everything-else stays fully protected.

    :param relax_arm: unused now that relaxation covers the whole robot;
        kept for call-site compatibility.
    :param targets: a Body, an iterable of Body, or None.
    """
    if not GiskardExecutable.collision_avoidance or not targets:
        return []
    if not isinstance(targets, (list, tuple, set)):
        targets = [targets]
    targets = [t for t in targets if t is not None]
    if not targets:
        return []
    manipulator_bodies = context.robot.bodies_with_collision
    return [
        UpdateTemporaryCollisionRules(
            temporary_rules=[AllowCollisionBetweenGroups(targets, manipulator_bodies)]
        )
    ]


def _patched_move_tcp_motion_chart(self):
    tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
    root = (
        self.world.root
        if isinstance(self.robot, HasMobileBase)
        and self.robot.mobile_base.full_body_controlled
        else self.robot.root
    )
    if self.movement_type == gripper_module.MovementType.TRANSLATION:
        task = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=self.target.to_position(),
            name="MoveTCP",
            weight=DefaultWeights.WEIGHT_BELOW_CA,
            reference_velocity=CARTESIAN_LINEAR_VELOCITY,
        )
    else:
        task = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=self.target,
            name="MoveTCP",
            weight=DefaultWeights.WEIGHT_BELOW_CA,
            threshold=0.005,
            reference_linear_velocity=CARTESIAN_LINEAR_VELOCITY,
            reference_angular_velocity=CARTESIAN_ANGULAR_VELOCITY,
        )
    narrow_target = getattr(self, "_narrow_collision_target", None)
    # Last round's test (skipping the Parallel-wrap whenever collision_
    # avoidance was False) came back with lift's tilt unchanged -- rules out
    # the Parallel-wrapping-causes-premature-convergence hypothesis. Left the
    # guard in place below anyway since it's still correct (no reason to wrap
    # when there's nothing to relax), just noting it wasn't the cause.
    nodes = _narrow_collision_relaxation_nodes(self.arm, narrow_target)
    if nodes:
        nodes.append(task)
        return Parallel(nodes)
    if self.allow_gripper_collision and GiskardExecutable.collision_avoidance:
        nodes = self._only_allow_gripper_collision_rules(self.arm)
        nodes.append(task)
        return Parallel(nodes)
    return task


gripper_module.MoveToolCenterPointMotion._motion_chart = property(
    _patched_move_tcp_motion_chart
)

# WORKAROUND #9: with collision_avoidance now True (this round), cube0's
# gripper CLOSE tumbled ~66 degrees -- MoveGripperMotion._motion_chart (its
# ORIGINAL body, unpatched until now) just returns a bare JointPositionList
# with no collision handling at all. ExternalCollisionAvoidance now gets
# added to every motion chart automatically once collision_avoidance=True
# (per ExecutionEnvironment's own docstring) -- including this one -- so
# closing the fingers onto a cube that ISN'T attached yet (AttachNode runs
# AFTER close) is being fought by avoidance treating that cube as an
# obstacle. Same fix pattern as everywhere else: narrow relaxation scoped to
# the specific cube about to be grasped, set as an ad-hoc
# _narrow_collision_target on the CLOSE motion specifically (never on OPEN,
# which doesn't need to touch anything).
def _patched_move_gripper_motion_chart(self):
    arm_view = ViewManager().get_end_effector_view(self.gripper, self.robot)
    # FIX (this round): the new log_gripper_state diagnostic showed OPEN/
    # CLOSE consistently NOT reaching their commanded targets -- e.g. OPEN
    # (target 0.04) landing around 0.030 at pick time and only ~0.012-0.015
    # at place time, despite plenty of tick budget and a finger DOF velocity
    # limit slow enough that time was never the constraint. Root cause:
    # JointPositionList's default threshold=0.01 (joint_tasks.py) is the
    # "close enough, call it done" cutoff that ends the whole tick loop once
    # crossed -- but the gripper's entire OPEN/CLOSE stroke is only ~0.022m
    # (0.018 <-> 0.04), so a 0.01 threshold is ~45% of the ENTIRE stroke, not
    # a small tolerance. The motion was reporting "done" and exiting roughly
    # halfway there, which is exactly the "doesn't really open/close" look --
    # note this was never actually about AttachNode's kinematic reparenting
    # papering over a bad grasp (that mechanism is real too, see
    # _hold_arm_joints_in_mujoco's docstring, but it's not why this
    # specifically looked wrong). Tightened to 2mm, which is small relative
    # to the ~22mm stroke.
    # UPDATE (this round): CLOSE's target just got tightened (panda.py,
    # 0.018 -> 0.014) to generate real squeeze/grip force now that the grasp
    # is physical, not kinematically welded -- but that means CLOSE will now
    # genuinely be stopped short of 0.014 by real contact with the cube (that
    # IS the grip), and 0.002 is tight enough that "stopped a few mm short by
    # a real object" may never register as converged, unlike OPEN which has
    # nothing in its way. Loosen CLOSE's own threshold so a real, physical
    # stall against the cube still counts as done; keep OPEN at the tight
    # 2mm since nothing should ever block it from reaching 0.04 for real.
    task = JointPositionList(
        goal_state=arm_view.get_joint_state_by_type(self.motion),
        name=("OpenGripper" if self.motion == GripperState.OPEN else "CloseGripper"),
        threshold=0.002 if self.motion == GripperState.OPEN else 0.01,
    )
    narrow_target = getattr(self, "_narrow_collision_target", None)
    # DIAGNOSTIC (this round): confirm whether the ad-hoc attribute set on
    # the instance BEFORE .perform() is still visible HERE, at chart-build
    # time -- if id(self) differs from what was logged at the call site, or
    # narrow_target prints as MISSING, the object got reconstructed
    # somewhere in between (see BaseMotion.motion_chart's alternative-motion
    # reconstruction path) and every _narrow_collision_target fix so far has
    # been silently dropped, not ineffective.
    log(
        f"[diag] _patched_move_gripper_motion_chart: id(self)={id(self)} "
        f"motion={self.motion} narrow_target={narrow_target!r}"
    )
    nodes = _narrow_collision_relaxation_nodes(self.gripper, narrow_target)
    if nodes:
        nodes.append(task)
        return Parallel(nodes)
    return task


gripper_module.MoveGripperMotion._motion_chart = property(
    _patched_move_gripper_motion_chart
)

# WORKAROUND #4: use PickUpAction/PlaceAction directly (bundled, one call
# each) instead of manually reconstructing open/reach/close/attach/lift as
# separate calls. The manual reconstruction was only ever a diagnostic tool
# to get per-substep visibility while hunting for where tumbling happened --
# but doing that reintroduced exactly the isolated-call arm-drift problem we
# found and fixed several rounds ago (bundled charts hold the arm's position
# across all their internal nodes; separate standalone calls don't reliably).
# Now that the actual bug (lift_to_pose's reference frame becoming circularly
# defined once AttachNode reparents the cube -- see below) is found and fixed,
# there's no reason to keep the manual version around. Patch _action_plan to
# fold that one fix into the framework's own bundled sequence.
import coraplex.robot_plans.actions.core.pick_up as pick_up_module


def _patched_pickup_action_plan(self):
    # lift_to_pose is expressed relative to the cube's own frame -- freezing
    # it into world.root's frame now gives a fixed absolute target regardless
    # of how the cube ends up being held.
    _, _, lift_to_pose = self.grasp_description.grasp_pose_sequence(self.object_designator)
    lift_to_pose = self.world.transform(lift_to_pose, self.world.root)
    # allow_gripper_collision=True here would (per base.py's actual
    # _only_allow_gripper_collision_rules body) make the gripper immune to
    # EVERY body in the world, not just self.object_designator -- exactly
    # what would let it plow through a neighboring cube during lift instead
    # of protecting one. Use the narrow, single-body relaxation instead (see
    # _patched_move_tcp_motion_chart's _narrow_collision_target handling).
    lift_motion = MoveToolCenterPointMotion(lift_to_pose, self.arm)
    lift_motion._narrow_collision_target = self.object_designator
    close_motion = MoveGripperMotion(motion=GripperState.CLOSE, gripper=self.arm)
    close_motion._narrow_collision_target = self.object_designator
    # REMOVED (kept removed): AttachNode's kinematic FixedConnection welding
    # -- not because of physically_simulated_dofs (confirmed inert, see the
    # setup code above), but because the cube is a genuine free body once
    # nothing forces its pose, so it's held by real friction/contact instead.
    return sequential(
        children=[
            MoveGripperMotion(motion=GripperState.OPEN, gripper=self.arm),
            ReachAction(
                target_pose=self.object_designator.global_pose,
                object_designator=self.object_designator,
                arm=self.arm,
                grasp_description=self.grasp_description,
            ),
            close_motion,
            lift_motion,
        ],
    )


pick_up_module.PickUpAction._action_plan = property(_patched_pickup_action_plan)


# WORKAROUND #8: ReachAction's own _action_plan (pick_up.py) hard-codes
# allow_gripper_collision=False on BOTH its internal MoveToolCenterPointMotion
# calls -- including the final approach pose, which legitimately needs ~0
# clearance to whatever object it's reaching for (that's the whole point of a
# reach: grasp calls it targeting the not-yet-held cube, place calls it
# targeting the still-held cube during the placement approach). With
# collision_avoidance=False this was a no-op, but it's very likely the actual
# cause of the historical "every ExternalCollisionAvoidance task fails at
# once" symptom (see the comment further down) once avoidance is switched on
# -- the final approach pose being unable to relax collision against its own
# target is a direct structural conflict, not a tuning problem. Give the
# final approach the same narrow, single-body relaxation lift/park-with-
# object already have; leave the pre-approach pose fully protected (it's not
# supposed to be anywhere near anything yet).
#
# WORKAROUND #8b (this round): confirmed the above wasn't the whole story --
# place's own reverse-reach failed with EVERY ExternalCollisionAvoidance task
# failing for EVERY link, even with object_designator's relaxation in place.
# Reason: by the time place's reach runs, the cube is RIGIDLY ATTACHED (via
# pick's AttachNode) -- relaxing collision against the cube doesn't help,
# because the actual conflict is between the gripper+cube assembly and
# whatever SURFACE the cube is being lowered onto (the floor, for cube0's
# round trip; the stack_pad or the previously-placed cube, for cube1/2/3) --
# a different body than the held object, and nothing was relaxing collision
# against it. Added an optional `_narrow_collision_extra_targets` ad-hoc
# attribute (a list) that diagnostic_place sets before calling perform(), so
# the landing surface gets the same relaxation as the held cube.
def _patched_reach_action_plan(self):
    target_pre_pose, target_pose, _ = self.grasp_description.pose_sequence(
        self.target_pose, self.object_designator, reverse=self.reverse_reach_order
    )
    pre_motion = MoveToolCenterPointMotion(
        target_pre_pose, self.arm, allow_gripper_collision=False
    )
    final_motion = MoveToolCenterPointMotion(
        target_pose,
        self.arm,
        allow_gripper_collision=False,
        movement_type=gripper_module.MovementType.CARTESIAN,
    )
    narrow_targets = []
    if self.object_designator is not None:
        narrow_targets.append(self.object_designator)
    extra_targets = getattr(self, "_narrow_collision_extra_targets", None)
    # DIAGNOSTIC (this round): same purpose as the MoveGripperMotion one --
    # confirm id(self) and extra_targets actually arrive here as set at the
    # call site, rather than assuming the fix took effect just because it
    # compiles and runs without error.
    log(
        f"[diag] _patched_reach_action_plan: id(self)={id(self)} "
        f"object_designator={self.object_designator} extra_targets={extra_targets!r}"
    )
    narrow_targets.extend(extra_targets or [])
    if narrow_targets:
        final_motion._narrow_collision_target = narrow_targets
    return sequential(children=[pre_motion, final_motion])


pick_up_module.ReachAction._action_plan = property(_patched_reach_action_plan)


# WORKAROUND #7: robot_body.py (now seen) confirms MoveJointsMotion's
# _motion_chart is just `return JointPositionList(goal_state=...)` -- no
# allow_gripper_collision, no collision hook of any kind. And ParkArmsAction
# ._action_plan (also now seen) constructs that MoveJointsMotion internally,
# with no way to pass one in. So park_until_converged's "park-with-object"
# call currently has no way to tell the solver "cube0 is rigidly attached,
# don't fight that contact" -- the same narrow-relaxation gap flagged for
# collision_avoidance below. Patched the same way as MoveToolCenterPointMotion:
# an ad-hoc _narrow_collision_target attribute, set on the ParkArmsAction
# instance (which the patched _action_plan below forwards onto the
# MoveJointsMotion it builds internally, since MoveJointsMotion itself has no
# `arm` field to reuse _only_allow_gripper_collision_rules with -- built the
# manipulator_bodies lookup directly off the module-level `end_effector`
# instead, since this is a single-arm Panda).
# UNVALIDATED: same caveat as WORKAROUND #6 -- reasoned from source, not run.
import coraplex.robot_plans.actions.core.robot_body as robot_body_actions_module
import coraplex.robot_plans.motions.robot_body as robot_body_motions_module


def _patched_move_joints_motion_chart(self):
    dofs = [self.world.get_connection_by_name(name) for name in self.names]
    task = JointPositionList(
        goal_state=GiskardJointState.from_mapping(dict(zip(dofs, self.positions))),
        max_velocity=JOINT_PARK_VELOCITY,
    )
    narrow_target = getattr(self, "_narrow_collision_target", None)
    nodes = _narrow_collision_relaxation_nodes(arm, narrow_target)
    if nodes:
        nodes.append(task)
        return Parallel(nodes)
    return task


robot_body_motions_module.MoveJointsMotion._motion_chart = property(
    _patched_move_joints_motion_chart
)


def _patched_park_arms_action_plan(self):
    joint_names, joint_poses = self.get_joint_poses()
    motion = robot_body_motions_module.MoveJointsMotion(
        names=joint_names, positions=joint_poses
    )
    narrow_target = getattr(self, "_narrow_collision_target", None)
    if narrow_target is not None:
        motion._narrow_collision_target = narrow_target
    return execute_single(motion)


robot_body_actions_module.ParkArmsAction._action_plan = property(
    _patched_park_arms_action_plan
)


# NEW (this round): per-tick velocity trace, to answer "is the motion smooth"
# with actual data instead of screen-recording stills (which can't show
# jitter happening between two frames ~125ms apart anyway). Ros2Executor.tick
# (confirmed source, this round) is the one place every single control cycle
# passes through: compute_collisions() -> motion_statechart.tick() ->
# qp_controller.compute_command() -> world.apply_control_commands(). The
# commanded velocity itself (next_cmd) is a local variable inside that method
# with no hook to intercept, but world.state.velocities is confirmed (from
# Executor._set_velocity_acceleration_jerk_to_zero, `state.velocities[:] = 0`)
# to be a plain numpy array reflecting the CURRENT velocity right after a
# tick applies its command -- reading that from outside tick() avoids having
# to reimplement/guess at tick()'s internals.
# Gated by _VELOCITY_LOG_ARMED so this only records during one deliberately
# chosen motion (cube0's first ReachAction) instead of every tick of the
# entire run.
VELOCITY_LOG: List[float] = []
_VELOCITY_LOG_ARMED = [False]

_original_ros2_executor_tick = Ros2Executor.tick

_original_ros2_executor_tick = Ros2Executor.tick

# REVERTED (this round): the real-time pacing sleep added last round is taken
# back out. A/B evidence across rounds points at it, not the hold-actuator
# resync gap (already fixed separately): the round WITHOUT pacing (physically_
# simulated_dofs + gravcomp, original hold-actuator resync untouched) had park
# converge in a single attempt and pick fully succeed -- only place's reach
# failed. The round WITH pacing added on top of the since-corrected hold-
# actuator resync couldn't even complete a single park attempt, described as
# vibrating in place with no progress at all. Since park's own resync
# granularity was identical in both rounds, pacing is the one new variable
# that lines up with the regression.
# Now that executables.py's actual source has been seen (this round):
# _execute_simulation builds a plain, synchronous `while ...: executor.tick()`
# loop with no threading of its own visible at this level -- so the
# regression isn't an obvious Python-level race in that loop itself. The more
# likely explanation is that MujocoSim (physically_simulated_dofs,
# real_time_factor, sync_rate_hz -- semantic_digital_twin/adapters/multi_sim.py,
# still not seen) runs its own background physics stepping at some rate
# already, and pacing Giskard's tick loop on top of that at a possibly
# different assumed rate could under/over-sample the feedback loop rather
# than fix the earlier divergence. Not guessing further at this without that
# file -- reverting to the configuration that's actually confirmed to work
# furthest, and asking for multi_sim.py before touching pacing again.


def _patched_executor_tick(self):
    _original_ros2_executor_tick(self)
    if _VELOCITY_LOG_ARMED[0]:
        VELOCITY_LOG.append(float(np.linalg.norm(self.context.world.state.velocities)))


Ros2Executor.tick = _patched_executor_tick


def log_velocity_trace(label: str) -> None:
    """Print the recorded per-tick velocity trace and a simple smoothness
    check (largest single-tick jump in speed) so a jerky/oscillating
    trajectory shows up as data, not just an impression from a video."""
    if not VELOCITY_LOG:
        log(f"[diag] {label}: no velocity samples recorded")
        return
    trace = ", ".join(f"{v:.4f}" for v in VELOCITY_LOG)
    log(f"[diag] {label}: velocity trace ({len(VELOCITY_LOG)} ticks) = [{trace}]")
    jumps = [abs(b - a) for a, b in zip(VELOCITY_LOG, VELOCITY_LOG[1:])]
    if jumps:
        log(
            f"[diag] {label}: max tick-to-tick jump in |velocity| = {max(jumps):.4f} "
            f"(at tick {jumps.index(max(jumps))})"
        )


# WORKAROUND #2 (retired): manually decomposing PickUpAction/PlaceAction into
# separate perform() calls was earlier suspected of causing arm drift between
# calls. Root cause has since been confirmed: it was our OWN settle() call
# (a bare time.sleep() with nothing holding position, run after every step in
# the main loop) letting the arm droop under gravity -- not the decomposition
# itself. park_until_converged already proved repeated separate perform()
# calls with NO sleep in between are safe (park now converges exactly, every
# time). So we go back to explicit per-substep construction below (open,
# reach, close, attach, lift / reach, open, detach, retract) instead of the
# bundled PickUpAction/PlaceAction, purely to regain fine-grained visibility
# into exactly which sub-step the remaining grasp disturbance happens at --
# with no settle() anywhere in it this time.


time.sleep(8)  # Wait for the launch file to start

# DIAGNOSTIC: if the process dies from a native crash (segfault in the MuJoCo
# viewer or giskardpy's solver) rather than a clean Python exception, a normal
# try/except never runs. faulthandler.enable() with no `file=` argument writes
# straight to stderr, which shows up directly in the same console/Run window
# you're already looking at -- no file path to go hunting for.
faulthandler.enable()

# Backup text log, placed next to this script (not /tmp) and with its resolved
# path printed immediately, so there's no ambiguity about where to find it even
# if stdout itself gets lost on a hard crash.
_log_path = Path(__file__).resolve().parent / "panda_stacking_demo.log"
print(f"[diag] writing backup log to: {_log_path}", flush=True)
_log_file = open(_log_path, "w", buffering=1)


def log(msg: str) -> None:
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()


def pose_repr(pose) -> str:
    """Best-effort readable dump of a Pose/TransformationMatrix: try to_np() for
    real numbers, fall back to the object's own repr if that's not available."""
    try:
        return str(pose.to_np())
    except Exception:
        return repr(pose)


execition_mode = ExecutionType.SIMULATED

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

world = MJCFParser(
    "/home/nvasant/workspace/ros/src/manipulation_experiments/resources/generated/stacking_scene.xml"
).parse()
Panda.from_world(world)
publisher = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

# It is important to have the ros_node in the context for a real robot
context = Context(
    world=world,
    robot=world.get_semantic_annotations_by_type(Panda)[0],
    ros_node=node,
    evaluate_conditions=False,
)

# DIAGNOSTIC: Context.debug (dataclasses.py) sets the "coraplex" logger to DEBUG,
# but that alone won't print anything without a handler -- basicConfig gives it one.
# Requires ros_node to be set (it is, above), or Context.debug's setter raises.
logging.basicConfig(level=logging.DEBUG)
context.debug = True

# Panda is single-armed (HasOneArm). ViewManager.get_all_arm_views checks
# len(robot_view.get_arms()) == 1 first and short-circuits to that one arm
# regardless of which Arms value is passed -- so LEFT/RIGHT/BOTH are all
# equivalent here. Arms.LEFT is just picked for readability.
arm = Arms.LEFT
arm_view = context.robot.get_arms()[0]
end_effector = arm_view.end_effector

# REVERTED (this round): physically_simulated_dofs and gravity compensation,
# both taken back out, after finally seeing semantic_digital_twin/adapters/
# multi_sim.py's actual source. Two things it confirms:
# 1. MultiSim.__init__ (MujocoSim's base) only accepts world/headless/
#    step_size explicitly -- physically_simulated_dofs, real_time_factor, and
#    sync_rate_hz all silently fall into **kwargs -> a `config` dict handed to
#    MujocoSimulator (never seen). MujocoSynchronizer.sync_rate_hz -- the
#    thing that actually matters for the sim<->world sync -- keeps its own
#    hardcoded default of 30 regardless of what's passed to MujocoSim.
# 2. MujocoSynchronizer._on_state_change writes world.state straight into
#    mj_data.qpos for EVERY non-fixed connection, unconditionally -- there is
#    no per-DOF "physically simulated" concept anywhere in this synchronizer.
#    The arm has been kinematically driven this entire session, before and
#    after every change made here -- physically_simulated_dofs never did
#    anything real.
# What actually made pick work when AttachNode was removed: the CUBE (a free
# body with nothing forcing its pose anymore) got held by genuine friction
# from the kinematically-perfect gripper -- real progress, just unrelated to
# physically_simulated_dofs. AttachNode/DetachNode stay removed below; only
# the parts that were never real are reverted here.
#
# Gravity compensation specifically is now suspected actively HARMFUL, not
# just inert: MujocoBody.gravitation_compensation_factor is real (confirmed in
# MujocoBuilder._build_mujoco_body) and cancels gravity's pull on the arm's
# links. But _sim_to_world reads MuJoCo's own physics state (on a background
# thread, throttled to sync_rate_hz=30Hz) back into Giskard's belief for the
# arm too, alongside Giskard's own direct kinematic writes -- a standing race
# between the two. Gravity previously gave that background drift a small,
# predictable, gravity-shaped pull; cancelling it removes the one consistent
# force and leaves the arm's own real actuators (sitting at ctrl=0, never
# explicitly commanded by anything in this file) as the dominant, less
# predictable disturbance in between Giskard's corrections -- a better fit
# for "vibrating in place" than the real-time-pacing theory (which didn't
# fix it when reverted alone last round).


# WORKAROUND #5: coraplex/semantic_digital_twin does NOT automatically hold a
# robot's joints in MuJoCo -- confirmed via another working demo on this same
# framework (experiments.montessori.montessori_demo), whose own docstring
# states plainly: "the robot's controlled joints must already be held... or
# they sag/spin under gravity... and are left in that pose". That demo builds
# its own explicit MuJoCo PD position-hold actuator for every joint
# (_hold_controlled_joints_in_mujoco/_position_hold_actuator), called once
# right after the robot is spawned, before any MuJoCo simulation starts --
# without it, a joint only holds its position while Giskard is actively
# driving it at that exact instant; the rest of the time it's free to droop
# or drift under gravity/contacts. This is very likely the actual root cause
# behind most of the arm-drift symptoms chased this session (the isolated
# gripper-call drift, the settle()-caused droop, and possibly the place/
# retract disturbance too) -- different symptoms of the same missing piece,
# rather than separate bugs. Panda has no mobile base, so (unlike HSRB in
# that demo) there are no wheel joints to handle separately -- just the
# arm/finger joints.
# REVERTED from 1000/50: that change was based on a misreading of what this
# actuator actually holds. Since ctrl is never explicitly set (defaults to
# 0), the actuator's force reduces to -position_gain*length - velocity_gain*
# velocity -- a restoring force toward each joint's ZERO position, not
# toward wherever Giskard currently wants it held. At the original 100/10
# this pull was weak enough that Giskard's own active qpos-overwriting
# dominated it; at 1000/50 it became strong enough to visibly drag the arm
# back toward its all-zeros pre-park configuration the instant Giskard
# wasn't actively re-asserting those joints (e.g. during a finger-only
# gripper motion) -- confirmed directly: end_effector jumped ~28cm and
# rotated substantially during a plain gripper OPEN call, worse than any
# isolated-call drift we'd seen before. Back to 100/10.
# Confirmed from logged pre_pose/grasp_target/lift_pose prints: grasp_pose_sequence's
# own lift_pose is always exactly 0.05 higher (z) than its grasp_target -- used to
# rebuild a same-height lift target from the current end-effector pose instead of
# the frozen, pre-grasp one (see diagnostic_pick).
#
# RAISED from 0.05, then 0.18, now further (this round): the 3.4 fix (freezing
# lift's target from the CURRENT end-effector pose) removed lift's
# *orientation*-correction sweep, but a bystander (cube1) still gets clipped
# during park-with-object in some runs -- confirmed intermittent (doesn't
# reproduce every run under identical settings), and confirmed NOT fixable
# via collision_avoidance=True: executables.py showed ExternalCollisionAvoidance
# is added once, separately, outside of and unaware of any per-task
# UpdateTemporaryCollisionRules wrapping, and compiled once before any node
# (including a relaxation rule) ever ticks -- three rounds of scoping that
# rule (gripper-only, then whole-robot) changed nothing, consistent with the
# relaxation structurally arriving too late to matter. Confirming that for
# certain needs UpdateTemporaryCollisionRules/ExternalCollisionAvoidance's own
# source (giskardpy.motion_statechart.goals.collision_avoidance), which we
# don't have -- not chasing that path further blind. More vertical clearance
# before park-with-object's joint-space transit starts is the lever actually
# available: raising this further so the arm is well clear of any neighboring
# cube's height before that transit begins, reducing (not eliminating -- this
# is inherently a physical/contact-chance thing, not something fully
# controllable without collision avoidance) the odds of a repeat.
LIFT_HEIGHT_OFFSET = 0.25

ARM_ACTUATOR_TIME_CONSTANT = 0.1
ARM_ACTUATOR_POSITION_GAIN = 100.0
ARM_ACTUATOR_VELOCITY_GAIN = 10.0


# NEW (this round): every CartesianPose/CartesianPosition task in this file
# funnels through _patched_move_tcp_motion_chart below, and every
# JointPositionList (park, park-with-object) through
# _patched_move_joints_motion_chart -- two choke points, so setting a
# slower, uniform tempo here applies everywhere consistently instead of
# leaving park ~3x faster than every Cartesian reach (its unset default,
# 1.0 rad/s, vs Cartesian tasks' unset default of 0.2 m/s and 0.2 rad/s --
# not directly comparable units, but different enough in character that
# park visibly "snaps" relative to the slower, deliberate reach/place
# motions). Scaled all three down by the same ~4x factor so the whole
# sequence reads as one consistent, slow tempo rather than fast-then-slow.
#
# CORRECTION (this round): confirmed from cartesian_tasks.py's own source --
# CartesianPose.reference_linear_velocity/reference_angular_velocity are
# documented in-line as "used for normalization, for real limits use
# CartesianVelocityLimit"; JointPositionList.max_velocity (joint_tasks.py)
# is fed into add_equality_constraint the same way. Neither ever was a real
# speed cap -- these two constants only ever affected QP normalization, not
# actual commanded velocity. The true, always-enforced ceiling has been
# panda.py's own DOF velocity limits the whole time (previously left at
# Franka's near-max 2.175/2.61 rad/s -- now lowered there, see panda.py's
# _setup_velocity_limits). Kept below unchanged (still a reasonable
# normalization target, and harmless) -- see the TRIED AND REVERTED note
# right below for why an actual CartesianVelocityLimit hard cap was tried
# here and then taken back out.
CARTESIAN_LINEAR_VELOCITY = 0.05  # m/s, normalization only -- see correction above
CARTESIAN_ANGULAR_VELOCITY = 0.05  # rad/s, normalization only -- see correction above
JOINT_PARK_VELOCITY = 0.25  # rad/s, normalization only -- see correction above

# TRIED AND REVERTED (this round): added a genuine CartesianVelocityLimit
# hard cap here (CARTESIAN_MAX_LINEAR_VELOCITY/CARTESIAN_MAX_ANGULAR_VELOCITY,
# wired into _patched_move_tcp_motion_chart). A follow-up run's velocity
# trace (cube0's first ReachAction) showed real per-tick jitter -- isolated
# spikes/dips scattered throughout, not just the two-hump shape you'd expect
# from ReachAction's own two sub-motions. The most likely explanation: a
# second, independent hard constraint (the velocity limit) sitting in the
# same QP tick as the goal-position task is a textbook way to get tick-to-
# tick active-set toggling, i.e. exactly this kind of jitter. Reverted rather
# than layer another unvalidated guess on top -- panda.py's DOF velocity
# limits (a plain per-joint box constraint, not a competing task) are the
# safer lever for genuinely slower motion, since a box constraint on a
# decision variable can't conflict with another task's constraint the way
# two Cartesian-space tasks can. See panda.py's _setup_velocity_limits.


# REVERTED (this round): the removal of _hold_arm_joints_in_mujoco/
# _position_hold_actuator/_sync_position_holds last round is undone. The
# very next run showed ParkArmsAction itself failing to converge on the
# FIRST attempt (MotionDidNotFinish, joints stuck partway -- e.g. joint4 at
# -0.36 of a -2.36 target), with the arm visibly shaking in place rather
# than moving -- exactly the symptom this hold-actuator was originally built
# to prevent ("joints sag/spin under gravity... left in that pose" once
# Giskard isn't actively driving them). That's a worse, earlier failure than
# last round's (which at least converged through park/pick/lift before
# diverging at place) -- strong evidence removing this was the wrong half of
# last round's two changes, not the real-time pacing addition (which has
# direct evidence behind it from the print_positions divergence data).
# Restoring it while keeping real-time pacing, to isolate which change
# actually helps. UNVALIDATED whether keeping BOTH the real actuator (now
# genuinely driven, via physically_simulated_dofs) and this separate custom
# hold-spring on the same joints causes some other conflict -- but "arm
# can't hold still at all" is a strictly worse failure mode than any
# hold-actuator/real-actuator tension seen so far, so restoring this is the
# safer default until proven otherwise.
def _position_hold_actuator(position_gain: float, velocity_gain: float) -> MujocoActuator:
    """A MuJoCo actuator that holds its DOF at whatever position it had when
    the simulation started, resisting gravity/contacts with a PD law."""
    return MujocoActuator(
        dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
        dynamics_parameters=[ARM_ACTUATOR_TIME_CONSTANT] + [0.0] * 9,
        gain_type=mujoco.mjtGain.mjGAIN_FIXED,
        gain_parameters=[position_gain] + [0.0] * 9,
        bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
        bias_parameters=[0, -position_gain, -velocity_gain] + [0.0] * 7,
    )


def _hold_arm_joints_in_mujoco(robot) -> List[str]:
    """Add a position-hold actuator for every arm/finger joint, so MuJoCo
    itself keeps them in place independent of whether Giskard is actively
    commanding something at any given instant. Returns the joint/DOF names
    (they coincide for these single-DOF connections, per MJCFParser.parse_dof)
    so their actuators can be resolved and kept synced once MuJoCo compiles.

    Finger joints are excluded: _sync_position_holds only refreshes each hold
    spring's ctrl at boundaries BETWEEN motions, never during a motion's own
    internal tick loop, so pinning the finger's hold spring at its pre-call
    position for an entire OPEN/CLOSE call would fight the gripper's own real
    actuator for the whole duration.
    """
    held_joint_names = []
    seen_dof_names = set()
    with robot._world.modify_world():
        for dof in robot.degrees_of_freedom_with_hardware_interface:
            if "finger" in dof.name.name.lower():
                continue
            if dof.name.name in seen_dof_names:
                continue
            seen_dof_names.add(dof.name.name)
            actuator = Actuator(name=PrefixedName(f"hold_{dof.name.name}"))
            actuator.add_dof(dof=dof)
            actuator.simulator_additional_properties.append(
                _position_hold_actuator(
                    ARM_ACTUATOR_POSITION_GAIN, ARM_ACTUATOR_VELOCITY_GAIN
                )
            )
            robot._world.add_actuator(actuator=actuator)
            held_joint_names.append(dof.name.name)
    return held_joint_names


HELD_JOINT_NAMES = _hold_arm_joints_in_mujoco(context.robot)

cubes = [
    world.get_body_in_branch_by_name(world.root, "cube0"),
    world.get_body_in_branch_by_name(world.root, "cube1"),
    world.get_body_in_branch_by_name(world.root, "cube2"),
    world.get_body_in_branch_by_name(world.root, "cube3"),
]

# DIAGNOSTIC: quick eyeball check -- do the cube positions look like they're
# actually within reach of the arm, before we even try to plan anything?
log(f"[diag] arm root pose:\n{context.robot.get_arms()[0].root.global_pose}")
log(f"[diag] end_effector/tool_frame pose:\n{end_effector.tool_frame.global_pose}")
for name, cube in zip(["cube0", "cube1", "cube2", "cube3"], cubes):
    log(f"[diag] {name} pose:\n{cube.global_pose}")

# stack_pad sits at pos="0.5 0.15 0.001" with half-thickness 0.001 (top surface at
# z=0.002); each cube has half-size 0.02 (size="0.02 0.02 0.02" in stacking_scene.xml),
# so cube centers stack 0.04m apart starting 0.02m above the pad surface.
STACK_X = 0.5
STACK_Y = 0.15
PAD_TOP_Z = 0.002
CUBE_HALF_HEIGHT = 0.02
CUBE_HEIGHT = 2 * CUBE_HALF_HEIGHT

# cube0: picked up and placed right back at its own original position (from
# stacking_scene.xml: pos="0.265 -0.14 0.02") -- a round-trip sanity check,
# decoupled from the stacking math, before attempting the real 3-cube tower.
# CORRECTED (this round): this was still hardcoded to 0.34 -- cube0's OLD
# starting x before stacking_scene.xml's cube row was widened to 0.14m
# spacing (0.265/0.405/0.545/0.685). Left unchanged, cube0 was picked up from
# its new spot at 0.265 but placed back down at the stale 0.34 -- only 6.5cm
# from cube1's new position at 0.405, LESS clearance than the original 0.09m
# layout had, despite the whole point of widening being more clearance. This
# is very likely the direct cause of cube1 still getting knocked over during
# cube0's place sequence after the XML edit, confirmed in the log: cube1
# stays untouched through cube0's park-with-object, then progressively
# tumbles across place-ReachAction/gripper-OPEN/DetachNode/retract -- i.e.
# during the approach to this now-too-close target, not the transit.
CUBE0_ORIGINAL_POSE = Pose.from_xyz_rpy(
    0.265, -0.14, 0.02, yaw=0, reference_frame=world.root
)

# cube1, cube2, cube3 form the actual stack on stack_pad.
stacked_cubes = cubes[1:]
target_poses = [CUBE0_ORIGINAL_POSE] + [
    Pose.from_xyz_rpy(
        STACK_X,
        STACK_Y,
        PAD_TOP_Z + CUBE_HALF_HEIGHT + i * CUBE_HEIGHT,
        yaw=0,
        reference_frame=world.root,
    )
    for i in range(len(stacked_cubes))
]

FLOOR_BODY = world.get_body_by_name("floor")
STACK_PAD_BODY = world.get_body_by_name("stack_pad")

stack_actions = []
for i, (cube, target_pose) in enumerate(zip(cubes, target_poses)):
    # ROOT CAUSE FOUND (rotations.py + panda.py's front_facing_orientation,
    # both computed and confirmed numerically): with ApproachDirection.FRONT,
    # VerticalAlignment.TOP, rotate_gripper=False (the previous default here),
    # the resulting grasp orientation puts:
    #   - reach axis            -> world -Z  (correct: straight down, as intended for a TOP grasp)
    #   - finger-spread axis    -> world +Y  (perpendicular to the cube row)
    #   - hand-width axis       -> world -X  (ALONG the cube row -- cube0/1/2/3
    #                                         differ only in X, all at y=-0.14)
    # The Franka hand's own housing/knuckle width (the real mesh, not just the
    # finger travel) is close to the full 9cm cube-to-cube spacing -- so this
    # orientation points the gripper's WIDEST dimension directly down the row
    # toward whichever neighbor is next, independent of timing, collision
    # wiring, or lift height. This is the actual "approach isn't proper, so it
    # sweeps" mechanism. rotate_gripper=True swaps this: fingers -> world +X
    # (spreading along the row, which is fine -- a single cube is only 4cm
    # wide) and hand-width -> world +Y (perpendicular, into open space, away
    # from every other cube). Confirmed by direct quaternion computation, not
    # guessed. This one line is likely more impactful than the lift-height/
    # collision-scoping fixes combined, since it addresses actual grasp
    # geometry rather than mitigating around it.
    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.TOP, end_effector, rotate_gripper=True
    )
    # WORKAROUND #8b's landing_surface: whatever body the cube actually rests
    # against once placed -- needed so place's reverse-reach can relax
    # collision against it (see diagnostic_place). cube0 returns to the bare
    # floor; cube1 is the first one set down on stack_pad; cube2/cube3 each
    # land on the cube stacked immediately before them, not the pad itself.
    if i == 0:
        landing_surface = FLOOR_BODY
    elif i == 1:
        landing_surface = STACK_PAD_BODY
    else:
        landing_surface = cubes[i - 1]
    stack_actions.append((f"cube{i}", cube, grasp_description, target_pose, landing_surface))


SETTLE_SECONDS = 0.5


def settle(seconds: float = SETTLE_SECONDS) -> None:
    """Let MuJoCo's physics settle a moment after a motion completes, before
    reading/printing poses or starting the next bundled action."""
    time.sleep(seconds)


def log_poses(label: str, cube) -> None:
    """
    Prints the cube's pose AND the end-effector's pose together. If the cube
    ends up somewhere wild, this tells us whether the arm is right there too
    (a genuine collision/carry) or whether the arm looks completely normal
    while the cube is elsewhere (a numerical instability unrelated to the
    arm's actual location).
    """
    log(f"[diag] {label}: cube pose =\n{pose_repr(cube.global_pose)}")
    log(f"[diag] {label}: end_effector pose =\n{pose_repr(end_effector.tool_frame.global_pose)}")


def log_other_cube_poses(label: str, current_cube) -> None:
    """
    Prints every OTHER cube's pose (not the one currently being picked/placed),
    so we can see exactly which step during one cube's own sequence is what
    bumps a NEIGHBORING cube out of its resting pose -- rather than only
    finding out at the start of that neighbor's own turn, by which point it's
    too late to tell which earlier step actually caused it.
    """
    for name, other_cube in zip(["cube0", "cube1", "cube2", "cube3"], cubes):
        if other_cube is current_cube:
            continue
        log(f"[diag] {label}: {name} pose (bystander) =\n{pose_repr(other_cube.global_pose)}")


# The intended arm_park target from panda.py's PandaArm.setup_joint_states, in
# joint1..joint7 order -- used below to check whether ParkArmsAction is
# actually reaching this configuration, or only getting partway there.
ARM_PARK_TARGET = [
    0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397
]
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def log_joint_angles(label: str) -> None:
    """
    Prints each arm joint's ACTUAL current position next to its intended
    arm_park target, so we can tell whether ParkArmsAction is genuinely
    converging to that configuration or only getting partway there (which
    would leave the arm in an unexpected starting pose for whatever motion
    comes next, regardless of how that next motion's own convergence looks).
    """
    for name, target in zip(JOINT_NAMES, ARM_PARK_TARGET):
        connection = world.get_connection_by_name(name)
        log(f"[diag] {label}: {name} actual={connection.position:.4f} target={target:.4f}")


# NEW (this round): nothing in this file has ever logged the fingers'
# ACTUAL positions -- log_joint_angles only covers the 7 arm joints. Because
# AttachNode's FixedConnection reparenting makes cube.global_pose a pure
# kinematic projection off the gripper once attached (see §2.5-equivalent
# comment on _hold_arm_joints_in_mujoco below: "a gripper that visually looks
# open can still 'hold' a cube perfectly via this mechanism"), a cube that
# tracks the gripper correctly in every log_poses print does NOT by itself
# prove the fingers ever actually closed on it -- only that AttachNode ran.
# This is exactly the gap needed to check "does the gripper open/close
# properly at pick time" directly rather than inferring it from the cube's
# pose, which the mechanism above can make misleading.
GRIPPER_JOINT_NAMES = ["/finger_joint1", "/finger_joint2"]
# Approximate open/close targets from panda.py's PandaGripper.setup_joint_states
# (gripper_open=[0.04, 0.04], gripper_close=[0.018, 0.018]), printed alongside
# the actual value purely for a quick by-eye sanity check in the log --
# not used in any control decision here.
GRIPPER_OPEN_TARGET = 0.04
GRIPPER_CLOSE_TARGET = 0.014


def log_gripper_state(label: str) -> None:
    """
    Prints each finger joint's ACTUAL current position next to the
    open/close target it's supposed to be at, so a gripper that's failing to
    reach (or overshooting) its commanded position shows up as data instead
    of only being guessable from how the cube's pose looks afterward.
    """
    for name in GRIPPER_JOINT_NAMES:
        connection = world.get_connection_by_name(name)
        log(
            f"[diag] {label}: {name} actual={connection.position:.4f} "
            f"(open_target={GRIPPER_OPEN_TARGET:.4f}, close_target={GRIPPER_CLOSE_TARGET:.4f})"
        )


PARK_TOLERANCE = 0.05
PARK_MAX_RETRIES = 20


def park_until_converged(held_body=None) -> None:
    """
    A single ParkArmsAction call has repeatedly only made partial progress
    toward arm_park (confirmed via joint-angle diagnostics, independent of
    velocity limits -- reverting all the way to original speed didn't change
    how far it got, ruling out speed as the cause). Whatever's limiting a
    single call's progress, calling it again should make further progress
    from wherever it left off. ParkArmsAction has no side effects (unlike
    PickUpAction/PlaceAction with their attach/detach/gripper actions), so
    retrying it is safe. Stops once every joint is within PARK_TOLERANCE of
    its target, or after PARK_MAX_RETRIES calls.

    :param held_body: pass the currently-attached cube during "park-with-
        object" (see WORKAROUND #7) so the narrow collision relaxation can be
        applied to THIS park call specifically -- leave None for the bare
        "park" step before anything is picked up, where no relaxation is
        needed at all.
    """
    for attempt in range(PARK_MAX_RETRIES):
        park_action = ParkArmsAction(Arms.BOTH)
        if held_body is not None:
            park_action._narrow_collision_target = held_body
        sequential([park_action], context=context).perform()
        # RESTORED (this round): re-anchor the hold actuators to wherever
        # this attempt just left the joints, before the next attempt's own
        # perform() call gives the hold spring a chance to pull toward its
        # stale, frozen-at-startup ctrl value instead.
        _sync_position_holds()
        errors = [
            abs(world.get_connection_by_name(name).position - target)
            for name, target in zip(JOINT_NAMES, ARM_PARK_TARGET)
        ]
        log(f"[diag] park_until_converged: attempt {attempt + 1}, max joint error = {max(errors):.4f}")
        if max(errors) < PARK_TOLERANCE:
            log(f"[diag] park_until_converged: converged after {attempt + 1} attempt(s)")
            return
    log(f"[diag] park_until_converged: did NOT converge within {PARK_MAX_RETRIES} attempts")


def diagnostic_pick(cube_label: str, cube, grasp_description: GraspDescription) -> None:
    """
    TEMPORARY diagnostic: decomposes PickUpAction into the same five separate
    calls as its own patched _action_plan above -- open, reach, close, attach,
    lift -- with a pose print (including every OTHER cube, via
    log_other_cube_poses) after each, to find exactly which sub-step is what
    physically bumps a neighboring cube. Confirmed via log_other_cube_poses
    that cube1 stays clean through park but is already disturbed by the time
    "pick" (currently bundled) finishes -- this narrows down which part of
    pick specifically.
    """
    sequential(
        [MoveGripperMotion(motion=GripperState.OPEN, gripper=arm)], context=context
    ).perform()
    log_poses(f"{cube_label} after gripper OPEN", cube)
    log_other_cube_poses(f"{cube_label} after gripper OPEN", cube)
    log_gripper_state(f"{cube_label} after gripper OPEN")

    if cube_label == "cube0":
        _VELOCITY_LOG_ARMED[0] = True
    sequential(
        [
            ReachAction(
                target_pose=cube.global_pose,
                object_designator=cube,
                arm=arm,
                grasp_description=grasp_description,
            )
        ],
        context=context,
    ).perform()
    if cube_label == "cube0":
        _VELOCITY_LOG_ARMED[0] = False
        log_velocity_trace(f"{cube_label} ReachAction")
    log_poses(f"{cube_label} after ReachAction", cube)
    log_other_cube_poses(f"{cube_label} after ReachAction", cube)

    # REVERTED: the arm-hold wrapper on CLOSE reproduced the exact same
    # regression as when first tried, many rounds ago -- end_effector stays
    # perfectly still (confirming the hold works) but the cube itself still
    # spins substantially (~50 degrees this run), since it's held only by
    # friction (not yet attached) and any correction on the arm can still
    # transmit through the finger-cube contact. The CLOSE-target fix (0.018)
    # doesn't prevent this -- it's a different mechanism than an unreachable
    # target. Back to a plain, unwrapped CLOSE call.
    #
    # NEW (this round, collision_avoidance=True): CLOSE tumbled ~66 degrees --
    # ExternalCollisionAvoidance now applies to this motion too and was
    # fighting the fingers closing onto a cube that isn't attached yet. Narrow
    # relaxation scoped to `cube` (WORKAROUND #9) fixes the same way as
    # everywhere else.
    close_motion = MoveGripperMotion(motion=GripperState.CLOSE, gripper=arm)
    close_motion._narrow_collision_target = cube
    log(f"[diag] {cube_label}: set _narrow_collision_target on close_motion, id={id(close_motion)}")
    sequential([close_motion], context=context).perform()
    log_poses(f"{cube_label} after gripper CLOSE", cube)
    log_other_cube_poses(f"{cube_label} after gripper CLOSE", cube)
    log_gripper_state(f"{cube_label} after gripper CLOSE")

    # REMOVED (kept removed): AttachNode's kinematic FixedConnection welding.
    # CORRECTED reasoning (this round, after seeing multi_sim.py's actual
    # source): this was originally justified by physically_simulated_dofs
    # supposedly making the arm physically driven -- confirmed false, that
    # kwarg does nothing real in this codebase. The actual reason this still
    # works: without AttachNode's kinematic override, the cube is a genuine
    # free body (Connection6DoF) with nothing forcing its pose, so it's held
    # purely by real friction/contact between the fingers and the cube once
    # CLOSE grips it -- confirmed by the earlier log showing the cube
    # tracking correctly through lift. Keeping this removed on that basis.

    # Confirmed the actual mechanism behind lift's disturbance: lift_to_pose's
    # orientation was frozen back when grasp_pose_sequence was first computed
    # (before any grasping happened) -- clean, near-identity. But the wrist
    # has since deflected during CLOSE's contact reaction (confirmed:
    # end_effector orientation substantially rotated right after AttachNode).
    # So lift wasn't just a small vertical move -- it was also correcting
    # that wrist misalignment back to the frozen orientation, and THAT
    # corrective rotation is large enough to sweep the arm's body into a
    # neighboring cube. Building the lift target from the CURRENT
    # end-effector pose (whatever orientation it actually has right now,
    # deflection included) plus only a vertical offset means lift never has
    # any orientation correction to make at all -- just the intended 5cm
    # translation.
    current_pose = world.transform(end_effector.tool_frame.global_pose, world.root)
    lift_to_pose = HomogeneousTransformationMatrix.from_point_rotation_matrix(
        current_pose.to_position() + Vector3(0, 0, LIFT_HEIGHT_OFFSET, reference_frame=world.root),
        current_pose.to_rotation_matrix(),
    )

    # allow_gripper_collision=True here would allow the gripper to collide
    # with EVERY body in the world (see base.py's actual
    # _only_allow_gripper_collision_rules, which scopes to self.world.
    # bodies_with_collision, not just the held cube) -- exactly wrong for
    # protecting cube1/cube2/cube3 during this step. Use the narrow,
    # single-body relaxation instead, scoped to just the cube being lifted.
    lift_motion = MoveToolCenterPointMotion(lift_to_pose, arm)
    lift_motion._narrow_collision_target = cube
    sequential([lift_motion], context=context).perform()
    log_poses(f"{cube_label} after lift", cube)
    log_other_cube_poses(f"{cube_label} after lift", cube)


def diagnostic_place(
    cube_label: str,
    cube,
    grasp_description: GraspDescription,
    target_pose: Pose,
    landing_surface=None,
) -> None:
    """
    TEMPORARY diagnostic: decomposes PlaceAction into the same four separate
    calls as its own _action_plan (placing.py) -- reach(reverse), gripper
    OPEN, DetachNode, retract -- with a pose print after each, to find exactly
    where place's ~90-degree disturbance happens. This was unsafe to do
    before the MuJoCo position-hold actuator fix (isolated calls let the arm
    drift), but should be safe now that MuJoCo itself holds the joints
    regardless of whether Giskard is actively driving them at any instant.

    :param landing_surface: the body the held cube is being lowered onto
        (floor / stack_pad / the previously-placed cube) -- with
        collision_avoidance on, the reverse-reach approach needs this
        relaxed too, not just the held cube itself (WORKAROUND #8b).
    """
    target_pre_pose, _, retract_pose = grasp_description.pose_sequence(
        target_pose, cube, reverse=True
    )

    # FOLDED IN (this round): grasp_description.pose_sequence's own FINAL
    # target uses a THEORETICAL grasp offset -- purely a function of
    # approach_direction/vertical_alignment, with no knowledge of the REAL
    # offset AttachNode actually captured at CLOSE time. That gap was enough
    # to land cube0's round-trip ~2.4cm BELOW the floor. Fixed last round by
    # reaching there anyway and then adding a SEPARATE corrective move
    # afterward -- which worked (placement is now accurate to ~1mm), but
    # looked like a visible double-lurch (reach, then an extra jerk to fix
    # itself). target_pre_pose (the backed-off approach point, not at
    # touching height) doesn't have this problem -- a few mm/degrees of
    # theoretical-vs-real error there doesn't cause interpenetration, so it's
    # still safe to use as-is. Only the FINAL, touching-height target is
    # replaced with one computed from the REAL, currently-observed gripper-
    # to-cube offset (constant since AttachNode, so this can be computed any
    # time before opening the gripper) -- then both moves run as ONE combined
    # motion, back to ReachAction's original two-step shape instead of three.
    gripper_pose_now = world.transform(
        end_effector.tool_frame.global_pose, world.root
    ).to_homogeneous_matrix()
    cube_pose_now = world.transform(cube.global_pose, world.root).to_homogeneous_matrix()
    gripper_T_cube = gripper_pose_now.inverse() @ cube_pose_now
    target_pose_world = world.transform(target_pose, world.root).to_homogeneous_matrix()
    corrected_transform = target_pose_world @ gripper_T_cube.inverse()
    corrected_gripper_pose = HomogeneousTransformationMatrix.from_point_rotation_matrix(
        corrected_transform.to_position(), corrected_transform.to_rotation_matrix()
    )

    pre_motion = MoveToolCenterPointMotion(target_pre_pose, arm, allow_gripper_collision=False)
    final_motion = MoveToolCenterPointMotion(corrected_gripper_pose, arm)
    final_motion._narrow_collision_target = (
        [cube] + ([landing_surface] if landing_surface is not None else [])
    )
    log(
        f"[diag] {cube_label}: place final target corrected from real gripper-cube "
        f"offset, id(final_motion)={id(final_motion)}, landing_surface={landing_surface}"
    )
    sequential([pre_motion, final_motion], context=context).perform()
    log_poses(f"{cube_label} after place-ReachAction", cube)
    log_other_cube_poses(f"{cube_label} after place-ReachAction", cube)

    # REVERTED (this round): un-bundling OPEN from hold_motion made things
    # WORSE, not better -- place-gripper-OPEN dropped to ~0.0095-0.0128
    # (target 0.04) from the prior round's ~0.0175-0.0204, across all four
    # cubes. That falsifies the "hold_motion's instant convergence truncates
    # OPEN" theory from last round -- noting that plainly rather than
    # layering a third guess on an already-wrong one.
    #
    # NEW theory, based on what's actually constant across every failed
    # attempt regardless of bundling: place's OPEN always runs while the
    # cube is STILL ATTACHED. Pick's OPEN (which reliably reaches ~0.038)
    # never has anything between the fingers. AttachNode's FixedConnection
    # (confirmed architecture, see _hold_arm_joints_in_mujoco's docstring)
    # makes the cube's pose a pure kinematic slave to the gripper's
    # tool_frame -- not a real physics body. As the fingers try to open,
    # they're pushing against a cube MuJoCo's solver can never actually move
    # out of the way (it snaps back to its fixed offset every tick
    # regardless of contact force) -- the fingers' own actuator is weak
    # (10N, from stacking_scene.xml's /actuator8), so it plausibly stalls
    # against that immovable obstacle instead of reaching 0.04, independent
    # of how OPEN itself is sequenced -- which matches both this round's and
    # last round's data equally well.
    # FIX: detach BEFORE opening, not after -- so by the time the fingers
    # try to open, the cube is already a free, physics-driven body resting
    # on landing_surface (it's already at the correct target pose, per
    # place-ReachAction's corrected final target above) instead of a
    # kinematic obstacle wedged between the fingers. UNVALIDATED -- next
    # concrete thing to check is whether place-gripper-OPEN's finger values
    # finally approach ~0.04 like pick's do.
    # REMOVED (kept removed): both AttachNode (diagnostic_pick) and DetachNode
    # (here) stay gone -- see the corrected reasoning in diagnostic_pick
    # (nothing to do with physically_simulated_dofs, which is inert; the cube
    # was simply never kinematically reparented in the first place, so there's
    # nothing to detach). Release is purely physical: once the fingers open
    # enough, friction can no longer hold the cube against gravity and it
    # settles onto landing_surface on its own.
    current_pose = world.transform(end_effector.tool_frame.global_pose, world.root)
    # Same reasoning as WORKAROUND #8b -- at this exact moment the cube is
    # resting against landing_surface (now as a free body, detached above),
    # so opening the gripper and holding this pose needs the same relaxation
    # as the reach that brought it here, or avoidance will fight release/
    # hold the same way it fought CLOSE and the reverse-reach.
    place_open_targets = [cube] + ([landing_surface] if landing_surface is not None else [])
    open_motion = MoveGripperMotion(motion=GripperState.OPEN, gripper=arm)
    open_motion._narrow_collision_target = place_open_targets
    sequential([open_motion], context=context).perform()
    hold_motion = MoveToolCenterPointMotion(current_pose, arm)
    hold_motion._narrow_collision_target = place_open_targets
    sequential([hold_motion], context=context).perform()
    log_poses(f"{cube_label} after place-gripper-OPEN", cube)
    log_other_cube_poses(f"{cube_label} after place-gripper-OPEN", cube)
    log_gripper_state(f"{cube_label} after place-gripper-OPEN")

    sequential([MoveToolCenterPointMotion(retract_pose, arm)], context=context).perform()
    log_poses(f"{cube_label} after retract", cube)
    log_other_cube_poses(f"{cube_label} after retract", cube)



# REMOVED: SpatialTypePublisher/SpatialTypeVisualization here (visualizing
# end_effector.tool_frame.global_pose in RViz) is exactly what the segfault
# trace implicates. It subscribes to world.notify_state_change(), which fires
# from BOTH the physics-sim background thread (multi_sim.py's _sim_to_world)
# AND the main thread's Giskard tick loop (plans/executables.py) once real
# motion starts. Both call the same to_quaternion() -> from_rotation_matrix()
# -> krrood.symbolic_math.substitute() -> casadi.substitute() path concurrently,
# and casadi's substitution isn't safe to call from two threads at once -- that's
# the actual segfault, unrelated to PickUpAction/GraspDescription/MoveGripperMotion.
log("Perform Plan")

multi_sim = MujocoSim(
    world=world,
    headless=False,
    step_size=0.0001,
)
time_start = time.time()

tool_frame = end_effector.tool_frame


def print_positions() -> None:
    """
    Prints the tool_frame's and cube0's position as seen by the world model
    (Giskard's kinematic belief) side by side with MuJoCo's own live simulated
    position, so a divergence between "where Giskard thinks it is" and "where
    it actually, physically is" is visible directly -- exactly the kind of
    divergence physically_simulated_dofs can introduce if real_time_pacing
    isn't also active (see the note on real_time_pacing elsewhere in this
    file). Ported from the linked commit.
    UNVALIDATED: world.compute_forward_kinematics and
    multi_sim.simulator.get_body_position are used as given in that commit --
    neither has been directly confirmed to exist in this installed version,
    so this is wrapped defensively; a failure here prints a warning once
    instead of crashing the rest of the run.
    """
    try:
        tool_frame_kinematic = np.array(
            world.compute_forward_kinematics(world.root, tool_frame).to_position().evaluate()[:3],
            dtype=float,
        )
        box_kinematic = np.array(
            world.compute_forward_kinematics(world.root, cubes[0]).to_position().evaluate()[:3],
            dtype=float,
        )
        tool_frame_mujoco = np.array(
            multi_sim.simulator.get_body_position(tool_frame.name.name).result[:3], dtype=float
        )
        box_mujoco = np.array(
            multi_sim.simulator.get_body_position(cubes[0].name.name).result[:3], dtype=float
        )
        log(
            f"[diag] tool_frame: kinematic={tool_frame_kinematic} mujoco={tool_frame_mujoco} | "
            f"cube0: kinematic={box_kinematic} mujoco={box_mujoco}"
        )
    except Exception as exc:
        log(f"[diag] print_positions unavailable in this framework version: {exc!r}")

# REVERTED (this round): restoring HOLD_ACTUATOR_IDS resolution and
# _sync_position_holds along with the rest of the hold-actuator infra above.
HOLD_ACTUATOR_IDS: Dict[str, int] = {}
for _joint_name in HELD_JOINT_NAMES:
    _actuator_id = mujoco.mj_name2id(
        multi_sim.simulator._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hold_{_joint_name}"
    )
    if _actuator_id == -1:
        log(f"[diag] WARNING: hold actuator for {_joint_name} not found in compiled model")
        continue
    HOLD_ACTUATOR_IDS[_joint_name] = _actuator_id


def _sync_position_holds() -> None:
    """
    Write each hold actuator's ctrl to that joint's CURRENT position, so the
    PD law in _position_hold_actuator resists drifting away from wherever the
    joint actually is right now, instead of always pulling toward qpos=0.
    Call this right after anything that leaves the joints briefly
    un-commanded by Giskard -- most importantly park_until_converged's own
    retry loop, and the boundary between each named step in the main loop.
    """
    mj_data = multi_sim.simulator._mj_data
    for joint_name, actuator_id in HOLD_ACTUATOR_IDS.items():
        mj_data.ctrl[actuator_id] = world.get_connection_by_name(joint_name).position


multi_sim.start_simulation()
_sync_position_holds()

# Let physics settle before reading any cube pose or computing a grasp target --
# if the cubes are still drifting/settling the instant start_simulation() returns,
# every downstream pose (grasp target, pre_pose, lift_pose) is computed against a
# moving target, and Giskard chasing a target that keeps moving under it is a
# very plausible way to get MotionDidNotFinish regardless of how correct the
# grasp math is.
log("[diag] letting physics settle for 3s before reading cube poses...")
time.sleep(3.0)
print_positions()

# collision_avoidance=False (reverted): enabling it exposed a harder problem --
# reaching grasp height means getting the gripper within ~2cm of the floor (the
# fingertips end up near the same height as the pinch point, which sits right
# at cube height), and no buffer size seems to reconcile "stay away from
# everything" with "get this close to the floor to grasp a tabletop object" --
# every attempt failed with the exact same signature (every single link's
# ExternalCollisionAvoidance task failing at once), even after tightening the
# buffers in panda.py. Properly fixing that would mean excluding the floor
# specifically as an expected-proximity surface, which needs source we don't
# have (AvoidExternalCollisions's own implementation). Reverting so the reach
# can converge at all -- the actual fixes for the original violence (reduced
# joint velocity limits, softened cube contact solref/solimp, capped gripper
# actuator forcerange) have never been tested without collision_avoidance
# fighting against them in the process.
#
# TRIED True across three rounds (this file's git-equivalent history, kept
# here since it's the real record): rotate_gripper=True fixed the static
# grasp-clearance problem, but a SEPARATE transit-sweep problem remained
# under collision_avoidance=False -- cube1 getting knocked from a clean pose
# into a ~95-degree tip during cube0's park-with-object/retract in one run
# (not every run -- did NOT reproduce in another run under the identical
# settings, meaning it's intermittent contact-dynamics variance, not
# deterministic). Flipping collision_avoidance=True was meant to fix that
# properly, with narrow relaxation added for every step that legitimately
# needs contact with a specific body: lift, park-with-object (WORKAROUND #7),
# ReachAction's final approach (WORKAROUND #8), MoveGripperMotion's CLOSE
# (WORKAROUND #9), and place's landing surface (WORKAROUND #8b).
#
# REVERTED back to False. Three consecutive rounds with collision_avoidance=
# True all produced the SAME failure: CLOSE tumbles the cube 66-84 degrees
# (worsening slightly, not improving, across attempts), and place crashes
# with the byte-for-byte IDENTICAL "every ExternalCollisionAvoidance task
# fails" node list every time, regardless of two different scoping fixes:
#   1. Narrow relaxation scoped to just the end effector's own bodies -- no
#      change.
#   2. Widened to the WHOLE ROBOT's collidable bodies -- still no change
#      (this round).
# id()-matching diagnostics (added between rounds 2 and 3) directly confirmed
# the relaxation mechanism was reaching the right object with the right
# scope both times -- ruling out "the fix isn't applying." Since widening
# from narrowest to broadest possible scope didn't move the outcome at all,
# this isn't a collision-avoidance-vs-target conflict -- something else about
# collision_avoidance=True is destabilizing these two specific motions, and
# chasing it further needs source this session doesn't have (coraplex/plans/
# executables.py's actual solver loop, or Giskard's QP internals) rather than
# another scoping guess.
#
# Net: collision_avoidance=False is the only configuration that has ever
# completed the full 4-cube sequence without an exception, more than once.
# Its one known issue (the transit-sweep above) is intermittent, not
# guaranteed, unlike True's 3-for-3 identical crash. All the WORKAROUND #6-9
# infrastructure is left in place (harmless no-ops with this False, and
# already-written scaffolding if this is revisited with deeper source later)
# -- only this flag changes.
with ExecutionEnvironment(execution_type=execition_mode, collision_avoidance=False):
    for cube_label, cube, grasp_description, target_pose, landing_surface in stack_actions:
        # Print the REAL absolute-frame target ReachAction will command --
        # grasp_pose_sequence(cube) (used earlier) is expressed relative to the
        # cube's own frame (so a (0,0,0) translation there is correct, not a bug).
        # PickUpAction/ReachAction actually call pose_sequence(cube.global_pose,
        # cube), which is in world.root's frame -- that's the number that matters
        # for checking whether the target dips below the table (z near/below 0).
        pre_pose, target_pose_abs, lift_pose = grasp_description.pose_sequence(
            cube.global_pose, cube
        )
        log(f"[diag] {cube_label}: pre_pose (world frame) =\n{pose_repr(pre_pose)}")
        log(f"[diag] {cube_label}: grasp target (world frame) =\n{pose_repr(target_pose_abs)}")
        log(f"[diag] {cube_label}: lift_pose (world frame) =\n{pose_repr(lift_pose)}")

        steps = [
            ("park", park_until_converged),
            (
                "pick",
                lambda: diagnostic_pick(cube_label, cube, grasp_description),
            ),
            (
                "park-with-object",
                lambda: park_until_converged(held_body=cube),
            ),
            (
                "place",
                lambda: diagnostic_place(
                    cube_label, cube, grasp_description, target_pose, landing_surface
                ),
            ),
        ]
        log_poses(f"{cube_label} before park (baseline)", cube)
        log_joint_angles(f"{cube_label} before park (baseline)")
        log_other_cube_poses(f"{cube_label} before park (baseline)", cube)
        for step_label, step_fn in steps:
            log(f"[diag] --- {cube_label}: starting {step_label} ---")
            try:
                step_fn()
                # NOT calling settle() here anymore -- confirmed it was letting
                # the arm droop back under gravity during the idle sleep
                # (nothing actively holds position during time.sleep()),
                # undoing whatever progress the step just made. Direct proof:
                # park_until_converged's own immediate joint-error check
                # showed near-perfect convergence (0.0018) right after park,
                # but log_joint_angles -- printed moments later, after the old
                # settle() call -- showed the arm back near its unparked
                # position. Same mechanism as the earlier arm-drift bug, just
                # reintroduced by our own diagnostic harness in a new place.
                #
                # RESTORED (this round): _sync_position_holds() was dropped
                # from here when the hold-actuator infra was removed, but
                # never re-added when that removal itself got reverted --
                # leaving the hold spring's ctrl frozen at whatever the arm's
                # pose was at startup, fighting every subsequent motion for
                # the rest of the run (two competing controllers pulling in
                # different directions). That's a direct, mechanistic match
                # for "shaking while standing upright, can't converge."
                _sync_position_holds()
                log(f"[diag] --- {cube_label}: {step_label} done ---")
                log_poses(f"{cube_label} after {step_label}", cube)
                log_joint_angles(f"{cube_label} after {step_label}")
                log_other_cube_poses(f"{cube_label} after {step_label}", cube)
                print_positions()
            except Exception:
                log(f"[diag] --- {cube_label}: {step_label} FAILED, traceback below ---")
                log_joint_angles(f"{cube_label} at {step_label} FAILURE")
                _log_file.write(traceback.format_exc())
                _log_file.flush()
                traceback.print_exc()
                break
        else:
            continue
        break
    log("[diag] --- final park ---")
    try:
        park_until_converged()
        log("[diag] --- final park done ---")
    except Exception:
        log("[diag] --- final park FAILED, traceback below ---")
        _log_file.write(traceback.format_exc())
        _log_file.flush()
        traceback.print_exc()

# DIAGNOSTIC: keep the window/process alive after the plan finishes (or fails) so
# there's time to read the printed output above before anything closes.
log("[diag] --- final positions ---")
print_positions()
log("[diag] Plan sequence finished. Keeping simulation open for inspection...")
try:
    while rclpy.ok():
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    _log_file.close()