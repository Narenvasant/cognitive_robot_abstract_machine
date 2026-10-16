"""
The Toyota HSR opens and closes a drawer in a RoboCasa kitchen, simulated in MuJoCo:
RoboCasa's ``OpenDrawer`` and ``CloseDrawer`` tasks. The HSR takes hold of the
drawer's handle, and pulls or pushes the drawer along its rails.

Run the opening task with::

    python -m experiments.hsr_robocasa.drawers
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.container import (
    CloseAction,
    ContainerAction,
    OpenAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.world_entity import Body

from experiments.hsr_robocasa.robocasa_task import ArmLink, HSRRoboCasaTask


class FixtureReference(StrEnum):
    """
    The names RoboCasa's tasks refer to their fixtures by.
    """

    DRAWER = "drawer"
    """
    The drawer a drawer task works.
    """

    COUNTER = "counter"
    """
    The counter above the drawer.
    """


@dataclass
class HSRTaskAtDrawer(HSRRoboCasaTask, ABC):
    """
    One of RoboCasa's tasks at its top drawer, performed by the HSR.
    """

    counter_height: float = 0.84
    """
    Lowered less than for the other tasks: RoboCasa takes a drawer for the top drawer
    its drawer tasks work only while the drawer's middle stands 0.7 m to 0.9 m high.
    """

    handle_reach: float = 0.6
    """
    How far in front of its base's centre the HSR takes hold of the handle: from the
    front, and with its hand level the arm reaches 0.37 m to 0.70 m ahead at the
    handle's height.
    """

    @property
    def drawer(self) -> Drawer:
        """
        The annotation of the drawer the task works: the part sliding out of the
        housing the task refers to.
        """
        housing = self.task.referenced_fixtures[FixtureReference.DRAWER]
        below_the_housing = (
            self.world.compute_descendent_child_kinematic_structure_entities(housing)
        )
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(Drawer)
            if annotation.root in below_the_housing
        ][0]

    @property
    def handle_middle(self) -> np.ndarray:
        """
        The middle of the drawer's handle.
        """
        return self.extent(self.drawer.handle.root).center.to_np()[:3]

    def handle_grasp(self) -> GraspCandidate:
        """
        :return: The grasp that takes the handle at its middle, approaching it level
            from the HSR's side of the counter, the fingers closing across the bar from
            above and below.
        """
        handle = self.drawer.handle
        world_T_grasp = HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.handle_middle,
            roll=math.pi / 2,
            yaw=math.pi / 2,
            reference_frame=self.world.root,
        )
        return GraspCandidate(
            handle, self.world.transform(world_T_grasp.pose, handle.root)
        )

    def drawer_plan(self, context: Context, action_type: type[ContainerAction]) -> Plan:
        """
        :param context: The context the actions are built in.
        :param action_type: The action that moves the drawer.
        :return: The plan that stands in front of the drawer's handle, moves the arm
            to where taking hold of the handle starts its approach, and moves the
            drawer.
        """
        robot = context.robot
        arm = robot.all_arms[0]
        grasp = self.handle_grasp()
        action = action_type(self.drawer.handle, arm, grasp=grasp)
        pre_grasp = action.grasp_pose_sequence(
            grasp.world_T_grasp, arm.end_effector, grasp
        ).pre_grasp
        standing_pose = self.standing_pose(self.handle_middle, self.handle_reach)
        return sequential(
            [
                ParkArmsAction(robot.all_arms),
                NavigateAction(standing_pose),
                *self.arm_motions_to(pre_grasp, standing_pose),
                action,
                ParkArmsAction(robot.all_arms),
            ],
            context=context,
        ).plan


@dataclass
class HSRDrawerTask(HSRTaskAtDrawer, ABC):
    """
    One of RoboCasa's drawer tasks, performed by the HSR: it takes hold of the
    drawer's handle and moves the drawer along its rails.
    """

    tool_frame_tolerance: float = 0.02
    """
    Following the drawer, the base moves sideways too, by 7 cm in the closing task,
    and the arm, swinging in its plane, cannot make up for that when backing off the
    released handle: it stops 1.6 cm to the side of where it backs off to.
    """

    @property
    @abstractmethod
    def drawer_action(self) -> type[ContainerAction]:
        """
        The action that moves the drawer as the task asks.
        """

    @property
    def handled_bodies(self) -> list[Body]:
        """
        The drawer and its handle.
        """
        return self.colliding_bodies_below(self.drawer.root)

    @property
    def bodies_touching_handled_bodies(self) -> list[Body]:
        """
        The wrist's links and the forearm, which reaches out level with the hand and
        which the drawer's front comes towards as the drawer opens.
        """
        return self.arm_links(list(ArmLink))

    @property
    def first_target(self) -> np.ndarray:
        """
        The middle of the drawer's handle.
        """
        return self.handle_middle

    def build_plan(self, context: Context) -> Plan:
        return self.drawer_plan(context, self.drawer_action)


# %% the tasks


@dataclass
class OpenDrawer(HSRDrawerTask):
    """
    Open the drawer.
    """

    @property
    def drawer_action(self) -> type[ContainerAction]:
        return OpenAction


@dataclass
class CloseDrawer(HSRDrawerTask):
    """
    Close the drawer, which RoboCasa has opened.
    """

    @property
    def drawer_action(self) -> type[ContainerAction]:
        return CloseAction


if __name__ == "__main__":
    experiment = OpenDrawer()
    experiment.build_world()
    experiment.run()
