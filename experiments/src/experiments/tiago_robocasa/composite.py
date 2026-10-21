"""
Tiago carries out RoboCasa's composite kitchen tasks, simulated in MuJoCo: each one a
sequence of the steps Tiago already performs on their own - working a door, sliding a
rack, carrying an object - carried out one after the other in the same kitchen, by
physics alone.

Run a task with::

    python -m experiments.tiago_robocasa.composite ServeWarmCroissant --record
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import Callable

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.plan import Plan
from experiments.scene_recording import ViewerCamera
from experiments.tiago_robocasa.appliances import AppliancePart, SlideToasterOvenRack
from experiments.tiago_robocasa.doors import TiagoOpensADoor
from experiments.tiago_robocasa.pick_place import TiagoPickPlaceTask
from experiments.tiago_robocasa.robocasa_task import TiagoRoboCasaTask
from experiments.tiago_robocasa.trials import Trials, parser_of, settings_from
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


class CompositeObject(StrEnum):
    """
    The names RoboCasa gives the bodies of the objects its composite tasks move.
    """

    CROISSANT = "croissant_main"
    """
    The croissant warming in the toaster oven.
    """

    PLATE = "plate_main"
    """
    The plate a croissant is served on.
    """


# %% a task made of steps


@dataclass
class TiagoCompositeTask(TiagoRoboCasaTask, ABC):
    """
    One of RoboCasa's composite tasks, performed by Tiago as a sequence of steps, each
    a task Tiago performs on its own, carried out in the kitchen this task composes.
    """

    recording_distance: float = 2.2
    """
    How far the recording camera stands from where the first step reaches first, in
    metres: close enough to see the hand, far enough to follow it between the steps.
    """

    steps: list[TiagoRoboCasaTask] = field(init=False, default_factory=list)
    """
    The steps, in the order they are carried out, once the kitchen is composed.
    """

    @abstractmethod
    def step_sequence(self) -> list[TiagoRoboCasaTask]:
        """
        :return: The steps the task is carried out in, not yet in any kitchen.
        """

    def prepare_for_the_kitchen(self) -> None:
        """
        Put every step into the kitchen just composed.
        """
        self.steps = self.step_sequence()
        for step in self.steps:
            step.work_in(self)

    @property
    def handled_bodies(self) -> list[Body]:
        """
        :return: What any of the steps works among.
        """
        return [body for step in self.steps for body in step.handled_bodies]

    @property
    def first_target(self) -> np.ndarray:
        """
        :return: Where the first step reaches first.
        """
        return self.steps[0].first_target

    @property
    def recording_camera(self) -> ViewerCamera:
        """
        :return: A camera on where the first step reaches first.
        """
        return self.camera_on(self.first_target, self.recording_distance)

    def build_plan(self, context: Context) -> Plan:
        """
        :param context: The context to plan in.
        :return: The plan of the first step, the others following as :attr:`stages`.
        """
        return self.steps[0].build_plan(context)

    @property
    def stages(self) -> list[Callable[[Context], Plan]]:
        """
        :return: The stages of every step, one step after the other.
        """
        return [stage for step in self.steps for stage in step.stages]

    def planned_with_the_arm_alone(
        self, build: Callable[[Context], Plan], context: Context
    ) -> Plan:
        """
        Plan the stage as the step it belongs to plans it, whose base may follow its
        arm where this task's does not.

        :param build: One of the stages of a step.
        :param context: The context to plan in.
        :return: The plan.
        """
        return build.__self__.planned_with_the_arm_alone(build, context)

    @property
    def task_is_done(self) -> bool:
        """
        :return: Whether every step left the scene as it asks.
        """
        return all(step.task_is_done for step in self.steps)

    @property
    def outcome(self) -> str:
        """
        :return: What became of every step.
        """
        return "; ".join(
            f"{type(step).__name__}: {step.outcome}" for step in self.steps
        )


# %% the steps


@dataclass
class OpenToasterOvenDoor(TiagoOpensADoor):
    """
    Tiago swings the toaster oven's door down open.
    """

    handle_reach: float = 0.7

    waits_in_joint_space: bool = True

    @property
    def fixture(self) -> Body:
        """
        :return: The toaster oven's housing.
        """
        return self.task.referenced_fixtures[AppliancePart.TOASTER_OVEN]


# %% the tasks


@dataclass
class ServeWarmCroissant(TiagoCompositeTask):
    """
    Tiago opens the toaster oven's door, pulls its rack out, and moves the croissant
    warming on it onto the plate on the counter beside the toaster oven.
    """

    def step_sequence(self) -> list[TiagoRoboCasaTask]:
        return [
            OpenToasterOvenDoor(),
            SlideToasterOvenRack(),
            TiagoPickPlaceTask(
                object_name=CompositeObject.CROISSANT,
                receptacle_name=CompositeObject.PLATE,
            ),
        ]


TASKS = {task.__name__: task for task in (ServeWarmCroissant,)}
"""
The composite tasks Tiago can be asked for, by the name RoboCasa gives them.
"""


# %% running one from the command line


def main() -> None:
    """
    Run the composite task the command line names over one kitchen after another.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    arguments = parser_of(__doc__, list(TASKS)).parse_args()
    Trials(
        task=TASKS[arguments.task],
        settings=settings_from(arguments),
        attempts=arguments.attempts,
        tries_every_kitchen=arguments.benchmark,
    ).run()


if __name__ == "__main__":
    main()
