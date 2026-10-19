"""
Tiago opens and closes a drawer in a RoboCasa kitchen, simulated in MuJoCo: RoboCasa's
``OpenDrawer`` and ``CloseDrawer`` tasks. It takes hold of the drawer's handle and pulls
or pushes the drawer along its rails, holding the handle by the grip of its fingers
alone.

Run the opening task with::

    python -m experiments.tiago_robocasa.drawers OpenDrawer --record
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass

from experiments.hsr_robocasa.robocasa_task import FixtureReference
from experiments.tiago_robocasa.handles import TiagoHandleTask
from experiments.tiago_robocasa.trials import Trials, parser_of, settings_from
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


# %% a task at the kitchen's top drawer


@dataclass
class TiagoDrawerTask(TiagoHandleTask, ABC):
    """
    One of RoboCasa's drawer tasks, performed by Tiago: it takes hold of the drawer's
    handle and draws the drawer along its rails by it.
    """

    @property
    def fixture(self) -> Body:
        """
        :return: The housing the drawer slides out of.
        """
        return self.task.referenced_fixtures[FixtureReference.DRAWER]

    @property
    def drawer(self) -> Drawer:
        """
        :return: The annotation of the part that slides out of the housing the task
            refers to.
        """
        below_the_housing = (
            self.world.compute_descendent_child_kinematic_structure_entities(
                self.fixture
            )
        )
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(Drawer)
            if annotation.root in below_the_housing
        ][0]

    @property
    def handle(self) -> Handle:
        """
        :return: The drawer's handle.
        """
        return self.drawer.handle


# %% the tasks


@dataclass
class OpenDrawer(TiagoDrawerTask):
    """
    Tiago pulls the drawer out.
    """

    @property
    def goal_position(self) -> float:
        """
        :return: The end of the drawer's rails.
        """
        return float(self.mechanism.dof.limits.upper.position)

    @property
    def task_is_done(self) -> bool:
        """
        :return: Whether the drawer now stands open.
        """
        return self.travel >= self.open_ratio


@dataclass
class CloseDrawer(TiagoDrawerTask):
    """
    Tiago pushes the drawer back in, which RoboCasa has opened.
    """

    @property
    def goal_position(self) -> float:
        """
        :return: The shut end of the drawer's rails.
        """
        return float(self.mechanism.dof.limits.lower.position)

    @property
    def task_is_done(self) -> bool:
        """
        :return: Whether the drawer now stands closed.
        """
        return self.travel <= self.closed_ratio


TASKS = {"OpenDrawer": OpenDrawer, "CloseDrawer": CloseDrawer}
"""
The drawer tasks Tiago can be asked for, by the name RoboCasa gives them.
"""


# %% running one from the command line


def main() -> None:
    """
    Run the drawer task the command line names over one kitchen after another.
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
