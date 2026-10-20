"""
Tiago turns a toaster on in a RoboCasa kitchen, simulated in MuJoCo: RoboCasa's
``TurnOnToaster`` task. It takes hold of the toaster's lever and pushes it down along
its slot, holding it by the grip of its fingers alone.

Run the task with::

    python -m experiments.tiago_robocasa.toaster TurnOnToaster --record
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import Callable

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from experiments.scene_recording import ViewerCamera
from experiments.tiago_robocasa.handles import TiagoHandleTask
from experiments.tiago_robocasa.trials import Trials, parser_of, settings_from
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


class ToasterPart(StrEnum):
    """
    How RoboCasa names the bodies of a toaster this task works with.
    """

    PREFIX = "toaster_"
    """
    The start of the name of every body of a toaster, and of a toaster oven.
    """

    OVEN = "toaster_oven_"
    """
    The start of the name of every body of a toaster oven, which is not a toaster.
    """

    MAIN = "_main"
    """
    The end of the name of the toaster's housing.
    """

    LEVER = "lever"
    """
    The end of the name of a lever that lowers the bread into a slot.
    """


# %% the task


@dataclass
class TurnOnToaster(TiagoHandleTask):
    """
    Tiago pushes the toaster's lever down, which turns the toaster on.

    The lever springs back up once the hand lets go of it, so whether the toaster was
    turned on is read while the hand still holds it down.
    """

    handle_reach: float = 0.7
    """
    How far in front of its base's centre Tiago takes hold of the lever, in metres:
    the toaster stands at the back of the counter.
    """

    turned_on_ratio: float = 0.9
    """
    How far down its slot the lever has to be pushed, as a share of its travel, for
    RoboCasa to count the toaster turned on.
    """

    deepest_travel: float = field(init=False, default=0.0)
    """
    How far down its slot the lever stood, as a share of its travel, while the hand
    held it down.
    """

    @property
    def fixture(self) -> Body:
        """
        :return: The toaster's housing.
        """
        return next(
            body
            for body in self.world.bodies
            if body.name.name.startswith(ToasterPart.PREFIX)
            and not body.name.name.startswith(ToasterPart.OVEN)
            and body.name.name.endswith(ToasterPart.MAIN)
        )

    @property
    def handle(self) -> Handle:
        """
        :return: The first of the toaster's levers.
        """
        prefix = self.fixture.name.name.removesuffix(ToasterPart.MAIN)
        levers = sorted(
            (
                body
                for body in self.world.bodies
                if body.name.name.startswith(prefix)
                and body.name.name.endswith(ToasterPart.LEVER)
            ),
            key=lambda body: body.name.name,
        )
        return Handle(root=levers[0])

    @property
    def handle_box(self) -> tuple[np.ndarray, np.ndarray]:
        """
        :return: The lowest and highest corner of the piece of the lever that stands
            furthest out of the toaster, which the fingers close on: the lever's body
            also takes in the shaft it slides on inside the housing.
        """
        housing = self.extent(self.fixture).center.to_np()[:2]
        pieces = self.handle.root.collision.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_boxes
        outermost = max(
            pieces,
            key=lambda piece: np.linalg.norm(piece.center.to_np()[:2] - housing),
        )
        return self.corners_around([outermost])

    @property
    def handled_bodies(self) -> list[Body]:
        """
        :return: The toaster and the counter it stands on: the lever stands a few
            centimetres above the counter, and the arm reaching it passes over the
            counter closer than the clearance it is otherwise held to.
        """
        return super().handled_bodies + self.bodies_under_the_handle

    @property
    def headings_to_try(self) -> tuple[float, ...]:
        """
        :return: The one way that faces the toaster, read off which way the lever
            stands out of its housing.
        """
        outward = self.handle_middle - self.extent(self.fixture).center.to_np()[:3]
        return (float(np.arctan2(-outward[1], -outward[0])),)

    recording_distance: float = 1.6
    """
    How far the recording camera stands from the lever, in metres: close enough to
    see the fingers on a lever a few centimetres wide.
    """

    @property
    def recording_camera(self) -> ViewerCamera:
        """
        :return: A camera looking down at the lever from close by.
        """
        return self.camera_on(self.handle_middle, self.recording_distance)

    @property
    def goal_position(self) -> float:
        """
        :return: The bottom of the lever's slot.
        """
        return float(self.mechanism.dof.limits.upper.position)

    @property
    def stages(self) -> list[Callable[[Context], Plan]]:
        """
        :return: Pushing the lever down and holding it there, then letting go of it.
        """
        return [self.pushing_the_lever_down, self.letting_go_of_the_lever]

    def pushing_the_lever_down(self, context: Context) -> Plan:
        """
        :param context: The context to plan in.
        :return: Standing at the toaster, taking hold of the lever and pushing it down.
        """
        self.stand_at_the_handle()
        return sequential(self.drawing_the_fixture(), context=context).plan

    def letting_go_of_the_lever(self, context: Context) -> Plan:
        """
        Note how far down the lever was pushed before the hand lets go of it.

        :param context: The context to plan in.
        :return: Letting go of the lever and backing the hand away.
        """
        self.deepest_travel = self.travel
        return sequential(self.letting_go_of_the_handle(), context=context).plan

    @property
    def task_is_done(self) -> bool:
        """
        :return: Whether the lever was pushed far enough down to turn the toaster on.
        """
        return self.deepest_travel >= self.turned_on_ratio

    @property
    def outcome(self) -> str:
        """
        :return: How far down the lever was pushed and where it stands now.
        """
        report = (
            f"{self.handle.root.name.name} was pushed down {self.deepest_travel:.0%}"
            f" of its travel and stands at {self.travel:.0%}"
        )
        if self.disturbed_fixtures:
            report += f"; also moved: {', '.join(self.disturbed_fixtures)}"
        return report


TASKS = {"TurnOnToaster": TurnOnToaster}
"""
The toaster task Tiago can be asked for, by the name RoboCasa gives it.
"""


# %% running it from the command line


def main() -> None:
    """
    Run the toaster task over one kitchen after another.
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
