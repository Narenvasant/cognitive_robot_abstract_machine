"""
A closed-form stand-in for the MuJoCo attempt, with the same shape and the same causal
structure the ten-milk mock is built to produce, so the pipelines can be exercised
without a simulator.

Friction lets the fingers hold the target; every adjacent neighbour is in the way of the
descending fingers and takes a share of that hold away; and the environment drives both
(see :data:`~experiments.causal_reasoning.tracy_rspn.layout_sampler.ENVIRONMENT_DISTRIBUTIONS`),
which is what makes it a confounder of friction and success.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import List

from experiments.causal_reasoning.tracy_rspn.domain import (
    FRICTION_LEVELS,
    ClutterPickOutcome,
    ClutterPickScene,
    ClutterSceneLayout,
    DistanceBand,
)
from experiments.causal_reasoning.tracy_rspn.layout_sampler import (
    ClutterLayoutSampler,
)

HOLD_PER_FRICTION = 1.0 / max(FRICTION_LEVELS)
"""
How much of a sure hold each unit of friction coefficient buys: the highest friction
level alone holds the target for certain.
"""

HOLD_LOST_PER_ADJACENT_NEIGHBOUR = 0.25
"""
How much of the hold each adjacent neighbour takes away.
"""

LIFTED_HEIGHT = 0.25
"""
How far, in metres, a held target rises: the hover clearance the gripper returns to,
less the closing swing it descended below it.
"""

ADJACENT_NEIGHBOUR_DISPLACEMENT = 0.03
"""
How far, in metres, an adjacent neighbour is shoved when the fingers land on it.
"""


@dataclass
class SyntheticPickOutcomes:
    """
    Draws the outcome of an attempt on a layout from the closed-form model above.
    """

    random_state: np.random.Generator
    """
    Source of randomness for the hold's own coin flip.
    """

    def simulate(self, layout: ClutterSceneLayout) -> ClutterPickOutcome:
        """
        :param layout: The layout to attempt.
        :return: The attempt's outcome.
        """
        target = layout.target
        adjacent = [
            neighbour
            for neighbour in layout.neighbours
            if DistanceBand.of(neighbour.distance_to(target)) == DistanceBand.ADJACENT
        ]
        hold_probability = float(
            np.clip(
                layout.friction_coefficient * HOLD_PER_FRICTION
                - len(adjacent) * HOLD_LOST_PER_ADJACENT_NEIGHBOUR,
                0.0,
                1.0,
            )
        )
        lifted = bool(self.random_state.uniform() < hold_probability)
        displacements = [
            (
                ADJACENT_NEIGHBOUR_DISPLACEMENT
                if DistanceBand.of(neighbour.distance_to(target))
                == DistanceBand.ADJACENT
                else 0.0
            )
            for neighbour in layout.neighbours
        ]
        return ClutterPickOutcome(
            lift_height=LIFTED_HEIGHT if lifted else 0.0,
            lifted=lifted,
            neighbour_displacements=displacements,
        )


def synthetic_clutter_pick_scenes(
    random_state: np.random.Generator, scene_count: int, object_count: int = 10
) -> List[ClutterPickScene]:
    """
    Draw random layouts and attempt each with :class:`SyntheticPickOutcomes`.

    :param random_state: Source of randomness for layouts and outcomes alike.
    :param scene_count: How many attempts to record.
    :param object_count: How many objects each layout holds, the target included.
    :return: The recorded attempts.
    """
    sampler = ClutterLayoutSampler(random_state, object_count=object_count)
    outcomes = SyntheticPickOutcomes(random_state)
    scenes = []
    for _ in range(scene_count):
        layout = sampler.sample()
        scenes.append(outcomes.simulate(layout).to_scene(layout))
    return scenes
