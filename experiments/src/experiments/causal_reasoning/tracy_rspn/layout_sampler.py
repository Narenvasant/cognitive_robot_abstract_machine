"""
Random clutter layouts for the ten-milk mock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Dict, List, Sequence, Tuple

from experiments.causal_reasoning.tracy_rspn.domain import (
    FRICTION_LEVELS,
    ClutterEnvironment,
    ClutterSceneLayout,
    ObjectCategory,
    PlacedObject,
)

CLUSTER_CENTRE_X = 0.8
CLUSTER_CENTRE_Y = 0.25
"""
Where the clutter is centred on the table, in the robot's root frame: in front of
Tracy's left arm, where a top-down reach has been confirmed to succeed.
"""

CLUSTER_COLUMNS = 4
"""
How many objects stand side by side along the table's x-axis in one row of the cluster.
"""

POSITION_JITTER = 0.008
"""
Largest offset, in metres, an object is shifted off its grid position along each axis.
"""

YAW_JITTER = 0.3
"""
Largest rotation, in radians, an object is turned away from the grid's own alignment.
"""

GRASP_FACE_YAWS: Sequence[float] = (0.0, math.pi / 2)
"""
The gripper yaws, relative to the target, that close the fingers across one of its two
pairs of faces.
"""

GRASP_YAW_ERROR = 0.1
"""
Largest error, in radians, the grasp's yaw is off the face pair it aims across.
"""


@dataclass(frozen=True)
class EnvironmentDistribution:
    """
    How one kind of environment's clutters are laid out and what friction their
    objects have.

    This is what makes the environment a confounder: it drives both how crowded the
    target is and how much friction the grasp gets, so the two are correlated in the
    recorded attempts without either causing the other.
    """

    minimum_spacing: float
    """
    Smallest centre-to-centre grid spacing, in metres.
    """

    maximum_spacing: float
    """
    Largest centre-to-centre grid spacing, in metres.
    """

    friction_levels: Tuple[float, ...]
    """
    The friction coefficients an attempt in this environment draws from.
    """


ENVIRONMENT_DISTRIBUTIONS: Dict[ClutterEnvironment, EnvironmentDistribution] = {
    ClutterEnvironment.TABLE: EnvironmentDistribution(
        0.09, 0.14, tuple(FRICTION_LEVELS)
    ),
    ClutterEnvironment.BIN: EnvironmentDistribution(
        0.065, 0.09, tuple(FRICTION_LEVELS[:3])
    ),
}
"""
The mock's environments: a table leaves room between the cartons and holds any
material, a bin packs them tightly and holds the slippery ones.
"""


@dataclass
class ClutterLayoutSampler:
    """
    Draws random clutter layouts: a jittered grid of objects in one of the mock's
    environments, one of them picked as the target.
    """

    random_state: np.random.Generator
    """
    Source of randomness.
    """

    object_count: int = 10
    """
    How many objects each layout holds, the target included.
    """

    environments: Tuple[ClutterEnvironment, ...] = field(
        default_factory=lambda: tuple(ENVIRONMENT_DISTRIBUTIONS)
    )
    """
    The environments a layout is drawn from, each equally likely.
    """

    def sample(self) -> ClutterSceneLayout:
        """
        :return: One random layout.
        """
        environment = self.environments[
            self.random_state.integers(len(self.environments))
        ]
        distribution = ENVIRONMENT_DISTRIBUTIONS[environment]
        spacing = float(
            self.random_state.uniform(
                distribution.minimum_spacing, distribution.maximum_spacing
            )
        )
        objects = self._grid(spacing)
        return ClutterSceneLayout(
            environment=environment,
            objects=objects,
            target_index=int(self.random_state.integers(len(objects))),
            friction_coefficient=float(
                distribution.friction_levels[
                    self.random_state.integers(len(distribution.friction_levels))
                ]
            ),
            grasp_yaw=float(
                GRASP_FACE_YAWS[self.random_state.integers(len(GRASP_FACE_YAWS))]
                + self.random_state.uniform(-GRASP_YAW_ERROR, GRASP_YAW_ERROR)
            ),
        )

    def _grid(self, spacing: float) -> List[PlacedObject]:
        """
        Lay the objects out on a jittered grid centred on the cluster centre.

        :param spacing: Centre-to-centre distance between grid positions, in metres.
        :return: The placed objects, row by row.
        """
        row_count = math.ceil(self.object_count / CLUSTER_COLUMNS)
        x_offset = (CLUSTER_COLUMNS - 1) / 2
        y_offset = (row_count - 1) / 2
        objects = []
        for index in range(self.object_count):
            row, column = divmod(index, CLUSTER_COLUMNS)
            jitter_x, jitter_y = self.random_state.uniform(
                -POSITION_JITTER, POSITION_JITTER, size=2
            )
            objects.append(
                PlacedObject(
                    category=ObjectCategory.MILK,
                    x=CLUSTER_CENTRE_X
                    + (column - x_offset) * spacing
                    + float(jitter_x),
                    y=CLUSTER_CENTRE_Y + (row - y_offset) * spacing + float(jitter_y),
                    yaw=float(self.random_state.uniform(-YAW_JITTER, YAW_JITTER)),
                )
            )
        return objects
