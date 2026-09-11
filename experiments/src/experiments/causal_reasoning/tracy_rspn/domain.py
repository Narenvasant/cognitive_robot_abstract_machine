"""
Domain classes for picking one object out of a clutter, shaped after the GraspClutter6D
dataset (https://sites.google.com/view/graspclutter6d): a scene is one of a few
environment kinds holding many objects with known poses, a grasp is scored by the
friction coefficient it relies on, and every object in the scene is an exchangeable part
of it.

The ten-milk MuJoCo mock produces the same classes, so the pipelines fitted on it apply
unchanged once GraspClutter6D's own scenes are annotated (see
:mod:`~experiments.causal_reasoning.tracy_rspn.graspclutter6d`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.entity_query_language.factories import entity, count_range, variable
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)
from typing_extensions import List, Sequence

# %% vocabulary


class ClutterEnvironment(StrEnum):
    """
    The kind of environment a clutter stands in; GraspClutter6D's 75 configurations are
    bins, shelves and tables.
    """

    TABLE = "table"
    BIN = "bin"
    SHELF = "shelf"


class ObjectCategory(StrEnum):
    """
    What kind of object stands in the clutter.

    GraspClutter6D distinguishes 200 objects; the ten-milk mock only has one.
    """

    MILK = "milk"


class DistanceBand(StrEnum):
    """
    How close a neighbour stands to the object being picked, in bands wide enough for a
    fitted model to be support-deterministic over.
    """

    ADJACENT = "adjacent"
    """
    Closer than :data:`ADJACENT_DISTANCE`: inside the space a gripper sweeps when it
    descends onto the target.
    """

    NEAR = "near"
    """
    Between :data:`ADJACENT_DISTANCE` and :data:`NEAR_DISTANCE`.
    """

    FAR = "far"
    """
    Further than :data:`NEAR_DISTANCE`.
    """

    @classmethod
    def of(cls, distance: float) -> DistanceBand:
        """
        :param distance: Centre-to-centre distance to the target, in metres.
        :return: The band the distance falls into.
        """
        if distance < ADJACENT_DISTANCE:
            return cls.ADJACENT
        if distance < NEAR_DISTANCE:
            return cls.NEAR
        return cls.FAR


class ClosingAxisSide(StrEnum):
    """
    Where a neighbour stands relative to the axis the gripper's fingers close along.
    """

    ALONG = "along"
    """
    In the direction the fingers close in, where a descending finger lands on it.
    """

    ACROSS = "across"
    """
    Off to the side of the fingers, where only the gripper's housing passes over it.
    """

    @classmethod
    def of(cls, bearing: float) -> ClosingAxisSide:
        """
        :param bearing: Angle, in radians, between the closing axis and the direction
            from the target to the neighbour.
        :return: The side the neighbour stands on.
        """
        if abs(math.cos(bearing)) > math.cos(ALONG_HALF_ANGLE):
            return cls.ALONG
        return cls.ACROSS


ALONG_HALF_ANGLE = math.pi / 4
"""
Largest angle, in radians, between the closing axis and the direction to a neighbour for
the neighbour to count as standing along the axis.
"""

ADJACENT_DISTANCE = 0.09
"""
Centre-to-centre distance, in metres, below which a neighbour counts as adjacent to the
target: a Robotiq 2F-85's open fingers reach about 4.5cm out from the target's centre on
either side, and a 6cm carton standing closer than 9cm leaves less than the finger's own
thickness of clearance between the two.
"""

NEAR_DISTANCE = 0.15
"""
Centre-to-centre distance, in metres, below which a neighbour counts as near the target:

still inside the gripper housing's own footprint while it descends.
"""

DISTURBANCE_THRESHOLD = 0.01
"""
How far, in metres, a neighbour has to move during an attempt to count as disturbed by
it.
"""

FRICTION_LEVELS: Sequence[float] = (0.125, 0.1875, 0.25, 0.375, 0.5, 0.75)
"""
The sliding friction coefficients an attempt's grasp contact is given, one level per
attempt.

A ladder in the spirit of the one GraspNet-1Billion and GraspClutter6D score their
grasps on -- a grasp's score is the smallest friction coefficient it still closes under
-- placed around the coefficient below which a friction-held carton slips out of the
Robotiq 2F-85's pads, at levels a single-precision float represents exactly: a circuit's
support is read back in single precision, and a level that rounds there would no longer
match the point its own leaves sit on.
"""


# %% objects and attempts


@dataclass
class ClutteredObject:
    """
    One object standing next to the one being picked, as an exchangeable part of a
    :class:`ClutterPickScene`.
    """

    category: ObjectCategory
    """
    What kind of object it is.
    """

    x: float
    """
    Position along the table's x-axis relative to the target's centre, in metres.
    """

    y: float
    """
    Position along the table's y-axis relative to the target's centre, in metres.
    """

    yaw: float
    """
    Rotation about the vertical, in radians.
    """

    distance_to_target: float
    """
    Centre-to-centre distance to the target, in metres.
    """

    distance_band: DistanceBand
    """
    :attr:`distance_to_target` in bands (see :class:`DistanceBand`).
    """

    closing_axis_side: ClosingAxisSide
    """
    Whether the object stands in the direction the fingers closed in.
    """

    displacement: float
    """
    How far the object moved during the attempt, in metres.
    """

    disturbed: bool
    """
    Whether the attempt moved the object by more than :data:`DISTURBANCE_THRESHOLD`.
    """


@dataclass
class ClutterPickScene:
    """
    One attempt to pick one object out of a clutter, with every other object of the
    clutter as an exchangeable part.
    """

    environment: ClutterEnvironment
    """
    The kind of environment the clutter stands in.
    """

    target_x: float
    """
    Where the picked object stood along the robot's x-axis, in metres.
    """

    target_y: float
    """
    Where the picked object stood along the robot's y-axis, in metres.
    """

    friction_coefficient: float
    """
    Sliding friction coefficient of the objects' contacts during the attempt, one of
    :data:`FRICTION_LEVELS`.
    """

    grasp_yaw: float
    """
    Rotation of the gripper about the vertical when it closed on the target, in radians,
    relative to the target's own yaw: ``0`` closes the fingers across the target's
    x-faces, a quarter turn across its y-faces.
    """

    neighbours: List[ClutteredObject]
    """
    Every other object of the clutter.
    """

    lifted: bool
    """
    Whether the target was still held once the gripper had risen back to its hover
    height.
    """

    lift_height: float
    """
    How far the target rose during the attempt, in metres.
    """


@dataclass
class ClutterPickSceneAggregations(AggregationStatistic[ClutterPickScene]):
    """
    Aggregation statistics of a :class:`ClutterPickScene` over its neighbours.
    """

    @aggregation_statistic("neighbours")
    def crowding_count(self) -> int:
        """
        How many neighbours stand adjacent to the target.
        """
        band = variable(ClutteredObject, self.instance.neighbours).distance_band
        [result] = (
            entity(count_range(band)).where(band == DistanceBand.ADJACENT).tolist()
        )
        return result


# %% layouts: the input side of an attempt


@dataclass
class PlacedObject:
    """
    An object standing on the surface, before anything is picked.
    """

    category: ObjectCategory
    """
    What kind of object it is.
    """

    x: float
    """
    Position along the scene's x-axis, in metres.
    """

    y: float
    """
    Position along the scene's y-axis, in metres.
    """

    yaw: float
    """
    Rotation about the vertical, in radians.
    """

    def distance_to(self, other: PlacedObject) -> float:
        """
        :param other: The object to measure to.
        :return: Centre-to-centre distance in the plane, in metres.
        """
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class ClutterSceneLayout:
    """
    Everything an attempt is given before it runs: the objects, which one to pick, and
    the attempt's own parameters.
    """

    environment: ClutterEnvironment
    """
    The kind of environment the clutter stands in.
    """

    objects: List[PlacedObject]
    """
    Every object of the clutter, the target included.
    """

    target_index: int
    """
    Index into :attr:`objects` of the object to pick.
    """

    friction_coefficient: float
    """
    Sliding friction coefficient the objects' contacts get, one of
    :data:`FRICTION_LEVELS`.
    """

    grasp_yaw: float
    """
    Rotation of the gripper about the vertical for the grasp, in radians, relative to
    the target's own yaw.
    """

    @property
    def target(self) -> PlacedObject:
        """
        The object to pick.
        """
        return self.objects[self.target_index]

    @property
    def neighbours(self) -> List[PlacedObject]:
        """
        Every object other than the target, in their order in :attr:`objects`.
        """
        return [
            placed
            for index, placed in enumerate(self.objects)
            if index != self.target_index
        ]


@dataclass
class ClutterPickOutcome:
    """
    What an attempt on a :class:`ClutterSceneLayout` did to the scene.
    """

    lift_height: float
    """
    How far the target rose, in metres.
    """

    lifted: bool
    """
    Whether the target was still held at the end of the attempt.
    """

    neighbour_displacements: List[float] = field(default_factory=list)
    """
    How far each neighbour moved, in metres, in the order of
    :attr:`ClutterSceneLayout.neighbours`.
    """

    def to_scene(self, layout: ClutterSceneLayout) -> ClutterPickScene:
        """
        Record the attempt as the relational scene the pipelines are fitted on.

        :param layout: The layout the attempt ran on.
        :return: The scene, its neighbours expressed relative to the target.
        """
        target = layout.target
        closing_axis_yaw = target.yaw + layout.grasp_yaw
        neighbours = [
            ClutteredObject(
                category=placed.category,
                x=placed.x - target.x,
                y=placed.y - target.y,
                yaw=placed.yaw,
                distance_to_target=placed.distance_to(target),
                distance_band=DistanceBand.of(placed.distance_to(target)),
                closing_axis_side=ClosingAxisSide.of(
                    math.atan2(placed.y - target.y, placed.x - target.x)
                    - closing_axis_yaw
                ),
                displacement=displacement,
                disturbed=displacement > DISTURBANCE_THRESHOLD,
            )
            for placed, displacement in zip(
                layout.neighbours, self.neighbour_displacements
            )
        ]
        return ClutterPickScene(
            environment=layout.environment,
            target_x=target.x,
            target_y=target.y,
            friction_coefficient=layout.friction_coefficient,
            grasp_yaw=layout.grasp_yaw,
            neighbours=neighbours,
            lifted=self.lifted,
            lift_height=self.lift_height,
        )
