"""
The causal questions both pipelines are asked, each as one ``cause``/``causes_effect``
EQL query that reads the same for either pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.entity_query_language.factories import a, cause, confounder
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List

from experiments.causal_reasoning.tracy_rspn.domain import (
    ClutteredObject,
    ClutterPickScene,
)
from experiments.causal_reasoning.tracy_rspn.flat_table import (
    NEIGHBOUR_FIELDS,
    SCENE_SCALAR_FIELDS,
)

# %% building blocks


def neighbour_query(**specified: Any) -> Match:
    """
    A query for one neighbour with every attribute left open but the given ones.

    :param specified: Attribute markers or values to set instead of leaving open.
    :return: The query.
    """
    return a(ClutteredObject)(
        **{name: specified.get(name, ...) for name in NEIGHBOUR_FIELDS}
    )


def scene_query(neighbours: List[Match], **specified: Any) -> Match:
    """
    A query for an attempt with the given neighbours and every scalar attribute left
    open but the given ones.

    :param neighbours: One query per neighbour.
    :param specified: Scalar attribute markers or values to set instead of leaving open;
        an aggregation statistic's name is accepted too.
    :return: The query.
    """
    return a(ClutterPickScene)(
        **{name: specified.get(name, ...) for name in SCENE_SCALAR_FIELDS},
        **{
            name: value
            for name, value in specified.items()
            if name not in SCENE_SCALAR_FIELDS
        },
        neighbours=neighbours,
    )


# %% the questions


@dataclass(frozen=True)
class CausalQueryCase(ABC):
    """
    One causal question, with the query that asks it.
    """

    neighbour_count: int
    """
    How many neighbours the queried attempt has.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A short identifier of the question.
        """

    @property
    @abstractmethod
    def question(self) -> str:
        """
        The question in plain words.
        """

    @abstractmethod
    def build(self) -> Match:
        """
        :return: The query asking the question, freshly built.
        """


@dataclass(frozen=True)
class FrictionCausesLift(CausalQueryCase):
    """
    Which grasp friction makes the target come up, once the environment -- which decides
    both how slippery and how crowded an attempt is -- is adjusted for?
    """

    @property
    def name(self) -> str:
        return f"friction_causes_lift_{self.neighbour_count}_neighbours"

    @property
    def question(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, which grasp friction "
            "coefficient causes the target to be lifted, adjusting for the environment?"
        )

    def build(self) -> Match:
        query = scene_query(
            [neighbour_query() for _ in range(self.neighbour_count)],
            friction_coefficient=cause,
            environment=confounder,
        )
        query.causes_effect(query.variable.lifted == True)
        return query


@dataclass(frozen=True)
class CrowdingCausesLift(CausalQueryCase):
    """
    How many adjacent neighbours can the target have and still come up, once the
    environment is adjusted for?

    The cause is a count over the exchangeable parts.
    """

    @property
    def name(self) -> str:
        return f"crowding_causes_lift_{self.neighbour_count}_neighbours"

    @property
    def question(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, how many of them "
            "standing adjacent to the target causes it to be lifted, adjusting for the "
            "environment?"
        )

    def build(self) -> Match:
        query = scene_query(
            [neighbour_query() for _ in range(self.neighbour_count)],
            crowding_count=cause,
            environment=confounder,
        )
        query.causes_effect(query.variable.lifted == True)
        return query


@dataclass(frozen=True)
class ClosingAxisSideCausesDisturbance(CausalQueryCase):
    """
    Does one neighbour standing where the fingers close cause that neighbour to be
    shoved aside?

    Cause and effect both live on one exchangeable part.
    """

    neighbour_index: int
    """
    Which neighbour the question is about.
    """

    @property
    def name(self) -> str:
        return (
            f"closing_axis_side_causes_disturbance_of_neighbour_{self.neighbour_index}"
            f"_of_{self.neighbour_count}"
        )

    @property
    def question(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, does neighbour "
            f"{self.neighbour_index} standing along the fingers' closing axis cause it "
            "to be disturbed by the pick?"
        )

    def build(self) -> Match:
        neighbours = [neighbour_query() for _ in range(self.neighbour_count)]
        neighbours[self.neighbour_index] = neighbour_query(closing_axis_side=cause)
        query = scene_query(neighbours)
        query.causes_effect(
            query.variable.neighbours[self.neighbour_index].disturbed == True
        )
        return query


SMALLER_CLUTTER_NEIGHBOUR_COUNT = 4
"""
Neighbour count of the questions asked about a smaller clutter than the attempts were
recorded in.
"""

LARGER_CLUTTER_NEIGHBOUR_COUNT = 12
"""
Neighbour count of the questions asked about a larger clutter than the attempts were
recorded in.
"""


def query_catalogue(recorded_neighbour_count: int) -> List[CausalQueryCase]:
    """
    Every question of the experiment: each kind of cause asked about a clutter of the
    recorded size, then again about clutters of other sizes.

    :param recorded_neighbour_count: How many neighbours the recorded attempts have.
    :return: The questions, in the order they are asked.
    """
    return [
        FrictionCausesLift(recorded_neighbour_count),
        CrowdingCausesLift(recorded_neighbour_count),
        ClosingAxisSideCausesDisturbance(recorded_neighbour_count, neighbour_index=0),
        FrictionCausesLift(SMALLER_CLUTTER_NEIGHBOUR_COUNT),
        CrowdingCausesLift(LARGER_CLUTTER_NEIGHBOUR_COUNT),
        ClosingAxisSideCausesDisturbance(
            LARGER_CLUTTER_NEIGHBOUR_COUNT,
            neighbour_index=LARGER_CLUTTER_NEIGHBOUR_COUNT - 1,
        ),
    ]
