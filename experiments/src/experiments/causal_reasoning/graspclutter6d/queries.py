"""
The causal questions every pipeline is asked, each as one ``cause``/``causes_effect``
EQL query that reads the same for either pipeline.

Every query lists at least one object and one viewpoint with every attribute left open:
that is what makes grounding retain the scene's aggregation counts as variables instead
of computing them from an empty part list, and the flat table ignores parts a query says
nothing about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from krrood.entity_query_language.factories import a, cause, confounder
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List

from experiments.causal_reasoning.graspclutter6d.domain import (
    GraspClutterObject,
    GraspClutterScene,
    GraspClutterViewpoint,
    Graspability,
    Occlusion,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import SceneSchema

# %% building blocks


def object_query(**specified: Any) -> Match:
    """
    A query for one object instance with every attribute left open but the given ones.

    :param specified: Attribute markers or values to set instead of leaving open.
    :return: The query.
    """
    return a(GraspClutterObject)(
        **{
            object_field.name: specified.get(object_field.name, ...)
            for object_field in fields(GraspClutterObject)
        }
    )


def viewpoint_query(**specified: Any) -> Match:
    """
    A query for one viewpoint with every attribute left open but the given ones.

    :param specified: Attribute markers or values to set instead of leaving open.
    :return: The query.
    """
    return a(GraspClutterViewpoint)(
        **{
            viewpoint_field.name: specified.get(viewpoint_field.name, ...)
            for viewpoint_field in fields(GraspClutterViewpoint)
        }
    )


def scene_query(
    objects: List[Match],
    viewpoints: List[Match],
    schema: SceneSchema = SceneSchema(),
    **specified: Any,
) -> Match:
    """
    A query for a scene with the given objects and viewpoints and every scalar attribute
    left open but the given ones.

    :param objects: One query per object instance.
    :param viewpoints: One query per viewpoint.
    :param schema: How the scene's attributes are named.
    :param specified: Scalar attribute markers or values to set instead of leaving open;
        an aggregation count's name is accepted too.
    :return: The query.
    """
    scalar_fields = schema.scalar_fields
    return a(GraspClutterScene)(
        **{name: specified.get(name, ...) for name in scalar_fields},
        **{
            name: value
            for name, value in specified.items()
            if name not in scalar_fields
        },
        objects=objects,
        viewpoints=viewpoints,
    )


# %% the questions


@dataclass(frozen=True, kw_only=True)
class CausalQueryCase(ABC):
    """
    One question, as a query and as words.
    """

    object_count: int = 1
    """
    How many object instances the query lists with every attribute open.
    """

    viewpoint_count: int = 1
    """
    How many viewpoints the query lists with every attribute open.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A short identifier for tables.
        """

    @property
    @abstractmethod
    def question(self) -> str:
        """
        The question in words.
        """

    @abstractmethod
    def build(self) -> Match:
        """
        :return: The query, freshly built, with its cause, confounders and effect
            marked.
        """

    @abstractmethod
    def describe_cause(self, region: str) -> str:
        """
        :param region: A region of the cause, written out.
        :return: The cause set to that region, in words.
        """

    @property
    @abstractmethod
    def effect(self) -> str:
        """
        The effect in words.
        """

    def _open_objects(self) -> List[Match]:
        """
        :return: One fully open query per listed object instance.
        """
        return [object_query() for _ in range(self.object_count)]

    def _open_viewpoints(self) -> List[Match]:
        """
        :return: One fully open query per listed viewpoint.
        """
        return [viewpoint_query() for _ in range(self.viewpoint_count)]


@dataclass(frozen=True, kw_only=True)
class CountCausesGraspability(CausalQueryCase):
    """
    Does one of the scene's aggregation counts cause every object in it to stay
    graspable, once the spread of the clutter is adjusted for?
    """

    statistic_name: str
    """
    The aggregation statistic of
    :class:`~experiments.causal_reasoning.graspclutter6d.domain.GraspClutterSceneAggregations`
    that is the cause.
    """

    count_noun: str
    """
    What the count counts, in words, such as ``small objects``.
    """

    confounder_name: str = "extent"
    """
    The scene attribute to adjust for.
    """

    confounder_noun: str = "how far the clutter is spread out"
    """
    The confounder in words.
    """

    @property
    def name(self) -> str:
        return f"{self.statistic_name}_causes_graspability"

    @property
    def question(self) -> str:
        return (
            f"How many {self.count_noun} cause every object of a scene to stay "
            f"graspable, adjusting for {self.confounder_noun}?"
        )

    def build(self) -> Match:
        query = scene_query(
            self._open_objects(),
            self._open_viewpoints(),
            **{self.statistic_name: cause, self.confounder_name: confounder},
        )
        query.causes_effect(query.variable.all_objects_graspable == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"{region} {self.count_noun}"

    @property
    def effect(self) -> str:
        return "every object of the scene stays graspable"


@dataclass(frozen=True, kw_only=True)
class CatalogueCausesGraspability(CausalQueryCase):
    """
    Does which object catalogue a scene is built from cause every object in it to stay
    graspable, once one other attribute of the scene is adjusted for?
    """

    confounder_name: str
    """
    The scene attribute or aggregation count to adjust for.
    """

    confounder_noun: str
    """
    The confounder in words, such as ``small-object count``.
    """

    @property
    def name(self) -> str:
        return f"catalogue_causes_graspability_adjusting_{self.confounder_name}"

    @property
    def question(self) -> str:
        return (
            "Does the object catalogue a scene is built from cause every object of it "
            f"to stay graspable, adjusting for its {self.confounder_noun}?"
        )

    def build(self) -> Match:
        query = scene_query(
            self._open_objects(),
            self._open_viewpoints(),
            object_catalogue=cause,
            **{self.confounder_name: confounder},
        )
        query.causes_effect(query.variable.all_objects_graspable == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"the {region} catalogue"

    @property
    def effect(self) -> str:
        return "every object of the scene stays graspable"


@dataclass(frozen=True, kw_only=True)
class CatalogueCausesOcclusion(CausalQueryCase):
    """
    Does which object catalogue a scene is built from cause one of its objects to be
    heavily occluded?

    The cause is a scene attribute, the effect one object's own.
    """

    object_index: int = 0
    """
    Which listed object the effect is about.
    """

    @property
    def name(self) -> str:
        return f"catalogue_causes_occluded_object_{self.object_index}"

    @property
    def question(self) -> str:
        return (
            "Does the object catalogue a scene is built from cause object "
            f"{self.object_index} of it to be heavily occluded?"
        )

    def build(self) -> Match:
        query = scene_query(
            self._open_objects(), self._open_viewpoints(), object_catalogue=cause
        )
        query.causes_effect(
            query.variable.objects[self.object_index].occlusion
            == Occlusion.HEAVILY_OCCLUDED
        )
        return query

    def describe_cause(self, region: str) -> str:
        return f"the {region} catalogue"

    @property
    def effect(self) -> str:
        return f"object {self.object_index} is heavily occluded"


@dataclass(frozen=True, kw_only=True)
class OccludedObjectsCauseBlockedObject(CausalQueryCase):
    """
    Does the number of occluded objects in a scene cause one of its objects to lose
    every grasp?

    The cause is a count over the objects, the effect one object's own attribute.
    """

    object_index: int = 0
    """
    Which listed object the effect is about.
    """

    @property
    def name(self) -> str:
        return f"occluded_object_count_causes_blocked_object_{self.object_index}"

    @property
    def question(self) -> str:
        return (
            "How many occluded objects cause object "
            f"{self.object_index} of a scene to lose every grasp?"
        )

    def build(self) -> Match:
        query = scene_query(
            self._open_objects(), self._open_viewpoints(), occluded_object_count=cause
        )
        query.causes_effect(
            query.variable.objects[self.object_index].graspability
            == Graspability.BLOCKED
        )
        return query

    def describe_cause(self, region: str) -> str:
        return f"{region} occluded objects"

    @property
    def effect(self) -> str:
        return f"object {self.object_index} loses every grasp"


@dataclass(frozen=True, kw_only=True)
class SizeCausesBlockedObject(CausalQueryCase):
    """
    Does an object's size cause it to lose every grasp?

    Cause and effect both live on one object.
    """

    object_index: int = 0
    """
    Which listed object the question is about.
    """

    @property
    def name(self) -> str:
        return f"size_causes_blocked_object_{self.object_index}"

    @property
    def question(self) -> str:
        return (
            f"Does the size of object {self.object_index} of a scene cause it to lose "
            "every grasp?"
        )

    def build(self) -> Match:
        objects = self._open_objects()
        objects[self.object_index] = object_query(size=cause)
        query = scene_query(objects, self._open_viewpoints())
        query.causes_effect(
            query.variable.objects[self.object_index].graspability
            == Graspability.BLOCKED
        )
        return query

    def describe_cause(self, region: str) -> str:
        return f"object {self.object_index} being {region}"

    @property
    def effect(self) -> str:
        return f"object {self.object_index} loses every grasp"


def scene_level_cases() -> List[CausalQueryCase]:
    """
    :return: The questions whose cause and effect are both scene attributes or counts, in
        the order they are asked.
    """
    return [
        CountCausesGraspability(
            statistic_name="small_object_count", count_noun="small objects"
        ),
        CountCausesGraspability(
            statistic_name="occluded_object_count", count_noun="occluded objects"
        ),
        CountCausesGraspability(
            statistic_name="clear_viewpoint_count", count_noun="clear viewpoints"
        ),
        CatalogueCausesGraspability(
            confounder_name="extent", confounder_noun="spread (extent)"
        ),
        CatalogueCausesGraspability(
            confounder_name="small_object_count", confounder_noun="small-object count"
        ),
    ]


def object_level_cases() -> List[CausalQueryCase]:
    """
    :return: The questions whose cause or effect lives on one object, in the order they
        are asked.
    """
    return [
        CatalogueCausesOcclusion(),
        OccludedObjectsCauseBlockedObject(),
        SizeCausesBlockedObject(),
    ]


def query_catalogue() -> List[CausalQueryCase]:
    """
    Every question of the experiment: the scene-level causes of graspability first, then
    the questions whose cause or effect lives on one object.

    :return: The questions, in the order they are asked.
    """
    return scene_level_cases() + object_level_cases()
