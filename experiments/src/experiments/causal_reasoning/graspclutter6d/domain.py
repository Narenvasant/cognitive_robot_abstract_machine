"""
Domain classes for the GraspClutter6D dataset (https://sites.google.com/view/graspclutter6d),
1000 real, densely cluttered bin, shelf and table scenes recorded from 52 camera frames
each, with 6D ground-truth poses, per-object visibility and analytic 6-DoF grasp
annotations.

A scene is the relational example: its own geometry and object catalogue, one
exchangeable part per object instance, and one per camera viewpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from krrood.entity_query_language.factories import entity, count_range, or_, variable
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)
from typing_extensions import Generic, List, Self, Sequence, Tuple, TypeVar

from experiments.causal_reasoning.comparison.domain import RelationalDomain

# %% reading a measurement as a level


LevelType = TypeVar("LevelType", bound=StrEnum)


@dataclass(frozen=True)
class MeasurementLevels(Generic[LevelType]):
    """
    The levels a continuous measurement is read as, and the boundaries between them.

    An aggregation counts the parts matching a condition, and a condition on a part a
    query leaves open can only compare for equality, so every measurement an aggregation
    filters on needs a discrete counterpart. The boundaries come from the dataset's own
    distribution rather than from chosen cutoffs, so each level holds an equal share of
    the values it was built from.
    """

    levels: Tuple[LevelType, ...]
    """
    The levels, from the lowest measurement to the highest.
    """

    boundaries: Tuple[float, ...]
    """
    The measurement at which one level gives way to the next, one fewer than
    :attr:`levels`; a measurement equal to a boundary falls in the higher level.
    """

    @classmethod
    def from_values(cls, values: Sequence[float], levels: Sequence[LevelType]) -> Self:
        """
        :param values: The measurements the boundaries are read off.
        :param levels: The levels to split them into, from the lowest to the highest.
        :return: The scale splitting the values into equally sized levels.
        """
        shares = np.linspace(0, 100, len(levels) + 1)[1:-1]
        return cls(
            levels=tuple(levels),
            boundaries=tuple(float(bound) for bound in np.percentile(values, shares)),
        )

    def level_of(self, value: float) -> LevelType:
        """
        :param value: A measurement.
        :return: The level it falls in.
        """
        return self.levels[int(np.searchsorted(self.boundaries, value, side="right"))]


# %% what the dataset records


class ObjectSize(StrEnum):
    """
    How large an object is, read off its model's diameter.
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Occlusion(StrEnum):
    """
    How much of an object the cameras of its scene actually see, read off its mean
    visible share over the scene's frames.
    """

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially occluded"
    HEAVILY_OCCLUDED = "heavily occluded"


class Graspability(StrEnum):
    """
    Whether an object of a scene can still be grasped where it lies.
    """

    GRASPABLE = "graspable"
    BLOCKED = "blocked"


class CameraModel(StrEnum):
    """
    The four cameras GraspClutter6D records every viewpoint of a scene with.
    """

    REALSENSE_D415 = "realsense-d415"
    REALSENSE_D435 = "realsense-d435"
    AZURE_KINECT = "azure-kinect"
    ZIVID = "zivid"

    @classmethod
    def of_frame(cls, image_id: int) -> Self:
        """
        :param image_id: A frame's number within its scene. The dataset numbers its
            frames ``4 * viewpoint + camera`` from one, with the cameras in the order
            this enum lists them.
        :return: The camera that recorded the frame.
        """
        return list(cls)[(image_id - 1) % len(cls)]


class Proximity(StrEnum):
    """
    How close a viewpoint's camera stands to the objects it looks at.
    """

    NEAR = "near"
    FAR = "far"


class ViewClarity(StrEnum):
    """
    How much of the scene's objects one viewpoint sees unobstructed.
    """

    CLEAR = "clear"
    OBSTRUCTED = "obstructed"


class ObjectCatalogue(StrEnum):
    """
    Which of the dataset's two object catalogues a scene's objects are drawn from.
    """

    GRASP = "grasp"
    """
    The dataset's own 200 novel objects only.
    """

    YCB_VIDEO = "ycb-video"
    """
    The standard YCB-Video objects only.
    """

    MIXED = "mixed"
    """
    Objects of both catalogues.
    """


# %% the scene and its exchangeable parts


@dataclass
class GraspClutterObject:
    """
    One object instance of a :class:`GraspClutterScene`, as an exchangeable part: the
    scene lists its instances in the order its annotation file happens to, and nothing
    ties a position in that list to an identity.
    """

    size: ObjectSize
    """
    How large the object is.
    """

    diameter: float
    """
    The diameter of the object's model, in meters.
    """

    visibility: float
    """
    The share of the object the cameras see, averaged over every frame of the scene, as
    recorded by the dataset's ``visib_fract``.
    """

    occlusion: Occlusion
    """
    The level :attr:`visibility` falls in.
    """

    graspability: Graspability
    """
    Whether any of the object's annotated antipodal grasps survives the rest of the
    scene, which is what makes graspability a property of the clutter rather than of the
    object alone.
    """


@dataclass
class GraspClutterViewpoint:
    """
    One annotated camera frame of a :class:`GraspClutterScene`, as an exchangeable part.

    Every scene is recorded from thirteen poses by four cameras, and the frames carry no
    order beyond the numbering of the files.
    """

    camera: CameraModel
    """
    The camera that recorded the frame.
    """

    distance: float
    """
    Distance from the camera to the objects' centre, in meters.
    """

    proximity: Proximity
    """
    The level :attr:`distance` falls in.
    """

    clarity: ViewClarity
    """
    Whether this frame sees the scene's objects clearly, read off the mean visible share
    of the objects in it.
    """


@dataclass
class GraspClutterScene:
    """
    One scene of the GraspClutter6D dataset, with its object instances and camera
    viewpoints as exchangeable parts.
    """

    object_catalogue: ObjectCatalogue
    """
    Which object catalogues the scene draws from.
    """

    extent: float
    """
    The diagonal of the box bounding the objects' positions, in meters: how far the
    clutter is spread out.
    """

    height_span: float
    """
    The vertical spread of the objects' positions, in meters: how far the clutter is
    stacked or shelved.
    """

    all_objects_graspable: bool
    """
    Whether every object instance of the scene keeps at least one grasp; the outcome the
    causal questions ask about.
    """

    objects: List[GraspClutterObject]
    """
    The scene's object instances.
    """

    viewpoints: List[GraspClutterViewpoint]
    """
    The scene's camera frames.
    """


@dataclass
class GraspClutterSceneAggregations(AggregationStatistic[GraspClutterScene]):
    """
    Aggregation statistics for :class:`GraspClutterScene` over its ``objects`` and
    ``viewpoints`` fields.
    """

    @aggregation_statistic("objects")
    def object_count(self) -> int:
        """
        Count of object instances, of any size.

        A scene with more objects holds more small ones and has more chances of one
        losing every grasp, so it confounds every question about a count.
        """
        size_variable = variable(GraspClutterObject, self.instance.objects).size
        [result] = (
            entity(count_range(size_variable))
            .where(or_(*(size_variable == size for size in ObjectSize)))
            .tolist()
        )
        return result

    @aggregation_statistic("objects")
    def small_object_count(self) -> int:
        """
        Count of small objects.
        """
        size_variable = variable(GraspClutterObject, self.instance.objects).size
        [result] = (
            entity(count_range(size_variable))
            .where(size_variable == ObjectSize.SMALL)
            .tolist()
        )
        return result

    @aggregation_statistic("objects")
    def occluded_object_count(self) -> int:
        """
        Count of objects the cameras do not see whole.
        """
        occlusion_variable = variable(
            GraspClutterObject, self.instance.objects
        ).occlusion
        [result] = (
            entity(count_range(occlusion_variable))
            .where(
                or_(
                    occlusion_variable == Occlusion.PARTIALLY_OCCLUDED,
                    occlusion_variable == Occlusion.HEAVILY_OCCLUDED,
                )
            )
            .tolist()
        )
        return result

    @aggregation_statistic("viewpoints")
    def near_viewpoint_count(self) -> int:
        """
        Count of viewpoints taken from close to the objects.
        """
        proximity_variable = variable(
            GraspClutterViewpoint, self.instance.viewpoints
        ).proximity
        [result] = (
            entity(count_range(proximity_variable))
            .where(proximity_variable == Proximity.NEAR)
            .tolist()
        )
        return result

    @aggregation_statistic("viewpoints")
    def clear_viewpoint_count(self) -> int:
        """
        Count of viewpoints that see the scene's objects clearly.
        """
        clarity_variable = variable(
            GraspClutterViewpoint, self.instance.viewpoints
        ).clarity
        [result] = (
            entity(count_range(clarity_variable))
            .where(clarity_variable == ViewClarity.CLEAR)
            .tolist()
        )
        return result


# %% the scene as a relational example


class PartField(StrEnum):
    """
    The scene's exchangeable-part fields.
    """

    OBJECTS = "objects"
    VIEWPOINTS = "viewpoints"


def scene_domain() -> RelationalDomain:
    """
    :return: The scene as the comparison sees it: its objects and viewpoints as
        exchangeable parts, whether every object stays graspable as the effect.
    """
    return RelationalDomain(
        example_class=GraspClutterScene,
        aggregation_class=GraspClutterSceneAggregations,
        effect_field="all_objects_graspable",
        noun="scene",
        plural="scenes",
        effect_phrase="every object stays graspable",
        part_nouns={PartField.OBJECTS: "object", PartField.VIEWPOINTS: "viewpoint"},
    )
