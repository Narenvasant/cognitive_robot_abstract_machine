"""
A structural causal model over the scene domain, with the interventional probabilities
it implies computed by construction.

The model exists so that a pipeline's causal answer can be marked *wrong*, not merely
different from another pipeline's. Its mechanism is one function that samples every
variable of a scene from its parents, with any cause the caller forces set instead of
sampled; the forced version is the ``do`` operator, so the true interventional
probability of an effect is the rate of the effect over many forced samples and needs no
inference machinery.

The confounder is the number of objects in the scene: it drives how many of them are
small and how likely each is to keep a grasp, so it opens a backdoor between every count
and graspability that adjusting for it closes. The scenes list their small objects
first, so a column addressing an object by position is systematically misleading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import Any, Dict, List, Optional, Sequence, Tuple

from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    KnownTruth,
)
from experiments.causal_reasoning.comparison.queries import CausalQueryCase
from experiments.causal_reasoning.graspclutter6d.domain import (
    CameraModel,
    GraspClutterObject,
    GraspClutterScene,
    GraspClutterViewpoint,
    Graspability,
    ObjectCatalogue,
    ObjectSize,
    Occlusion,
    Proximity,
    ViewClarity,
)
from experiments.causal_reasoning.graspclutter6d.queries import (
    CatalogueCausesGraspability,
    CatalogueCausesOcclusion,
    CountCausesGraspability,
    OccludedObjectsCauseBlockedObject,
    SizeCausesBlockedObject,
)

# %% what can be intervened on and what is read off


class Cause(StrEnum):
    """
    The variables of the model an intervention can force.
    """

    SMALL_OBJECT_COUNT = "small_object_count"
    OCCLUDED_OBJECT_COUNT = "occluded_object_count"
    CLEAR_VIEWPOINT_COUNT = "clear_viewpoint_count"
    CATALOGUE = "object_catalogue"
    OBJECT_SIZE = "object size"


class Effect(StrEnum):
    """
    The events of the model a question asks the probability of.
    """

    ALL_OBJECTS_GRASPABLE = "every object graspable"
    OBJECT_HEAVILY_OCCLUDED = "an object heavily occluded"
    OBJECT_BLOCKED = "an object blocked"


@dataclass(frozen=True)
class Intervention:
    """
    One cause forced to one value.
    """

    cause: Cause
    """
    The variable forced.
    """

    value: Any
    """
    The value it is forced to: a count, a catalogue or a size.
    """


@dataclass(frozen=True)
class CausalQuestion:
    """
    What a query of the catalogue asks, in the model's own terms.
    """

    cause: Cause
    """
    The variable the query marks as its cause.
    """

    effect: Effect
    """
    The event the query asks the probability of.
    """

    @classmethod
    def of(cls, case: CausalQueryCase) -> CausalQuestion:
        """
        :param case: A question of the catalogue.
        :return: The cause it forces and the effect it reads, in the model's terms.
        """
        if isinstance(case, CountCausesGraspability):
            return cls(Cause(case.statistic_name), Effect.ALL_OBJECTS_GRASPABLE)
        if isinstance(case, CatalogueCausesGraspability):
            return cls(Cause.CATALOGUE, Effect.ALL_OBJECTS_GRASPABLE)
        if isinstance(case, CatalogueCausesOcclusion):
            return cls(Cause.CATALOGUE, Effect.OBJECT_HEAVILY_OCCLUDED)
        if isinstance(case, OccludedObjectsCauseBlockedObject):
            return cls(Cause.OCCLUDED_OBJECT_COUNT, Effect.OBJECT_BLOCKED)
        if isinstance(case, SizeCausesBlockedObject):
            return cls(Cause.OBJECT_SIZE, Effect.OBJECT_BLOCKED)
        raise NotImplementedError(f"The model has no reading of {type(case).__name__}")


# %% the sampled scenes as arrays


@dataclass
class SampledScenes:
    """
    Many scenes sampled from the model at once, one row per scene, with the objects and
    viewpoints of every scene padded to the same width; a padded position is masked out.
    """

    object_count: np.ndarray
    """
    Per scene, how many objects it holds.
    """

    catalogue_is_grasp: np.ndarray
    """
    Per scene, whether it is built from the dataset's own catalogue rather than the YCB-
    Video one.
    """

    extent: np.ndarray
    """
    Per scene, how far its clutter is spread, in meters.
    """

    object_present: np.ndarray
    """
    Per scene and object position, whether the position holds an object.
    """

    object_small: np.ndarray
    """
    Per scene and object position, whether the object is small rather than large.
    """

    object_occluded: np.ndarray
    """
    Per scene and object position, whether the object is heavily occluded.
    """

    object_graspable: np.ndarray
    """
    Per scene and object position, whether the object keeps a grasp.
    """

    viewpoint_clear: np.ndarray
    """
    Per scene and viewpoint, whether the viewpoint sees the objects clearly.
    """

    picked: Optional[np.ndarray] = None
    """
    Per scene, the position of the object an object-level intervention was forced on, if
    one was.
    """

    @property
    def scene_count(self) -> int:
        """
        How many scenes were sampled.
        """
        return len(self.object_count)

    def effect_rate(self, effect: Effect, random_state: np.random.Generator) -> float:
        """
        :param effect: The event to read.
        :param random_state: Source of randomness for picking the object an object-level
            effect is read off, one at random per scene, since every object is
            exchangeable.
        :return: The share of scenes in which the event holds.
        """
        if effect is Effect.ALL_OBJECTS_GRASPABLE:
            return float(
                np.all(self.object_graspable | ~self.object_present, axis=1).mean()
            )
        picked = (random_state.random(self.scene_count) * self.object_count).astype(int)
        rows = np.arange(self.scene_count)
        if effect is Effect.OBJECT_HEAVILY_OCCLUDED:
            return float(self.object_occluded[rows, picked].mean())
        return float((~self.object_graspable[rows, picked]).mean())


# %% the model


@dataclass
class SceneStructuralCausalModel:
    """
    The mechanism every scene is sampled from.

    Per scene, the number of objects comes first and everything else follows from it:
    the catalogue, the spread, how many objects are small, how many are occluded, and
    whether each keeps a grasp, which a small, visible object of the dataset's own
    catalogue in a scene with few objects does most often. The viewpoints' clarity
    follows the object count too but nothing follows from it, so a count of clear
    viewpoints has no effect on anything; a pipeline that finds one is wrong.
    """

    object_count_centre: int = 10
    """
    The typical number of objects in a scene; the count is spread evenly between half
    and one and a half times it.
    """

    confounding_strength: float = 0.6
    """
    How far, between 0 and 1, the number of objects moves the share of small objects and
    each object's chance of keeping a grasp across its range.
    """

    viewpoint_count: int = 8
    """
    How many viewpoints every scene has.
    """

    small_share: float = 0.5
    """
    The share of small objects in a scene of typical size.
    """

    blocking_rate_small: float = 0.03
    """
    How often a small, visible object of the dataset's catalogue loses every grasp in a
    scene of ten objects; see :attr:`blocking_scale`.
    """

    blocking_rate_large: float = 0.12
    """
    The same for a large object.
    """

    occlusion_penalty: float = 0.08
    """
    How much being heavily occluded adds to an object's chance of losing every grasp, in
    a scene of ten objects.
    """

    catalogue_penalty: float = 0.04
    """
    How much being built from the YCB-Video catalogue adds to every object's chance of
    losing every grasp, in a scene of ten objects.
    """

    confounding_penalty: float = 0.12
    """
    How much the most crowded scene adds to every object's chance of losing every grasp,
    at full confounding strength, in a scene of ten objects.
    """

    occluded_share_small: float = 0.2
    """
    How often a small object is heavily occluded.
    """

    occluded_share_large: float = 0.5
    """
    How often a large object is heavily occluded.
    """

    @property
    def blocking_scale(self) -> float:
        """
        What every object's chance of losing a grasp is multiplied by, so that a scene
        of typical size leaves every object graspable about as often whatever that
        size is: the chances are stated for a scene of ten objects and fall in
        proportion for a larger one.
        """
        return 10 / self.object_count_centre

    @property
    def object_count_range(self) -> range:
        """
        The numbers of objects a scene can hold.
        """
        return range(
            max(2, self.object_count_centre // 2), self.object_count_centre * 3 // 2 + 1
        )

    @property
    def max_object_count(self) -> int:
        """
        The most objects a scene can hold.
        """
        return self.object_count_range.stop - 1

    def _crowding(self, object_count: np.ndarray) -> np.ndarray:
        """
        :param object_count: Per scene, how many objects it holds.
        :return: Where each count sits between the fewest and the most, from -0.5 to
            0.5.
        """
        counts = self.object_count_range
        return (object_count - counts.start) / max(
            1, counts.stop - 1 - counts.start
        ) - 0.5

    def sample(
        self,
        scene_count: int,
        random_state: np.random.Generator,
        intervention: Optional[Intervention] = None,
    ) -> SampledScenes:
        """
        Sample scenes from the mechanism, with one cause forced if asked.

        A count is forced by choosing exactly that many objects or viewpoints at random
        to carry the property and the rest not to; a scene that cannot hold that many
        objects is left out, so the forced count is read over the scenes it is possible
        in. A size is forced on the one object an object-level effect is read off.

        :param scene_count: How many scenes to sample.
        :param random_state: Source of randomness.
        :param intervention: The cause to force, if any.
        :return: The scenes.
        """
        counts = self.object_count_range
        object_count = random_state.integers(counts.start, counts.stop, scene_count)
        if intervention is not None and intervention.cause is Cause.SMALL_OBJECT_COUNT:
            object_count = object_count[object_count >= intervention.value]
            scene_count = len(object_count)
        if (
            intervention is not None
            and intervention.cause is Cause.OCCLUDED_OBJECT_COUNT
        ):
            object_count = object_count[object_count >= intervention.value]
            scene_count = len(object_count)
        crowding = self._crowding(object_count)
        width = self.max_object_count
        rows = np.arange(scene_count)
        positions = np.arange(width)[None, :]
        object_present = positions < object_count[:, None]

        catalogue_is_grasp = random_state.random(scene_count) < (
            0.5 - self.confounding_strength * crowding
        )
        if intervention is not None and intervention.cause is Cause.CATALOGUE:
            catalogue_is_grasp = np.full(
                scene_count, intervention.value is ObjectCatalogue.GRASP
            )
        extent = np.clip(
            0.45 + 0.4 * crowding + random_state.normal(0.0, 0.05, scene_count),
            0.1,
            1.0,
        )

        small_probability = np.clip(
            self.small_share + self.confounding_strength * crowding, 0.05, 0.95
        )
        object_small = (
            random_state.random((scene_count, width)) < small_probability[:, None]
        )
        if intervention is not None and intervention.cause is Cause.SMALL_OBJECT_COUNT:
            object_small = self._exactly(
                intervention.value, object_count, width, random_state
            )
        object_small &= object_present
        # small objects are listed first: the listing order is a function of the data
        order = np.argsort(~object_small & object_present, axis=1, kind="stable")
        object_small = np.take_along_axis(object_small, order, axis=1)

        occluded_probability = np.where(
            object_small, self.occluded_share_small, self.occluded_share_large
        )
        object_occluded = (
            random_state.random((scene_count, width)) < occluded_probability
        )
        if (
            intervention is not None
            and intervention.cause is Cause.OCCLUDED_OBJECT_COUNT
        ):
            object_occluded = self._exactly(
                intervention.value, object_count, width, random_state
            )
        object_occluded &= object_present

        picked = (random_state.random(scene_count) * object_count).astype(int)
        if intervention is not None and intervention.cause is Cause.OBJECT_SIZE:
            object_small[rows, picked] = intervention.value is ObjectSize.SMALL

        blocking_probability = self.blocking_scale * (
            np.where(object_small, self.blocking_rate_small, self.blocking_rate_large)
            + self.occlusion_penalty * object_occluded
            + self.catalogue_penalty * ~catalogue_is_grasp[:, None]
            + self.confounding_penalty
            * self.confounding_strength
            * (crowding[:, None] + 0.5)
        )
        object_graspable = random_state.random((scene_count, width)) >= np.clip(
            blocking_probability, 0.0, 0.98
        )
        object_graspable &= object_present

        viewpoint_clear = random_state.random((scene_count, self.viewpoint_count)) < (
            0.6 - 0.5 * crowding[:, None]
        )
        if (
            intervention is not None
            and intervention.cause is Cause.CLEAR_VIEWPOINT_COUNT
        ):
            viewpoint_clear = self._exactly(
                intervention.value,
                np.full(scene_count, self.viewpoint_count),
                self.viewpoint_count,
                random_state,
            )

        return SampledScenes(
            object_count=object_count,
            catalogue_is_grasp=catalogue_is_grasp,
            extent=extent,
            object_present=object_present,
            object_small=object_small,
            object_occluded=object_occluded,
            object_graspable=object_graspable,
            viewpoint_clear=viewpoint_clear,
            picked=(
                picked
                if intervention is not None and intervention.cause is Cause.OBJECT_SIZE
                else None
            ),
        )

    @staticmethod
    def _exactly(
        count: int,
        present_count: np.ndarray,
        width: int,
        random_state: np.random.Generator,
    ) -> np.ndarray:
        """
        :param count: How many positions of every row to mark.
        :param present_count: Per row, how many positions hold a part.
        :param width: How many positions a row has.
        :param random_state: Source of randomness for which positions are marked.
        :return: Per row, exactly ``count`` of its held positions marked, chosen at
            random.
        """
        held = np.arange(width)[None, :] < present_count[:, None]
        scores = np.where(
            held, random_state.random((len(present_count), width)), np.inf
        )
        ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
        return ranks < count

    def interventional_probability(
        self,
        intervention: Intervention,
        effect: Effect,
        random_state: np.random.Generator,
        sample_count: int = 200_000,
    ) -> float:
        """
        The probability of the effect under the intervention, read off forced samples.

        :param intervention: The cause and the value it is forced to.
        :param effect: The event to read.
        :param random_state: Source of randomness.
        :param sample_count: How many scenes to sample.
        :return: The share of forced scenes in which the effect holds.
        """
        sampled = self.sample(sample_count, random_state, intervention)
        if sampled.picked is None:
            return sampled.effect_rate(effect, random_state)
        rows = np.arange(sampled.scene_count)
        if effect is Effect.OBJECT_BLOCKED:
            return float((~sampled.object_graspable[rows, sampled.picked]).mean())
        return float(sampled.object_occluded[rows, sampled.picked].mean())

    def scenes(
        self, scene_count: int, random_state: np.random.Generator
    ) -> List[GraspClutterScene]:
        """
        Sample scenes and build them as domain objects, with the small objects listed
        first.

        :param scene_count: How many scenes to build.
        :param random_state: Source of randomness.
        :return: The scenes.
        """
        sampled = self.sample(scene_count, random_state)
        scenes = []
        for row in range(scene_count):
            objects = [
                self._object(sampled, row, position, random_state)
                for position in range(sampled.object_count[row])
            ]
            scenes.append(
                GraspClutterScene(
                    object_catalogue=(
                        ObjectCatalogue.GRASP
                        if sampled.catalogue_is_grasp[row]
                        else ObjectCatalogue.YCB_VIDEO
                    ),
                    extent=float(sampled.extent[row]),
                    height_span=float(random_state.uniform(0.0, 0.3)),
                    all_objects_graspable=all(
                        one.graspability is Graspability.GRASPABLE for one in objects
                    ),
                    objects=objects,
                    viewpoints=[
                        self._viewpoint(bool(clear), random_state)
                        for clear in sampled.viewpoint_clear[row]
                    ],
                )
            )
        return scenes

    @staticmethod
    def _object(
        sampled: SampledScenes,
        row: int,
        position: int,
        random_state: np.random.Generator,
    ) -> GraspClutterObject:
        """
        :param sampled: The sampled scenes.
        :param row: The scene.
        :param position: The object's position in the scene's listing.
        :param random_state: Source of randomness for the continuous attributes, which
            follow the discrete ones and drive nothing.
        :return: The object.
        """
        small = bool(sampled.object_small[row, position])
        occluded = bool(sampled.object_occluded[row, position])
        return GraspClutterObject(
            size=ObjectSize.SMALL if small else ObjectSize.LARGE,
            diameter=float(
                random_state.uniform(0.05, 0.15)
                if small
                else random_state.uniform(0.3, 0.4)
            ),
            visibility=float(
                random_state.uniform(0.0, 0.3)
                if occluded
                else random_state.uniform(0.7, 1.0)
            ),
            occlusion=Occlusion.HEAVILY_OCCLUDED if occluded else Occlusion.VISIBLE,
            graspability=(
                Graspability.GRASPABLE
                if sampled.object_graspable[row, position]
                else Graspability.BLOCKED
            ),
        )

    @staticmethod
    def _viewpoint(
        clear: bool, random_state: np.random.Generator
    ) -> GraspClutterViewpoint:
        """
        :param clear: Whether the viewpoint sees the objects clearly.
        :param random_state: Source of randomness for the camera and the distance.
        :return: The viewpoint.
        """
        return GraspClutterViewpoint(
            camera=CameraModel.of_frame(int(random_state.integers(1, 53))),
            distance=float(
                random_state.uniform(0.5, 0.8)
                if clear
                else random_state.uniform(1.0, 1.5)
            ),
            proximity=Proximity.NEAR if clear else Proximity.FAR,
            clarity=ViewClarity.CLEAR if clear else ViewClarity.OBSTRUCTED,
        )


# %% the model as known truth


@dataclass(frozen=True)
class ModelSetting:
    """
    One setting of the synthetic model.
    """

    object_count_centre: int
    """
    The typical number of objects in a scene.
    """

    confounding_strength: float
    """
    How strongly the number of objects drives both the causes and the effect.
    """

    def model(self) -> SceneStructuralCausalModel:
        """
        :return: The model under this setting.
        """
        return SceneStructuralCausalModel(
            object_count_centre=self.object_count_centre,
            confounding_strength=self.confounding_strength,
        )


def region_value(effect: InterventionalEffect, cause: Cause) -> Any:
    """
    :param effect: One region's effect, whose region is written out.
    :param cause: The cause the region is a value of.
    :return: The value the region names, as the model forces it.
    """
    if cause is Cause.CATALOGUE:
        return ObjectCatalogue(effect.cause_region)
    if cause is Cause.OBJECT_SIZE:
        return ObjectSize(effect.cause_region)
    return int(effect.ordinal)


@dataclass
class SceneTruth(KnownTruth):
    """
    The synthetic model's interventional probabilities under several settings, each
    computed once per intervention and effect.
    """

    settings: Sequence[ModelSetting] = (
        ModelSetting(5, 0.6),
        ModelSetting(10, 0.0),
        ModelSetting(10, 0.3),
        ModelSetting(10, 0.6),
        ModelSetting(20, 0.6),
    )
    """
    The settings to run.
    """

    random_seed: int = 0
    """
    Seed of the forced samples, the same for every intervention.
    """

    sample_count: int = 200_000
    """
    How many forced scenes each probability is read off.
    """

    known: Dict[Tuple[ModelSetting, Intervention, Effect], float] = field(
        default_factory=dict
    )
    """
    The probabilities computed so far.
    """

    @property
    def configurations(self) -> Sequence[ModelSetting]:
        return self.settings

    def describe(self, configuration: ModelSetting) -> Dict[str, str]:
        return {
            "objects per scene": str(configuration.object_count_centre),
            "confounding strength": f"{configuration.confounding_strength:.1f}",
        }

    def examples(
        self, configuration: ModelSetting, count: int, random_state: np.random.Generator
    ) -> List[GraspClutterScene]:
        return configuration.model().scenes(count, random_state)

    def probability(
        self,
        configuration: ModelSetting,
        case: CausalQueryCase,
        effect: InterventionalEffect,
    ) -> float:
        question = CausalQuestion.of(case)
        intervention = Intervention(
            question.cause, region_value(effect, question.cause)
        )
        key = (configuration, intervention, question.effect)
        if key not in self.known:
            self.known[key] = configuration.model().interventional_probability(
                intervention,
                question.effect,
                np.random.default_rng(self.random_seed),
                self.sample_count,
            )
        return self.known[key]


def scenes_of_size(
    object_count_centre: int, scene_count: int, random_state: np.random.Generator
) -> List[GraspClutterScene]:
    """
    :param object_count_centre: The typical number of objects per scene.
    :param scene_count: How many scenes to sample.
    :param random_state: Source of randomness.
    :return: Scenes sampled from the model at that size.
    """
    return SceneStructuralCausalModel(object_count_centre=object_count_centre).scenes(
        scene_count, random_state
    )
