"""
Building :class:`~experiments.causal_reasoning.graspclutter6d.domain.GraspClutterScene`
instances out of the dataset's annotations, object models and grasp labels, plus a
synthetic generator of the same shape for tests that must run without the dataset.

The levels the scenes' measurements are read as come from the dataset itself: object
sizes from the catalogue's diameters, occlusion from the visibility the cameras actually
record, and a viewpoint's proximity and clarity from the frames themselves. Reading the
annotations of every scene once is cheap; reading the grasp labels is not, so their
counts are kept in an index beside the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.loader import (
    GraspClutter6DDatasetLoader,
    GraspClutter6DModelVariant,
    GraspClutter6DObjectSet,
    GraspClutter6DSplit,
)
from typing_extensions import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeVar,
)

from experiments.causal_reasoning.graspclutter6d.annotations import (
    MILLIMETERS_PER_METER,
    SceneAnnotationSource,
    SceneAnnotations,
    scene_annotation_source,
)
from experiments.causal_reasoning.graspclutter6d.domain import (
    CameraModel,
    GraspClutterObject,
    GraspClutterScene,
    GraspClutterViewpoint,
    Graspability,
    MeasurementLevels,
    ObjectCatalogue,
    ObjectSize,
    Occlusion,
    Proximity,
    ViewClarity,
)
from experiments.causal_reasoning.graspclutter6d.grasp_labels import (
    GraspCountIndex,
    GraspLabels,
)
from experiments.causal_reasoning.graspclutter6d.scene_store import SceneStore

T = TypeVar("T")

MODELS_INFO_FILE = "models_info.json"
"""
The file an extracted models directory describes every object model in.
"""

GRASP_COUNT_INDEX_FILE = "grasp_count_index.json"
"""
The file the grasps read off the labels are kept in, beside the dataset.
"""

# %% the object catalogue


class ModelInfoField(StrEnum):
    """
    The fields the dataset's ``models_info.json`` gives each object model.
    """

    DIAMETER = "diameter"


@dataclass(frozen=True)
class ObjectModels:
    """
    The dataset's object models, as much of them as a scene's objects are described by.
    """

    diameters: Dict[int, float]
    """
    Per object model id, the diameter of its mesh, in meters.
    """

    @classmethod
    def read(cls, models_directory: Path) -> Self:
        """
        :param models_directory: An extracted models directory, holding
            ``models_info.json``.
        :return: The models it describes.
        """
        info = json.loads((models_directory / MODELS_INFO_FILE).read_text())
        return cls(
            diameters={
                int(object_id): float(entry[ModelInfoField.DIAMETER])
                / MILLIMETERS_PER_METER
                for object_id, entry in info.items()
            }
        )

    def size_levels(self) -> MeasurementLevels[ObjectSize]:
        """
        :return: The levels an object's diameter is read as, split so that each holds a
            third of the catalogue.
        """
        return MeasurementLevels.from_values(
            list(self.diameters.values()), list(ObjectSize)
        )


# %% which catalogues a scene draws from


@dataclass(frozen=True)
class SceneCatalogues:
    """
    Which of the dataset's two object catalogues each scene draws from, read off the
    split files.
    """

    scene_ids_by_catalogue: Dict[GraspClutter6DObjectSet, FrozenSet[str]]
    """
    Per catalogue, the ids of the scenes listed under it.
    """

    @classmethod
    def read(cls, loader: GraspClutter6DDatasetLoader) -> Self:
        """
        :param loader: The loader holding the dataset's split files.
        :return: What the split files say about every scene.
        """
        return cls(
            scene_ids_by_catalogue={
                object_set: frozenset(
                    scene_id
                    for split in GraspClutter6DSplit
                    for scene_id in loader.available_scene_ids(object_set, split)
                )
                for object_set in GraspClutter6DObjectSet
            }
        )

    @property
    def scene_ids(self) -> List[str]:
        """
        Every scene listed under either catalogue, in ascending id.
        """
        return sorted(frozenset().union(*self.scene_ids_by_catalogue.values()))

    def catalogue_of(self, scene_id: str) -> ObjectCatalogue:
        """
        :param scene_id: One of :attr:`scene_ids`.
        :return: Which catalogues it draws from.
        """
        listed = {
            object_set
            for object_set, scene_ids in self.scene_ids_by_catalogue.items()
            if scene_id in scene_ids
        }
        if len(listed) > 1:
            return ObjectCatalogue.MIXED
        if listed == {GraspClutter6DObjectSet.YCBV}:
            return ObjectCatalogue.YCB_VIDEO
        return ObjectCatalogue.GRASP


# %% the levels the scenes' measurements are read as


@dataclass(frozen=True)
class SceneLevels:
    """
    How every continuous measurement of a scene is read as a level.
    """

    size: MeasurementLevels[ObjectSize]
    """
    An object's diameter.
    """

    occlusion: MeasurementLevels[Occlusion]
    """
    An object's mean visible share over the scene's frames.
    """

    proximity: MeasurementLevels[Proximity]
    """
    A viewpoint's distance to the objects.
    """

    clarity: MeasurementLevels[ViewClarity]
    """
    The mean visible share of the objects in one frame.
    """

    @classmethod
    def from_annotations(
        cls, models: ObjectModels, annotations: Sequence[SceneAnnotations]
    ) -> Self:
        """
        :param models: The dataset's object models.
        :param annotations: The annotations of every scene the dataset is built from.
        :return: The levels those scenes split into.
        """
        visibilities = np.concatenate(
            [scene.mean_visible_fractions() for scene in annotations]
        )
        distances = [
            frame.distance_to_objects for scene in annotations for frame in scene.frames
        ]
        visible_shares = [
            frame.visible_share for scene in annotations for frame in scene.frames
        ]
        return cls(
            size=models.size_levels(),
            occlusion=MeasurementLevels.from_values(
                visibilities,
                (
                    Occlusion.HEAVILY_OCCLUDED,
                    Occlusion.PARTIALLY_OCCLUDED,
                    Occlusion.VISIBLE,
                ),
            ),
            proximity=MeasurementLevels.from_values(
                distances, (Proximity.NEAR, Proximity.FAR)
            ),
            clarity=MeasurementLevels.from_values(
                visible_shares, (ViewClarity.OBSTRUCTED, ViewClarity.CLEAR)
            ),
        )


# %% building the scenes


@dataclass
class SceneBuilder:
    """
    Turns one scene's annotations and grasp counts into a
    :class:`~experiments.causal_reasoning.graspclutter6d.domain.GraspClutterScene`.
    """

    models: ObjectModels
    """
    The dataset's object models.
    """

    catalogues: SceneCatalogues
    """
    Which catalogues each scene draws from.
    """

    levels: SceneLevels
    """
    How the scenes' measurements are read as levels.
    """

    def build(
        self, annotations: SceneAnnotations, grasp_counts: Sequence[int]
    ) -> GraspClutterScene:
        """
        :param annotations: What the scene's annotation files record.
        :param grasp_counts: How many grasps each of its object instances keeps, in the
            order the ground truth lists them.
        :return: The scene.
        """
        visibilities = annotations.mean_visible_fractions()
        extent, height_span = annotations.object_position_spread()
        return GraspClutterScene(
            object_catalogue=self.catalogues.catalogue_of(annotations.scene_id),
            extent=extent,
            height_span=height_span,
            all_objects_graspable=all(count > 0 for count in grasp_counts),
            objects=[
                self._object(object_id, visibility, count)
                for object_id, visibility, count in zip(
                    annotations.object_ids, visibilities, grasp_counts
                )
            ],
            viewpoints=[
                self._viewpoint(
                    frame.image_id, frame.distance_to_objects, frame.visible_share
                )
                for frame in annotations.frames
            ],
        )

    def _object(
        self, object_id: int, visibility: float, grasp_count: int
    ) -> GraspClutterObject:
        """
        :param object_id: The object model's id.
        :param visibility: The share of the object the cameras see.
        :param grasp_count: How many grasps it keeps in the scene.
        :return: The object instance.
        """
        diameter = self.models.diameters[object_id]
        return GraspClutterObject(
            size=self.levels.size.level_of(diameter),
            diameter=diameter,
            visibility=float(visibility),
            occlusion=self.levels.occlusion.level_of(visibility),
            graspability=(
                Graspability.GRASPABLE if grasp_count > 0 else Graspability.BLOCKED
            ),
        )

    def _viewpoint(
        self, image_id: int, distance: float, visible_share: float
    ) -> GraspClutterViewpoint:
        """
        :param image_id: The frame's number within the scene.
        :param distance: Distance from the camera to the objects, in meters.
        :param visible_share: The mean visible share of the objects in the frame.
        :return: The viewpoint.
        """
        return GraspClutterViewpoint(
            camera=CameraModel.of_frame(image_id),
            distance=distance,
            proximity=self.levels.proximity.level_of(distance),
            clarity=self.levels.clarity.level_of(visible_share),
        )


def build_graspclutter_scenes(
    scene_limit: Optional[int] = None,
    annotations: Optional[SceneAnnotationSource] = None,
    labels: Optional[GraspLabels] = None,
    loader: Optional[GraspClutter6DDatasetLoader] = None,
    index_path: Optional[Path] = None,
) -> List[GraspClutterScene]:
    """
    Build the dataset's scenes from its annotations, object models and grasp labels.

    :param scene_limit: Build only the first scenes of the dataset, in ascending id; all
        of them if not given.
    :param annotations: Where to read the scenes' annotation files from; whatever the
        environment offers if not given.
    :param labels: The extracted grasp and collision labels; the loader's own directory
        if not given.
    :param loader: The loader holding the dataset's split files and object models.
    :param index_path: Where the grasp counts read off the labels are kept.
    :return: One scene per scene id listed under either object catalogue.
    """
    loader = loader or GraspClutter6DDatasetLoader()
    annotations = annotations or scene_annotation_source()
    labels = labels or GraspLabels(directory=loader.directory)
    index_path = index_path or loader.directory / GRASP_COUNT_INDEX_FILE
    models = ObjectModels.read(loader.download_models(GraspClutter6DModelVariant.EVAL))
    catalogues = SceneCatalogues.read(loader)
    scene_ids = catalogues.scene_ids[:scene_limit]

    read = {scene_id: annotations.scene(scene_id) for scene_id in scene_ids}
    levels = SceneLevels.from_annotations(models, list(read.values()))
    builder = SceneBuilder(models=models, catalogues=catalogues, levels=levels)

    index = GraspCountIndex.read(index_path)
    scenes = [
        builder.build(scene, index.counts(labels, scene_id, scene.object_ids))
        for scene_id, scene in read.items()
    ]
    index.write(index_path)
    return scenes


def fetch_graspclutter_scenes(
    scene_limit: Optional[int] = None,
    store: Optional[SceneStore] = None,
    rebuild: bool = False,
    loader: Optional[GraspClutter6DDatasetLoader] = None,
) -> List[GraspClutterScene]:
    """
    Read the dataset's scenes, from the store if it holds them and by building them
    otherwise.

    The store holds every scene of the dataset; a limit is applied to what it returns.
    Building the scenes needs the annotations and the grasp labels and takes a while, so
    it happens once, and again only when asked.

    :param scene_limit: Return only the first scenes of the dataset, in ascending id;
        all of them if not given.
    :param store: Where the built scenes are kept; the store the environment names, or a
        database file beside the dataset, if not given.
    :param rebuild: Build the scenes afresh and replace what the store holds, which is
        what a change to the domain classes calls for.
    :param loader: The loader holding the dataset's split files and object models.
    :return: One scene per scene id listed under either object catalogue.
    """
    loader = loader or GraspClutter6DDatasetLoader()
    store = store or SceneStore.from_environment(loader.directory)
    if rebuild or store.scene_count == 0:
        store.write(build_graspclutter_scenes(loader=loader))
    return store.read()[:scene_limit]


# %% a synthetic stand-in


def synthetic_graspclutter_scenes(
    random_state: np.random.Generator,
    scene_count: int = 20,
    object_count: int = 3,
    viewpoint_count: int = 4,
) -> List[GraspClutterScene]:
    """
    Generate a small, dataset-free set of scenes with the same shape as
    :func:`fetch_graspclutter_scenes`, for tests that must run without the dataset.

    A scene holds a number of small objects that rises from none to all of them across
    the scenes, and the rest are large. A small object is usually visible and usually
    keeps its grasps, a large one usually neither, so the more of a scene is small the
    likelier it is to leave every object graspable, without the count settling it. The
    scenes' extent and their viewpoints follow the same divide, so that adjusting for
    either changes the answer, and the small objects are listed first, so that a column
    addressing one object by position means something different once the objects are
    reordered.

    :param random_state: Source of randomness for everything the relation does not fix.
    :param scene_count: How many scenes to generate.
    :param object_count: How many objects each scene has.
    :param viewpoint_count: How many viewpoints each scene has.
    :return: The generated scenes.
    """
    scenes = []
    for index in range(scene_count):
        small_count = index % (object_count + 1)
        mostly_small = small_count * 2 > object_count
        objects = [
            _synthetic_object(random_state, small=position < small_count)
            for position in range(object_count)
        ]
        scenes.append(
            GraspClutterScene(
                object_catalogue=_synthetic_catalogue(random_state, mostly_small),
                extent=float(random_state.uniform(0.2, 0.4 if mostly_small else 0.8)),
                height_span=float(random_state.uniform(0.0, 0.3)),
                all_objects_graspable=all(
                    one.graspability is Graspability.GRASPABLE for one in objects
                ),
                objects=objects,
                viewpoints=[
                    _synthetic_viewpoint(random_state, clear=mostly_small)
                    for _ in range(viewpoint_count)
                ],
            )
        )
    return scenes


USUAL_SHARE = 0.85
"""
How often a synthetic object behaves the way its size suggests, leaving the rest as the
noise a real dataset carries.
"""


def _synthetic_object(
    random_state: np.random.Generator, small: bool
) -> GraspClutterObject:
    """
    :param random_state: Source of randomness for the diameter, the visibility and
        whether the object behaves the way its size suggests.
    :param small: Whether the object is one of the scene's small ones.
    :return: The object.
    """
    visible = random_state.random() < (USUAL_SHARE if small else 1 - USUAL_SHARE)
    graspable = random_state.random() < (USUAL_SHARE if small else 1 - USUAL_SHARE)
    return GraspClutterObject(
        size=ObjectSize.SMALL if small else ObjectSize.LARGE,
        diameter=float(
            random_state.uniform(0.05, 0.15)
            if small
            else random_state.uniform(0.3, 0.4)
        ),
        visibility=float(
            random_state.uniform(0.8, 1.0)
            if visible
            else random_state.uniform(0.0, 0.2)
        ),
        occlusion=Occlusion.VISIBLE if visible else Occlusion.HEAVILY_OCCLUDED,
        graspability=(Graspability.GRASPABLE if graspable else Graspability.BLOCKED),
    )


def _synthetic_catalogue(
    random_state: np.random.Generator, mostly_small: bool
) -> ObjectCatalogue:
    """
    :param random_state: Source of randomness for the catalogue.
    :param mostly_small: Whether more than half of the scene's objects are small.
    :return: The catalogue the scene is built from, which follows the scene's size mix
        without being settled by it.
    """
    usual = ObjectCatalogue.GRASP if mostly_small else ObjectCatalogue.YCB_VIDEO
    other = ObjectCatalogue.YCB_VIDEO if mostly_small else ObjectCatalogue.GRASP
    return usual if random_state.random() < USUAL_SHARE else other


def _synthetic_viewpoint(
    random_state: np.random.Generator, clear: bool
) -> GraspClutterViewpoint:
    """
    :param random_state: Source of randomness for the camera and the distance.
    :param clear: Whether the viewpoint usually sees the scene's objects clearly.
    :return: The viewpoint.
    """
    clear = random_state.random() < (USUAL_SHARE if clear else 1 - USUAL_SHARE)
    return GraspClutterViewpoint(
        camera=CameraModel.of_frame(int(random_state.integers(1, 53))),
        distance=float(
            random_state.uniform(0.5, 0.8) if clear else random_state.uniform(1.0, 1.5)
        ),
        proximity=Proximity.NEAR if clear else Proximity.FAR,
        clarity=ViewClarity.CLEAR if clear else ViewClarity.OBSTRUCTED,
    )


# %% a set of scenes


@dataclass(frozen=True)
class GraspableRate:
    """
    How often a group of scenes leaves every object graspable.
    """

    scene_count: int
    """
    How many scenes the group holds.
    """

    graspable_count: int
    """
    How many of them leave every object graspable.
    """

    @property
    def rate(self) -> float:
        """
        The share of scenes that do.
        """
        return self.graspable_count / self.scene_count


@dataclass
class GraspClutterDataset:
    """
    A set of scenes, as read and as the pipelines see them.
    """

    scenes: List[GraspClutterScene] = field(default_factory=list)
    """
    The scenes.
    """

    @property
    def graspable_rate(self) -> float:
        """
        Share of scenes that leave every object graspable.
        """
        return sum(scene.all_objects_graspable for scene in self.scenes) / len(
            self.scenes
        )

    def graspable_rate_by(
        self, key: Callable[[GraspClutterScene], T]
    ) -> Dict[T, GraspableRate]:
        """
        How often the scenes sharing a value leave every object graspable.

        :param key: What to group the scenes by.
        :return: Each value's rate, by value.
        """
        by_value: Dict[T, List[GraspClutterScene]] = {}
        for scene in self.scenes:
            by_value.setdefault(key(scene), []).append(scene)
        return {
            value: GraspableRate(
                scene_count=len(scenes),
                graspable_count=sum(scene.all_objects_graspable for scene in scenes),
            )
            for value, scenes in sorted(by_value.items())
        }

    def with_shuffled_parts(self, random_state: np.random.Generator) -> Self:
        """
        The same scenes with their objects and viewpoints in a random order each.

        :param random_state: Source of randomness for the orders.
        :return: The dataset with reordered parts.
        """
        return type(self)(
            [
                replace(
                    scene,
                    objects=[
                        scene.objects[index]
                        for index in random_state.permutation(len(scene.objects))
                    ],
                    viewpoints=[
                        scene.viewpoints[index]
                        for index in random_state.permutation(len(scene.viewpoints))
                    ],
                )
                for scene in self.scenes
            ]
        )

    def split(
        self, train_fraction: float, random_state: np.random.Generator
    ) -> Tuple[Self, Self]:
        """
        Shuffle the scenes and split them in two.

        :param train_fraction: Share of scenes that go into the first part.
        :param random_state: Source of randomness for the shuffle.
        :return: The first and second part.
        """
        order = random_state.permutation(len(self.scenes))
        split_index = int(train_fraction * len(self.scenes))
        first = [self.scenes[index] for index in order[:split_index]]
        second = [self.scenes[index] for index in order[split_index:]]
        return type(self)(first), type(self)(second)
