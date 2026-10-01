"""
Fixtures shared by the GraspClutter6D causal-reasoning tests.

Everything here runs without the dataset: the scenes are the synthetic generator's, and
the annotations are one scene of the real dataset trimmed to three frames and four object
instances and checked in under ``dataset``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.causal_reasoning.graspclutter6d.annotations import (
    ExtractedScenes,
    SceneAnnotations,
)
from experiments.causal_reasoning.comparison.dataset import ExampleDataset
from experiments.causal_reasoning.graspclutter6d.dataset import (
    graspclutter_dataset,
    synthetic_graspclutter_scenes,
)

SAMPLE_SCENE_ID = "000000"
"""
The scene the checked-in annotations were trimmed from.
"""


@pytest.fixture
def scenes_directory() -> Path:
    """
    The checked-in ``scenes`` directory, laid out the way the extracted dataset is.
    """
    return Path(__file__).parent / "dataset" / "scenes"


@pytest.fixture
def sample_annotations(scenes_directory) -> SceneAnnotations:
    """
    The checked-in scene's annotations.
    """
    return ExtractedScenes(directory=scenes_directory).scene(SAMPLE_SCENE_ID)


SYNTHETIC_SCENE_COUNT = 40
"""
How many synthetic scenes the tests fit on.
"""


@pytest.fixture
def synthetic_scenes():
    """
    Synthetic scenes with a graspability relation baked in.
    """
    return synthetic_graspclutter_scenes(
        np.random.default_rng(0), scene_count=SYNTHETIC_SCENE_COUNT
    )


@pytest.fixture
def synthetic_dataset(synthetic_scenes) -> ExampleDataset:
    """
    The synthetic scenes as a dataset.
    """
    return graspclutter_dataset(synthetic_scenes)


@pytest.fixture(scope="module")
def shared_synthetic_dataset() -> ExampleDataset:
    """
    The same scenes, fitted on once per module by the tests that compare pipelines.
    """
    return graspclutter_dataset(
        synthetic_graspclutter_scenes(
            np.random.default_rng(0), scene_count=SYNTHETIC_SCENE_COUNT
        )
    )
