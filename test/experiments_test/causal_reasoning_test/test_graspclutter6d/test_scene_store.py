"""
Keeping built scenes in a database and reading them back.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import pytest

from experiments.causal_reasoning.graspclutter6d.dataset import (
    fetch_graspclutter_scenes,
)
from experiments.causal_reasoning.graspclutter6d.scene_store import (
    DEFAULT_DATABASE_FILE,
    SceneStore,
    SceneStoreVariable,
)


@pytest.fixture
def store(tmp_path) -> SceneStore:
    return SceneStore(database_uri=f"sqlite:///{tmp_path / 'scenes.sqlite'}")


def test_a_fresh_store_holds_nothing(store):
    assert store.scene_count == 0


def test_scenes_come_back_equal_and_in_order(store, synthetic_scenes):
    store.write(synthetic_scenes)
    assert store.scene_count == len(synthetic_scenes)
    assert store.read() == synthetic_scenes


def test_writing_again_replaces_what_the_store_held(store, synthetic_scenes):
    store.write(synthetic_scenes)
    store.write(synthetic_scenes[:3])
    assert store.read() == synthetic_scenes[:3]


def test_fetching_reads_the_store_when_it_holds_scenes(store, synthetic_scenes):
    store.write(synthetic_scenes)
    assert fetch_graspclutter_scenes(scene_limit=5, store=store) == synthetic_scenes[:5]


def test_the_environment_names_the_database(monkeypatch, tmp_path):
    monkeypatch.setenv(SceneStoreVariable.DATABASE_URI, "sqlite:///named.sqlite")
    assert SceneStore.from_environment(tmp_path).database_uri == "sqlite:///named.sqlite"


def test_without_a_name_the_database_sits_beside_the_dataset(monkeypatch, tmp_path):
    monkeypatch.delenv(SceneStoreVariable.DATABASE_URI, raising=False)
    assert SceneStore.from_environment(tmp_path).database_uri == (
        f"sqlite:///{tmp_path / DEFAULT_DATABASE_FILE}"
    )
