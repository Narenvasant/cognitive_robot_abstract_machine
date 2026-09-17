"""
What each pipeline fits, which questions it can answer, and which it refuses.
"""

from __future__ import annotations

from dataclasses import replace

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import pytest

from experiments.causal_reasoning.graspclutter6d.exceptions import (
    FlatTableSchemaMismatchError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import (
    FlatTable,
    SceneView,
    TableLayout,
)
from experiments.causal_reasoning.graspclutter6d.pipelines import (
    CauseStratification,
    FlatTablePipeline,
    RelationalPipeline,
    pipelines,
)
from experiments.causal_reasoning.graspclutter6d.queries import (
    CatalogueCausesGraspability,
    CountCausesGraspability,
    SizeCausesBlockedObject,
)

LEAF_SHARE = 0.1
"""
The share of its training rows a leaf may hold in these tests, small enough for forty
synthetic scenes to still split.
"""


@pytest.fixture
def fitted_pipelines(synthetic_scenes):
    """
    Every pipeline fitted on the synthetic scenes.
    """
    fitted = pipelines(synthetic_scenes)
    for pipeline in fitted:
        pipeline.min_samples_per_leaf = LEAF_SHARE
        pipeline.plain_min_samples_per_leaf = LEAF_SHARE
        pipeline.fit(synthetic_scenes)
    return {pipeline.name: pipeline for pipeline in fitted}


def test_every_layout_and_the_relational_circuit_are_compared(synthetic_scenes):
    assert [pipeline.name for pipeline in pipelines(synthetic_scenes)] == [
        "relational circuit",
        f"{TableLayout.PROPOSITIONAL} tree",
        f"{TableLayout.UNROLLED} tree",
        f"{TableLayout.SCALARS} tree",
    ]


def test_a_pipeline_that_was_never_fitted_says_so():
    with pytest.raises(PipelineNotFittedError):
        RelationalPipeline().registry_for(None)


def test_fitting_reports_one_model_before_any_cause_is_asked_about(fitted_pipelines):
    for pipeline in fitted_pipelines.values():
        assert pipeline.fit_report.model_count == 1


def test_asking_about_a_cause_fits_one_further_model_once(fitted_pipelines):
    pipeline = fitted_pipelines["relational circuit"]
    cause_name = "GraspClutterSceneAggregations.small_object_count()"
    pipeline.registry_for(cause_name)
    after_first = pipeline.fit_report.model_count
    pipeline.registry_for(cause_name)
    assert (after_first, pipeline.fit_report.model_count) == (2, 2)


def test_only_the_pipelines_that_model_parts_say_so(fitted_pipelines):
    assert {
        name: pipeline.models_parts for name, pipeline in fitted_pipelines.items()
    } == {
        "relational circuit": True,
        f"{TableLayout.PROPOSITIONAL} tree": False,
        f"{TableLayout.UNROLLED} tree": True,
        f"{TableLayout.SCALARS} tree": False,
    }


def test_a_pipeline_without_the_counts_refuses_a_question_about_them(
    fitted_pipelines,
):
    pipeline = fitted_pipelines[f"{TableLayout.SCALARS} tree"]
    with pytest.raises(FlatTableSchemaMismatchError):
        pipeline.registry_for("GraspClutterSceneAggregations.small_object_count()")


def test_a_pipeline_without_the_parts_refuses_a_question_about_one(fitted_pipelines):
    pipeline = fitted_pipelines[f"{TableLayout.PROPOSITIONAL} tree"]
    with pytest.raises(FlatTableSchemaMismatchError):
        pipeline.registry_for("GraspClutterScene.objects[0].size")


def test_a_scene_level_cause_stratifies_the_class_circuit_not_a_template():
    stratification = CauseStratification.for_variable("GraspClutterScene.extent")
    assert stratification.class_columns == ["GraspClutterScene.extent"]
    assert stratification.part_attributes == {}


def test_a_part_level_cause_stratifies_that_parts_template_not_the_class_circuit():
    stratification = CauseStratification.for_variable(
        "GraspClutterScene.objects[3].size"
    )
    assert stratification.class_columns is None
    assert stratification.part_attributes == {"objects": ["size"]}


def test_only_the_pipelines_that_model_parts_score_a_whole_scene(
    fitted_pipelines, synthetic_scenes
):
    scored = {
        name: pipeline.log_likelihood(synthetic_scenes, SceneView.WHOLE_SCENE)
        is not None
        for name, pipeline in fitted_pipelines.items()
    }
    assert scored == {
        name: pipeline.models_parts for name, pipeline in fitted_pipelines.items()
    }


def test_every_pipeline_covers_the_scenes_it_was_fitted_on(
    fitted_pipelines, synthetic_scenes
):
    for pipeline in fitted_pipelines.values():
        report = pipeline.log_likelihood(synthetic_scenes, SceneView.SCALARS)
        assert report.coverage == 1.0


def test_a_scene_the_table_has_no_room_for_lies_outside_the_support(
    synthetic_scenes,
):
    narrow = FlatTablePipeline(
        flat_table=FlatTable.unrolled_for(synthetic_scenes),
        min_samples_per_leaf=LEAF_SHARE,
        plain_min_samples_per_leaf=LEAF_SHARE,
    )
    narrow.fit(synthetic_scenes)
    scene = synthetic_scenes[0]
    larger = replace(scene, objects=scene.objects + [scene.objects[0]])
    report = narrow.log_likelihood([larger], SceneView.WHOLE_SCENE)
    assert report.covered_scene_count == 0


def test_the_questions_name_the_variables_the_pipelines_are_asked_about():
    assert (
        CountCausesGraspability(
            statistic_name="small_object_count", count_noun="small objects"
        ).name
        == "small_object_count_causes_graspability"
    )
    assert (
        CatalogueCausesGraspability(
            confounder_name="extent", confounder_noun="spread"
        ).name
        == "catalogue_causes_graspability_adjusting_extent"
    )
    assert SizeCausesBlockedObject().name == "size_causes_blocked_object_0"
