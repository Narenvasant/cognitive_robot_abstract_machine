"""
What each pipeline fits, which questions it can answer, and which it refuses.
"""

from __future__ import annotations

from dataclasses import replace

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest

from experiments.causal_reasoning.comparison.domain import ExampleView
from experiments.causal_reasoning.comparison.evaluation import QuestionAsker
from experiments.causal_reasoning.comparison.exceptions import (
    FlatTableSchemaMismatchError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.comparison.flat_table import (
    FlatTable,
    Schema,
    TableLayout,
)
from experiments.causal_reasoning.comparison.pipelines import (
    CauseStratification,
    FlatTablePipeline,
    HybridPipeline,
    RelationalPipeline,
    pipelines,
)
from experiments.causal_reasoning.graspclutter6d.domain import PartField, scene_domain
from experiments.causal_reasoning.graspclutter6d.queries import (
    CatalogueCausesGraspability,
    CountCausesGraspability,
    SizeCausesBlockedObject,
    object_level_cases,
)

LEAF_SHARE = 0.1
"""
The share of its training rows a leaf may hold in these tests, small enough for forty
synthetic scenes to still split.
"""


@pytest.fixture
def schema() -> Schema:
    return Schema(scene_domain())


def scene_pipelines(synthetic_scenes):
    """
    :param synthetic_scenes: The scenes the pipelines will be fitted on.
    :return: Every pipeline the experiment compares, unfitted.
    """
    return pipelines(
        scene_domain(), synthetic_scenes, positional_fields=(PartField.VIEWPOINTS,)
    )


@pytest.fixture
def fitted_pipelines(synthetic_scenes):
    """
    Every pipeline fitted on the synthetic scenes.
    """
    fitted = scene_pipelines(synthetic_scenes)
    for pipeline in fitted:
        pipeline.min_samples_per_leaf = LEAF_SHARE
        pipeline.plain_min_samples_per_leaf = LEAF_SHARE
        pipeline.fit(synthetic_scenes)
    return {pipeline.name: pipeline for pipeline in fitted}


def test_every_layout_and_the_relational_circuit_are_compared(synthetic_scenes):
    assert [pipeline.name for pipeline in scene_pipelines(synthetic_scenes)] == [
        "relational circuit",
        "hybrid circuit",
        f"{TableLayout.PROPOSITIONAL} tree",
        f"{TableLayout.UNROLLED} tree",
        f"{TableLayout.SCALARS} tree",
    ]


def test_a_pipeline_that_was_never_fitted_says_so():
    with pytest.raises(PipelineNotFittedError):
        RelationalPipeline(domain=scene_domain()).registry_for(None)


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
        "hybrid circuit": True,
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


def test_a_scene_level_cause_stratifies_the_class_circuit_not_a_template(schema):
    stratification = CauseStratification.for_variable(
        "GraspClutterScene.extent", schema
    )
    assert stratification.class_columns == ["GraspClutterScene.extent"]
    assert stratification.part_attributes == {}


def test_a_part_level_cause_stratifies_that_parts_template_not_the_class_circuit(
    schema,
):
    stratification = CauseStratification.for_variable(
        "GraspClutterScene.objects[3].size", schema
    )
    assert stratification.class_columns is None
    assert stratification.part_attributes == {"objects": ["size"]}


def test_only_the_pipelines_that_model_parts_score_a_whole_scene(
    fitted_pipelines, synthetic_scenes
):
    scored = {
        name: pipeline.log_likelihood(synthetic_scenes, ExampleView.WHOLE) is not None
        for name, pipeline in fitted_pipelines.items()
    }
    assert scored == {
        name: pipeline.models_parts for name, pipeline in fitted_pipelines.items()
    }


def test_every_pipeline_covers_the_scenes_it_was_fitted_on(
    fitted_pipelines, synthetic_scenes
):
    for pipeline in fitted_pipelines.values():
        report = pipeline.log_likelihood(synthetic_scenes, ExampleView.SCALARS)
        assert report.coverage == 1.0


def test_a_scene_the_table_has_no_room_for_lies_outside_the_support(
    synthetic_scenes, schema
):
    narrow = FlatTablePipeline(
        domain=scene_domain(),
        layout=TableLayout.UNROLLED,
        part_widths=FlatTable.unrolled_for(schema, synthetic_scenes).part_widths,
        min_samples_per_leaf=LEAF_SHARE,
        plain_min_samples_per_leaf=LEAF_SHARE,
    )
    narrow.fit(synthetic_scenes)
    scene = synthetic_scenes[0]
    larger = replace(scene, objects=scene.objects + [scene.objects[0]])
    report = narrow.log_likelihood([larger], ExampleView.WHOLE)
    assert report.covered_example_count == 0


def test_the_questions_name_the_variables_the_pipelines_are_asked_about():
    assert (
        CountCausesGraspability(
            statistic_name="small_object_count", count_noun="small objects"
        ).name
        == "small_object_count_causes_graspability_adjusting_extent"
    )
    assert (
        CatalogueCausesGraspability(
            confounder_name="extent", confounder_noun="spread"
        ).name
        == "catalogue_causes_graspability_adjusting_extent"
    )
    assert SizeCausesBlockedObject().name == "size_causes_blocked_object_0"


def test_the_relational_circuit_gives_the_same_answers_whatever_order_the_parts_come_in(
    synthetic_dataset,
):
    """
    Refitting on the same scenes with their parts reordered must give the same
    refusals, regions and probabilities, to floating-point precision: the circuit is
    the same, and only the order its sums are taken in can differ.
    """
    asker = QuestionAsker(random_seed=0, min_region_support=1)
    answers = []
    for ordering in range(3):
        reordered = synthetic_dataset.with_shuffled_parts(
            np.random.default_rng(ordering)
        )
        pipeline = RelationalPipeline(
            domain=scene_domain(),
            min_samples_per_leaf=0.2,
            plain_min_samples_per_leaf=0.2,
        )
        pipeline.fit(reordered.examples)
        outcomes = [asker.ask(pipeline, case) for case in object_level_cases()]
        answers.append(
            (
                [outcome.refusal for outcome in outcomes],
                [
                    [effect.cause_region for effect in outcome.effects]
                    for outcome in outcomes
                ],
                [
                    [effect.adjusted_probability for effect in outcome.effects]
                    for outcome in outcomes
                ],
                pipeline.log_likelihood(
                    reordered.examples, ExampleView.WHOLE
                ).mean_log_likelihood,
            )
        )
    first = answers[0]
    for refusals, regions, probabilities, likelihood in answers[1:]:
        assert refusals == first[0]
        assert regions == first[1]
        for asked, first_asked in zip(probabilities, first[2]):
            assert asked == pytest.approx(first_asked, abs=1e-9)
        assert likelihood == pytest.approx(first[3], abs=1e-9)


def test_the_hybrid_answers_object_questions_exactly_as_the_relational_circuit(
    synthetic_scenes,
):
    asker = QuestionAsker(random_seed=0, min_region_support=1)
    relational = RelationalPipeline(
        domain=scene_domain(), min_samples_per_leaf=0.2, plain_min_samples_per_leaf=0.2
    )
    hybrid = HybridPipeline(
        domain=scene_domain(),
        positional_fields=(PartField.VIEWPOINTS,),
        min_samples_per_leaf=0.2,
        plain_min_samples_per_leaf=0.2,
    )
    relational.fit(synthetic_scenes)
    hybrid.fit(synthetic_scenes)
    for case in object_level_cases():
        expected = asker.ask(relational, case)
        answered = asker.ask(hybrid, case)
        assert answered.refusal == expected.refusal
        assert [effect.cause_region for effect in answered.effects] == [
            effect.cause_region for effect in expected.effects
        ]
        for one, other in zip(answered.effects, expected.effects):
            assert one.adjusted_probability == pytest.approx(
                other.adjusted_probability, abs=1e-9
            )


def test_the_hybrid_holds_the_viewpoints_by_position_and_not_the_objects(
    synthetic_scenes,
):
    hybrid = HybridPipeline(
        domain=scene_domain(),
        positional_fields=(PartField.VIEWPOINTS,),
        min_samples_per_leaf=0.2,
        plain_min_samples_per_leaf=0.2,
    )
    hybrid.fit(synthetic_scenes)
    assert hybrid.table.unrolled_fields == [PartField.VIEWPOINTS]
    assert not any("objects[" in column for column in hybrid.table.columns)
    assert hybrid.log_likelihood(synthetic_scenes, ExampleView.WHOLE) is not None
    assert not hybrid.order_invariant


def test_a_table_unrolls_only_the_part_fields_it_is_told_to(synthetic_scenes, schema):
    table = FlatTable.unrolled_for(
        schema, synthetic_scenes, part_fields=[PartField.VIEWPOINTS]
    )
    assert table.unrolled_fields == [PartField.VIEWPOINTS]
    row = table.row(synthetic_scenes[0])
    assert all("objects[" not in column for column in row)
    assert any("viewpoints[" in column for column in row)
