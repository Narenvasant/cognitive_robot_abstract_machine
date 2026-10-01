"""
The permutation invariant neural estimator: what it reads, what it answers and that its
answers do not depend on the order the parts are listed in.
"""

from __future__ import annotations

from dataclasses import replace

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest

from experiments.causal_reasoning.comparison.neural_baseline import (
    NeuralAdjustmentBaseline,
    is_numeric,
    pooled_features,
)
from experiments.causal_reasoning.graspclutter6d.domain import scene_domain
from experiments.causal_reasoning.graspclutter6d.queries import (
    CountCausesGraspability,
    OccludedObjectsCauseBlockedObject,
    SizeCausesBlockedObject,
)

LEAF_SUPPORT = 1
"""
The fewest training rows a cause region may hold in these tests.
"""


@pytest.fixture
def baseline(synthetic_scenes) -> NeuralAdjustmentBaseline:
    fitted = NeuralAdjustmentBaseline(
        domain=scene_domain(), min_region_support=LEAF_SUPPORT, max_iterations=200
    )
    fitted.fit(synthetic_scenes)
    return fitted


def test_pooling_does_not_depend_on_the_order_of_the_parts(synthetic_scenes):
    scene = synthetic_scenes[0]
    attributes = list(scene_domain().part_attribute_types("objects"))
    categories = {
        attribute: sorted({vars(one)[attribute] for one in scene.objects}, key=str)
        for attribute in attributes
        if not all(is_numeric(vars(one)[attribute]) for one in scene.objects)
    }
    forwards = pooled_features(scene.objects, attributes, categories)
    backwards = pooled_features(list(reversed(scene.objects)), attributes, categories)
    assert forwards == backwards


def test_it_answers_a_count_question_with_one_effect_per_value(
    baseline, synthetic_scenes
):
    outcome = baseline.ask(
        CountCausesGraspability(
            statistic_name="small_object_count", count_noun="small objects"
        )
    )
    assert outcome.answered
    counts = sorted(
        {
            sum(one.size.value == "small" for one in scene.objects)
            for scene in synthetic_scenes
        }
    )
    assert [effect.cause_region for effect in outcome.effects] == [
        str(count) for count in counts
    ]
    assert sum(effect.support_count for effect in outcome.effects) == len(
        synthetic_scenes
    )
    assert all(0.0 <= effect.adjusted_probability <= 1.0 for effect in outcome.effects)


def test_it_answers_a_question_whose_effect_lives_on_a_part(baseline, synthetic_scenes):
    outcome = baseline.ask(OccludedObjectsCauseBlockedObject())
    assert outcome.answered
    part_count = sum(len(scene.objects) for scene in synthetic_scenes)
    assert sum(effect.support_count for effect in outcome.effects) == part_count


def test_it_answers_a_question_whose_cause_and_effect_live_on_one_part(
    baseline, synthetic_scenes
):
    outcome = baseline.ask(SizeCausesBlockedObject())
    assert outcome.answered
    assert {effect.cause_region for effect in outcome.effects} == {
        str(one.size) for scene in synthetic_scenes for one in scene.objects
    }


def test_its_answers_do_not_move_when_the_parts_are_reordered(synthetic_scenes):
    random_state = np.random.default_rng(0)
    reordered = [
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
        for scene in synthetic_scenes
    ]
    case = OccludedObjectsCauseBlockedObject()
    answers = []
    for examples in (synthetic_scenes, reordered):
        fitted = NeuralAdjustmentBaseline(
            domain=scene_domain(), min_region_support=LEAF_SUPPORT, max_iterations=200
        )
        fitted.fit(examples)
        answers.append(
            [effect.adjusted_probability for effect in fitted.ask(case).effects]
        )
    assert answers[0] == pytest.approx(answers[1], abs=1e-9)
