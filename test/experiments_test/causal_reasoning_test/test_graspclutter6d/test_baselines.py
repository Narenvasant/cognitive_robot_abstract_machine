"""
Regression adjustment on the propositional table.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import pytest

from experiments.causal_reasoning.graspclutter6d.baselines import (
    RegressionAdjustmentBaseline,
)
from experiments.causal_reasoning.graspclutter6d.evaluation import Refusal
from experiments.causal_reasoning.graspclutter6d.queries import (
    ground_truth_cases,
    object_level_cases,
)


@pytest.fixture
def baseline(synthetic_scenes) -> RegressionAdjustmentBaseline:
    fitted = RegressionAdjustmentBaseline(min_region_support=1)
    fitted.fit(synthetic_scenes)
    return fitted


def test_it_answers_a_count_question_with_one_effect_per_value(
    baseline, synthetic_scenes
):
    outcome = baseline.ask(ground_truth_cases()[0])
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


def test_it_answers_a_symbolic_cause(baseline):
    outcome = baseline.ask(ground_truth_cases()[3])
    assert outcome.answered
    assert all(effect.ordinal is None for effect in outcome.effects)


def test_it_refuses_a_question_about_one_object(baseline):
    for case in object_level_cases():
        assert baseline.ask(case).refusal is Refusal.SCHEMA_MISMATCH
