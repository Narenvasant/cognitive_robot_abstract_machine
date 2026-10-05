"""
What the permutation invariant neural estimator is allowed to read on the attempts: not
the fields a neighbour only gets after the grasp.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest
from krrood.parametrization.parameterizer import UnderspecifiedParameters

from experiments.causal_reasoning.comparison.neural_baseline import (
    NeuralAdjustmentBaseline,
)
from experiments.causal_reasoning.comparison.pipelines import cause_variable_name
from experiments.causal_reasoning.tracy_clutter_picking.domain import attempt_domain
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    CrowdingCausesLift,
    FrictionCausesLift,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    synthetic_clutter_pick_scenes,
)

NEIGHBOUR_COUNT = 5
"""
Neighbours per synthetic attempt, fewer than a recording has, to keep the tests quick.
"""

LEAF_SUPPORT = 1
"""
The fewest training rows a cause region may hold in these tests.
"""


@pytest.fixture(scope="module")
def baseline() -> NeuralAdjustmentBaseline:
    fitted = NeuralAdjustmentBaseline(
        domain=attempt_domain(), min_region_support=LEAF_SUPPORT, max_iterations=200
    )
    fitted.fit(
        synthetic_clutter_pick_scenes(
            np.random.default_rng(0),
            scene_count=150,
            object_count=NEIGHBOUR_COUNT + 1,
        )
    )
    return fitted


def input_columns(fitted: NeuralAdjustmentBaseline, case) -> list[str]:
    """
    :param fitted: A fitted estimator.
    :param case: The question to assemble the rows for.
    :return: The columns the network is fitted on for that question.
    """
    parameters = UnderspecifiedParameters(case.build())
    schema = fitted.schema
    cause_name = cause_variable_name(parameters)
    [effect_variable] = parameters.effect_variables_from_causes_effect
    [simple_event] = (
        parameters.truncation_assignments_from_where_conditions.simple_sets
    )
    frame, _ = fitted._rows(
        cause_name,
        schema.part_attribute(cause_name),
        [variable.name for variable in parameters.search_confounder_variables],
        effect_variable,
        simple_event[effect_variable],
        schema.part_attribute(effect_variable.name),
    )
    return list(frame.columns)


@pytest.mark.parametrize(
    "case",
    [
        FrictionCausesLift(open_part_count=NEIGHBOUR_COUNT),
        CrowdingCausesLift(open_part_count=NEIGHBOUR_COUNT),
    ],
    ids=["friction", "crowding"],
)
def test_a_lift_question_reads_nothing_recorded_after_the_grasp(baseline, case):
    columns = input_columns(baseline, case)

    assert columns
    assert not [
        column
        for column in columns
        if "displacement" in column or "disturbed" in column
    ]
