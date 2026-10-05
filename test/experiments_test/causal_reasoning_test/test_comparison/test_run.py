"""
The knobs every experiment's run shares, as read from its command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
from experiments.causal_reasoning.comparison.evaluation import (
    Comparison,
    baselines_of,
)
from experiments.causal_reasoning.comparison.run import RunSettings
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    attempt_domain,
)


def test_the_shared_flags_are_read_back_into_settings():
    parser = argparse.ArgumentParser()
    RunSettings.add_arguments(parser, Path("results.md"))
    arguments = parser.parse_args(
        [
            "--seed",
            "3",
            "--orderings",
            "2",
            "--splits",
            "1",
            "--min-region-support",
            "4",
        ]
    )
    assert RunSettings.from_arguments(arguments) == RunSettings(
        output=Path("results.md"),
        seed=3,
        ordering_count=2,
        split_count=1,
        min_region_support=4,
    )


def test_the_defaults_leave_the_leaf_sizes_to_the_pipelines():
    parser = argparse.ArgumentParser()
    RunSettings.add_arguments(parser, Path("results.md"))
    settings = RunSettings.from_arguments(parser.parse_args([]))
    assert settings.min_samples_per_leaf is None
    assert settings.plain_min_samples_per_leaf is None
    assert settings.split_count == 0


def test_a_narrowed_comparison_runs_only_the_named_pipelines():
    domain = attempt_domain()
    comparison = Comparison(
        domain=domain, selected_pipelines=("neural adjustment",)
    )

    kept = comparison.selected(baselines_of(domain, 10))

    assert [baseline.name for baseline in kept] == ["neural adjustment"]


def test_a_comparison_that_names_no_pipeline_runs_them_all():
    domain = attempt_domain()
    built = baselines_of(domain, 10)

    kept = Comparison(domain=domain).selected(built)

    assert [baseline.name for baseline in kept] == [
        baseline.name for baseline in built
    ]
