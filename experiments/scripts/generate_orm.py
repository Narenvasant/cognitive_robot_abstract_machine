import logging
from pathlib import Path

import experiments
import experiments.control_loop_experiments.benchmark
import experiments.control_loop_experiments.scenarios
import coraplex.orm.ormatic_interface

from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
import experiments.control_loop_experiments.control_loop_profiler
from experiments.causal_reasoning.comparison import (
    baselines as comparison_baselines,
    dataset as comparison_dataset,
    domain as comparison_domain,
    evaluation as comparison_evaluation,
    exceptions as comparison_exceptions,
    neural_baseline as comparison_neural_baseline,
    flat_table as comparison_flat_table,
    pipelines as comparison_pipelines,
    queries as comparison_queries,
    report as comparison_report,
    run as comparison_run,
)
from experiments.causal_reasoning.graspclutter6d import (
    annotations as graspclutter6d_annotations,
    exceptions as graspclutter6d_exceptions,
    grasp_labels as graspclutter6d_grasp_labels,
    queries as graspclutter6d_queries,
    run_pipeline as graspclutter6d_run_pipeline,
    scene_store as graspclutter6d_scene_store,
    synthetic_scm as graspclutter6d_synthetic_scm,
)

# benchmarking measures a running system instead of describing it
ignored_classes = set(classes_of_module(experiments.control_loop_experiments.scenarios))
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.benchmark)
)
ignored_classes |= set(
    classes_of_module(experiments.control_loop_experiments.control_loop_profiler)
)

# the causal-query comparison fits and measures models over its domain instead of
# describing the domain, and reads the dataset's own files instead of the domain's
for comparison_module in (
    comparison_baselines,
    comparison_dataset,
    comparison_domain,
    comparison_evaluation,
    comparison_exceptions,
    comparison_flat_table,
    comparison_neural_baseline,
    comparison_pipelines,
    comparison_queries,
    comparison_report,
    comparison_run,
    graspclutter6d_annotations,
    graspclutter6d_exceptions,
    graspclutter6d_grasp_labels,
    graspclutter6d_queries,
    graspclutter6d_run_pipeline,
    graspclutter6d_scene_store,
    graspclutter6d_synthetic_scm,
):
    ignored_classes |= set(classes_of_module(comparison_module))

# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package(
    [experiments], [coraplex.orm.ormatic_interface], ignored_classes, type_mappings={}
)
logging.getLogger("krrood").setLevel(logging.DEBUG)

# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parent.parent
    / "src"
    / "experiments"
    / "orm"
    / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
