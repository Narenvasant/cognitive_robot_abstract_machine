import logging
from pathlib import Path

import experiments
import experiments.control_loop_experiments.benchmark
import experiments.control_loop_experiments.scenarios
import coraplex.orm.ormatic_interface

from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
import experiments.control_loop_experiments.control_loop_profiler
from experiments.causal_reasoning.graspclutter6d import (
    annotations as graspclutter6d_annotations,
    evaluation as graspclutter6d_evaluation,
    exceptions as graspclutter6d_exceptions,
    flat_table as graspclutter6d_flat_table,
    grasp_labels as graspclutter6d_grasp_labels,
    pipelines as graspclutter6d_pipelines,
    queries as graspclutter6d_queries,
    report as graspclutter6d_report,
    run_pipeline as graspclutter6d_run_pipeline,
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
    graspclutter6d_annotations,
    graspclutter6d_evaluation,
    graspclutter6d_exceptions,
    graspclutter6d_flat_table,
    graspclutter6d_grasp_labels,
    graspclutter6d_pipelines,
    graspclutter6d_queries,
    graspclutter6d_report,
    graspclutter6d_run_pipeline,
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
