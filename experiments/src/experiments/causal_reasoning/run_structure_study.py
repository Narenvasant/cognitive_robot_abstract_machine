"""
Run the structure study on the clutter-picking and the scene model, and write every
answer and every learned graph to a JSON file.

python -m experiments.causal_reasoning.run_structure_study --output results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
from experiments.causal_reasoning.comparison.structure_study import (
    StructureStudy,
    StructureStudyReport,
    background_knowledge_of,
)
from experiments.causal_reasoning.graspclutter6d.domain import scene_domain
from experiments.causal_reasoning.graspclutter6d.queries import (
    CountCausesGraspability,
    ground_truth_cases,
)
from experiments.causal_reasoning.graspclutter6d.synthetic_scm import SceneTruth
from experiments.causal_reasoning.tracy_clutter_picking.domain import attempt_domain
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    CrowdingCausesLift,
    FrictionCausesLift,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import ClutterTruth

# %% the two studies


def clutter_study(**settings) -> StructureStudy:
    """
    :return: The clutter-picking model's questions, the friction's and the crowding's
        effect on a lift, each adjusting for the environment.
    """
    domain = attempt_domain()
    return StructureStudy(
        domain=domain,
        truth=ClutterTruth(),
        cases=[
            FrictionCausesLift(open_part_count=9),
            CrowdingCausesLift(open_part_count=9),
        ],
        knowledge=background_knowledge_of(
            domain,
            first=["ClutterPickScene.environment"],
            derived_fields={"lift_height": "lifted"},
        ),
        **settings,
    )


def scene_study(**settings) -> StructureStudy:
    """
    :return: The scene model's questions as its known-truth study asks them, adjusting
        for the number of objects or for nothing, and the count questions as the
        catalogue asks them, adjusting for how far the clutter is spread out.
    """
    domain = scene_domain()
    return StructureStudy(
        domain=domain,
        truth=SceneTruth(),
        cases=ground_truth_cases()
        + [
            CountCausesGraspability(statistic_name=name, count_noun=noun)
            for name, noun in (
                ("small_object_count", "small objects"),
                ("occluded_object_count", "occluded objects"),
                ("clear_viewpoint_count", "clear viewpoints"),
            )
        ],
        knowledge=background_knowledge_of(
            domain, first=["GraspClutterSceneAggregations.object_count()"]
        ),
        **settings,
    )


def report_as_json(report: StructureStudyReport) -> dict:
    """
    :param report: A study.
    :return: Everything in it, with every setting written out.
    """
    return json.loads(
        json.dumps(
            {
                "questions": [dataclasses.asdict(row) for row in report.questions],
                "structures": [dataclasses.asdict(row) for row in report.structures],
                "descriptions": {
                    str(setting): description
                    for setting, description in report.descriptions.items()
                },
            },
            default=str,
        )
    )


def main() -> None:
    """
    Run the studies the command line names and write them out.
    """
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", default=["clutter", "scene"])
    parser.add_argument("--example-counts", nargs="+", type=int, default=[400, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settings", type=int, default=None)
    parser.add_argument("--setting", type=int, default=None)
    arguments = parser.parse_args()
    results = {}
    for name in arguments.studies:
        study = {"clutter": clutter_study, "scene": scene_study}[name](
            example_counts=tuple(arguments.example_counts), repeat_count=arguments.repeats
        )
        if arguments.settings is not None:
            study.truth.settings = study.truth.settings[: arguments.settings]
        if arguments.setting is not None:
            study.truth.settings = study.truth.settings[
                arguments.setting : arguments.setting + 1
            ]
        results[name] = report_as_json(
            study.run(log=lambda line: print(line, flush=True))
        )
        arguments.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
