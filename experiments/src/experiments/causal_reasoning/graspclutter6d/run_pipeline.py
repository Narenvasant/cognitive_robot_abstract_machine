"""
Fit every pipeline on the GraspClutter6D scenes, ask them every question of the
catalogue, repeat that over random orderings of the parts, score every pipeline against
the known truth of a synthetic model of the same domain, follow the relational answers
as grounding draws more samples, measure how the likelihoods grow with the training set
and how the cost grows with the number of objects, and write it all out as Markdown.

Run with::

    python -m experiments.causal_reasoning.graspclutter6d.run_pipeline
        [--output RESULTS.md] [--scenes N] [--rebuild] [--train-fraction F] [--seed N]
        [--min-samples-per-leaf F] [--plain-min-samples-per-leaf F]
        [--orderings N] [--splits N] [--min-region-support N]

The built scenes are read from the database ``GRASPCLUTTER6D_DATABASE_URI`` names, or
from a database file beside the dataset. The first run, and a run with ``--rebuild``,
builds them from the annotations read from the dataset server named by
``SEMANTIC_DIGITAL_TWIN_DATASET_SERVER`` and from the extracted ``grasp_label`` and
``collision_label`` folders, which is the slow part. The data access objects the
relational pipeline fits on come from the ``experiments`` package's generated ORM
interface; build it with ``scripts/regenerate_all_orm.py`` first.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional

from experiments.causal_reasoning.comparison.evaluation import Comparison
from experiments.causal_reasoning.comparison.report import ReportText
from experiments.causal_reasoning.comparison.run import (
    Experiment,
    RunSettings,
    ScalingSetup,
    run,
)
from experiments.causal_reasoning.graspclutter6d.dataset import (
    fetch_graspclutter_scenes,
    graspability_summaries,
    graspclutter_dataset,
)
from experiments.causal_reasoning.graspclutter6d.domain import PartField, scene_domain
from experiments.causal_reasoning.graspclutter6d.queries import (
    OccludedObjectsCauseBlockedObject,
    ground_truth_cases,
    monte_carlo_cases,
    object_level_cases,
    query_catalogue,
)
from experiments.causal_reasoning.graspclutter6d.synthetic_scm import (
    SceneTruth,
    scenes_of_size,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentFiles:
    """
    Where this experiment writes its comparison.
    """

    package_directory: Path = Path(__file__).parent
    """
    Where this experiment lives.
    """

    @property
    def results(self) -> Path:
        """
        Where the comparison is written.
        """
        return self.package_directory / "results.md"


def report_text() -> ReportText:
    """
    :return: What the report says about the dataset and the pipelines.
    """
    return ReportText(
        title="GraspClutter6D: relational circuit against flat-table trees",
        introduction=(
            "The GraspClutter6D dataset records a thousand real, densely cluttered "
            "bin, shelf and table scenes, each photographed from thirteen poses by "
            "four cameras, with the ground-truth pose and the visible share of every "
            "object instance in every frame, and with analytic antipodal grasps "
            "annotated on every object model and checked for collision against every "
            "scene it stands in. A scene here is its own attributes (which object "
            "catalogue it is built from, how far its clutter is spread, how far it is "
            "stacked, and whether every object in it keeps a grasp) with one "
            "exchangeable part per object instance (size, diameter, visibility, "
            "occlusion, graspability) and one per camera frame (camera, distance, "
            "proximity, clarity). A scene holds between five and twenty object "
            "instances, and they have no canonical order; the annotation file lists "
            "them in the order they were labelled, and nothing ties a position to an "
            "identity.",
            "Five pipelines were fitted on the same scenes and asked the same "
            "`cause`/`causes_effect` EQL queries:",
            "- **relational circuit**: a relational probabilistic circuit fitted on "
            "the scenes' relational structure, one circuit over the scene's own "
            "attributes and its aggregation counts (small objects, occluded objects, "
            "near viewpoints, clear viewpoints), one template over an object's "
            "attributes and one over a viewpoint's, grounded per query into a circuit "
            "over exactly the queried scene, objects and viewpoints and registered as "
            "a causal circuit;\n"
            "- **hybrid circuit**: the same, except that the viewpoints, whose "
            "position in a scene the recording rig fixes, are held by position "
            "rather than pooled into a template;\n"
            "- **propositional tree**: a joint probability tree fitted on the scenes "
            "flattened into one table of the scene's own attributes and the same four "
            "counts, the classic propositional summary of a relational example, "
            "registered as a causal circuit the same way;\n"
            "- **unrolled tree**: the same tree on a table that also carries every "
            "object's and viewpoint's attributes under the part's position, padded "
            "with an absent marker past a scene's last part, so that a column means "
            "whatever part a scene happens to list at that position;\n"
            "- **scalars-only tree**: the same tree on the scene's own attributes "
            "alone, what a flat learner sees without the relational feature "
            "extraction.",
            "Every flat tree answers a query by backdoor adjustment on a table column; "
            "the relational circuit does the same on the variable of a grounded "
            "circuit. In both, the model is stratified so it is support-deterministic "
            "over the cause, the effect's probability is read off every region of the "
            "cause, and any variable the query marks as a confounder is summed out of "
            "that reading. Every query lists one object and one viewpoint with all "
            "their attributes open, which is what makes grounding retain the scene's "
            "counts as variables; a flat table ignores parts a query says nothing "
            "about and refuses a query that constrains a column it does not have.",
        ),
        effect_summary=(
            "the object catalogue the scene is built from, by how many of its objects "
            "are small, by how many of them the cameras do not see whole, and by how "
            "many objects it holds at all"
        ),
        answerability_note=(
            "A question about counts needs the counts: the scalars-only tree refuses "
            "it. A question whose effect is one object's own attribute needs the "
            "objects: the propositional tree refuses it, the unrolled tree answers it "
            "about whatever object the scenes list at that position, and the "
            "relational circuit answers it about an exchangeable object. What an "
            'answer about "object 0" is worth is what the reordering below measures.'
        ),
        reordering_note=(
            "The dataset's order is not arbitrary throughout: a scene's frames are "
            "numbered by the recording rig, four cameras per pose in a fixed sequence, "
            "so which camera took frame *i* is the same in every scene, and a column "
            "that addresses a viewpoint by position addresses a real thing. Its "
            "objects carry no such order."
        ),
        ground_truth_note=(
            "The cause is forced to a value in the mechanism and the effect's rate "
            "read off 200,000 forced scenes. The number of objects is the model's one "
            "confounder, driving both the causes and the effect, and the scenes list "
            "their small objects first, so a column that addresses an object by "
            "position is systematically misleading. A count's extreme values are rare "
            "in the data and rarer still within every stratum of the confounder, so "
            "no estimator recovers their interventional probability well; the "
            "questions whose cause or effect lives on one object are where the "
            "pipelines differ."
        ),
        learning_curve_note=(
            "Every object of every training scene, and every frame of it, goes into "
            "the templates."
        ),
    )


def graspclutter_experiment() -> Experiment:
    """
    :return: The GraspClutter6D comparison: what is compared and what is asked.
    """
    return Experiment(
        comparison=Comparison(
            domain=scene_domain(),
            positional_fields=(PartField.VIEWPOINTS,),
            summaries=graspability_summaries,
        ),
        text=report_text(),
        cases=query_catalogue(),
        part_cases=object_level_cases(),
        monte_carlo_cases=monte_carlo_cases(),
        truth=SceneTruth(),
        truth_cases=ground_truth_cases(),
        scaling=ScalingSetup(
            examples_of_size=scenes_of_size, case=OccludedObjectsCauseBlockedObject()
        ),
        results=ExperimentFiles().results,
    )


def main(settings: RunSettings, scene_limit: Optional[int], rebuild: bool) -> None:
    """
    Run the comparison and every study around it, and write them out.

    :param settings: The run's knobs.
    :param scene_limit: Read only the first scenes of the dataset; all of them if not
        given.
    :param rebuild: Build the scenes afresh instead of reading the stored ones.
    """
    import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes

    dataset = graspclutter_dataset(
        fetch_graspclutter_scenes(scene_limit=scene_limit, rebuild=rebuild)
    )
    logger.info("Read %d scenes", len(dataset.examples))
    run(graspclutter_experiment(), dataset, settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    RunSettings.add_arguments(parser, ExperimentFiles().results)
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    main(RunSettings.from_arguments(arguments), arguments.scenes, arguments.rebuild)
