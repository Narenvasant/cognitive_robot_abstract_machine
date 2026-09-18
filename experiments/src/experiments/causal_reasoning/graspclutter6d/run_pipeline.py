"""
Fit every pipeline on the GraspClutter6D scenes, ask them every question of the
catalogue, repeat that over random orderings of the parts and over random splits,
measure how the likelihoods grow with the training set, score every pipeline against the
known truth of a synthetic model of the same domain, and write it all out as Markdown.

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

from experiments.causal_reasoning.graspclutter6d.dataset import (
    GraspClutterDataset,
    fetch_graspclutter_scenes,
)
from experiments.causal_reasoning.graspclutter6d.evaluation import (
    evaluate,
    ground_truth_study,
    learning_curve,
    monte_carlo_study,
    permutation_study,
    split_study,
)
from experiments.causal_reasoning.graspclutter6d.report import MarkdownReport


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


logger = logging.getLogger(__name__)


def main(
    output: Path,
    scene_limit: Optional[int],
    rebuild: bool,
    train_fraction: float,
    seed: int,
    min_samples_per_leaf: Optional[float],
    plain_min_samples_per_leaf: Optional[float],
    ordering_count: int,
    split_count: int,
    min_region_support: int,
) -> None:
    """
    Run the comparison and every study around it, and write them out.

    :param output: The Markdown file to write.
    :param scene_limit: Read only the first scenes of the dataset; all of them if not
        given.
    :param rebuild: Build the scenes afresh instead of reading the stored ones.
    :param train_fraction: Share of scenes to fit on.
    :param seed: Seed of the split and of the questions' Monte-Carlo grounding; the
        repeated splits use the seeds counting up from it.
    :param min_samples_per_leaf: The fewest training rows a leaf of a cause-specific
        model may hold; the pipelines' own default if not given.
    :param plain_min_samples_per_leaf: The fewest training rows a leaf of the plain
        model may hold; the pipelines' own default if not given.
    :param ordering_count: How many random orderings of the parts to try.
    :param split_count: How many random splits to repeat the comparison over.
    :param min_region_support: The fewest training scenes a cause region may hold for
        its effect to be read as an answer.
    """
    import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes

    dataset = GraspClutterDataset(
        fetch_graspclutter_scenes(scene_limit=scene_limit, rebuild=rebuild)
    )
    logger.info("Read %d scenes", len(dataset.scenes))
    settings = dict(
        train_fraction=train_fraction,
        min_samples_per_leaf=min_samples_per_leaf,
        plain_min_samples_per_leaf=plain_min_samples_per_leaf,
        min_region_support=min_region_support,
    )
    logger.info("Comparing on one split")
    report = evaluate(dataset, random_seed=seed, **settings)
    logger.info("Reordering the parts %d times", ordering_count)
    permutations = permutation_study(
        dataset, ordering_count=ordering_count, random_seed=seed, **settings
    )
    logger.info("Repeating over %d splits", split_count)
    splits = split_study(
        dataset, random_seeds=range(seed, seed + split_count), **settings
    )
    logger.info("Measuring the learning curve")
    curve = learning_curve(
        dataset,
        random_seeds=range(seed, seed + min(split_count, 3)),
        plain_min_samples_per_leaf=plain_min_samples_per_leaf,
    )
    logger.info("Following the answers as grounding draws more samples")
    monte_carlo = monte_carlo_study(dataset, random_seed=seed, **settings)
    logger.info("Scoring against the synthetic model's truth")
    truth = ground_truth_study(
        random_seed=seed,
        min_samples_per_leaf=min_samples_per_leaf,
        plain_min_samples_per_leaf=plain_min_samples_per_leaf,
        min_region_support=min_region_support,
    )
    output.write_text(
        MarkdownReport(
            report,
            permutations=permutations,
            splits=splits,
            curve=curve,
            truth=truth,
            monte_carlo=monte_carlo,
        ).render()
    )
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ExperimentFiles().results)
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-samples-per-leaf", type=float, default=None)
    parser.add_argument("--plain-min-samples-per-leaf", type=float, default=None)
    parser.add_argument("--orderings", type=int, default=20)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--min-region-support", type=int, default=10)
    arguments = parser.parse_args()
    main(
        arguments.output,
        arguments.scenes,
        arguments.rebuild,
        arguments.train_fraction,
        arguments.seed,
        arguments.min_samples_per_leaf,
        arguments.plain_min_samples_per_leaf,
        arguments.orderings,
        arguments.splits,
        arguments.min_region_support,
    )
