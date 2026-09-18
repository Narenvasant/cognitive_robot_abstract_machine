"""
Writing the comparison out as Markdown, with every table explained and the answers put
into words.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from typing_extensions import Dict, Iterable, List, Optional, Sequence, Tuple

from experiments.causal_reasoning.graspclutter6d.evaluation import (
    EvaluationReport,
    InterventionalEffect,
    LearningCurveReport,
    PermutationReport,
    PipelineReport,
    QueryOutcome,
    SplitReport,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import SceneView
from experiments.causal_reasoning.graspclutter6d.pipelines import LikelihoodReport
from experiments.causal_reasoning.graspclutter6d.queries import count_cases_by_statistic


class Verdict(StrEnum):
    """
    How an outcome is marked in the answerability table.
    """

    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass
class MarkdownReport:
    """
    Renders a comparison as Markdown.
    """

    report: EvaluationReport
    """
    The comparison on one split, with every question asked and timed.
    """

    permutations: Optional[PermutationReport] = None
    """
    The same questions under random orderings of the objects and viewpoints, if run.
    """

    splits: Optional[SplitReport] = None
    """
    The comparison repeated over several splits, if run.
    """

    curve: Optional[LearningCurveReport] = None
    """
    The likelihoods over growing training sets, if run.
    """

    def render(self) -> str:
        """
        :return: The whole document.
        """
        sections = [
            self._setup(),
            self._graspability(),
            self._answerability(),
            self._trends(),
            self._adjustments(),
            self._quantities(),
            self._latencies(),
        ]
        if self.permutations is not None:
            sections.append(self._permutations())
        if self.splits is not None:
            sections.append(self._splits())
        if self.curve is not None:
            sections.append(self._learning_curve())
        sections.append(self._findings())
        for case_index in range(len(self._first_pipeline.outcomes)):
            sections.append(self._effects(case_index))
        return "\n".join(line for section in sections for line in section + [""])

    @property
    def _first_pipeline(self) -> PipelineReport:
        return self.report.pipelines[0]

    @staticmethod
    def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
        """
        :param header: The column titles.
        :param rows: The cells, row by row.
        :return: The lines of a Markdown table.
        """
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return lines

    @staticmethod
    def _number(value: float, digits: int = 3) -> str:
        """
        :param value: A number, possibly ``nan``.
        :param digits: How many decimals to keep.
        :return: The number rounded, or a dash for ``nan``.
        """
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"

    @staticmethod
    def _percent(share: float) -> str:
        """
        :param share: A share between 0 and 1.
        :return: The share as a percentage with one decimal.
        """
        return f"{100 * share:.1f}%"

    def _mean_and_spread(self, values: Sequence[float], digits: int = 2) -> str:
        """
        :param values: Numbers, possibly ``nan``.
        :param digits: How many decimals to keep.
        :return: Their mean and standard deviation over the finite ones, or a dash.
        """
        finite = [value for value in values if not math.isnan(value)]
        if not finite:
            return "-"
        return (
            f"{self._number(float(np.mean(finite)), digits)} ± "
            f"{self._number(float(np.std(finite)), digits)}"
        )

    @staticmethod
    def _region_order(region: str) -> Tuple[int, float, str]:
        """
        :param region: A cause region, written out.
        :return: A sort key putting numeric regions in numeric order ahead of the others
            in alphabetical order.
        """
        if re.fullmatch(r"-?\d+(\.\d+)?", region):
            return (0, float(region), region)
        return (1, 0.0, region)

    # %% sections

    def _setup(self) -> List[str]:
        report = self.report
        return [
            "# GraspClutter6D: relational circuit against flat-table trees",
            "",
            "The GraspClutter6D dataset records a thousand real, densely cluttered bin, "
            "shelf and table scenes, each photographed from thirteen poses by four "
            "cameras, with the ground-truth pose and the visible share of every object "
            "instance in every frame, and with analytic antipodal grasps annotated on "
            "every object model and checked for collision against every scene it stands "
            "in. A scene here is its own attributes (which object catalogue it is built "
            "from, how far its clutter is spread, how far it is stacked, and whether "
            "every object in it keeps a grasp) with one exchangeable part per object "
            "instance (size, diameter, visibility, occlusion, graspability) and one per "
            "camera frame (camera, distance, proximity, clarity). A scene holds between "
            "five and twenty object instances, and they have no canonical order; the "
            "annotation file lists them in the order they were labelled, and nothing "
            "ties a position to an identity.",
            "",
            "Four pipelines were fitted on the same scenes and asked the same "
            "`cause`/`causes_effect` EQL queries:",
            "",
            "- **relational circuit**: a relational probabilistic circuit fitted on the "
            "scenes' relational structure, one circuit over the scene's own attributes "
            "and its aggregation counts (small objects, occluded objects, near "
            "viewpoints, clear viewpoints), one template over an object's attributes "
            "and one over a viewpoint's, grounded per query into a circuit over exactly "
            "the queried scene, objects and viewpoints and registered as a causal "
            "circuit;",
            "- **propositional tree**: a joint probability tree fitted on the scenes "
            "flattened into one table of the scene's own attributes and the same four "
            "counts, the classic propositional summary of a relational example, "
            "registered as a causal circuit the same way;",
            "- **unrolled tree**: the same tree on a table that also carries every "
            "object's and viewpoint's attributes under the part's position, padded with "
            "an absent marker past a scene's last part, so that a column means whatever "
            "part a scene happens to list at that position;",
            "- **scalars-only tree**: the same tree on the scene's own attributes "
            "alone, what a flat learner sees without the relational feature "
            "extraction.",
            "",
            "Every flat tree answers a query by backdoor adjustment on a table column; "
            "the relational circuit does the same on the variable of a grounded "
            "circuit. In both, the model is stratified so it is support-deterministic "
            "over the cause, the effect's probability is read off every region of the "
            "cause, and any variable the query marks as a confounder is summed out of "
            "that reading. Every query lists one object and one viewpoint with all "
            "their attributes open, which is what makes grounding retain the scene's "
            "counts as variables; a flat table ignores parts a query says nothing about "
            "and refuses a query that constrains a column it does not have.",
            "",
            "## Setup",
            "",
            f"- scenes: {report.training_scene_count + report.test_scene_count}"
            f" ({report.training_scene_count} to fit on, "
            f"{report.test_scene_count} held out)",
            "- scenes leaving every object graspable: "
            f"{self._percent(report.graspable_rate)}",
            "- fewest training rows per leaf, as a share of the rows fitted on: "
            f"{report.min_samples_per_leaf} in a cause-specific model, "
            f"{report.plain_min_samples_per_leaf} in the plain model that scores "
            "held-out scenes",
            f"- split seed: {report.random_seed}",
            f"- fewest training scenes a cause region may hold for its effect to be "
            f"read as an answer: {report.min_region_support}; a region below that is "
            "marked † in the tables and takes no part in any summary",
        ]

    def _graspability(self) -> List[str]:
        lines = [
            "## How often a scene leaves every object graspable",
            "",
            "The scenes themselves, before any model: the share that leave every object "
            "instance with at least one antipodal grasp the rest of the scene does not "
            "block, grouped by the object catalogue the scene is built from, by how "
            "many of its objects are small, by how many of them the cameras do not "
            "see whole, and by how many objects it holds at all. This is the signal "
            "the models are asked to explain.",
            "",
        ]
        for title, rates in (
            ("object catalogue", self.report.graspable_by_catalogue),
            ("small objects", self.report.graspable_by_small_object_count),
            ("occluded objects", self.report.graspable_by_occluded_object_count),
            ("objects", self.report.graspable_by_object_count),
        ):
            lines += self._table(
                [title, "scenes", "every object graspable"],
                [
                    [str(value), str(rate.scene_count), self._percent(rate.rate)]
                    for value, rate in rates.items()
                ],
            )
            lines.append("")
        return lines

    def _answerability(self) -> List[str]:
        lines = [
            "## Which questions each pipeline can answer",
            "",
            "One row per question, one column per pipeline. An answered cell says, in "
            "words, which setting of the cause makes the effect most likely after "
            "adjustment and how likely, against the least favourable setting, over "
            "the regions that hold enough training scenes to be read; a refused cell "
            "says why the pipeline could not answer at all.",
            "",
        ]
        names = [pipeline.name for pipeline in self.report.pipelines]
        rows = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            row = [outcome.case.question]
            for pipeline in self.report.pipelines:
                row.append(self._verdict(pipeline.outcomes[case_index]))
            rows.append(row)
        lines += self._table(["question"] + names, rows)
        lines += [
            "",
            "A question about counts needs the counts: the scalars-only tree refuses "
            "it. A question whose effect is one object's own attribute needs the "
            "objects: the propositional tree refuses it, the unrolled tree answers it "
            "about whatever object the scenes list at that position, and the relational "
            "circuit answers it about an exchangeable object. What an answer about "
            '"object 0" is worth is what the reordering below measures.',
        ]
        return lines

    def _verdict(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its answer put into words, or its refusal, in one cell.
        """
        if not outcome.answered:
            return f"{Verdict.REFUSED}: {outcome.refusal}."
        case = outcome.case
        best = outcome.most_effective
        worst = outcome.least_effective
        if best is None:
            return (
                f"{Verdict.ANSWERED}, but no region of the cause holds "
                f"{outcome.min_region_support} training scenes."
            )
        return (
            f"{Verdict.ANSWERED}: with {case.describe_cause(best.cause_region)}, "
            f"{case.effect} with probability "
            f"{self._number(best.adjusted_probability, 2)}, the highest of any setting; "
            f"with {case.describe_cause(worst.cause_region)} it is only "
            f"{self._number(worst.adjusted_probability, 2)}."
        )

    def _trends(self) -> List[str]:
        lines = [
            "## Trend and contrast",
            "",
            "The most effective setting is an argmax over up to twenty sparse regions "
            "and moves with the split. Two summaries that do not: *trend* is "
            "Spearman's rank correlation between the cause's value and the adjusted "
            "probability over the supported regions, for a numeric cause; *contrast* "
            "is the adjusted probability at the highest supported region minus at "
            "the lowest (for a symbolic cause, at the most effective minus at the "
            "least), with Newcombe's interval from the Wilson intervals of the two "
            "regions' support.",
            "",
        ]
        header = ["question"]
        for pipeline in self.report.pipelines:
            header += [f"{pipeline.name}, trend", f"{pipeline.name}, contrast"]
        rows = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            row = [outcome.case.name]
            for pipeline in self.report.pipelines:
                asked = pipeline.outcomes[case_index]
                row += [self._trend_cell(asked), self._contrast_cell(asked)]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _trend_cell(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its trend, or a dash.
        """
        if not outcome.answered or outcome.trend is None:
            return "-"
        return self._number(outcome.trend, 2)

    def _contrast_cell(self, outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its contrast with its interval and the regions it is between, or a
            dash.
        """
        if not outcome.answered or outcome.contrast is None:
            return "-"
        contrast = outcome.contrast
        return (
            f"{self._number(contrast.difference, 2)} "
            f"[{self._number(contrast.interval.lower, 2)}, "
            f"{self._number(contrast.interval.upper, 2)}] "
            f"({contrast.low_region} → {contrast.high_region})"
        )

    def _adjustments(self) -> List[str]:
        lines = [
            "## What adjusting for changes",
            "",
            "The same count question adjusted for the spread of the clutter, for the "
            "number of objects, and for both, read off the relational circuit (the "
            "propositional tree gives the same numbers on these columns). *n* is how "
            "many training scenes hold that value of the cause; † marks a region "
            "below the support threshold.",
            "",
        ]
        outcomes = {
            outcome.case.name: outcome for outcome in self._first_pipeline.outcomes
        }
        for statistic_name, cases in count_cases_by_statistic().items():
            asked = [outcomes[case.name] for case in cases if case.name in outcomes]
            if not asked or not all(outcome.answered for outcome in asked):
                continue
            lines += [f"### {statistic_name}", ""]
            by_region: Dict[str, Dict[str, InterventionalEffect]] = {}
            for outcome in asked:
                for effect in outcome.effects:
                    by_region.setdefault(effect.cause_region, {})[
                        outcome.case.name
                    ] = effect
            header = ["cause region", "n", "naive"] + [
                "adjusted for "
                + " and ".join(confounder.noun for confounder in case.confounders)
                for case in cases
            ]
            rows = []
            for region in sorted(by_region, key=self._region_order):
                effects = by_region[region]
                first = next(iter(effects.values()))
                row = [
                    self._region_label(first, asked[0].min_region_support),
                    str(first.support_count),
                    self._number(first.naive_probability),
                ]
                for case in cases:
                    effect = effects.get(case.name)
                    row.append(
                        "-"
                        if effect is None
                        else self._number(effect.adjusted_probability)
                    )
                rows.append(row)
            lines += self._table(header, rows)
            lines.append("")
        return lines

    @staticmethod
    def _region_label(effect: InterventionalEffect, min_region_support: int) -> str:
        """
        :param effect: One region's effect.
        :param min_region_support: The support threshold.
        :return: The region, marked if it is below the threshold.
        """
        if effect.is_supported(min_region_support):
            return effect.cause_region
        return f"{effect.cause_region} †"

    def _quantities(self) -> List[str]:
        lines = [
            "## Fit and likelihood",
            "",
            "What each pipeline cost. *Models fitted* counts the plain model plus one "
            "support-deterministic model per distinct cause the questions asked about "
            "and the pipeline could fit; *training seconds* and the *nodes*/*edges* of "
            "every fitted circuit are summed over them, which for the relational "
            "circuit includes the object and viewpoint templates.",
            "",
        ]
        lines += self._table(
            ["pipeline", "models fitted", "training seconds", "nodes", "edges"],
            [
                [
                    pipeline.name,
                    str(pipeline.fit.model_count),
                    self._number(pipeline.fit.training_duration, 2),
                    str(pipeline.fit.size.node_count),
                    str(pipeline.fit.size.edge_count),
                ]
                for pipeline in self.report.pipelines
            ],
        )
        lines += [
            "",
            "How well each explains scenes it never saw, on three views of a scene: its "
            "own scalars, which every pipeline models; its scalars and counts; and the "
            "whole scene, objects and viewpoints included, which only the pipelines "
            "that model the parts can score. The relational circuit scores a whole "
            "scene as its class circuit over the scalars and counts times each part "
            "template over one object or viewpoint given the counts; the unrolled tree "
            "scores it as one row. *Held-out coverage* is the share of held-out scenes "
            "that lie inside the plain model's support at all, since a tree's leaves "
            "span only the value ranges they were fitted on, and a whole scene is "
            "covered only if every one of its objects and viewpoints is. The *mean "
            "log-likelihood* is over the covered scenes only; the last column restricts "
            "it to the scenes every pipeline in the table covers, so the numbers are "
            "over the same rows.",
        ]
        for view in SceneView:
            scored = [
                pipeline
                for pipeline in self.report.pipelines
                if pipeline.likelihoods[view] is not None
            ]
            lines += ["", f"### {view}", ""]
            lines += self._table(
                [
                    "pipeline",
                    "held-out coverage",
                    "mean log-likelihood (covered)",
                    "mean log-likelihood (covered by all)",
                ],
                [
                    [
                        pipeline.name,
                        self._percent(pipeline.likelihoods[view].coverage),
                        self._number(pipeline.likelihoods[view].mean_log_likelihood, 2),
                        self._number(
                            self.report.shared_coverage_log_likelihoods[view][
                                pipeline.name
                            ],
                            2,
                        ),
                    ]
                    for pipeline in scored
                ],
            )
        return lines

    def _latencies(self) -> List[str]:
        lines = [
            "## Seconds per question",
            "",
            "Wall-clock time from asking to the answer or the refusal. The *first ask* "
            "of a cause includes fitting that cause's own support-deterministic model; "
            "*asked again* repeats the question with every model fitted, so only "
            "grounding (for the relational circuit), verification and backdoor "
            "adjustment remain. A refusal is fast when it is a schema check; a "
            "relational answer draws Monte-Carlo samples for every count the query "
            "leaves open and grounds one part template per sampled value, which is "
            "where its time goes.",
            "",
        ]
        header = ["question"]
        for pipeline in self.report.pipelines:
            header += [f"{pipeline.name}, first ask", f"{pipeline.name}, asked again"]
        rows = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            row = [outcome.case.name]
            for pipeline in self.report.pipelines:
                asked = pipeline.outcomes[case_index]
                row += [
                    self._number(asked.duration, 2),
                    self._number(asked.repeat_duration, 2),
                ]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _permutations(self) -> List[str]:
        permutations = self.permutations
        lines = [
            "## Does the order of the objects matter?",
            "",
            f"Every scene's objects and viewpoints were put in a random order, "
            f"{permutations.ordering_count} times over, and each time the pipelines "
            "that model the parts were refitted on the same split and asked the "
            "questions about objects again; the parts in the order the dataset lists "
            "them is the baseline every reordering is measured against. A relational "
            "circuit treats the objects as exchangeable, so nothing about it can "
            "depend on the order; an unrolled table's column `objects[0]` holds a "
            "different object of every scene after each reordering. Per question and "
            "pipeline: how many reorderings were answered; over the cause regions "
            "every answered ordering distinguishes, the mean standard deviation and "
            "the widest range of the adjusted probability; the share of reorderings "
            "whose most effective region is not the dataset-order one; and the share "
            "whose trend changed sign.",
            "",
        ]
        lines += self._table(
            [
                "question",
                "pipeline",
                "reorderings answered",
                "mean sd of adjusted P(effect)",
                "widest range",
                "argmax moved",
                "trend sign flipped",
            ],
            [
                [
                    question.case.name,
                    question.pipeline_name,
                    f"{len(question.answered)} of {len(question.reordered)}",
                    self._number(question.mean_adjusted_standard_deviation, 3),
                    self._number(question.largest_adjusted_difference, 2),
                    self._share(question.argmax_flip_share),
                    self._share(question.trend_sign_flip_share),
                ]
                for question in permutations.questions
            ],
        )
        lines += [
            "",
            "The whole-scene likelihood of the same held-out scenes with the parts in "
            "the order the dataset lists them, and over the reorderings. The "
            "dataset's order is not arbitrary throughout: a scene's frames are "
            "numbered by the recording rig, four cameras per pose in a fixed sequence, "
            "so which camera took frame *i* is the same in every scene, and a column "
            "that addresses a viewpoint by position addresses a real thing. Its "
            "objects carry no such order. *Largest drop* is how far below the "
            "dataset-order likelihood the worst reordering took each pipeline.",
            "",
        ]
        lines += self._table(
            [
                "pipeline",
                "dataset order, coverage / mean log-likelihood",
                "reorderings, coverage / mean log-likelihood (mean ± sd)",
                "largest drop",
            ],
            [
                [
                    name,
                    self._likelihood_cell(permutations.in_dataset_order(name)),
                    f"{self._mean_and_spread([report.coverage for report in reordered], 3)}"
                    f" / {self._mean_and_spread([report.mean_log_likelihood for report in reordered])}",
                    self._number(permutations.largest_likelihood_drop(name), 2),
                ]
                for name in permutations.whole_scene_likelihoods
                for reordered in [permutations.reordered(name)]
            ],
        )
        return lines

    def _share(self, share: float) -> str:
        """
        :param share: A share between 0 and 1, or ``nan``.
        :return: The share as a percentage, or a dash.
        """
        if math.isnan(share):
            return "-"
        return self._percent(share)

    def _likelihood_cell(self, likelihood: LikelihoodReport) -> str:
        """
        :param likelihood: A likelihood report.
        :return: Its coverage and mean log-likelihood in one cell.
        """
        return (
            f"{self._percent(likelihood.coverage)} / "
            f"{self._number(likelihood.mean_log_likelihood, 2)}"
        )

    def _splits(self) -> List[str]:
        splits = self.splits
        lines = [
            "## Over several splits",
            "",
            f"The comparison repeated over {len(splits.reports)} random splits (seeds "
            f"{', '.join(str(report.random_seed) for report in splits.reports)}), "
            "mean ± standard deviation. The likelihoods are over the scenes every "
            "pipeline modelling the view covers.",
            "",
        ]
        for view in SceneView:
            names = [
                name
                for name in splits.pipeline_names
                if splits.reports[0].pipeline(name).likelihoods[view] is not None
            ]
            lines += [f"### {view}", ""]
            lines += self._table(
                [
                    "pipeline",
                    "held-out coverage",
                    "mean log-likelihood (covered by all)",
                ],
                [
                    [
                        name,
                        self._mean_and_spread(splits.coverage(name, view), 3),
                        self._mean_and_spread(splits.shared_log_likelihood(name, view)),
                    ]
                    for name in names
                ],
            )
            lines.append("")
        lines += [
            "Per question, how many splits each pipeline answered, and the mean ± "
            "standard deviation over the splits of its trend and of its contrast:",
            "",
        ]
        header = ["question"]
        for name in splits.pipeline_names:
            header += [f"{name}, answered", f"{name}, trend", f"{name}, contrast"]
        rows = []
        for case_index, outcome in enumerate(splits.reports[0].pipelines[0].outcomes):
            row = [outcome.case.name]
            for name in splits.pipeline_names:
                outcomes = splits.outcomes(name, case_index)
                answered = [one for one in outcomes if one.answered]
                row += [
                    f"{len(answered)} of {len(outcomes)}",
                    self._mean_and_spread(
                        [
                            float("nan") if one.trend is None else one.trend
                            for one in answered
                        ]
                    ),
                    self._mean_and_spread(
                        [
                            (
                                float("nan")
                                if one.contrast is None
                                else one.contrast.difference
                            )
                            for one in answered
                        ]
                    ),
                ]
            rows.append(row)
        lines += self._table(header, rows)
        return lines

    def _learning_curve(self) -> List[str]:
        curve = self.curve
        seeds = sorted({point.random_seed for point in curve.points})
        lines = [
            "## How much training data it takes",
            "",
            f"Every pipeline's plain model fitted on a growing share of the scenes and "
            f"scored on the same held-out fifth, over {len(seeds)} splits, mean ± "
            "standard deviation of the held-out coverage and of the mean log-likelihood "
            "over the covered scenes. The relational circuit's templates pool every "
            "object of every training scene, and every frame of it, where the unrolled "
            "tree sees one row per scene.",
        ]
        for view in (SceneView.SCALARS_AND_COUNTS, SceneView.WHOLE_SCENE):
            names = [
                name
                for name in curve.pipeline_names
                if curve.points_of(name, curve.train_fractions[0])[0].likelihoods[view]
                is not None
            ]
            lines += ["", f"### {view}", ""]
            header = ["training share"]
            for name in names:
                header += [f"{name}, coverage", f"{name}, mean log-likelihood"]
            rows = []
            for train_fraction in curve.train_fractions:
                row = [self._percent(train_fraction)]
                for name in names:
                    likelihoods = [
                        point.likelihoods[view]
                        for point in curve.points_of(name, train_fraction)
                    ]
                    row += [
                        self._mean_and_spread(
                            [likelihood.coverage for likelihood in likelihoods], 3
                        ),
                        self._mean_and_spread(
                            [
                                likelihood.mean_log_likelihood
                                for likelihood in likelihoods
                            ]
                        ),
                    ]
                rows.append(row)
            lines += self._table(header, rows)
        return lines

    def _findings(self) -> List[str]:
        """
        What the numbers add up to, read off the report itself.
        """
        lines = ["## What the results show", ""]
        for pipeline in self.report.pipelines:
            answered = [outcome for outcome in pipeline.outcomes if outcome.answered]
            refused = [outcome for outcome in pipeline.outcomes if not outcome.answered]
            sentence = (
                f"- The {pipeline.name} answered {len(answered)} of "
                f"{len(pipeline.outcomes)} questions"
            )
            if refused:
                sentence += ", refusing " + "; ".join(
                    f"`{outcome.case.name}` because {outcome.refusal}"
                    for outcome in refused
                )
            lines.append(sentence + ".")
        lines += self._agreement_findings()
        lines += self._likelihood_findings()
        if self.permutations is not None:
            lines += self._permutation_findings()
        lines += self._latency_findings()
        return lines

    def _agreement_findings(self) -> List[str]:
        """
        Where several pipelines answered the same question, whether they agree.
        """
        lines = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            answers = [
                pipeline.outcomes[case_index]
                for pipeline in self.report.pipelines
                if pipeline.outcomes[case_index].most_effective is not None
            ]
            if len(answers) < 2:
                continue
            best_regions = {answer.most_effective.cause_region for answer in answers}
            probabilities = ", ".join(
                f"{answer.pipeline_name} "
                f"{self._number(answer.most_effective.adjusted_probability, 2)}"
                for answer in answers
            )
            if len(best_regions) == 1:
                lines.append(
                    f"- On `{outcome.case.name}`, every pipeline that answered finds "
                    f"{outcome.case.describe_cause(best_regions.pop())} the most "
                    f"effective setting (adjusted probabilities: {probabilities})."
                )
                continue
            lines.append(
                f"- On `{outcome.case.name}`, the pipelines disagree on the most "
                "effective setting: "
                + "; ".join(
                    f"the {answer.pipeline_name} says "
                    f"{outcome.case.describe_cause(answer.most_effective.cause_region)} "
                    f"({self._number(answer.most_effective.adjusted_probability, 2)})"
                    for answer in answers
                )
                + "."
            )
        return lines

    def _likelihood_findings(self) -> List[str]:
        """
        Which pipeline explains the held-out scenes best, per view.
        """
        lines = []
        for view in SceneView:
            shared = self.report.shared_coverage_log_likelihoods[view]
            if len(shared) < 2:
                continue
            finite = {
                name: value for name, value in shared.items() if not math.isnan(value)
            }
            if not finite:
                continue
            best = max(finite, key=finite.get)
            if len(set(finite.values())) == 1:
                lines.append(
                    f"- On the {view}, every pipeline modelling it assigns the same "
                    f"mean log-likelihood ({self._number(finite[best], 2)}) to the "
                    "held-out scenes: on those columns they are the same tree fitted on "
                    "the same rows."
                )
                continue
            others = ", ".join(
                f"{name} {self._number(value, 2)}"
                for name, value in finite.items()
                if name != best
            )
            coverage = ", ".join(
                f"{pipeline.name} {self._percent(pipeline.likelihoods[view].coverage)}"
                for pipeline in self.report.pipelines
                if pipeline.likelihoods[view] is not None
            )
            lines.append(
                f"- On the {view}, the {best} assigns the highest mean log-likelihood "
                f"({self._number(finite[best], 2)}, against {others}) to the held-out "
                f"scenes every pipeline covers; coverage: {coverage}."
            )
        return lines

    def _permutation_findings(self) -> List[str]:
        """
        Which pipelines' answers moved when the parts were reordered.
        """
        lines = []
        by_pipeline = {}
        for question in self.permutations.questions:
            by_pipeline.setdefault(question.pipeline_name, []).append(question)
        for name, questions in by_pipeline.items():
            ranges = [
                question.largest_adjusted_difference
                for question in questions
                if not math.isnan(question.largest_adjusted_difference)
            ]
            flips = [
                question.argmax_flip_share
                for question in questions
                if not math.isnan(question.argmax_flip_share)
            ]
            sentence = (
                f"- Over {self.permutations.ordering_count} reorderings, the {name}'s "
                "adjusted effect probabilities ranged by up to "
                f"{self._number(max(ranges), 2) if ranges else '-'} and its most "
                "effective region moved in "
                f"{self._share(float(np.mean(flips))) if flips else '-'} of the "
                "reorderings; its whole-scene mean log-likelihood fell by up to "
                f"{self._number(self.permutations.largest_likelihood_drop(name), 2)}"
                " from the dataset's own order"
            )
            lines.append(sentence + ".")
        return lines

    def _latency_findings(self) -> List[str]:
        """
        How the pipelines compare on time per answered question.
        """
        lines = []
        for pipeline in self.report.pipelines:
            answered = [outcome for outcome in pipeline.outcomes if outcome.answered]
            if not answered:
                continue
            mean_repeat = sum(outcome.repeat_duration for outcome in answered) / len(
                answered
            )
            lines.append(
                f"- The {pipeline.name} takes {self._number(mean_repeat, 2)} seconds per "
                "answered question on average once its models are fitted."
            )
        return lines

    def _effects(self, case_index: int) -> List[str]:
        case = self._first_pipeline.outcomes[case_index].case
        lines = [
            f"## {case.question}",
            "",
            "One row per region of the cause the model distinguishes. *n* is how many "
            "training rows the region holds, and † marks a region below the support "
            "threshold; *P(region)* is how much of the fitted population it holds; "
            "*naive P(effect)* is the effect's probability simply conditioned on the "
            "region; *adjusted* is the interventional probability after summing out "
            "the question's confounders, which is what the question asks for, and the "
            "*interval* is its Wilson interval over the region's n. Where naive and "
            "adjusted agree, the confounders carried no further information within "
            "that region.",
            "",
        ]
        for pipeline in self.report.pipelines:
            outcome = pipeline.outcomes[case_index]
            lines.append(f"### {pipeline.name}")
            lines.append("")
            if not outcome.answered:
                lines.append(f"Refused: {outcome.refusal}.")
                lines.append("")
                continue
            lines += self._table(
                [
                    "cause region",
                    "n",
                    "P(region)",
                    "naive P(effect)",
                    "adjusted P(effect | do(cause))",
                    "95% interval",
                ],
                [
                    [
                        self._region_label(effect, outcome.min_region_support),
                        str(effect.support_count),
                        self._number(effect.region_probability),
                        self._number(effect.naive_probability),
                        self._number(effect.adjusted_probability),
                        f"[{self._number(effect.adjusted_interval.lower, 2)}, "
                        f"{self._number(effect.adjusted_interval.upper, 2)}]",
                    ]
                    for effect in sorted(
                        outcome.effects,
                        key=lambda effect: self._region_order(effect.cause_region),
                    )
                ],
            )
            lines += [
                "",
                f"EQL's own `cause` search settles on {outcome.best_region}: the region "
                "most probable once the effect is required to hold, from which the "
                "query's samples are drawn (P(effect | do) = "
                f"{self._number(outcome.effect_probability_given_best_region, 2)}).",
                "",
            ]
        return lines
