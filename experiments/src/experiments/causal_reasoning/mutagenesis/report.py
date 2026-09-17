"""
Writing an
:class:`~experiments.causal_reasoning.mutagenesis.evaluation.EvaluationReport` out as
Markdown, with every table explained and the answers put into words.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Iterable, List, Sequence, Tuple

from experiments.causal_reasoning.mutagenesis.evaluation import (
    EvaluationReport,
    PipelineReport,
    QueryOutcome,
)


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
    The comparison to render.
    """

    def render(self) -> str:
        """
        :return: The whole document.
        """
        sections = [
            self._setup(),
            self._mutagenicity(),
            self._answerability(),
            self._quantities(),
            self._latencies(),
            self._findings(),
        ]
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

    @staticmethod
    def _region_order(region: str) -> Tuple[int, float, str]:
        """
        :param region: A cause region, written out.
        :return: A sort key putting numeric regions in numeric order ahead of the
            others in alphabetical order.
        """
        if re.fullmatch(r"-?\d+(\.\d+)?", region):
            return (0, float(region), region)
        return (1, 0.0, region)

    # %% sections

    def _setup(self) -> List[str]:
        report = self.report
        return [
            "# Mutagenesis: relational circuit against flat-table tree",
            "",
            "The CTU Mutagenesis dataset records 188 nitroaromatic molecules, each as "
            "its own attributes (the `ind1` structural indicator, `logp`, `lumo`, and "
            "whether it tested mutagenic) with one exchangeable part per atom "
            "(element, atom-type code, partial charge, number of bonds) and one per "
            "bond (its type). A molecule has between 14 and 40 atoms, and its atoms "
            "have no canonical order.",
            "",
            "Two pipelines were fitted on the same molecules and asked the same "
            "`cause`/`causes_effect` EQL queries:",
            "",
            "- **relational circuit**: a relational probabilistic circuit fitted on the "
            "molecules' relational structure, one circuit over the molecule's own "
            "attributes and its aggregation counts (chlorine atoms, branching atoms, "
            "double bonds, aromatic bonds), one template over an atom's attributes "
            "and one over a bond's, grounded per query into a circuit over exactly "
            "the queried molecule, atoms and bonds and registered as a causal "
            "circuit;",
            "- **flat-table tree**: a joint probability tree fitted on the same "
            "molecules flattened into one table of the molecule's own attributes and "
            "the same four aggregation counts, registered as a causal circuit the "
            "same way. There is no fixed-width unrolling of the atoms in which one "
            "column would mean the same thing in two molecules, so the table is the "
            "classic propositional summary of a relational example: scalars plus "
            "counts, and no atoms or bonds.",
            "",
            "Both answer a query by backdoor adjustment: the model is stratified so it "
            "is support-deterministic over the cause, the effect's probability is read "
            "off every region of the cause, and any variable the query marks as a "
            "confounder is summed out of that reading. Every query lists one atom and "
            "one bond with all their attributes open, which is what makes grounding "
            "retain the molecule's counts as variables; the flat table ignores parts "
            "a query says nothing about and refuses a query that constrains one.",
            "",
            "## Setup",
            "",
            f"- molecules: {report.training_molecule_count + report.test_molecule_count}"
            f" ({report.training_molecule_count} to fit on, "
            f"{report.test_molecule_count} held out)",
            f"- molecules that tested mutagenic: {self._percent(report.mutagenic_rate)}",
            f"- fewest training rows per leaf: {report.min_samples_per_leaf} in a "
            f"cause-specific model, {report.plain_min_samples_per_leaf} in the plain "
            "model that scores held-out molecules",
        ]

    def _mutagenicity(self) -> List[str]:
        lines = [
            "## How often a molecule is mutagenic",
            "",
            "The molecules themselves, before any model: the share that tested "
            "mutagenic, grouped by the `ind1` indicator, by how many branching atoms "
            "(atoms with three or four bonds, the ring-fusion and branch points of the "
            "molecular graph) the molecule has, and by how many of its bonds are "
            "aromatic. This is the signal the models are asked to explain.",
            "",
        ]
        for title, rates in (
            ("ind1", self.report.mutagenic_by_indicator),
            ("branching atoms", self.report.mutagenic_by_branching_atom_count),
            ("aromatic bonds", self.report.mutagenic_by_aromatic_bond_count),
        ):
            lines += self._table(
                [title, "molecules", "mutagenic"],
                [
                    [str(value), str(rate.molecule_count), self._percent(rate.rate)]
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
            "adjustment and how likely, against the least favourable setting; a "
            "refused cell says why the pipeline could not answer at all.",
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
            "The questions whose cause and effect are both molecule-level attributes "
            "or counts can be put to either pipeline. A question whose effect is one "
            "atom's own attribute has no column in the flat table, so only a model "
            "that grounds itself for the queried atoms can answer it. A question whose "
            "cause is one atom's own attribute is refused by both. The flat table has "
            "no column for it. The relational circuit, grounding with the molecule's "
            "counts left open, mixes one copy of the atom template per sampled count, "
            "and those copies overlap on the atom's element without being identical "
            "(a copy for a molecule with no chlorine has no chlorine atom, a copy for "
            "one with some has), so the grounded circuit is not support-deterministic "
            "over the element and there is no disjoint region of it to intervene on.",
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
        return (
            f"{Verdict.ANSWERED}: with {case.describe_cause(best.cause_region)}, "
            f"{case.effect} with probability "
            f"{self._number(best.adjusted_probability, 2)}, the highest of any "
            f"setting; with {case.describe_cause(worst.cause_region)} it is only "
            f"{self._number(worst.adjusted_probability, 2)}."
        )

    def _quantities(self) -> List[str]:
        lines = [
            "## Fit and likelihood",
            "",
            "What each pipeline cost and how well it explains molecules it never saw. "
            "*Models fitted* counts the plain model plus one support-deterministic "
            "model per distinct cause the questions asked about; *training seconds* "
            "and the *nodes*/*edges* of every fitted circuit are summed over them, "
            "which for the relational circuit includes the atom and bond templates. "
            "*Held-out coverage* is the share of held-out molecules that lie inside "
            "the plain model's support at all, since a tree's leaves span only the "
            "value ranges they were fitted on. The *mean log-likelihood* is over the "
            "covered molecules only, on the molecule's own attributes and its four "
            "counts, the variables both pipelines model; the last column restricts it "
            "to the molecules both pipelines cover, so the two numbers are over the "
            "same rows.",
            "",
        ]
        rows = []
        for pipeline in self.report.pipelines:
            rows.append(
                [
                    pipeline.name,
                    str(pipeline.fit.model_count),
                    self._number(pipeline.fit.training_duration, 2),
                    str(pipeline.fit.size.node_count),
                    str(pipeline.fit.size.edge_count),
                    self._percent(pipeline.likelihood.coverage),
                    self._number(pipeline.likelihood.mean_log_likelihood, 2),
                    self._number(
                        self.report.shared_coverage_log_likelihoods[pipeline.name], 2
                    ),
                ]
            )
        lines += self._table(
            [
                "pipeline",
                "models fitted",
                "training seconds",
                "nodes",
                "edges",
                "held-out coverage",
                "mean log-likelihood (covered)",
                "mean log-likelihood (covered by both)",
            ],
            rows,
        )
        lines += self._whole_molecule_likelihood()
        return lines

    def _whole_molecule_likelihood(self) -> List[str]:
        """
        The likelihood of whole molecules, atoms and bonds included, for the pipelines
        that model them.
        """
        scored = [
            pipeline
            for pipeline in self.report.pipelines
            if pipeline.whole_molecule_likelihood is not None
        ]
        if not scored:
            return []
        lines = [
            "",
            "A held-out molecule is more than its scalars and counts: it is also every "
            "one of its atoms and bonds. Only a pipeline that models the parts can "
            "score those, as the class circuit over the scalars and counts times each "
            "part template over one atom or bond given the counts. A molecule is "
            "covered only if every one of its atoms and bonds lies inside its "
            "template's leaves, so one atom of a rare element, or one partial charge "
            "outside the fitted ranges, puts the whole molecule outside.",
            "",
        ]
        lines += self._table(
            [
                "pipeline",
                "held-out coverage (whole molecule)",
                "mean log-likelihood (whole molecule, covered)",
            ],
            [
                [
                    pipeline.name,
                    self._percent(pipeline.whole_molecule_likelihood.coverage),
                    self._number(
                        pipeline.whole_molecule_likelihood.mean_log_likelihood, 2
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
            "leaves open and grounds one atom template per sampled value, which is "
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
        lines += self._latency_findings()
        return lines

    def _agreement_findings(self) -> List[str]:
        """
        Where the two pipelines answered the same question, whether they agree.
        """
        lines = []
        for case_index, outcome in enumerate(self._first_pipeline.outcomes):
            answers = [
                pipeline.outcomes[case_index]
                for pipeline in self.report.pipelines
                if pipeline.outcomes[case_index].answered
            ]
            if len(answers) < 2:
                continue
            best_regions = {answer.most_effective.cause_region for answer in answers}
            probabilities = ", ".join(
                self._number(answer.most_effective.adjusted_probability, 2)
                for answer in answers
            )
            if len(best_regions) == 1:
                lines.append(
                    f"- On `{outcome.case.name}`, both pipelines find "
                    f"{outcome.case.describe_cause(best_regions.pop())} the most "
                    f"effective setting (adjusted probabilities {probabilities})."
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
        Which pipeline covers more held-out molecules, and which explains the shared
        ones better.
        """
        by_coverage = max(
            self.report.pipelines, key=lambda pipeline: pipeline.likelihood.coverage
        )
        shared = self.report.shared_coverage_log_likelihoods
        by_likelihood = max(shared, key=shared.get)
        if len(set(shared.values())) == 1:
            return [
                "- On a molecule's own attributes and counts, the pipelines assign the "
                f"same mean log-likelihood ({self._number(shared[by_likelihood], 2)}) "
                f"to the held-out molecules and cover the same share of them "
                f"({self._percent(by_coverage.likelihood.coverage)}): the relational "
                "circuit's class-level circuit and the flat-table tree are fitted on "
                "the same rows with the same settings, so they are the same tree. The "
                "relational circuit differs in what it models besides: the atoms and "
                "bonds.",
            ]
        return [
            f"- The {by_coverage.name} covers the most held-out molecules "
            f"({self._percent(by_coverage.likelihood.coverage)}); on the molecules "
            f"both cover, the {by_likelihood} assigns the higher mean log-likelihood "
            f"({self._number(shared[by_likelihood], 2)}).",
        ]

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
            "One row per region of the cause the model distinguishes. *P(region)* is "
            "how much of the training population that region holds; *naive P(effect)* "
            "is the effect's probability simply conditioned on the region; *adjusted* "
            "is the interventional probability after summing out the question's "
            "confounders, which is what the question asks for. Where the two columns "
            "agree, the confounder carried no extra information within that region.",
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
                    "P(region)",
                    "naive P(effect)",
                    "adjusted P(effect | do(cause))",
                ],
                [
                    [
                        effect.cause_region,
                        self._number(effect.region_probability),
                        self._number(effect.naive_probability),
                        self._number(effect.adjusted_probability),
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
