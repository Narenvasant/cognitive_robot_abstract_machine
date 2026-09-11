"""
Writing an :class:`~experiments.causal_reasoning.tracy_rspn.evaluation.EvaluationReport`
out as Markdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from typing_extensions import Iterable, List, Sequence

from experiments.causal_reasoning.tracy_rspn.evaluation import (
    EvaluationReport,
    PipelineReport,
    QueryOutcome,
)

ANSWERED_MARK = "answered"
REFUSED_MARK = "refused"
"""
How an outcome is marked in the answerability table.
"""


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


def _number(value: float, digits: int = 3) -> str:
    """
    :param value: A number, possibly ``nan``.
    :return: The number rounded, or a dash for ``nan``.
    """
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


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
            self._success(),
            self._answerability(),
            self._quantities(),
            self._latencies(),
        ]
        for case_index in range(len(self.report.pipelines[0].outcomes)):
            sections.append(self._effects(case_index))
        return "\n".join(line for section in sections for line in section + [""])

    def _setup(self) -> List[str]:
        report = self.report
        return [
            "# Tracy clutter picking: relational circuit against flat-table tree",
            "",
            "Both pipelines were fitted on the same recorded attempts and asked the "
            "same `cause`/`causes_effect` EQL queries.",
            "",
            f"- recorded attempts: {report.training_scene_count + report.test_scene_count}"
            f" ({report.training_scene_count} to fit on, {report.test_scene_count} held out)",
            f"- neighbours per attempt: {report.recorded_neighbour_count}",
            f"- attempts whose target was lifted: {100 * report.success_rate:.1f}%",
        ]

    def _success(self) -> List[str]:
        lines = ["## How often the pick came up", ""]
        for title, rates in (
            ("environment", self.report.success_by_environment),
            ("friction coefficient", self.report.success_by_friction),
            ("adjacent neighbours", self.report.success_by_crowding),
        ):
            lines += _table(
                [title, "attempts", "lifted %"],
                [
                    [str(value), str(rate.attempt_count), _number(100 * rate.rate, 1)]
                    for value, rate in rates.items()
                ],
            )
            lines.append("")
        return lines

    def _answerability(self) -> List[str]:
        lines = ["## Which questions each pipeline can answer", ""]
        names = [pipeline.name for pipeline in self.report.pipelines]
        rows = []
        for case_index, outcome in enumerate(self.report.pipelines[0].outcomes):
            row = [outcome.case.question]
            for pipeline in self.report.pipelines:
                row.append(self._verdict(pipeline.outcomes[case_index]))
            rows.append(row)
        lines += _table(["question"] + names, rows)
        lines += [
            "",
            "An answer names the cause region whose intervention makes the effect most "
            "likely after adjustment, and the region EQL's own `cause` search settles "
            "on: the region most probable once the effect is required to hold, from "
            "which the query's samples are drawn.",
        ]
        return lines

    @staticmethod
    def _verdict(outcome: QueryOutcome) -> str:
        """
        :param outcome: One outcome.
        :return: Its answer or refusal, in one cell.
        """
        if not outcome.answered:
            return f"{REFUSED_MARK}: {outcome.refusal}"
        most_effective = outcome.most_effective
        return (
            f"{ANSWERED_MARK}: do(cause = {most_effective.cause_region}) gives the "
            f"effect with probability {_number(most_effective.adjusted_probability)}; "
            f"EQL's search settles on {outcome.best_region} "
            f"({_number(outcome.effect_probability_given_best_region)})"
        )

    def _quantities(self) -> List[str]:
        lines = ["## Fit and likelihood", ""]
        rows = []
        for pipeline in self.report.pipelines:
            rows.append(
                [
                    pipeline.name,
                    str(pipeline.fit.model_count),
                    _number(pipeline.fit.training_duration, 2),
                    str(pipeline.fit.size.node_count),
                    str(pipeline.fit.size.edge_count),
                    _number(100 * pipeline.likelihood.coverage, 1),
                    _number(pipeline.likelihood.mean_log_likelihood, 2),
                    _number(
                        self.report.shared_coverage_log_likelihoods[pipeline.name], 2
                    ),
                ]
            )
        lines += _table(
            [
                "pipeline",
                "models fitted",
                "training seconds",
                "nodes",
                "edges",
                "held-out coverage %",
                "mean log-likelihood (covered)",
                "mean log-likelihood (covered by both)",
            ],
            rows,
        )
        lines += [
            "",
            "Coverage is the share of held-out attempts inside the model's support; a "
            "tree's leaves span only the ranges they saw. Log-likelihoods are over an "
            "attempt's observed attributes, its own and its neighbours'.",
        ]
        return lines

    def _latencies(self) -> List[str]:
        lines = ["## Seconds per question", ""]
        names = [pipeline.name for pipeline in self.report.pipelines]
        rows = []
        for case_index, outcome in enumerate(self.report.pipelines[0].outcomes):
            row = [outcome.case.name]
            for pipeline in self.report.pipelines:
                row.append(_number(pipeline.outcomes[case_index].duration, 2))
            rows.append(row)
        lines += _table(["question"] + names, rows)
        lines += [
            "",
            "A question's time includes fitting the cause-specific model the first time "
            "that cause is asked about, grounding, verification and adjustment.",
        ]
        return lines

    def _effects(self, case_index: int) -> List[str]:
        case = self.report.pipelines[0].outcomes[case_index].case
        lines = [f"## {case.question}", ""]
        for pipeline in self.report.pipelines:
            outcome = pipeline.outcomes[case_index]
            lines.append(f"### {pipeline.name}")
            lines.append("")
            if not outcome.answered:
                lines.append(f"Refused: {outcome.refusal}.")
                lines.append("")
                continue
            lines += _table(
                [
                    "cause region",
                    "P(region)",
                    "naive P(effect)",
                    "adjusted P(effect | do(cause))",
                ],
                [
                    [
                        effect.cause_region,
                        _number(effect.region_probability),
                        _number(effect.naive_probability),
                        _number(effect.adjusted_probability),
                    ]
                    for effect in sorted(
                        outcome.effects, key=lambda effect: effect.cause_region
                    )
                ],
            )
            lines.append("")
        return lines
