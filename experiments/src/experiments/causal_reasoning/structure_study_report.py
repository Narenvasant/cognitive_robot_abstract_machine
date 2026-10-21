"""
Write the structure study's results, as the runner stored them, out as Markdown tables.

python -m experiments.causal_reasoning.structure_study_report results*.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from typing_extensions import Any, Dict, List

# %% reading the results


def load(paths: List[Path]) -> Dict[str, Dict[str, List[Any]]]:
    """
    :param paths: The runner's JSON files.
    :return: Per study, every question row and every structure row of every file.
    """
    merged: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {"questions": [], "structures": []}
    )
    for path in paths:
        for study, rows in json.loads(path.read_text()).items():
            merged[study]["questions"] += rows["questions"]
            merged[study]["structures"] += rows["structures"]
    return merged


def errors(question: Dict[str, Any], arm: str) -> List[float]:
    """
    :param question: One question row.
    :param arm: ``declared``, ``learned`` or ``unadjusted``.
    :return: The absolute error of every scored region the arm answered.
    """
    return [
        abs(region[arm] - region["true_probability"])
        for region in question["regions"]
        if region[arm] is not None
    ]


def mean(values: List[float]) -> float:
    """
    :return: The mean, or ``nan`` without values.
    """
    return float(np.mean(values)) if values else float("nan")


def share(flags: List[bool]) -> str:
    """
    :return: How many of the flags hold, as a count and a percentage.
    """
    if not flags:
        return "–"
    return f"{sum(flags)}/{len(flags)} ({100 * sum(flags) / len(flags):.0f}%)"


# %% the tables


def accuracy_table(questions: List[Dict[str, Any]], key) -> List[str]:
    """
    :param questions: Question rows.
    :param key: What groups the rows in one table line.
    :return: Mean absolute error of each arm, and how often each arm answered, per
        group.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for question in questions:
        groups[key(question)].append(question)
    lines = [
        "| group | MAE declared | MAE learned | MAE no adjustment | learned beats declared | declared answered | learned answered |",
        "|---|---|---|---|---|---|---|",
    ]
    for group, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        both = [
            row for row in rows if errors(row, "declared") and errors(row, "learned")
        ]
        lines.append(
            f"| {group} | {mean([e for row in rows for e in errors(row, 'declared')]):.3f} "
            f"| {mean([e for row in rows for e in errors(row, 'learned')]):.3f} "
            f"| {mean([e for row in rows for e in errors(row, 'unadjusted')]):.3f} "
            f"| {share([mean(errors(row, 'learned')) < mean(errors(row, 'declared')) for row in both])} "
            f"| {share([bool(errors(row, 'declared')) for row in rows])} "
            f"| {share([bool(errors(row, 'learned')) for row in rows])} |"
        )
    return lines


def validity_table(questions: List[Dict[str, Any]]) -> List[str]:
    """
    :param questions: Question rows.
    :return: Per question, the sets each arm adjusted for and how often each was valid
        in the true graph.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for question in questions:
        groups[question["case_name"]].append(question)
    lines = [
        "| question | declared set | declared valid | learned sets (count) | learned valid | learned refused |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows in sorted(groups.items()):
        learned_sets = defaultdict(int)
        for row in rows:
            if row["learned_set"] is not None:
                learned_sets[
                    ", ".join(short(n) for n in row["learned_set"]) or "∅"
                ] += 1
        refusals = defaultdict(int)
        for row in rows:
            if row["learned_refusal"]:
                refusals[row["learned_refusal"]] += 1
        lines.append(
            f"| {name} | {', '.join(short(n) for n in rows[0]['declared_set']) or '∅'} "
            f"| {share([row['declared_valid'] for row in rows if row['declared_valid'] is not None])} "
            f"| {'; '.join(f'{s} ({c})' for s, c in sorted(learned_sets.items(), key=lambda i: -i[1]))} "
            f"| {share([row['learned_valid'] for row in rows if row['learned_valid'] is not None])} "
            f"| {', '.join(f'{r} ({c})' for r, c in refusals.items()) or '–'} |"
        )
    return lines


def structure_table(structures: List[Dict[str, Any]]) -> List[str]:
    """
    :param structures: Structure rows.
    :return: Per setting and size, how close the learned graphs are to the true one.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in structures:
        groups[(setting_label(row["configuration"]), row["example_count"])].append(row)
    lines = [
        "| setting | examples | SHD | adjacency precision | adjacency recall | orientation precision | seconds to learn |",
        "|---|---|---|---|---|---|---|",
    ]
    for (setting, count), rows in sorted(
        groups.items(), key=lambda i: (str(i[0][0]), i[0][1])
    ):
        lines.append(
            f"| {setting} | {count} "
            f"| {mean([r['structural_hamming_distance'] for r in rows]):.1f} "
            f"| {np.nanmean([r['adjacency_precision'] for r in rows]):.2f} "
            f"| {np.nanmean([r['adjacency_recall'] for r in rows]):.2f} "
            f"| {np.nanmean([r['orientation_precision'] for r in rows]):.2f} "
            f"| {mean([r['seconds_to_learn'] for r in rows]):.1f} |"
        )
    return lines


def setting_label(configuration: Any) -> str:
    """
    :param configuration: A model setting as the runner stored it.
    :return: The setting written out.
    """
    if isinstance(configuration, dict):
        return ", ".join(f"{key}={value}" for key, value in configuration.items())
    return str(configuration)


def short(name: str) -> str:
    """
    :return: A variable's name without its class.
    """
    return name.split(".", 1)[1] if "." in name else name


def report(results: Dict[str, Dict[str, List[Any]]]) -> str:
    """
    :param results: Per study, its rows.
    :return: The Markdown report.
    """
    lines = []
    for study, rows in results.items():
        questions, structures = rows["questions"], rows["structures"]
        lines += [f"## {study}", "", "### Accuracy by example count", ""]
        lines += accuracy_table(questions, lambda q: q["example_count"])
        lines += ["", "### Accuracy by setting", ""]
        lines += accuracy_table(questions, lambda q: setting_label(q["configuration"]))
        lines += ["", "### Accuracy by question", ""]
        lines += accuracy_table(questions, lambda q: q["case_name"])
        lines += ["", "### Adjustment sets against the true graph", ""]
        lines += validity_table(questions)
        lines += ["", "### Learned graphs against the true graph", ""]
        lines += structure_table(structures)
        lines += [
            "",
            f"Choosing a learned set took {mean([q['seconds_to_choose'] for q in questions]):.3f} s per question on average.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    """
    Print the report of the files the command line names.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    print(report(load(parser.parse_args().paths)))


if __name__ == "__main__":
    main()
