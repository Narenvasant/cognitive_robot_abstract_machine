"""Summarise domino_study.json into markdown tables. Usage: summarize_domino.py <json>"""
import json
import sys
from collections import defaultdict

import numpy as np

rows = json.load(open(sys.argv[1]))
methods = ["new", "current", "plain"]
by_setting = defaultdict(list)
for row in rows:
    by_setting[(row["question"], row["spillover"], row["scene_count"])].append(row)

print("| question | q | scenes | true contrast | MAE new | MAE current | MAE plain conditioning | contrast new (mean ± sd over 3) | new beats current |")
print("|---|---|---|---|---|---|---|---|---|")
for (question, spillover, scene_count), group in sorted(by_setting.items()):
    truth = np.array(group[0]["truth"])
    mae = {
        method: np.nanmean([np.mean(np.abs(np.array(row[method]) - truth)) for row in group])
        for method in methods
    }
    contrasts = np.array([row["new"][1] - row["new"][0] for row in group])
    print(
        f"| {question} | {spillover} | {scene_count} | {truth[1] - truth[0]:+.3f} | "
        f"{mae['new']:.3f} | {mae['current']:.3f} | {mae['plain']:.3f} | "
        f"{contrasts.mean():+.3f} ± {contrasts.std(ddof=1):.3f} | "
        f"{'yes' if mae['new'] < mae['current'] else 'no'} |"
    )

print()
print("No invented spillover (q = 0): 95% interval of the new model's contrast over the 3 repeats")
for (question, spillover, scene_count), group in sorted(by_setting.items()):
    if spillover != 0.0 or question == "no_touching_pairs":
        continue
    contrasts = np.array([row["new"][1] - row["new"][0] for row in group])
    half_width = 4.30 * contrasts.std(ddof=1) / np.sqrt(len(contrasts))
    low, high = contrasts.mean() - half_width, contrasts.mean() + half_width
    print(f"- {question}, {scene_count} scenes: [{low:+.3f}, {high:+.3f}] contains 0: {low <= 0 <= high}")
