"""Score one or more predictions.jsonl files and print a markdown table (optionally CSV).

    python -m physwm.eval.evaluate --data $WM/data --splits id_test ood_forces \
        --pred static=$WM/runs/baselines/static.jsonl zeroshot=$WM/runs/zs/pred.jsonl --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import math

from physwm.eval.metrics import load_predictions, summarize, validate_coverage
from physwm.schema import load_split

COLUMNS = [
    ("method", "{}"), ("split", "{}"), ("task", "{}"), ("episodes", "{}"), ("parse_rate", "{:.2f}"),
    ("missing_rate", "{:.3f}"),
    ("all_mean", "{:.3f}"), ("all_median", "{:.3f}"),
    ("target_mean", "{:.3f}"), ("target_median", "{:.3f}"), ("target_skill", "{:+.2f}"),
    ("moved_mean", "{:.3f}"), ("unmoved_mean", "{:.3f}"),
]


def fmt(v, f):
    if isinstance(v, float) and math.isnan(v):
        return "–"
    return f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    episodes = {ep.episode_id: ep for s in args.splits for _, ep in load_split(args.data, s)}
    rows = []
    for item in args.pred:
        name, path = item.split("=", 1)
        predictions = load_predictions(path)
        groups = {(episodes[p["episode_id"]].split, p["task"]) for p in predictions
                  if p["episode_id"] in episodes}
        if not groups:
            raise ValueError(f"no predictions for requested splits: {path}")
        for split, task in groups:
            cohort = {eid: ep for eid, ep in episodes.items() if ep.split == split}
            validate_coverage([p for p in predictions if p["task"] == task], cohort, (task,))
        for r in summarize(predictions, episodes):
            rows.append({"method": name, **r})
    rows.sort(key=lambda r: (r["task"], r["split"], r["method"]))

    print("errors in metres (camera-frame 3D centre L2); skill = 1 - err / err_static")
    print("| " + " | ".join(c for c, _ in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(fmt(r[c], f) for c, f in COLUMNS) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
