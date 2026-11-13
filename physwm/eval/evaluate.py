"""Score one or more predictions.jsonl files and print a markdown table (optionally CSV).

    python -m physwm.eval.evaluate --data $WM/data --splits id_test ood_forces \
        --pred static=$WM/runs/baselines/static.jsonl zeroshot=$WM/runs/zs/pred.jsonl --csv out.csv

Records that carry ``extents`` (centre + extents answer format) additionally get the
``ext_l2`` and ``iou`` columns; files without extents keep the original table. ``ext_l2`` /
``iou`` are penalized (missing objects enter with the 1 m penalty / IoU 0); ``ext_l2_obs`` /
``iou_obs`` average valid predictions only, and ``target_missing`` is the fraction of
episodes whose target prediction is missing/invalid.

Task N is scored at the first stored frame after the record's ``frames`` (legacy records
without the field default to ``(0,)``, i.e. t = 0.1 s).
"""

from __future__ import annotations

import argparse
import csv
import math

from physwm.eval.boxmetrics import summarize_boxes
from physwm.eval.metrics import load_predictions, summarize, validate_coverage
from physwm.schema import load_split

COLUMNS = [
    ("method", "{}"), ("split", "{}"), ("task", "{}"), ("episodes", "{}"), ("parse_rate", "{:.2f}"),
    ("missing_rate", "{:.3f}"),
    ("all_mean", "{:.3f}"), ("all_median", "{:.3f}"),
    ("target_mean", "{:.3f}"), ("target_median", "{:.3f}"), ("target_skill", "{:+.2f}"),
    ("moved_mean", "{:.3f}"), ("unmoved_mean", "{:.3f}"),
]
BOX_COLUMNS = [("ext_l2", "{:.3f}"), ("iou", "{:.3f}"), ("ext_l2_obs", "{:.3f}"),
               ("iou_obs", "{:.3f}"), ("target_missing", "{:.3f}")]
BOX_KEYS = tuple(c for c, _ in BOX_COLUMNS)


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
    rows, has_boxes = [], False
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
        boxes = ({(b["split"], b["task"]): b for b in summarize_boxes(predictions, episodes)}
                 if any("extents" in p for p in predictions) else {})
        for r in summarize(predictions, episodes):
            box = boxes.get((r["split"], r["task"]))
            if box is not None:
                has_boxes = True
                r = {**r, **{k: box[k] for k in BOX_KEYS}}
            rows.append({"method": name, **r})
    if has_boxes:  # keep the CSV columns rectangular across methods
        for r in rows:
            for k in BOX_KEYS:
                r.setdefault(k, float("nan"))
    rows.sort(key=lambda r: (r["task"], r["split"], r["method"]))
    columns = COLUMNS + (BOX_COLUMNS if has_boxes else [])

    print("errors in metres (camera-frame 3D centre L2); skill = 1 - err / err_static")
    print("| " + " | ".join(c for c, _ in columns) + " |")
    print("|" + "---|" * len(columns))
    for r in rows:
        print("| " + " | ".join(fmt(r[c], f) for c, f in columns) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
