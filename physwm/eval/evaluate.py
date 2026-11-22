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
from physwm.eval.idpmetrics import summarize_ctx, summarize_idp
from physwm.eval.metrics import load_predictions, summarize, validate_coverage
from physwm.schema import load_split
from physwm.vlm.prompts import IDP_TASK

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
IDP_COLUMNS = [("method", "{}"), ("split", "{}"), ("episodes", "{}"), ("parse_rate", "{:.2f}"),
               ("exact_acc", "{:.2f}"), ("target_acc", "{:.2f}"), ("direction_acc", "{:.2f}"),
               ("strength_acc", "{:.2f}"), ("no_motion_acc", "{:.2f}"),
               ("n_action_gt", "{}"), ("n_no_motion_gt", "{}")]
CTX_COLUMNS = [("method", "{}"), ("split", "{}"), ("episodes", "{}"), ("ctx_parse_rate", "{:.2f}"),
               ("ctx_target_acc", "{:.2f}"), ("ctx_direction_acc", "{:.2f}"),
               ("ctx_obstacles_acc", "{:.2f}"), ("ctx_edge_mae_m", "{:.3f}")]


def fmt(v, f):
    if isinstance(v, float) and math.isnan(v):
        return "–"
    return f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--limit", type=int, default=None, help="episodes per split, as in inference")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    episodes = {ep.episode_id: ep for s in args.splits
                for _, ep in load_split(args.data, s)[:args.limit]}
    rows, idp_rows, ctx_rows, has_boxes = [], [], [], False
    for item in args.pred:
        name, path = item.split("=", 1)
        predictions = load_predictions(path)
        idp = [p for p in predictions if p.get("task") == IDP_TASK]
        core = [p for p in predictions if p.get("task") != IDP_TASK]
        groups = {(episodes[p["episode_id"]].split, p["task"]) for p in core
                  if p["episode_id"] in episodes}
        if not groups and not idp:
            raise ValueError(f"no predictions for requested splits: {path}")
        for split, task in groups:
            cohort = {eid: ep for eid, ep in episodes.items() if ep.split == split}
            validate_coverage([p for p in core if p["task"] == task], cohort, (task,))
        for split in {episodes[p["episode_id"]].split for p in idp if p["episode_id"] in episodes}:
            cohort = {eid: ep for eid, ep in episodes.items() if ep.split == split}
            validate_coverage(idp, cohort, (IDP_TASK,))
        boxes = ({(b["split"], b["task"]): b for b in summarize_boxes(core, episodes)}
                 if any("extents" in p for p in core) else {})
        for r in summarize(core, episodes):
            box = boxes.get((r["split"], r["task"]))
            if box is not None:
                has_boxes = True
                r = {**r, **{k: box[k] for k in BOX_KEYS}}
            rows.append({"method": name, **r})
        idp_rows += [{"method": name, **r} for r in summarize_idp(idp, episodes)]
        ctx_rows += [{"method": name, **r} for r in summarize_ctx(predictions, episodes)]
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
    if idp_rows:
        print("\nIDP accuracy (auxiliary task sanity; not a gate -- research/PREREG.md gates on Task B)")
        print("| " + " | ".join(c for c, _ in IDP_COLUMNS) + " |")
        print("|" + "---|" * len(IDP_COLUMNS))
        for r in idp_rows:
            print("| " + " | ".join(fmt(r[c], f) for c, f in IDP_COLUMNS) + " |")
    if ctx_rows:
        print("\npredicted action context (ctx3d diagnostic; pairs with --oracle-context: "
              "predicted vs simulator-provided context)")
        print("| " + " | ".join(c for c, _ in CTX_COLUMNS) + " |")
        print("|" + "---|" * len(CTX_COLUMNS))
        for r in ctx_rows:
            print("| " + " | ".join(fmt(r[c], f) for c, f in CTX_COLUMNS) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
