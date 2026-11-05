"""Metrics for grounded 3D centre prediction.

Prediction records (one JSON object per line in predictions.jsonl):
    {"episode_id": str, "task": "A" | "B",
     "objects": {"<object id>": [x, y, z] | null},   # camera frame (OpenCV), metres
     "parsed": bool, "raw": str (optional)}

Task A = current-state grounding (t = 0, perception ceiling).
Task B = future-state prediction (t = 1 s after the push).

A missing or unparsable object prediction is scored with MISSING_PENALTY metres.
Always report missing rate: this finite penalty can be lower than a very bad numeric answer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physwm import geometry as G
from physwm.schema import Episode

TASK_TIME = {"A": 0.0, "B": 1.0}
MISSING_PENALTY = 1.0  # m
MOVED_THRESH = 0.01  # m, GT displacement that counts as "moved"


def gt_centers_cam(ep: Episode, t: float) -> np.ndarray:
    pos = np.array([s.pos for s in ep.state_at(t)])
    return G.world_to_cam(pos, ep.camera.cam2world)


def load_predictions(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_object_rows(pred: dict, ep: Episode) -> list[dict]:
    """One row per object with its error and grouping flags."""
    t = TASK_TIME[pred["task"]]
    gt = gt_centers_cam(ep, t)
    gt0 = gt_centers_cam(ep, 0.0)
    rows = []
    for i in range(len(ep.objects)):
        p = pred.get("objects", {}).get(str(i))
        try:
            xyz = np.asarray(p, dtype=float)
            valid = xyz.shape == (3,) and bool(np.all(np.isfinite(xyz)))
        except (TypeError, ValueError):
            valid = False
        err = float(np.linalg.norm(xyz - gt[i])) if valid else MISSING_PENALTY
        rows.append({
            "err": err,
            "err_static": float(np.linalg.norm(gt0[i] - gt[i])),
            "missing": not valid,
            "target": i == ep.action.target,
            "moved": float(np.linalg.norm(gt[i] - gt0[i])) > MOVED_THRESH,
        })
    return rows


def validate_coverage(preds: list[dict], episodes: dict[str, Episode], tasks: tuple[str, ...]) -> None:
    """Fail instead of silently scoring a truncated, duplicated or empty cohort."""
    expected = {(eid, task) for eid in episodes for task in tasks}
    seen = set()
    for pred in preds:
        if pred["episode_id"] not in episodes:
            continue  # a prediction file may contain additional, unrequested splits
        key = (pred["episode_id"], pred["task"])
        if key in seen:
            raise ValueError(f"duplicate prediction: {key}")
        if key not in expected:
            raise ValueError(f"unexpected prediction task: {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"incomplete prediction coverage: {len(expected - seen)} episode/task records missing")


def summarize(preds: list[dict], episodes: dict[str, Episode]) -> list[dict]:
    """Aggregate per (split, task). Returns a list of result dicts."""
    groups: dict[tuple, list] = defaultdict(list)
    parsed: dict[tuple, list] = defaultdict(list)
    for p in preds:
        ep = episodes.get(p["episode_id"])
        if ep is None:
            continue
        key = (ep.split, p["task"])
        groups[key] += per_object_rows(p, ep)
        parsed[key].append(bool(p.get("parsed", True)))

    out = []
    for (split, task), rows in sorted(groups.items()):
        def agg(sel):
            sub = [r for r in rows if sel(r)]
            if not sub:
                return dict(n=0, mean=np.nan, median=np.nan, skill=np.nan)
            e = np.array([r["err"] for r in sub])
            s = np.array([r["err_static"] for r in sub])
            skill = 1 - e.mean() / s.mean() if s.mean() > 0 else np.nan
            return dict(n=len(sub), mean=float(e.mean()), median=float(np.median(e)), skill=float(skill))

        res = {"split": split, "task": task, "episodes": len(parsed[(split, task)]),
               "parse_rate": float(np.mean(parsed[(split, task)])),
               "missing_rate": float(np.mean([r["missing"] for r in rows]))}
        for name, sel in (
            ("all", lambda r: True),
            ("target", lambda r: r["target"]),
            ("moved", lambda r: r["moved"]),
            ("unmoved", lambda r: not r["moved"]),
        ):
            for k, v in agg(sel).items():
                res[f"{name}_{k}"] = v
        out.append(res)
    return out
