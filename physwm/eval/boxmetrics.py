"""Metrics for centre + extents 3D box predictions (phase-2 answer format).

Prediction records extend ``predictions.jsonl`` with ``"extents"``::

    {"episode_id": str, "task": "A" | "B" | "N",
     "objects": {"<object id>": [x, y, z] | null},   # centres, camera frame, metres
     "extents": {"<object id>": [ex, ey, ez] | null},  # full body-axis sizes
     "parsed": bool, "raw": str (optional)}

Per object we report the centre L2 (identical to :mod:`physwm.eval.metrics`), the extents
L2 in metres and the IoU of the two boxes after transforming the predicted centre into the
GT body frame (body-axis-aligned boxes there; exact axis-aligned intersection/union).
Missing predictions use the same fixed :data:`MISSING_PENALTY` for both L2 terms and score
IoU 0. Aggregates are per (split, task) means with an episode-level percentile bootstrap CI.

Penalized vs. observed columns
------------------------------
``ext_l2`` / ``iou`` are the penalized means: a missing or invalid prediction enters with
:data:`MISSING_PENALTY` metres (or IoU 0), which conflates omission with numeric error.
``ext_l2_obs`` / ``iou_obs`` are means over objects with a valid prediction only and never
apply the penalty. ``target_missing`` is the fraction of episodes whose TARGET object
prediction is missing/invalid, per task.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from physwm import geometry as G
from physwm.eval.metrics import MISSING_PENALTY, gt_centers_cam, gt_extents, task_time
from physwm.schema import Episode


def _xyz(value) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.shape == (3,) and bool(np.all(np.isfinite(arr))) else None


def _volume_intersection(a_lo, a_hi, b_lo, b_hi) -> tuple[float, float, float]:
    inter = float(np.prod(np.clip(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0, None)))
    vol_a = float(np.prod(np.clip(a_hi - a_lo, 0.0, None)))
    vol_b = float(np.prod(np.clip(b_hi - b_lo, 0.0, None)))
    return inter, vol_a, vol_b


def aligned_iou(pred_centre_cam: np.ndarray, pred_extents, gt_centre_cam: np.ndarray,
                gt_extents_xyz, obj_quat, cam2world) -> float:
    """IoU of body-axis-aligned boxes, in the GT object's body frame.

    ``p_body = R_obj^T R_cam (p_cam - c_cam)`` puts the predicted centre in the GT body
    frame; the GT box is then centred at the origin with half extents
    ``gt_extents / 2`` and the prediction keeps its (unrotated) extents.
    """
    R_cam = np.asarray(cam2world, dtype=float)[:3, :3]
    R_obj = G.quat_to_mat(obj_quat)
    p_body = R_obj.T @ (R_cam @ (np.asarray(pred_centre_cam, float) - np.asarray(gt_centre_cam, float)))
    half_pred = 0.5 * np.asarray(pred_extents, dtype=float)
    half_gt = 0.5 * np.asarray(gt_extents_xyz, dtype=float)
    inter, vol_a, vol_b = _volume_intersection(
        p_body - half_pred, p_body + half_pred, -half_gt, half_gt
    )
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def per_object_box_rows(pred: dict, ep: Episode, t: float | None = None,
                        frames: tuple[int, ...] | None = None) -> list[dict]:
    """One row per object: centre L2, extents L2, body-frame IoU, plus grouping flags.

    Task N is resolved from the record's ``frames`` (legacy records default to ``(0,)``,
    i.e. t = 0.1 s) unless an explicit ``t`` or ``frames`` is passed.
    """
    frames = (tuple(pred.get("frames", (0,)) or (0,)) if frames is None
              else tuple(frames) or (0,))
    t = task_time(ep, pred["task"], frames) if t is None else t
    gt = gt_centers_cam(ep, t)
    gt0 = gt_centers_cam(ep, 0.0)
    gt_ext = gt_extents(ep)
    pred_c = pred.get("objects", {})
    pred_e = pred.get("extents", {})
    rows = []
    for i, obj in enumerate(ep.objects):
        c = _xyz(pred_c.get(str(i)))
        e = _xyz(pred_e.get(str(i)))
        centre_l2 = float(np.linalg.norm(c - gt[i])) if c is not None else MISSING_PENALTY
        ext_l2 = float(np.linalg.norm(e - gt_ext[i])) if e is not None else MISSING_PENALTY
        iou = (aligned_iou(c, e, gt[i], gt_ext[i], ep.state_at(t)[i].quat, ep.camera.cam2world)
               if c is not None and e is not None else 0.0)
        rows.append({
            "centre_l2": centre_l2,
            "ext_l2": ext_l2,
            "iou": iou,
            "err_static": float(np.linalg.norm(gt0[i] - gt[i])),
            "missing": c is None,
            "missing_extents": e is None,
            "target": i == ep.action.target,
            "moved": float(np.linalg.norm(gt[i] - gt0[i])) > 0.01,
        })
    return rows


def _bootstrap_ci(values, repetitions: int = 1000, seed: int = 0) -> list[float] | None:
    """Percentile bootstrap CI of the mean, resampling episodes (``values`` per episode)."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(len(x), size=(repetitions, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def summarize_boxes(preds: list[dict], episodes: dict[str, Episode], repetitions: int = 1000,
                    seed: int = 0) -> list[dict]:
    """Per (split, task) means and bootstrap CIs of centre L2, extents L2 and IoU.

    ``ext_l2``/``iou`` are penalized (missing objects enter as :data:`MISSING_PENALTY` /
    IoU 0); ``ext_l2_obs``/``iou_obs`` average valid predictions only, and
    ``target_missing`` is the per-episode fraction of missing target predictions.
    """
    groups: dict[tuple, list[list[dict]]] = defaultdict(list)  # per episode rows
    parsed: dict[tuple, list[bool]] = defaultdict(list)
    for p in preds:
        ep = episodes.get(p["episode_id"])
        if ep is None or "extents" not in p:
            continue
        key = (ep.split, p["task"])
        groups[key].append(per_object_box_rows(p, ep))
        parsed[key].append(bool(p.get("parsed", True)))

    out = []
    for (split, task), per_ep in sorted(groups.items()):
        rows = [r for ep_rows in per_ep for r in ep_rows]

        def mean(name):
            return float(np.mean([r[name] for r in rows])) if rows else float("nan")

        def obs_mean(name, valid):
            vals = [r[name] for r in rows if valid(r)]
            return float(np.mean(vals)) if vals else float("nan")

        out.append({
            "split": split, "task": task, "episodes": len(per_ep),
            "parse_rate": float(np.mean(parsed[(split, task)])),
            "missing_rate": float(np.mean([r["missing"] for r in rows])),
            "missing_extents_rate": float(np.mean([r["missing_extents"] for r in rows])),
            "centre_l2": mean("centre_l2"), "ext_l2": mean("ext_l2"), "iou": mean("iou"),
            "ext_l2_obs": obs_mean("ext_l2", lambda r: not r["missing_extents"]),
            "iou_obs": obs_mean("iou", lambda r: not r["missing"] and not r["missing_extents"]),
            "target_missing": float(np.mean([
                1.0 if next(r["missing"] for r in ep_rows if r["target"]) else 0.0
                for ep_rows in per_ep
            ])),
            "centre_l2_ci95": _bootstrap_ci([np.mean([r["centre_l2"] for r in ep_rows])
                                             for ep_rows in per_ep], repetitions, seed),
            "ext_l2_ci95": _bootstrap_ci([np.mean([r["ext_l2"] for r in ep_rows])
                                          for ep_rows in per_ep], repetitions, seed),
            "iou_ci95": _bootstrap_ci([np.mean([r["iou"] for r in ep_rows])
                                       for ep_rows in per_ep], repetitions, seed),
        })
    return out