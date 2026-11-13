"""Paired motion-attribution read-outs on saved prediction files (CPU only).

For the pushed target of every episode with valid predictions:

- ``err_B``         ||B̂ − GT₁||                 model endpoint error at t = 1 s
- ``err_ownstatic`` ||Â − GT₁||                 hold the perceived t = 0 centre as the t = 1 s answer
- ``err_disp``      ||GT₀ + (B̂ − Â) − GT₁||     privileged anchor + the model's predicted displacement
- ``err_N``         ||N̂ − GT||                 next-frame error at t = frame_times[1]
- ``err_Nstatic``   ||Â − GTₙ||                 hold-Â at the next-frame horizon
- ``err_gtstatic``  ||GT₀ − GT₁||               oracle perception held static (t = 1 s)
- ``err_gtstaticN`` ||GT₀ − GTₙ||               oracle perception held static (next frame)

Paired episode-level bootstrap CIs; the printed differences (B − own-static,
disp − B, N − N-static) isolate motion information from perception error without
subtracting aggregate means (research/PLAN.md reading rule 4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physwm import geometry as G
from physwm.schema import load_split

KEYS = ("err_B", "err_ownstatic", "err_gtstatic", "err_disp",
        "err_N", "err_Nstatic", "err_gtstaticN")
DIFFS = (("B - own-static", "err_B", "err_ownstatic"),
         ("B - GT-static", "err_B", "err_gtstatic"),
         ("disp - B (privilege the anchor)", "err_disp", "err_B"),
         ("N - own-static @ next frame", "err_N", "err_Nstatic"),
         ("N - GT-static @ next frame", "err_N", "err_gtstaticN"))


def _gt_cam(ep, t: float, oid: int) -> np.ndarray:
    pos = np.array([s.pos for s in ep.state_at(t)], float)
    return G.world_to_cam(pos, ep.camera.cam2world)[oid]


def target_rows(ep, recs: list[dict]) -> dict:
    """Per-episode paired errors for the pushed target; empty dict if A or B missing."""
    oid = ep.action.target
    got = {r["task"]: r["objects"].get(str(oid)) for r in recs}
    a, b, n = got.get("A"), got.get("B"), got.get("N")
    out: dict = {}
    if a is not None and b is not None:
        a_v, b_v = np.asarray(a, float), np.asarray(b, float)
        gt0, gt1 = _gt_cam(ep, 0.0, oid), _gt_cam(ep, 1.0, oid)
        out["err_B"] = float(np.linalg.norm(b_v - gt1))
        out["err_ownstatic"] = float(np.linalg.norm(a_v - gt1))
        out["err_gtstatic"] = float(np.linalg.norm(gt0 - gt1))
        out["err_disp"] = float(np.linalg.norm(gt0 + (b_v - a_v) - gt1))
    if a is not None and n is not None:
        tn = ep.frame_times[1]
        gtn, gt0n = _gt_cam(ep, tn, oid), _gt_cam(ep, 0.0, oid)
        out["err_N"] = float(np.linalg.norm(np.asarray(n, float) - gtn))
        out["err_Nstatic"] = float(np.linalg.norm(np.asarray(a, float) - gtn))
        out["err_gtstaticN"] = float(np.linalg.norm(gt0n - gtn))
    return out


def _ci(rng: np.random.Generator, vals: list[float], boot: int) -> tuple[float, float]:
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(boot, len(v)))
    return tuple(float(x) for x in np.percentile(v[idx].mean(axis=1), [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--splits", nargs="+",
                    default=["val_id", "val_ood_objects", "val_ood_forces", "val_ood_config"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    by_ep: dict = {}
    for line in Path(a.pred).read_text().splitlines():
        r = json.loads(line)
        by_ep.setdefault(r["episode_id"], []).append(r)
    rng = np.random.default_rng(a.seed)
    print("| split | quantity | mean | 95% CI | n |")
    print("|---|---|---|---|---|")
    for split in a.splits:
        rows = []
        for _, ep in load_split(a.data, split):
            recs = by_ep.get(ep.episode_id)
            if recs:
                row = target_rows(ep, recs)
                if row:
                    rows.append(row)
        if not rows:
            print(f"| {split} | (no paired predictions) | - | - | 0 |")
            continue
        for k in KEYS:
            v = [r[k] for r in rows if k in r]
            if v:
                lo, hi = _ci(rng, v, a.boot)
                print(f"| {split} | {k} | {np.mean(v):.4f} | [{lo:.4f}, {hi:.4f}] | {len(v)} |")
        for name, x, y in DIFFS:
            v = [r[x] - r[y] for r in rows if x in r and y in r]
            if v:
                lo, hi = _ci(rng, v, a.boot)
                print(f"| {split} | {name} | {np.mean(v):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {len(v)} |")


if __name__ == "__main__":
    main()
