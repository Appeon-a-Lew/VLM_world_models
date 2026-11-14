"""IDEA-103: is the PE-on grounding error a surface read-out? (CPU only)

Decomposes every Task-A (t = 0) per-object centre error into a component along the
camera view ray and a tangential component, and compares the signed radial error with
an analytic proxy for the visible-surface-to-centre offset.

Geometry and sign convention
----------------------------
Predictions and GT centres are in the OpenCV camera frame (x right, y down, z
forward); the camera is at the origin, so the view ray of a GT centre ``c`` is
``r = c / ||c||`` (pointing *away* from the camera). With ``e = pred - c``:

- ``rad = e . r``  signed radial error. Moving **towards the camera** gives
  ``rad < 0``.
- ``tan = ||e - rad * r||`` tangential error.

Surface proxy. In the body frame the box surface closest to the camera along the ray
is the support plane ``min_i (k_i . r_body) = -sum_a h_a |r_body_a|`` over the eight
corners ``k_i`` of the half-extent box. Define the positive distance

    d_near = sum_a h_a |r_body_a| >= 0,

i.e. the distance from the body centre to the nearest body-surface point along the
ray (exact for boxes; an axis-aligned-box proxy for the other shapes). A pure
visible-surface read-out then predicts

    pred = c - d_near * r   =>   rad = -d_near,  tan = 0,  rad / d_near = -1,

while a body-centre read-out predicts ``rad = 0`` (``rad / d_near = 0``). The
report additionally lists ``-rad / d_near = +1`` under the same surface hypothesis
so the sign is unambiguous either way.

Per-arm and per-split means carry episode-level percentile bootstrap CIs (objects
within an episode are clustered), and the ``s_f1 - s_f1_nope`` radial difference is
bootstrapped paired on episodes using object ids present in both arms.

Usage::

    python -m physwm.eval.analyze_pe_surface \
        --pred-dir /tmp/opencode/predsdl/site/v3/api/pred \
        --meta-cache /tmp/opencode/metas \
        --splits val_id val_ood_config \
        --arm s_f1 --arm s_f1_nope
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from physwm import geometry as G
from physwm.schema import Episode

HF_META_URL = (
    "https://huggingface.co/datasets/Appeon-a-Lew/physwm-tabletop-pushes"
    "/resolve/main/v3/{split}/{episode}/meta.json"
)
DEFAULT_SPLITS = ("val_id", "val_ood_config")
DEFAULT_ARMS = ("s_f1", "s_f1_nope")
PAIR = ("s_f1", "s_f1_nope")


# --------------------------------------------------------------------------- geometry
def error_components(pred, gt) -> tuple[float, float, float]:
    """(||e||, rad, tan) for camera-frame ``pred`` and ``gt`` (camera at origin)."""
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    e = pred - gt
    r = gt / np.linalg.norm(gt)
    rad = float(e @ r)
    tan = float(np.linalg.norm(e - rad * r))
    return float(np.linalg.norm(e)), rad, tan


def near_surface_offset(quat, half_extents, cam2world, gt_cam) -> float:
    """Distance from the body centre to the nearest surface point along the view ray.

    Exact for boxes (support function of the half-extent box); a box proxy for the
    corpus' other primitives.
    """
    gt_cam = np.asarray(gt_cam, float)
    r = gt_cam / np.linalg.norm(gt_cam)
    r_wc = np.asarray(cam2world, float)[:3, :3].T          # world -> camera rotation
    r_bc = r_wc @ G.quat_to_mat(quat)                      # body -> camera rotation
    r_body = r_bc.T @ r
    h = np.abs(np.asarray(half_extents, float))
    return float(np.sum(h * np.abs(r_body)))


# --------------------------------------------------------------------------- metadata
def _fetch_one(args: tuple[str, str, Path]) -> tuple[str, str, bool]:
    split, episode, path = args
    if path.exists() and path.stat().st_size > 0:
        return split, episode, True
    url = HF_META_URL.format(split=split, episode=episode)
    last: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "physwm-idea103/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            json.loads(data)  # validate before caching
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return split, episode, True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    print(f"[warn] failed to fetch {split}/{episode}: {last}", file=sys.stderr)
    return split, episode, False


def ensure_metas(splits, episodes_by_split, cache: Path, workers: int) -> dict[str, dict[str, Path]]:
    """Download missing ``meta.json`` files to ``cache/<split>/<episode>.json``."""
    jobs = [
        (split, eid, cache / split / f"{eid}.json")
        for split in splits
        for eid in sorted(episodes_by_split.get(split, ()))
    ]
    missing = [j for j in jobs if not (j[2].exists() and j[2].stat().st_size > 0)]
    if missing:
        print(f"[meta] fetching {len(missing)} of {len(jobs)} (workers={workers})", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_fetch_one, missing))
    ok: dict[str, dict[str, Path]] = {s: {} for s in splits}
    for split, eid, path in jobs:
        if path.exists() and path.stat().st_size > 0:
            ok[split][eid] = path
    return ok


# --------------------------------------------------------------------------- records
def load_arm_rows(pred_dir: Path, splits, metas, arm: str) -> list[dict]:
    """One row per object with a valid Task-A prediction and a cached meta."""
    rows: list[dict] = []
    for split in splits:
        split_dir = pred_dir / arm / split
        if not split_dir.is_dir():
            print(f"[warn] missing pred split: {split_dir}", file=sys.stderr)
            continue
        for eid, meta_path in sorted(metas.get(split, {}).items()):
            pred_path = split_dir / f"{eid}.json"
            if not pred_path.exists():
                continue
            try:
                recs = json.loads(pred_path.read_text())
                ep = Episode.load(meta_path)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"[warn] skip {arm}/{split}/{eid}: {exc}", file=sys.stderr)
                continue
            arec = next((r for r in recs if r.get("task") == "A"), None)
            if not arec or not arec.get("objects") or not arec.get("parsed", True):
                continue
            pos_world = np.array([s.pos for s in ep.state_at(0.0)], float)
            pos_cam = G.world_to_cam(pos_world, ep.camera.cam2world)
            ep_rows = []
            for oid, obj in enumerate(ep.objects):
                raw = arec["objects"].get(str(oid))
                if raw is None:
                    continue
                try:
                    pred = np.asarray(raw, float)
                except (TypeError, ValueError):
                    continue
                if pred.shape != (3,) or not np.all(np.isfinite(pred)):
                    continue
                gt = pos_cam[oid]
                err, rad, tan = error_components(pred, gt)
                st = ep.state_at(0.0)[oid]
                d_near = near_surface_offset(st.quat, obj.half_extents,
                                             ep.camera.cam2world, gt)
                ep_rows.append({
                    "arm": arm, "split": split, "episode": eid, "oid": oid,
                    "shape": obj.shape, "err": err, "rad": rad, "tan": tan,
                    "d_near": d_near,
                    "pred_dist": float(np.linalg.norm(pred)),
                    "gt_dist": float(np.linalg.norm(gt)),
                    "_pred": pred, "_gt": gt,
                })
            # Per-episode range scale: pred = s_ep * gt (no intercept), least squares.
            if ep_rows:
                p = np.array([r["_pred"] for r in ep_rows])
                g = np.array([r["_gt"] for r in ep_rows])
                s = float(np.sum(p * g) / np.sum(g * g))
                for r in ep_rows:
                    r["ep_scale"] = s
                    r["scale_resid"] = float(np.linalg.norm(r["_pred"] - s * r["_gt"]))
            rows.extend(ep_rows)
    for r in rows:
        del r["_pred"], r["_gt"]
    return rows


# --------------------------------------------------------------------------- stats
def cluster_bootstrap(values_by_ep: dict[str, list[float]], boot: int, seed: int):
    """Mean over objects with an episode-clustered percentile bootstrap CI."""
    eids = sorted(values_by_ep)
    if not eids:
        return float("nan"), (float("nan"), float("nan")), 0, 0
    means = np.array([np.mean(values_by_ep[e]) for e in eids], float)
    counts = np.array([len(values_by_ep[e]) for e in eids], float)
    point = float(np.sum(means * counts) / np.sum(counts))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(eids), size=(boot, len(eids)))
    w = counts[idx]
    sims = np.sum(means[idx] * w, axis=1) / np.sum(w, axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return point, (float(lo), float(hi)), len(eids), int(np.sum(counts))


def paired_rad_diff(rows_by_arm, arm_a: str, arm_b: str, split: str, boot: int, seed: int):
    """Paired episode bootstrap of mean rad(arm_a) - mean rad(arm_b)."""
    def per_ep(arm):
        out: dict[str, dict[int, float]] = {}
        for r in rows_by_arm.get(arm, []):
            if r["split"] == split:
                out.setdefault(r["episode"], {})[r["oid"]] = r["rad"]
        return out
    a, b = per_ep(arm_a), per_ep(arm_b)
    eids = sorted(set(a) & set(b))
    diffs, counts = [], []
    for eid in eids:
        common = sorted(set(a[eid]) & set(b[eid]))
        if not common:
            continue
        diffs.append(np.mean([a[eid][o] - b[eid][o] for o in common]))
        counts.append(len(common))
    if not diffs:
        return float("nan"), (float("nan"), float("nan")), 0, 0
    diffs = np.array(diffs, float)
    counts = np.array(counts, float)
    point = float(np.sum(diffs * counts) / np.sum(counts))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(boot, len(diffs)))
    w = counts[idx]
    sims = np.sum(diffs[idx] * w, axis=1) / np.sum(w, axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return point, (float(lo), float(hi)), int(np.sum(counts > 0)), int(np.sum(counts))


def slope_rad_on_dnear(rows: list[dict], boot: int, seed: int):
    """OLS slope of signed rad on d_near, episode-clustered bootstrap CI."""
    if len(rows) < 2:
        return float("nan"), (float("nan"), float("nan"))
    x = np.array([r["d_near"] for r in rows], float)
    y = np.array([r["rad"] for r in rows], float)
    slope = float(np.polyfit(x, y, 1)[0]) if np.var(x) > 0 else float("nan")
    by_ep: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_ep.setdefault(r["episode"], []).append(i)
    eids = sorted(by_ep)
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(boot):
        pick = rng.integers(0, len(eids), size=len(eids))
        ii = [i for e in pick for i in by_ep[eids[e]]]
        xx, yy = x[ii], y[ii]
        sims.append(np.polyfit(xx, yy, 1)[0] if np.var(xx) > 0 else np.nan)
    sims = np.asarray(sims, float)
    lo, hi = np.nanpercentile(sims, [2.5, 97.5])
    return slope, (float(lo), float(hi))


def mechanism_stats(rows: list[dict], boot: int, seed: int) -> dict:
    """Diagnostics separating per-object surface tracking from a per-episode depth scale."""
    by_ep: dict[str, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["episode"], []).append(r)
    eids = sorted(by_ep)
    corr, s_vals, ep_dp, ep_dg, range_corr = [], [], [], [], []
    for eid in eids:
        rs = by_ep[eid]
        rad = np.array([r["rad"] for r in rs])
        dn = np.array([r["d_near"] for r in rs])
        dp = np.array([r["pred_dist"] for r in rs])
        dg = np.array([r["gt_dist"] for r in rs])
        s_vals.append(float(np.sum(dp) / np.sum(dg)))
        ep_dp.append(float(dp.mean()))
        ep_dg.append(float(dg.mean()))
        if len(rs) > 1 and np.var(dn) > 0 and np.var(rad) > 0:
            corr.append(float(np.corrcoef(rad, dn)[0, 1]))
        if len(rs) > 1 and np.var(dp) > 0 and np.var(dg) > 0:
            range_corr.append(float(np.corrcoef(dp, dg)[0, 1]))

    def within_slope(ids):
        num = den = 0.0
        for eid in ids:
            rs = by_ep[eid]
            if len(rs) < 2:
                continue
            x = np.array([r["d_near"] for r in rs])
            y = np.array([r["rad"] for r in rs])
            xc, yc = x - x.mean(), y - y.mean()
            num += float(xc @ yc)
            den += float(xc @ xc)
        return num / den if den > 0 else float("nan")

    point = within_slope(eids)
    rng = np.random.default_rng(seed)
    sims = [within_slope([eids[i] for i in rng.integers(0, len(eids), len(eids))])
            for _ in range(boot)]
    sims = np.asarray(sims, float)
    lo, hi = np.nanpercentile(sims, [2.5, 97.5])
    rad2 = np.array([r["rad"] ** 2 for r in rows])
    err2 = np.array([r["err"] ** 2 for r in rows])
    return {
        "radial_share": float(np.mean(rad2) / np.mean(err2)),
        "within_corr_rad_dnear": float(np.mean(corr)) if corr else float("nan"),
        "within_slope": point, "within_slope_ci": (float(lo), float(hi)),
        "ep_scale_mean": float(np.mean(s_vals)),
        "ep_scale_sd": float(np.std(s_vals)),
        "ep_mean_pred_dist_sd": float(np.std(ep_dp)),
        "ep_mean_gt_dist_sd": float(np.std(ep_dg)),
        "within_corr_pred_gt_dist": float(np.mean(range_corr)) if range_corr else float("nan"),
        "scale_resid_rms": float(np.nanmean([r.get("scale_resid", np.nan) for r in rows])),
        "surface_pred_err": float(np.mean([r["d_near"] for r in rows])),
    }


def _fmt(mean: float, ci, digits: int = 4) -> str:
    return f"{mean:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def summarize(rows: list[dict], boot: int, seed: int) -> dict:
    def by_ep(key):
        d: dict[str, list[float]] = {}
        for r in rows:
            d.setdefault(r["episode"], []).append(r[key])
        return d

    out = {}
    for key in ("err", "rad", "tan", "abs_rad", "d_near", "neg_rad"):
        if key == "abs_rad":
            d = by_ep("rad")
            d = {k: [abs(v) for v in vals] for k, vals in d.items()}
        elif key == "neg_rad":
            d = by_ep("rad")
            d = {k: [-v for v in vals] for k, vals in d.items()}
        else:
            d = by_ep(key)
        out[key] = cluster_bootstrap(d, boot, seed)
    out["slope"], out["slope_ci"] = slope_rad_on_dnear(rows, boot, seed)
    out["frac_neg_rad"] = float(np.mean([r["rad"] < 0 for r in rows])) if rows else float("nan")
    out["n"] = len(rows)
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default="/tmp/opencode/predsdl/site/v3/api/pred")
    ap.add_argument("--meta-cache", default="/tmp/opencode/metas")
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--arm", action="append", default=None,
                    help="arm(s) to analyse; repeatable (default: s_f1 and s_f1_nope)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-fetch", action="store_true", help="never touch the network")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    arms = args.arm or list(DEFAULT_ARMS)
    pred_dir = Path(args.pred_dir)
    cache = Path(args.meta_cache)
    # union across arms so paired episodes survive
    merged: dict[str, set[str]] = {s: set() for s in args.splits}
    for split in args.splits:
        for arm in arms:
            merged[split] |= {p.stem for p in (pred_dir / arm / split).glob("*.json")}
    metas = {s: {} for s in args.splits} if args.no_fetch else \
        ensure_metas(args.splits, merged, cache, args.workers)
    if args.no_fetch:
        for split in args.splits:
            if (cache / split).is_dir():
                metas[split] = {p.stem: p for p in (cache / split).glob("*.json")}

    rows_by_arm = {arm: load_arm_rows(pred_dir, args.splits, metas, arm) for arm in arms}

    print("## Per-arm error geometry\n")
    print("| arm | split | n_obj | n_ep | mean \\|e\\| (m) | mean rad (m) | mean tan (m) "
          "| mean \\|rad\\| (m) | mean d_near (m) | rad/d_near | -rad/d_near "
          "| slope rad~d_near | frac rad<0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary: dict = {}
    for arm in arms:
        for split in args.splits:
            rows = [r for r in rows_by_arm[arm] if r["split"] == split]
            s = summarize(rows, args.boot, args.seed)
            summary[f"{arm}/{split}"] = s
            ratio = s["rad"][0] / s["d_near"][0] if s["d_near"][0] else float("nan")
            print(f"| {arm} | {split} | {s['n']} | {s['rad'][2]} "
                  f"| {_fmt(*s['err'][:2], 4).replace('+', '')} "
                  f"| {_fmt(*s['rad'][:2])} | {_fmt(*s['tan'][:2], 4).replace('+', '')} "
                  f"| {s['abs_rad'][0]:.4f} | {s['d_near'][0]:.4f} "
                  f"| {ratio:+.2f} | {-ratio:+.2f} "
                  f"| {_fmt(s['slope'], s['slope_ci'], 2)} "
                  f"| {s['frac_neg_rad']:.2f} |")

    if all(a in arms for a in PAIR):
        print("\n## Paired radial difference (per episode, object-paired)\n")
        print("| split | mean rad(s_f1) - mean rad(s_f1_nope) (m) | n_obj | n_ep |")
        print("|---|---|---|---|")
        for split in args.splits:
            d, ci, n_ep, n_obj = paired_rad_diff(rows_by_arm, PAIR[0], PAIR[1],
                                                 split, args.boot, args.seed)
            print(f"| {split} | {_fmt(d, ci)} | {n_obj} | {n_ep} |")
            summary[f"diff/{split}"] = {"mean": d, "ci": list(ci),
                                        "n_obj": n_obj, "n_ep": n_ep}

        print("\n## Box-only robustness (exact surface proxy)\n")
        print("| arm | split | n_obj | mean rad (m) | mean d_near (m) | rad/d_near |")
        print("|---|---|---|---|---|---|")
        for arm in arms:
            for split in args.splits:
                rows = [r for r in rows_by_arm[arm]
                        if r["split"] == split and r["shape"] == "box"]
                if not rows:
                    continue
                rad = float(np.mean([r["rad"] for r in rows]))
                dn = float(np.mean([r["d_near"] for r in rows]))
                print(f"| {arm} | {split} | {len(rows)} | {rad:+.4f} | {dn:.4f} "
                      f"| {rad / dn if dn else float('nan'):+.2f} |")
                summary[f"box/{arm}/{split}"] = {"mean_rad": rad, "mean_d_near": dn,
                                                 "n_obj": len(rows)}

    print("\n## Mechanism diagnostics\n")
    print("`radial_share` = mean(rad^2)/mean(|e|^2); `within slope` is the "
          "episode-fixed-effects OLS slope of rad on d_near (surface = -1, centre = 0). "
          "`s_ep` is the per-episode range ratio mean|pred|/mean|GT|.")
    print("| arm | split | radial share | within corr(rad, d_near) | within slope [95% CI] "
          "| s_ep mean | s_ep sd | sd(ep-mean pred range) | sd(ep-mean GT range) "
          "| within corr(pred,GT range) | scale-model resid mean (m) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for split in args.splits:
            rows = [r for r in rows_by_arm[arm] if r["split"] == split]
            m = mechanism_stats(rows, args.boot, args.seed)
            summary[f"mechanism/{arm}/{split}"] = m
            print(f"| {arm} | {split} | {m['radial_share']:.3f} "
                  f"| {m['within_corr_rad_dnear']:+.2f} "
                  f"| {_fmt(m['within_slope'], m['within_slope_ci'], 2)} "
                  f"| {m['ep_scale_mean']:.3f} | {m['ep_scale_sd']:.3f} "
                  f"| {m['ep_mean_pred_dist_sd']:.3f} | {m['ep_mean_gt_dist_sd']:.3f} "
                  f"| {m['within_corr_pred_gt_dist']:.2f} "
                  f"| {m['scale_resid_rms']:.4f} |")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=1))
        print(f"\n[json] wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()