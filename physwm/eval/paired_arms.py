"""Paired arm contrast on matched counterfactual families with the pre-registered M0 gate.

For each paired seed of arm A and arm B, the target-scope 3D effect error is computed per
matched scene family and action variant (reverse/half/double) exactly as
``physwm/eval/counterfactual.py`` computes it, averaged over variants within a family,
and the paired reduction R = err(A) - err(B) is bootstrapped over families. The static
baseline fixes E_static and the gate applies the threshold pinned in research/PREREG.md:

    R_seed >= max(0.10 * E_static, 0.02 m)   with the bootstrap CI95 of R excluding 0
    and the arm-B missing-pair rate not exceeding arm A's by more than 2 points.

Per-seed "earned" is the cheaper seed-2 spending rule (CI excluding 0 at half the
threshold, also guarded by the missing rate); the gate itself requires the two
preregistered, distinct seed pairs -- verified against each run's recorded training seed,
with identical files rejected -- and every seed must clear individually. Same direction
alone does not clear it. Each seed is reported separately; pooled numbers are never a
gate input. ``--calibrate`` prints E_static and the threshold from the static baseline
alone (no arm prediction files needed).

    python -m physwm.eval.paired_arms --data $WM/data --splits val_cf --calibrate \
        --out $R/calib/threshold.json
    python -m physwm.eval.paired_arms --data $WM/data --splits val_cf \
        --a fwd=$R/s1_fwd/pred_val.jsonl --a fwd=$R/s2_fwd/pred_val.jsonl \
        --b both=$R/s1_both/pred_val.jsonl --b both=$R/s2_both/pred_val.jsonl \
        --out $R/paired/m0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from physwm.eval.counterfactual import _world_prediction, load_families, static_predictions
from physwm.eval.metrics import MISSING_PENALTY, load_predictions
from physwm.schema import load_split

ACTION_VARIANTS = ("reverse", "half", "double")
THRESH_REL = 0.10    # of E_static -- research/PREREG.md (frozen 2026-11-21)
THRESH_ABS = 0.02    # m; the tiny-gains floor
MISSING_GUARD = 0.02
N_BOOT = 2000
REQUIRED_SEED_PAIRS = 2   # one seed pair can earn seed 2; only two distinct seeds clear


def per_family_variant(families: dict, preds: list[dict], *, variants=ACTION_VARIANTS):
    """{(family, variant): mean target-scope effect error in m} and missing-pair rates.

    The per-object math mirrors ``counterfactual.analyze``: a missing endpoint on either
    side of a pair costs MISSING_PENALTY, and world-frame differences are taken before
    the norm.
    """
    b_preds = [p for p in preds if p["task"] == "B"]
    lookup = {p["episode_id"]: p for p in b_preds}
    if len(lookup) != len(b_preds):
        raise ValueError("duplicate Task-B records in prediction file")
    ids = {ep.episode_id for f in families.values() for ep, _ in f.values()}
    missing = ids - set(lookup)
    if missing:
        raise ValueError(f"prediction file lacks {len(missing)} family episodes, "
                         f"e.g. {sorted(missing)[:3]}")
    err, miss = {}, {}
    for fid, family in sorted(families.items()):
        base, _ = family["original"]
        base_gt = {o.id: np.asarray(s.pos) for o, s in zip(base.objects, base.state_at(1))}
        p0 = {o.id: _world_prediction(lookup[base.episode_id], base, o.id) for o in base.objects}
        for variant in variants:
            ep, _ = family[variant]
            vals, n_miss = [], 0
            for oid in (ep.action.target,):
                n_obj = 1
                p = _world_prediction(lookup[ep.episode_id], ep, oid)
                delta = np.asarray(ep.state_at(1)[oid].pos) - base_gt[oid]
                if p is None or p0[oid] is None:
                    vals.append(MISSING_PENALTY)
                    n_miss += 1
                else:
                    vals.append(float(np.linalg.norm((p - p0[oid]) - delta)))
            err[(fid, variant)] = float(np.mean(vals))
            miss[(fid, variant)] = n_miss / n_obj
    return err, miss


def family_reductions(err_a: dict, err_b: dict) -> np.ndarray:
    """Per-family reduction R_f = mean over action variants of err(A) - err(B)."""
    fids = sorted({fid for fid, _ in err_a})
    return np.array([np.mean([err_a[(f, v)] - err_b[(f, v)] for v in ACTION_VARIANTS])
                     for f in fids], dtype=float)


def paired_stats(families: dict, err_a: dict, err_b: dict, n_boot: int, seed: int) -> dict:
    red = family_reductions(err_a, err_b)
    rng = np.random.default_rng(seed)
    n = len(red)
    boots = np.array([red[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) if n else np.array([])
    ci = (np.quantile(boots, [.025, .975]).tolist() if n else [float("nan")] * 2)
    per_variant = {v: float(np.mean([err_a[(f, v)] - err_b[(f, v)]
                                     for f in sorted({fid for fid, _ in err_a})]))
                   for v in ACTION_VARIANTS}
    return {"n_families": n, "reduction_mean": float(red.mean()) if n else float("nan"),
            "reduction_ci95": [float(v) for v in ci], "per_variant_reduction_m": per_variant}


def seed_flags(stats: dict, threshold: float, miss_rate_a: float, miss_rate_b: float) -> dict:
    """The PREREG per-seed verdicts: seed-2 'earned' at half the threshold, gate 'cleared'
    at the full threshold; both require the bootstrap CI to exclude 0, and the missing-rate
    guard applies to 'earned' as well (a gamed earn still authorizes compute).
    research/PREREG.md addendum 2026-11-21."""
    ci_excludes_zero = bool(stats["reduction_ci95"][0] > 0)
    guard_ok = bool(miss_rate_b <= miss_rate_a + MISSING_GUARD)
    return {
        "missing_guard_ok": guard_ok,
        "earned": bool(guard_ok and ci_excludes_zero
                       and stats["reduction_mean"] >= 0.5 * threshold),
        "seed_cleared": bool(guard_ok and ci_excludes_zero
                             and stats["reduction_mean"] >= threshold),
    }


def run_seed(path: str) -> int | None:
    """Training seed recorded in the run's config.json next to the prediction file."""
    cfg = Path(path).parent / "config.json"
    if not cfg.exists():
        return None
    try:
        return int(json.loads(cfg.read_text()).get("seed"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def gate_verdict(seed_rows: list[dict], verification_ok: bool) -> dict:
    """The both-seeds rule: two distinct, verified seed pairs with every seed individually
    cleared. A single seed pair can at most earn the second; it can never clear the gate."""
    enough = len(seed_rows) >= REQUIRED_SEED_PAIRS
    all_cleared = bool(seed_rows) and all(s["seed_cleared"] for s in seed_rows)
    if verification_ok and enough and all_cleared:
        reason = "cleared"
    elif not verification_ok:
        reason = "seed verification failed"
    elif not enough:
        reason = (f"insufficient seed pairs ({len(seed_rows)} of {REQUIRED_SEED_PAIRS}) -- "
                  "a single seed can earn seed 2 but can never clear the gate")
    else:
        reason = "one or more seeds did not clear the threshold"
    return {"gate_cleared": bool(verification_ok and enough and all_cleared), "reason": reason}


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--a", action="append", default=None, metavar="NAME=PATH",
                    help="arm-A prediction file; repeat once per seed (paired with --b order)")
    ap.add_argument("--b", action="append", default=None, metavar="NAME=PATH",
                    help="arm-B prediction file; repeat once per seed (paired with --a order)")
    ap.add_argument("--calibrate", action="store_true",
                    help="E_static + threshold from the static baseline alone; no arm files "
                         "needed (the preregistered alternative is counterfactual.py --static)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.calibrate and (not args.a or not args.b):
        ap.error("arm contrasts need --a and --b (or use --calibrate)")

    a_names = {item.split("=", 1)[0] for item in (args.a or [])}
    b_names = {item.split("=", 1)[0] for item in (args.b or [])}
    if not args.calibrate:
        if len(a_names) != 1 or len(b_names) != 1:
            raise SystemExit("all --a entries must share one arm name, likewise all --b entries")
        if len(args.a) != len(args.b):
            raise SystemExit(f"paired seeds: {len(args.a)} --a files vs {len(args.b)} --b files")
    name_a = (args.a or ["static"])[0].split("=", 1)[0]
    name_b = (args.b or ["static"])[0].split("=", 1)[0]

    items = [item for split in args.splits for item in load_split(args.data, split)]
    families = load_families(items)

    err_static, _ = per_family_variant(families, static_predictions(families))
    fids = sorted({fid for fid, _ in err_static})
    e_static = float(np.mean([err_static[(f, v)] for f in fids for v in ACTION_VARIANTS]))
    threshold = max(THRESH_REL * e_static, THRESH_ABS)

    if args.calibrate:
        per_variant = {v: float(np.mean([err_static[(f, v)] for f in fids]))
                       for v in ACTION_VARIANTS}
        result = {
            "units": "metres", "n_families": len(fids),
            "e_static_m": e_static, "per_variant_static_m": per_variant,
            "threshold_m": threshold,
            "threshold_rule": f"max({THRESH_REL} * E_static, {THRESH_ABS} m) -- research/PREREG.md",
            "provenance": {"data_root": str(Path(args.data).resolve()), "splits": args.splits,
                           "source_sha256": sha256(Path(__file__).resolve())},
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"E_static = {e_static:.4f} m over {len(fids)} families "
              f"(per variant: {per_variant})")
        print(f"threshold = max({THRESH_REL:.2f} x E_static, {THRESH_ABS} m) = "
              f"{threshold:.4f} m   (research/PREREG.md; wrote {out})")
        if e_static < 0.08:
            print("WARNING: E_static < 0.08 m -- the matched families are too weak; regenerate "
                  "with stronger effects before training (PREREG pre-launch calibration)")
        return

    seeds = []
    for a_item, b_item in zip(args.a, args.b):
        _, a_path = a_item.split("=", 1)
        _, b_path = b_item.split("=", 1)
        err_a, miss_a = per_family_variant(families, load_predictions(a_path))
        err_b, miss_b = per_family_variant(families, load_predictions(b_path))
        stats = paired_stats(families, err_a, err_b, args.n_boot, args.seed)
        miss_rate_a = float(np.mean(list(miss_a.values())))
        miss_rate_b = float(np.mean(list(miss_b.values())))
        stats.update({
            "a": str(Path(a_path).resolve()), "b": str(Path(b_path).resolve()),
            "seed_a": run_seed(a_path), "seed_b": run_seed(b_path),
            "missing_pair_rate_a": miss_rate_a, "missing_pair_rate_b": miss_rate_b,
            **seed_flags(stats, threshold, miss_rate_a, miss_rate_b),
        })
        seeds.append(stats)

    # Seed verification: positional pairing is checked against each run's recorded training
    # seed and identical files are rejected -- a repeated file must not count as a second seed.
    hashes = [sha256(item.split("=", 1)[1]) for item in args.a + args.b]
    seeds_a = [s["seed_a"] for s in seeds]
    seeds_b = [s["seed_b"] for s in seeds]
    problems = []
    if len(set(hashes)) != len(hashes):
        problems.append("duplicate prediction file (identical sha256 counted as a distinct seed)")
    if any(s is None for s in seeds_a + seeds_b):
        problems.append("run config.json with the training seed missing next to a prediction file")
    else:
        if len(set(seeds_a)) != len(seeds_a) or len(set(seeds_b)) != len(seeds_b):
            problems.append("seed values are not distinct within an arm")
        if seeds_a != seeds_b:
            problems.append("positional pairing does not match the recorded seeds (a_k vs b_k)")
    verification_ok = not problems
    verdict = gate_verdict(seeds, verification_ok)

    result = {
        "units": "metres", "bootstrap_unit": "scene_family", "bootstrap_samples": args.n_boot,
        "arm_a": name_a, "arm_b": name_b, "n_seed_pairs": len(seeds),
        "e_static_m": e_static, "threshold_m": threshold,
        "threshold_rule": f"max({THRESH_REL} * E_static, {THRESH_ABS} m) -- research/PREREG.md",
        "missing_guard": f"arm-B missing-pair rate <= arm-A + {MISSING_GUARD} "
                         "(applies to 'earned' as well as 'cleared')",
        "seeds": seeds,
        "seed_verification": {"ok": verification_ok, "problems": problems,
                              "seeds_a": seeds_a, "seeds_b": seeds_b},
        "gate_cleared": verdict["gate_cleared"], "gate_reason": verdict["reason"],
        "gate_note": f"clears only with {REQUIRED_SEED_PAIRS} distinct verified seed pairs, "
                     "each individually meeting the threshold with CI95 excluding 0; "
                     "same-direction sub-threshold gains do not clear",
        "provenance": {
            "data_root": str(Path(args.data).resolve()), "splits": args.splits,
            "predictions": {item.split("=", 1)[1]: h for item, h in zip(args.a + args.b, hashes)},
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(f"# Paired arm contrast: {name_a} -> {name_b}")
    print(f"E_static = {e_static:.4f} m, threshold = {threshold:.4f} m "
          f"(max({THRESH_REL:.2f} x E_static, {THRESH_ABS} m), research/PREREG.md)")
    print("| seed pair | seeds (a/b) | R (m) | CI95 | reverse | half | double "
          "| missing A | missing B | earned | cleared |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---|---|")
    for k, s in enumerate(seeds, 1):
        pv = s["per_variant_reduction_m"]
        print(f"| {k} | {s['seed_a']}/{s['seed_b']} | {s['reduction_mean']:.4f} | "
              f"[{s['reduction_ci95'][0]:.4f}, {s['reduction_ci95'][1]:.4f}] | "
              f"{pv['reverse']:.4f} | {pv['half']:.4f} | {pv['double']:.4f} | "
              f"{s['missing_pair_rate_a']:.2%} | {s['missing_pair_rate_b']:.2%} | "
              f"{s['earned']} | {s['seed_cleared']} |")
    for problem in problems:
        print(f"seed verification: {problem}")
    print(f"GATE: {verdict['reason']} ({len(seeds)} seed pair(s); wrote {out})")


if __name__ == "__main__":
    main()
