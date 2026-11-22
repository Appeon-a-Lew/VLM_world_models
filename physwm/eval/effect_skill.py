"""Score saved matched-pair predictions relative to a no-change baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physwm.eval.counterfactual import VISUAL, load_families, static_predictions
from physwm.eval.metrics import load_predictions
from physwm.eval.paired_arms import ACTION_VARIANTS, per_family_variant
from physwm.schema import load_split


def analyze(families, predictions, *, tolerance_m=0.02, n_boot=2000, seed=0):
    variants = ACTION_VARIANTS + VISUAL
    names = sorted(families)

    def as_matrix(scores, columns):
        return np.array([[scores[fid, variant] for variant in columns] for fid in names])

    static, _ = per_family_variant(families, static_predictions(families))
    truth = as_matrix(static, ACTION_VARIANTS)
    cutoffs = np.quantile(truth, [1 / 3, 2 / 3])
    bins = np.searchsorted(cutoffs, truth, side="right")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(names), size=(n_boot, len(names)))

    errors, missing, invariance = {}, {}, {}
    for name, preds in predictions.items():
        scores, misses = per_family_variant(families, preds, variants=variants)
        errors[name] = as_matrix(scores, ACTION_VARIANTS)
        missing[name] = as_matrix(misses, ACTION_VARIANTS)
        invariance[name] = {}
        for variant in VISUAL:
            values = as_matrix(scores, (variant,))
            observed = as_matrix(misses, (variant,)) == 0
            invariance[name][variant] = {
                "families": len(names),
                "within_tolerance_rate": float(np.mean(observed & (values <= tolerance_m))),
                "mean_error_m": float(values.mean()),
                "missing_pair_rate": float(1 - observed.mean()),
            }

    groups = {}
    for label, mask in zip(("all", "weak", "medium", "strong"),
                           (np.ones_like(bins, dtype=bool), bins == 0, bins == 1, bins == 2)):
        baseline_by_family = np.where(mask, truth, 0).sum(axis=1)
        baseline = baseline_by_family.sum()
        boot_baseline = baseline_by_family[draws].sum(axis=1)
        valid_draws = boot_baseline > 0
        count = int(mask.sum())
        rows = {}
        for name, values in errors.items():
            error_by_family = np.where(mask, values, 0).sum(axis=1)
            boot_errors = error_by_family[draws[valid_draws]].sum(axis=1)
            boot_scores = 1 - boot_errors / boot_baseline[valid_draws]
            rows[name] = {
                "skill": float(1 - error_by_family.sum() / baseline) if baseline > 0 else None,
                "skill_ci95": np.quantile(boot_scores, [.025, .975]).tolist()
                if len(boot_scores) else None,
                "mean_error_m": float(values[mask].mean()) if count else None,
                "missing_pair_rate": float(missing[name][mask].mean()) if count else None,
            }
        groups[label] = {
            "pairs": count,
            "families": int(mask.any(axis=1).sum()),
            "mean_static_error_m": float(baseline / count) if count else None,
            "methods": rows,
        }
    return {
        "effect_cutoffs_m": cutoffs.tolist(),
        "effect_binning": "Thirds of pooled simulator outcome changes, shared across methods",
        "bootstrap_unit": "scene_family",
        "bootstrap_samples": n_boot,
        "bootstrap_seed": seed,
        "groups": groups,
        "invariance_tolerance_m": tolerance_m,
        "invariance": invariance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = {}
    for item in args.pred:
        name, path = item.split("=", 1)
        if name in methods:
            parser.error(f"duplicate method: {name}")
        methods[name] = load_predictions(path)
    report = analyze(load_families(load_split(args.data, args.split)), methods)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
