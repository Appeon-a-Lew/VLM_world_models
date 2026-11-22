"""Compare Task B with static and constant-velocity baselines across OOD push speeds."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physwm.eval.analyze_baseline import xyz
from physwm.eval.metrics import MISSING_PENALTY, gt_centers_cam, load_predictions, validate_coverage
from physwm.schema import load_split


def analyze(episodes, methods):
    # Use pooled target-mass quartiles and equal mass-stratum weights for both speed groups.
    masses = {eid: ep.objects[ep.action.target].mass for eid, ep in episodes.items()}
    cuts = np.unique(np.quantile(list(masses.values()), [.25, .5, .75]))
    strata = defaultdict(list)
    for eid, ep in episodes.items():
        speed = ep.action.speed
        bucket = "weak" if speed < .3 else "strong" if speed > 1.6 else "in_range"
        strata[bucket, int(np.searchsorted(cuts, masses[eid], side="right"))].append(eid)
    common = sorted({mass for bucket, mass in strata if bucket == "weak"}
                    & {mass for bucket, mass in strata if bucket == "strong"})
    truth = {eid: (gt_centers_cam(ep, 0)[ep.action.target], gt_centers_cam(ep, 1)[ep.action.target])
             for eid, ep in episodes.items()}
    static = {eid: float(np.linalg.norm(initial - future))
              for eid, (initial, future) in truth.items()}

    rows = []
    for name, preds in methods.items():
        task_b = [p for p in preds if p["task"] == "B"]
        validate_coverage(task_b, episodes, ("B",))
        by_id = {p["episode_id"]: p for p in task_b}
        errors, missing = {}, {}
        for eid, (_, future) in truth.items():
            prediction = xyz(by_id[eid], episodes[eid].action.target)
            missing[eid] = prediction is None
            errors[eid] = (MISSING_PENALTY if prediction is None
                           else float(np.linalg.norm(prediction - future)))
        for bucket in sorted({bucket for bucket, _ in strata}):
            groups = [strata[bucket, mass] for speed, mass in sorted(strata) if speed == bucket]
            eids = [eid for group in groups for eid in group]
            raw_error = float(np.mean([errors[eid] for eid in eids]))
            raw_static = float(np.mean([static[eid] for eid in eids]))
            row = dict(method=name, speed=bucket, episodes=len(eids), endpoint_error_m=raw_error,
                       skill=1 - raw_error / raw_static if raw_static > 0 else None,
                       missing_rate=float(np.mean([missing[eid] for eid in eids])))
            if common and bucket in ("weak", "strong"):
                counts = {mass: len(strata[bucket, mass]) for mass in common}
                adjusted = float(np.mean([np.mean([errors[eid] for eid in strata[bucket, mass]])
                                          for mass in common]))
                adjusted_static = float(np.mean([np.mean([static[eid] for eid in strata[bucket, mass]])
                                                 for mass in common]))
                row.update(common_mass_counts=counts, mass_standardized_error_m=adjusted,
                           mass_standardized_skill=1 - adjusted / adjusted_static
                           if adjusted_static > 0 else None)
            rows.append(row)
    return {"mass_cutoffs_kg": cuts.tolist(), "common_mass_strata": common, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val_ood_forces")
    parser.add_argument("--pred", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes = {ep.episode_id: ep for _, ep in load_split(args.data, args.split)}
    methods = {}
    for item in args.pred:
        name, path = item.split("=", 1)
        methods[name] = load_predictions(path)
    report = analyze(episodes, methods)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    print("| Method | Speed | Episodes | Endpoint (m) | Skill | Missing | Mass-standardized skill |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in report["rows"]:
        print(f"| {row['method']} | {row['speed']} | {row['episodes']} | "
              f"{fmt(row['endpoint_error_m'])} | {fmt(row['skill'])} | "
              f"{row['missing_rate']:.1%} | {fmt(row.get('mass_standardized_skill'))} |")


if __name__ == "__main__":
    main()
