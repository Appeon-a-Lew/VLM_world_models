#!/usr/bin/env python3
"""Offline sanity check of an experiment spec (laptop side, no GPU box needed).

    python3 /home/appeon_a_lew/adl4cv-worlmodels/scripts/check_spec.py <worktree>/research/experiments/<id>/spec.yaml

Mirrors the cheap parts of physwm/agent/exec.py:validate_spec so mistakes show up
before submit. The executor's checks on the box remain authoritative.
"""
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
REQUIRED = ("id", "hypothesis", "change", "baseline", "kind", "budget_hours", "seed")
# keep in sync with physwm/agent/exec.py ABLATION_MODES (cf_* = counterfactual action text, normal image)
IMAGE_MODES = {"normal", "none", "blank", "swap", "cf_flip", "cf_impulse_0.5", "cf_impulse_2"}
STYLES = {"center", "qwen3d", "uvz", "box3d"}


def main(path):
    spec_path = Path(path).resolve()
    spec = yaml.safe_load(spec_path.read_text())
    policy = yaml.safe_load((REPO / "configs/research_agent_policy.yaml").read_text())
    errors = [f"missing '{k}'" for k in REQUIRED if k not in spec]
    if spec.get("kind") not in ("train", "eval_only"):
        errors.append("kind must be train | eval_only")
    max_h = policy["resources"]["large_experiment_max_hours"]
    try:
        if not 0 < float(spec.get("budget_hours", 0)) <= max_h:
            errors.append(f"budget_hours must be in (0, {max_h}]")
    except (TypeError, ValueError):
        errors.append("budget_hours must be a number")
    if spec.get("id") != spec_path.parent.name:
        errors.append(f"id '{spec.get('id')}' != directory name '{spec_path.parent.name}'")
    bad_modes = set(spec.get("ablations", [])) - IMAGE_MODES
    if bad_modes:
        errors.append(f"unknown ablations {sorted(bad_modes)}; allowed {sorted(IMAGE_MODES)}")
    if spec.get("style", "center") not in STYLES:
        errors.append(f"unknown style '{spec.get('style')}'")
    frames = spec.get("frames")
    if frames is not None and not (isinstance(frames, list) and frames
                                   and all(isinstance(f, int) and not isinstance(f, bool) and f >= 0 for f in frames)):
        errors.append("frames must be a non-empty list of non-negative int indices")
    views = spec.get("views")
    if views is not None and not (isinstance(views, list) and views
                                  and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in views)):
        errors.append("views must be a non-empty list of non-negative int indices")
    if "use_pe" in spec and not isinstance(spec["use_pe"], bool):
        errors.append("use_pe must be a bool")
    tasks = spec.get("tasks")
    if tasks is not None and (not isinstance(tasks, list) or not tasks or set(tasks) - {"A", "B", "N"}):
        errors.append("tasks must be a non-empty subset of [A, B, N]")
    if spec.get("kind") == "train":
        worktree = spec_path.parents[3]
        cfg = worktree / str(spec.get("train_config", ""))
        if not cfg.is_file():
            errors.append(f"train_config not found: {cfg}")
        else:
            train_cfg = yaml.safe_load(cfg.read_text())
            split = train_cfg.get("train_split")
            if train_cfg.get("seed") != spec.get("seed"):
                errors.append("spec seed must match the training config seed")
            held_out = (set(policy["data"]["protected_splits"]) | set(policy["data"]["validation_splits"])
                        | set(policy["data"].get("diagnostic_splits", [])))
            if split in held_out or str(split).startswith(("val", "ood", "id_test")):
                errors.append(f"train_split '{split}' is a held-out split")
    print(json.dumps({"spec": str(spec_path), "ok": not errors, "errors": errors,
                      "summary": {k: spec.get(k) for k in ("id", "kind", "style", "budget_hours", "ablations")}},
                     indent=1))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) == 2 else "usage: check_spec.py <spec.yaml>")
