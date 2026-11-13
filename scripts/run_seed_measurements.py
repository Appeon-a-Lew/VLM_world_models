#!/usr/bin/env python3
"""Bounded, resumable multi-seed training and validation inference.

The source snapshot and corrected dataset are copied into ``ROOT/source``;
seed-specific outputs live in ``ROOT/seeds``. The source snapshot is never
edited. This runner imports the colocated supervisor module for its fail-closed
GPU check, process-group cleanup, and prediction coverage validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import supervise_measurements as supervisor

SPLITS = ["val_id", "val_ood_objects", "val_ood_forces", "val_ood_config",
          "val_diag_elev", "val_diag_count", "val_diag_corner", "val_diag_elev_corner"]
EXPECTED = {"train": 10000, "val_id": 500, "val_ood_objects": 500,
            "val_ood_forces": 500, "val_ood_config": 500,
            "val_diag_elev": 300, "val_diag_count": 300,
            "val_diag_corner": 300, "val_diag_elev_corner": 300}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(root: Path) -> tuple[Path, Path, dict]:
    source = root / "source"
    code, data = source / "code", source / "data"
    if not (code / "physwm/vlm/sft.py").is_file():
        raise FileNotFoundError("frozen source/code/physwm/vlm/sft.py is missing")
    launch = json.loads((source / "launch.json").read_text())
    archive = source / "physwm-audit-v2-20261027.tar.gz"
    if not archive.is_file() or file_sha256(archive) != launch["archive_sha256"]:
        raise RuntimeError("source archive SHA-256 does not match launch.json")
    critical = ("physwm/schema.py", "physwm/vlm/sft.py", "physwm/vlm/infer.py",
                "physwm/vlm/prompts.py", "physwm/vlm/parse.py", "physwm/eval/metrics.py",
                "physwm/eval/evaluate.py", "configs/sft_qwen3vl2b.yaml")
    with tarfile.open(archive, "r:gz") as packed:
        for rel in critical:
            packed_file = packed.extractfile(rel)
            local_file = code / rel
            if packed_file is None or not local_file.is_file():
                raise FileNotFoundError(f"critical snapshot file missing: {rel}")
            if hashlib.sha256(packed_file.read()).digest() != bytes.fromhex(file_sha256(local_file)):
                raise RuntimeError(f"extracted source differs from archived snapshot: {rel}")
        archived_cfg = yaml.safe_load(packed.extractfile("configs/sft_qwen3vl2b.yaml"))
    run_cfg = yaml.safe_load((source / "sft_config.yaml").read_text())
    for key in archived_cfg:
        if key not in ("data", "out") and archived_cfg[key] != run_cfg.get(key):
            raise RuntimeError(f"corrected run config differs from frozen {key}")
    if run_cfg.get("seed") != 0:
        raise RuntimeError("corrected run config seed is not 0")
    sys.path.insert(0, str(code))
    from physwm.schema import load_split
    counts = {split: len(load_split(data, split)) for split in EXPECTED}
    if counts != EXPECTED:
        raise RuntimeError(f"corrected dataset count mismatch: {counts}")
    for name in ("static.jsonl", "const_velocity.jsonl"):
        if not (source / "baselines" / name).is_file():
            raise FileNotFoundError(f"source baseline missing: {name}")
    return code, data, launch


def seed_config(root: Path, data: Path, seed: int) -> Path:
    source_cfg = yaml.safe_load((root / "source/sft_config.yaml").read_text())
    configured_seed = source_cfg.get("seed")
    if configured_seed != 0:
        raise RuntimeError(f"expected unchanged seed-0 source config; found {configured_seed}")
    cfg = dict(source_cfg)
    cfg["seed"] = seed
    cfg["data"] = str(data)
    cfg["out"] = str(root / "seeds" / f"seed{seed}" / "model")
    dest = root / "seeds" / f"seed{seed}" / "sft_config.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def seed_artifacts_valid(seed_root: Path, expected_seed: int) -> bool:
    adapter = seed_root / "model/adapter/adapter_config.json"
    cfg = seed_root / "model/config.json"
    weight_files = list((seed_root / "model/adapter").glob("adapter_model*.safetensors"))
    weight_files += list((seed_root / "model/adapter").glob("adapter_model*.bin"))
    if not adapter.is_file() or not cfg.is_file() or not weight_files:
        return False
    if any(p.stat().st_size == 0 for p in weight_files):
        return False
    saved = json.loads(cfg.read_text())
    return saved.get("seed") == expected_seed


def seed_is_complete(seed_root: Path, expected_seed: int) -> bool:
    marker = seed_root / "train_complete.json"
    if not marker.is_file() or not seed_artifacts_valid(seed_root, expected_seed):
        return False
    done = json.loads(marker.read_text())
    weights = sorted((seed_root / "model/adapter").glob("adapter_model*.safetensors"))
    weights += sorted((seed_root / "model/adapter").glob("adapter_model*.bin"))
    actual = {p.name: file_sha256(p) for p in weights}
    return done.get("seed") == expected_seed and done.get("weights") == actual


def record_gpu_usage(root: Path, seconds: float) -> None:
    usage_path = root / "gpu_usage.json"
    existing = json.loads(usage_path.read_text()) if usage_path.exists() else {"seconds": 0.0}
    existing["seconds"] = float(existing.get("seconds", 0.0)) + seconds
    existing["updated_utc"] = supervisor.utc()
    tmp = root / "gpu_usage.tmp"
    tmp.write_text(json.dumps(existing, indent=2) + "\n")
    tmp.replace(usage_path)


def bounded_gpu_stage(root: Path, name: str, command: list[str], timeout: float, args) -> float:
    elapsed = time.monotonic() - args.run_started
    wall_remaining = args.max_wall_seconds - elapsed
    gpu_remaining = args.max_gpu_seconds - args.used_gpu_seconds
    if wall_remaining <= 0 or gpu_remaining <= 0:
        raise TimeoutError("remaining wall-time or GPU-process budget is exhausted")
    supervisor.STAGE = name
    try:
        seconds = supervisor.run_stage(name, command, min(timeout, gpu_remaining), gpu=True,
                                       gpu_wait=min(args.gpu_wait, wall_remaining),
                                       wall_deadline=args.run_started + args.max_wall_seconds)
    except (supervisor.StageTimeout, supervisor.StageInterrupted, supervisor.StageFailure) as exc:
        args.used_gpu_seconds += exc.elapsed
        record_gpu_usage(root, exc.elapsed)
        raise
    args.used_gpu_seconds += seconds
    record_gpu_usage(root, seconds)
    return seconds


def run_seed(root: Path, code: Path, data: Path, seed: int, args) -> float:
    seed_root = root / "seeds" / f"seed{seed}"
    model_root = seed_root / "model"
    adapter = model_root / "adapter"
    seed_root.mkdir(parents=True, exist_ok=True)
    cfg_path = seed_config(root, data, seed)
    elapsed_gpu = 0.0

    if seed_is_complete(seed_root, seed):
        supervisor.status("running", f"seed {seed}: validated existing adapter; skip training")
    else:
        if model_root.exists():
            raise RuntimeError(f"incomplete seed {seed} model directory exists; refusing overwrite")
        command = [args.python, "-m", "physwm.vlm.sft", "--config", str(cfg_path)]
        seconds = bounded_gpu_stage(root, f"train_seed{seed}", command, args.train_timeout, args)
        elapsed_gpu += seconds
        if not seed_artifacts_valid(seed_root, seed):
            raise RuntimeError(f"seed {seed} training returned without a valid adapter/config")
        weights = sorted((model_root / "adapter").glob("adapter_model*.safetensors"))
        weights += sorted((model_root / "adapter").glob("adapter_model*.bin"))
        marker = {"seed": seed, "completed_utc": supervisor.utc(),
                  "weights": {p.name: file_sha256(p) for p in weights}}
        (seed_root / "train_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
        if not seed_is_complete(seed_root, seed):
            raise RuntimeError(f"seed {seed} completion marker or adapter validation failed")
    if args.stop_after_train:
        return elapsed_gpu

    expected = supervisor.expected_rows(data, SPLITS)
    pred_dir = seed_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    final = pred_dir / "normal.jsonl"
    ok, detail = supervisor.valid_predictions(final, expected, "normal") if final.exists() else (False, "missing")
    if ok:
        supervisor.status("running", f"seed {seed}: skip complete normal validation predictions: {detail}")
    else:
        partial = pred_dir / "normal.jsonl.partial"
        partial.unlink(missing_ok=True)
        command = [args.python, "-m", "physwm.vlm.infer", "--data", str(data),
                   "--splits", *SPLITS, "--lora", str(adapter), "--batch-size", "32",
                   "--image-mode", "normal", "--out", str(partial)]
        seconds = bounded_gpu_stage(root, f"infer_normal_seed{seed}", command, args.infer_timeout, args)
        elapsed_gpu += seconds
        ok, detail = supervisor.valid_predictions(partial, expected, "normal")
        if not ok:
            raise RuntimeError(f"seed {seed} predictions failed coverage: {detail}")
        if supervisor.STOP:
            raise InterruptedError("stop requested before installing normal predictions")
        if final.exists():
            final.rename(pred_dir / f"normal.jsonl.invalid.{int(time.time())}")
        partial.replace(final)
    return elapsed_gpu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="isolated run root with source/ and seeds/")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    ap.add_argument("--stop-after-train", action="store_true",
                    help="train selected seed(s), then return for parent pilot evaluation")
    ap.add_argument("--train-timeout", type=float, default=2 * 3600)
    ap.add_argument("--infer-timeout", type=float, default=60 * 60)
    ap.add_argument("--gpu-wait", type=float, default=2 * 3600)
    ap.add_argument("--max-gpu-seconds", type=float, default=6 * 3600)
    ap.add_argument("--max-wall-seconds", type=float, default=6 * 3600)
    args = ap.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    supervisor.ROOT = root
    supervisor.STAGE = "seed_measurements"
    supervisor.STOP = False
    lock_path = root / "seed_measurements.lock"
    import fcntl
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("seed measurement runner already active", file=sys.stderr)
        return 2
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, supervisor.on_signal)

    try:
        code, data, launch = validate_source(root)
        base_cfg = yaml.safe_load((root / "source/sft_config.yaml").read_text())
        if launch.get("seed") != 0 or base_cfg.get("seed") != 0:
            raise RuntimeError("source snapshot is not the expected seed-0 corrected baseline")
        os.chdir(code)
        os.environ.update(PYTHONPATH=str(code), HF_HOME="/var/tmp/toyi/wm/hf",
                          XDG_CACHE_HOME="/var/tmp/toyi/wm/cache", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                          TOKENIZERS_PARALLELISM="false", MUJOCO_GL="egl",
                          OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                          PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        start = time.monotonic()
        args.run_started = start
        prior_usage = json.loads((root / "gpu_usage.json").read_text()) if (root / "gpu_usage.json").exists() else {"seconds": 0.0}
        args.used_gpu_seconds = float(prior_usage["seconds"])
        if args.used_gpu_seconds >= args.max_gpu_seconds:
            supervisor.status("timed_out", "cumulative GPU process budget already exhausted")
            return 124
        for seed in args.seeds:
            if supervisor.STOP:
                raise InterruptedError("stop requested before next seed")
            if time.monotonic() - start > args.max_wall_seconds:
                supervisor.status("timed_out", "six-hour wall clock budget exhausted")
                return 124
            run_seed(root, code, data, seed, args)
            supervisor.status("running", f"seed {seed} complete; GPU budget used {args.used_gpu_seconds:.0f}s")
            if args.used_gpu_seconds > args.max_gpu_seconds:
                supervisor.status("timed_out", "six-hour cumulative GPU process budget exceeded")
                return 124
            if args.stop_after_train:
                supervisor.status("done", f"seed {seed} adapter ready; stopped for parent pilot")
                return 0
        if supervisor.STOP:
            raise InterruptedError("stop requested before completion")
        supervisor.status("done", f"completed requested seeds {args.seeds}; GPU process budget {args.used_gpu_seconds:.0f}s")
        return 0
    except TimeoutError as exc:
        supervisor.status("timed_out", str(exc))
        return 124
    except Exception as exc:
        supervisor.status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
