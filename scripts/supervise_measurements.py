#!/usr/bin/env python3
"""Resume corrected-baseline image measurements without retraining.

Run from the frozen code snapshot with the run root as the only positional
argument. Completed prediction files are retained after validating coverage;
incomplete files are regenerated through a temporary path and atomically
installed only after validation succeeds.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

SPLITS = ["val_id", "val_ood_objects", "val_ood_forces", "val_ood_config",
          "val_diag_elev", "val_diag_count", "val_diag_corner", "val_diag_elev_corner"]
MODES = ("blank", "swap")
STOP = False
ROOT = Path()
STAGE = "startup"


class StageTimeout(TimeoutError):
    def __init__(self, stage: str, elapsed: float):
        super().__init__(f"{stage} exceeded its runtime limit")
        self.elapsed = elapsed


class StageInterrupted(InterruptedError):
    def __init__(self, stage: str, elapsed: float):
        super().__init__(f"{stage} interrupted")
        self.elapsed = elapsed


class StageFailure(subprocess.CalledProcessError):
    def __init__(self, returncode: int, command: list[str], elapsed: float):
        super().__init__(returncode, command)
        self.elapsed = elapsed


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def status(state: str, detail: str = "") -> None:
    rec = {"state": state, "stage": STAGE, "detail": detail,
           "host": socket.gethostname(), "pid": os.getpid(), "updated_utc": utc()}
    tmp = ROOT / "status_measurements.tmp"
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.replace(ROOT / "status_measurements.json")
    with (ROOT / "events_measurements.jsonl").open("a") as stream:
        stream.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def on_signal(signum, _frame):
    global STOP
    STOP = True
    status("failed", f"received signal {signum}; stage will stop at the next safe boundary")


def gpu_ready() -> tuple[bool, str]:
    try:
        rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
        if rc:
            return False, f"libcuda cuInit returned {rc}"
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()
        if len(q) != 1:
            return False, "expected exactly one GPU telemetry row"
        mem, util = [x.strip() for x in q[0].split(",", 1)]
        if not mem.isdigit() or not util.isdigit():
            return False, f"non-numeric GPU telemetry: {q[0]}"
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        app_rows = [x.strip() for x in apps.splitlines() if x.strip()]
        if any(not row.isdigit() for row in app_rows):
            return False, f"non-numeric compute-process telemetry: {app_rows}"
        active = app_rows
        if int(mem) >= 2500 or int(util) > 5 or active:
            return False, f"GPU busy (memory={mem} MiB, util={util}%, compute_pids={active})"
        return True, "CUDA initializes; numeric telemetry and compute-process check pass"
    except Exception as exc:
        return False, f"GPU health check failed: {type(exc).__name__}: {exc}"


def expected_rows(data: Path, splits: list[str]) -> set[tuple[str, str]]:
    from physwm.schema import load_split
    expected: set[tuple[str, str]] = set()
    for split in splits:
        for _, ep in load_split(data, split):
            for task in ("A", "B"):
                expected.add((ep.episode_id, task))
    return expected


def valid_predictions(path: Path, expected: set[tuple[str, str]], mode: str) -> tuple[bool, str]:
    try:
        found = set()
        with path.open() as stream:
            for line_no, line in enumerate(stream, 1):
                obj = json.loads(line)
                key = (obj["episode_id"], obj["task"])
                if key in found:
                    return False, f"duplicate prediction key at line {line_no}: {key}"
                found.add(key)
                if obj.get("image_mode") != mode:
                    return False, f"unexpected image_mode at line {line_no}"
        if found != expected:
            return False, f"coverage mismatch: got {len(found)}, expected {len(expected)}"
        return True, f"validated {len(found)} unique predictions"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def stop_process_group(proc: subprocess.Popen, grace: float = 30) -> None:
    """Stop the whole new session, even if its leader exits before descendants."""
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def run_stage(name: str, command: list[str], timeout: float, gpu: bool = True,
              gpu_wait: float | None = None, kill_after: float = 30,
              wall_deadline: float | None = None) -> float:
    global STAGE
    STAGE = name
    log_path = ROOT / "logs" / f"measure_{name}.log"
    if STOP:
        raise InterruptedError("stop requested")
    if gpu:
        start = time.monotonic()
        wait_limit = timeout if gpu_wait is None else gpu_wait
        while True:
            ok, why = gpu_ready()
            if ok:
                break
            if time.monotonic() - start > wait_limit:
                status("timed_out", f"GPU wait exceeded {wait_limit:.0f}s: {why}")
                raise TimeoutError(why)
            status("waiting", why + "; retry in 60 seconds")
            for _ in range(60):
                if STOP:
                    raise InterruptedError("stop requested while waiting for GPU")
                time.sleep(1)
    if STOP:
        raise InterruptedError("stop requested before child launch")
    if wall_deadline is not None:
        timeout = min(timeout, wall_deadline - time.monotonic())
    if timeout <= 0:
        status("timed_out", "no wall-time remains for child stage")
        raise TimeoutError("wall-time budget exhausted before child launch")
    status("running", " ".join(command))
    child_started = time.monotonic()
    with log_path.open("a") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ.copy())
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if STOP or time.monotonic() >= deadline:
                timed_out = not STOP
                stop_process_group(proc, kill_after)
                if timed_out:
                    status("timed_out", f"stage exceeded {timeout:.0f}s; process group stopped")
                    raise StageTimeout(name, time.monotonic() - child_started)
                raise StageInterrupted(name, time.monotonic() - child_started)
            time.sleep(2)
        if proc.returncode or STOP:
            stop_process_group(proc, kill_after)
        elif process_group_exists(proc.pid):
            # A successful leader can still leave workers behind. Drain them too.
            stop_process_group(proc, kill_after)
        if proc.returncode:
            raise StageFailure(proc.returncode, command, time.monotonic() - child_started)
        if STOP:
            raise InterruptedError("stop requested while child was exiting")
    return time.monotonic() - child_started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu-wait", type=float, default=4 * 3600)
    ap.add_argument("--stage-timeout", type=float, default=2 * 3600)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    args = ap.parse_args()
    global ROOT
    ROOT = args.run_root.resolve()
    data = (args.data or ROOT / "data").resolve()
    os.chdir(Path(__file__).resolve().parents[1])
    os.environ.update(PYTHONUNBUFFERED="1", HF_HOME="/var/tmp/toyi/wm/hf",
                      XDG_CACHE_HOME="/var/tmp/toyi/wm/cache", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", MUJOCO_GL="egl", OMP_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "measurements.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("measurement supervisor already active", file=sys.stderr)
        return 2
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    try:
        if not (ROOT / "model/adapter/adapter_config.json").is_file():
            raise FileNotFoundError("seed-0 adapter missing; refusing to train or infer")
        all_expected = expected_rows(data, args.splits)
        id_expected = expected_rows(data, ["val_id"])
        common = [args.python, "-m", "physwm.vlm.infer", "--data", str(data),
                  "--lora", str(ROOT / "model/adapter"), "--batch-size", "32"]
        for mode in MODES:
            final = ROOT / "pred" / f"{mode}.jsonl"
            final.parent.mkdir(parents=True, exist_ok=True)
            valid, detail = valid_predictions(final, id_expected, mode) if final.exists() else (False, "missing")
            if valid:
                status("running", f"skip completed {mode}: {detail}")
                continue
            tmp = final.with_suffix(".jsonl.partial")
            tmp.unlink(missing_ok=True)
            cmd = [*common, "--splits", "val_id", "--image-mode", mode, "--out", str(tmp)]
            run_stage(f"infer_{mode}", cmd, args.stage_timeout, gpu_wait=args.gpu_wait)
            valid, detail = valid_predictions(tmp, id_expected, mode)
            if not valid:
                raise RuntimeError(f"{mode} output failed validation: {detail}")
            if STOP:
                raise InterruptedError("stop requested before installing prediction output")
            tmp.replace(final)
        pred_args = [f"static={ROOT}/baselines/static.jsonl",
                     f"const_velocity={ROOT}/baselines/const_velocity.jsonl"]
        for mode, expected, recorded_mode in (("normal", all_expected, "normal"),
                                               ("none", id_expected, "none"),
                                               ("blank", id_expected, "blank"),
                                               ("swap", id_expected, "swap")):
            path = ROOT / "pred" / f"{mode}.jsonl"
            valid, detail = valid_predictions(path, expected, recorded_mode)
            if not valid:
                raise RuntimeError(f"existing {mode} output is not complete: {detail}")
            pred_args.append(f"sft_{mode}={path}")
        STAGE = "evaluate"
        (ROOT / "results").mkdir(parents=True, exist_ok=True)
        run_stage("evaluate", [args.python, "-m", "physwm.eval.evaluate", "--data", str(data),
                                "--splits", *args.splits, "--csv", str(ROOT / "results/metrics.csv"),
                                "--pred", *pred_args], args.stage_timeout, gpu=False)
        (ROOT / "results/table.md").write_text((ROOT / "logs/measure_evaluate.log").read_text())
        if STOP:
            raise InterruptedError("stop requested before marking measurements done")
        STAGE = "complete"
        status("done", "blank/swap measurements and scoring complete; seed-0 adapter unchanged")
        return 0
    except TimeoutError as exc:
        if json.loads((ROOT / "status_measurements.json").read_text()).get("state") != "timed_out":
            status("timed_out", str(exc))
        return 124
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
