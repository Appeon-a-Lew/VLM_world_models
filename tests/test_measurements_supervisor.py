import json
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

_spec = importlib.util.spec_from_file_location(
    "supervise_measurements", Path(__file__).resolve().parents[1] / "scripts/supervise_measurements.py")
supervisor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(supervisor)


def _rows(path: Path, mode: str, keys=(('ep-1', 'A'), ('ep-1', 'B'))):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({"episode_id": eid, "task": task,
                                       "image_mode": mode}) + '\n'
                           for eid, task in keys))


def _reset_root(tmp_path, monkeypatch):
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    monkeypatch.setattr(supervisor, "ROOT", root)
    return root


def test_gpu_waits_for_busy_then_runs(monkeypatch, tmp_path):
    _reset_root(tmp_path, monkeypatch)
    checks = iter([(False, "busy at 100%"), (True, "healthy")])
    monkeypatch.setattr(supervisor, "gpu_ready", lambda: next(checks))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    states = []
    monkeypatch.setattr(supervisor, "status", lambda state, detail="": states.append(state))

    supervisor.run_stage("probe", [sys.executable, "-c", "pass"], timeout=2)

    assert states == ["waiting", "running"]


def test_gpu_n_a_telemetry_is_not_idle(monkeypatch):
    class Driver:
        @staticmethod
        def cuInit(_):
            return 0

    monkeypatch.setattr(supervisor.ctypes, "CDLL", lambda _: Driver())
    monkeypatch.setattr(supervisor.subprocess, "check_output", lambda *a, **k: "314, [N/A]\n")

    ready, detail = supervisor.gpu_ready()

    assert not ready
    assert "non-numeric" in detail


def test_complete_prediction_files_are_skipped(monkeypatch, tmp_path):
    root = tmp_path / "run"
    (root / "model/adapter").mkdir(parents=True)
    (root / "model/adapter/adapter_config.json").write_text("{}")
    keys = {("ep-1", "A"), ("ep-1", "B")}
    for mode in ("normal", "none", "blank", "swap"):
        _rows(root / "pred" / f"{mode}.jsonl", mode)
    monkeypatch.setattr(supervisor, "expected_rows", lambda *_: keys)
    calls = []

    def run_stage(name, command, timeout, gpu=True, **kwargs):
        calls.append((name, gpu))
        if name == "evaluate":
            (supervisor.ROOT / "logs/measure_evaluate.log").write_text("table")

    monkeypatch.setattr(supervisor, "run_stage", run_stage)
    monkeypatch.setattr(sys, "argv", ["supervise_measurements.py", str(root)])

    assert supervisor.main() == 0

    assert calls == [("evaluate", False)]
    assert json.loads((root / "status_measurements.json").read_text())["state"] == "done"


def test_incomplete_output_uses_partial_then_installs_atomically(monkeypatch, tmp_path):
    root = tmp_path / "run"
    (root / "model/adapter").mkdir(parents=True)
    (root / "model/adapter/adapter_config.json").write_text("{}")
    keys = {("ep-1", "A"), ("ep-1", "B")}
    for mode in ("normal", "none", "swap"):
        _rows(root / "pred" / f"{mode}.jsonl", mode)
    monkeypatch.setattr(supervisor, "expected_rows", lambda *_: keys)
    calls = []

    def run_stage(name, command, timeout, gpu=True, **kwargs):
        calls.append((name, gpu))
        if name == "infer_blank":
            output = Path(command[command.index("--out") + 1])
            _rows(output, "blank")
        if name == "evaluate":
            (supervisor.ROOT / "logs/measure_evaluate.log").write_text("table")

    monkeypatch.setattr(supervisor, "run_stage", run_stage)
    monkeypatch.setattr(sys, "argv", ["supervise_measurements.py", str(root)])

    assert supervisor.main() == 0

    assert calls == [("infer_blank", True), ("evaluate", False)]
    assert (root / "pred/blank.jsonl").is_file()
    assert not (root / "pred/blank.jsonl.partial").exists()


def test_timeout_records_state_and_kills_child_process_group(monkeypatch, tmp_path):
    root = _reset_root(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "gpu_ready", lambda: (True, "healthy"))
    states = []
    monkeypatch.setattr(supervisor, "status", lambda state, detail="": states.append(state))
    pid_file = tmp_path / "grandchild.pid"
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',\"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)\"]); "
            "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)")

    with pytest.raises(TimeoutError):
        supervisor.run_stage("timeout", [sys.executable, "-c", code, str(pid_file)],
                             timeout=0.5, gpu=False, kill_after=0.2)

    assert states[-1] == "timed_out"
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        proc = Path(f"/proc/{child_pid}/stat")
        if not proc.exists() or proc.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"child process {child_pid} survived supervisor timeout")


def test_successful_leader_exit_does_not_leave_child(monkeypatch, tmp_path):
    _reset_root(tmp_path, monkeypatch)
    states = []
    monkeypatch.setattr(supervisor, "status", lambda state, detail="": states.append(state))
    pid_file = tmp_path / "orphan.pid"
    code = ("import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "open(sys.argv[1],'w').write(str(p.pid))")

    supervisor.run_stage("leader_exit", [sys.executable, "-c", code, str(pid_file)],
                         timeout=2, gpu=False, kill_after=0.2)

    child_pid = int(pid_file.read_text())
    proc = Path(f"/proc/{child_pid}/stat")
    assert not proc.exists() or proc.read_text().split()[2] == "Z"
    assert states == ["running"]


def test_sigterm_handler_records_failed_state(monkeypatch):
    states = []
    monkeypatch.setattr(supervisor, "status", lambda state, detail="": states.append((state, detail)))
    monkeypatch.setattr(supervisor, "STOP", False)

    supervisor.on_signal(signal.SIGTERM, None)

    assert supervisor.STOP
    assert states[-1][0] == "failed"
