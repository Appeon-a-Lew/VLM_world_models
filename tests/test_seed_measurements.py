import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

script_dir = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(script_dir))
_spec = importlib.util.spec_from_file_location("run_seed_measurements", script_dir / "run_seed_measurements.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
supervisor = runner.supervisor


def _source(root):
    (root / "source").mkdir(parents=True, exist_ok=True)
    config = {"model": "test/model", "data": "/old/data", "train_split": "train",
              "tasks": ["A", "B"], "out": "/old/model",
              "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "q_proj"},
              "epochs": 2, "batch_size": 4, "grad_accum": 4, "seed": 0}
    (root / "source/sft_config.yaml").write_text(yaml.safe_dump(config))
    (root / "source/code").mkdir(exist_ok=True)
    return config


def _predictions(path, mode="normal"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"episode_id": "e1", "task": task, "image_mode": mode} for task in ("A", "B")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_seed_config_changes_only_seed_and_redirected_paths(tmp_path):
    base = _source(tmp_path)

    path = runner.seed_config(tmp_path, tmp_path / "source/data", 1)
    result = yaml.safe_load(path.read_text())

    expected = {**base, "seed": 1, "data": str(tmp_path / "source/data"),
                "out": str(tmp_path / "seeds/seed1/model")}
    assert result == expected


def test_complete_seed_skips_training_and_inference(monkeypatch, tmp_path):
    _source(tmp_path)
    seed_root = tmp_path / "seeds/seed1"
    (seed_root / "model/adapter").mkdir(parents=True)
    (seed_root / "model/adapter/adapter_config.json").write_text("{}")
    weights = b"model"
    (seed_root / "model/adapter/adapter_model.safetensors").write_bytes(weights)
    (seed_root / "model/config.json").write_text(json.dumps({"seed": 1}))
    (seed_root / "train_complete.json").write_text(json.dumps({
        "seed": 1, "weights": {"adapter_model.safetensors": hashlib.sha256(weights).hexdigest()}}))
    _predictions(seed_root / "pred/normal.jsonl")
    expected = {("e1", "A"), ("e1", "B")}
    monkeypatch.setattr(supervisor, "expected_rows", lambda *_: expected)
    monkeypatch.setattr(supervisor, "valid_predictions",
                        lambda *_: (True, "validated"))
    monkeypatch.setattr(supervisor, "run_stage",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should skip")))
    supervisor.ROOT = tmp_path
    supervisor.STOP = False
    args = SimpleNamespace(python="python", stop_after_train=False,
                           train_timeout=100, infer_timeout=100, gpu_wait=100)

    elapsed = runner.run_seed(tmp_path, tmp_path / "source/code", tmp_path / "source/data", 1, args)

    assert elapsed == 0


def test_incomplete_model_is_never_overwritten(monkeypatch, tmp_path):
    _source(tmp_path)
    model_root = tmp_path / "seeds/seed2/model"
    model_root.mkdir(parents=True)
    (model_root / "partial.txt").write_text("preserve")
    supervisor.ROOT = tmp_path
    args = SimpleNamespace(python="python", stop_after_train=True,
                           train_timeout=100, infer_timeout=100, gpu_wait=100)

    try:
        runner.run_seed(tmp_path, tmp_path / "source/code", tmp_path / "source/data", 2, args)
    except RuntimeError as exc:
        assert "refusing overwrite" in str(exc)
    else:
        raise AssertionError("incomplete model was accepted")
    assert (model_root / "partial.txt").read_text() == "preserve"


def test_stop_after_train_marks_adapter_ready_for_parent_pilot(monkeypatch, tmp_path):
    _source(tmp_path)
    monkeypatch.setattr(runner, "validate_source",
                        lambda _: (tmp_path / "source/code", tmp_path / "source/data", {"seed": 0}))
    calls = []
    monkeypatch.setattr(runner, "run_seed", lambda *a: calls.append(a[3]) or 80.0)
    states = []
    monkeypatch.setattr(supervisor, "status", lambda state, detail="": states.append((state, detail)))
    monkeypatch.setattr(sys, "argv", ["run_seed_measurements.py", str(tmp_path),
                                      "--seeds", "1", "--stop-after-train"])

    assert runner.main() == 0

    assert calls == [1]
    assert states[-1][0] == "done"
    assert "parent pilot" in states[-1][1]


def test_gpu_stage_uses_remaining_cumulative_budget(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(run_started=runner.time.monotonic(), max_wall_seconds=300,
                           max_gpu_seconds=100, used_gpu_seconds=70, gpu_wait=200)
    calls = []
    monkeypatch.setattr(supervisor, "run_stage",
                        lambda *a, **kw: calls.append((a, kw)) or 25.0)

    elapsed = runner.bounded_gpu_stage(root, "train_seed1", ["python", "train"], 90, args)

    assert elapsed == 25
    assert calls[0][0][2] == 30
    assert calls[0][1]["gpu_wait"] == 200
    assert args.used_gpu_seconds == 95
    assert json.loads((root / "gpu_usage.json").read_text())["seconds"] == 25
