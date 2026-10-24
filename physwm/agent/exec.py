"""Deterministic experiment executor -- the only way the research agent touches the GPU box.

Runs on the lab machine from the *main* code copy (~/adl4cv-worlmodels). Experiment code is a
snapshot of one git commit (sent by scripts/agent.sh as a tar on stdin) and is only used for
training / inference. Evaluation, metrics, baselines and the policy always come from the main copy,
whose hashes are pinned by a human (`pin`), so an experiment cannot change what counts as success.

Agent-facing ops (via scripts/agent.sh): status, submit, gen, results, log, cancel, compare, final-eval
Human-only ops (run on the box directly): pin, reset-failures
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

MAIN = Path(__file__).resolve().parents[2]
WM = Path(os.environ.get("WM", "/var/tmp/toyi/wm"))
AGENT = WM / "agent"
EXPS = WM / "experiments"
DATA = WM / "data"
LEDGER = AGENT / "ledger.jsonl"
POLICY = yaml.safe_load((MAIN / "configs" / "research_agent_policy.yaml").read_text())
PINNED_FILES = [
    "configs/research_agent_policy.yaml", "manifests/final_test.json", "physwm/schema.py",
    "physwm/geometry.py", "physwm/eval/metrics.py", "physwm/eval/evaluate.py",
    "physwm/eval/baselines.py", "physwm/agent/exec.py", "physwm/agent/manifest.py",
]
EXP_ID = re.compile(r"^exp\d{3}_[a-z0-9_]{1,48}$")
SPLIT_NAME = re.compile(r"^[a-z0-9_]{1,40}$")
BASELINE_NAMES = ("static", "const_velocity")
DIAG_SPLITS = POLICY["data"].get("diagnostic_splits", [])


class Denied(Exception):
    pass


# ---------------------------------------------------------------- helpers
def now() -> float:
    return time.time()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]


def ledger_append(rec: dict) -> None:
    AGENT.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def exp_status(exp: str) -> dict:
    p = EXPS / exp / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def set_status(exp: str, **kw) -> None:
    st = exp_status(exp)
    st.update(kw, updated=now())
    (EXPS / exp / "status.json").write_text(json.dumps(st, indent=1))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_jobs() -> list[str]:
    out = []
    for st in EXPS.glob("*/status.json") if EXPS.exists() else []:
        s = json.loads(st.read_text())
        if s.get("state") == "running" and alive(s.get("pid", -1)):
            out.append(st.parent.name)
    return out


def gpu_hours_used(window_h: float) -> float:
    """Hours of GPU jobs started within the window; running jobs count with their full budget."""
    t0, total = now() - window_h * 3600, 0.0
    for r in ledger():
        if r.get("event") != "job" or r["start"] < t0:
            continue
        total += r["hours"] if "hours" in r else r["budget_hours"]
    return total


def gpu_mem_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def check_pinned() -> None:
    pin = AGENT / "pinned.json"
    if not pin.exists():
        raise Denied("main code not pinned yet (human: python -m physwm.agent.exec pin)")
    pinned = json.loads(pin.read_text())
    bad = [f for f in PINNED_FILES if sha(MAIN / f) != pinned.get(f)]
    if bad:
        raise Denied(f"pinned files changed since pin: {bad} (needs human review + re-pin)")


def consecutive_failures() -> int:
    n = 0
    for r in reversed(ledger()):
        if r.get("event") == "reset_failures":
            break
        if r.get("event") == "job_end":
            if r["state"] == "done":
                break
            n += 1
    return n


# ---------------------------------------------------------------- validation
def validate_spec(spec: dict, code: Path) -> dict:
    res = POLICY["resources"]
    for k in ("id", "hypothesis", "change", "baseline", "kind", "budget_hours", "seed"):
        if k not in spec:
            raise Denied(f"spec missing '{k}'")
    if spec["kind"] not in ("train", "eval_only"):
        raise Denied("kind must be train | eval_only")
    if not (0 < float(spec["budget_hours"]) <= res["large_experiment_max_hours"]):
        raise Denied(f"budget_hours must be in (0, {res['large_experiment_max_hours']}]")
    if sha(code / "pyproject.toml") != sha(MAIN / "pyproject.toml"):
        raise Denied("dependency changes are not allowed (pyproject.toml differs from main)")
    for f in ("configs/research_agent_policy.yaml", "manifests/final_test.json"):
        if (code / f).exists() and sha(code / f) != sha(MAIN / f):
            raise Denied(f"experiment modifies protected file {f}")
    protected = set(POLICY["data"]["protected_splits"]) | set(POLICY["data"]["validation_splits"])
    if spec["kind"] == "train":
        cfg_path = code / spec["train_config"]
        tcfg = yaml.safe_load(cfg_path.read_text())
        if tcfg["train_split"] in protected:
            raise Denied(f"training on {tcfg['train_split']} is not allowed")
        if Path(tcfg["data"]).resolve() != DATA.resolve():
            raise Denied("train config must read data from $WM/data")
    lora = spec.get("lora")
    if lora and not Path(lora).resolve().is_relative_to(EXPS.resolve()):
        raise Denied("lora must point into $WM/experiments")
    return spec


def check_resources(budget_h: float) -> None:
    res = POLICY["resources"]
    if consecutive_failures() >= POLICY["stopping_conditions"]["max_consecutive_failed_experiments"]:
        raise Denied("too many consecutive failed experiments; human must review (reset-failures)")
    if len(running_jobs()) >= res["max_concurrent_gpu_jobs"]:
        raise Denied(f"job already running: {running_jobs()}")
    if gpu_hours_used(24) + budget_h > res["max_gpu_hours_per_day"]:
        raise Denied(f"daily GPU budget exceeded ({gpu_hours_used(24):.1f} h used)")
    if gpu_hours_used(24 * 7) + budget_h > res["max_gpu_hours_per_week"]:
        raise Denied(f"weekly GPU budget exceeded ({gpu_hours_used(24 * 7):.1f} h used)")
    used = gpu_mem_used_mib()
    if used > res["gpu_free_max_used_mib"]:
        raise Denied(f"GPU busy ({used} MiB in use by someone else)")


# ---------------------------------------------------------------- ops
def op_status(args) -> None:
    res = POLICY["resources"]
    print(json.dumps({
        "host": os.uname().nodename,
        "running": running_jobs(),
        "gpu_mem_used_mib": gpu_mem_used_mib(),
        "gpu_hours_24h": round(gpu_hours_used(24), 2), "limit_24h": res["max_gpu_hours_per_day"],
        "gpu_hours_7d": round(gpu_hours_used(24 * 7), 2), "limit_7d": res["max_gpu_hours_per_week"],
        "consecutive_failures": consecutive_failures(),
        "experiments": {p.name: exp_status(p.name).get("state") for p in sorted(EXPS.glob("exp*"))}
        if EXPS.exists() else {},
        "splits": sorted(p.name for p in DATA.iterdir() if p.is_dir()),
    }, indent=1))


def op_submit(args) -> None:
    check_pinned()
    if not EXP_ID.match(args.exp):
        raise Denied("experiment id must match exp###_lowercase_words")
    exp_dir = EXPS / args.exp
    if exp_dir.exists() and exp_status(args.exp).get("state") not in ("failed", None):
        raise Denied(f"{args.exp} already exists with state {exp_status(args.exp).get('state')}")
    code = exp_dir / "code"
    if code.exists():  # resubmission after a failure: start from a clean snapshot
        shutil.rmtree(code)
    exp_dir.mkdir(parents=True, exist_ok=True)
    tar_bytes = sys.stdin.buffer.read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        tf.extractall(code, filter="data")
    spec = yaml.safe_load((code / "research" / "experiments" / args.exp / "spec.yaml").read_text())
    if spec.get("id") != args.exp:
        raise Denied("spec id does not match experiment id")
    validate_spec(spec, code)
    check_resources(float(spec["budget_hours"]))
    # preflight tests on the experiment's own code
    env = dict(os.environ, PYTHONPATH=str(code), MUJOCO_GL="egl")
    pre = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "tests"], cwd=code, env=env,
                         capture_output=True, text=True)
    (exp_dir / "preflight.log").write_text(pre.stdout + pre.stderr)
    if pre.returncode != 0:
        set_status(args.exp, state="failed", reason="preflight tests failed")
        raise Denied("preflight tests failed:\n" + pre.stdout[-2000:])
    (exp_dir / "spec.json").write_text(json.dumps({**spec, "commit": args.commit}, indent=1))
    log = open(exp_dir / "job.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "physwm.agent.exec", "job", "--exp", args.exp],
                         cwd=MAIN, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         env=dict(os.environ, PYTHONPATH=str(MAIN)))
    set_status(args.exp, state="running", pid=p.pid, commit=args.commit, started=now(),
               budget_hours=float(spec["budget_hours"]))
    ledger_append({"event": "job", "exp": args.exp, "commit": args.commit, "start": now(),
                   "budget_hours": float(spec["budget_hours"])})
    print(f"submitted {args.exp} (commit {args.commit[:10]}, pid {p.pid}, budget {spec['budget_hours']} h)")


def _run(cmd: list[str], env: dict, deadline: float, log) -> None:
    remaining = deadline - now()
    if remaining <= 0:
        raise TimeoutError("budget exhausted")
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True, timeout=remaining, stdout=log, stderr=subprocess.STDOUT)


def op_job(args) -> None:
    exp_dir = EXPS / args.exp
    spec = json.loads((exp_dir / "spec.json").read_text())
    code = exp_dir / "code"
    deadline = now() + float(spec["budget_hours"]) * 3600
    env_code = dict(os.environ, PYTHONPATH=str(code), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    t_start, state, reason = now(), "done", ""
    log = sys.stdout
    try:
        lora = spec.get("lora")
        if spec["kind"] == "train":
            _run([sys.executable, "-m", "physwm.vlm.sft", "--config", str(code / spec["train_config"]),
                  "--out", str(exp_dir / "model")], env_code, deadline, log)
            lora = str(exp_dir / "model" / "adapter")
        model = spec.get("model", "Qwen/Qwen3-VL-2B-Instruct")
        infer = [sys.executable, "-m", "physwm.vlm.infer", "--data", str(DATA), "--model", model,
                 "--style", spec.get("style", "center"), "--batch-size", str(spec.get("batch_size", 32))]
        infer += ["--lora", lora] if lora else []
        infer += ["--limit", str(spec["eval_limit"])] if spec.get("eval_limit") else []
        pred = exp_dir / "pred"
        _run(infer + ["--splits", *POLICY["data"]["validation_splits"], *DIAG_SPLITS,
                      "--out", str(pred / "normal.jsonl")], env_code, deadline, log)
        for mode in spec.get("ablations", ["none"]):
            _run(infer + ["--splits", POLICY["data"]["ablation_split"], "--image-mode", mode,
                          "--out", str(pred / f"{mode}.jsonl")], env_code, deadline, log)
        evaluate_experiment(args.exp)
    except Exception as e:  # noqa: BLE001
        state, reason = "failed", f"{type(e).__name__}: {e}"
        print("FAILED:", reason, flush=True)
    hours = (now() - t_start) / 3600
    set_status(args.exp, state=state, reason=reason, finished=now(), hours=round(hours, 3))
    # replace the reservation with the actual usage
    recs = ledger()
    for r in recs:
        if r.get("event") == "job" and r["exp"] == args.exp and "hours" not in r:
            r["hours"] = round(hours, 3)
    AGENT.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in recs))
    ledger_append({"event": "job_end", "exp": args.exp, "state": state, "hours": round(hours, 3), "t": now()})


def evaluate_experiment(exp: str, splits_key: str = "validation_splits", tag: str = "") -> dict:
    """Score with the pinned main-copy metrics. Writes results{tag}.md and metrics{tag}.json."""
    import numpy as np

    from physwm.eval.metrics import load_predictions, summarize
    from physwm.schema import load_split

    check_pinned()
    exp_dir = EXPS / exp
    splits = POLICY["data"][splits_key]  # the metrics below use these only
    shown = splits + (DIAG_SPLITS if splits_key == "validation_splits" else [])
    episodes = {ep.episode_id: ep for s in shown for _, ep in load_split(DATA, s)}
    base_dir = AGENT / ("val_baselines" if splits_key == "validation_splits" else "final_baselines")
    preds = {"static": base_dir / "static.jsonl", "const_velocity": base_dir / "const_velocity.jsonl"}
    for f in sorted((exp_dir / f"pred{tag}").glob("*.jsonl")):
        preds[f"{exp}:{f.stem}"] = f
    rows = []
    for name, path in preds.items():
        for r in summarize(load_predictions(path), episodes):
            rows.append({"method": name, **r})

    def get(method, split, task, col):
        for r in rows:
            if r["method"] == method and r["split"] == split and r["task"] == task:
                return r[col]
        return float("nan")

    me = f"{exp}:normal"
    ood = [s for s in splits if "ood" in s]
    id_split = [s for s in splits if "ood" not in s][0]
    metrics = {
        "primary_metric": POLICY["evaluation"]["primary_metric"] if not tag else "final_ood_target_mean",
        "value": float(np.mean([get(me, s, "B", "target_mean") for s in ood])),
        f"{id_split}_target_mean": get(me, id_split, "B", "target_mean"),
        f"{id_split}_all_mean": get(me, id_split, "B", "all_mean"),
        f"taskA_{id_split}_all_mean": get(me, id_split, "A", "all_mean"),
        "parse_rate_min": min(r["parse_rate"] for r in rows if r["method"] == me and r["split"] in splits),
        "static_ood_target_mean": float(np.mean([get("static", s, "B", "target_mean") for s in ood])),
    }
    if not tag:
        noimg = get(f"{exp}:none", id_split, "B", "target_mean")
        metrics["image_dependence_gap"] = noimg - metrics[f"{id_split}_target_mean"]
    for k, v in metrics.items():
        if isinstance(v, float) and not np.isfinite(v):
            raise ValueError(f"invalid metric {k}={v}")
    cols = ["method", "split", "task", "episodes", "parse_rate", "all_mean", "target_mean",
            "target_median", "target_skill", "moved_mean", "unmoved_mean"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (r["task"], r["split"], r["method"])):
        lines.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (exp_dir / f"results{tag}.md").write_text("\n".join(lines) + "\n")
    (exp_dir / f"metrics{tag}.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def op_results(args) -> None:
    d = EXPS / args.exp
    print(json.dumps(exp_status(args.exp), indent=1))
    for f in ("metrics.json", "results.md", "metrics_final.json", "results_final.md"):
        if (d / f).exists():
            print(f"\n## {f}\n" + (d / f).read_text())


def op_log(args) -> None:
    d = EXPS / args.exp
    for f in ("preflight.log", "job.log"):
        if (d / f).exists():
            lines = [x for x in (d / f).read_text().splitlines() if "it/s]" not in x and "s/it]" not in x]
            print(f"## {f}\n" + "\n".join(lines[-args.n:]))


def _episode_errors(method: str, splits: list[str], task: str) -> dict[str, dict[str, float]]:
    """split -> {episode_id: error} for `exp_id[:image_mode]` or a val baseline name.
    Task B: the pushed object's error; Task A: mean over all objects. Missing predictions score as
    unparsed (worst case), but a split with no predictions at all is refused."""
    import numpy as np

    from physwm.eval.metrics import load_predictions, per_object_rows
    from physwm.schema import load_split

    exp, _, mode = method.partition(":")
    if exp in BASELINE_NAMES and not mode:
        path = AGENT / "val_baselines" / f"{exp}.jsonl"
    elif EXP_ID.match(exp) and re.fullmatch(r"[a-z]*", mode):
        path = EXPS / exp / "pred" / f"{mode or 'normal'}.jsonl"
    else:
        raise Denied(f"method must be static, const_velocity or exp_id[:image_mode], got {method!r}")
    if not path.exists():
        raise Denied(f"no validation predictions for {method}")
    preds = {p["episode_id"]: p for p in load_predictions(path) if p["task"] == task}
    out = {}
    for s in splits:
        eps = [ep for _, ep in load_split(DATA, s)]
        if not any(ep.episode_id in preds for ep in eps):
            raise Denied(f"{method} has no Task {task} predictions on {s}")
        out[s] = {}
        for ep in eps:
            rows = per_object_rows(preds.get(ep.episode_id, {"task": task, "objects": {}}), ep)
            out[s][ep.episode_id] = (next(r["err"] for r in rows if r["target"]) if task == "B"
                                     else float(np.mean([r["err"] for r in rows])))
    return out


def op_compare(args) -> None:
    """Paired bootstrap between two methods, on the aggregate (mean over splits) and per split.
    Default: the primary metric (val OOD, Task B target mean). diff = a - b; negative means `a` is better."""
    import numpy as np

    allowed = POLICY["data"]["validation_splits"] + DIAG_SPLITS
    splits = args.splits or [s for s in POLICY["data"]["validation_splits"] if "ood" in s]
    bad = [s for s in splits if s not in allowed]
    if bad:
        raise Denied(f"compare runs on validation / diagnostic splits only {allowed}, got {bad}")
    ea, eb = _episode_errors(args.a, splits, args.task), _episode_errors(args.b, splits, args.task)
    rng = np.random.default_rng(0)
    per_split = {s: (np.array([ea[s][k] for k in sorted(ea[s])]), np.array([eb[s][k] for k in sorted(ea[s])]))
                 for s in splits}
    diffs = {s: float(a.mean() - b.mean()) for s, (a, b) in per_split.items()}
    point = float(np.mean(list(diffs.values())))
    boots, split_boots = [], {s: [] for s in splits}
    for _ in range(args.n_boot):
        d = []
        for s, (a, b) in per_split.items():
            idx = rng.integers(0, len(a), len(a))
            d.append(a[idx].mean() - b[idx].mean())
            split_boots[s].append(d[-1])
        boots.append(np.mean(d))
    thr = POLICY["evaluation"]["improvement_threshold"]

    def verdict(point, lo, hi):
        return ("a better" if hi < 0 and point < -thr else
                "b better" if lo > 0 and point > thr else "comparable / not significant")

    def ci(b):
        return [float(x) for x in np.percentile(b, [2.5, 97.5])]

    lo, hi = ci(boots)
    default = args.task == "B" and not args.splits
    print(json.dumps({
        "metric": POLICY["evaluation"]["primary_metric"] if default
        else f"task{args.task}_{'target' if args.task == 'B' else 'all'}_mean over {splits}",
        "a": args.a, "b": args.b,
        "a_value": float(np.mean([a.mean() for a, _ in per_split.values()])),
        "b_value": float(np.mean([b.mean() for _, b in per_split.values()])),
        "diff_a_minus_b": point, "ci95": [lo, hi],
        "per_split_diff": diffs,
        "per_split_ci95": {s: ci(b) for s, b in split_boots.items()},
        "per_split_verdict": {s: verdict(diffs[s], *ci(b)) for s, b in split_boots.items()},
        "per_split_n": {s: len(a) for s, (a, _) in per_split.items()},
        "improvement_threshold": thr, "verdict": verdict(point, lo, hi),
    }, indent=1))


def op_cancel(args) -> None:
    st = exp_status(args.exp)
    if st.get("state") == "running" and alive(st["pid"]):
        os.killpg(st["pid"], signal.SIGTERM)
        print("cancelled", args.exp)
    else:
        print("not running")


def op_gen(args) -> None:
    d = POLICY["data"]
    if not SPLIT_NAME.match(args.split) or not any(args.split.startswith(p) for p in d["agent_split_prefixes"]):
        raise Denied(f"split name must start with one of {d['agent_split_prefixes']}")
    if args.split in d["protected_splits"] + d["validation_splits"] + DIAG_SPLITS:
        raise Denied("protected / validation splits are fixed")
    if (DATA / args.split).exists():
        raise Denied(f"split {args.split} exists; datasets are immutable once generated")
    lo, hi = args.seed_start, args.seed_start + args.n
    for a, b in d["reserved_seed_ranges"]:
        if lo < b and a < hi:
            raise Denied(f"seed range overlaps reserved [{a}, {b})")
    if args.n > 50000:
        raise Denied("max 50000 episodes per generation call")
    cfg = (MAIN / args.config).resolve()
    if not cfg.is_relative_to((MAIN / "configs").resolve()):
        raise Denied("config must be a file in the main repo's configs/")
    log = open(AGENT / f"gen_{args.split}.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "physwm.sim.generate", "--config", str(cfg),
                          "--split", args.split, "--seed-start", str(lo), "--n", str(args.n),
                          "--out", str(DATA), "--workers", "20"], cwd=MAIN, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    print(f"generating {args.split} ({args.n} episodes) in background, pid {p.pid}; "
          f"log: {AGENT / f'gen_{args.split}.log'}")


def op_final_eval(args) -> None:
    check_pinned()
    token = AGENT / "approvals" / f"{args.exp}.ok"
    st = exp_status(args.exp)
    if not token.exists() or token.read_text().strip() != st.get("commit"):
        raise Denied("final test needs human approval: scripts/approve.sh " + args.exp)
    spec = json.loads((EXPS / args.exp / "spec.json").read_text())
    check_resources(1.0)
    code = EXPS / args.exp / "code"
    env = dict(os.environ, PYTHONPATH=str(code))
    lora = str(EXPS / args.exp / "model" / "adapter") if spec["kind"] == "train" else spec.get("lora")
    cmd = [sys.executable, "-m", "physwm.vlm.infer", "--data", str(DATA), "--splits",
           *POLICY["data"]["protected_splits"], "--style", spec.get("style", "center"),
           "--model", spec.get("model", "Qwen/Qwen3-VL-2B-Instruct"), "--batch-size", "32",
           "--out", str(EXPS / args.exp / "pred_final" / "normal.jsonl")]
    cmd += ["--lora", lora] if lora else []
    t0 = now()
    ledger_append({"event": "job", "exp": args.exp + ":final", "start": t0, "budget_hours": 1.0})
    subprocess.run(cmd, env=env, check=True, timeout=3 * 3600)
    recs = ledger()
    recs[-1]["hours"] = round((now() - t0) / 3600, 3)
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in recs))
    token.unlink()  # one approval = one final evaluation
    print(json.dumps(evaluate_experiment(args.exp, "protected_splits", "_final"), indent=1))


def op_pin(args) -> None:
    """Human only: verify the final-test manifest, compute baselines, pin eval-code hashes."""
    from physwm.eval import baselines as B
    from physwm.schema import load_split

    ok = subprocess.run([sys.executable, "-m", "physwm.agent.manifest", "--data", str(DATA),
                         "--verify", str(MAIN / POLICY["data"]["protected_manifest"])])
    if ok.returncode != 0:
        sys.exit("final-test manifest mismatch; refusing to pin")
    for key, sub in (("validation_splits", "val_baselines"), ("protected_splits", "final_baselines")):
        out = AGENT / sub
        out.mkdir(parents=True, exist_ok=True)
        files = {n: open(out / f"{n}.jsonl", "w") for n in B.BASELINES}
        for split in POLICY["data"][key] + (DIAG_SPLITS if key == "validation_splits" else []):
            for ep_dir, ep in load_split(DATA, split):
                for n, fn in B.BASELINES.items():
                    c = fn(ep, ep_dir)
                    files[n].write(json.dumps({"episode_id": ep.episode_id, "task": "B", "parsed": True,
                                               "objects": {str(i): c[i].tolist() for i in range(len(c))}}) + "\n")
        for f in files.values():
            f.close()
    pinned = {f: sha(MAIN / f) for f in PINNED_FILES}
    pinned["_pinned_at"] = dt.datetime.now().isoformat()
    (AGENT / "pinned.json").write_text(json.dumps(pinned, indent=1))
    ledger_append({"event": "pin", "t": now()})
    print("pinned", len(PINNED_FILES), "files; baselines written")


def op_reset_failures(args) -> None:
    ledger_append({"event": "reset_failures", "t": now()})
    print("failure counter reset")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    sub.add_parser("status")
    for name in ("submit", "job", "results", "cancel", "final-eval"):
        p = sub.add_parser(name)
        p.add_argument("--exp", required=True)
        if name == "submit":
            p.add_argument("--commit", required=True)
    p = sub.add_parser("log")
    p.add_argument("--exp", required=True)
    p.add_argument("--n", type=int, default=40)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--splits", nargs="+")
    p.add_argument("--task", choices=["A", "B"], default="B")
    p = sub.add_parser("gen")
    p.add_argument("--config", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    sub.add_parser("pin")
    sub.add_parser("reset-failures")
    args = ap.parse_args()
    EXPS.mkdir(parents=True, exist_ok=True)
    AGENT.mkdir(parents=True, exist_ok=True)
    try:
        {"status": op_status, "submit": op_submit, "job": op_job, "results": op_results,
         "log": op_log, "cancel": op_cancel, "compare": op_compare, "gen": op_gen, "final-eval": op_final_eval,
         "pin": op_pin, "reset-failures": op_reset_failures}[args.op](args)
    except Denied as e:
        ledger_append({"event": "denied", "op": args.op, "reason": str(e), "t": now()})
        print(f"DENIED: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
