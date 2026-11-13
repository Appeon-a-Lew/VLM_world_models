# Corrected baseline measurements — 2026-10-29

This work resumes the seed-0 corrected baseline at
`/var/tmp/toyi/wm/audit_v2_20261027`. It adds only blank-image and swapped-image
inference plus CPU scoring. It never invokes training and requires the existing
`model/adapter/adapter_config.json` before proceeding.

The local supervisor is `scripts/supervise_measurements.py`. Copy that file as
an extra file into the isolated remote run root (do not edit or replace
`code/`), then launch it with the run's venv Python and a detached process whose
PID is retained by the launcher. Use a six-hour wall-time cap. The supervisor
uses the existing frozen code snapshot and reads the existing data and adapter.
It creates `status_measurements.json`, `events_measurements.jsonl`, and
`logs/measure_*.log` separately from the original pipeline state files.
Supervisor SHA-256: `d10711bc00556fa22825a51591e70dae56485a9a9647898557294b6e2e4542bd`.

Before launch, verify on the target host that `libcuda.so.1:cuInit(0)` returns
zero, `nvidia-smi` reports numeric GPU utilization at or below 5 and memory
below 2500 MiB, and the compute-process query is empty. Repeat the check between
the two inference stages. Busy or unhealthy GPUs are polled for up to four
hours; no other user's process is stopped.

Existing blank/swap outputs are skipped only after checking unique
`(episode_id, task)` coverage against every `val_id` episode and both tasks.
New output goes to a `.partial` file and replaces the final path atomically
only after validation. Normal and no-image predictions and all baseline files
are validated before scoring. SIGTERM records a failed state and terminates the
child process group; stage timeout records `timed_out`. A later launch resumes
from completed, validated output files. Evaluation is CPU-only.

No seed-1/seed-2 job is launched as part of this measurement queue. The two
requested unchanged-config runs need separate budgeted scheduling after the
parent confirms timing and an available, CUDA-healthy GPU.
