# Corrected remote baseline — 2026-10-27

Host: `nc-10-4-2` (RTX 5080, 16 GB). The previous host `nc-10-4-3` was occupied by another user's GPU job. Existing data and checkpoints remain there.

Run root: `/var/tmp/toyi/wm/audit_v2_20261027`

Launch: background `nohup` + `timeout`, maximum six hours including setup/waits, with a three-hour limit for training. Launcher PID: `303145`; pipeline shell PID: `303146`. The source is an uploaded working-tree snapshot verified against the SHA-256 in [launch.json](launch.json), not a committed research-loop experiment. The pinned main executor and frozen test manifest were not changed.

Pipeline:
1. Remote tests (29 passed).
2. Generate 10,000 corrected training episodes, 2,000 validation episodes, and 1,200 diagnostic episodes in the isolated run root.
3. Compute static and constant-velocity baselines.
4. Train the unchanged seed-0 Qwen3-VL-2B LoRA configuration, with only dataset/output paths redirected.
5. Infer on every validation/diagnostic episode; run no-image, blank-image, and swapped-image ablations on `val_id`.
6. Produce `results/metrics.csv` and `results/table.md`.

A minimal simulation environment starts data generation while the existing full environment/model cache transfers from the previous host. Training waits for the transfer-complete marker. The GPU is checked between stages; a busy GPU causes a bounded wait rather than launching alongside someone else's compute job. No frozen final-test generation or evaluation is queued.

Inspect remotely:

```bash
ssh -o BatchMode=yes -o LogLevel=ERROR nc-10-4-2 \
  'cat /var/tmp/toyi/wm/audit_v2_20261027/status.json; tail -15 /var/tmp/toyi/wm/audit_v2_20261027/pipeline.log'
```

Detailed logs are under `logs/` (`generate_train.log`, `train.log`, `infer_normal.log`, etc.). The adapter will be under `model/adapter`. `events.jsonl` records stage transitions and `status.json` reports running/waiting/done/failed. These corrected-corpus results must not be merged into the legacy leaderboard without identifying the dataset version.

## Recovery on 2026-10-27

All 13,200 episodes and both baseline files finished. Training on `nc-10-4-2` failed during model loading at 14:12 UTC with `RuntimeError: No CUDA GPUs are available`; no training steps completed. Full `nvidia-smi` subsequently showed ERR!/N/A device telemetry. The identical torch environment initializes CUDA successfully on `nc-10-4-3`, identifying a host/driver problem rather than a missing torch installation. Low memory plus N/A utilization was an insufficient initial availability check.

Copied the completed run intact to **nc-10-4-3**, same run-root path; retained the original failed run on nc-10-4-2. Started a single six-hour recovery queue, launcher PID **3380643**, using `scripts/resume_corrected_training.py` (deployed outside the frozen code snapshot). It verifies the source archive checksum and all split counts, preserves the failed model directory, then waits for CUDA initialization, numeric low utilization, low memory and absence of compute jobs before starting training. Other users' jobs are untouched. Training remains capped at three hours. Failed-start details remain in `logs/train.log`; retry output will be `logs/train_retry.log`.

**Use nc-10-4-3 for monitoring now**, with `recovery_pipeline.log` and `status.json` under the same run root. The original `launch.json` intentionally records the first launch; this recovery note records the host change. The local availability guard has also been corrected to reject failed CUDA initialization and unknown utilization.
