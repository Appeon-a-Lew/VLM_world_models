# Research state

_Maintained by the research agent; the human may edit it at any time._

> **HUMAN NOTE (2026-10-24, executor bug, fixed in d4de761 and re-pinned).** Until now, every job ran **main's** `physwm` code, not the experiment snapshot. `python -m` puts the cwd (MAIN) on sys.path ahead of PYTHONPATH. Consequences:
> - exp000 is valid, since it changed no code.
> - **exp001 did not run its reviewed harness change.** It used main's infer.py, with 48 tokens per object instead of the 128-token qwen3d cap and no per-split stats print. That is why the stats lines are missing from job.log; `op_log` is not at fault. The val_ood_config Task A median of exactly 1.000 m is likely truncation under the 48-token cap.
> - exp001's numbers therefore describe main's harness, not the spec. Mark it invalid as specified, and re-run the same spec under a new id (e.g. `exp002_zeroshot_qwen3d_rerun`). The run is cheap (0.35 GPU-h). Jobs now run from the snapshot dir, so worktree code changes take effect.
> - The outbox request `exp001_joblog_missing_rate.md` is answered by this note: the lines were never printed.

## Objective
Improve the Task B target error on the validation OOD splits (`val_ood_objects`, `val_ood_forces`, `val_ood_config`) relative to `static` and `exp000_sft_baseline`, and measure the image-dependence gap.

## Leaderboard (val, metres; lower is better)
| exp | primary (val OOD target mean) | val_id target mean | Task A val_id | image gap | parse | verdict |
|---|---|---|---|---|---|---|
| static | 0.379 | 0.265 | – | – | – | trivial baseline (oracle t=0 perception) |
| exp000_sft_baseline | 0.314 (−0.065 vs static, CI [−0.075, −0.056]) | 0.155 | 0.016 | 0.331 | 1.000 | reference; gain mostly from val_ood_forces; val_ood_config worse than static |
| ~~exp001_zeroshot_qwen3d~~ | ~~0.755~~ | ~~0.600~~ | ~~0.619~~ | – | – | **invalid as specified**: ran main's infer.py (48-token cap, no stats); superseded by exp002_zeroshot_qwen3d_rerun |
| exp002_zeroshot_qwen3d_rerun | 0.735 (+0.356 vs static, CI [0.338, 0.375]) | 0.561 | 0.564 | uninterpretable (`none` missing 0.998); swap−normal A −0.063 [−0.089, −0.036] | 0.996 min | zero-shot reference; worse than static and exp000 on every split (C-004); (A) rejected: the correct image is worse than a swapped one |

Per-split Task B target mean (exp000 / static): objects 0.161 / 0.232, forces 0.510 / 0.664, config 0.269 / 0.241.

## Known facts (Phase 1 setup, 2026-10-24)
- Pushed objects mostly stop within ~0.2 s because of friction. Constant velocity extrapolated from 0.1 s is worse than static.
- Balls roll; about a third of pushed balls leave the table, which produces large errors. Look at medians as well as means.
- In a smoke test, zero-shot Qwen3-VL-2B with the `center` prompt outputs `[0,0,0]`. The native `bbox_3d` prompt gives real but poor estimates (~0.7 m error at t = 0).
- SFT overfits 100 episodes (loss 0.6 → 0.02), so the pipeline works.
- exp000: SFT Task B depends heavily on the image. no-image / blank / swap give 0.486 / 0.472 / 0.400 on val_id, all worse than static (0.265) (C-001).
- exp000: Task A perception holds on objects and forces (0.024 / 0.016) but degrades on val_ood_config (0.136).
- exp001 is invalid as specified (see HUMAN NOTE); exp002 re-ran it with the reviewed harness.
- exp002 (zero-shot Qwen3-VL-2B, qwen3d prompt, valid; gate passed, missing ≤ 0.017 on every val split):
  - Worse than static and exp000 on every val OOD split, for both tasks (C-004, pipeline-level).
  - On val_id, swap beats normal. Task A −0.063 [−0.089, −0.036], Task B −0.059 [−0.088, −0.028]. This is not a missing artefact. The candidate mechanisms are a frame flip or offset, or hedging toward a default under swap (IDEA-024/028, triggers met).
  - `none` yields no parseable output, so `image_dependence_gap` is uninterpretable zero-shot. Use swap.
  - val_ood_config Task A is 0.972 with missing 0.017: a real pipeline-level error, not truncation.
  - Raw Task A on the diag splits: count 0.887, elev 0.771, corner 0.519, against val_id 0.564. This may reflect distance from the model's default output rather than perception (IDEA-032 constant-centre baseline).
- `compare` weights each episode, while results.md all_mean weights each object. They can differ by several cm, e.g. exp002 Task A val_id 0.493 vs 0.564.
- Until d4de761, all jobs ran main's `physwm`, not the snapshot. Any earlier experiment with a code change must be re-run. exp000 had none, so it is valid.
- `sim_ood_config` changes 3 factors at once: camera elevation 12–25° (train 30–60°), 4–5 objects (train 1–3), corner cluster. The intrinsics in the prompt are constant across episodes.

## Queue
1. exp002_zeroshot_qwen3d_rerun: done (0.722 GPU h), reported; post-review AMEND adopted.
2. exp003_exp000_diag_eval (IDEA-017 + IDEA-006 ownstatic ride-along): running since 2026-10-24 (commit 6cec893; pre-review APPROVE round 2).
   - Hypothesis: elevation drives exp000's config failure. Verdict order: FALSIFIED → SUPPORTED → PARTIAL → INCONCLUSIVE.
   - Reproduction must hold first (primary 0.3136 ± 0.005, val_id 0.155 ± 0.005).
   - Fetch the log with `log … 400`.
   - Post-review notes: flag point-estimate-only R_f calls in the verdict sentence; the R_f interval accounts only for numerator noise.

   After it: IDEA-025/026, IDEA-006, IDEA-002. Zero-shot follow-ups IDEA-024/028/032 are triggered, but they are diagnostics of the zero-shot pipeline, not of the primary objective.
3. Cheap diagnostics from the exp000 review (IDEA-006/007/008), mostly analysis or tooling, not GPU. IDEA-009 done by human; IDEA-017 unblocked (diag splits exist; exp000 numbers need an eval_only spec with the exp000 adapter).

## Tooling (human, 2026-10-24)
- `compare` now reports `per_split_ci95` / `per_split_verdict`, takes `--splits` (any val_* or val_diag_* split), `--task A|B`, and `exp_id:mode` for ablations. Use it for exp001's Task A normal-vs-swap criterion in addition to the fixed 0.05 m margin.
- Diagnostic splits `val_diag_elev`, `val_diag_count`, `val_diag_corner` (300 episodes each) are inferred in every job and listed in results.md, not in the primary metric.
- Qwen3-VL bbox_3d centre/frame/units match our parser (camera frame, metres). Our qwen3d prompt is not Qwen's native template (adds K, frame sentence, multiple objects per query); exp001 measures "Qwen3-VL with our prompt" (IDEA-023).

## Open questions
- Why does zero-shot exp002 do worse with the correct image than with a swapped one: a frame flip or offset, or hedging toward a default output? (IDEA-024/028/032)
- Why does exp000 fail on val_ood_config: camera elevation, object count / omission (missing_rate), or cluster dynamics? (IDEA-006, IDEA-007; IDEA-017 now runnable via val_diag_elev/count/corner with an eval_only spec on the exp000 adapter.)
- Does exp000 use the stated impulse on val_ood_forces, or does it predict training-typical motion? (IDEA-008)
- Are the per-split differences outside the noise? `compare` now gives per-split CIs (IDEA-009 done); exp000 per-split CIs not yet computed.
- If the camera turns out to be the cause, which geometric input fixes it: per-patch xyz (IDEA-013), uv + depth output (IDEA-014), camera pose text (IDEA-016), or K-varying crops (IDEA-019)?
