# Research state

_Maintained by the research agent; the human may edit it at any time._

> **HUMAN NOTE (2026-10-24, GPU box moved to nc-10-1-1).** nc-11-1-2 went offline ("No route to host"). The executor now runs on **nc-10-1-1** (RTX 5080, same stack: torch 2.11, transformers 5.20.2, peft 0.21.2, MuJoCo 3.17; 13/13 tests pass). Consequences:
> - **Everything stored only on the old box is unreachable**: the exp000–003 adapters, predictions, job logs and the GPU ledger. The reports in `research/experiments/` and the numbers in this file remain the record. The ledger restarts at 0, `scripts/agent.sh results/log/compare` on exp000–003 will fail, and `experiments` in `status` is empty.
> - **All data was regenerated** from the same configs and seeds. meta.json, traj.npz, depth and seg are byte-identical; the RGB PNGs differ by ±1/255 in ≤8 pixels per image (EGL render noise, also seen between two runs on the same box). Static val OOD is 0.3789670302327517, the same as before, and all per-split static/const-vel baselines match.
> - The final-test manifest is now **v2** (human-approved): RGB files enter its hash by name only. Re-pinned.
> - **exp004 cannot run as written**: its `lora:` points at the exp000 adapter, which is gone. First re-train exp000 unchanged under a new id (e.g. `exp005_exp000_repro`, ~1.4 GPU-h), then point exp004's `lora:` at that adapter. The repro also checks whether the new box reproduces exp000's numbers within noise. Report any drift; don't hide it.
> - **IDEA-036 is done on the data side**: `val_diag_elev_corner` (300 episodes, `configs/sim_diag_elev_corner.yaml`, seeds 8.8M) is in `diagnostic_splits`, inferred in every job and scorable with `compare --splits`. Static 0.279, const-vel 0.574.
> - Another lab user may be using the GPU's compute (seen at 91 % utilisation, <1 GB memory). Jobs still launch, but they may run slower than the old timings.

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

| exp005_exp000_repro | 0.316 (−0.063 vs static, CI [−0.073, −0.054]) | 0.157 | 0.015 | 0.343 | 1.000 | exp000 retrained unchanged on nc-10-1-1: REPRODUCED (tentative, C-009); now the adapter-bearing reference; blank-ablation order flipped vs exp000 |

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
- exp003 (exp000 adapter on the diag splits; reproduces exp000 bit-exactly, C-005):
  - No single config factor reproduces exp000's config failure. Every single-factor split beats static (elev −0.053, count −0.072, corner −0.055), while config is +0.028 [0.013, 0.042] (C-006).
  - Shares of the config shift: R_elev 0.417, R_count 0.273, R_corner 0.396. All three are partial, and the result is point-estimate "diffuse".
  - exp000 Task A perception: count 0.019 (no degradation, C-008 tentative), elev 0.082, corner 0.078, config 0.136.
  - Task B beats own-perception static on all 7 splits (C-007).
  - On config, the own-perception cost is +0.056 [0.048, 0.065] (CI-solid). The dynamics gain falls from −0.113 (id) to −0.028. Motion-normalised skill K is 0.424 (id) vs 0.116 (config). Which part is larger isn't CI-separable.
  - missing_rate is 0.000 everywhere for exp000.
- `compare` weights each episode, while results.md all_mean weights each object. They can differ by several cm, e.g. exp002 Task A val_id 0.493 vs 0.564.
- Until d4de761, all jobs ran main's `physwm`, not the snapshot. Any earlier experiment with a code change must be re-run. exp000 had none, so it is valid.
- `sim_ood_config` changes 3 factors at once: camera elevation 12–25° (train 30–60°), 4–5 objects (train 1–3), corner cluster. The intrinsics in the prompt are constant across episodes.

## Queue
1. exp002_zeroshot_qwen3d_rerun: done (0.722 GPU h), reported; post-review AMEND adopted.
2. exp003_exp000_diag_eval: done (0.214 GPU h), reported; post-review AMEND adopted. Next: the planner picks between the cheap discriminators IDEA-034 (re-anchor on true t=0) / IDEA-035 (contact split) and the first improvement attempts. IDEA-036 (elev × corner split) needs a human data config.

   Older queue: IDEA-025/026, IDEA-006, IDEA-002. Zero-shot follow-ups IDEA-024/028/032 are triggered, but they are diagnostics of the zero-shot pipeline, not of the primary objective.
3. Cheap diagnostics from the exp000 review (IDEA-006/007/008), mostly analysis or tooling, not GPU. IDEA-009 done by human; IDEA-017 unblocked (diag splits exist; exp000 numbers need an eval_only spec with the exp000 adapter).

## Tooling (human, 2026-10-24)
- `compare` now reports `per_split_ci95` / `per_split_verdict`, takes `--splits` (any val_* or val_diag_* split), `--task A|B`, and `exp_id:mode` for ablations. Use it for exp001's Task A normal-vs-swap criterion in addition to the fixed 0.05 m margin.
- Diagnostic splits `val_diag_elev`, `val_diag_count`, `val_diag_corner` (300 episodes each) are inferred in every job and listed in results.md, not in the primary metric.
- Qwen3-VL bbox_3d centre/frame/units match our parser (camera frame, metres). Our qwen3d prompt is not Qwen's native template (adds K, frame sentence, multiple objects per query); exp001 measures "Qwen3-VL with our prompt" (IDEA-023).

## Unblocked (2026-10-25)
- The torchvision gap was fixed by the human; exp005 reran and finished (1.41 GPU-h). Adapter: `/var/tmp/toyi/wm/experiments/exp005_exp000_repro/model/adapter`. Read exp004 and later exp000-derived diagnostics against exp005, not exp000's recorded numbers. Next: exp004 (prepare in progress), then IDEA-040 (no-GPU prior baseline), IDEA-025/026.

## Open questions
- Why does zero-shot exp002 do worse with the correct image than with a swapped one: a frame flip or offset, or hedging toward a default output? (IDEA-024/028/032)
- exp000's val_ood_config failure isn't explained by any single factor (exp003). Is it an interaction (elev × corner, IDEA-036) or additive? Is the lost dynamics gain lost skill, or harder multi-body contact (IDEA-035)? Does the split depend on the decomposition path (IDEA-034)?
- Does exp000 use the stated impulse on val_ood_forces, or does it predict training-typical motion? (IDEA-008)
- Are the per-split differences outside the noise? `compare` now gives per-split CIs (IDEA-009 done); exp000 per-split CIs not yet computed.
- If the camera turns out to be the cause, which geometric input fixes it: per-patch xyz (IDEA-013), uv + depth output (IDEA-014), camera pose text (IDEA-016), or K-varying crops (IDEA-019)?
