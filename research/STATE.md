# Research state

_Maintained by the research agent; the human may edit it at any time._

## Objective
Improve the Task B target error on the validation OOD splits (`val_ood_objects`, `val_ood_forces`, `val_ood_config`) relative to `static` and `exp000_sft_baseline`, and measure the image-dependence gap.

## Leaderboard (val, metres; lower is better)
| exp | primary (val OOD target mean) | val_id target mean | Task A val_id | image gap | parse | verdict |
|---|---|---|---|---|---|---|
| static | 0.379 | 0.265 | – | – | – | trivial baseline (oracle t=0 perception) |
| exp000_sft_baseline | 0.314 (−0.065 vs static, CI [−0.075, −0.056]) | 0.155 | 0.016 | 0.331 | 1.000 | reference; gain mostly from val_ood_forces; val_ood_config worse than static |
| exp001_zeroshot_qwen3d | 0.755 (+0.376 vs static, CI [0.358, 0.396]) | 0.600 | 0.619 | 0.400 (artefact: `none` parse 0) | 0.996 min | zero-shot, worse than static and exp000 on every split; config = harness result; provisional (missing_rate not visible) |

Per-split Task B target mean (exp000 / static): objects 0.161 / 0.232, forces 0.510 / 0.664, config 0.269 / 0.241.

## Known facts (Phase 1 setup, 2026-10-24)
- Pushed objects mostly stop within ~0.2 s because of friction. Constant velocity extrapolated from 0.1 s is worse than static.
- Balls roll; about a third of pushed balls leave the table, which produces large errors. Look at medians as well as means.
- In a smoke test, zero-shot Qwen3-VL-2B with the `center` prompt outputs `[0,0,0]`. The native `bbox_3d` prompt gives real but poor estimates (~0.7 m error at t = 0).
- SFT overfits 100 episodes (loss 0.6 → 0.02), so the pipeline works.
- exp000: SFT Task B depends heavily on the image. no-image / blank / swap give 0.486 / 0.472 / 0.400 on val_id, all worse than static (0.265) (C-001).
- exp000: Task A perception holds on objects and forces (0.024 / 0.016) but degrades on val_ood_config (0.136).
- exp001 (zero-shot Qwen3-VL-2B, qwen3d prompt): every val OOD split is worse than static (C-003, tentative, pipeline-level). On Task A val_id, swap beats normal: −0.051, CI [−0.074, −0.028]. The correct image yields larger errors than a wrong one. The cause is open: label mismatch, frame flip, variance shrinkage, or missing-rate differences (IDEA-028/024).
- exp001: without an image the base model outputs nothing parseable (`none` parse 0.000), so `image_dependence_gap` is uninterpretable for zero-shot runs. Use `swap` as the reference.
- exp001: val_ood_config Task A target_median is 1.000, so most objects there are probably missing (1.0 m penalty). The reason (omission, label mismatch, truncation) is unknown.
- `agent.sh log` hides stdout lines that share a physical line with a tqdm frame (`exec.py:356` filter), so per-split stats printed by `infer.py` are invisible (IDEA-029, needs-human). Until that's fixed, don't route pre-review logging requirements through job.log.
- `sim_ood_config` changes 3 factors at once: camera elevation 12–25° (train 30–60°), 4–5 objects (train 1–3), corner cluster. The intrinsics in the prompt are constant across episodes.

## Queue
1. exp001_zeroshot_qwen3d: done (0.351 GPU h). Reported, post-review AMEND adopted. Verdicts are provisional until the human supplies `joblog_rates.txt` (outbox/exp001_joblog_missing_rate.md).
2. Cheap diagnostics from the exp000 review (IDEA-006/007/008), mostly analysis or tooling, not GPU. IDEA-009 done by human; IDEA-017 unblocked (diag splits exist; exp000 numbers need an eval_only spec with the exp000 adapter).

## Tooling (human, 2026-10-24)
- `compare` now reports `per_split_ci95` / `per_split_verdict`, takes `--splits` (any val_* or val_diag_* split), `--task A|B`, and `exp_id:mode` for ablations. Use it for exp001's Task A normal-vs-swap criterion in addition to the fixed 0.05 m margin.
- Diagnostic splits `val_diag_elev`, `val_diag_count`, `val_diag_corner` (300 episodes each) are inferred in every job and listed in results.md, not in the primary metric.
- Qwen3-VL bbox_3d centre/frame/units match our parser (camera frame, metres). Our qwen3d prompt is not Qwen's native template (adds K, frame sentence, multiple objects per query); exp001 measures "Qwen3-VL with our prompt" (IDEA-023).

## Open questions
- Why does zero-shot exp001 do worse with the correct image than with a swapped one, and what are its per-split missing rates? Needs the human to supply the job.log stats lines (outbox) or IDEA-028 read-out.
- Why does exp000 fail on val_ood_config: camera elevation, object count / omission (missing_rate), or cluster dynamics? (IDEA-006, IDEA-007; IDEA-017 now runnable via val_diag_elev/count/corner with an eval_only spec on the exp000 adapter.)
- Does exp000 use the stated impulse on val_ood_forces, or does it predict training-typical motion? (IDEA-008)
- Are the per-split differences outside the noise? `compare` now gives per-split CIs (IDEA-009 done); exp000 per-split CIs not yet computed.
- If the camera turns out to be the cause, which geometric input fixes it: per-patch xyz (IDEA-013), uv + depth output (IDEA-014), camera pose text (IDEA-016), or K-varying crops (IDEA-019)?
