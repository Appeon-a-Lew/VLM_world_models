# Research state

_Maintained by the research agent; the human may edit it at any time._

## Objective
Improve the Task B target error on the validation OOD splits (`val_ood_objects`, `val_ood_forces`, `val_ood_config`) relative to `static` and `exp000_sft_baseline`, and measure the image-dependence gap.

## Leaderboard (val, metres; lower is better)
| exp | primary (val OOD target mean) | val_id target mean | Task A val_id | image gap | parse | verdict |
|---|---|---|---|---|---|---|
| static | 0.379 | 0.265 | – | – | – | trivial baseline (oracle t=0 perception) |
| exp000_sft_baseline | 0.314 (−0.065 vs static, CI [−0.075, −0.056]) | 0.155 | 0.016 | 0.331 | 1.000 | reference; gain mostly from val_ood_forces; val_ood_config worse than static |

Per-split Task B target mean (exp000 / static): objects 0.161 / 0.232, forces 0.510 / 0.664, config 0.269 / 0.241.

## Known facts (Phase 1 setup, 2026-10-24)
- Pushed objects mostly stop within ~0.2 s because of friction. Constant velocity extrapolated from 0.1 s is worse than static.
- Balls roll; about a third of pushed balls leave the table, which produces large errors. Look at medians as well as means.
- In a smoke test, zero-shot Qwen3-VL-2B with the `center` prompt outputs `[0,0,0]`. The native `bbox_3d` prompt gives real but poor estimates (~0.7 m error at t = 0).
- SFT overfits 100 episodes (loss 0.6 → 0.02), so the pipeline works.
- exp000: SFT Task B depends heavily on the image. no-image / blank / swap give 0.486 / 0.472 / 0.400 on val_id, all worse than static (0.265) (C-001).
- exp000: Task A perception holds on objects and forces (0.024 / 0.016) but degrades on val_ood_config (0.136).
- `sim_ood_config` changes 3 factors at once: camera elevation 12–25° (train 30–60°), 4–5 objects (train 1–3), corner cluster. The intrinsics in the prompt are constant across episodes.

## Queue
1. exp001_zeroshot_qwen3d (eval only): being prepared
2. Cheap diagnostics from the exp000 review (IDEA-006/007/008/009), mostly analysis or tooling, not GPU

## Open questions
- Why does exp000 fail on val_ood_config: camera elevation, object count / omission (missing_rate), or cluster dynamics? (IDEA-006, IDEA-007; IDEA-017 needs a human for seed ranges.)
- Does exp000 use the stated impulse on val_ood_forces, or does it predict training-typical motion? (IDEA-008)
- Are the per-split differences outside the noise? There are no per-split CIs yet (IDEA-009).
- If the camera turns out to be the cause, which geometric input fixes it: per-patch xyz (IDEA-013), uv + depth output (IDEA-014), camera pose text (IDEA-016), or K-varying crops (IDEA-019)?
