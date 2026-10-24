# Research state

_Maintained by the research agent; the human may edit it at any time._

## Objective
Improve the Task B target error on the validation OOD splits (`val_ood_objects`, `val_ood_forces`, `val_ood_config`) relative to `static` and `exp000_sft_baseline`, and measure the image-dependence gap.

## Leaderboard (val, metres; lower is better)
| exp | primary (val OOD target mean) | val_id target mean | Task A val_id | image gap | parse | verdict |
|---|---|---|---|---|---|---|
| static | – | – | – | – | – | trivial baseline |

## Known facts (Phase 1 setup, 2026-10-24)
- Pushed objects mostly stop within ~0.2 s because of friction. Constant velocity extrapolated from 0.1 s is worse than static.
- Balls roll; about a third of pushed balls leave the table, which produces large errors. Look at medians as well as means.
- In a smoke test, zero-shot Qwen3-VL-2B with the `center` prompt outputs `[0,0,0]`. The native `bbox_3d` prompt gives real but poor estimates (~0.7 m error at t = 0).
- SFT overfits 100 episodes (loss 0.6 → 0.02), so the pipeline works.

## Queue
1. exp000_sft_baseline (train, style center)
2. exp001_zeroshot_qwen3d (eval only)

## Open questions
