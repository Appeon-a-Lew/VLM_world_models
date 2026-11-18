# Phase 1 — archived (v1 corpus)

Phase 1 ran on the first corpus (v1). Phase-2 numbers come from the corrected v3 corpus, so
**do not compare the levels across the two tables**; compare the *pattern*. Frozen record:
`research/STATE_phase1_archive.md` (2026-11-14), reports in `research/experiments/*/report.md`.
Every number below is quoted from those files.

## Leaderboard (validation, metres, lower is better)

| arm | val OOD target mean (primary) | val_id target (Task B) | Task A val_id | verdict |
|---|---|---|---|---|
| static (hold true t=0 pose) | 0.379 | 0.265 | – | trivial baseline |
| exp000 SFT baseline (centre) | 0.314 (−0.065 vs static) | 0.155 | 0.016 | fine-tuned reference |
| exp002 zero-shot Qwen3-VL | 0.735 (+0.356 vs static) | 0.561 | 0.564 | worse than static on every split |
| exp005 SFT reproduction | 0.316 (−0.063 vs static) | 0.157 | 0.015 | exp000 reproduced on a new box |
| exp008 uvz output | 0.326 (vs exp005 +0.010) | – | – | relative offsets did not help |

exp001 is excluded: it ran the wrong harness (48-token cap) and was re-run as exp002.
exp010 (camera pose in the prompt) was prepared but never submitted; it has no result.

## Per-split Task B (exp000 vs static)

| split | exp000 | static | reading |
|---|---|---|---|
| val_ood_objects | 0.161 | 0.232 | fine-tuning wins |
| val_ood_forces | 0.510 | 0.664 | fine-tuning wins, but large error remains |
| val_ood_config | 0.269 | 0.241 | **worse than doing nothing** |

## What phase 1 established

- **Fine-tuning is what makes the model usable.** Zero-shot Qwen3-VL with our prompt is worse
  than the do-nothing baseline (0.735 vs 0.379); the trained model beats it by ~6.5 cm. So the
  phase-1 pipeline works, even though the prompt/zero-shot path does not.
- **The gain is concentrated in the forces split, and the config split fails.** On the compound
  configuration shift the trained model is 2.8 cm *worse* than holding the true first pose.
- **No single factor explains the config failure** (exp003): elevation, object count and corner
  placement each beat static on their own, while the combined shift is worse — the failure is
  an interaction, not one axis.
- **The model does use the action text** (exp009): flipping the push direction raises Task B
  val_id from 0.157 to 0.417 m; halving/doubling the impulse costs +0.056/+0.037 m.
- **The camera-frame and per-axis stories were not supported** (exp006/exp007), and relative
  offsets (uvz) did not help (exp008).
- **Privileged anchoring does not rescue the config split** (exp004): even given the true t=0
  pose, motion skill on config is 0.152 vs 0.420 on val_id.

Phase 1's `val_ood_config` is the legacy compound shift (camera 12–25° + 4–5 objects + corner
cluster). Phase 2 replaced it with the camera-in-distribution `val_ood_config_nocam` axis
(`research/RESULTS_phase2.md` §5).
