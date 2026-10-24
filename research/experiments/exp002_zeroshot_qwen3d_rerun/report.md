# exp002_zeroshot_qwen3d_rerun: report

Status: done (0.722 GPU h, commit 9e70064). This re-runs the exp001 spec on the snapshot executor (≥ d4de761). exp001 ran main's harness and is invalid as specified.

Every number comes from these `scripts/agent.sh` calls:
- `results exp002_zeroshot_qwen3d_rerun`
- `log exp002_zeroshot_qwen3d_rerun`
- `compare exp002_zeroshot_qwen3d_rerun static` (default splits, and `--splits val_diag_elev val_diag_count val_diag_corner`)
- `compare exp002_zeroshot_qwen3d_rerun exp000_sft_baseline`
- `compare exp002_zeroshot_qwen3d_rerun:swap exp002_zeroshot_qwen3d_rerun:normal --task A|B --splits val_id`

## Hypothesis (unchanged from exp001)
- (A) The zero-shot Qwen3-VL-2B qwen3d pipeline uses the image to locate current centres.
  - Supported if Task A val_id all_mean(swap) − all_mean(normal) ≥ 0.05 m.
  - Rejected if < 0.02 m.
- (B) Its Task B target error is worse than static: `compare` ci95[0] > 0. This half is descriptive only, because static has oracle perception.
- Validity gate: missing_rate ≤ 5% on every validation split.

## Change
No training. The native 3D-grounding prompt. Harness instrumentation: a qwen3d cap of 128 tokens per object, and a per-split stats print.

## Snapshot check and validity gate: PASSED
Stats lines appear in job.log for all 7 normal-pass splits plus `none` and `swap`. This confirms that the snapshot's `infer.py` ran, which was the reviewer's condition. The rates pool Task A and Task B answers. The truncation rate is an upper bound: it counts answers that don't end in `]`.

| split | mode | answers | parse_rate | missing_rate | truncation_rate (upper bound) |
|---|---|---|---|---|---|
| val_id | normal | 1000 | 1.000 | 0.000 | 0.000 |
| val_ood_objects | normal | 1000 | 0.998 | 0.001 | 0.000 |
| val_ood_forces | normal | 1000 | 1.000 | 0.002 | 0.002 |
| val_ood_config | normal | 1000 | 1.000 | 0.017 | 0.019 |
| val_diag_elev | normal | 600 | 1.000 | 0.000 | 0.000 |
| val_diag_count | normal | 600 | 1.000 | 0.017 | 0.023 |
| val_diag_corner | normal | 600 | 1.000 | 0.003 | 0.002 |
| val_id | none | 1000 | 0.003 | 0.998 | 0.998 |
| val_id | swap | 1000 | 0.997 | 0.009 | 0.003 |

Every validation split is at or below 0.017 missing, so **the gate passes and every split is a model result**. The `none` ablation produces almost no parseable output (missing 0.998). This is a no-image ablation, not a validation split, so it is not gated.

## Primary metric (Task B target, mean 3D L2 in m; average over val OOD)
| | value |
|---|---|
| exp002 | 0.7350 |
| static | 0.3790 |
| exp002 − static | +0.3560, 95% CI [0.3381, 0.3747] → `b better` |
| exp000_sft_baseline | 0.3136 |
| exp002 − exp000 | +0.4214, 95% CI [0.4039, 0.4389] → `b better` |

Per split (Task B target). No improvement was expected: this is a zero-shot reference.

| split | exp002 mean | exp002 median | static mean | exp002 − static (CI95) | exp002 − exp000 (CI95) | const_velocity mean |
|---|---|---|---|---|---|---|
| val_ood_objects | 0.449 | 0.374 | 0.232 | +0.217 [0.192, 0.242] | +0.288 [0.263, 0.312] | 0.587 |
| val_ood_forces | 0.868 | 0.839 | 0.664 | +0.204 [0.171, 0.236] | +0.358 [0.326, 0.389] | 0.644 |
| val_ood_config | 0.888 | 0.892 | 0.241 | +0.647 [0.609, 0.685] | +0.619 [0.582, 0.655] | 0.570 |
| val_id (ref.) | 0.561 | 0.492 | 0.265 | – | – | 0.567 |

## Diagnostic splits (single-factor; descriptive)
| split | Task A all_mean | Task B target mean | static B | exp002 − static B (CI95) |
|---|---|---|---|---|
| val_diag_elev | 0.771 | 0.627 | 0.243 | +0.384 [0.342, 0.426] |
| val_diag_count | 0.887 | 0.846 | 0.213 | +0.633 [0.584, 0.683] |
| val_diag_corner | 0.519 | 0.531 | 0.315 | +0.216 [0.180, 0.251] |
| val_id (ref.) | 0.564 | 0.561 | 0.265 | – |

exp000 has no diag-split predictions, so it can't be compared here (IDEA-017).

## Task A (current positions, from results.md)
| split / mode | all_mean | target_median | parse |
|---|---|---|---|
| val_id normal | 0.564 | 0.458 | 1.000 |
| val_id swap | 0.475 | 0.371 | 1.000 |
| val_id none | 1.000 | 1.000 | 0.006 |
| val_ood_objects | 0.448 | 0.362 | 0.996 |
| val_ood_forces | 0.585 | 0.476 | 1.000 |
| val_ood_config | 0.972 | 0.983 | 1.000 |

## Image dependence
- `image_dependence_gap` = 0.439. **It can't be interpreted.** `none` parses to almost nothing (missing 0.998), so the gap is a penalty artefact. It must not be compared with exp000's 0.331.
- `swap` vs normal on val_id (from `compare`, which weights each episode equally):
  - Task A: −0.063, CI [−0.089, −0.036].
  - Task B target: −0.059, CI [−0.088, −0.028].
  - Swap's missing_rate is 0.009, against 0.000 for normal. Missingness can only shrink swap's advantage, so the advantage is not a missing-rate artefact (see the verdict).

## Verdict (after post-review)
- **(A) rejected, under both weightings.**
  - The criterion uses results.md all_mean, which weights each object: 0.475 − 0.564 = −0.089 m.
  - The paired `compare` weights each episode (`exec.py:390`), which is why its means are 0.430 (swap) and 0.493 (normal). It gives −0.063, CI [−0.089, −0.036].
  - Both are far below the 0.02 m rejection threshold, and the CI excludes 0. The gate passes, and the harness is now the reviewed one. With the correct image, this zero-shot pipeline is further from the truth than with another scene's image.
- **Missing objects can't explain it.**
  - A missing object scores 1.0 m, which is worse than either mode's mean error, so missingness penalises swap. Swap's Task A missing rate is at most 0.018.
  - At most this shrinks swap's advantage, by about 0.01 m. It can't create the advantage.
- **Explanations, restructured.**
  - Variance shrinkage alone would predict swap ≈ normal.
  - Swap can only win if the correct-image predictions are worse than an image-independent guess. Two mechanisms do that:
    - (i) a frame flip or offset, under which image-tracking predictions are anti-correlated with the truth;
    - (ii) hedging toward a default output under swap.
  - Label-to-box misassignment matters only together with (ii).
  - The trigger conditions for IDEA-024 (frame/scale audit) and IDEA-028 (read-out) are now met.
- **(B) supported (descriptive, pipeline-level).** CI [0.338, 0.375], and every per-split CI excludes 0 (C-004). As before, this says nothing about dynamics knowledge.
- **val_ood_config is a pipeline-level failure, not a format artefact.**
  - Missing objects account for at most 0.017 m of the Task A 0.972. Scoring has no clipping, so the 0.983 median is a real distance.
  - Mechanism (i) would also inflate config, so this isn't necessarily "perception" in the narrow sense.
  - The error persists at the exp001 level with truncation ≤ 0.019. exp001 is context only; its missing rate was never measured.
- **Factor breakdown (descriptive, for this pipeline's raw error).**
  - Task A error is 0.887 with 4–5 objects (count) and 0.771 at low elevation (elev), against 0.564 on val_id.
  - The corner cluster gives 0.519: no evidence that clustering raises the error.
  - There's no paired CI vs val_id (different episodes).
  - Caveat: this pipeline loses to a wrong image. So the per-split error may partly reflect how far each split's true positions sit from the model's default output, not perceptual difficulty. A constant-centre baseline scored per split (IDEA-032, `pred/constprior.jsonl`) is needed before reading this as perception. Whether exp000 shows the same ranking is IDEA-017's question.
- **vs exp000:** exp000 is better on every split, for both tasks.
  - Task B: see the table above.
  - Task A all_mean (`compare --task A`, 4 val splits): +0.547 [0.534, 0.560]. Every per-split CI excludes 0: id +0.478, objects +0.374, forces +0.502, config +0.833.
  - Training and answer style both differ between the two, so the gap can't be pinned on either one.

## Reviewer (post): AMEND, adopted
See `review_post.md`. All 7 amendments are adopted:
1. The (A) rejection is stated under both weightings, with the reason the means differ.
2. The missing-rate bound and its direction are added.
3. The explanations are restructured, and the IDEA-024/028 triggers noted.
4. The config failure is reworded as pipeline-level, and the exp001 sentence narrowed.
5. The factor ranking is hedged, with the default-output caveat and the IDEA-032 pointer.
6. The Task A comparison vs exp000 is added.
7. This section.

CLAIMS: C-004 (solid, pipeline-level), which replaces the withdrawn C-003.
