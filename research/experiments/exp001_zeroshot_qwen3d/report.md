# exp001_zeroshot_qwen3d: report

Status: done (0.351 GPU h, commit 0776d91). Every number comes from one of these `scripts/agent.sh` calls:
- `results exp001_zeroshot_qwen3d`
- `compare exp001_zeroshot_qwen3d static`
- `compare exp001_zeroshot_qwen3d exp000_sft_baseline` (Task B; Task A with `--splits val_id val_ood_*`)
- `compare exp001_zeroshot_qwen3d:swap exp001_zeroshot_qwen3d:normal --task A|B --splits val_id`

**Both verdicts are provisional.** The validity gate can't be checked until the per-split missing rates are available (see below).

## Hypothesis (spec, fixed criteria)
- (A) Zero-shot Qwen3-VL-2B with the qwen3d (bbox_3d) prompt localises current 3D centres using the image.
  - Supported if Task A val_id all_mean(swap) − all_mean(normal) ≥ 0.05 m.
  - Rejected if < 0.02 m.
- (B) Its Task B target error is worse than static: `compare` ci95 lower bound > 0. This half is descriptive only, because static has oracle perception.
- Validity gate: object-missing rate ≤ 5% on every split. Otherwise those splits count as a harness/format result, not a model result.

## Change
No training. The base model with the qwen3d prompt. Harness instrumentation only: a qwen3d token cap of 128 per object, plus per-split parse/missing/truncation logging.

## Validity gate: NOT PASSED on any split (unverifiable)
The worktree `infer.py` prints one `[split] answers=… parse_rate=… missing_rate=… truncation_rate=…` line per split, with `flush=True`. `scripts/agent.sh log exp001_zeroshot_qwen3d 400` shows blank lines where those lines should be: 5 for the normal pass (4 splits), and 2 each for none and swap (1 split each). The reviewer confirmed this: of 19 log lines, only the three command lines have text.

The likely cause is that `op_log` (`physwm/agent/exec.py:356`) drops every line containing `s/it]` or `it/s]`. Each stats line probably shares a physical line with tqdm's final progress frame. This is unverified, because the raw job.log isn't readable through `agent.sh`. The human request is in `research/outbox/exp001_joblog_missing_rate.md`; the tooling fix is IDEA-029 (needs-human, pinned file).

Once the rates arrive, a per-split table goes here: parse / missing / "truncation_rate (upper bound: non-`]`-terminated)". Then the gate is re-applied. The logged rates pool Task A and Task B, so they settle the gate but can't separate the missing rate for normal vs swap by task (IDEA-028).

**val_ood_config is a harness/format result, not a model result.**
- Task A all_mean is 0.966 and Task A target_median is 1.000 there. A missing object scores exactly 1.0 m (`MISSING_PENALTY`), so a target median of 1.000 means at least half of the target objects got the penalty.
- Rough estimate, assuming errors of about 0.5 m when an object is found: a mean of 0.966 implies about 90% of objects missing.
- Object names are unique within a scene, so parser name collisions are ruled out (reviewer).
- The cause is unknown: omission, label mismatch, or truncation.

## Primary metric (Task B target, mean 3D L2 in m; average over val OOD)
| | value |
|---|---|
| exp001 | 0.7554 |
| static | 0.3790 |
| exp001 − static | +0.3764, 95% CI [0.3578, 0.3964] → `b better` |
| exp000_sft_baseline | 0.3136 |
| exp001 − exp000 | +0.4418, 95% CI [0.4243, 0.4606] → `b better` |

Per split (Task B target):
| split | exp001 mean | exp001 median | static mean | exp001 − static (CI95) | exp001 − exp000 (CI95) | parse |
|---|---|---|---|---|---|---|
| val_ood_objects | 0.505 | 0.404 | 0.232 | +0.274 [0.244, 0.303] | +0.344 [0.316, 0.372] | 1.000 |
| val_ood_forces | 0.870 | 0.922 | 0.664 | +0.206 [0.169, 0.240] | +0.360 [0.325, 0.392] | 1.000 |
| val_ood_config | 0.891 | 0.909 | 0.241 | +0.650 [0.612, 0.688] | +0.621 [0.585, 0.658] | 1.000 |
| val_id (ref.) | 0.600 | 0.525 | 0.265 | – | – | 1.000 |

Against const_velocity (target mean; no CI), exp001 wins only on objects:
| split | const_velocity | exp001 |
|---|---|---|
| val_ood_objects | 0.587 | 0.505 (better) |
| val_id | 0.567 | 0.600 |
| val_ood_forces | 0.644 | 0.870 |
| val_ood_config | 0.570 | 0.891 |

## Task A (current positions)
From `results.md`:
| split / mode | all_mean | target_median | parse |
|---|---|---|---|
| val_id normal | 0.619 | 0.473 | 1.000 |
| val_id swap | 0.548 | 0.392 | 1.000 |
| val_id none | 1.000 | 1.000 | 0.000 |
| val_ood_objects | 0.531 | 0.374 | 0.996 |
| val_ood_forces | 0.634 | 0.500 | 1.000 |
| val_ood_config | 0.966 | 1.000 | 1.000 |

`compare` reports a different Task A val_id mean for normal: 0.529 (swap 0.478). The likely reason is that results.md weights each object while `compare` weights each episode; this isn't verified in code. If so, the gap suggests per-object error is higher in scenes with more objects.

exp001 − exp000, Task A all_mean (`compare --task A`):
| split | diff (CI95) |
|---|---|
| val_id | +0.514 [0.488, 0.540] |
| val_ood_objects | +0.431 [0.411, 0.454] |
| val_ood_forces | +0.535 [0.509, 0.559] |
| val_ood_config | +0.828 [0.800, 0.857] |

All four CIs exclude 0. The two runs differ in two ways, training and answer style, so the gap can't be pinned on either one.

## Image dependence
- **`image_dependence_gap` = 0.400 is uninterpretable for this run.** It must not be compared with exp000's 0.331.
  - The `none` ablation has parse_rate 0.000, so every object scores the 1.0 m penalty, and the gap is exactly 1.000 − 0.600.
  - Without an image the base model produces no parseable bbox_3d.
- The `swap` reference is a real scene image, but the wrong one. It beats normal on val_id on both tasks:
  - Task A: swap − normal = −0.051, CI [−0.074, −0.028].
  - Task B target: −0.062, CI [−0.090, −0.035].
  - Swap's Task B parse rate is 0.994.

## Verdict (after post-review, provisional until joblog_rates.txt exists)
- **(A) rejected, at harness level (model + prompt + parser).** swap − normal on Task A val_id is −0.051 m, CI [−0.074, −0.028]. That is below the 0.02 m rejection threshold and of the wrong sign. The image changes the output, but the correct image gives *larger* errors than a wrong one. Open explanations, none tested:
  1. **Label mismatch.** With the true image the model names what it sees rather than echoing the prompt names, so slots go missing. With a wrong image it echoes the prompt.
  2. **Frame flip or offset** relative to our camera frame. Image-tracking predictions would then be anti-correlated with the truth, while swap predictions are uncorrelated (IDEA-024, raised in priority).
  3. **Variance shrinkage.** With a mismatched image the model falls back to low-variance stock coordinates, which beat noisy scene-specific guesses.

  Different missing rates under normal vs swap could also cause the result on their own. IDEA-028 (a read-out of existing pred files, no generation) separates these explanations.
- **(B) supported, at pipeline level.** CI [0.358, 0.396], and every per-split CI excludes 0 (smallest: forces +0.206 [0.169, 0.240]). Missing objects only inflate exp001's error, so (B) holds whatever the missing rate is. It is still a pipeline result, not a statement about the model's dynamics knowledge.
  - Task A error alone exceeds static's whole Task B error on val_id (0.619 vs 0.265) and objects (0.531 vs 0.232).
  - It does not on forces (Task A 0.634 vs static 0.664).
- **val_ood_config:** a harness/format result (see the gate section).
- **exp001 vs exp000:** exp000 is better on every split for both tasks. This is not attributable to training or to answer style alone.

Not available: the diagnostic splits (`val_diag_*`). This job's commit predates them, so they aren't in results.md.

## Reviewer (post): AMEND, adopted
See `review_post.md`. All 8 amendments are adopted:
1. Both verdicts are provisional, (B) worded at pipeline level and (A) at harness level, with the explanations listed.
2. Swap-vs-normal CIs added, and the results.md vs `compare` difference in the Task A mean noted.
3. const_velocity comparison corrected.
4. The "Task A > static Task B" claim restricted to val_id and objects.
5. The config paragraph strengthened.
6. The image gap declared uninterpretable.
7. Task A per-split CIs vs exp000 added.
8. A placeholder left for the rates table.

CLAIMS: C-003 (tentative, pipeline-level).
