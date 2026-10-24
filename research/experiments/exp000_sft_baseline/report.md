# exp000_sft_baseline: report

Status: done (1.372 GPU h, commit 345017f). Every number below comes from `scripts/agent.sh results exp000_sft_baseline` or `scripts/agent.sh compare exp000_sft_baseline static`.

## Hypothesis
Plain LoRA SFT of Qwen3-VL-2B on Task A + B (cross-entropy loss on JSON answers, `style: center`) learns grounded push outcomes in-distribution but generalises worse to the OOD validation splits.

## Change
None. This is the reference baseline (`configs/sft_qwen3vl2b.yaml`, seed 0). Ablations `none`, `blank` and `swap` run on val_id.

## Primary metric (Task B, target, mean 3D L2 in m; average over the val OOD splits)
| | value |
|---|---|
| exp000_sft_baseline | 0.3136 |
| static | 0.3790 |
| diff (exp000 − static) | −0.0654, 95% CI [−0.0754, −0.0557] (paired bootstrap) |
| verdict from compare | a better |

Per split (Task B target; mean / median):
| split | exp000 mean | static mean | diff | exp000 median | static median | const_velocity mean |
|---|---|---|---|---|---|---|
| val_ood_objects | 0.161 | 0.232 | −0.071 | 0.080 | 0.154 | 0.587 |
| val_ood_forces | 0.510 | 0.664 | −0.154 | 0.089 | 0.050 | 0.644 |
| val_ood_config | 0.269 | 0.241 | **+0.028** | 0.162 | 0.108 | 0.570 |
| val_id (ref.) | 0.155 | 0.265 | | 0.062 | 0.145 | 0.567 |

## Perception (Task A, all objects, mean)
val_id 0.016, val_ood_objects 0.024, val_ood_forces 0.016, **val_ood_config 0.136**.

## Image dependence (val_id, Task B target mean)
| ablation | target mean |
|---|---|
| normal | 0.155 |
| none (no image) | 0.486 |
| blank | 0.472 |
| swap | 0.400 |

`image_dependence_gap` = 0.331 (err(no image) − err(image), per `physwm/agent/exec.py:326`). All three ablations are worse than `static` (0.265), so the model's Task B answers depend heavily on the image. Task A shows the same pattern (normal 0.016 vs none 0.223, blank 0.240, swap 0.249).

Parse rate: 1.000 on every split and ablation.

## Observations
1. The SFT model beats static on the aggregate primary metric. The gain comes from val_ood_objects and val_ood_forces.
2. **val_ood_config is the failure case.** SFT is worse than static on Task B there (+0.028), and Task A perception error rises from 0.016 to 0.136. The camera/scene-configuration shift breaks the model's 2D→3D mapping. The dynamics are likely not the problem: perception alone explains an error of that size.
3. On val_ood_forces the mean improves a lot (0.510 vs 0.664), but the median is worse than static (0.089 vs 0.050). The mean is dominated by objects that travel far (static moved_mean 0.997). The model partly predicts those, but it adds error on the many objects that barely move (unmoved_mean 0.019 vs 0.001).
4. In-distribution the model clearly learned something: val_id target 0.155 vs static 0.265, with target_skill 0.416.

## Provisional verdict
Hypothesis **partly supported**. SFT learns grounded push outcomes in-distribution, and the model relies on the image. OOD generalisation is uneven: it holds for new objects and forces (in mean) but fails for new configurations, and the failure looks perception-driven. exp000 is the reference point for all later experiments.

## Reviewer (post): AMEND, adopted
See `review_post.md`. I adopt all three amendments:
1. **The aggregate gain is concentrated in one split.** val_ood_forces supplies about 78% of the −0.065. Only the aggregate has a CI, so "config +0.028" is a point estimate with no CI (IDEA-009 adds per-split CIs). Static copies the GT t=0 positions (oracle perception), so exp000 vs static mixes perception and dynamics. Point estimates *support* the hypothesis: target_skill falls from 0.416 (val_id) to 0.305 / 0.232 / −0.118 on objects / forces / config.
2. **Observation 2 is withdrawn as stated.** Task A target error on val_ood_config (0.133) is about half of the Task B error (0.269), so perception does not explain it alone. The split also changes object count (4–5 vs 1–3), clustering (corner) and camera elevation (12–25° vs 30–60°) together. Other explanations still open: object omission scored at 1.0 m (`missing_rate` is not reported, and parse_rate only needs one object), camera-frame miscalibration at low elevation, and new dynamics (more collisions). Tests: IDEA-006 (own-static + missing_rate) and IDEA-007 (single-factor breakdown).
3. **Observation 3's mechanism is withdrawn.** unmoved_mean is the perception noise floor. val_ood_forces is bimodal by design (weak and strong pushes): the median reflects weak pushes and the mean strong ones. The leading untested hypothesis is that the model predicts training-typical motion regardless of the stated impulse. Test: IDEA-008.

**Final verdict:** hypothesis supported in point estimates (single seed). The aggregate OOD gain over static is significant (CI [−0.0754, −0.0557]) but driven by val_ood_forces. val_ood_config is worse than static, cause unresolved. Claims C-001 and C-002 are in `research/paper/CLAIMS.md`.
