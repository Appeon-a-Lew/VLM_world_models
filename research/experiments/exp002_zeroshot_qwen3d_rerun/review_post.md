# exp002_zeroshot_qwen3d_rerun: post-review

Verdict: **AMEND** (wording only; every number checks out and every verdict stands)

Calls run by the reviewer:
- `results exp002_zeroshot_qwen3d_rerun`
- `log exp002_zeroshot_qwen3d_rerun`
- `compare exp002 static` (default splits; and `--splits val_diag_elev val_diag_count val_diag_corner`)
- `compare exp002 exp000_sft_baseline` (default)
- `compare exp002 exp000_sft_baseline --task A --splits val_id val_ood_objects val_ood_forces val_ood_config` (new; not in the report)
- `compare exp002:swap exp002:normal --task A --splits val_id` and the same with Task B

I also read `exec.py:362-441`, which holds `_episode_errors` and `op_compare`, and `prompts.py:91-108`, which holds `swap_partners`. They tell how the compare numbers are weighted and how swap partners are chosen.

## Number check
Every number in the report matches tool output after rounding. That covers:
- the gate table (9 stats lines)
- the primary metric and its CIs against static and against exp000
- the per-split table, including medians and const_velocity
- the diagnostic table, including CIs
- the Task A table
- `image_dependence_gap` 0.439
- both swap−normal compares

Run hours are 0.722 against a budget of 1.5. Commit 9e70064 matches the approved pre-review.

## 1. Gate and snapshot check: PASSES (confirmed)
- `agent.sh log` shows all 9 stats lines: 7 for the normal pass, plus `[val_id] mode=none` and `[val_id] mode=swap`.
  - This meets pre-review condition 1: the snapshot's `infer.py` ran.
  - It also confirms that job.log prints are visible, so IDEA-029's question is closed.
- Missing on gated splits (val_id, val_ood_*): 0.000, 0.001, 0.002, 0.017. All are ≤ 0.05.
- Missing on the diagnostic splits (descriptive only): 0.000, 0.017, 0.003.
- The truncation upper bound is ≤ 0.023 everywhere, so the 128-token cap is sufficient.
- The pooled rates agree with results.md. `none` has pooled parse 0.003, and results.md has Task A 0.006 with Task B 0.000.

## 2. (A) rejected; swap beats normal: CONFIRMED, with additions
**The verdict holds under either weighting.**

| weighting | swap − normal (Task A, val_id) |
|---|---|
| object-pooled (results.md all_mean; the criterion as written) | 0.475 − 0.564 = −0.089 |
| episode-weighted (`compare`) | −0.063, CI [−0.089, −0.036] |

- I checked the code: `_episode_errors` with Task A takes each episode's mean over its objects, and the bootstrap then averages over episodes. Its a/b values (swap 0.430, normal 0.493) therefore differ from results.md's 0.475 and 0.564. That is expected, not a discrepancy.
- The gap between the two weightings (0.564 vs 0.493 for normal) means episodes with more objects have higher per-object error *within* val_id. That fits the count effect on the diagnostic splits.
- The CI upper bound (−0.036) is below the 0.02 m rejection threshold, and also below 0. So (A) is rejected however the criterion is computed. More than that: swap is significantly *better*.
- Task B shows the same: −0.059 [−0.088, −0.028].

**"Not a missing artefact" is justified, and the report should state the bound.**
- Missing slots score 1.0 m, which is above both modes' mean error (about 0.43-0.56). Missing objects therefore inflate swap's error and work *against* the observed direction.
- Swap's pooled missing rate is 0.009. Task A and Task B have the same slots per episode, so Task A missing is at most 0.018, even if every missing slot was Task A.
- Results.md gives swap Task A parse 1.000.
- So missingness could at most shrink swap's advantage, by about 0.018 × (1.0 − 0.43) ≈ 0.01 m. It cannot create it. The same bound covers truncation (0.003).

**What remains: the three named explanations are not on equal footing.**
- **Variance shrinkage alone cannot produce swap < normal.**
  - Suppose the predictions were image-independent and shrunk to a default. Then normal and swap would be the same predictor, and their errors would be equal in expectation.
  - `swap_partners` pairs within the same object-count group (prompts.py:91), so count does not differ between the modes either.
  - The result therefore requires the correct-image predictions to be *worse than an image-independent predictor*.
- There are two ways to get that:
  - (i) **Systematic anti-informative transform under the correct image.** Examples: an axis sign flip, a mirrored or offset frame, or a depth-axis confusion.
    - For a mirrored axis with GT ~ N(0, σ²), E|2x| ≈ 1.60σ, while E|x − x'| for an unrelated scene's x' is ≈ 1.13σ. Wrong-image predictions win.
    - A uniform scale error does not have this property. IDEA-024's own estimate (≤ 0.09 m) also rules out the Cube R-CNN virtual-depth convention as the main cause.
  - (ii) **Differential hedging.**
    - With the correct image, the model commits to precise but wrong locations. With a mismatched image, it does not find the named objects (prompt labels vs other scene) and falls back toward a central default that happens to be closer.
    - Under this explanation, the normal predictions are worse than a constant prior, and std(swap) < std(normal).
- **Label-to-box misassignment is not a third independent mechanism.**
  - Permuted labels within the correct scene give errors of about the inter-object distance. A random same-count scene gives errors of a similar size.
  - Misassignment explains swap < normal only together with (ii), i.e. when swap hedges.
  - It is still worth measuring, because it would steer the fix (IDEA-004 vs frame correction).
- **Discriminator (no GPU, existing exp002 preds): IDEA-024 + IDEA-028, whose trigger conditions are now met.**
  - Per-axis Pearson r(pred, GT) under normal. r < 0 on some axis points to (i).
  - std(normal) vs std(swap), plus a constant-centre prior scored by pinned code. The prior beating normal, with std(swap) < std(normal), points to (ii).
  - Hungarian-matched vs label-matched Task A error under normal. A large drop means misassignment.

## 3. val_ood_config: wording needs narrowing
- **Not a format artefact: confirmed.**
  - Config has missing 0.017 and truncation ≤ 0.019. Missing therefore contributes at most 0.017 m of the 0.972 Task A all_mean, so at least 0.955 m is error on parsed predictions.
  - metrics.py does not clip errors (`MISSING_PENALTY` is used only for `p is None`). The median of 0.983 is a real L2 distance that happens to lie near 1.0, not a penalty floor.
- **"Genuine model error" is too narrow. The finding is pipeline-level:** Qwen3-VL-2B base, plus our qwen3d prompt (text K + FRAME sentence + multi-object query; IDEA-023 is untested), plus our parser and frame read-out.
  - Section 2 shows that this pipeline loses to a wrong image on val_id. A frame or convention error (explanation i) would therefore also inflate config, and it would not be "the model cannot localise".
- **"exp001's config collapse was not caused by truncation" goes beyond the evidence.** exp001's missing rate was never measured. What exp002 shows is narrower: under the reviewed harness the config error stays at the same level (Task B vs static +0.647 here, against +0.650 in exp001) with negligible truncation. exp001 is context only.

## 4. Diagnostic factor ranking: not adequately hedged
- It is right that the report says there is no paired CI. The diagnostic splits and val_id are different episodes. The big gaps vs val_id are large next to plausible unpaired sampling error at n = 300/500: count +0.32 m, elev +0.21 m. As descriptive statements, these are fine.
- "Clustering doesn't matter" and "corner ≈ val_id" claim a null without any test, from 0.519 vs 0.564 unpaired. Replace them with "no evidence that corner clustering raises the error".
- **Substantive caveat that is missing.** Section 2 shows this pipeline is worse than an image-independent predictor on val_id. Per-split error may therefore track how far each split's GT centres lie from the pipeline's default output, rather than "localisation difficulty".
  - Example: 4-5 objects spread wider and low elevation lengthens depths. Either changes the error of a near-constant predictor.
  - The ranking can't be read as a ranking of perceptual factors until a constant-centre prior is scored on each diagnostic split (IDEA-032 below). "Object count is the largest single factor" should be prefixed with "for this pipeline's raw error".

## Other checks
- **Parse rates**: ≥ 0.996 for normal on every split. The parse-rate comparison between swap and normal is covered under 2.
- **Image-dependence gap**: correctly declared uninterpretable (`none` gives missing 0.998). Swap is the right control, and it shows *negative* image dependence. For a zero-shot model this is not shortcut learning (nothing was learned). It is a pipeline read-out problem to diagnose, not a model property to cite.
- **vs exp000**: the report's "better on every split" is Task B only. Task A (my compare) agrees: aggregate +0.547 [0.534, 0.560]. Per split: val_id +0.478 [0.452, 0.502], objects +0.374 [0.355, 0.395], forces +0.502 [0.477, 0.526], config +0.833 [0.803, 0.864]. The two-factor confound (training and answer style) is stated correctly.
- **(B)**: supported, descriptive. Per-split CIs all exclude 0. The caveat about oracle perception is stated.

## Alternative explanations for the headline (swap < normal) and the cheapest discriminators
1. Frame or convention transform under the correct image: an axis flip or offset. Read per-axis r and a fitted sign/offset correction (IDEA-024 (ii) plus IDEA-028 (c)). No GPU.
2. Differential hedging: swap shrinks toward a default that beats committed wrong outputs. Read std(pred) per mode plus the constant-centre prior (IDEA-028 (c)(d)). No GPU.
3. Label-to-box misassignment, together with 2: compare Hungarian-matched and label-matched error (IDEA-032 (b)). No GPU.
4. Prompt additions (text K, FRAME sentence) push the model off its native output convention, which yields 1 as a side effect: IDEA-023. About 0.3 GPU-h; run it only after 1-3.

Backlog entries are added or updated in `research/ideas/BACKLOG.md`: notes on IDEA-024 and IDEA-028, and a new IDEA-032, all with `source: review of exp002_zeroshot_qwen3d_rerun`.

## Required amendments to report.md
1. **(A) paragraph.** State that the rejection holds under both weightings. The CI upper bound −0.036 is below the 0.02 threshold and below 0. Quote compare's a/b values (swap 0.430, normal 0.493) and note that they are episode-weighted (`exec.py:390`), which is why they differ from results.md.
2. **"Not a missing artefact."** Add the bound and its direction. Missing scores 1.0 m, above both modes' mean, so it penalises swap. Task A swap missing is ≤ 0.018 even in the worst case, so the effect on the diff is at most about 0.01 m, and in the opposite direction to the observed advantage.
3. **Explanations.** Replace "three explanations remain open" with the structure from section 2:
   - Shrinkage alone predicts swap ≈ normal. The result requires normal to be worse than an image-independent predictor, through (i) an anti-informative transform or (ii) differential hedging.
   - Misassignment matters only together with (ii).
   - State that the IDEA-024/028 trigger conditions are now met.
4. **Config paragraph.**
   - Replace "real model failure" with "pipeline-level failure (model + our qwen3d prompt + parser + frame read-out), not a parse, missing or truncation artefact: missing contributes ≤ 0.017 m of 0.972".
   - Replace "exp001's config collapse was not caused by truncation" with "under the reviewed harness the config error persists at the exp001 level with truncation ≤ 0.019; exp001 is context only".
5. **Factor breakdown.**
   - Replace "clustering doesn't matter" with "no evidence that corner clustering raises the error (0.519 vs 0.564, unpaired)".
   - Prefix the ranking with "for this pipeline's raw error".
   - Add the caveat that the ranking may reflect GT dispersion relative to the pipeline's default output (see 2) until a constant-centre prior is scored per diagnostic split (IDEA-032).
6. **vs exp000.** Add the Task A compare above, or state that "every split" refers to Task B only.
7. Fill the Reviewer section with a pointer to this file.

## CLAIMS.md
- **C-004 added** (it replaces C-003 at pipeline level). The zero-shot qwen3d pipeline is far worse than static and exp000 on every val OOD split for Task B, and worse than exp000 on every val split for Task A. The gate passes. Status: **solid (pipeline-level)**.
- Swap < normal is recorded under C-004 as an observation, **not a claim**. It is measured on val_id only, so the split-consistency bar cannot be checked, and its mechanism is open.
- C-003 stays withdrawn, with a forward link to C-004.
