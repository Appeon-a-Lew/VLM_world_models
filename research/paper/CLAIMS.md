# Evidence-backed claims (for the eventual paper)

Maintained by the reviewer. Each claim needs:
- **claim**
- **evidence**: experiment ids, numbers, paired-bootstrap CI
- **status**: tentative / solid / contested
- **caveats**

All numbers are on validation splits unless marked FINAL (final test, human-approved).

## C-001 LoRA-SFT Qwen3-VL-2B's Task B predictions depend on image content
- **claim**: After SFT on Task A + B, the model's future-position predictions rely on the image. Removing it or replacing it makes the predictions worse than the trivial static baseline.
- **evidence**: exp000_sft_baseline, val_id, Task B target mean: normal 0.155 m; swap (another scene's image, same object count) 0.400; blank 0.472; no image 0.486; static 0.265. image_dependence_gap = 0.331. Task A shows the same pattern (0.016 vs 0.223–0.249). Parse rate 1.000. No paired CI for the ablation diffs: `compare` covers only the primary metric.
- **status**: tentative
- **caveats**: single seed; val_id only; no CI. Blank and none are inputs the model never saw in training, so they overstate reliance; swap is the cleanest control. Shows that the model uses the image, not that it uses it for *physics*: the Task B ablation drop could be mostly the loss of perception (where objects are now).

## C-002 SFT beats static on the aggregate val-OOD Task B target mean, but not on every split
- **claim**: Plain LoRA SFT lowers the mean Task B target error averaged over the three val OOD splits relative to static (which copies GT t=0 positions). The gain does not hold on val_ood_config.
- **evidence**: exp000_sft_baseline vs static: 0.3136 vs 0.3790, diff −0.0654, 95% CI [−0.0754, −0.0557] (paired bootstrap, 500 episodes/split). Per-split diffs (no CI): objects −0.071, forces −0.154, config +0.028. forces contributes 78% of the aggregate gain. On forces the target median is *worse* than static (0.089 vs 0.050). target_skill: val_id 0.416, objects 0.305, forces 0.232, config −0.118.
- **status**: tentative. Fails the split-consistency bar, so cite only as "aggregate", never as "SFT generalises OOD".
- **caveats**: single seed; static has oracle perception, so the comparison mixes perception and dynamics; per-split CIs are missing; val_ood_forces is bimodal (weak / strong pushes), so its mean gain may come from a training-displacement prior rather than force-conditional dynamics (see review_post of exp000, IDEA-008); `missing_rate` is not yet reported.

## C-003 [WITHDRAWN: invalid as specified] Zero-shot Qwen3-VL-2B with the native bbox_3d (qwen3d) prompt is far worse than static and than LoRA-SFT on every val OOD split
- **status (2026-10-24, reviewer): withdrawn. Do not cite.** Per the HUMAN NOTE in `research/STATE.md`, until `d4de761` every job ran main's `physwm` instead of the experiment snapshot, because `python -m` put cwd=MAIN first on sys.path. exp001 therefore ran main's `infer.py`: a 48-token/object cap and no stats print. It never ran the reviewed harness, which had a 128-token qwen3d cap and per-split missing/truncation logging. The numbers below describe main's harness under a cap that the exp001 pre-review predicted would truncate (review_pre §1), especially on val_ood_config. They are not results of the spec. The swap-beats-normal observation in the caveats is withdrawn with the claim. Re-test: exp002_zeroshot_qwen3d_rerun (pre-review APPROVE, `research/experiments/exp002_zeroshot_qwen3d_rerun/review_pre.md`). A new claim is written only from exp002's results. The original entry is kept below for the record. **Superseded by C-004 (2026-10-24, review of exp002_zeroshot_qwen3d_rerun).**
- **claim**: Without training, the qwen3d pipeline (Qwen3-VL-2B base + our qwen3d prompt + our parser) has a much higher Task B target error than static on each of the three val OOD splits. It also has a much higher error than the exp000 SFT model on both Task B and Task A.
- **evidence**: exp001_zeroshot_qwen3d.
  - Task B target, val-OOD mean: 0.755 vs static 0.379, diff +0.376, CI [0.358, 0.396]. Per split vs static: objects +0.274 [0.244, 0.303], forces +0.206 [0.169, 0.240], config +0.650 [0.612, 0.688].
  - vs exp000_sft_baseline: +0.442 [0.424, 0.461]; objects +0.344, forces +0.360, config +0.621, all per-split CIs > 0.
  - Task A vs exp000 (compare --task A): val_id +0.514 [0.488, 0.540], and every OOD split's CI > 0.
  - Parse rate ≥ 0.996.
- **status** (original): tentative. Superseded by "withdrawn" above.
- **caveats**:
  - **Pipeline-level only.** The object-missing rate (validity gate ≤ 5%) is unverified on every split, because the stats lines were lost from the job log (outbox/exp001_joblog_missing_rate.md). val_ood_config is almost certainly dominated by missing objects (Task A target_median = 1.000 = penalty). The claim must not be stated as "the model cannot localise / predict" until missing_rate is known.
  - Static has oracle perception, so "worse than static" says nothing about dynamics knowledge.
  - The comparison with exp000 differs in two factors (training and answer style).
  - Single prompt: the verbatim native template (IDEA-023) is untested.
  - Single seed (greedy decoding).
  - Related unresolved observation (val_id only, not a claim): with the image of another scene (swap), the score is *better* than with the correct image. Task A −0.051 [−0.074, −0.028]; Task B −0.062 [−0.090, −0.035]. The cause is open: label matching vs frame convention vs variance shrinkage (IDEA-028, IDEA-024).

## C-004 The zero-shot qwen3d pipeline (Qwen3-VL-2B base + our qwen3d prompt + our parser) is far worse than static and than LoRA-SFT on every val OOD split, and the failure is not a format artefact
- **claim**: Without training, the qwen3d pipeline has a much higher Task B target error than static on each of the three val OOD splits, and than exp000_sft_baseline on each of them. Its Task A error is much higher than exp000's on every val split. Object-missing rates are ≤ 0.017 on every validation split, so the errors are on parsed predictions, not penalties. This replaces the withdrawn C-003.
- **evidence**: exp002_zeroshot_qwen3d_rerun (commit 9e70064, snapshot harness confirmed by 9/9 stats lines in job.log). Paired bootstrap with 500 episodes per split:
  - Task B target, val-OOD mean: 0.735 vs static 0.379, diff +0.356, CI [0.338, 0.375].
  - Task B per split vs static: objects +0.217 [0.192, 0.242], forces +0.204 [0.171, 0.236], config +0.647 [0.609, 0.685].
  - Task B vs exp000_sft_baseline: +0.421 [0.404, 0.439]. Per split: objects +0.288 [0.263, 0.312], forces +0.358 [0.326, 0.389], config +0.619 [0.582, 0.655].
  - Task A vs exp000 (`compare --task A`, episode-weighted, val_id + 3 OOD): +0.547 [0.534, 0.560]. Per split: val_id +0.478 [0.452, 0.502], objects +0.374 [0.355, 0.395], forces +0.502 [0.477, 0.526], config +0.833 [0.803, 0.864].
  - Validity (pooled Task A + B, from job.log): missing rates are val_id 0.000, objects 0.001, forces 0.002, config 0.017. The truncation upper bound is ≤ 0.019. Parse rate is ≥ 0.996.
  - The diag splits are also far worse than static: elev +0.384 [0.342, 0.426], count +0.633 [0.584, 0.683], corner +0.216 [0.180, 0.251].
- **status**: solid at the pipeline level. Every per-split CI excludes 0 by a wide margin, the validity gate passes, and the direction and size match exp001. exp001 used a different (invalid) harness and is context only, not evidence.
- **caveats**:
  - **Pipeline-level only.** Do not write "Qwen3-VL-2B cannot localise / predict zero-shot". Only one prompt was tested: our qwen3d prompt adds text K, a FRAME sentence and multi-object queries, and the verbatim native template (IDEA-023) is untested. The observation below suggests that the pipeline's frame or read-out may be at fault rather than the model's perception.
  - Static has oracle perception, so "worse than static" says nothing about dynamics knowledge.
  - The comparison with exp000 differs in two factors: training and answer style (qwen3d vs center).
  - Single seed (greedy decoding).
  - Related observation (not a claim: val_id only, mechanism open). It reproduces exp001 under the valid harness. With another scene's image (swap, a same-count partner), the error is *lower* than with the correct image.
    - Task A: −0.063 [−0.089, −0.036] episode-weighted, or −0.089 object-pooled.
    - Task B target: −0.059 [−0.088, −0.028].
    - Missing is 0.009 for swap and 0.000 for normal. That penalises swap, so missingness can't explain the advantage.
    - This requires the correct-image predictions to be worse than an image-independent predictor. The candidates are an anti-informative frame transform, or hedging under swap combined with miscalibrated commitments under normal.
    - To be resolved by IDEA-024, IDEA-028 and IDEA-032 on the exp002 preds. Promote it to a claim only if the read-out identifies the mechanism, or if swap < normal reproduces on a second split.
