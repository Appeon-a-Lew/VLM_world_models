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
- **status (2026-10-24, reviewer): withdrawn. Do not cite.** Per the HUMAN NOTE in `research/STATE.md`, until `d4de761` every job ran main's `physwm` instead of the experiment snapshot, because `python -m` put cwd=MAIN first on sys.path. exp001 therefore ran main's `infer.py`: a 48-token/object cap and no stats print. It never ran the reviewed harness, which had a 128-token qwen3d cap and per-split missing/truncation logging. The numbers below describe main's harness under a cap that the exp001 pre-review predicted would truncate (review_pre §1), especially on val_ood_config. They are not results of the spec. The swap-beats-normal observation in the caveats is withdrawn with the claim. Re-test: exp002_zeroshot_qwen3d_rerun (pre-review APPROVE, `research/experiments/exp002_zeroshot_qwen3d_rerun/review_pre.md`). A new claim is written only from exp002's results. The original entry is kept below for the record.
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
