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
