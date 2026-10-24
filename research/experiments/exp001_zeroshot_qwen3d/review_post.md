# exp001_zeroshot_qwen3d: post-review

Verdict: **AMEND**

The numbers in the draft match `results` and `compare`. Both verdicts point the right way and hold up under the new per-split and Task A CIs. However: the validity gate is unverified on every split, not only config; two descriptive sentences are factually wrong; and the (A) wording claims a model property where the evidence so far only supports a harness-level result. All amendments are to the report text. No re-run is needed.

## Numbers checked
- `results exp001_zeroshot_qwen3d`: primary 0.7554, static 0.3790, val_id Task B 0.600, Task A val_id 0.619, parse_rate_min 0.996, image_dependence_gap 0.400. These match the draft.
- `compare exp001 static`: +0.3764, CI [0.3578, 0.3964]. Per split: objects +0.274 [0.244, 0.303], forces +0.206 [0.169, 0.240], config +0.650 [0.612, 0.688]. **All three are "b better" with the CI excluding 0.** This matches the draft.
- `compare exp001 exp000_sft_baseline`: +0.4418 [0.4243, 0.4606]. Per split: objects +0.344 [0.316, 0.372], forces +0.360 [0.325, 0.392], config +0.621 [0.585, 0.658]. All three "b better".
- `compare exp001:swap exp001:normal --task A --splits val_id`: swap − normal = **−0.051, CI [−0.074, −0.028], "a better"**. In Task B (val_id) it is −0.062 [−0.090, −0.035].
- `compare exp001 exp000 --task A` on all 4 splits: exp001 is worse on every split. val_id +0.514 [0.488, 0.540], objects +0.431, forces +0.535, config +0.828. All CIs exclude 0.
- **Averaging.** The `compare --task A` value for normal val_id is 0.529, while `results.md` has all_mean 0.619. `compare` is presumably episode-weighted and results.md object-weighted (unverified; I can't read exec.py's `compare` internals through agent.sh). The spec criterion names results.md all_mean (−0.071). Both versions put the difference below −0.02, so the verdict does not depend on which average is used. The gap between 0.529 and 0.619 is informative in its own right: per-object error is higher in episodes with more objects. That fits object omission (scored at 1.0 m) in multi-object scenes, and it fits harder multi-object scenes equally well.

## Ruling on issue 1: validity gate / missing_rate
- I confirmed the stats lines are absent. `agent.sh log exp001 400` has 19 lines, and the only non-blank ones are the three command lines. The exec.py:356 tqdm-filter explanation is plausible but unverified. The outbox request is the right action.
- **Treating val_ood_config as a harness/format result: appropriate, and the evidence is stronger than the draft says.** A Task A target_median of exactly 1.000 means at least half of target objects scored the 1.0 m penalty, or else an implausible pile-up of real errors at exactly 1.000. A Task A all_mean of 0.966 with real errors around 0.5 m (the level on the other splits) implies about 90% of slots missing. parse_rate 1.000 doesn't contradict this, since one parsed object is enough to count as parsed (parse.py:63). Object names are unique per scene (`scene.py:111`, colours drawn without replacement), so dictionary collisions in `by_name` are ruled out. With a 688-token cap for 5 objects, truncation is not the obvious cause either. The more likely causes are the model listing fewer objects or labels that don't match. Still unknown.
- **The other splits: the gate is unverified, so it has *not* passed.** The spec makes the gate a precondition ("otherwise the run is reported as a harness/format result"), so every verdict stays **provisional** until `joblog_rates.txt` exists. The two verdicts depend on missing_rate in different ways:
  - **(B) does not depend on it.** Missing objects can only raise exp001's error, and (B) says exp001 is worse than static. Whatever the missing rate, (B) holds at the pipeline level (model + prompt + parser). What stays open is whether it is a property of the *model*. (B) is descriptive anyway.
  - **(A) does depend on it.** A *differential* missing rate between normal and swap would by itself shift the swap − normal difference. The rejection therefore stands as a harness-level measurement; it can only be read as a model-level result once the per-mode rates are known.
  - Note: the logged missing_rate pools Task A and Task B answers within a split (infer.py `work` loops over both tasks). The human-supplied lines will settle the gate. They will not give a Task A-only missing rate for normal vs swap, which IDEA-028 provides.

## Ruling on issue 2: swap better than normal on Task A
(A) is **rejected**, and more strongly than "below the 0.02 m threshold": the CI lies entirely on the wrong side (−0.074 to −0.028). Normal is not merely no better than swap; it is worse. That is itself evidence that the image *changes* the output, so the model's output depends on the image. The image just doesn't make the output *correct*. Alternative explanations:
1. **Label matching / format (harness).** With the correct image, the model may name what it sees (e.g. a different colour word, "cylinder", "block") instead of copying the prompt names. `_norm` maps only sphere→ball and cube→box, so those slots become `None` = 1.0 m. With a mismatched image it may just echo the prompt list, so more labels match. The result is lower missing_rate under swap, and a lower error score that reflects better label matching, not better localisation.
2. **Frame or axis convention mismatch (systematic, image-driven error).** Suppose the model localises from the image but one axis is flipped or offset relative to our camera frame. Then normal predictions are *anti*-correlated with GT deviations along that axis, while swap predictions are uncorrelated. Under L2, an anti-correlated estimator loses to an uncorrelated one. This is the one simple mechanism that produces "image helps negatively". IDEA-024 already lists the sign-flip test; this result raises its priority.
3. **Variance shrinkage.** When the named objects are not visible (swap), the model may fall back to stereotyped "table centre" coordinates with low variance. With the true image, it produces scene-specific but weakly correlated and high-variance guesses. When the correlation is low, a low-variance prior beats a noisy estimator. This predicts that a constant-prior predictor also beats normal.
- **Cheapest discriminator: IDEA-028, a read-out of existing pred files with no generation.**
  - Per-mode, Task-A-only missing rate and unmatched-label rate: (1) predicts swap < normal.
  - Error on objects matched in *both* modes: if normal ≤ swap there, (1) explains the result.
  - Per-axis Pearson r(pred, GT) and std(pred) in each mode: (2) predicts r < 0 on some axis under normal; (3) predicts r ≈ 0+ and std(normal) > std(swap).
  - A constant-centre prior as a derived pred file.

## Ruling on issue 3: image_dependence_gap
Agreed, it's an artefact. none has parse_rate 0.000 on both tasks, so 1.000 − 0.600 = 0.400 exactly. The report must state that the metric is **uninterpretable for this run** and must not use it anywhere as evidence of image use. It also must not be compared with exp000's 0.331. For zero-shot runs, `none` measures whether the model follows the format without an image, not whether it relies on the image. swap is the right reference, and its CI is above.

## Parse rates
Task B: normal 1.000 on every split, swap 0.994 on val_id, none 0.000. Task A: objects 0.996, everything else 1.000. The 0.6% swap parse deficit works *against* swap, since unparsed answers score 1.0 m, so it can't produce swap's advantage. For this run, parse_rate says very little: it is satisfied by a single parsed object. Missing slots are where the risk lies (issue 1).

## Per-split
(B) is consistent across the splits. All three per-split CIs are above 0, and the smallest is forces at +0.206 [0.169, 0.240]. val_ood_config dominates the aggregate (+0.650) and is a harness result, but without config the other two splits alone still exclude 0 by a wide margin. The same holds for exp001 vs exp000.

## Factual errors in the draft
- "exp001 is also worse than const_velocity on every split except val_id, where the two are close": **wrong.**
  - On val_ood_objects exp001 is *better* than const_velocity by point estimate (0.505 vs 0.587).
  - On val_id it is worse (0.600 vs 0.567).
  - On forces (0.870 vs 0.644) and config (0.891 vs 0.570) it is worse.
  - No CI exists for any of these comparisons.
- "Task A error alone (0.53–0.63 m on the non-config splits) exceeds static's whole Task B error": **wrong on forces.** There static's Task B target mean is 0.664, above exp001's Task A all_mean (0.634) and target mean (0.552). The claim holds for val_id and objects only.

## Required amendments (numbered)
1. **Mark both verdicts provisional pending `joblog_rates.txt`, and scope them.** (B): "supported at pipeline level (model + qwen3d prompt + parser); not yet attributable to the model". (A): "rejected at harness level; swap − normal = −0.071 m (results.md all_mean; compare episode-weighted −0.051, CI [−0.074, −0.028])". Replace "The scene-specific image content does not help zero-shot Qwen3-VL-2B … Its answers don't reflect the actual scene better than answers for a random other scene" with wording that describes the scored output and lists the three alternatives above. Don't claim a model property until missing_rate per mode is known.
2. **Add the Task A CI** (`compare exp001:swap exp001:normal --task A --splits val_id`) and the Task B swap-vs-normal CI. Remove the claim that no Task A CI exists. Note the object-weighted vs episode-weighted difference (0.619 vs 0.529) and what it implies.
3. **Fix the const_velocity sentence**: exp001 beats const_velocity on val_ood_objects by point estimate and is worse on val_id, forces and config, with no CI for any of these.
4. **Fix the "Task A error exceeds static's whole Task B error" sentence**: it holds for val_id and objects, not forces. The conclusion that (B) carries no dynamics information still follows from the oracle-perception argument.
5. **Strengthen the val_ood_config paragraph** with the arithmetic above (median exactly 1.000 implies ≥50% of target slots at the penalty; all_mean 0.966 implies roughly 90% of slots missing if real errors are ~0.5 m). State that name collisions are ruled out and that the cause (omission vs label mismatch vs truncation) is unknown.
6. **image_dependence_gap**: state that it is uninterpretable for this run and must not be compared with exp000's 0.331.
7. **Add the exp001 vs exp000 Task A per-split CIs** (all exclude 0). Keep the existing caveat that two things differ between exp000 and exp001 (training and answer style).
8. When `joblog_rates.txt` arrives, add the per-split parse/missing/truncation table. Label the truncation column "upper bound: non-`]`-terminated", as pre-review round 2 required. Then re-apply the gate: splits with missing_rate > 5% are harness results.

## Backlog / claims
- BACKLOG: IDEA-028 (exp001 pred read-out: per-mode missing/label rates, matched-only error, per-axis correlation, constant prior) and IDEA-029 (op_log filter fix, needs-human). IDEA-024 gets a priority note.
- CLAIMS: added C-003 (tentative): the zero-shot qwen3d pipeline is worse than static and than exp000 on every val OOD split, with all CIs excluding 0. The swap < normal result is single-split and its cause is unresolved, so it is recorded only as a caveat, not as a claim.
