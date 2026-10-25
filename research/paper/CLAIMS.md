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
- **update (reviewer, 2026-10-24, review of exp003_exp000_diag_eval)**:
  - The CI now exists. `compare exp003_exp000_diag_eval:none exp003_exp000_diag_eval:normal --splits val_id` gives a gap of +0.331 [0.298, 0.363]. exp003 is bit-identical to exp000 (C-005).
  - On the last caveat: C-007 shows that the Task B answer beats the model's own-perception static baseline on val_id by 0.113 [0.097, 0.131]. So the Task B predictions carry information beyond t=0 localisation.
  - The none/blank/swap drop itself still mixes perception and dynamics. The status is unchanged.
- **update (reviewer, 2026-10-25, review of exp005_exp000_repro)**: a second, independently trained adapter (same recipe, new box) replicates the qualitative result on val_id Task B target: normal 0.157, none 0.500, blank 0.527, swap 0.406, static 0.265, gap 0.3425. All three ablations are worse than static by at least 0.14. none and swap are within 0.015 of exp000. blank is +0.055 off exp000, and blank > none in exp005 (+0.027 [+0.011, +0.043]) but blank < none in exp000, so do not cite blank-vs-none. Status still tentative: both runs use seed 0 and there is no seed variance estimate.

## C-002 SFT beats static on the aggregate val-OOD Task B target mean, but not on every split
- **claim**: Plain LoRA SFT lowers the mean Task B target error averaged over the three val OOD splits relative to static (which copies GT t=0 positions). The gain does not hold on val_ood_config.
- **evidence**: exp000_sft_baseline vs static: 0.3136 vs 0.3790, diff −0.0654, 95% CI [−0.0754, −0.0557] (paired bootstrap, 500 episodes/split). Per-split diffs (no CI): objects −0.071, forces −0.154, config +0.028. forces contributes 78% of the aggregate gain. On forces the target median is *worse* than static (0.089 vs 0.050). target_skill: val_id 0.416, objects 0.305, forces 0.232, config −0.118.
- **status**: tentative. Fails the split-consistency bar, so cite only as "aggregate", never as "SFT generalises OOD".
- **caveats**: single seed; static has oracle perception, so the comparison mixes perception and dynamics; per-split CIs are missing; val_ood_forces is bimodal (weak / strong pushes), so its mean gain may come from a training-displacement prior rather than force-conditional dynamics (see review_post of exp000, IDEA-008); `missing_rate` is not yet reported.
- **update (reviewer, 2026-10-24, review of exp003_exp000_diag_eval)**: per-split CIs are now available (`compare exp000_sft_baseline static`, pinned per_split_ci95):
  - objects −0.071 [−0.080, −0.062];
  - forces −0.154 [−0.178, −0.130];
  - config +0.028 [+0.013, +0.042] ("b better").

  **Replication (exp005_exp000_repro, 2026-10-25)**: retraining on the new box gives the same picture. Aggregate -0.0629 [-0.0725, -0.0535]; objects -0.071 [-0.080, -0.062]; forces -0.150 [-0.172, -0.126]; config +0.031 [+0.017, +0.045]. See C-009.

  The config failure is therefore CI-backed (single seed). missing_rate is 0.000 on every split for Task A and Task B (exp003 job.log, bit-identical to exp000 per C-005). The "per-split CIs are missing" and "missing_rate not yet reported" caveats are resolved. The status stays tentative: single seed, and the claim still fails the split-consistency bar by construction.

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

## C-005 (methods) Greedy inference with a fixed adapter is bit-reproducible under the snapshot executor
- **claim**: Re-running the exp000 LoRA adapter (center style, batch 32, greedy decoding) reproduces every exp000 prediction exactly.
- **evidence**: exp003_exp000_diag_eval (commit 6cec893) vs exp000_sft_baseline:
  - `compare` diff 0.0 with CI [0, 0] on val_id, val_ood_objects, val_ood_forces and val_ood_config, for Task B and for `--task A`;
  - `exp003:none` vs `exp000:none` on val_id: 0.0;
  - primary 0.31356803 for both; image_dependence_gap 0.33129 for both.
- **status**: solid (for inference).
- **caveats**:
  - One repeat, on the same box and software stack.
  - exp000 ran under the pre-d4de761 executor, whose infer path was identical to the snapshot's (exp003 review_pre §1).
  - Says nothing about training reproducibility or seed variance of SFT, which is untested for every claim here.
  - A torch/transformers/CUDA change would need a fresh check.

## C-006 No single factor of val_ood_config reproduces exp000's config failure
- **claim**: Changing only camera elevation, only object count (4–5), or only the corner-cluster layout from the training config leaves the exp000 adapter *better* than static on Task B target. Only the full val_ood_config combination is worse than static. Camera elevation is not shown to be the dominant driver.
- **evidence**: exp003_exp000_diag_eval, C1 = `compare exp003_exp000_diag_eval static --splits val_id val_ood_config val_diag_*` (paired bootstrap; 500 / 300 episodes).

  | split | D (exp000 − static) | CI |
  |---|---|---|
  | val_diag_elev | −0.053 | [−0.075, −0.031] |
  | val_diag_count | −0.072 | [−0.083, −0.063] |
  | val_diag_corner | −0.055 | [−0.075, −0.038] |
  | val_ood_config | +0.028 | [+0.013, +0.044] |
  | val_id | −0.110 | [−0.129, −0.095] |

  Share of the config shift R_f = (D_f − D_id)/(D_cfg − D_id), with conservative numerator-only intervals:
  - elev 0.417 [0.144, 0.705];
  - count 0.273 [0.087, 0.471];
  - corner 0.396 [0.148, 0.658].

  Every R_f upper bound is < 1. Pre-registered hypothesis "elevation drives it" (R_elev ≥ 0.5): FALSIFIED, point-estimate only.
- **status**: solid at the adapter level (exp000, seed 0). Every diag CI lies below 0, the config CI lies above 0, and the parse and missing rates are 1.000 / 0.000 everywhere.
- **caveats**:
  - Which combination drives the failure is unresolved. "Diffuse/additive" (outcome 5, Σ R = 1.086) vs "interaction" (outcome 4) is point-estimate only: the conservative Σ interval is [0.38, 1.83].
  - Elev or corner could each still account for ≥ 0.5 of the shift.
  - The cross-split comparison is unpaired: the diag splits use different episodes from val_id, and n = 300 vs 500.
  - The corner split changes three config fields (min_gap, region, region_center).
  - Single adapter and seed; not a statement about LoRA-SFT in general.

## C-007 exp000's Task B predictions beat its own-perception static baseline on every val split, by much less on val_ood_config
- **claim**: Replacing exp000's Task B answer with its own Task A answer, i.e. "no motion, from where the model thinks objects are", makes Task B target error worse on every val split. The model's predicted motion adds value beyond its own localisation, including on config. On config the margin is about a quarter of val_id's.
- **evidence**: G = `compare exp003_exp000_diag_eval exp003_exp000_diag_eval:ownstatic --splits …` (Task B minus own-static; paired):

  | split | G | CI |
  |---|---|---|
  | val_id | −0.113 | [−0.131, −0.097] |
  | val_ood_objects | −0.074 | [−0.083, −0.065] |
  | val_ood_forces | −0.160 | [−0.184, −0.136] |
  | val_ood_config | −0.028 | [−0.040, −0.015] |
  | val_diag_elev | −0.081 | [−0.102, −0.061] |
  | val_diag_count | −0.074 | [−0.084, −0.065] |
  | val_diag_corner | −0.075 | [−0.092, −0.059] |

  Normalised by true target motion, K = target_skill(:normal) − target_skill(:ownstatic) = −G / static target_mean (results.md): val_id 0.424, elev 0.332, count 0.346, corner 0.237, config 0.116.
- **status**: solid for the sign (every CI excludes 0, 7/7 splits). Tentative for the size of the config reduction: the raw G CIs do not overlap (config [−0.040, −0.015] vs val_id [−0.131, −0.097]), but there is no CI on K.
- **caveats**:
  - Own-static uses a *separate* generation (the Task A answer) as the t=0 estimate. Task B unmoved-object error ≈ own-static unmoved error on every split (config 0.132 vs 0.134), which supports the substitution for non-targets only.
  - "Adds value beyond perception" does not mean action-conditioned dynamics. On forces it could be a training-displacement prior (IDEA-008/025/026).
  - The split of the config shift into perception (0.054) and dynamics (0.085) is point-estimate only and path-dependent (exp003 review_post Point 2), and is **not** part of this claim.
  - Single adapter and seed.

## C-008 exp000's t=0 localisation is not materially degraded by an unseen object count (4–5) when camera and layout are in-distribution
- **claim**: On val_diag_count (sim_train with only the count changed to 4–5), exp000's Task A error stays at the val_id level, and its own-perception cost on Task B is not significant.
- **evidence**: exp003_exp000_diag_eval.
  - Task A all_mean: count 0.019 vs val_id 0.016 (results.md point estimates). Perception share P_count = 0.03, against elev 0.55 and corner 0.52.
  - C3 (own-static − static): count +0.001 [−0.000, 0.003], not significant, vs val_id +0.002 [0.001, 0.003], elev +0.028 [0.020, 0.035] and corner +0.019 [0.013, 0.025].
  - Task A missing_rate is 0.000 (job.log).
- **status**: tentative.
- **caveats**:
  - Absence-of-effect claim. Task A has no cross-split CI (`compare --task A … static` is refused, and the splits are unpaired), so the bound is only the point gap of 0.003 m plus C3's upper bound of 0.003 m in Task B space.
  - Count may still interact with elevation or layout inside val_ood_config.
  - The opposite of zero-shot exp002, which ranked count worst. This is training-specific, single adapter and seed.

## C-009 (methods) The exp000 SFT recipe retrains to the same val results on a new box (one repeat, same seed)
- **claim**: Retraining the exp000 recipe unchanged (seed 0, new box nc-10-1-1, regenerated data with ±1/255 RGB noise) reproduces the exp000 headline numbers and the per-split pattern. The primary metric moved by only +0.0024.
- **evidence**: exp005_exp000_repro (commit 90b1cc3, 1.41 GPU-h) vs the exp000 report. Primary 0.3160 vs 0.3136. Task B target: objects 0.161 vs 0.161, forces 0.515 vs 0.510, config 0.272 vs 0.269, val_id 0.157 vs 0.155. Task A val_id 0.0152 vs 0.016. Gap 0.3425 vs 0.331. Parse rate 1.000. exp005 vs static: -0.0629 [-0.0725, -0.0535]; per split objects/forces better, config worse (+0.031 [+0.017, +0.045]).
- **status**: tentative
- **caveats**:
  - Single repeat with the same seed, so this does not estimate seed-to-seed variance. A different seed could move the primary by more.
  - No paired CI against exp000 (predictions lost). Agreement is on aggregates only.
  - The blank ablation did not reproduce (+0.055; the order against none flipped). Ablations built from inputs the model never saw in training are the least reproducible.
  - The box change and the render noise are confounded.
  - C-005 (inference bit-reproducibility) is separate and still holds only for a fixed adapter.
  - Retraining is not bit-exact, though the numbers are close.

## C-010 (privileged diagnostic) With GT t=0 re-anchoring, the exp005 adapter's displacement skill is lower on val_ood_config than on val_id, in every physics stratum; the stratum mix explains almost none of it
- **claim**: Replacing the model's own t=0 with GT t=0 (oracleanchor = GT_t0 + (B^ - A^); uses privileged GT, not a model result) still leaves a large loss of motion-normalised skill K from val_id (0.420) to val_ood_config (0.152). The loss is not a stratum-mix effect (Comp about 0.01; negative in absolute-diff space). Isolated pushes: K 0.482 -> 0.217. About 43% of the config shift is perception and 57% dynamics (point estimates, path-averaged).
- **evidence**: exp004_exp000_oracle_anchor, recomputed by the reviewer from `compare` (all CIs per split, vs static):
  - oracleanchor diff: val_id -0.1115 [-0.128, -0.097], config -0.0366 [-0.048, -0.024].
  - iso K (rescaled): id 0.482 [0.423, 0.539], config 0.217 [0.075, 0.341]. The two K intervals do not overlap.
  - rho = 0.451, conservative interval [0.14, 0.81]: crosses 0.5 and 0.75, so the pre-registered FALSIFIED call is point-estimate only.
  - cont K: id 0.529 (n 31), config 0.224. off K: id 0.363, config 0.106 [0.063, 0.162].
  - Decomposition: S_cfg 0.1396 = P_oa 0.0647 + D_oa 0.0750 = delta C3 0.0543 + delta G 0.0854. Shapley 0.0595 / 0.0802.
  - Diag iso K: elev 0.337, count 0.444, corner 0.444 (no CI computed).
- **status**: tentative. Direction of the K loss on the oracle path is CI-supported (config K_oa CI excludes val_id's by a wide margin); the size and the "dynamics, not perception" attribution are not.
- **caveats**:
  - The K drop on isolated pushes is mostly a denominator effect: iso target displacement is 0.137 m (id) vs 0.103 m (config), while the oracle iso absolute error is 0.071 m vs 0.081 m. In absolute terms, isolated-push error rose by about 0.010 m. Do not write "dynamics skill is lost on simple pushes" without the absolute-error CI (IDEA-051).
  - In D space the config loss splits about 0.0388 iso / 0.0369 off-table / -0.0008 contact. Roughly half is off-table episodes (edge falls), which neither IDEA-002 nor camera conditioning addresses (IDEA-052).
  - Camera-frame confound is untested: actions are camera-relative, away/toward map to elevation-dependent camera y/z, left/right do not (IDEA-050). The diag splits argue against elevation alone: elev iso K is 0.337, config 0.217, and no single factor or pair reproduces config (3-way interaction or regime change).
  - **Update from exp006_exp005_direction_split (2026-10-25, review_post; bit-exact rerun of exp005, privileged oracle path):** the pre-registered camera-frame prediction (AT rises more than LR, DiD >= +0.02) is not supported on config: DiD -0.0196 [-0.057, 0.017]. The upper bound is below +0.02, so the pre-registered magnitude is excluded. Delta_e on config is LR +0.020 [-0.008, 0.051] and AT +0.0002 [-0.019, 0.020]. Neither is resolved from 0. Using constant-e arithmetic on the iso means, roughly two thirds of the pooled iso K drop (0.48 -> 0.22) is the shrunken denominator and about one third (e +0.010 m, unresolved by CI) is error. The earlier "mostly denominator" caveat therefore stands, with "mostly" meaning about 2/3. On val_diag_elev, LR rises more than AT (DiD -0.035 [-0.071, -0.002]), but this is one split with a CI edge at -0.002, and val_diag_elev_corner has the opposite ordering (exploratory, no CI). Do not read it as evidence against a camera-frame effect: LR cross-push error lies on the elevation-dependent y/z axes (IDEA-053). Config off-table fall recall 0.351 -> 0.090 (drop 0.262 [0.141, 0.382]), but recall is also 0.06 to 0.17 on objects, count and corner, so it is not config-specific (IDEA-054).
  - **Update from exp007_exp005_peraxis (2026-10-25, review_post; bit-exact rerun, privileged oracle path):** the y/z-localisation prediction (A-yz) is not supported and the probe is underpowered, so the camera-frame idea is neither supported nor refuted. did_yz (AT minus LR yz rise) is negative with CI including 0 on all three pairs (elev -0.037 [-0.077, 0.003], elev_corner -0.004 [-0.031, 0.024], config -0.022 [-0.065, 0.016]); the CI half-widths (about 0.04 m) cannot exclude a +0.03 m effect. LR yz rise: elev +0.035 [0.003, 0.072], elev_corner -0.003, config +0.016 [-0.011, 0.049]. The LR rise on elev is in x as well (+0.027 [0.007, 0.046]), so it is not y/z-localised; it does not replicate on elev_corner. Outcome D-mixed. The pre-registered LR validity limit (0.01 m) was miscalibrated (val_id LR GT |y|+|z| is 0.030 m), so the formal "invalid probe" call is a design miss; the routing is the same without it. AT pooled signed means are uninformative (away and toward cancel). Exploratory, not a result: AT mean-abs per-axis error is roughly 0.45-0.55 of GT mean |motion| on that axis on val_id and elev (y 0.43 -> 0.55, z 0.49 -> 0.46), i.e. error scales with GT motion.
  - Oracle path mixes separate Task A and Task B generations (unmoved error 0.010 m config vs 0.006 id; gate passed at 0.05 m).
  - Single adapter and seed. n_cont = 31 on val_id.
  - Extends but does not replace C-006 and C-007. The C-007 own-path perception/dynamics split is unchanged and still point-estimate only.
