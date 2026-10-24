# exp000_sft_baseline: post-review

Verdict: **AMEND**. The numbers in the draft are transcribed correctly. Two of the interpretations (Obs. 2 and Obs. 3) go further than the evidence, and one diagnostic the result depends on (`missing_rate`) is not reported.

## What I checked myself
- `scripts/agent.sh results` and `scripts/agent.sh compare exp000_sft_baseline static`: every number in the report tables matches.
- The aggregate is the unweighted mean of the three splits: (−0.0707 − 0.1539 + 0.0284) / 3 = −0.0654. The CI [−0.0754, −0.0557] excludes 0 and the improvement threshold (0.005).
- `baseline` = static (spec). For exp000 there is no SFT baseline to compare against, so the `compare <exp> <baseline>` and `compare <exp> static` runs are the same comparison.
- Parse rate is 1.000 everywhere. **Caveat:** `parse_rate` is True as soon as one object is parsed (`physwm/vlm/parse.py:63`). Objects that are missing get `MISSING_PENALTY` = 1.0 m (`physwm/eval/metrics.py:27,49`). `summarize` computes `missing_rate`, but `results.md` does not show it, so "parse 1.000" does not rule out dropped objects.
- Image dependence (val_id): normal 0.155; swap 0.400, blank 0.472, none 0.486. All three ablations are worse than static (0.265). The most informative control is swap: it is a real scene image with the same object count, and it still costs +0.245 m. No red flag here. The model uses the image heavily, and this is the first experiment, so there is no earlier gap to compare against.

## Is the verdict supported?
1. **The aggregate gain is mostly one split.** val_ood_forces contributes −0.051 of the −0.065 aggregate (78%). Without forces, the objects+config mean diff is −0.021. compare gives no per-split CI, so the split-level claims in the draft (including "worse on config, +0.028") are point estimates. The aggregate CI excludes 0, but the result is **not consistent across splits**, so it does not meet the bar for a solid claim.
2. **"Hypothesis partly supported" is mislabelled.** The hypothesis says "learns ID, generalises worse OOD". Measured as target_skill against static, skill drops on *every* OOD split: val_id 0.416 → objects 0.305, forces 0.232, config −0.118. In point-estimate terms the hypothesis is therefore **supported** on both halves. The draft's "partly" mixes up "beats static OOD" with "generalises as well as ID", and the hypothesis never claimed the first. Note also that the hypothesis had no pre-registered numeric threshold, so it was only weakly falsifiable.
3. **Static is a privileged baseline.** It copies the GT t=0 positions (`physwm/eval/baselines.py:26`), so it has perfect perception. The SFT model has to perceive. Every "worse than static" statement therefore conflates perception error with dynamics error. The report should say this explicitly.

## (a) val_ood_config: "perception-driven" is not established
The draft says that "perception alone explains an error of that size" and that "dynamics are likely not the problem". The numbers do not show that:
- Task A *target* error is 0.133, but Task B target error is 0.269. Perception therefore accounts for at most about half of the Task B target error. If the val_id dynamics error (~0.14 beyond perception) carried over unchanged, combining it with 0.133 would give anything from 0.19 (quadrature) to 0.27 (linear). So the data are *consistent with* "unchanged dynamics + worse perception", but they do not show it, and they also fit a world where dynamics got worse.
- The one clean sign of a perception effect: SFT Task B `unmoved_mean` = 0.132 ≈ Task A error 0.136. The non-moving objects inherit the perception error.
- **ood_config is a compound shift**, not a camera shift (`configs/sim_ood_config.yaml`). It combines four things: 4–5 objects (train: 1–3), a tight cluster (min_gap 0.01 vs 0.03) near a table corner, a low camera elevation of 12–25° (train: 30–60°), and the same lens/distance. The draft's "camera/scene-configuration shift breaks the 2D→3D mapping" picks one factor without evidence.

Alternative explanations:
1. **Object-count / omission artefact.** The model never saw more than 3 objects. If it lists only 3 of 4–5, each omitted object scores 1.0 m, and parse_rate still reads 1.000. Task A mean − median = 0.024 leaves room for a few % omissions. This would be an output-format failure, not perception.
2. **Camera-frame miscalibration at low elevation.** A systematic (not noisy) bias in depth/y from the unseen elevation range, i.e. the model memorised the training camera prior. Task A median 0.112 ≈ mean 0.136 points to a consistent offset rather than outliers.
3. **Dynamics are novel too.** Dense clusters at a corner mean more multi-body collisions and more objects pushed off the table edge. Under this explanation the extra ~0.14 Task B error beyond perception is real dynamics error.

Cheapest discriminators. None of these needs training, and all can run on existing predictions:
- report `missing_rate` per split/task. This settles (1) directly.
- **perception-matched static** ("own-static"): score the model's own Task A answer as its Task B prediction. If SFT-B ≈ own-static or worse on config, the dynamics part fails there too (supports 3). If SFT-B is below own-static by a val_id-sized margin, dynamics survive and perception is the bottleneck (supports the draft).
- break the config results down by object count (4 vs 5) and by elevation bin.
- if those are ambiguous: single-factor val splits (elevation-only, count-only, cluster-only).

## (b) val_ood_forces: mean better, median worse
The draft's mechanism is wrong. It attributes the worse median to "added error on the many objects that barely move (unmoved_mean 0.019 vs 0.001)". `unmoved_mean` averages over *all* objects with GT displacement ≤ 1 cm, and these are mostly non-targets. 0.019 is the perception floor (Task A 0.016). That cannot explain a 0.04 m gap in the *target* median (0.089 vs 0.050).

Key fact the draft does not use: the forces split is **bimodal by construction**. Speeds are drawn from [0.08, 0.25] or [2.0, 2.8] m/s (train: [0.3, 1.6]), and the impulse is given numerically in the prompt (`physwm/sim/push.py:157`). The static target median of 0.050 sits in the weak mode, so the median comparison is effectively a weak-push comparison, and the mean comparison is effectively a strong-push comparison.

Alternative explanations:
1. **Regression to the training displacement prior.** The model ignores out-of-range impulse numbers and predicts training-typical motion. Weak pushes are then over-predicted, which worsens the median, and strong pushes are under-predicted. Any forward motion still beats static's ~1 m error there (moved_mean 0.765 vs 0.997), which improves the mean. Under this explanation the mean gain is not force-conditional dynamics.
2. **Partial extrapolation.** The model does scale displacement with impulse, but compresses it. The two can be told apart by the slope of predicted |Δ| vs impulse within each mode.
3. **Off-table events dominate the mean.** At 2.0–2.8 m/s many targets slide or roll off the 0.5 m half-width table (z → −0.75). The mean gain could come almost entirely from predicting "near the edge" for these episodes.

Cheapest discriminators (no training): split forces by speed mode (weak/strong) and by off-table vs on-table; regress predicted |Δ| on impulse; add a trivial **"training-prior displacement" baseline** (own Task A position + mean training displacement along the stated direction). If that baseline matches SFT on forces, the forces gain is a prior, not dynamics.

## Required amendments to report.md
1. Obs. 2: replace "perception alone explains / dynamics likely not the problem" with "Task A error rises 8× (0.016 → 0.136) and accounts for at most ~half of the Task B target error (0.269); ood_config is a compound shift (count, clustering, camera); cause not yet isolated." Report `missing_rate`.
2. Obs. 3: remove the unmoved_mean mechanism. State that the split is bimodal, that the median reflects weak pushes and the mean reflects strong pushes, and that the leading hypothesis is regression to the training prior (untested).
3. Obs. 1: add that 78% of the aggregate gain comes from val_ood_forces, and that per-split diffs have no CIs.
4. Note that static uses GT t=0 positions (oracle perception).
5. Verdict: "supported in point estimates (skill drops on all three OOD splits); single seed; hypothesis had no numeric criterion". Do not call it "partly supported", and do not say "generalisation holds for objects and forces".

## Claims
`research/paper/CLAIMS.md`: added C-001 (image dependence, tentative) and C-002 (aggregate OOD gain vs static, tentative, explicitly split-inconsistent). Neither is solid, because of the single seed, the missing per-split / ablation CIs, and the failure of the split-consistency bar for C-002.

## Backlog
Added IDEA-006 to IDEA-009 (source: review of exp000_sft_baseline). IDEA-002 (displacement output) is directly relevant to (a), because it separates perception error from dynamics error by design.
