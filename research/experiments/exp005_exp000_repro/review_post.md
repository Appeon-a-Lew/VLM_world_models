# Post-review exp005_exp000_repro

**Verdict: REPRODUCED (confirmed), tentative. Single seed, new box, no paired CI against exp000.**

## Table check (report.md vs exp000 report.md and results.md/metrics.json)
- Every exp000 value in the table matches exp000's report: 0.3136, 0.161, 0.510, 0.269, 0.155, 0.016, gap 0.331.
- Every exp005 value matches results.md / metrics.json. The gap 0.3425 = none 0.500 - normal 0.157.
- Ablation values for exp000 (none 0.486, blank 0.472, swap 0.400) match exp000's report.
- The pre-registered criteria all pass, with margin. Primary +0.0024 vs a ±0.010 band, per-split at most 0.005 vs 0.015, gap +0.0115 vs 0.03.
- Smaller points:
  - The report says "Static val OOD unchanged" (0.3790). This is true, and expected since static uses GT.
  - Primary vs static is -0.0629 [-0.0725, -0.0535] (exp000 -0.0654 [-0.0754, -0.0557]).
  - Per-split CIs vs static (new, from `compare exp005 static`): objects -0.071 [-0.080, -0.062]; forces -0.150 [-0.172, -0.126]; config +0.031 [+0.017, +0.045]. The ordering and signs are identical to exp000's (config still worse than static, CI-backed).
  - The caveat "+0.0024 is a single-run noise estimate" is right. It also means the ±0.010 band was not shown to be wider than the seed noise. Do not describe it as "within noise" beyond "one extra draw landed close".

## Does the blank +0.055 deserve more than a note? Yes, a small one, but it does not change the verdict.
Evidence from the table and a fresh `compare`:
- It is not a gated criterion, so the verdict stands. But "most likely noise" is currently unsupported.
- Within exp005, `compare exp005:blank exp005:none --splits val_id` gives +0.027 [+0.011, +0.043]. blank is significantly worse than none.
- In exp000 the order was reversed (blank 0.472 < none 0.486, -0.014). The sign flipped between training runs, with the paired CI in exp005 excluding 0. A pure image-free noise story does not predict that blank and none differ much.
- Blank Task B: mean 0.527, median 0.383. For none the median is 0.387. The median is the same, so the +0.055 sits in the tail, probably a few episodes with large errors. Task A blank barely moved (0.233 vs 0.240; none 0.228 vs 0.223). So the shift is in Task B dynamics output under an all-constant image, not perception.
- Interpretation: blank is an input never seen in training. The adapter's response to it is an unconstrained extrapolation and depends on training details (init, data order, render noise), so it is the least reproducible ablation. That is consistent with what C-001 already says (blank and none overstate reliance; swap is the clean control).
- Consequences:
  - C-001 is safe because all three ablations are worse than static (0.265) in both runs, with margins above 0.14. swap (0.400 vs 0.406) and none (0.486 vs 0.500) replicate.
  - Do not cite blank-vs-none comparisons, or "blank ~ none", anywhere. Do not use blank as a precise baseline for later exps (exp004 oracleblank, IDEA-037/041).
  - Cost to settle it is zero GPU: inspect per-episode blank errors in exp005 (tail count, top-5 episodes). A second seed would give the true seed noise floor for all numbers.

## Alternative explanations for REPRODUCED
1. Same seed, same recipe: close agreement may reflect a deterministic training path (seed 0, same data generator seed), with the ±1/255 render noise and the box having small effects. Then the 0.0024 says little about seed-to-seed variance (a different seed could move the primary by more). Cheapest discriminator: one repeat with seed 1 (about 1.4 GPU-h), or a repeat of exp005 on the same box to see bit-exactness of training.
2. Agreement of the aggregate hides compensating per-split differences (config +0.031 vs +0.028, forces -0.150 vs -0.154). Per-split values are all within 0.005, so this is unlikely. Cheapest check: per-episode correlation, unavailable because exp000's predictions are gone. So the stand-in claim rests on aggregates only.

## Verdict on use as exp000 stand-in
Fine, with two cautions: read exp004 against exp005's numbers, not exp000's, and treat blank-based numbers as non-reproducible at the 0.05 level.

## Claims
CLAIMS.md updated: C-009 added (training reproduction), C-001 and C-002 given updates, C-005 caveat on training reproducibility partly addressed.
