# exp003_exp000_diag_eval: report

Status: done (0.214 GPU h, commit 6cec893). This is an eval-only re-run of the exp000 adapter on the three single-factor diagnostic splits. It also writes an own-static read-out (`pred/ownstatic.jsonl`), as declared instrumentation.

Every raw number comes from `scripts/agent.sh results`, `log … 400` or `compare` (the calls are listed in each section). The derived quantities (S_f, R_f, P_f, G_s, intervals) are arithmetic on those outputs, using the spec's formulas.

## Hypothesis (spec)
Camera elevation drives exp000's val_ood_config failure. On Task B target error relative to static, the hypothesis needs three things:
- R_elev ≥ 0.5 and separable from val_id;
- R_elev is the largest R_f;
- R_corner ≤ 0.25.

It is falsified if R_elev < 0.5, or if another R_f exceeds R_elev. Verdict order: FALSIFIED → SUPPORTED → PARTIALLY SUPPORTED → INCONCLUSIVE.

## Change
None to the model: the same exp000 adapter, `center` style, seed 0. The instrumentation is the own-static derived pred plus per-split, per-task stats lines.

## Snapshot and validity
All 16 stats lines are present: 7 splits × 2 tasks for mode=normal, plus val_id × 2 for mode=none. parse_rate is 1.000, and missing_rate and truncation_rate are 0.000 on every line. **No omission anywhere, including val_ood_config.**

## Reproduction: PASSED (exact)
| check | result |
|---|---|
| (i) primary vs 0.3136 | 0.31357, identical to exp000's 0.31356803… |
| (ii) `compare exp003 exp000 --splits val_id val_ood_*` | diff 0.0 on every split, CI [0, 0]; val_id_target_mean 0.15506 vs 0.155 |
| (ii) same with `--task A` | diff 0.0 on every split |
| (iii) `compare exp003:none exp000:none --splits val_id` | diff 0.0; image_dependence_gap 0.3313, within 0.331 ± 0.010 |
| (iv) Task A all_mean, val_id / val_ood_config | 0.016 / 0.136, vs 0.016 / 0.136 |

So exp003's numbers stand in for exp000 on every split, and greedy decoding under the snapshot executor is bit-reproducible.

## Primary metric (unchanged reference)
`compare exp003 static`: 0.3136 vs 0.3790, diff −0.0654, CI [−0.0754, −0.0557]. Per split:
- objects −0.071
- forces −0.154
- config +0.028 [0.013, 0.042]

## C1: Task B target vs static, per split (`compare exp003 static --splits val_id val_ood_config val_diag_*`)
| split | D_s (exp003 − static) | CI95 | S_s = D_s − D_val_id | R_f = S_f / S_cfg | conservative R_f interval |
|---|---|---|---|---|---|
| val_id | −0.1104 | [−0.1288, −0.0951] | – | – | – |
| val_ood_config | +0.0284 | [+0.0134, +0.0440] | S_cfg = +0.1388 | 1 | – |
| val_diag_elev | −0.0525 | [−0.0751, −0.0309] | +0.0578 | **0.417** | [0.144, 0.705] |
| val_diag_count | −0.0725 | [−0.0831, −0.0634] | +0.0379 | **0.273** | [0.087, 0.471] |
| val_diag_corner | −0.0554 | [−0.0746, −0.0375] | +0.0550 | **0.396** | [0.148, 0.658] |

**Precondition met.** S_cfg = 0.139 ≥ 0.05, and config's CI lower bound (+0.013) is above val_id's CI upper bound (−0.095).

The intervals are S_f bounds from the C1 CI endpoints, divided by the S_cfg point estimate. They capture numerator noise only and are not full CIs.

**CI-robust finding: no single factor reproduces the config failure.** Every single-factor split has its C1 CI below 0, i.e. better than static. Config's CI is above 0. So every R_f's upper bound is below 1, whatever the noise.

**Per factor (point estimates):** every R_f < 0.5 and every R_f > 0.25, so **all three are PARTIAL**. Both labels are point-estimate only in places:
- "Not implicated" is point-estimate only for elev [0.144, 0.705] and corner [0.148, 0.658], whose intervals cross 0.5.
- "Not cleared" is point-estimate only for count [0.087, 0.471], whose interval crosses 0.25.

**Outcome (5), "diffuse" (point-estimate only).** R_elev + R_count + R_corner = 1.086 ≥ 0.75. But the conservative interval on the sum, [0.38, 1.83], crosses 0.75, so the data can't separate (4) "interaction" from (5) "diffuse". Each single factor reproduces 27–42% of the config shift at the point estimate. The sum is close to 1, but whether the contributions are additive is not established (IDEA-036, a two-factor split).

## Hypothesis verdict: FALSIFIED (point-estimate only: R_elev 0.417, conservative interval [0.144, 0.705], crosses 0.5)
- R_elev < 0.5, so FALSIFIED by the first rule. R_elev is the largest point estimate, so the second clause doesn't fire.
- The corner clause (R_corner ≤ 0.25) also fails, but only at the point estimate: 0.396 [0.148, 0.658].
- The ranking elev > corner > count isn't separable either, since all three intervals overlap.

What the data robustly support:
- No single factor reproduces the config failure; each single-factor split alone stays better than static.
- Elevation is not shown to be the dominant driver.

## Mechanism read-outs (routing only, not the verdict)
**(a) Perception share** P_f = (A_f − A_id) / (A_cfg − A_id), with A_id = 0.016 and A_cfg = 0.136 (results.md Task A all_mean):

| split | Task A all_mean | P_f |
|---|---|---|
| val_diag_elev | 0.082 | 0.55 |
| val_diag_count | 0.019 | 0.03 |
| val_diag_corner | 0.078 | 0.52 |

Perception degrades at low elevation and in the corner cluster. It does **not** degrade with 4–5 objects. (Zero-shot exp002 ranked count worst, a different model; context only.)

**(b) Omission:** missing_rate is 0.000 on every split, so there is no omission artefact.

**(c) Dynamics**, C2 = `compare exp003 exp003:ownstatic --splits …`, G_s = Task B minus own-static. The precondition holds: val_id's CI upper bound is −0.097 < 0. PRESERVED requires a CI upper bound < 0 and G_s ≤ 0.5·G_id = −0.056.

| split | G_s | CI95 | status |
|---|---|---|---|
| val_id | −0.1125 | [−0.1310, −0.0972] | (ref.) |
| val_diag_elev | −0.0807 | [−0.1019, −0.0605] | PRESERVED |
| val_diag_count | −0.0737 | [−0.0839, −0.0647] | PRESERVED |
| val_diag_corner | −0.0746 | [−0.0924, −0.0589] | PRESERVED |
| val_ood_config | −0.0279 | [−0.0399, −0.0154] | REDUCED |

**(d) Cost of own vs oracle t=0 positions** on Task B, C3 = `compare exp003:ownstatic static --splits …`:

| split | diff | CI95 |
|---|---|---|
| val_id | +0.002 | [0.001, 0.003] |
| val_ood_config | +0.056 | [0.048, 0.065] |
| val_diag_elev | +0.028 | [0.020, 0.035] |
| val_diag_count | +0.001 | [−0.000, 0.003] |
| val_diag_corner | +0.019 | [0.013, 0.025] |

**Decomposition.** By construction, D_s = C3_s + G_s (exp003 − static = (ownstatic − static) + (exp003 − ownstatic)).
- val_id: +0.002 − 0.113 = −0.110.
- val_ood_config: +0.056 − 0.028 = +0.028.

**The decomposition assumes** that the Task A answer stands in for the perception inside the Task B answer. Supporting check: on every split, Task B unmoved_mean ≈ own-static unmoved_mean (results.md, :normal vs :ownstatic):

| split | :normal | :ownstatic |
|---|---|---|
| val_id | 0.017 | 0.016 |
| config | 0.132 | 0.134 |
| elev | 0.081 | 0.084 |
| count | 0.020 | 0.020 |
| corner | 0.081 | 0.082 |
| objects | 0.025 | 0.025 |
| forces | 0.019 | 0.016 |

On the own-static path, and at the point estimate, the config shift of 0.139 splits into:
- about 0.054 from worse perception (ΔC3; conservative interval [0.045, 0.063]);
- about 0.085 from a smaller dynamics gain (ΔG; [0.057, 0.116]).

The two intervals overlap, so **which part is larger is not CI-separable**. The split is also path-dependent; IDEA-034 (re-anchoring on true t=0 positions) gives the second path.
- The perception cost on config, +0.056 [0.048, 0.065] vs val_id's +0.002, is the CI-solid part.
- No single factor breaks dynamics: G stays between −0.074 and −0.081 on each diagnostic split.

**Additivity:**
- The dynamics side is *sub*-additive. The single-factor losses vs val_id sum to 0.109, against config's 0.085.
- The perception side is *super*-additive. The single-factor ΔC3 sum to 0.042, against config's 0.054.

**Motion confound** (corrected). G_s scales with how far the target moves, so normalise by static target_mean: K_s = −G_s / static target_mean_s = target_skill(:normal) − target_skill(:ownstatic).

| split | K_s |
|---|---|
| val_id | 0.424 |
| elev | 0.332 |
| count | 0.346 |
| corner | 0.237 |
| config | 0.116 |

- Less motion explains only about 0.010 of config's 0.085 drop in G; dynamics skill per unit of motion really is lower.
- On corner, K is 0.56 of val_id's, close to the 0.5 line, so its PRESERVED status is borderline once motion is normalised.

Own-static on the other OOD splits (`compare exp003 exp003:ownstatic`, default splits): objects −0.074 [−0.083, −0.065], forces −0.160 [−0.184, −0.136]. Task B beats its own-perception static on all 7 splits (C-007).

## Routing (spec `if_it_fails` / `routing`, as amended)
- R_elev < 0.5 (point estimate), so the "camera prior alone" story is not supported for the fine-tuned model.
- No factor is implicated and dynamics are not BROKEN, so the spec's routing rules don't fire.
- **IDEA-004 dropped:** count perception is fine (P = 0.03).
- **IDEA-002 is not ranked above the geometry levers on config grounds.** The dynamics-vs-perception split is not CI-separable, and the perception cost on config is the CI-solid component.
- **Next are the cheap discriminators.** Both are diagnostics on existing exp000 predictions or metadata, and their routing value is higher than that of a training run:
  - IDEA-034: re-anchor exp000's implied displacement on the true t=0 positions (the second decomposition path).
  - IDEA-035: split the dynamics gain by whether the target makes contact with other objects (lost skill vs harder multi-body physics).
- **IDEA-036** (two-factor elev × corner split, needs a human data config) stays open for (4) vs (5).
- IDEA-033 can only diagnose an IDEA-002 model, so it doesn't route anything for exp000.

## Verdict (after post-review)
- **Hypothesis FALSIFIED, point-estimate only** (R_elev 0.417 [0.144, 0.705]).
- **CI-robust:** no single config factor reproduces exp000's config failure, since every single-factor split beats static (C-006).
- **Outcome (5), "diffuse", point-estimate only.** The sum of R is 1.086, with interval [0.38, 1.83], so (4) can't be told from (5).
- **Perception degrades at low elevation and in the corner cluster, not with 4–5 objects** (C-008, tentative).
- **On config,** the perception cost (+0.056, CI-solid) and the lost dynamics gain (0.085 at the point estimate, motion-normalised K 0.116 vs 0.424) both contribute. Which is larger is not separable.
- **exp000 reproduces bit-exactly** under the snapshot executor (C-005).

## Reviewer (post): AMEND, adopted
See `review_post.md`. All 8 amendments are adopted:
1. The interval is in the verdict sentence, the corner clause is flagged, and the section leads with the CI-robust finding.
2. Outcome (5) is flagged, with the sum interval and count's "not cleared" flag.
3. The decomposition is restated as point-estimate only and path-dependent, with the unmoved check.
4. The additivity sentence is corrected (sub-/super-additive).
5. The motion caveat uses K_s.
6. R_corner is 0.396.
7. The routing is fixed (IDEA-034/035 first, IDEA-036 open).
8. Own-static on objects and forces is quoted.

CLAIMS: C-005..C-008 added; notes on C-001 and C-002.
