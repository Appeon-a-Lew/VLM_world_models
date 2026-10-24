# Post-review: exp003_exp000_diag_eval

Reviewer, 2026-10-24. Run done at 6cec893 (0.214 GPU-h of 1.0). Draft report at
`research/experiments/exp003_exp000_diag_eval/report.md`.

## Verdict: AMEND

The run is valid. The reproduction is exact, and the mechanical hypothesis verdict (FALSIFIED, flagged
point-estimate only) is correct. Every raw number in the report matches the tool outputs. Three
interpretive statements go beyond what the CIs support, and two read-outs use the wrong quantity:
- outcome (5) "diffuse" is reported unflagged;
- "the larger part is lost dynamics skill" is reported unflagged;
- "the single-factor losses roughly add up" is wrong for the dynamics side;
- the motion caveat uses the wrong columns;
- IDEA-033 is routed as a diagnostic for exp000, which it cannot be.

The amendments are text-only. No re-run is needed.

## 1. Numbers checked

Sources: `agent.sh results`, `agent.sh log exp003_exp000_diag_eval 400`, and every `compare` call listed below.

**Validity.** All 16 stats lines are present: 7 splits × A/B for normal, plus val_id × A/B for none.
parse_rate is 1.000 and missing_rate and truncation_rate are 0.000 on every line. results.md parse_rate is
1.000 on every row. Parse rates did not change, so no accuracy change can be a parse artefact.

**Reproduction (i)-(iv): PASSED, exact.**
- metrics.json value is 0.31356803, identical to exp000.
- `compare exp003 exp000 --splits val_id val_ood_objects val_ood_forces val_ood_config` gives diff 0.0 with
  CI [0, 0] on every split. The same holds with `--task A`, and for `exp003:none` vs `exp000:none` on val_id.
- val_id_target_mean is 0.15506.
- image_dependence_gap is 0.33129. `compare exp003:none exp003:normal --splits val_id` gives
  +0.331 [0.298, 0.363], unchanged from exp000.
- This is not a training experiment, so there is no shortcut concern.

**Primary metric.** `compare exp003 static` gives −0.0654 [−0.0754, −0.0557]. Per split:
- objects −0.071 [−0.080, −0.062];
- forces −0.154 [−0.178, −0.130];
- config +0.028 [+0.013, +0.042] ("b better").

`compare exp000_sft_baseline static` returns the same numbers, so these are exp000's own per-split CIs.
C-002 is updated with them.

**C1 (recomputed).**

| split | D_s | CI | S_f | R_f | conservative interval |
|---|---|---|---|---|---|
| val_id | −0.11038 | [−0.12877, −0.09509] | – | – | – |
| config | +0.02840 | [+0.01339, +0.04401] | 0.13878 | 1 | – |
| elev | −0.05253 | [−0.07508, −0.03093] | 0.05785 | 0.417 | [0.144, 0.705] |
| count | −0.07250 | [−0.08306, −0.06337] | 0.03789 | 0.273 | [0.087, 0.471] |
| corner | −0.05536 | [−0.07459, −0.03751] | 0.05502 | **0.396** (report: 0.397) | [0.148, 0.658] |

- Sum of R_f = 1.086.
- Precondition met: S_cfg is 0.139, and the config CI lower bound (0.013) is above the val_id CI upper bound (−0.095).
- Every interval matches the report.

**C2 (G_s) and status (recomputed).**
- G_id = −0.11252, CI upper bound −0.0972 < 0, so the precondition holds. 0.5·G_id = −0.05626.
- elev −0.0807 [−0.1019, −0.0605], count −0.0737 [−0.0839, −0.0647] and corner −0.0746 [−0.0924, −0.0589] are all PRESERVED.
- config −0.0279 [−0.0399, −0.0154] is REDUCED.
- All statuses match the report.
- Added: objects −0.074 [−0.083, −0.065] and forces −0.160 [−0.184, −0.136] (`compare exp003 exp003:ownstatic --splits val_ood_objects val_ood_forces`). So G < 0 with CI on all 7 splits.

**C3 (recomputed).** All five rows match the report:
- id +0.0021 [0.0012, 0.0031];
- config +0.0563 [0.0477, 0.0646];
- elev +0.0282 [0.0205, 0.0355];
- count +0.0012 [−0.0004, 0.0028];
- corner +0.0193 [0.0131, 0.0254].

**Decomposition.** D_s = C3_s + G_s holds exactly on all five splits; the residual is ≤ 1e-6. On config:
- ΔC3 = 0.05420;
- ΔG = 0.08458;
- sum = 0.13878 = S_cfg.

**P_f.** A_id = 0.0157 (metrics.json), A_cfg = 0.136. This gives elev 0.55, count 0.03 and corner 0.52, matching the report. These are point estimates with no CI.

## 2. Rulings

### Point 1: verdict wording and outcome (5)

**The verdict is correct and the wording meets the round-2 note.** The heading and the provisional-verdict bullet both put
"point-estimate only" in the same sentence as FALSIFIED. Amendment A1 asks for the interval to be put in that sentence as well.

**Outcome (5) is applied correctly to the point estimates, but it is not flagged, and it must be.**

Outcome (5) needs "none implicated" and a sum ≥ 0.75. Neither condition holds robustly:
- **Elevation and corner:** their "not implicated" calls are point-estimate only. Their intervals reach 0.705 and 0.658, which cross 0.5.
- **Count:** the only robust non-implication (upper bound 0.471 < 0.5).
- **The sum:** the conservative interval (Σ D_f lo − 3·D_id hi, Σ D_f hi − 3·D_id lo, over S_cfg) is [0.38, 1.83]. That crosses 0.75.

So the data do not separate (5) "diffuse" from (4) "interaction". Most of the sum's width comes from D_id's CI, which enters all three R_f with the same sign.

The report's line "together they roughly add up to it" therefore reads more into 1.086 than the data allow.

Line 51, "None is cleared either", is also point-estimate only for count, whose interval [0.087, 0.471] crosses 0.25.

**What is CI-robust:**
- Each single-factor D_f has its CI upper bound below 0, at −0.031, −0.063 and −0.038.
- Config's CI lower bound is +0.013.
- Under the conservative interval every R_f < 1 (maximum 0.705).

"No single factor reproduces the config failure" is therefore robust. "Diffuse" is not. The report should lead with the robust statement and label (5) as a point-estimate classification.

### Point 2: "the larger part is lost dynamics skill"

**Not justified as written. It needs two flags and one correction. The motion-magnitude confound turns out to be small.**

(a) **The CIs do not separate the two parts.** Conservative cross-split intervals:
- ΔG ∈ [G_cfg lo − G_id hi, G_cfg hi − G_id lo] = [0.057, 0.116];
- ΔC3 ∈ [0.045, 0.063].

These overlap. "ΔG > ΔC3" is point-estimate only. The pinned outputs cannot give a tighter paired interval: the needed quantity, exp003 + static − 2·ownstatic, is not a `compare` pair.

(b) **The split depends on the decomposition path.** Errors are Euclidean, so perception error and displacement error do not add linearly. One example: config Task A target error is 0.133, yet it costs only 0.056 in own-static Task B error.

The report decomposes along one path, oracle → own perception → own dynamics. The other path, oracle perception with the model's dynamics, could give a different share. Only that path isolates dynamics with oracle perception, and it is not available for exp000 from the current outputs (see IDEA-034).

The decomposition also assumes that the Task A answer stands in for Task B's implicit t=0 perception. There is partial support for that. Task B unmoved_mean ≈ own-static unmoved_mean on every split:

| split | Task B unmoved_mean | own-static unmoved_mean |
|---|---|---|
| val_id | 0.017 | 0.016 |
| config | 0.132 | 0.134 |
| elev | 0.081 | 0.084 |
| corner | 0.081 | 0.082 |
| count | 0.020 | 0.020 |

Whether this also holds for the target object specifically is untested.

(c) **The additivity remark is wrong for the dynamics side.**
- **Dynamics:** single-factor ΔG are elev 0.032, count 0.039 and corner 0.038, which sum to 0.109. Config's ΔG is 0.085, so the dynamics side is *sub*-additive.
- **Perception:** single-factor ΔC3 are 0.026, −0.001 and 0.017, which sum to 0.042. Config's ΔC3 is 0.054, so the perception side is *super*-additive.

Both are point estimates. Some of the super-additivity may come from the Euclidean nonlinearity, since on Task A the single-factor perception losses sum to 0.131 vs config's 0.120.

(d) **Better normalisation, available from the outputs.** The report's caveat uses the wrong columns:
- It uses static `moved_mean`, which averages over *all* moved objects (metrics.py:55, 89), not the target. The target displacement is static **target_mean**: id 0.265, elev 0.243, count 0.213, corner 0.315, config 0.241. That holds because static's error is |p1 − p0| (metrics.py:52).
- It uses `target_skill` of `:normal`, which is 1 − err/static_err on targets. That is −D_s/static, so it includes the perception cost.

The dynamics-only normalised quantity is

  **K_s = target_skill(:normal) − target_skill(:ownstatic) = −G_s / static target_mean_s**

which can be read off results.md:

| split | K_s |
|---|---|
| val_id | 0.424 |
| elev | 0.332 |
| count | 0.346 |
| corner | 0.237 |
| config | 0.116 |

What K_s shows:
- **Config.** At val_id's skill, config's G would be −0.424 × 0.241 = −0.102. The observed G is −0.028. Of ΔG = 0.085, about 0.010 is less motion and about 0.074 is lower skill per metre of target motion. On config the confound is real but minor.
- **Corner.** The confound runs the other way. Corner has *more* target motion (0.315), so the raw "PRESERVED" hides a larger skill loss: K = 0.237 = 0.56·K_id, close to the 0.5 line. The spec's status rule is on raw G, so the status stays PRESERVED, but the report should say this.
- **Count.** About 0.022 of its 0.039 ΔG is motion and about 0.016 is skill.

The report's conclusion survives normalisation as a point estimate, but (a) and (b) still apply. The statement to make is:

"On the own-static path, at point estimate, more of the config shift is lost dynamics gain (0.085, of which ~0.074 is lower skill per metre of motion) than perception cost (0.054). The two parts are not CI-separable, and the share depends on the path."

### Point 3: routing

- **Dropping IDEA-004: agreed.** On the count split:
  - Task A is 0.019 vs 0.016;
  - C3 is +0.001 [−0.000, 0.003], not significant;
  - missing is 0.000.

  Count's shift is almost entirely on the dynamics side. Nothing supports a binding or omission lever. This holds for exp000 only; zero-shot exp002 ranked count worst, a different model.
- **Geometry levers: partly agreed.** They lose the "elevation alone" justification. However, perception cost is the *CI-robust* part of the config failure: C3_cfg is +0.056 [0.048, 0.065] vs val_id's +0.002, and Task A rises 8.5×. The dynamics > perception ranking is point-estimate only and path-dependent (Point 2). The geometry levers should not be demoted below IDEA-002 on config grounds alone.
- **IDEA-002 → IDEA-033: disagree with the ordering rationale.** IDEA-033 *depends on* IDEA-002 and diagnoses an IDEA-002 model, not exp000. It cannot tell us whether exp000's config loss is dynamics. IDEA-002's case rests on forces (78% of the gain; IDEA-025/026), and on config it is not yet established.

  Before spending 1.4 GPU-h on IDEA-002 *for config*, run two cheap discriminators on exp000's predictions:
  - IDEA-034, which gives the other decomposition path;
  - IDEA-035, which tests "lost skill" against "harder multi-body physics in clustered 4–5-object scenes".

  Since (4) vs (5) is unresolved, the outcome-(4) follow-up (a two-factor split, IDEA-036) also stays live.

### Point 4: claims

I added or updated the following in `research/paper/CLAIMS.md`:

| entry | status | content |
|---|---|---|
| C-005 | solid, methods | Inference is bit-reproducible under the snapshot executor. |
| C-006 | solid at adapter level | No single config factor reproduces exp000's config failure. |
| C-007 | solid for the sign | New. exp000's Task B beats its own-perception static on all 7 val splits. The config reduction is tentative. |
| C-008 | tentative | exp000's perception is not degraded by 4–5 objects. It is an absence-of-effect claim with no Task A CI. |
| C-002 | updated | Gains per-split CIs, so the config failure is now CI-backed. |
| C-001 | note added | A link to C-007. |

Not added: the decomposition shares (point-estimate only) and "diffuse" (point-estimate only).

## 3. Alternative explanations and discriminating experiments

1. **Diffuse/additive vs interaction.** The sum interval [0.38, 1.83] spans both readings. On the perception side, elev × corner may be super-additive: a low camera plus a tight cluster causes occlusion.
   - Cheapest test: a two-factor elev × corner diag split, evaluated on the exp000 adapter (IDEA-036).
   - Cost: needs a human config on main; under 0.1 GPU-h.
2. **The config dynamics loss is harder physics, not lost skill.** Config clusters 4–5 objects in a corner, so the target probably hits neighbours more often, and multi-body outcomes are harder to predict at any viewpoint.
   - Cheapest test: stratify G/K by "isolated target" (no non-target moves, from GT metadata) on val_id vs config vs diag (IDEA-035).
   - Expected result if this explanation holds: isolated-target K on config ≈ val_id's.
   - Cost: derived files only.
3. **The share depends on the decomposition path, or Task B's target perception differs from Task A's.**
   - Cheapest test: score the oracle-anchored implied displacement GT_t0 + (B̂ − Â) on exp000's existing predictions (IDEA-034). This approximates the oracle-perception path without training IDEA-002.
4. **Motion magnitude** (the report's caveat). Quantified above: about 0.010 of ΔG. It is mostly ruled out on config, so no experiment is needed.
5. **Corner split.** It changes three fields: min_gap, region and region_center. Its perception cost could come from a far-from-camera or image-edge placement rather than from tightness. That is unresolved, and the interpretation of R_corner should say so.

Items 1-3 are added to `research/ideas/BACKLOG.md` as IDEA-034/035/036, each with `source: review of exp003_exp000_diag_eval`.

## Amendments (text only)

1. **Verdict sentence.** Write "FALSIFIED (point-estimate only: R_elev = 0.417, conservative interval [0.144, 0.705] crosses 0.5)". Add that the corner clause also fails at point estimate (0.396 > 0.25, interval [0.148, 0.658]). Lead the section with the CI-robust result: every D_f CI < 0 while D_cfg CI > 0, and every R_f upper bound < 1.
2. **Outcome (5).** Label it "point-estimate only". "None implicated" is point-estimate only for elev and corner. The conservative sum interval is [0.38, 1.83], which crosses 0.75, so (4) is not excluded. Remove "together they roughly add up to it", or qualify it. Also flag count's "not cleared" (interval [0.087, 0.471] crosses 0.25). Update provisional-verdict bullet 2 accordingly.
3. **Decomposition claim.** Replace "The larger part of exp000's config failure is lost dynamics skill" with the statement given at the end of Point 2(d). Add the conservative intervals ΔG [0.057, 0.116] and ΔC3 [0.045, 0.063], and add the Task A ≈ Task B perception assumption together with the unmoved_mean check. Update provisional-verdict bullet 3 the same way.
4. **Additivity sentence.** Correct it: the dynamics side is sub-additive (Σ single ΔG 0.109 vs 0.085) and the perception side is super-additive (Σ ΔC3 0.042 vs 0.054). Both are point estimates.
5. **Motion caveat.** Replace `moved_mean` with static `target_mean`. Replace `target_skill(:normal)` with K_s = target_skill(:normal) − target_skill(:ownstatic): id 0.424, elev 0.332, count 0.346, corner 0.237, config 0.116. State that motion accounts for about 0.010 of config's ΔG, and that corner's normalised skill is 0.56·K_id.
6. **R_corner.** Change 0.397 to 0.396.
7. **Routing.**
   - IDEA-033 depends on IDEA-002 and diagnoses an IDEA-002 model, so it is not a follow-up diagnostic for exp000.
   - Route config to IDEA-034/035 first; they are cheap and need no training. Keep IDEA-036 for (4) vs (5).
   - Do not rank IDEA-002 above the geometry levers *on config grounds*, because C3_cfg is the CI-robust component.
   - The IDEA-004 drop stands.
8. **Context line.** Add the C2 values for objects and forces (−0.074 [−0.083, −0.065], −0.160 [−0.184, −0.136]), so that "G < 0 on all 7 splits" is quoted with CIs.
