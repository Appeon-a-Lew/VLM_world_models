# Review (post): exp004_exp000_oracle_anchor

Verdict: the numeric content of the report is APPROVED (point-estimate FALSIFIED for the pre-registered hypothesis stands, flagged point-estimate only). The routing R3 -> "promote IDEA-002 on config grounds" is NOT approved. AMEND routing (below).

## 1. Recomputation
Recomputed from `agent.sh compare` (oracleanchor, iso/cont/off anchors, ownstatic, P_oa) and results.md. All of the following match the report:
- D_oa, K_oa: id -0.11152 / 0.420, config -0.03657 / 0.152; Q = 0.361.
- Rescaled D_k and K_k for all six stratum rows (iso id 0.482, config 0.217; cont 0.529, 0.224; off 0.363, 0.106). Iso K CIs match.
- rho = 0.451; conservative interval [0.139, 0.805].
- K_own id 0.415, config 0.102, ratio 0.247. dC3 0.0543, dG 0.0854, S_cfg 0.1396, P_oa 0.0647, D_oa 0.0750, Shapley 0.0595 / 0.0802 (43% / 57%).
- Comp: K_cfg_comp 0.4177, Comp 0.007 (report 0.01). Also checked in absolute-diff space: config's stratum mix at val_id per-stratum D predicts -0.122 on config vs actual -0.037, so the mix explains none of the loss there either.
- Diag iso K: elev 0.337, count 0.444, corner 0.444.
Small discrepancies (no effect on any call): K CI for cont id I get [0.321, 0.756] (report [0.332, 0.743]); cont config [0.031, 0.394] (report [0.015, 0.389]); off config [0.063, 0.162] (report [0.059, 0.164]). Fix the table.

## 2. Validity items not checked by the author
- Stats lines x 2 tasks: 16/16 present (8 splits x A,B), parse 1.000, missing at most 0.001 (Task A on config and count), truncation 0.000.
- None-mode lines: 2/2 present (val_id A and B), parse 1.000, missing 0.000.
- Strata: 8/8, iso + cont + off = N on all, disp_all within 0.001 of static target_mean on all 8 splits.
- Reproduction and gate: as the report states; oracle unmoved_mean config 0.010, id 0.006, gate passed.
Validity passes.

## 3. Does the CI support the verdict?
- The K_iso drop is real at the interval level: config upper 0.341 is below id lower 0.423.
- The threshold call is not: rho interval [0.14, 0.81] crosses 0.5 and 0.75. The report says so correctly. Treat as "rho < 0.75 likely, <= 0.5 not established". A paired bootstrap of the ratio (CPU, minutes) would tighten it, but it is not needed for routing.
- Per-split: the K drop shows up in every stratum on config, but cont is small and wide (n 31 on id). The consistent pieces are iso and off; cont adds nothing (cont D contributes -0.0008 to the loss).

## 4. Main finding that changes the interpretation
The K drop on isolated pushes is largely a denominator effect. Iso target displacement is 0.137 m (id) vs 0.103 m (config). The oracle-path iso absolute error is 0.1374 - 0.0663 = 0.071 m (id) vs 0.1033 - 0.0224 = 0.081 m (config). So in metres the model's error on isolated pushes rose by about 0.010 m, while the pushes themselves became shorter. "Dynamics skill is lost even on single-body pushes" is true of the ratio but weakly so of the metres. The off-table stratum carries the metre-level loss: oracle error 0.619 -> 0.827 m.
In D space (diff vs static, per-split units) the 0.0750 config loss splits iso 0.0388, off 0.0369, cont -0.0008. About half is edge-fall episodes.
A caveat on K itself: a ratio to static displacement is not rotation-invariant in what matters (norms are, but the error floor is not scaled by displacement). Report absolute error alongside K whenever a stratum's disp differs by 25 percent.

## 5. The camera-frame confound and the R3 routing
Facts from the repo: actions are camera-relative (left/right/away/toward, configs/sim_train.yaml:31) and the prompt gives intrinsics and the action text but not elevation. Displacement norms are rotation-invariant, but the model must output a camera-frame vector. left/right pushes lie mostly along camera x (elevation-independent). away/toward pushes mix camera y and z with an angle equal to elevation (train 30-60 degrees, config 12-25 degrees), so a model with a train-elevation prior would put cross-push error into the away/toward half of the episodes.
Evidence from this run on that confound:
- Against elevation alone: elev-only iso K is 0.337 (ratio 0.70 to id), count 0.444, corner 0.444, elev_corner roughly 0.38 (from rounded results.md values). Config iso K is 0.217 and is below every single factor and the pair. That points to a three-way interaction or a different regime, not one camera-frame mechanism.
- For some elevation role: elev has the largest single-factor drop on iso (no CI).
- Off-table episodes (half of the loss) are a layout/physics effect, not a frame effect.
What this means for the lever:
- IDEA-002 (displacement output) removes the t=0 anchoring error, but the oracle path already does that, and B^ - A^ is already the model's implicit displacement. It still outputs a camera-frame vector. It would not fix a camera-frame confound, and nothing here shows that a different output head fixes generalisation of the learned displacement map. Its justification stays the forces share (C-002, 78% of the aggregate gain), not config.
- IDEA-016 (camera pose in prompt) or a table/world-frame output would address only the away/toward component, and only if IDEA-050 shows it.
- Nothing in this run selects IDEA-002 over IDEA-016/014. The pre-registered R3 wording ("IDEA-002 becomes the next training run, config as well as forces as justification, IDEA-014 runner-up") assumed that the dynamics loss is a lever-addressable viewpoint/layout loss. The data support only "a dynamics-side loss exists", about 57% of the shift, point estimate, with a large off-table component.
Required routing amendment:
1. Do not launch a 1.4 GPU-h run on R3 as written. Run IDEA-050 first (CPU, minutes): K_iso by action direction (left/right vs away/toward) on val_id, val_diag_elev, val_ood_config, plus along/cross-push error split and absolute iso error with a paired CI. Decision rule written before looking: if the K loss on config iso is similar for left/right and away/toward (difference within CI), the camera-frame explanation is out and the loss is generic; if it is concentrated in away/toward with cross-push error, camera conditioning or table-frame output (IDEA-016) goes ahead of IDEA-002.
2. Run IDEA-052 (off-table edge-fall analysis, CPU) in the same pass: half the loss.
3. IDEA-002 stays a candidate on forces grounds (also pending IDEA-025/026). Keep IDEA-014 as the geometry lever for the perception half (about 0.060 of 0.140, point estimate).
4. The report must state "R3 (pre-registered) fired; reviewer amends the lever choice" so the pre-registration is not silently overridden.

## 6. Parse rates, image dependence
Parse rate 1.000 everywhere, no change vs exp005 (bit-exact normal.jsonl). Image-dependence gap 0.3425 identical to exp005. No model changed, so no shortcut-learning signal can arise here.

## 7. Alternative explanations (at least two) and discriminators
1. Denominator/magnitude effect: config iso pushes are shorter, error floor roughly constant (0.071 vs 0.081 m). Discriminator: absolute iso error with a paired CI (IDEA-051/050, CPU).
2. Camera-frame (elevation) confound on away/toward pushes: discriminator is the direction split above (IDEA-050).
3. Off-table / layout (corner near the edge) regime: half the D loss. Discriminator: edge-fall analysis, fall-prediction rate (IDEA-052).
4. A/B inconsistency noise on config (unmoved 0.010 vs 0.006 m): small, direction biases toward the finding. Discriminator: not needed beyond the gate.
5. Single seed: IDEA-043(b).
Added to BACKLOG as IDEA-050, 051, 052 (source: review of exp004_exp000_oracle_anchor).

## 8. CLAIMS
Added C-010 (tentative) with the denominator and off-table caveats. C-007's split remains point-estimate only and unchanged. No claim contested.

## 9. Does the verdict stand?
- Hypothesis FALSIFIED (point-estimate only, rho 0.451, gate passed): stands as recorded, with the caveat that it is a ratio and that the metre-level iso loss is only about 0.010 m.
- Routing R3 to IDEA-002 on config grounds: does not stand. Replace by IDEA-050/052 (CPU), then pick between IDEA-016/014 and IDEA-002.
