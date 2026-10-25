# exp004_exp000_oracle_anchor: report (provisional, pending post-review)

**Privileged diagnostic, uses GT t=0; not a model result.** This sentence applies to every `oracleanchor`, `isoanchor`, `contanchor` and `offanchor` number below. Task B only. K = skill = `-diff_vs_static / static target_mean` (= results.md `target_skill`).

## Hypothesis and change
Hypothesis (pre-registered): exp000's loss of dynamics gain on val_ood_config comes from its own perception error and from harder multi-body or off-table episodes, not from worse displacement prediction. Prediction: on isolated episodes, rho = K_iso_cfg / K_iso_id >= 0.75; falsified if rho <= 0.5 (with the consistency gate passed).
Change: none to model or prompts. Eval-only run of the exp005 adapter (exp000 retrained unchanged) with instrumentation only (strata lines, derived pred files). Job done, 0.231 GPU-h (budget 0.75), commit 6052307, 18 preflight tests passed. Third submit attempt; the first two were denied because another user held the GPU.

## Validity and reproduction (vs exp005)
- metrics.json: primary 0.3160281875 (exp005: identical), image_dependence_gap 0.34253104 (exp005: identical), parse_rate_min 1.000.
- (i) PASS (diff 0). (ii) `compare exp004 exp005_exp000_repro` over all 8 splits: every per_split_diff is exactly 0.0, CI [0, 0], so `normal.jsonl` is bit-exact. (iii) Task A :normal rows in results.md match exp005's (val_id 0.015, objects 0.023, forces 0.016, config 0.137, count 0.021, elev 0.084, corner 0.079, elev_corner 0.122). (iv) PASS (0.3425).
- Strata lines, 8 present. iso + cont + off = N on every split (id 395+31+74, objects 410+42+48, forces 260+22+218, config 301+121+78, elev 246+21+33, count 215+58+27, corner 196+32+72, elev_corner 202+40+58). `disp_all` equals the static target_mean in results.md (e.g. id 0.2654, config 0.2410).
- **Not checked by me:** the "stats line x 2 tasks" and "2 none-mode lines" parts of the validity rule. I only grepped the strata lines. The reviewer should check them. The parse and missing rates I saw in the log tail were 1.000 and at most 0.001.
- Sum check: for each split the three stratum diffs add up to the oracleanchor diff (val_id -0.052341 - 0.006934 - 0.052247 = -0.111522; config -0.013513 - 0.007734 - 0.015318 = -0.036566).
- Merge rule: val_id has n_cont = 31 (>= 30) and config 121, n_off >= 74, so no merge. n_cont = 31 on val_id makes the cont K wide.

## Own-path anchors (computed from `exp004:ownstatic` first, before the oracle outcomes were read)
G_s = `compare exp004 exp004:ownstatic` per_split_diff; K_own_s = -G_s / static target_mean.
| split | G | K_own | C3 (ownstatic vs static) |
|---|---|---|---|
| val_id | -0.11006 | 0.415 | +0.00187 |
| val_ood_config | -0.02468 | 0.102 | +0.05612 |
| val_diag_elev | -0.07418 | 0.305 | +0.02880 |
| val_diag_count | -0.07603 | 0.357 | +0.00122 |
| val_diag_corner | -0.07063 | 0.224 | +0.01923 |

K_own_cfg / K_own_id = 0.247 (exp000 reference from exp003: 0.116 / 0.424 = 0.27). delta C3 = 0.0543, delta G = 0.0854.

## Oracle path (privileged), S5, Task B target mean vs static
| split | N | D_oa (diff vs static) | CI95 | K_oa |
|---|---|---|---|---|
| val_id | 500 | -0.11152 | [-0.12804, -0.09726] | 0.420 |
| val_ood_config | 500 | -0.03657 | [-0.04843, -0.02359] | 0.152 |
| val_diag_elev | 300 | -0.08603 | [-0.10691, -0.06574] | 0.354 |
| val_diag_count | 300 | -0.07708 | [-0.08705, -0.06886] | 0.362 |
| val_diag_corner | 300 | -0.08094 | [-0.09874, -0.06548] | 0.257 |

K_oa agrees with results.md target_skill (id 0.420, config 0.152). Other splits from results.md: objects 0.326, forces 0.231, elev_corner 0.248.
Consistency gate: results.md unmoved_mean of :oracleanchor on val_ood_config is 0.010 m (val_id 0.006), against the 0.05 m limit, so **the gate passed**. The own-path comparator is `ownstatic` unmoved_mean on config, 0.136.

## Per-stratum skill (rescaled: D_k = diff * N / n_k; K_k = -D_k / disp_k; CI endpoints scale the same way)
| split | stratum | n | disp_k | D_k | K_k | K_k CI (from rescaled CI) |
|---|---|---|---|---|---|---|
| val_id | iso | 395 | 0.1374 | -0.06626 | 0.482 | [0.423, 0.539] |
| val_id | cont | 31 | 0.2115 | -0.11184 | 0.529 | [0.321, 0.756] (corrected by reviewer; I had [0.332, 0.743]) |
| val_id | off | 74 | 0.9716 | -0.35302 | 0.363 | [0.260, 0.482] |
| val_ood_config | iso | 301 | 0.1033 | -0.02245 | 0.217 | [0.075, 0.341] |
| val_ood_config | cont | 121 | 0.1427 | -0.03196 | 0.224 | [0.031, 0.394] (corrected by reviewer; I had [0.015, 0.389]) |
| val_ood_config | off | 78 | 0.9252 | -0.09819 | 0.106 | [0.063, 0.162] (corrected by reviewer; I had [0.059, 0.164]) |

Precondition met: n_iso >= 30 on both splits, and the val_id iso CI upper bound is below 0 (-0.0459 before rescaling).
Diag-split iso K (same method): elev 0.337 (n_iso 246), count 0.444 (215), corner 0.444 (196). I did not compute the cont and off K on the diag splits.

## Verdict (provisional): hypothesis FALSIFIED, point-estimate only
- rho = 0.217 / 0.482 = 0.451, which is <= 0.5 with the gate passed. By the pre-registered order that is **FALSIFIED**.
- The conservative interval is [0.075/0.539, 0.341/0.423] = [0.14, 0.81]. It crosses both the 0.5 and the 0.75 thresholds, so the call is point-estimate only (the pre-registration allows exactly this flag).
- In words: with GT t=0 removing the perception offset, the model's displacement skill still falls on config, even on isolated single-body pushes (K 0.48 -> 0.22), and in every stratum (cont 0.53 -> 0.22, off 0.36 -> 0.11).

## Routing readouts (point estimates)
- (a) Q = K_oa_cfg / K_oa_id = 0.152 / 0.420 = 0.361, against the own-path ratio 0.247. The oracle path recovers some skill but not close to 0.75.
- (b) K_oa_id - K_oa_cfg = 0.269 (>= 0.05, so reported). K_cfg_comp (config's stratum mix at val_id's per-stratum K) = 0.418, so Comp = (0.420 - 0.418) / 0.269 = 0.01. **The stratum mix explains about 1% of the loss.** More cont/off episodes on config is not the cause.
- (c) In distribution: K_cont_id 0.529 and K_iso_id 0.482 are alike (cont wide, n = 31); K_off_id 0.363 is lower.
- (d) S_cfg = 0.1396 (config minus val_id normal-vs-static diff, from -0.108196 and +0.031432). `compare exp004 exp004:oracleanchor` on val_id and config gives per-split diffs 0.00333 and 0.06800 (CIs [0.0023, 0.0044] and [0.0597, 0.0764]). Perception part P_oa = 0.0647, dynamics part D_oa_cfg - D_oa_id = 0.0750. Own-path split: delta C3 = 0.0543, delta G = 0.0854. Shapley-style average: perception 0.0595, dynamics 0.0802 (exp000 reference own-path split 0.054 / 0.085). So about 43% perception and 57% dynamics, point estimates.
- (e) Diag splits (overall K_oa): elev 0.354, count 0.362, corner 0.257, against val_id 0.420 and config 0.152. In iso only: elev 0.337, count 0.444, corner 0.444. Count shows no loss, corner is lost mostly outside isolated pushes, elev loses on isolated pushes too, and config is lower than any single factor.

## Routing (pre-registered R3, amended by the post-review)
Pre-registered R3 fired (FALSIFIED), which would promote IDEA-002. **The reviewer amended the lever choice and I adopt that, stated openly so the pre-registration is not silently overridden:**
- Denominator effect: iso target displacement falls from 0.137 m (val_id) to 0.103 m (config), but the iso oracle absolute error only rises from 0.071 m to 0.081 m. So much of the K drop on isolated pushes is a smaller denominator. In D space the config loss splits about iso 0.0388, off-table 0.0369, cont -0.0008, so roughly half is edge-fall episodes.
- Camera-frame confound (my caveat, confirmed): actions are camera-relative and the prompt carries no elevation. Left/right pushes are elevation-independent in the camera frame; away/toward pushes are not.
- iso K is 0.337 for elev alone and 0.444 for count or corner alone, against 0.217 on config. That points to a three-way interaction or a different regime, not one camera mechanism.
- IDEA-002 still outputs a camera-frame vector and the oracle path already removes the t=0 anchor error. Nothing here selects it over IDEA-016 or IDEA-014. It stays a candidate on forces grounds only.
- Next step before any 1.4 GPU-h run: IDEA-050 on the existing oracleanchor files (K_iso for left/right vs away/toward pushes, along/cross-push error split, absolute iso error with CI). No new generation is needed. Also IDEA-051 (diag-split strata with CIs) and IDEA-052 (edge-fall analysis).

## Post-review (adopted)
Reviewer: the hypothesis stays falsified on the point estimate only (rho 0.451, gate passed, interval [0.14, 0.81] crosses both thresholds). All my hand-computed D_k, K_k, rho, Q, Comp (about 0.007; I wrote 0.01), S_cfg and the Shapley split were recomputed and match, except the three CI endpoints corrected above. Validity passes: stats lines 16 of 16 (8 splits x 2 tasks, parse 1.000, missing at most 0.001), both none-mode lines present, strata sum to N, disp_all matches static on all 8 splits. C-010 added to CLAIMS.md as tentative.

## Caveats
- The oracle path mixes separate Task A and Task B generations. The gate (0.010 m on config vs 0.05 m) says that noise is small.
- Camera-frame confound: displacement is predicted in camera coordinates, so the same physical push maps to a different camera-frame vector when elevation changes. Skill loss on isolated pushes may reflect missing camera-pose knowledge, not lost dynamics. If so, a displacement-output change (IDEA-002) would not fix it, while camera conditioning (IDEA-016) or world-frame output could. exp004 cannot separate these.
- Single adapter, single seed (exp005, no seed-noise estimate; IDEA-043). rho's interval spans both thresholds.
- n_cont = 31 on val_id is small.
