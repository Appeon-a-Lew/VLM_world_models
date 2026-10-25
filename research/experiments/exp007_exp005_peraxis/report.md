# exp007_exp005_peraxis: report (provisional, pending post-review)

**Privileged diagnostic, uses GT t=0; not a model result.** This applies to every `[axis]`, `[axis-did]`, `[axis-signed]` and `[gtmotion]` number below (oracle-anchored path: GT t=0 plus the model's predicted displacement). Task B target, isolated (iso) episodes, camera frame, metres. LR = left/right pushes, AT = away/toward pushes. yz = mean(|r_y| + |r_z|), r = pred - gt1.

## Hypothesis and change
Hypothesis (A-yz): the LR iso error rise from val_id to the shifted splits sits in the y/z axes (what an elevation-dependent camera-frame mapping would produce) and the AT yz rise exceeds the LR yz rise (did_yz = AT minus LR > 0, CI lower bound > 0), against the alternative that generic perception noise inflates the LR y/z error.
Change: none to model, prompts or generation. Eval-only rerun of the exp005 adapter; print-only per-axis read-outs. Done in 0.231 GPU-h, commit 2724516 (pre-review: REVISE in round 1, APPROVE in round 2).

## Validity
- `compare exp007 exp005_exp000_repro`: per_split_diff exactly 0.0 with CI [0, 0] (the reviewer verified the 3 OOD splits; I saw 0.0 on all 8 in my own run). metrics.json value 0.3160281875 (same as exp005); the reviewer confirms the strata lines hash identically to exp006's. Bit-exact, so the stack did not drift; no FAILED line in the log. n_iso >= 30 per group on every split (smallest: val_diag_elev_corner LR 92).
- **LR probe validity check fails on every split, including val_id (formally invalid by a miscalibrated rule).** The spec required LR mean |gt y| + |gt z| <= 0.01 m on val_id and on every compared split. Printed `[gtmotion]` LR |y|+|z|: val_id 0.0299, val_diag_elev 0.0331, val_diag_elev_corner 0.0296, val_ood_config 0.0239. So by the pre-registered rule no split counts toward A-yz. The reviewer's point: LR GT y/z motion is small and stable (0.030, 0.033, 0.030, 0.024), so the elevation LR rise of 0.035 is not driven by GT motion; the invalidity is a formal one and does not change the route (A-yz, B-x and C-flat fail without the check too). This is a design miss on my side: the 0.01 limit was never calibrated against the in-distribution value (LR true y/z motion is about 0.015 m per axis even on val_id).

## Results
Mean-abs per axis (iso; AT group |gt| motion in last columns, from `[gtmotion]`):
| split | group | n | x abs | y abs | z abs | yz | yz/2 | gt \|y\|+\|z\| |
|---|---|---|---|---|---|---|---|---|
| val_id (baseline) | LR | 197 | 0.0563 | 0.0189 | 0.0192 | 0.0381 | 0.0191 | 0.0299 |
| val_id (baseline) | AT | 198 | 0.0250 | 0.0416 | 0.0476 | 0.0893 | 0.0446 | 0.1951 |
| val_diag_elev | LR | 134 | 0.0832 | 0.0366 | 0.0369 | 0.0735 | 0.0368 | 0.0331 |
| val_diag_elev | AT | 112 | 0.0303 | 0.0254 | 0.0627 | 0.0881 | 0.0441 | 0.1828 |
| val_diag_elev_corner | LR | 92 | 0.0490 | 0.0099 | 0.0253 | 0.0351 | 0.0176 | 0.0296 |
| val_diag_elev_corner | AT | 110 | 0.0171 | 0.0252 | 0.0568 | 0.0821 | 0.0410 | 0.1326 |
| val_ood_config | LR | 143 | 0.0687 | 0.0243 | 0.0300 | 0.0543 | 0.0271 | 0.0239 |
| val_ood_config | AT | 158 | 0.0237 | 0.0309 | 0.0523 | 0.0832 | 0.0416 | 0.1224 |

Changes from val_id (bootstrap 95% CIs, from `[axis-did]`):
| pair | d_yz LR | d_x LR | d_yz AT | d_x AT | did_yz (AT-LR) | did_x |
|---|---|---|---|---|---|---|
| val_diag_elev | +0.0354 [0.0025, 0.0716] | +0.0269 [0.0066, 0.0463] | -0.0012 [-0.0213, 0.0172] | +0.0053 [-0.0007, 0.0120] | -0.0366 [-0.0766, 0.0028] | -0.0216 [-0.0422, 0.0003] |
| val_diag_elev_corner | -0.0030 [-0.0179, 0.0105] | -0.0072 [-0.0198, 0.0059] | -0.0072 [-0.0312, 0.0145] | -0.0079 [-0.0138, -0.0025] | -0.0042 [-0.0314, 0.0237] | -0.0007 [-0.0149, 0.0128] |
| val_ood_config | +0.0161 [-0.0105, 0.0494] | +0.0124 [-0.0069, 0.0349] | -0.0061 [-0.0303, 0.0178] | -0.0013 [-0.0077, 0.0057] | -0.0222 [-0.0651, 0.0155] | -0.0137 [-0.0371, 0.0069] |

Signed means (descriptive; CI excludes 0 only where stated): LR y on val_diag_elev +0.0255 [0.0062, 0.0497] and z +0.0167 [0.0051, 0.0308]; LR y on config +0.0167 [0.0022, 0.0345]; AT z on config -0.0219 [-0.0337, -0.0097] and on elev_corner -0.0195 [-0.0383, -0.0033]; LR z on elev_corner -0.0083 [-0.0158, -0.0006]. No consistent sign across elev, elev_corner and config.

## Which rule fired (pre-registered)
- **A-yz: no.** Independently of the invalid LR probe: did_yz is negative on all three pairs and its CI includes 0 on all three (val_diag_elev upper end +0.0028); the LR yz rise reaches 0.02 only on val_diag_elev (config 0.0161, elev_corner -0.0030), so the two-split bar fails.
- **B-x: no.** x rise >= 0.02 with CI > 0 holds only for LR on val_diag_elev (+0.0269), where the yz rise is also >= 0.01; config LR x rise is +0.0124 with a CI including 0.
- **C-flat: no.** Only val_diag_elev_corner has both rises under 0.01 (and the LR probe there is also invalid); the yz rise on config is 0.0161 and on elev 0.0354.
- **D-mixed: yes.** Any other combination, including the invalid LR probe and the opposite ordering on elev_corner. Pre-registered routing: IDEA-014 next, no IDEA-016 training run, per the spec. (Not-A-yz, as the spec defines "falsified".)

## Verdict (provisional)
The hypothesis is **not supported, and not refuted**: A-yz fails because did_yz is not positive on any split (the real reason), with did_yz CIs about ±0.04, wide enough not to exclude a +0.03 m effect. The formally invalid LR probe is a second, weaker reason. The only visible pattern, an LR error rise on val_diag_elev in both yz (CI [0.0025, 0.0716], barely above 0) and x, is exploratory: about 12 uncorrected CIs, and no replication on val_diag_elev_corner or config. The result is uninformative about noise versus camera frame; it is not evidence for noise.
Route: IDEA-014 (u, v, depth output) next, per D-mixed. IDEA-016 not promoted (not refuted: the probe is weak, as noted in two earlier reviews). IDEA-025/026 still need the pinned-file change requested in `research/outbox/exp007_cf_flip_needs_pinned_change.md`.

## Descriptive observation (not pre-registered, not a result about the model)
`[gtmotion]` shows how GT AT motion projects onto the camera axes: on val_id |y| 0.097 and |z| 0.098 (about equal), on val_ood_config |y| 0.030 and |z| 0.092, on val_diag_elev |y| 0.046 and |z| 0.137, on val_diag_elev_corner |y| 0.035 and |z| 0.098. So the y:z split of an away/toward push in camera coordinates changes with elevation, as the camera-frame story assumes, and the model is not told the elevation. Control (reviewer): val_diag_corner, which has no elevation change, gives |y| 0.076 and |z| 0.074, so the asymmetry is elevation-specific; config and corner AT motion is also smaller overall (0.12 to 0.13 vs 0.19), a separate effect. This is a fact about the data; whether the model's prediction follows it is not tested by this run. Pooled AT signed means cancel (away and toward have opposite sign) and the LR-vs-AT DiD has unequal floors (AT yz 0.089 vs LR 0.038 on val_id), so a better-posed test is IDEA-055 (per-direction, sign-normalised error and per-axis gain on val_id, val_diag_elev and val_diag_corner; no GPU). IDEA-025/026 remain the only test of whether the prompt direction is used.

## Post-review (adopted): the verdict stands
Reviewer: D-mixed and IDEA-014 next are the correct pre-registered application; every table entry and rule call recomputed and matches the log. Wording fixes above applied. C-010 gets an exp007 bullet (status tentative). IDEA-053 done; IDEA-055 and IDEA-056 (calibrate validity limits from val_id before pre-registering) added.

## Caveats
- One adapter, one seed (exp005). About 12 uncorrected 95% CIs; the splits are nested.
- The LR probe check I pre-registered was miscalibrated (0.01 m versus an in-distribution 0.03 m); the invalid-probe call follows the rule as written.
- Oracle-path numbers mix separate Task A and Task B generations (exp004 gate: 0.010 m on config).
