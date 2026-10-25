# exp006_exp005_direction_split: report (provisional, pending post-review)

**Privileged diagnostic, uses GT t=0; not a model result.** This applies to every `e`, `K`, `along`, `cross` and DiD number below. They come from the oracle-anchored path (GT t=0 plus the model's predicted displacement). The off-table recall and false-alarm rows are privileged too (reviewer correction: `derived.py` computes the displacement from the oracle-anchored prediction; I had wrongly called them the model's own predictions). Task B only. LR = left/right pushes, AT = away/toward pushes (camera-relative).

## Hypothesis and change
Hypothesis: from val_id to val_ood_config the iso (isolated-push) oracle-path absolute error `e` rises more for AT pushes than for LR pushes (DiD = Delta_e_AT - Delta_e_LR >= 0.02 m, bootstrap CI lower bound > 0, cross-push share of Delta_e_AT >= 0.5), which would point at camera-frame mapping (IDEA-016).
Change: none to model or prompts. Eval-only re-run of the exp005 adapter with exp004's code plus print-only direction / off-table lines (IDEA-050 + IDEA-052). Job done in 0.231 GPU-h, commit 75ebbfd. The pre-review was APPROVE at b362554; the try/except guard (a93dfa4, the reviewer's recommendation) and the pre-registered routing gaps G1-G3 in the spec were added afterwards and not re-reviewed.

## Validity
- metrics.json equals exp005's: primary 0.3160281875, gap 0.34253104, parse 1.000. `compare exp006 exp005_exp000_repro` gives per_split_diff exactly 0.0 with CI [0, 0] (reviewer verified on the 3 OOD splits; I ran it on all 8 and saw 0.0 on each). So the re-run is bit-exact and the determinism check passes.
- No `[direction] FAILED` line in the log. Every group has >= 30 iso episodes: val_id LR 197 / AT 198, config LR 143 / AT 158, val_diag_elev LR 134 / AT 112.
- Reviewer checked the job.log: 8 of 8 strata lines (identical to exp004's), 16 normal plus 2 none stats lines, parse rate 1.000 everywhere.

## Results (iso stratum, oracle path; e in m)
| split | group | n | disp | e | K | along | cross |
|---|---|---|---|---|---|---|---|
| val_id | LR | 197 | 0.1318 | 0.0672 | 0.490 | 0.0587 | 0.0217 |
| val_id | AT | 198 | 0.1430 | 0.0750 | 0.476 | 0.0644 | 0.0261 |
| val_ood_config | LR | 143 | 0.1052 | 0.0870 | 0.173 | 0.0680 | 0.0391 |
| val_ood_config | AT | 158 | 0.1015 | 0.0752 | 0.259 | 0.0585 | 0.0352 |
| val_diag_elev | LR | 134 | 0.1426 | 0.1086 | 0.239 | 0.0847 | 0.0499 |
| val_diag_elev | AT | 112 | 0.1482 | 0.0815 | 0.450 | 0.0674 | 0.0351 |

(The log also has count, corner, elev_corner and objects and forces; all LR/AT iso K on count, corner and objects are 0.39 to 0.47. On val_diag_elev_corner LR 0.450 vs AT 0.335.) The `gain` column is defined in the spec (I wrongly said it was undefined). Config gain is 1.26 (LR) and 1.11 (AT) against 0.92 and 0.97 on val_id: the model overshoots where the true displacement shrank, which supports the denominator reading.

### Pre-registered pairs (job.log `[direction-did]`; bootstrap CIs)
| pair | Delta_e_LR | Delta_e_AT | DiD | recall drop |
|---|---|---|---|---|
| val_id -> val_ood_config | +0.0198 [-0.0075, 0.0510] | +0.0002 [-0.0194, 0.0203] | -0.0196 [-0.0571, 0.0172] | 0.2616 [0.1414, 0.3818] |
| val_id -> val_diag_elev | +0.0413 [0.0103, 0.0735] | +0.0065 [-0.0118, 0.0233] | -0.0349 [-0.0706, -0.0019] | -0.0729 [-0.2785, 0.1365] |

cross_share_AT printed as 41.35 [-11.99, 12.94] on config and 1.40 on elev. That is meaningless: its denominator, Delta_e_AT, is about 0.0002 (config) and CI-indistinguishable from 0. I do not read it.

### Off-table (model's own predictions)
| split | n_off | recall | false_alarm | pred/gt disp |
|---|---|---|---|---|
| val_id | 74 | 0.351 | 0.0047 | 0.535 |
| val_ood_config | 78 | 0.090 | 0.0213 | 0.262 |
| val_diag_elev | 33 | 0.424 | 0.0225 | 0.633 |

Exploratory, no CI and not pre-registered: recall on the other splits is objects 0.063, forces 0.248, count 0.074, corner 0.167, elev_corner 0.121.

## Which rules fired (spec `pre_registered_routing`; point estimates, then CI)
- **R-A (camera-frame): no.** DiD on config is -0.0196 (needs >= +0.02).
- **R-B (generic): no.** It needs both Delta_e >= 0.01; Delta_e_AT is 0.0002.
- **R-C (denominator): yes on the letter.** Both Delta_e CIs include 0 on config, but they are about ±0.03 wide, so the conclusion is weaker than "denominator only". The reviewer's rough constant-error arithmetic on the iso means puts about two thirds of the iso K drop (0.48 to 0.22) on the denominator and about one third (about +0.010 m real error rise) unresolved by the CIs.
- **Stronger falsification of R-A (reviewer):** the config DiD upper bound (0.017) is below the +0.02 threshold, so R-A's required magnitude is excluded at 95%.
- **G3 (DiD <= -0.01, LR rises more): yes on the point estimate.** On config the CI [-0.057, 0.017] includes 0, so it is point-estimate only. On val_diag_elev DiD is -0.0349 with a CI that excludes 0 (upper end -0.0019): LR error rises, AT does not. I first called this a contradiction of the camera-frame story; the reviewer says that is too strong. LR pushes have zero true y/z camera motion, so their cross-push error is the y/z error, and it rose from 0.022 to 0.050 (the largest change in the table). That is consistent with an elevation effect showing up as spurious y/z motion on LR pushes. LR vs AT cannot separate a frame effect from other causes; a per-axis error read-out can (IDEA-053). The result is fragile: one split, a CI edge at -0.002, no multiplicity control, and val_diag_elev_corner shows the reverse ordering (K 0.450 LR vs 0.335 AT), a non-replication. The push direction is also given to the model as text, so the LR/AT split is a weak probe either way.
- **R-D (off-table): yes on the point estimate.** Recall drops from 0.351 to 0.090 (drop 0.262, CI [0.141, 0.382]). The spec reads two ways: its R-D text (drop >= 0.15 with CI lower bound > 0) fires, while the generic "CI crosses the threshold" clause makes it point-estimate only (lower bound 0.141 < 0.15). Both readings are reported; the route is the same. It is also not config-specific: recall is 0.06 to 0.17 on objects, count and corner too, and only val_diag_elev is high (0.42) (IDEA-054). The off-table false-alarm rate also rises (0.0047 to 0.0213).

## Verdict (provisional)
The hypothesis is **not supported**: DiD on config has the wrong sign and a CI around zero, and the one corroboration split (val_diag_elev) shows the opposite, CI-solid. IDEA-016 is not promoted by this result. The config loss on isolated pushes is not localised to away/toward pushes.
Routing per the pre-registered rules: every call is point-estimate only or falls in the "other combination" bucket, so the next spec is **IDEA-025/026** (counterfactual actions on the exp005 adapter, about 0.15 GPU-h each). R-D also names IDEA-049 (state-only ceiling) as the next diagnostic for the off-table half.

## Post-review (adopted): the verdict stands
Reviewer: hypothesis not supported; routing to IDEA-025/026 is right; the seven wording fixes above are applied. Alternatives to the config null (review_post.md): underpowered (about 150 episodes per cell; pool splits or add seeds), perception noise from the separate Task A and B generations (per-axis read-out, IDEA-053), and text direction making LR/AT a weak probe of image-frame mapping (IDEA-025/026). C-010's caveats now include the exp006 evidence; no new claim (nothing clears CI-excludes-0 plus cross-split consistency).

## Caveats
- One adapter, one seed (exp005); no seed-noise floor (IDEA-043).
- Oracle-path numbers mix separate Task A and B generations (exp004 gate: 0.010 m on config).
- DiD can reflect GT displacement differences between cells; `disp` is quoted above per cell (LR vs AT disp differ by 0.01 to 0.02 m on most splits).
- The recall table beyond id/config/elev is descriptive only.
