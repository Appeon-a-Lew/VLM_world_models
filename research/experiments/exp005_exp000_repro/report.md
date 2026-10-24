# exp005_exp000_repro: report (provisional, pending post-review)

## Hypothesis and change
Re-training exp000 unchanged on the new box (nc-10-1-1, regenerated data) reproduces exp000's numbers within the pre-registered tolerances. **Change: none** (only the id differs). Purpose: (1) check box and data reproducibility; (2) produce the adapter for exp004.

Job: done, 1.41 GPU-h (budget 3 h), commit 90b1cc3. The first submit failed for an environment reason (torchvision missing, fixed by the human); the resubmit used the same spec. Adapter: `/var/tmp/toyi/wm/experiments/exp005_exp000_repro/model/adapter`, confirmed in job.log. exp004's `lora:` points at it.

## Numbers (from `agent.sh results`; exp000 values from its report / STATE)
| quantity | exp005 | exp000 | diff | tolerance | pass |
|---|---|---|---|---|---|
| val OOD target mean (primary) | 0.3160 | 0.3136 | +0.0024 | ±0.010 | yes |
| val_ood_objects Task B target | 0.161 | 0.161 | 0.000 | 0.015 | yes |
| val_ood_forces Task B target | 0.515 | 0.510 | +0.005 | 0.015 | yes |
| val_ood_config Task B target | 0.272 | 0.269 | +0.003 | 0.015 | yes |
| val_id Task B target | 0.157 | 0.155 | +0.002 | 0.015 | yes |
| Task A val_id all_mean | 0.0152 | 0.016 | about -0.001 | 0.005 | yes |
| image_dependence_gap | 0.3425 | 0.331 | +0.0115 | ±0.03 | yes |
| parse_rate_min | 1.000 | 1.000 | – | 1.000 | yes |

Static val OOD is 0.3790 (unchanged). Primary vs static: -0.063 (exp000: -0.065). Per-split ordering is the same as exp000 (objects and forces beat static; config 0.272 vs static 0.241 is worse).

Image-dependence ablations on val_id Task B target (exp005 / exp000, exp000 values from STATE): none 0.500 / 0.486, blank 0.527 / 0.472, swap 0.406 / 0.400. All worse than static (0.265). `none` and `swap` are within 0.015 of exp000. **`blank` differs by +0.055.** This is outside the pre-registered criteria (the criteria only cover the gap), so it is noted rather than counted as a failure.

Task A per-split (exp005): objects 0.023, forces 0.016, config 0.137 (exp000: 0.024 / 0.016 / 0.136). Diag Task A: count 0.021, elev 0.084, corner 0.079 (exp003 on exp000: 0.019 / 0.082 / 0.078). New split val_diag_elev_corner: Task A 0.122, Task B target 0.264 vs static 0.279 and const-vel 0.574.

## Verdict (provisional): REPRODUCED
Every pre-registered criterion passes. The primary metric differs by +0.0024, well inside ±0.010. exp005 is therefore usable as the exp000 stand-in for exp004 and for later exp000-derived diagnostics.

Caveats:
- **One seed.** The box effect and seed noise cannot be separated. The +0.0024 is a single-run noise estimate, not a noise floor. `compare exp005 exp000` is unavailable because exp000's predictions are gone, so no paired CI or significance claim is made.
- The blank ablation moved by +0.055, which is larger than the other differences. Most likely noise in a degenerate-input condition, but not tested.
- exp005 is not bit-exact. exp003 reproduced exp000 bit-exactly, but that was an eval-only run on a fixed adapter.

## Post-review (adopted)
Reviewer: **REPRODUCED, confirmed; tentative** (one same-seed repeat, no paired CI vs exp000). Adopted, with these changes to my wording:
- Do not say "within noise" for the +0.0024. The training path is likely deterministic at a fixed seed, so closeness says little about seed variance (IDEA-043 proposes a seed-1 retrain, about 1.4 GPU-h, for a real floor).
- exp005 vs static: -0.0629 [-0.0725, -0.0535]. Per-split CIs: objects -0.071 [-0.080, -0.062], forces -0.150 [-0.172, -0.126], config +0.031 [+0.017, +0.045]. Same signs and ordering as exp000.
- The blank ablation deserves more than a note, but does not change the verdict. Within exp005, blank minus none on val_id is +0.027 [+0.011, +0.043]; in exp000 blank was below none (-0.014), so the order flipped. The blank median (0.383) equals none's (0.387), so the shift is in the tail. Task A blank barely moved. C-001 stands (all three ablations are worse than static in both runs). Do not cite blank-vs-none or use blank as a precise baseline.
- Claims: C-009 added (tentative); replication notes on C-001 and C-002. IDEA-043 added to the backlog.

## Next
- exp004 (oracle anchor, IDEA-034+035) can go through experimenter prepare and reviewer pre. Its numbers must be read against exp005's, not exp000's recorded ones.
- IDEA-025/026 (counterfactual actions) can use this adapter.
