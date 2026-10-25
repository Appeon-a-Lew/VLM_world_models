# exp008_uvz_output: report (provisional, pending post-review)

## Hypothesis and change
(IDEA-014) Same LoRA SFT, answers written as `[u, v, z]` (pixel centre on a 0-1000 grid plus depth in metres), decoded to camera xyz with the per-episode K. Prediction: Task A val_ood_config <= 0.100 (exp005 0.137) and primary (val OOD Task B target mean) <= 0.311 (exp005 0.3160 minus 0.005), CI vs exp005 excluding 0. Done: 1.51 GPU-h, commit 804614437a, train run, seed 0, one adapter. Bundled change (spec confound_note): the FRAME sentence was removed with the format change.

## Deviation from the spec (stated first)
The spec's real-episode pre-submit check (50 train episodes uvz round trip within 0.002 m; 5 center-style episodes byte-identical to main) was **not run**: no `agent.sh` operation runs custom code. The orchestrator waived it; only the synthetic tests in `tests/test_uvz.py` ran in preflight. Indirect evidence the encode/decode works: val_id Task A median |du| 0.4 px, |dv| 0.5 px, |dz| 0.012 m and val_id Task A all_mean 0.0154 (exp005 0.0152). It does not cover the center-style byte-identity part (exp005 was not rerun on this commit).

## Validity (spec gate)
- parse_rate_min 1.000 (gate >= 0.99). Missing rate per split is **not verified** (not readable from job.log; pred files are on the executor; IDEA-058).
- val_id Task A all_mean 0.0154 (gate <= 0.030): pass. val_id Task B target 0.174 (gate <= 0.200): pass. So: parse rate, val_id Task A and Task B limits pass; per-split missing rate not verified.
- **Off-image fraction (spec report_must_quote): not available.** It is not printed in job.log and I cannot compute it from pred files in-loop. Open item; matters only for route R4's "u/v outside 0-1000" reading.

## Results
| | exp008 | exp005 | diff (CI) |
|---|---|---|---|
| primary val OOD Task B target | 0.3264 | 0.3160 | +0.0104 [0.0041, 0.0165] (exp005 better) |
| per split objects / forces / config | 0.170 / 0.524 / 0.285 | 0.161 / 0.515 / 0.272 | +0.0091 [0.0014, 0.0166] / +0.0095 [-0.0058, 0.0245] / +0.0125 [0.0047, 0.0201] |
| vs static (0.3790) | -0.0526 [-0.0621, -0.0429] | | |
| val_id Task B target | 0.174 | 0.157 | |
| val_id Task A all_mean | 0.0154 | 0.0152 | |
| Task A val_ood_config | 0.145 | 0.137 | |
| Task A diag elev / count / corner / elev_corner | 0.090 / 0.020 / 0.093 / 0.137 | 0.084 / 0.021 / 0.079 / 0.122 | |
| Task A objects / forces | 0.021 / 0.016 | 0.023 / 0.016 | |

Image dependence (val_id Task B target): normal 0.174, none 0.553, swap 0.400. Gap none-normal 0.378 [0.346, 0.409] (exp005 0.343 [0.310, 0.376]; difference 0.035 is under the 0.05 margin, not interpreted); swap-normal 0.226 (exp005 0.249). Both none and swap are worse than static 0.265 (C-001 retained). Caveat from spec: none-mode differs by prompt text, so a gap change under 0.05 is not interpreted.

Print-only Task A line (median |du| px / |dv| px / |dz| m vs GT): val_id 0.4/0.5/0.012; objects 0.5/1.1/0.014; forces 0.4/0.5/0.012; count 0.4/0.6/0.014; config 0.8/1.2/**0.128**; elev 0.5/0.7/**0.078**; corner 0.4/0.6/**0.082**; elev_corner 0.6/0.8/**0.120**. none: 85.2/29.2/0.105; swap: 46.2/36.3/0.153.

## Which rule fired (pre-registered)
- Primary 0.3264 > 0.321: "worse". Task A config 0.145 > 0.120. Mechanism check (config Task A <= 0.100) fails; guards: forces 0.524 <= 0.545 and objects 0.170 within +0.03 of 0.161, both hold.
- R1 no, R2 no (config Task A not <= 0.100). The spec lists "primary > 0.321 with config Task A > 0.120" as an **inconclusive** case: R3 and R4 overlap, report both readings. R4's own condition ("validity fails, or primary > 0.321 with val_id Task A fine") holds on the letter (val_id Task A fine). R3's ("config Task A > 0.120 and primary comparable or worse") holds on the letter too.
- Default action (spec): do not adopt uvz as the new base; go to IDEA-002, or to IDEA-025/026 if the dynamics half looks like the bottleneck.

## Verdict (provisional)
Hypothesis **not supported**: Task A is worse on every shifted split (config +0.0081 [0.0058, 0.0102], elev +0.0064, corner +0.0148, elev_corner +0.0152; post-review `compare --task A`) and equal on val_id; the primary is slightly worse (+0.010, CI excludes 0; one seed; the 0.0024 exp000-vs-exp005 difference is a two-run aggregate repro, not a seed floor). Task B per split vs exp005: worse on objects, config, val_id (+0.017), corner, elev_corner; forces not significant. The waived pre-submit check and the unverified per-split missing rate and off-image fraction are caveats on this headline; exp005 predictions come from commit 90b1cc3, not 804614437a. Dropping IDEA-014 is conditional on the off-image fraction (IDEA-058); R4's mechanism is untested. Verdict wording: **inconclusive per spec routing (R3 and R4 readings overlap)**; the representation is not a lever for the config perception error.
Descriptive finding, not pre-registered: under uvz the 2D part of the Task A error is small everywhere including config (median 0.8 px / 1.2 px) while the depth error carries the shift (median |dz| 0.012 m on val_id to 0.128 m on config, 0.120 on elev_corner, 0.078 elev, 0.082 corner; count and forces unchanged). About uvz outputs only (no exp005 comparison); 1 px is about 0.003 m laterally (z~1.3 m, fx~520), so the shifted-view error is largely depth. A description, not a cause, and it does not discriminate against camera conditioning (a viewpoint change moves depth systematically). Medians only; no CIs; counts are objects not episodes. IDEA-057 compares exp005's per-axis Task A error.

## Caveats
- One adapter, one seed; no seed-noise floor (IDEA-043).
- FRAME sentence removal bundled with the format change.
- Real-episode pre-submit check waived (above); off-image fraction not reported.
- exp005 not rerun on commit 804614437a; the comparison uses exp005's stored predictions.
