# Post-review exp008_uvz_output

**Verdict: hypothesis NOT SUPPORTED; uvz is slightly WORSE than exp005 (small, CI-backed, one seed). Routing: inconclusive per spec (R3 and R4 letters both hold); default action applies (do not adopt uvz; go to IDEA-002 or IDEA-025/026). Report numbers verified.**

## Verification (via `compare`)
- Primary 0.3264 vs exp005 0.3160: +0.0104 [0.0041, 0.0165]; verdict "b better". Matches the report. vs static -0.0526 [-0.0621, -0.0429]. Matches.
- Per split Task B vs exp005: objects +0.0091 [0.0014, 0.0166] (b better), forces +0.0095 [-0.0058, 0.0245] (not significant), config +0.0125 [0.0047, 0.0201] (b better). The report's CIs are right. The aggregate worsening is consistent in sign across the three splits, but only objects and config carry per-split CIs. Do not say "forces worse". On the other splits: val_id +0.0172 [0.0042, 0.0294], elev_corner +0.0196 [0.0094, 0.0306], corner +0.0168 [0.0010, 0.0330], elev +0.0046 [-0.011, 0.018], count +0.0021 [-0.003, 0.007].
- vs static per split: config +0.0439 [0.0302, 0.0576] (still worse than static), objects and forces better. Same as exp005's pattern (C-002).
- Task A vs exp005 (`--task A`): config +0.0081 [0.0058, 0.0102], elev +0.0064 [0.0043, 0.0085], corner +0.0148 [0.0121, 0.0176], elev_corner +0.0152 [0.0125, 0.0178], val_id +0.0002 [-0.0006, 0.0010]. So the report's "no Task A gain" understates it: Task A is significantly worse on every shifted split and equal on val_id. objects -0.0026 [-0.0034, -0.0018] (uvz slightly better, 0.002 m, immaterial), forces and count flat. The in-distribution parity says the codec is fine; the shifted-camera degradation says uvz extrapolates worse across viewpoint.
- Image-dependence gap: none-normal 0.378 [0.346, 0.409] vs exp005 0.343 [0.310, 0.376]. Difference 0.035 is under the pre-registered 0.05 non-interpretation margin and the CIs overlap. Gap in [0.30, 0.40]: similar image use. swap-normal 0.226 vs 0.249: not interpreted. Both ablations (0.553, 0.400) are worse than static 0.265, C-001 retained. No shortcut flag: the model did not improve while relying less on the image (it got slightly worse and relies on it slightly more or equally).

## Gate
- parse_rate_min 1.000 on all 8 normal splits (results.md): pass. **Missing rate per split cannot be verified**: `agent.sh log` shows only the tail of job.log with no missing line, and the pred files are on nc-10-4-3, not readable from here. Print-only Task A "n=" is 989 on val_id, below 1000 objects (500 episodes), but I do not know the object count per episode, so I can not tell whether 11 objects (1.1%) were dropped as missing (z<=0 or unparsed) or the episodes just have fewer objects. If it is a drop, the 0.01 limit is exceeded on the letter. The report's "gate passed" should read "parse rate and Task A/B val_id limits pass; per-split missing rate not verified". Cheap check: the infer stats line on the executor, or compare n to GT object counts (a CPU job, IDEA-058).
- val_id Task A 0.0154 <= 0.030 and Task B 0.174 <= 0.200: pass.

## Rule calls
- Primary 0.3264 > 0.321 ("worse", CI excludes 0), config Task A 0.145 > 0.120: the spec's inconclusive case "R3 and R4 overlap". The report's call (inconclusive, both readings, default action) is correct. R1 and R2 are excluded. Mechanism check fails. Guards hold (forces 0.524 <= 0.545, objects within +0.03).
- R3 reading (output parameterisation is not the cause) is supported; R4's mechanism (dynamics answer leaves image) is untested because the off-image fraction was not computed. The report's sentence "the representation is not a lever for the config perception error" is the R3 reading and is fine; "drop IDEA-014" (R4) should not be asserted.

## Waived pre-submit check
It does not undermine validity of the headline numbers. The two things it guarded: (a) the round trip encode/decode: val_id Task A 0.0154 vs 0.0152 and median |du| 0.4 px, |dz| 0.012 m show that a codec bug would have to be invisible in-distribution, which is implausible; a systematic bias would have shown on val_id. (b) center-style byte identity: this matters only for the exp005 comparison being like-for-like in code, and exp005 predictions were not rerun on this commit. The center path was reviewed pre-hoc and tests cover it on mocked GT. Residual risk: a real-episode K or W/H edge case (e.g. non-square images, cx/cy rounding) that appears only off-distribution. The config split uses different cameras, so a K-specific decode bug could in principle contribute to the shifted-split worsening. Evidence against: the 2D error stays about 1 px on config, which a K bug would break. So I consider it low risk but the waiver should be stated as a caveat on the headline, not only in a Deviation section. Also note: the in-distribution Task B val_id is worse by +0.017 [0.004, 0.029], which a decode bug would not explain but which the FRAME removal or the format could.

## Descriptive Task A claim (depth vs 2D)
Numbers copied correctly from job.log (checked all 8 lines). Issues:
1. It is a claim about uvz outputs only. exp005 has no printed per-axis error, so it is not known whether the center format has the same depth-dominated split. The report's step "this points at IDEA-015 and depth ideas rather than camera conditioning" over-reaches: a viewpoint change moves depth and pixel position differently in the image, and camera conditioning (K/pose text) is exactly what would fix a depth bias. A depth-dominated error is compatible with camera-elevation effects (depth along the view direction changes systematically with elevation for the same table layout). It does not discriminate against camera conditioning.
2. Medians of |dz| in metres versus medians of |du| in pixels are not commensurate; 1 px at z~1.3 m, fx~520 is about 2.5 mm, so lateral is about 0.003 m against 0.128 m depth. That makes the depth dominance real, and it matches the Task A all_mean 0.145 (3D). Suggest quoting the lateral error in metres.
3. No signed error, no CI, n counts objects. diag count and forces unchanged confirms viewpoint dependence of the depth error (elev 0.078, corner 0.082, elev_corner 0.120 roughly additive).
Wording: "under uvz, on val_ood_config and the elevation/corner diagnostics, the median depth error is 6-10x val_id while 2D is about 1 px" is supported; "the config/elevation perception error is largely depth" is supported for uvz as a description, not as a causal statement and not for exp005.

## Off-image fraction
Cannot be derived from existing logs: job.log prints only the Task A medians and loss lines. GT fraction can be computed from the dataset (K, centers) and pred fraction from pred/*.jsonl on the executor, both need a CPU/eval_only job (IDEA-058). Weak indirect evidence against R4's mechanism: val_id Task A 2D error is about 0.5 px and Task B worsening is also seen on non-shifted val_id (+0.017), and objects/count with the same camera are near flat; but a Task B off-image story is untested.

## Alternative explanations for primary +0.010 and Task A degradation on shifted views
1. FRAME-sentence removal (the camera-frame cue) hurts viewpoint generalisation; format irrelevant. Cheapest test: IDEA-059 (center, no FRAME).
2. Depth regression extrapolates worse than xyz: under camera elevation the depth range of objects shifts, and z as a free scalar regresses to the training depth range (shrinkage), whereas camera y/z in xyz entangle depth with height. Test: signed dz vs GT z; IDEA-057 on exp005 per-axis error.
3. Task B dynamics: pixel-space displacement under a push is non-linear in depth and perspective, so the same physical push gives different (u, v) deltas on the tilted camera; the model must learn a depth-dependent mapping. Test: error of Task B in metres vs in pixels on val_id vs config; off-image fraction.
4. Seed noise: the repro noise is 0.0024 and one-seed variation is unknown; +0.010 is about 4x the repro delta but not a seed estimate. Test: a second seed (IDEA-043); the Task A shifted-split degradation, which has tight CIs on the same 500/300 episodes, is the more stable evidence but also single adapter.
5. Quantisation or decode error near the image border/z extremes is a small contributor only (pre-review computed 6e-5 m).

## Wording fixes for report.md (not edited)
1. Validity: replace "Gate passed" with "parse rate, val_id Task A and Task B limits pass; missing rate per split not verified (cannot be read from job.log)".
2. Verdict line: replace "no Task A gain" with "Task A is worse on every shifted split (config +0.0081 [0.0058, 0.0102], elev +0.0064, corner +0.0148, elev_corner +0.0152) and equal on val_id".
3. State per-split Task B CIs: forces is not significant; objects, config, val_id, corner, elev_corner are worse. Currently the table gives them but the verdict says nothing on the sign consistency.
4. Gap: add CIs (0.378 [0.346, 0.409] vs 0.343 [0.310, 0.376]); difference 0.035 is under the 0.05 margin and is not interpreted.
5. Descriptive finding: add "uvz only, no exp005 comparison", quote lateral error in metres, and drop "points at IDEA-015 ... rather than camera conditioning"; say it does not discriminate against camera conditioning.
6. Say "drop IDEA-014" only conditionally; R4 is not confirmed without the off-image fraction.
7. Move the waived-check caveat into the Verdict, and add "exp005 predictions are from commit 90b1cc3, not 804614437a".
8. "0.0024 reproduction noise" is a two-run repro on aggregates, not a seed floor; phrase accordingly.

## Backlog and claims
Added IDEA-057 (exp005 per-axis Task A), IDEA-058 (off-image/missing post hoc), IDEA-059 (FRAME-removed center control) in research/ideas/BACKLOG.md. Added C-011 (tentative) to research/paper/CLAIMS.md.
