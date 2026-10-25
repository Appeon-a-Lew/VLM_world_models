# exp006_exp005_direction_split: review_post

Verdict: STAND (hypothesis not supported; routing to IDEA-025/026 is correct), with the required wording corrections below.

## Verified
- compare vs exp005_exp000_repro: per_split_diff 0.0 and CI [0, 0] on objects, forces, config; metrics.json value 0.3160281875 identical. The re-run is bit-exact. (compare covers the 3 OOD splits; the report says "all 8", which is wrong: fix it. The other splits are covered by identical strata lines and by the val_id numbers.)
- Strata lines: 8 of 8, identical to exp004's (val_id 395/31/74, config 301/121/78, elev 246/21/33, and the rest). Stats lines: 16 normal plus 2 none, parse_rate 1.000 everywhere, missing_rate 0.001 only on config and count Task A (same as before). No FAILED line. All [direction] numbers in the report match the log.

## Rule application
- R-A no, R-B no: agree. R-B needs both Delta_e >= 0.01 and AT is 0.0002.
- R-C yes: agree on the letter. Over-read in the conclusion: both Delta_e CIs are about +-0.03 wide, so "not lost skill" is absence of evidence. Constant-e arithmetic: pooled iso K would be 0.31 from the denominator alone versus the actual 0.22 (id 0.48), so about 2/3 denominator and 1/3 error. For AT the denominator explains all of it (0.476 -> 0.26 predicted, 0.259 actual). For LR it explains about 40% (0.49 -> 0.36 predicted, 0.17 actual). Say this instead.
- G3: point estimate on config only: agree. The stronger observation that was missed: the config DiD upper bound 0.017 is below the +0.02 threshold, so the pre-registered R-A magnitude is excluded at 95%. That is a real falsification, not just "wrong sign".
- R-D: the spec text defines it as drop >= 0.15 with CI lower bound > 0. Under that wording it fires CI-solid (0.141 > 0). The report applied the generic "CI crosses the threshold" clause instead. The two clauses conflict. Report both readings and say the spec is ambiguous. The route (IDEA-049 diagnostic, no training run) is the same either way.

## The val_diag_elev contradiction claim: partly unfair
1. G3 is pre-registered, so quoting it is fine. But "contradicts the camera-frame story" over-reads. The C-010 story is that AT pushes map to elevation-dependent camera y/z. LR pushes have zero GT y/z motion, so their cross-push error is exactly the y/z error. LR cross error on elev rose 0.022 -> 0.050, the largest change in the table. That is consistent with an elevation-dependent frame effect expressed as spurious y/z motion on LR pushes. LR vs AT does not isolate the frame. Only a per-axis error does (IDEA-053).
2. Fragility: the DiD CI edge is -0.0019, n 134/112, a single corroboration split, unpaired bootstrap, several quantities computed (no multiplicity control). val_diag_elev_corner has the reverse ordering (K LR 0.450, AT 0.335; e LR 0.061, AT 0.071). The report mentions this only in a parenthesis. It should be stated as a non-replication: 2 of 3 elevated-camera splits show LR > AT in e (elev, config), one the reverse.
3. The model gets the push direction as text, so the displacement could come from the prompt, not the image. Oracle anchoring then removes most position perception. The test is weak for both signs.

## Other corrections
- "Only the off-table recall and false-alarm rows read the model's own predictions" is wrong. derived.py computes pd = oracleanchor centre - gt0, and the [offtable] lines are labelled privileged. Only the displacement B^ - A^ is the model's; the anchor is GT.
- "I have not defined gain": the spec defines it (mean of pd.ghat / ||disp||). It is informative: config LR 1.26 and AT 1.11 vs id 0.92 / 0.97. The model overshoots on config, where GT displacement shrank, which supports the denominator reading. It is a ratio of noisy means; no CI.
- cross_share is meaningless here, as the report says. Fine.
- R-D "layout prior" reading: recall is 0.06 to 0.17 on objects, count, corner, elev_corner too and 0.42 on elev. The drop is val_id-vs-most-others, not config-specific (IDEA-054). False-alarm 0.0047 -> 0.0213 is 2 vs about 9 episodes: ignore.
- The report notes that the try/except guard and G1-G3 were added after the pre-review. Correct to flag; G1-G3 were written before the direction lines were read (git history: bf25b72 precedes the run) and are not gaming.

## Alternatives for the null on config
1. Underpowered: Delta_e CIs +-0.03 on 140-160 episodes per cell. Cheapest: pool the 4 non-id OOD-camera splits or add seeds (IDEA-043).
2. Task A/B perception-difference noise (B^ - A^ from separate generations) grows off-distribution and swamps direction effects. Cheapest: IDEA-053 per-axis, plus a Task-B-only displacement readout.
3. Direction given as text, so AT/LR is not a probe of image-frame mapping. Cheapest: IDEA-025/026 counterfactual actions, already routed.

## Claims
C-010 updated (not overwritten): camera-frame prediction not supported on config, with CIs; mixed elev evidence; recall caveat. No new claim: no finding clears the "CI excludes 0 and consistent across splits" bar.

## Fixes to apply to report.md
(a) "all 8 splits" -> the 3 OOD splits in compare. (b) off-table "model's own predictions" wording. (c) add the DiD upper-bound-below-0.02 statement. (d) rewrite the elev paragraph per above and cite elev_corner. (e) R-C: add the 2/3 vs 1/3 decomposition. (f) R-D: state both readings. (g) define gain.
