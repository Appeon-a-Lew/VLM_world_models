# exp009_exp005_cf_action: review_post

Verdict: ACCEPT R1 (action used on val_id) as tentative, with one pre-registered validity-gate deviation that must be reported, and with narrower wording than the draft.

## 1. Validity gates (checked from job.log / results / compare)
1. Primary 0.3160281875, per_split_diff vs exp005_exp000_repro = 0.0 on objects/forces/config. PASS.
2. Task A val_id all_mean vs normal (paired compare --task A): impulse_0.5 diff 0.0 exactly, impulse_2 diff 0.0 exactly, cf_flip +4.3e-05 [-1.0e-4, 2.0e-4] (0.014995 vs 0.014952). Inside 0.0005. PASS. The flip jitter is the padding effect the spec anticipated (flip changes token count; the impulse rewrites do not).
3. [cfprompt]: cf_flip 500/500, cf_impulse_2 500/500, **cf_impulse_0.5 499/500**. The spec gate says n_changed == n_total for each cf mode, and the table says a gate 1-4 failure is INC regardless of cell. **Formally FAILED for one mode.** The draft did not catch this (it left it to the reviewer). Cause is unverified (most likely an impulse that rounds to itself at 2 decimals, e.g. 0.01 x 0.5 -> "0.01"). Materiality: one of 500 episodes; the rise 0.0556 has CI lower bound 0.0435, i.e. a summed excess of about 21 m over 500 episodes, so a single episode cannot create or remove it; it is also a Task B text edit that was simply not applied, not a leakage. The same rounding makes small impulses off-factor (0.05 -> 0.03 = x0.6). Shown sentences for all three modes are visibly flipped/scaled. My call: R1 holds in substance, but the report must say "gate 3 missed by 1/500 on cf_impulse_0.5; by the letter of the spec this routes INC; judged immaterial for the reason above". Do not retro-fit the gate silently.
4. No FAILED line, parse 1.000, missing-rate lines absent (parse_rate_min 1.0). PASS.
5. Non-sphere n = 263 >= 50. PASS.
6. [cfsign] normal = 1.0000 >= 0.80, valid.

## 2. Is [cfdisp] the spec's median_dhat, and s
Yes. The worktree code (infer.py at ffe8ee228b) prints np.median of dhat and dgt; columns are `n median_dhat median_dgt`. The draft's "first column after n" is the median. Recomputed: s = ln(0.1972/0.0430)/ln 4 = 1.523/1.386 = 1.10 (ratio of medians, as pinned). d(0.5)=0.0430 < d(1)=0.0941 < d(2)=0.1972, ordering holds. alpha 0.5 vs 1 ratio 0.457; 2 vs 1 ratio 2.10 (log-slopes 1.13 and 1.07, so no saturation at 2). Sphere group (not pooled): slope ln(0.5170/0.0910)/ln 4 = 1.25. Caveat: no CI on s; margin to the 0.8 threshold is 0.30 but is unquantified at n = 263. The dhat floor (two separate generations, rounding) biases the 0.5 point up and s down, so the bias is conservative.

## 3. Rule calls
- F-read: E_flip 0.417 >= 0.265; rise +0.2595 [0.2354, 0.2849] > 0.05; frac_pos 0.0121 <= 0.35. FIRES. Agree. (The draft does not give a paired CI of E_flip vs static; not needed by the rule.)
- I-use: s 1.10 >= 0.8; rises +0.0556 [0.0435, 0.0695] and +0.0368 [0.0170, 0.0563], both > 0.03 point and CI lower > 0; ordering holds. FIRES. Agree. Note the alpha=2 CI lower bound (0.017) is below 0.03: the rule requires the point rise > 0.03 and CI lower > 0, which holds, but do not word it as "clearly above 0.03".
- Cell F-read x I-use = R1. Agree, subject to the gate-3 deviation in section 1.
- Not R4/R3/INC by the table. Not retro-fitted.

## 4. Alternative explanations
1. Lexical token lookup rather than direction semantics. Flip swaps tokens, and the model may map "left/right/away/toward" to a displacement sign. This does not undercut "text is used", but it limits what R1 licenses for camera conditioning: the flip is in-distribution camera and not split into LR vs AT. Cheapest discriminator: IDEA-062 (paraphrases) and IDEA-060 (flip on elev/config with cfsign per label).
2. Out-of-range or off-grid impulse text. 0.5 and 2 leave the training range and are quantised. The slope and ordering guard against this, and the near-power-1 behaviour at both ends and the monotone ordering argue for dose-response; however the alpha=2 rise (+0.037) is smaller than alpha=0.5 (+0.056) even though dhat doubles, so mixed explanations are not excluded. Cheapest: IDEA-061 (alpha 0.8/1.25, bootstrap CI on s).
3. GT-moved subset selection. [cfdisp] and [cfsign] are restricted to dgt > 0.05 under the real action, i.e. to strong-push episodes (414/500). Normal frac_pos = 1.000 is therefore on the easiest subset; the slope is for high-impulse episodes, not for all. The set is identical across modes, so the cf comparison is fair, but the 0.012 and 1.10 do not describe weak pushes. The pinned error rows (all 500) agree in direction.
4. dhat = |B - A| mixes two generations, so some of it is Task A/B jitter; and a model that simply emits typical-length displacements along the text axis would show sign flip with fixed size. That is what flip does here (0.0940 vs 0.0941), but the impulse result excludes "fixed size".
5. Exploratory side evidence (print-only, unregistered): with no image, [cfsign] is still 1.0000 (the text alone sets the sign), and on val_ood_forces the normal-pass non-sphere median dhat is 0.215 vs GT 1.257 and sphere 0.958 vs 1.746. Predicted displacement rises with the larger impulses (0.094 -> 0.215, x2.3) but the GT rises x8.6, so magnitude is strongly under-scaled OOD. This suggests the forces gain over static is partly direction + shrunken magnitude, not full impulse conditioning. Do not claim it from this run.

## 5. Image-dependence / parse
Image is normal in all cf modes, so no image-dependence claim. none 0.500, gap 0.3425 unchanged from exp005 (repro). Parse 1.000 in every mode; no parse-rate masquerade.

## 6. Implications
- exp006/007 (LR/AT): R1 removes one cheap objection to the probe on val_id (the action is not ignored), so the nulls there are not explained by "text unused". It does not rescue statistical power (CI half-widths about 0.04 m remain), does not test config/elev where the probe lives, and does not show the AT label maps to the elevation-dependent camera axes. The draft's "reverses the weak probe worry" is too strong; say "removes the action-ignored explanation on val_id".
- C-010: no number changes; update appended. Camera-frame confound (IDEA-050) remains open.
- "the forces gain is not purely a layout prior" (R1 text copied into the report): R1 never tests forces. Supported only as "on val_id the model reads the action"; the forces gain could still be partly a prior. Keep the pre-registered limit.
- Hypothesis (a) and (b) from the spec both hold; falsification band not approached.

## 7. Wording fixes for report.md (not edited)
1. Validity section: replace "reviewer must confirm" lines with the verified facts: cf_impulse_0.5 n_changed 499/500, others 500/500; Task A diffs 0.0 / 0.0 / +4.3e-5; state the gate-3 deviation and the INC-by-letter note.
2. Say the printed [cfdisp] columns are `n median_dhat median_dgt`, so s is the pinned one.
3. Quote the [cfprompt], [cfdisp], [cfsign] lines verbatim and the compare outputs (spec report_must_quote), and put "val_id only; val_ood_forces action-dependence not tested" next to the headline verdict.
4. Replace "reverses the weak-probe worry about LR/AT splits" with "removes the 'action ignored' explanation for the exp006/007 nulls on val_id only; config/elev action reading untested".
5. Replace "the forces gain is not purely a layout prior" with "val_id only; forces not tested".
6. Add that the [cfdisp]/[cfsign] numbers are on the GT-moved subset (414/500 episodes), and that s has no CI.
7. "Flip scores worse than static": fine, but add that the paired CI vs static was not computed.
8. Describe alpha=2 rise as the smallest, CI lower 0.017, not "clearly above 0.03".

## 8. Claims / backlog
- CLAIMS.md: C-012 added (tentative); C-010 update note appended.
- BACKLOG: IDEA-060 (cf on shifted splits, per-label sign), IDEA-061 (dose response, s CI, in-range), IDEA-062 (paraphrase).
