# Pre-review: exp004_exp000_oracle_anchor

## Round 2 verdict: **APPROVE** (reviewer's verdict, transcribed by the orchestrator because the reviewer's write was blocked)
- Worktree spec.yaml is byte-identical to main's at 58c848f (`cmp`); main is an ancestor of the worktree HEAD (merge 3090867); tree clean.
- Round-1 fixes 1-5 resolved: criteria re-anchored on exp005 (0.3160 ±0.005, compare vs exp005 exact 0.0 over 8 splits, gap 0.3425 ±0.005); exp003 numbers marked as exp000 references, exp005 K_own/C3/G to be computed from `ownstatic` before oracle outcomes are read; 8 splits and 8 strata lines; key-match assertion plus test in 6052307; main merged.
- No new confound: the only change is an assertion in the post-generation derived path.
- Residual risk, not blocking: `gt0` keys are `str(range(n))` and parser keys are `str(o.id)`. Non-contiguous object ids would fire the assertion after inference (GPU time lost, no silent wrong result). A crash there means a key mismatch, not drift.

## Round 1 verdict (superseded): **REVISE** (spec text only; the code was fine to run as is)

## Code check (cherry-pick 56a3432 from c2e2605)
`git diff main...HEAD` touches only physwm/vlm/derived.py, physwm/vlm/infer.py and tests/test_derived.py (+187 lines). The plain `git diff main` also shows deletions of exp005 report/review/CLAIMS/BACKLOG. That is only because the worktree branch is behind main. Rebase or merge main before the final commit so the committed branch carries no stale state. The code diff is unaffected.

- Generation is untouched: the `generate(...)` call, `max_new`, the prompt build, and the `f.write(json.dumps({... "raw": ans ...}))` for normal.jsonl are all unchanged. The new code runs after the write and only reads `objects`, `parsed` and `ep`. normal.jsonl is byte-identical in content and order. Stats lines are print-only. The `s` and `k` names do not leak: the comprehension variables are scoped, and `s = stats[task]` is not reused in a way that matters.
- Strata are computed only in normal mode, using the pinned `metrics.gt_centers_cam` and `MOVED_THRESH`. The target is "off" if not `on_table` at t=1, then "cont", else "iso". This matches the spec.
- Derived files are written only when mode is normal and both A and B are in tasks. The stems are lowercase letters only. The none pass writes no derived file and no strata line.
- Parse keys are `str(o.id)` and gt0 keys are `str(j)`, so oracleanchor slots line up and will not silently become None. If a non-contiguous object id ever occurred, `gt0[j]` indexing by position would misalign. This is low risk, but see fix 4.
- Tests cover oracleanchor, None propagation, hybrid, static and stratum. Preflight is `pytest -x tests`.
- There is no leakage into training, since this is eval_only. The GT-dependent files are clearly labelled privileged and diagnostic, and the primary metric, parse rate and image_dependence_gap read only :normal and :none.

## Required fixes (spec.yaml)
1. **Re-anchor the reproduction criteria on exp005, not exp000/exp003.** The adapter is the exp005 one, so exp004:normal must match exp005:normal. Replace criterion (i) with the exp005 value 0.3160 +/- 0.005 (the exp000 value was 0.3136). Replace (ii) with `compare exp004 exp005_exp000_repro` over all normal splits, with exact 0.0 expected. Without this, (ii) tests exp005's retrain noise rather than the stack. Replace (iv) with the exp005 image_dependence_gap of 0.3425. Criterion (iv) as written (0.331 +/- 0.010) already fails on exp005's recorded number, since |0.3425 - 0.331| = 0.0115. The run would be declared "drifted" before it starts. Criterion (iii), `ownstatic` vs exp003's ownstatic, is not meaningful either: it compares different adapters. Compare against exp005:normal Task A instead. Use `compare exp004:ownstatic` against the exp005 Task A numbers (all_mean/target_mean per split), or drop (iii).
2. **Re-register the numeric anchors that depend on the exp000 adapter.** The own-path ratio 0.116/0.424 = 0.27 comes from exp003 and so from exp000. Exp005 differs: its config Task B target_mean is 0.272 against static 0.241, which is a skill of -0.130. Compute the own-path K for exp005 from exp004:ownstatic before the results are read, and put the exp005 own-path values into the routing_readouts (a) and (d) comparisons. Mark the exp003 values as exp000 references only. The 0.054/0.085 Shapley numbers and the 0.139 S_cfg likewise need an exp005 version, or an explicit note that they are exp000 references. Do this now, before results exist. Then the success criterion and the thresholds (0.75/0.5, gate 0.05 m) stay fixed ahead of the data.
3. **Split count.** The spec says "7 normal splits / 7 strata lines". exp005's results contain 8 normal splits, because val_diag_elev_corner now exists. Confirm the split list that infer/exec will use (I could not read exec.py under the review shell restrictions). Then fix the validity wording, either "8 strata lines" with elev_corner added to the reporting, or an explicit statement that elev_corner is outside S5 but still gets derived files. As written, the validity check could fail on a counting error.
4. Minor: add one test or assertion that `set(gt0[eid]) == set(parsed objects keys)`. Then oracleanchor cannot silently emit None through a key mismatch. Also note that the unmoved_mean gate on config reads results.md for :oracleanchor, so the gate check needs those rows.
5. Rebase the worktree on current main before submission (see above).

## Other judgments
- One intentional change: instrumentation only. No seeds, epochs, prompts or data changed. OK.
- Budget 0.75 h (3.5x margin over the 0.214 h exp003 observed) is realistic. If exp005's box is slower, the resubmit rule is already stated.
- Hypothesis is falsifiable, and thresholds, gate, precondition and merge rule are pre-registered. This is good. Fix 2 keeps it that way.
- Caveat to carry into the report: oracleanchor mixes separate Task A and B generations, so A/B inconsistency biases K down. This is already captured by the gate.
