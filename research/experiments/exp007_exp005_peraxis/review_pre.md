# Pre-review exp007_exp005_peraxis, round 2 (worktree HEAD 2724516, main f6ce41b)

Verdict (round 2): **APPROVE**

## Round-1 fixes, verified against spec and diff
1. A-yz over-reading: resolved. did_yz (AT minus LR yz rise) > 0 with CI lower bound > 0 is now required in the hypothesis and in A-yz. If the did_yz CI includes 0 the routing is the cheapest discriminator, not a 016 training run.
2. LR probe: resolved. `gt_motion` and `print_gtmotion` print mean |gt1-gt0| per axis and |y|+|z| over the same iso rows (pred not None) as `axis_readout`. The 0.01 m rule is pre-registered, and an invalid split cannot count towards A-yz.
3. Multiplicity: resolved. The spec states that the splits are nested and that there are about 12 uncorrected CIs. A-yz and B-x need val_diag_elev plus one other split, same sign, CI lower bound > 0 on both. A lone CI > 0 is exploratory, and D is the default if elev_corner disagrees.
4. Thresholds: resolved. yz/2 is printed in `[axis]` and `[axis-base]` lines with the val_id baselines. The spec says the 0.02 threshold is not raised after the fact, and the call is labelled "near noise floor".
5. Falsification wording: resolved. "Falsified" means not A-yz, and C-flat, B-x and D are listed as the sub-cases. B-x fires when x >= 0.02 even if yz < 0.01.
6. Signed means: resolved. `axis_signed_ci` gives bootstrap CIs for signed y and z per split-group, and the spec calls them descriptive unless the CI excludes 0.
7. Cosmetic: resolved. `ap = argparse.ArgumentParser()` appears as unchanged context in the diff. The diff vs main is only derived.py, infer.py and tests/test_derived.py (627 insertions, 0 deletions). The worktree is clean. The worktree spec does not appear in `diff main`, so it is identical to main's. I could not run `cmp` or a diff against the exp006 branch in this environment, so the byte-identity of derived.py and test_derived.py to exp006 apart from the exp007 additions rests on the implementation note and is not independently verified. It is low risk, because those additions are pure functions and prints.
8. Unguarded key-assert: the risk is now stated in the spec as a known unguarded step.

## New-confound and crash check
- Generation, prompts, parse, pinned files, jsonl writers and pyproject.toml are untouched. All jsonl files are written before any new print runs. Nothing new is written to disk.
- `print_direction`, `print_gtmotion` and `print_axis` each run in their own `try/except Exception`. A bug in one prints a `FAILED` line and cannot kill the run or suppress the others. Output is also ordered so the jsonl and the final "wrote ..." line survive.
- `_iso_resid` reshapes to (n,3), so empty groups are safe. `axis_bootstrap` returns `({}, skipped)` when both groups are skipped, and `print_axis` handles that. `axis_signed_ci` skips n < 30. The DiD is dropped when either group is skipped.
- The `np.percentile` call in `axis_bootstrap` runs only when `point` is non-empty, which the `groups` check guarantees.
- The `[axis-did]` skipped-group line carries no privileged tag. That is cosmetic, because it contains no numbers.

## Residual notes, not blocking
- The "yz/2 baseline > 0.02 so near noise floor" label and the 0.01 m probe rule are applied by eye in the report. The report should quote the `[gtmotion]` and `[axis-base]` lines verbatim.
- The reference scale from exp006 mixes cross and along push-axis errors, and the spec already says it is not the same quantity.
- The validity gate (0.3160 +/- 0.005, OOD diff exactly 0.0, strata identical to exp006) is sound. The budget is fine.

Approved to submit.
