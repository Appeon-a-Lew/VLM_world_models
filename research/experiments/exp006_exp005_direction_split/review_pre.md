# Pre-review exp006_exp005_direction_split (worktree HEAD b362554)

Verdict: **APPROVE** (two non-blocking recommendations below).

## Checks
1. One intentional change: PASS. `git diff main...HEAD` is 3 files. Commit 48bf1bd is the exp004 cherry-pick. Commit b362554 adds `print_direction` and its call, which comes after all derived files are written. The generation call, `normal.jsonl`, `none.jsonl`, prompts, parser, pinned files and pyproject are untouched. No seed, epoch, data-size or prompt change. Bit-exact repro against exp005 (0.3160, per-split diff 0.0 on all 8 splits) is the right validity gate, and the spec says "stack drift, do not route" if it fails.
2. Baseline and budget: PASS. This is an eval_only re-run of the exp005 adapter, 0.23 GPU-h of work inside a 0.75 h budget.
3. Leakage and gaming: PASS. No training happens. The prompt is unchanged. GT t=0 is used only in the privileged oracle-path read-out, which is labelled in every printed line and which the spec requires in the report. No parser change, and missing predictions are skipped and counted (`skipped=`), not defaulted to t=0.
4. numpy: PASS. `numpy` is already in `pyproject.toml` (line 8) and is imported at top level in `physwm/eval/metrics.py`. The import in `derived.py` is lazy inside the functions, so the module is still importable without numpy. `infer.py` now imports numpy at the top, which is safe because torch, transformers and metrics already require it. The spec said "numpy only if already imported there", and derived.py did not import it. The deviation is harmless and documented in the code comment. No new dependency.
5. false_alarm: PASS. The code uses `stratum != "off"` (iso plus cont, GT on table at t=1). This matches the spec's "P(pred_fall | GT on table)", and the unit test covers it. The comment in the code says so.
6. direction-did pairs: PASS. The pairs are (val_id, val_ood_config) and (val_id, val_diag_elev), as in the spec. The code guards on the splits being present. The label groups are left/right to LR and away/toward to AT, per `schema.py` `direction_label`.
7. Bootstrap: PASS.
   - It uses a single `default_rng(0)`, so it is deterministic.
   - Each of the 6 (split, group) cells and the 2 off cells is resampled independently, which is correct for unpaired cross-split differences.
   - `e` and `cross` share one index, so they are resampled jointly.
   - NaN cross values (disp < 0.01) are dropped via `_nanmean`.
   - Empty cells are handled.
   - Percentile CIs are taken after dropping NaN.
   - 1000 resamples on arrays of a few hundred rows run in seconds.
   - The test checks point estimates and that the DiD CI brackets the point estimate.
8. Crash risk after inference: LOW. `oracleanchor_records` and its key assert run before `print_direction`, and the same code path ran on the same data in exp004, so the assert residual risk is unchanged. In `print_direction`, `dir_meta` is filled for every episode in normal mode and `rec["objects"].get(tgt)` uses str keys, which match the `str(tgt)` keys written by gt0. Format strings only meet floats, ints and NaN. `_GROUP.get` tolerates unexpected labels. The remaining unknown is a data-shape surprise that the synthetic tests do not cover.

## Recommendations (non-blocking)
- R1. Wrap the `print_direction(oa, dir_meta)` call in `try/except Exception: traceback.print_exc()`. All artifacts are already on disk, so a read-out bug cannot then fail the job or block metrics.json. This is cheap insurance for a 0.23 GPU-h run on a contended GPU.
- R2. The pre-registered routing leaves gaps. Pre-register them now, before any [direction] line is read.
  - DiD >= 0.02 with a CI lower bound > 0 but cross share < 0.5. Suggested rule: "camera-frame, along-push dominated"; report it and route to the cheaper 025/026.
  - 0.01 <= DiD < 0.02.
  - DiD <= -0.01, where LR rises more than AT.
  - Default for anything not covered: report it and route to 025/026.
  - Note that R-A requires a CI lower bound > 0, while the hypothesis text says falsified if the CI includes 0. These are consistent, but state that R-A not firing does not equal R-B firing.

## Interpretation caveats for the post-review
- `e` is absolute path error. The K-based framing and the shrunken denominator are handled by R-C, which is good.
- Iso LR and AT episodes may differ in GT displacement and in table geometry between splits. A DiD could reflect displacement differences. The report should quote `disp` per cell.
- Several routes are being read at once (4 rules, 2 pairs). Treat anything that lands near a threshold as point-estimate only, as the spec already says.
