# Pre-review: exp003_exp000_diag_eval

Reviewer, 2026-10-24. Spec at main d00ff59. Worktree `/home/appeon_a_lew/wm-worktrees/exp003_exp000_diag_eval`,
branch `exp/exp003_exp000_diag_eval`, commit c2e2605 (one commit on top of d00ff59, worktree clean).
`check_spec.py` reports ok, no errors.

## Verdict: REVISE (spec text only; the code is approved as is)

Both required fixes are edits to `spec.yaml`. No code change is needed and the code does not need
another review. After the edit, re-run `check_spec.py`, commit in the worktree (the snapshot tar
takes the spec from the worktree commit) and mirror the edit to main. A delta check of the spec diff
is enough for re-approval.

## What was checked

### 1. Byte-identical reproduction (OK)
- `git diff main...HEAD -- physwm/vlm/infer.py` contains additions only: `import re`, the
  `derived` import, `task_a_recs = []`, the per-split `stats` dict, counters, the Task A record list,
  the stats print and the derived-file write. The `max_new` line (85), the `generate` call (86), the
  `parse_answer` call and the `json.dumps` write of normal.jsonl records (95-99, same keys and order)
  are untouched. `generate`, `load_model`, prompts.py, parse.py, pyproject.toml, exec.py, metrics.py
  and configs are unchanged.
- Nothing mutates `objects` after it is written. `ownstatic_records` copies with `dict(...)`, and
  `json.dumps` runs before the append in any case.
- Code path vs exp000: `git log -- physwm/vlm/` shows only 6b3ceba, so infer/prompts/parse are the
  same as when exp000 ran at 345017f. exp000 ran under the pre-d4de761 executor (cwd = MAIN). At that
  commit MAIN's infer.py was identical to the snapshot's, so this makes no difference. The job now
  appends the diag splits to the same `--splits` call. Batching is per split (`work` is rebuilt per
  split), so the batch composition of the four validation splits does not change. The batch size
  stays at 32 (spec has no `batch_size` and no `eval_limit`).
- Remaining drift sources: the box's torch/transformers/CUDA state and GPU kernel nondeterminism.
  The reproduction gate exists to catch these.

### 2. Scorer side (OK, no blocker)
- `evaluate_experiment` keys methods by **filename stem**: `preds[f"{exp}:{f.stem}"] = f`
  (exec.py:304-305). Each file is loaded and summarised on its own, and `load_predictions` and
  `summarize` never read `image_mode`. The `image_mode: "normal"` field in ownstatic records is
  inert. Nothing concatenates files and nothing overwrites another file.
- The primary metric, `val_id_*`, `taskA_val_id_all_mean` and `parse_rate_min` read only
  `{exp}:normal` (exec.py:317-326). `image_dependence_gap` reads only `{exp}:none` (exec.py:330).
  The ownstatic rows cannot touch any of them.
- `_episode_errors` maps `exp:mode` to `pred/{mode}.jsonl` (exec.py:374-375). `ownstatic` matches
  `[a-z]*`, and the code filters `task == task`. `per_object_rows` picks GT time from `pred["task"]`
  (metrics.py:43). The relabelled records are therefore scored against t = 1 s GT, which is what
  own-static needs.
- results.md gains `exp003_exp000_diag_eval:ownstatic` rows, Task B only, on all 7 splits. Their
  `parse_rate` is the Task A parse rate. This is expected and should be stated in the report.
- I confirmed that the pinned static baseline covers the diag splits with Task B. `compare
  exp002_zeroshot_qwen3d_rerun static --splits val_id val_diag_*` returns n per split without
  refusal. C1 and C3 are therefore computable.

### 3. Single-change status (fair)
The model, adapter, prompts, parser, style, seed and batch size are all held fixed. The differences
from exp000's run are:
- the job also scores the diag splits (exec now does this for every job);
- `ablations: [none]` instead of none/blank/swap, which does not affect normal.jsonl;
- the executor runs from the snapshot dir;
- derived/printed instrumentation.

None of these can change a generated answer. "No intentional change + declared instrumentation" is
an accurate description.

### 4. Criteria and computability (two gaps, see required fixes)
- Leakage: none. The job is eval only, with no training, no data generation and no prompt change.
  The diag splits are validation only. ownstatic is built from model outputs only, with no GT.
- Metric gaming: none. ownstatic is a separately named method. It is never mixed into `:normal`.
- Everything in decision_criteria is computable from `agent.sh compare/results/log`. The spec
  routes around the `compare --task A ... static` refusal (static holds Task B only, exec.py:522 and
  384-385): Task A per-split numbers are results.md point estimates, and `compare --task A exp003
  exp000` is allowed because both have Task A. results.md does not show `missing_rate` (it is not in
  `cols`, exec.py:335), so the job.log stats lines are the only source for (b). The spec uses them.
- `agent.sh log` defaults to the last 40 lines. Use `agent.sh log exp003_exp000_diag_eval 400` so
  that the 14 normal-mode stats lines are not cut off by the none pass's load output.
- The expected values check out against exp000's results.md: D_val_id = 0.155 - 0.265 = -0.110,
  D_cfg = 0.269 - 0.241 = +0.028, S_cfg = 0.138. The budget also checks: 6800 samples, worst case
  ~0.55 h vs 1.0 h, ledger 2.45/12 h.

### 5. Final-eval path (does not matter; note only)
`op_final_eval` calls infer with the default `--image-mode normal` and tasks A B, so it would also
write `pred_final/ownstatic.jsonl`. `evaluate_experiment(..., "_final")` would add ownstatic rows to
results_final.md. `metrics_final.json` reads only `:normal` and computes no gap with a tag. No
protected number changes, and a final eval needs a human token anyway. If this infer.py is merged to
main, every future normal run will emit ownstatic.jsonl. That is harmless and arguably useful, but
the merge should be a stated decision, not a side effect.

### 6. The test (OK)
`tests/test_derived.py` sits in `tests/`, so preflight (`pytest -q -x tests`, exec.py:221) collects
it. `derived.py` is stdlib only. On inspection the test passes:
- [a1, b1, a2] yields 2 records in order e1, e2, and b1 is skipped by the `task != "A"` guard;
- objects compare equal and slot "1" stays None;
- metadata is carried through.

The test does not cover the duplicate-episode dedup. That is fine: (split, episode_id) is unique
per run.

## Required fixes

1. **Add val_id to the reproduction gate.** Criterion (ii) uses the default splits, which are the
   OOD splits only, so the normal-mode val_id Task B number is never checked directly. (iii) bounds
   it only indirectly, to about +/-0.02 through gap +/- none-diff. D_val_id is the reference for
   every S_f and R_f, so it must reproduce as tightly as the rest. Replace (ii)'s command with
   `compare exp003_exp000_diag_eval exp000_sft_baseline --splits val_id val_ood_objects
   val_ood_forces val_ood_config`, with every |per_split_diff| <= 0.010, and keep (i) for the
   aggregate. Alternatively add (v): |metrics.json val_id_target_mean - 0.1551| <= 0.005.

2. **Give the hypothesis an explicit verdict mapping.** The hypothesis asserts three things: R_elev
   >= 0.5 (implicated), R_elev is the largest, and R_corner <= 0.25. The falsification clause lists
   only R_elev < 0.5 and R_count/R_corner > R_elev. Two outcomes are left unassigned:
   - R_elev >= 0.5 but not CI-separable from val_id, so not IMPLICATED;
   - R_elev implicated and largest, but corner PARTIAL (0.25 < R_corner < R_elev).

   Both invite post hoc reading. Add a mapping that is evaluated only if reproduction (i)-(iv) and
   the precondition hold:
   - SUPPORTED = outcome (1) or (2) with f = elev, and R_corner <= 0.25.
   - PARTIALLY SUPPORTED = elev implicated and largest, but R_corner > 0.25.
   - FALSIFIED = R_elev < 0.5, or max(R_count, R_corner) > R_elev.
   - INCONCLUSIVE = anything else (e.g. R_elev >= 0.5 but not separable).
   - Precondition fails: no verdict.

## Recommended (not required)

- R_f is a ratio of noisy differences. The per-split CIs on D are roughly +/-0.02-0.03, which gives
  R_f an uncertainty of about +/-0.2-0.3. The "largest R_f" and "R_corner <= 0.25" calls are point
  comparisons. In the report, also give a conservative interval for each R_f from the CI endpoints:
  S_f in [D_f lo - D_id hi, D_f hi - D_id lo] over S_cfg. Flag any call whose interval crosses its
  threshold as "point-estimate only".
- Comparing D_diag against D_val_id is an unpaired cross-split comparison (300 vs 500 different
  episodes). The CI non-overlap rule is conservative, which is acceptable, but the report should
  say so.
- diag_corner changes three config fields (min_gap, region, region_center). It is one conceptual
  factor ("tight corner cluster") but not one parameter. Note this when interpreting R_corner.
