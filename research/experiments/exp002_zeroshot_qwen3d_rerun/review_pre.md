# exp002_zeroshot_qwen3d_rerun: pre-review

Verdict: **APPROVE**

This re-runs the exp001 spec, which was approved in round 2 (`research/experiments/exp001_zeroshot_qwen3d/review_pre.md`). As the spec asks, this review is a diff check against exp001's round-2 diff, plus the checks that changed since then: the executor fix, the added diagnostic splits, and the budget.

## Diff check
- `git -C ../wm-worktrees/exp002_zeroshot_qwen3d_rerun diff main...HEAD --stat`: only `physwm/vlm/infer.py` (+13/-1). Merge base is `f62f3b3`, which is current main. HEAD is `9e70064`. `spec.yaml` has no diff: the worktree uses main's committed spec unchanged.
- The pre-image blob of `infer.py` is `cd9f64a` in both `351537c..0776d91` (exp001 round 2) and `main...9e70064` (exp002). The hunk therefore applies to the same file.
- Hunk by hunk compared with `git diff 351537c 0776d91 -- physwm/vlm/infer.py`:
  - `import re`: identical.
  - The counter reset per split (`n_ans = n_parsed = n_trunc = n_slots = n_missing = 0`), inside the split loop: identical.
  - `per_obj = 128 if args.style == "qwen3d" else 48` and `max_new = 48 + per_obj * max(...)`: identical.
  - The five counter updates, including the truncation regex `r"\]\s*(```)?$"` on `ans.strip()` and the missing count `sum(v is None ...)` over the pre-filled slots: identical.
  - The pred-record `json.dumps` is untouched: identical.
  - The stats print is the only difference: `"\n[{split}] mode={args.image_mode} answers=..."` with the same three rates, the same formats, and `flush=True`, wrapped over 4 lines instead of 3. This matches the three declared deviations exactly: the leading newline, `mode=`, and the line wrap. Nothing else changed.
- **`center` is byte-identical in behaviour.** For `style=center`, `per_obj=48`, so `max_new = 48 + 48*n` exactly as on main. The same `generate` call runs with the same arguments, and the same record is written. The only new effect is one extra stdout line per split, which is logging. exp000 stays reproducible.
- **The snapshot fix is in effect.** `d4de761` is an ancestor of the merge base `f62f3b3`. On main, `_run` passes `cwd=code` for sft and every infer call (`exec.py:239-246, 261, 269, 272`). `op_final_eval` uses `cwd=code` as well. `op_job` sets `PYTHONPATH=code`, and `code` is the tar snapshot extracted in `op_submit`. With cwd = snapshot, `python -m physwm.vlm.infer` resolves to the snapshot's `infer.py`.
- `check_spec.py` on the worktree spec: ok, no errors.

## Checks that changed since exp001 round 2
- **Splits.** Main's `op_job` now also runs `DIAG_SPLITS` (`val_diag_elev/count/corner`, 300 episodes each, added in `be26128`, after exp001 ran). exp001's log shows only the 4 val splits. This is a change in the harness, not in the model or the prompt. The diag splits are excluded from the primary metric, and the spec treats them as descriptive. They are not a second intentional change.
- **Budget: realistic.** I ground the estimate in exp001's measured rate: 4000 samples in 916 s at a cap of 192 or 288 steps, which gives about 34-38 ms per decode step at batch 32. The worst case has every batch running to the new cap:
  - 3 ID-like val splits: 32 batches × 432 steps each, about 1580 s in total.
  - `val_ood_config`: 32 × 688 steps, about 840 s.
  - `diag_elev` and `diag_corner`: 19 × 432 steps each, about 310 s each.
  - `diag_count` (4-5 objects): 19 × 688 steps, about 500 s.
  - `none` and `swap`: 32 × 432 steps each, about 1050 s together.
  - 3 model loads plus evaluation: about 4 min.

  The total is about **1.34 h worst case**, against the 1.5 h budget. Greedy batches stop at EOS, so the expected time is about 0.6-0.8 h. The margin is thin but sufficient. The ledger charges actual hours. 24 h usage is 1.72 of 12 GPU-h.
- **Leakage: none.** No training and no data generation. `prompts.py`, `parse.py` and `exec.py` are untouched. Inference reads only `val_*` and `val_diag_*` (allowed by `exec.py:399`). No test split is touched.
- **Metric gaming: none.** Missing slots still score 1.0 m. A higher cap only lets answers finish; it does not fill in or default anything. It is declared in the `change` field as instrumentation, as it was in exp001.
- **Falsifiability.** The criteria for (A) and (B) are fixed in the spec before the run. They are carried over word for word from exp001 round 2, plus the `compare --task A` CI as an extra item to report. The validity gate is now stated over `val_id` and `val_ood_*`, with the diag rates reported descriptively. That is consistent with the diag splits not being part of the primary metric. exp001 is context only, not a baseline: correct, since it is invalid as specified.

## Conditions for the report (not blocking submission)
1. **Positive control for the snapshot fix.** job.log must contain `[<split>] mode=normal ...` lines for all 7 normal-pass splits, plus `[val_id] mode=none` and `[val_id] mode=swap`. If any are missing, the run again did not execute the snapshot. It must then be declared invalid before any number is interpreted. This also settles the "IDEA-029 premise" question. A missing line could also mean `op_log` filtering, which only affects lines containing `it/s]` or `s/it]`. The leading `\n` prevents that.
2. The stats lines pool Task A and Task B answers within a split. The validity gate therefore uses the pooled `missing_rate`, as in exp001 round 2. A Task-A-only per-mode missing rate needs IDEA-028, retargeted at exp002's preds.
3. Label the truncation column "truncation_rate (upper bound: non-`]`-terminated)", as required in exp001 round 2.
4. exp000 has no preds for the `val_diag_*` splits, because it ran before they existed. Do not attempt `compare exp002 exp000_sft_baseline --splits val_diag_*`. Report the diag splits against static and descriptively only.
5. Ordering risk: `swap` runs last. If the budget is exceeded, (A) is lost and `evaluate_experiment` does not run. The estimate makes this unlikely. If the job does time out, resubmit with `budget_hours: 2.0` and change nothing else.

## Housekeeping done with this review (CLAIMS maintainer)
- `research/paper/CLAIMS.md`: C-003 is marked **withdrawn (invalid as specified)**. Its original text is kept, with the reason and a link to the STATE HUMAN NOTE.
- `research/experiments/exp001_zeroshot_qwen3d/review_post.md`: an addendum records that the verdicts describe main's harness and not the spec, and that IDEA-029's premise was wrong. The stats lines were never printed.
