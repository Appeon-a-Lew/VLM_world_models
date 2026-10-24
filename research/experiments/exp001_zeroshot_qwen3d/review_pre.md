# exp001_zeroshot_qwen3d: pre-review

Verdict: **REVISE**

The experiment is sound in principle: eval-only, no training, no leakage, and the code path already exists. In its current form, though, the measurement instrument (token cap) is biased against the very thing being tested. The hypothesis also has no fixed numeric criterion, and its Task A half has no defined reference. All fixes are small.

## What I checked
- Diff vs main (6a699fd): empty. The commit 351537c only adds the spec. Inference runs from the snapshot (`env_code` PYTHONPATH=code, `physwm/agent/exec.py:250,260`). Scoring runs from pinned main.
- Code path for `style: qwen3d`:
  - prompt: `physwm/vlm/prompts.py:48-53`
  - parser: `physwm/vlm/parse.py`; the regex accepts bbox-first and label-first, and JSON is tried first
  - covered by `tests/test_parse.py:22`
- **Leakage: none.**
  - Nothing trains.
  - Inference touches only the `val_*` splits plus the `val_id` ablation. The policy allows this.
  - The prompt contains intrinsics, object names, and the action text. The action text gives direction and impulse only, with no mass or speed (`physwm/sim/push.py:157-160`), and is identical to exp000.
  - The prompt contains no GT positions or sizes.
- **Metric gaming: none.** Missing objects score `MISSING_PENALTY` = 1.0 m (`physwm/eval/metrics.py:27,49`), so every bias described below makes the zero-shot model look *worse*, never better.
- Split sizes: each val split has 500 episodes (exp000 `results.md`). The full normal pass is 4000 samples (A+B). For exp000 it took 482 s at about 30 tokens per object.

## The experimenter's four flagged risks

### 1. Truncation (max_new_tokens): real, and it needs fixing. **Required.**
`max_new = 48 + 48 * max_objects_in_batch` (`physwm/vlm/infer.py:81`). Qwen's tokenizer splits digits individually, so one bbox_3d entry costs about 60-65 tokens, or about 75-85 if pretty-printed. That leaves headroom of about +23 for 1 object and about −7 for 3, so `val_id`, `val_ood_objects` and `val_ood_forces` are borderline whenever a batch holds a 3-object episode (count ∈ [1,3]). `val_ood_config` has 4-5 objects (`configs/sim_ood_config.yaml`), giving a budget of 240/288 tokens against a need of about 260/325. **Nearly every config sample will lose its last 1-2 entries.**

The fallback regex can only pair a bbox with a *following* label. A truncated final entry is therefore dropped, scored at 1.0 m, and still counted as `parsed=True`, because `parse_rate` is True as soon as any one object parses (`parse.py:63`).

On `val_ood_config` the scoring is roughly: ~1/n of target objects get exactly 1.0 m. This systematically inflates `val_ood_config` and the primary metric. It would turn the Task B "worse than static" half of the hypothesis into a harness artefact rather than a model property. The cap is *experiment* code, not pinned (`infer.py` runs from the snapshot), so it can be fixed in this worktree.

`results.md` does not show `missing_rate` (`exec.py:330`; already flagged in the exp000 review), and pred files are not readable through `agent.sh`. **Truncation would therefore be invisible.**

### 2. Budget / eval_limit: do **not** set eval_limit; raise budget to 1.5 h
- `ZS_LIMIT=500` in `scripts/run_phase1.sh` applied to the larger test splits. On the 500-episode val splits it would be a no-op.
- **Any `eval_limit` < 500 would corrupt the comparison.** `_target_errors` (`exec.py:371`) scores every val episode that has no prediction as `{"objects": {}}`, which gives 1.0 m. `compare` would then penalise the episodes that were never run. `results.md` would also compare a subset (exp001) against full-split static rows.
- Runtime estimate:
  - exp000's measured rate is about 3.9 s per 32-sample batch at about 100 decode steps, i.e. about 35 ms/step including prefill.
  - With the raised cap from fix 1 (128 tokens per object), the worst case has every batch running to the cap: ID-like splits at 432 steps × 31 batches × 3 splits ≈ 13 min, and config at 688 steps × 31 batches ≈ 12 min.
  - The `none` and `swap` ablations add about 7 min each, and three model loads about 3 min.
  - Total: about 45 min worst case, about 20-25 min expected, since greedy batches stop at EOS.
- 1 h leaves too little margin in the worst case, and a timeout counts as a failed experiment. The ledger charges actual hours, so `budget_hours: 1.5` costs nothing extra if the run is fast.

### 3. Ablations [none] only: add `swap`. **Required.**
`none` is mandatory, because `image_dependence_gap` reads `exp:none` and a NaN would abort `evaluate_experiment`. For a *zero-shot* model, though, `none` is a weak reference: dropping the image changes the input modality. A text-only prompt may make the base model refuse or produce garbage, which scores 1.0 m per object and inflates the reference.

`swap` keeps the input distribution identical: a real scene image with the same object count (`prompts.py:91-108`). It removes only the episode-specific information. **It is the correct "trivial" reference for the Task A claim** (see fix 3), and truncation affects it equally, so the comparison stays fair. Cost is about 5-7 min. `blank` is optional.

### 4. Task A reference and numeric criteria: both missing. **Required.**
`static` is GT at t=0, so for Task A it would have error 0 and is not a valid reference. The pinned tooling also has no Task A bootstrap: `compare` covers only Task B target on val OOD. The Task A criterion therefore has to be a fixed margin that is large relative to plausible noise.

A second problem: the Task B half ("worse than static") is close to guaranteed by construction. Static has oracle perception (`baselines.py:26`; exp000 review point 3). If zero-shot Task A error is around 0.2 m, Task B loses to static even when the dynamics are perfect. "Worse than static" therefore cannot support "cannot predict push outcomes". It has to be stated as a descriptive result, and the dynamics wording dropped unless a perception-matched reference exists (optional item below).

## Other checks
- **One intentional change.** Against the declared baseline (static) the change is "zero-shot base model + native prompt". That is a single condition, so this is fine. Against exp000 there are **two** differences, no training and a different answer style. The report must not attribute any exp000-vs-exp001 difference to either one alone. The token-cap and logging fixes below are instrumentation. They do not count as a second change, provided `center`-style behaviour stays byte-identical, so that exp000 remains reproducible.
- **Baseline.** Static is the right primary comparator for this hypothesis. Run `compare exp001 exp000_sft_baseline` as a secondary comparison only.
- **Seed.** Greedy decoding is used, so the seed is irrelevant to inference. The swap pairing uses a fixed seed of 0. OK.

## Required fixes (numbered)
1. **Token cap (worktree `physwm/vlm/infer.py:81`).** Make the per-object allowance style-dependent: keep `48 + 48*n` for `center` (unchanged), and use `48 + 128*n` for `qwen3d`. Commit it, and add one line to the spec's `change` field: "harness: qwen3d token cap 128/object (instrumentation, center unchanged)".
2. **Make truncation visible.** At the end of each split, `infer.py` must print to stdout (so it lands in `job.log`) the following counts:
   - parse rate
   - **object-missing rate**, i.e. the fraction of object slots that are `None`
   - **truncation rate**, i.e. the fraction of answers whose `raw.strip()` does not end with `]` or `]` followed by an optional code fence

   The report must quote all three per split from `scripts/agent.sh log`. This is logging only, with no change to parsing or scoring.
3. **Replace the hypothesis with fixed numeric criteria** (edit `spec.yaml` before submission):
   ```yaml
   hypothesis: >
     (A) Zero-shot Qwen3-VL-2B with its native bbox_3d prompt localises current 3D centres using the
     image: Task A val_id all_mean(normal) is at least 0.05 m lower than Task A val_id all_mean(swap).
     (B) Its Task B target error is worse than static: compare exp001 static gives ci95 lower bound > 0
     (diff = exp001 - static).
   success_criteria:
     A_supported: "taskA val_id all_mean(swap) - taskA val_id all_mean(normal) >= 0.05 m"
     A_rejected:  "difference < 0.02 m"   # 0.02-0.05 m = inconclusive (no CI available for Task A)
     B_supported: "compare vs static: ci95[0] > 0"
     B_rejected:  "compare vs static: ci95[1] < 0"
     validity:    "object-missing rate <= 5% on every split after fix 1; otherwise the run is reported as
                   a harness/format result, not a model result, for the affected split(s)"
     note: "(B) is descriptive only: static has oracle perception, so (B) does not by itself show
            absence of dynamics knowledge."
   ```
   Additionally, report Task A per split against the exp000:swap Task A value of 0.249 m as context only, since it comes from a different model and is not a criterion.
4. **`ablations: [none, swap]`.**
5. **`budget_hours: 1.5`**, and **no `eval_limit`**. Add a comment in the spec that an eval_limit < 500 breaks `compare` (`exec.py:371`).

## Optional (recommended, not blocking)
- **Perception-matched "own-static".** Score exp001's Task A answer as its Task B prediction. Only this reference can support "cannot predict push outcomes". The pinned tooling cannot compute it today. If the experimenter wants it, have `infer.py` additionally write `pred/normal_ownstatic.jsonl`, which copies Task A objects into Task B records. `evaluate_experiment` globs `pred/*.jsonl`, so it would appear in `results.md` without touching pinned code. If this is done, it must be declared in the spec. If it is not done, it goes to the backlog during post-review.
- Report `val_ood_config` Task A separately. Truncation and the compound shift both concentrate there.

Once fixes 1-5 are committed and the spec is updated, this can be approved without another full review. A diff check of `infer.py` and `spec.yaml` is enough.
