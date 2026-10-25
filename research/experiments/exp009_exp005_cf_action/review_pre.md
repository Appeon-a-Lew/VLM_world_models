# review_pre exp009_exp005_cf_action (round 1)

Verdict: **REVISE** (one spec fix required, routing; code is fine)

Reviewed: spec.yaml on main 1c5405f, worktree commit 9486052 (3 files: infer.py, prompts.py, tests/test_cf_action.py). Nothing was executed (no pytest run here).

## What checks out
- One intentional change. Diff touches only the 3 files named in the spec. No pinned file, config, parse.py or pyproject change, and no seed, epoch, batch-size or eval_limit change. `--batch-size` still comes from exec.py (spec default 32), so the normal pass is bit-comparable to exp005.
- Normal path is byte-identical. `prompt_text(..., action_text=None)` keeps the same f-string. infer.py passes `None` unless `task == "B"` and the mode is in `CF_MODES`. The new test reconstructs the full prompt string literally and compares it. `load_image` sends every `cf_*` mode through the normal-image branch, and the comment says so.
- Task A is unaffected. The Task A branch ignores `action_text`, and a test pins that.
- Rewriting versus the real strings. `DIRECTION_PHRASE` in physwm/sim/push.py is {left: "to the left", right: "to the right", away: "away from the camera", toward: "toward the camera"}. The action text is built as `f"Push the {name} {DIRECTION_PHRASE[label]} with an impulse of {impulse:.2f} N·s."`.
  - Flip replaces exactly that phrase by the opposite label's phrase, with `assert count == 1`. The phrases are mutually non-overlapping and cannot occur in object names.
  - The impulse regex `impulse of ([0-9.]+) N·s` matches the real format. The new value is written with `:.2f`, which matches training.
  - A failed assert would raise and mark the job FAILED. That is the right failure mode, and the [cfprompt] n_changed check is a second guard.
- Leakage and policy.
  - Only val_id is run in the cf passes, which is the pinned `ablation_split`. val_ood_forces gets a print-only read-out in the normal pass, which already runs there. No training split is touched.
  - The GT direction label used for the rewrite is part of the episode input, not answer information.
  - The print-only read-out reads the GT only for printing, and nothing is written to pred files.
- exec.py names. `ABLATION_MODES = ("none","blank","swap","cf_flip","cf_impulse_0.5","cf_impulse_2")` matches the spec ablations list and the infer `--image-mode` choices. check_spec `IMAGE_MODES` matches. Line 379 accepts `<exp>:<mode>` for these modes, so the `compare ...:cf_flip` calls in the spec are valid.
- Print guards. `_cf_readout` is wrapped in try/except, with a leading newline and flush=True. Pred keys are `str(o.id)`, and the code uses `str(ep.action.target)` and `gt_centers_cam(...)[ep.action.target]`, which is consistent with `parse_answer`. Task A and B preds come from the same pass. The `dgt > 0.05` restriction and the sphere / non-sphere grouping follow the spec.
- The hypothesis is falsifiable. Thresholds are fixed before results and are calibrated to the 0.0024 repro difference, the 0.157 normal error and the 0.265 static error.

## Required fix (REVISE)
1. **Routing overlaps and gaps in `success_criterion.routing`.**
   - Overlap: R2 says "F-ignore or I-ignore (either one)", while R3 says "F-read and I-ignore". F-read with I-ignore satisfies both. The routing is not a function of the outcome. Make R2 = "F-ignore and (I-ignore or I-part)", or add an explicit precedence rule. R3 stays "F-read and (I-ignore, or I-part with s < 0.5)".
   - Gap, F-ignore with I-use (direction ignored, magnitude used): R2's current wording sends it to "the action is ignored", which contradicts the evidence. Put it explicitly under `inconclusive`, or add R4 "magnitude used, direction not". Do the same for F-part with any I outcome, which already falls to inconclusive.
   - F-read with I-part and s >= 0.5 falls to inconclusive. That is acceptable, but say so in the text.
   - Add a one-line decision table covering all 3 x 3 combinations of {F-read, F-ignore, F-part} x {I-use, I-ignore, I-part}, with the two validity-failure fallbacks ([cfsign] and [cfdisp] unavailable), so the routing is complete and unambiguous before any cf line is read.

## Non-blocking notes (no spec change required; say so in the report)
- Task A gate (2) at 0.0005. In the cf passes the batches still interleave A and B. A flip or impulse rewrite changes the B token length, so the left padding of the A rows in the same batch changes. That can cause bf16 jitter in Task A, which would show up in gate (2) as a false "not isolating" signal. If gate (2) fails by a tiny amount, check whether the A raw strings are identical before concluding anything.
- F-ignore needs a rise CI upper bound < 0.04 on val_id. If val_id is small, the CI may be wider than that and the outcome lands in F-part. This is acceptable because it is pre-registered and reported.
- Alternative explanation for the impulse rise: rewriting to 0.31 or 1.24 puts the text partly out of the training range [0.3, 1.6] speeds. An error rise can then come from out-of-distribution text confusion, not from dynamics. The slope s and the d(0.5) < d(1) < d(2) ordering are the discriminators, and they are already specified. Keep the rule that the pinned rises alone cannot give I-use.
- Alternative explanation for the flip rise: the model may read only a left/right cue as a position shift prior (for example, "left" means a negative x offset), without simulating a push. The [cfsign] check does not separate the two. Cheapest discriminator, for later: flip only with a label the scene does not support (a target against the table edge). This is not needed for exp009.
- Import risk. `physwm.sim.push` imports mujoco and `physwm.sim.render` at module level. mujoco is a core dependency, and the test imports it at the top, so preflight on the GPU box will catch a missing install before the run. The lazy import keeps the normal path clean.
- Budget 0.75 h. The normal pass over all val and diag splits plus four val_id passes looks adequate relative to the earlier eval-only runs. I did not verify exp005's wall-clock time here.

After fix 1, I expect to APPROVE without further code changes (round 2 only re-reads the spec).
