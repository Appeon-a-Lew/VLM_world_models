# Pre-submission review: exp010_exp005_camera_pose_text

Spec at main c93504d; diff is worktree commit 783c204 (`git diff main...HEAD`). Nothing was executed.

## Verdict: APPROVE (round 1)

Submit is conditional on the human adding `centerpose` to STYLES in `scripts/check_spec.py`. See `research/outbox/needs_human_check_spec_centerpose_style.md`. Re-run `python3 scripts/check_spec.py` on the spec and run the preflight pytest before submitting.

## Checks

1. **One intentional change.** The only experimental change is the sentence in the prompt style `centerpose`.
   - `configs/sft_qwen3vl2b_pose.yaml` matches the exp005 recipe: lora r16/alpha32, lr 2e-4, 2 epochs, batch 4 x accum 4, seed 0, same data and tasks. It differs only in `style` and `out`.
   - No seed, epoch, data or parser change rides along.
   - `parse.py` is untouched. `infer.py` passes `--style centerpose` through, and `exec.py:270` forwards the spec style.
   - Parsing of `centerpose` output goes through `parse_answer`, which takes no style argument. The spec says "explicit map"; the code simply calls the same parser, which is equivalent and fine.

2. **Byte-identity of `center` and `qwen3d`.**
   - In `prompts.py` the `frame` variable equals `FRAME` unless the style is `centerpose`. The `qwen3d` branch and `target_text` are untouched.
   - `tests/test_prompt_pose.py` rebuilds main's `center` prompt independently for tasks A and B and compares exactly.
   - Weak test: `test_target_text_identical_across_styles` only asserts that "degrees" is absent from the target. `target_text` takes no style argument, so it is trivially identical. This is harmless and non-blocking.
   - The `Episode`, `Camera`, `ObjectSpec` and `Action` positional constructors in the tests match `schema.py`.

3. **No leakage.**
   - The sentence uses only `ep.camera.elevation_deg` and `ep.camera.distance`. These are camera metadata, not object, target or mass information.
   - The metadata is legitimately available at test time, and the shift in `val_ood_config` is exactly in elevation. Giving the model the camera pose is the stated experimental input, not leakage of GT.
   - Only the `train` split is used in SFT. Inference evaluates `val_*` only. No new data is generated, so reserved seeds are not an issue.

4. **Metric gaming.**
   - The parser is unchanged.
   - Missing objects keep the existing penalty.
   - The `[poseerr]` diagnostic calls `per_object_rows` on in-memory records.
     - Their `objects` keys are `str(id)`, matching `parse.py:57`.
     - Their `task` field is present.
     - The diagnostic is print-only, wrapped in try/except, and never written to a pred file.
   - `records = []` is reset per split.

5. **`sft.py` does not alter exp005.** `SFTData` defaults to `style="center"`, and the call site uses `cfg.get("style", "center")`. The exp005 config has no `style` key, so its behaviour is unchanged.

6. **Swap/none ablations.**
   - Both keep the pose sentence, which is derived from the episode's own metadata.
   - The `none` gap will shrink, because text-only now knows the camera. The spec flags this and says not to over-read it.
   - Under `swap`, the sentence belongs to the prompt episode and the image to another episode. The pose text can therefore contradict the image. That makes swap a slightly different test from exp005's. It is informative about whether the model trusts text or pixels, but not strictly comparable to exp005's 0.249.
   - The report should say this.

7. **Calibration (IDEA-056) and routing.**
   - The thresholds derive from recorded exp005 values and repro noise.
   - P classes are total and disjoint. P_down requires P <= 0.311 and the CI below 0, P_up is P > 0.321, and everything else is P_flat.
   - T classes are total and disjoint at 0.107 and 0.127.
   - The 3x3 grid has all 9 cells mapped to exactly one route:

     | | T_big | T_part | T_none |
     |---|---|---|---|
     | P_down | R1 | R1 | R2 |
     | P_flat | R3 | R4 | R5 |
     | P_up | R3 | R6 | R6 |

   - Validity failure routes to R7 regardless of cell.
   - The validity gate has concrete numeric limits, including `n_with_pose == n_total`.
   - The hypothesis is falsifiable and fixed before results.
   - The budget (3 h) is realistic.

8. **Policy and pinned files.** No pinned file, `pyproject.toml` or `parse.py` is touched. Only `prompts.py`, `infer.py` and `sft.py` change, plus the config and the test file.

## Non-blocking notes for the report

- The 12-25 degree values lie outside the training range of 30-60. A flat `[poseerr]` curve means the digits did not extrapolate or were ignored (R5). Do not claim "pose text does not help" in general.
- Elevation and distance change together, so the report cannot attribute an effect to one of them.
- Single seed. Differences under 0.01 are not interpreted without a seed-floor run.
