# Pre-review exp008_uvz_output (worktree HEAD 9ab3687)

**Verdict: APPROVE** (with non-blocking notes; the check_spec STYLES edit is a known human step).

## Checks
1. One change: yes. Config differs from sft_qwen3vl2b.yaml only in `style` and `out`. The test `test_uvz_config_differs_only_in_style_and_out` reads both files, so it is real. Seed, lr, epochs, LoRA and data are untouched. The intended answer-format change also drops the FRAME sentence from the prompt (replaced by the uvz format line). This is part of the representation switch (the xyz frame sentence is meaningless for uvz) and the spec says so. I treat it as bundled, not incidental.
2. Default style is unchanged. `prompt_text` for "center" gives the same head, FRAME and fmt strings. `target_text` default is identical. The parser's center path uses the same keys (`center`, `bbox_3d`) and an equivalent regex. qwen3d falls through to the center parse as before. `sft.py` uses `cfg.get("style", "center")`. The tests cover target and prompt equality only on mocked GT, so the spec's pre-submit byte-identity check on 5 real episodes against main is required and must be run.
3. Evaluation is unchanged. None of the changed files is in PINNED_FILES, and the scorer is untouched. Decoding to camera xyz happens inside the parser, so the pinned scorer sees xyz. The infer.py print uses GT only after the prediction is written and does not touch the pred files. Note that `infer.py` calls `prompt_text(ep, task, args.style)`, so none and swap use the uvz prompt, which is correct.
4. Grid and quantisation: spec and code agree (0-1000 grid, u and v to 1 decimal, z to 3 decimals). At W=448, 0.1 grid unit is 0.045 px, which gives a lateral error of about 6e-5 m at z=1.3 m and fx=520. The z rounding is 0.0005 m, the same as the center format's 3 decimals. Round-trip error is well below 0.002 m and cannot confound the 0.005 threshold.
5. Calibration numbers match exp005 (results.md and report.md): primary 0.3160, static OOD 0.379, val_id Task B 0.157 (static 0.265), none 0.500, swap 0.406, gap 0.343, swap minus normal 0.249, Task A config 0.137, elev 0.084, elev_corner 0.122, objects 0.161. One slip: the forces guard cites 0.510, but that is exp000. exp005 forces is 0.515. Compute the guard against exp005's own value (0.515), so the limit is 0.545. The effect is small.
6. Routing gaps (non-blocking, fix in the report by stating them explicitly):
   - Primary <= 0.311 with config A > 0.100 is unrouted.
   - Config A in (0.100, 0.120] with primary comparable is unrouted.
   - Primary > 0.321 with config A > 0.120 falls under both R3 and R4.
   - Primary in (0.311, 0.321) counts as "comparable", and it is also what you get when the CI excludes 0 but the effect is under 0.005.
   Report these cases verbatim as "inconclusive" and do not retro-fit them into a route.
7. Ablations [none, swap] are comparable to exp005's [none, swap]. Dropping blank is justified. The image-gap definition (none minus normal) is the same. The caveat is that the uvz prompt differs in text, so the none-mode text-only prior differs. The gap is therefore not perfectly like-for-like, and a gap change below about 0.05 should not be over-read. The C-001 check against static is well defined.
8. Budget: exp005 took 1.41 h, and 3 h gives a margin above 2x. Crash risk is low. The parser handles z<=0, NaN, Infinity, a short list, a string value and truncated regex fallback, all covered by tests. A parse failure becomes a missing object, not a clamp. The validity gate caps missing at 0.01. The regex-fallback `float()` on NaN or Infinity literals is covered by the z check. Leakage: training reads `train` only, the decode uses the episode K (an input given in the prompt, not GT), and the print-only GT use is after generation. No leakage found.

## Non-blocking notes
- Hypothesis and thresholds are fixed before the result and are falsifiable. Single seed, so the CI on the paired bootstrap is the arbiter. Reproduction noise is 0.0024, so a primary of 0.311 to 0.315 is not a strong result.
- Report the off-image fraction (u or v outside 0-1000) for Task B predictions and GT, as R4 expects. The spec implies it but nothing in the code prints it. Compute it post hoc from the pred files.
- Confound to name in the report: the FRAME sentence removal and the output format changed together. Any gain cannot be attributed to the pixel representation alone without a center-style run with FRAME removed (cheap, to add to the backlog if R1 or R2 fires).
