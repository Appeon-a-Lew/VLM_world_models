# needs-human: add "uvz" to STYLES in scripts/check_spec.py (one line, blocks exp008 lint only)

**Experiment:** exp008_uvz_output (IDEA-014, (u, v, depth) output; the next training run by exp007's pre-registered routing).

**Problem:** `python3 scripts/check_spec.py research/experiments/exp008_uvz_output/spec.yaml` reports `unknown style 'uvz'` because `STYLES = {"center", "qwen3d"}` in `scripts/check_spec.py` on main. The loop must not edit main's scripts, and a worktree copy of the script is not runnable by the loop (`scripts/loop_bash_gate.py` allows only the main path). The experimenter confirmed that this is lint only: `exec.py` passes `spec.style` to `infer.py` without validating it, and `scripts/check_spec.py` is not in `PINNED_FILES`, so no re-pin is needed.

**Needed:** on main, add `"uvz"` to `STYLES` in `scripts/check_spec.py` (and commit). Nothing else.

**What the loop does meanwhile:** implements exp008 in its worktree (prompts.py, parse.py, infer.py, sft.py, new config `configs/sft_qwen3vl2b_uvz.yaml`, tests) and runs the pre-review. It will not submit until `check_spec.py` passes on the spec, unless you say a lint failure may be ignored.
