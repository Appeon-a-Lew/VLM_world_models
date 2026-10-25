# needs-human: add "centerpose" to STYLES in scripts/check_spec.py (one line, blocks exp010 lint only)

**Experiment:** exp010_exp005_camera_pose_text (IDEA-016, camera elevation and distance as a prompt sentence; routed by exp009 R1).

**Problem:** `scripts/check_spec.py` on main has `STYLES = {"center", "qwen3d", "uvz"}`, so the exp010 spec fails lint with `unknown style 'centerpose'`. Same situation as the uvz request (answered 2026-10-25): lint only, `exec.py` passes `spec.style` to `infer.py` unvalidated, and the script is not in PINNED_FILES.

**Needed:** on main, add `"centerpose"` to `STYLES` in `scripts/check_spec.py` (and commit). Nothing else.

**What the loop does meanwhile:** implements exp010 in its worktree and runs the pre-review. It will not submit until `check_spec.py` passes, unless you say a lint failure may be ignored.

---

# needs-human (optional, lower priority): counterfactual modes on shifted splits (IDEA-060) and in-range dose-response (IDEA-061)

exp009 showed the exp005 adapter uses the action text on val_id only (the pinned `ablation_split`). Whether it still does on val_ood_forces, val_ood_config and val_diag_elev (where the exp006/007 LR/AT probes live, and where forces carries the largest share of the primary) cannot be tested without either (a) allowing the cf_* modes to run on a configurable list of splits (pinned policy `ablation_split` and exec.py), or (b) new pinned modes. Also wanted: `cf_impulse_0.8` and `cf_impulse_1.25` (stay inside the training impulse range; IDEA-061). Exploratory print from exp009: on val_ood_forces the normal-pass non-sphere median predicted displacement is 0.215 m against GT 1.257 m, so the magnitude is strongly under-scaled OOD. Not blocking exp010.
