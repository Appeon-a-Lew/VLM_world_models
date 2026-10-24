# Idea backlog

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`, `needs-human: reason`).
The planner re-ranks this file; the researcher and reviewer append to it.

## Ranking (planner, 2026-10-24; re-rank only, no new spec)
Inputs: exp000 report + review_post, STATE open questions, literature notes 2025_limits_of_finetuning, 2026_onecanvas, 2025_sd_vlm, 2025_spacemind. exp001_zeroshot_qwen3d is in pre-review (REVISE) and is not touched here.
Principle: simple baseline first. Cheap diagnostics that settle *why* val_ood_config fails and *whether* the forces gain is force-conditioned come before any architectural change.

Feasibility facts I checked in the code (they shape the costs below):
- `results.md` and `compare` are produced by pinned `physwm/agent/exec.py` (PINNED_FILES, exec.py:38-42). Adding a `missing_rate` column or per-split CIs there is **needs-human** (re-pin).
- `agent.sh` cannot read pred files, so "analyse exp000's existing predictions" in practice means a cheap `kind: eval_only` re-run with `lora: $WM/experiments/exp000_sft_baseline/model/adapter`. validate_spec allows this (exec.py:161-163). The worktree's `physwm/vlm/infer.py` can write extra `pred/*.jsonl` files.
- Pinned `summarize` scores only the episodes that appear in a pred file (metrics.py:64-70). `evaluate_experiment` scores every `pred/*.jsonl` (exec.py:299-300), but the primary metric and `compare` read only `pred/normal.jsonl`. So **derived files** (own-static, prior baseline) and **subset files** (e.g. only val_ood_config episodes with 4 objects) get pinned-code scores in `results.md` without touching the primary metric. Subset content must never go into `normal.jsonl`.
- The episode metadata already holds every stratification variable: `camera.elevation_deg`, `len(objects)`, `action.speed` / `action.impulse`, `ObjectState.on_table` (schema.py).
- Decoding is greedy, so the re-run should reproduce exp000 `normal` (primary 0.3136, within bf16 batch noise). That gives a free determinism check.
- Recommendation: put #1-#3 (plus the interim part of #4) in **one** eval_only job, ~0.25 GPU-h. None of them changes the model or the predictions; they are read-outs of identical outputs and cannot confound one another. If the planner or reviewer reads this as more than one change, split it into config (006+007) and forces (008).

Ranked (rank. idea: for / against; effect on primary; cost; risk; information value):
1. **IDEA-006 own-static + missing_rate.** For: Task A target 0.133 is about half of the Task B 0.269 on config. Static has oracle perception (baselines.py:26), so every "vs static" statement mixes perception and dynamics. parse_rate 1.000 does not rule out dropped objects (parse.py:63), and config has 4-5 objects, never seen in training (exp000/review_post.md §a). Against: Task A and Task B are separate prompts, so own-static uses a different answer's t=0 positions; report it as is. Effect: none directly, but it picks the next lever. Cost: ~0.25 GPU-h, shared job. Risk: none protected. missing_rate goes to job.log from worktree code, because a results.md column would need a pinned edit. Info: **high**, and every outcome changes the plan: missing_rate > 5% on config → fix format/count (IDEA-004); SFT-B ≈ own-static on config → dynamics fail too (IDEA-002); SFT-B below own-static by about the val_id margin → perception bottleneck → geometry ideas (016/014/019/020).
2. **IDEA-008 forces stratification + training-prior baseline.** For: forces supplies 78% of the aggregate gain. The split is bimodal by construction (median ≈ weak pushes, mean ≈ strong pushes; review_post §b). limits_of_finetuning predicts shortcut/prior learning. Against: the prior baseline needs the mean training displacement, computed from `train` metadata (allowed) along `direction_world` mapped to the camera frame; the slope of |Δ| on impulse is unpinned (log only). Effect: none directly. If prior ≈ SFT on forces, the headline C-002 gain is not force-conditioned dynamics, which rewrites the claim and promotes IDEA-002. Cost: +0 in the shared job. Risk: none protected. Info: **high**.
3. **IDEA-007 step 1: stratify config by count (4/5) and elevation bin.** Also stratify val_id Task A over 30-60° to see whether error already rises toward low elevation in-distribution. For: review_post §a alternatives 1-2; the size-generalisation analogue in limits_of_finetuning. Against: a weak discriminator. The corner cluster is constant across the split, and all of 12-25° lies outside training's 30-60°, so a within-split trend is only suggestive and is likely ambiguous (→ IDEA-017). Cost: +0 in the shared job. Info: medium. Include it because it is free.
4. **IDEA-009 per-split CIs.** Status: needs-human for the pinned `compare` change. Interim: the diagnostic job prints a per-split paired bootstrap to job.log (SFT vs static, SFT vs own-static), labelled unpinned. Static is computed from GT t=0 in worktree code, and the reviewer cross-checks the point estimates against pinned results.md. Info: medium-high. It gates every per-split claim; "config +0.028" has no CI today.
5. **IDEA-022 frozen geometry models vs elevation.** For: SpaceMind note. It tells whether low-elevation, cluttered views are intrinsically harder (part of the config gap then is not attributable to SFT), and whether a predicted-depth route for 020/013 is viable. Against: the job runner only launches sft/infer (exec.py:256-269), so this has to be hacked into worktree infer.py; AbsRel is not a pinned metric; VGGT probably needs a new pip package (**needs-human** for that half). Depth-Anything-V2 should load through transformers (verify before planning). Worth running only if #1/#3 implicate perception/camera. Cost: ~0.5 day, < 0.2 GPU-h. Info: medium.
6. IDEA-002 displacement output: the first training candidate after the diagnostics. For: it separates perception from dynamics by design, and Task A is already good on objects/forces (0.024/0.016), so the error left there is dynamics. Against: own centre + delta can compound the Task A error. Effect: plausibly −0.01 to −0.03 if a dynamics prior is the issue (promote it if #2 shows prior ≈ SFT); uncertain on config. Cost: 1.4 GPU-h. Info: high.
7. IDEA-016 camera pose in the prompt: the cheapest geometry probe (prompt only). Run it if #1/#3 implicate the camera. Against: 12-25° is numerically outside training. A failure here is informative when contrasted with 020/021. Cost: 1.4 GPU-h.
8. IDEA-019 K-varying crops: tests whether the model uses K at all (K is constant now; limits note). Against: a crop does not change elevation, so it is an indirect fix for config. Cost: 1.4 GPU-h.
9. IDEA-014 (u, v, depth) output: 2D pointing is native to Qwen3-VL (OneCanvas), it splits the error into 2D and depth parts, and it enables IDEA-015. Against: depth along low-elevation rays may still be tied to the viewpoint; needs a parser change. Cost: 1.4 GPU-h.
10. IDEA-020 scalar depth PE: SD-VLM shows 46.73 → 56.31, with fixed encoding as good as learnable. Against: uses GT depth (privileged; claims need a predicted-depth follow-up, and the ablations must drop the PE), plus 0.5-1 day of surgery. Run only after the diagnostics confirm a camera/perception cause.
11. IDEA-004 set-of-marks: conditional. Jump it to #6 if #1 shows omission/binding errors with 4-5 objects; otherwise low.
12. IDEA-003 Task B only: cheap (1.2 GPU-h) but answers no open question, and dropping Task A removes the own-static diagnostic.
13. IDEA-015 depth-lookup oracle: depends on 014.
14. IDEA-013 per-patch xyz PE: a superset of 020. Run only if 020 helps partially.
15. IDEA-018 ViT LoRA: limits_of_finetuning adapted the ViT and still failed under visual shift; direction uncertain. Cost: 1.5 GPU-h.
16. IDEA-021 camera FiLM gate: conditional on a partial gain from 016.
17. IDEA-005 depth as a second image: SD-VLM found it hurt (22.64 vs 46.73). Deprioritised.
- IDEA-017: unblocked by the human (2026-10-24); the diagnostic splits exist. See its status below.
- IDEA-001: planned: exp001_zeroshot_qwen3d (in pre-review, untouched).

## IDEA-001 Zero-shot with the native 3D-grounding prompt
- source: Phase 1 smoke test (the `center` prompt yields `[0,0,0]`)
- change: no training; `style: qwen3d`
- status: planned: exp001_zeroshot_qwen3d

## IDEA-002 Predict the displacement instead of the absolute future position
- source: Phase 1 design. Absolute targets mix perception error and dynamics error.
- change: Task B answers both the current centre and `delta` = p(t=1) − p(t=0); the scored position is the model's own current centre + delta (never the GT t=0 position, which would leak)
- status: proposed

## IDEA-003 Task mix ablation: Task B only vs A + B
- source: Phase 1 design question. Does grounding supervision help dynamics?
- change: `tasks: [B]` in the SFT config
- status: proposed

## IDEA-004 Neutral object references (set-of-marks IDs instead of colour + shape names)
- source: Slack discussion (semantic vs geometric vs neutral references)
- change: draw numbered marks on the image; refer to objects as "object 1, 2, ..."
- status: proposed

## IDEA-005 Add metric depth as a second image (pseudo-colour)
- source: Bart, grounding via unprojection
- change: give the depth map as an extra image input
- note (researcher, 2026-10-24): SD-VLM (research/literature/2025_sd_vlm.md) found that depth as a second image *hurt* badly (MSMU-Bench 22.64 vs 46.73 with no depth), and concatenated depth tokens hurt too (35.72). Only additive per-patch depth PE helped (56.31). Prefer IDEA-020/013 first.
- status: proposed

## IDEA-006 Perception-matched static baseline ("own-static") plus missing_rate reporting
- source: review of exp000_sft_baseline
- change: eval-only, no GPU. Score each model's own Task A prediction as its Task B answer, per split, and add `missing_rate` to `results.md`. Static copies GT t=0 positions (oracle perception), so a model's gain over own-static isolates dynamics skill.
- discriminates: on val_ood_config, perception-only failure (SFT-B well below own-static) vs a dynamics failure as well (SFT-B ≈ own-static), and an object-omission artefact with 4–5 objects (missing_rate > 0).
- note (planner, 2026-10-24): a `missing_rate` column in results.md would need an edit to pinned exec.py. `summarize` already computes it (metrics.py:85), so print it per split/task to job.log from worktree infer.py instead. Own-static works without any pinned change: a derived `pred/normal_ownstatic.jsonl` (Task A objects copied into the Task B records) is scored automatically.
- status: proposed

## IDEA-007 Single-factor breakdown of val_ood_config
- source: review of exp000_sft_baseline
- change: first, zero-cost. Stratify existing exp000 val_ood_config results by object count (4 vs 5) and camera-elevation bin. If that is ambiguous, generate three small validation splits from the reserved val seed range, each changing one factor: elevation [12, 25] only, count [4, 5] only, corner cluster / min_gap 0.01 only. Then evaluate exp000 on them (eval_only).
- discriminates: camera-frame miscalibration vs object-count / format failure vs dense-cluster dynamics as the cause of the config failure.
- status: proposed

## IDEA-008 Stratified val_ood_forces analysis + training-prior displacement baseline
- source: review of exp000_sft_baseline
- change: eval-only. Split val_ood_forces by speed mode (weak [0.08, 0.25] vs strong [2.0, 2.8] m/s) and by off-table vs on-table outcome. Regress predicted |Δ| on the impulse. Add a trivial baseline: own Task A position + mean training displacement along the stated push direction.
- discriminates: regression to the training displacement prior (prior baseline ≈ SFT; no slope with impulse) vs partial extrapolation (positive slope within modes) vs gain driven by off-table events.
- status: proposed

## IDEA-009 Per-split paired-bootstrap CIs in `agent.sh compare`
- source: review of exp000_sft_baseline
- change: tooling. Report a CI per val split and for the val_id ablation diffs, not only for the aggregate primary metric.
- discriminates: whether split-level statements (e.g. config +0.028) are outside the noise. Needed before any per-split claim can become solid.
- note (planner, 2026-10-24): `compare` lives in pinned `physwm/agent/exec.py`, so the proper fix needs a human edit plus a re-pin. Interim: an unpinned per-split bootstrap printed to job.log by the diagnostic eval job (see Ranking #4).
- status: done (human, 2026-10-24). `compare` on main now reports `per_split_ci95` / `per_split_verdict`, takes `--splits` (any val_* or val_diag_* split), `--task A|B`, and `exp_id:mode` for ablation runs. The interim unpinned bootstrap is no longer needed.

## IDEA-013 Per-patch camera-frame xyz embedding from depth (OneCanvas-style, single view)
- source: research/literature/2026_onecanvas.md
- change: unproject each merged vision token's pixel block (median of `depth_t0`, constant K) to camera-frame xyz. Encode it with sinusoids (OneCanvas: per-axis sin/cos/raw at 16 log frequencies in [0.1, 100] rad/m, plus radial terms and the unit ray), pass it through a 2-layer MLP, and add it via a zero-initialised gate to the visual tokens after the merger. Train the MLP (higher LR) plus the existing LoRA. The image ablations (none/blank/swap) must also drop or swap the PE.
- expected: Task A error on val_ood_config falls toward the ID level (~0.02 m) because the metric coordinates are no longer inferred from the viewpoint. Task B on val_ood_config at or below static. A stronger version of IDEA-005, with geometry given per token instead of as a second image.
- cost: about 1–2 days of model surgery plus about 1.5 GPU h. A follow-up with predicted depth (e.g. Depth Anything) is needed before claiming RGB-only results.
- status: proposed

## IDEA-014 Answer as (u, v, depth) and unproject analytically
- source: research/literature/2026_onecanvas.md
- change: Task A/B targets become the pixel centre (Qwen 0–1000 grid) plus the centre depth z in metres. Camera xyz is computed with K for scoring (metrics unchanged).
- expected: lower Task A error on val_ood_config, since 2D pointing is native to Qwen3-VL and less viewpoint-tied. The error also splits into 2D and depth parts.
- cost: prompt/target and parser change plus 1 SFT run (about 1.4 GPU h)
- status: proposed

## IDEA-015 Depth-lookup oracle for Task A (diagnostic)
- source: research/literature/2026_onecanvas.md
- change: eval-only, on IDEA-014 outputs. Replace the predicted z with `depth_t0` at the predicted (u, v), plus a half-extent offset along the ray.
- expected: tells whether the val_ood_config failure is in 2D localisation or in metric depth
- cost: about 0.5 day, no training
- status: proposed

## IDEA-016 Camera pose in the prompt
- source: research/literature/2026_onecanvas.md
- change: add the camera elevation and distance (or the table-plane normal and offset in the camera frame, from `cam2world`) to the prompt text
- expected: some reduction of Task A error on val_ood_config. It may fail because 12–25° is numerically outside training, which would be informative when contrasted with IDEA-013.
- cost: prompt change plus 1 SFT run (about 1.4 GPU h)
- status: proposed

## IDEA-017 Factorised val_ood_config diagnostic splits
- source: research/literature/2025_limits_of_finetuning.md
- note: same as step 2 of IDEA-007. Try IDEA-007's zero-cost stratification of exp000 first; this one is needed only if that is ambiguous.
- change: eval only. Generate three small validation splits that each change one factor of `sim_ood_config.yaml`: (a) elevation 12–25° only, (b) 4–5 objects only, (c) the corner cluster only. Evaluate exp000.
- expected: attributes the Task A 0.016 → 0.136 m jump to the camera vs count/clutter, which decides between the geometry fixes (013/014/016) and a clutter/binding fix
- cost: sim generation plus about 0.2 GPU h eval
- note (planner, 2026-10-24): seeds are not actually the blocker. `op_gen` accepts `dev_*` splits with seeds outside [1M, 9M), e.g. ≥ 9,000,000 (exec.py:417-428). The real blockers: (1) `op_gen` only takes configs from main's `configs/`, so three new single-factor sim configs must land on main; (2) pinned `evaluate_experiment` loads only `validation_splits` episodes (exec.py:295-296), so predictions on `dev_*` splits would be silently skipped. Scoring them needs a pinned change, or unpinned scoring in worktree code, which a human should approve.
- status: unblocked (human, 2026-10-24). Splits `val_diag_elev` (elevation 12–25° only), `val_diag_count` (4–5 objects only) and `val_diag_corner` (corner cluster with min_gap 0.01 only), 300 episodes each, configs `configs/sim_diag_*.yaml`, policy `data.diagnostic_splits`. Every job now runs inference on them and lists them in results.md (not in the primary metric); `compare --splits` gives CIs. To get exp000 numbers, plan an eval_only spec with `lora: /var/tmp/toyi/wm/experiments/exp000_sft_baseline/model/adapter`.

## IDEA-018 LoRA on the vision encoder too
- source: research/literature/2025_limits_of_finetuning.md
- change: extend `lora.target_modules` to include the ViT attention/MLP layers
- expected: possibly lower ID Task A error. The val_ood_config effect is uncertain, and it may overfit the viewpoint distribution further (the paper adapted the ViT and still failed under visual shift).
- cost: config change plus 1 SFT run (about 1.5 GPU h)
- status: proposed

## IDEA-019 Random crop/zoom augmentation with intrinsics rewritten in the prompt
- source: research/literature/2025_limits_of_finetuning.md
- change: at training time, random crop and resize (scale 0.6–1.0, random offset). Update fx, fy, cx, cy in the prompt. The camera-frame targets are unchanged by a crop.
- expected: K is currently constant, so the model can ignore it and memorise a pixel→metric map. Varying K forces use of projective geometry, which should help on val_ood_config. Small ID cost.
- cost: data-loader change plus 1 SFT run (about 1.4 GPU h)
- status: proposed

## IDEA-020 Scalar metric-depth sinusoidal PE on visual tokens (cheap rung below IDEA-013)
- source: research/literature/2025_sd_vlm.md
- change: mean-pool `depth_t0` onto the merged-token grid. Encode depth in metres with a *fixed* scale (no per-image min-max) as standard sin/cos. Add it to the visual tokens after the merger through a single zero-initialised scalar gate. No MLP, no xyz. Train the gate plus the existing LoRA. The image ablations (none/blank/swap) must drop or swap the PE too.
- expected: most of IDEA-013's Task A gain on val_ood_config (SD-VLM: no depth 46.73 → DPE 56.31; fixed vs learnable encoding equal, 56.31 vs 56.18). If this closes the config gap, it also supports the camera-elevation hypothesis over count/clutter. If it works, the xyz/MLP of IDEA-013 may be unnecessary.
- cost: about 0.5–1 day of code plus about 1.5 GPU h
- status: proposed

## IDEA-021 Camera-pose FiLM/SwiGLU gate on visual tokens (non-text variant of IDEA-016)
- source: research/literature/2025_spacemind.md
- change: map a small camera vector from `cam2world` (gravity direction in the camera frame plus camera height above the table) through an MLP to a SwiGLU gate and bias applied residually (zero-init) to the merged visual tokens, as in SpaceMind's CGMF. Train the MLP plus the existing LoRA.
- expected: tests whether a continuous embedding conditions on viewpoint better than digits in the prompt. Run only if IDEA-016 gives a partial Task A gain on val_ood_config. The same extrapolation risk applies (12–25° lies outside training's 30–60°).
- cost: about 1 day of surgery plus about 1.5 GPU h
- status: proposed

## IDEA-022 Frozen geometry model (VGGT / Depth-Anything-V2) error vs camera elevation (diagnostic)
- source: research/literature/2025_spacemind.md
- change: eval only, no training. Run pretrained single-image VGGT and Depth-Anything-V2 on the t=0 images of val_id and val_ood_config. Report the depth AbsRel vs `depth_t0`, and the object-centre xyz error from the predicted point map averaged over the GT mask, by elevation bin and object count.
- discriminates: flat error across elevation while exp000 Task A jumps 0.016 → 0.136 m means the failure sits in the VLM's learned pixel→metric map, and an RGB-only geometry-encoder route (predicted-depth input for IDEA-020/013) is viable. If the geometry models also degrade at 12–25° or with 4–5 objects, the low-elevation, cluttered views are intrinsically harder and part of the config gap is not attributable to SFT.
- cost: about 0.5 day, under 0.2 GPU h
- note (planner, 2026-10-24): the job runner launches only `physwm.vlm.sft` / `physwm.vlm.infer`, so this has to run inside worktree infer.py. VGGT most likely needs a new pip package (not allowed). Check whether Depth-Anything-V2 loads through the existing transformers install before planning, and do only that half unless a human approves VGGT.
- status: proposed, Depth-Anything-V2 half only. Human, 2026-10-24: VGGT half dropped, no new dependencies. Depth-Anything-V2 is allowed only if it runs with the current install.
