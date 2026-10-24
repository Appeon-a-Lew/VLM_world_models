# Idea backlog

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`, `needs-human: reason`).
The planner re-ranks this file; the researcher and reviewer append to it.

## Ranking (planner, 2026-10-24, round 2; spec: exp002_zeroshot_qwen3d_rerun)
Inputs: exp001 report + review_post, the HUMAN NOTE in STATE.md (executor bug, fixed in d4de761), the coordinator directive (next spec = exp001 rerun), and literature notes 2025_qwen3_vl, 2026_quantiphy, 2023_omni3d_cube_rcnn, 2026_camera_robust_3d_tooluse.

What changed since round 1:
- **exp001 is invalid as specified.** It ran main's infer.py: a 48-token/object cap and no stats print. Its swap-beats-normal result, its config Task A median of 1.000 (likely truncation: 240/288 tokens for 4/5 objects against ~260/325 needed, exp001/review_pre.md §1) and C-003 describe main's harness, not the spec. Every exp001-specific read-out (IDEA-024, IDEA-028) has to be retargeted at exp002's preds and made conditional on exp002's outcome.
- **IDEA-029's premise is wrong.** The stats lines were never printed, so the tqdm filter was not at fault. Whether job.log prints are visible now is untested. exp002 tests it for free (stats lines with a leading newline). Until exp002 shows them, keep visible diagnostics in pinned-scored pred/*.jsonl files.
- exp000 stays valid (no code change). The round-1 feasibility notes below still hold.
- New pinned-visible read-out trick (IDEA-026 note): a counterfactual-action pred file scored against the *real* GT gives an "action-dependence gap" in results.md, analogous to the image gap, with no job.log needed.

Top 5 (for / against; effect on primary; cost; risk; information value):
1. **IDEA-001R exp001 rerun → planned: exp002_zeroshot_qwen3d_rerun.** For: human directive (STATE HUMAN NOTE). The reviewed harness never ran (review_pre §1 predicted config truncation), and the diag splits now come for free. Against: zero-shot is ~0.38 m above static (exp001, harness-level), so no primary gain is possible. Effect on primary: none (reference only). Cost: ~0.5 GPU-h expected, 1.5 budgeted (the diag splits add ~1800 samples). Risk: none protected; instrumentation only, center byte-identical. Info: high for validity. It settles whether exp001's (A) rejection and the config collapse were cap artefacts, it gates IDEA-024/028/023, and it tests job.log visibility, which every log-based diagnostic below depends on.
2. **IDEA-017 exp000 adapter, eval_only, on val_diag_elev/count/corner.** For: the top open question (why config is worse than static: 0.269 vs 0.241). Zero code change, since diag splits are inferred automatically. Greedy decoding gives a free determinism check (normal primary should reproduce 0.3136). Limits note and camera_robust note (K/viewpoint brittleness of RGB-only Qwen3-VL). Against: three single-factor splits can miss an interaction (the effect of all three together could exceed the sum). Effect: none directly. It picks the config lever: elev → 016/014/020, count → 004, corner → dynamics/002. Config is 1/3 of the primary, and closing it to static is worth ≥ 0.01 m. Cost: ~0.25 GPU-h (normal pass + `none` ablation, which is mandatory for image_dependence_gap). Risk: none. Info: **highest per GPU-h of any idea that bears on the primary.** Run it right after exp002.
3. **IDEA-025 + IDEA-026 counterfactual action tests on exp000 (forces).** For: forces gives 78% of exp000's gain (exp000 review_post §b). QuantiPhy shows VLMs anchor to priors under counterfactual inputs (−70-80% MRA). The direction flip, scored against real GT in `pred/cf_flip.jsonl`, is visible in results.md with no pinned change. Against: the impulse slope (025) needs a custom statistic, so it needs job.log (wait for exp002 to confirm visibility) or two pinned-scored α files (the error-vs-α curve is a weaker read-out than the slope). Effect: none directly. If the model ignores the action, C-002's gain is a prior and IDEA-002 is promoted. Cost: ~0.15 GPU-h. Risk: counterfactual files must never go into normal.jsonl. Two prompt interventions in one job are read-outs of the same adapter; split them if the reviewer counts them as two changes. Info: high.
4. **IDEA-006 own-static derived file (exp000 adapter).** For: it separates perception from dynamics on every split, diag splits included. That makes IDEA-017 interpretable: diag Task B vs own-static says whether a factor breaks perception only or dynamics as well. Against: one worktree infer.py addition (a derived file). It fits best as a ride-along in the IDEA-017 job: a read-out, no prediction change, declared in the spec. Effect: none directly. Cost: +0 if it rides along. Risk: none. Info: high, together with #2.
5. **IDEA-002 displacement output.** The first improvement attempt on the primary. For: Task A is already good on objects and forces (0.024/0.016), so the remaining Task B error there is dynamics. Against: it should wait for #2/#3/#4 to say whether config is a perception or dynamics failure and whether forces is prior-driven. Effect: plausibly −0.01 to −0.03 on the primary. Cost: 1.4 GPU-h. Risk: own centre + delta compounds the Task A error. Info: high.

Then:
6. IDEA-008 forces stratification + training-prior baseline. Mostly superseded by 025/026 (within-episode interventions). Keep only the prior baseline, as a derived pred file.
7. IDEA-028 / IDEA-024 read-outs, **retargeted at exp002 preds** and conditional. Run only if exp002 shows missing > 5% (028: labels, omission) or swap ≥ normal with low missing (024: frame/scale). The exp001 preds are not worth reading.
8. IDEA-023 verbatim native prompt. Only after exp002, and only if (A) stays unsupported with valid missing rates.
9. IDEA-016 camera pose in prompt: first, if IDEA-017 implicates elevation.
10. IDEA-014 (u, v, depth) output: Cube R-CNN and camera_robust evidence. Second geometry lever if elevation is implicated.
11. IDEA-004 set-of-marks: jumps to #5 if val_diag_count carries the config failure.
12. IDEA-022 Depth-Anything-V2 vs elevation (verified loadable): only if elevation is implicated.
13. IDEA-020 depth PE; 14. IDEA-019 K-crops (+ IDEA-027 eval, K is fixed in all val splits so it does not touch the primary); 15. IDEA-003; 16. IDEA-015; 17. IDEA-013; 18. IDEA-018; 19. IDEA-021; 20. IDEA-005 (SD-VLM: harmful).
- IDEA-029: rejected (premise wrong, see its entry).

## Ranking (planner, 2026-10-24, round 1; superseded by round 2 above, kept for the feasibility notes)
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
- history: exp001_zeroshot_qwen3d (2026-10-24) is **invalid as specified**. Per the HUMAN NOTE in STATE.md, it ran main's infer.py (48-token/object cap, no stats print), not its reviewed harness. Its (A)/(B) verdicts, swap-beats-normal and the config median of 1.000 describe main's harness only.
- status: planned: exp002_zeroshot_qwen3d_rerun (same spec, harness change re-applied on a fresh worktree off main ≥ d4de761). Follow-ups IDEA-028/024/023 are conditional on exp002's outcome.

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
- evidence (researcher, 2026-10-24): Cube R-CNN uses the same parameterisation, projected 2D centre plus depth back-projected with known K (research/literature/2023_omni3d_cube_rcnn.md, §4.1). Equation-anchored tool use (research/literature/2026_camera_robust_3d_tooluse.md, Table 3) shows that at fixed K, adding depth + K to Qwen3-VL-8B raises Det F1 from 38.43 to 44.13 and Ground from 50.13 to 55.17. Caveat from the same note: back-projection fixes the camera-frame centre, but Task B dynamics on the table plane still depend on elevation, so for val_ood_config Task B pair this with IDEA-016/021.
- status: proposed

## IDEA-015 Depth-lookup oracle for Task A (diagnostic)
- source: research/literature/2026_onecanvas.md
- change: eval-only, on IDEA-014 outputs. Replace the predicted z with `depth_t0` at the predicted (u, v), plus a half-extent offset along the ray.
- expected: tells whether the val_ood_config failure is in 2D localisation or in metric depth
- cost: about 0.5 day, no training
- note (researcher, 2026-10-24): research/literature/2026_camera_robust_3d_tooluse.md (§3.2) samples ≤ 5 valid pixels inside the object mask (depth < 0.1 m discarded) and averages them. That is more robust than a single pixel at the predicted (u, v). We have GT masks, so use the median `depth_t0` over the mask pixels nearest the predicted point. At 1.0× the paper's GT vs UniDepthV2 depth gap is small (Det 44.90 vs 44.13), which supports a later predicted-depth variant.
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
- note (planner, 2026-10-24, round 2): ranked #2. It is the next spec after exp002 (suggested id exp003_exp000_diag_eval). The change is none: same adapter, `style: center`, `ablations: [none]`. Optional ride-along: the IDEA-006 own-static derived file, declared in the spec.

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
- evidence (researcher, 2026-10-24): (1) research/literature/2026_camera_robust_3d_tooluse.md (Tables 1-2): RGB-only Qwen3-VL-8B "overfit[s] to a canonical training intrinsic". Det F1 is 10.75 / 38.20 / 15.05 at K × 0.5 / 1.0 / 1.5, so exp000 (one K) should be at least as brittle. Even with explicit K + depth, only 38-47% of the score is retained under shift, so crops with text K alone are unlikely to make the model fully K-equivariant. (2) research/literature/2023_omni3d_cube_rcnn.md (Table 1, §4.2): scale augmentation gives +3.1 AP3D only together with a K-consistent depth target (virtual depth z·H/f), and the text reports it as harmful without one. If IDEA-019 is run, consider a control arm whose depth target is virtual depth z·H/f′ (f′ = the crop's focal), converted back analytically, so the model never has to read K from text. (3) Neither paper studies elevation or extrinsic shift. val_ood_config has fixed K, so this is not evidence that 019 helps the primary metric. Its evaluation counterpart is IDEA-027.
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
- status: proposed, Depth-Anything-V2 half only. Human, 2026-10-24: VGGT half dropped, no new dependencies. Depth-Anything-V2 is allowed only if it runs with the current install. Verified by the human: `transformers.DepthAnythingForDepthEstimation` imports on the box (transformers 5.20.2). Weights download through the HF hub (e.g. `depth-anything/Depth-Anything-V2-Small-hf`), so no new package is needed.

## IDEA-023 Verbatim native 3D-grounding prompt for zero-shot (exp001 variant)
- source: research/literature/2025_qwen3_vl.md
- change: for `style: qwen3d`, use the Qwen3-VL report's App. B.5 template, one query per object: `Locate the {name} in the provided image and output their positions and dimensions using 3D bounding boxes. The results must be in the JSON format: [{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}]`. Drop the scene sentence, the text intrinsics and the `FRAME` sentence. For Task B keep the action line. Single change: the prompt text.
- expected: our current qwen3d prompt adds text K, a frame sentence and multiple objects per query. None of these appears in Qwen3-VL's 3D training or eval format, which canonicalises the intrinsics away into a virtual camera. A clear drop in the Task A val_id error means our additions hurt the zero-shot readout, so exp001's "native zero-shot" conclusion should use this prompt. No change means the prompt is not the bottleneck.
- cost: worktree prompt option plus about 0.3 GPU h eval (one query per object)
- status: proposed

## IDEA-024 Frame/scale audit of zero-shot Task A predictions (diagnostic)
- source: research/literature/2025_qwen3_vl.md
- change: analysis of exp001 val_id Task A outputs, no GPU. (i) Project the predicted and GT centres with K and report the pixel error separately from the depth (z) error. (ii) Fit one global depth scale s (z_gt ≈ s·z_pred) and per-axis sign/scale, then report the error after correction.
- discriminates: s ≈ 1.07 with a large error drop points to a Cube R-CNN/Omni3D virtual-depth convention (f_v = H_v = 512 vs our f/H ≈ 1.072), which is fixable post hoc. A small pixel error with a large z error means metric depth fails (→ IDEA-014/015). A large pixel error means poor 2D grounding of small primitives (→ IDEA-004). A sign flip on an axis would mean a frame-convention bug, although the cookbook projection says Qwen uses the same OpenCV frame as us.
- cost: about 0.5 day, no GPU. Needs the exp001 pred file; if `agent.sh` cannot read it, run it as a read-out inside a worktree eval job.
- note (review of exp001_zeroshot_qwen3d): priority up. On val_id, swap scores *better* than normal (Task A −0.051 m, CI [−0.074, −0.028]). An axis sign flip or offset is the one simple mechanism under which image-driven predictions lose to wrong-image predictions. Run (i)-(ii) together with IDEA-028 in the same read-out.
- note (planner, 2026-10-24, round 2): the "swap beats normal" motivation came from exp001, which is invalid as specified (48-token cap). Run this audit on exp002's preds, and only if exp002 reproduces swap ≥ normal with missing_rate ≤ 5% in both modes.
- refinement (researcher, 2026-10-24, research/literature/2023_omni3d_cube_rcnn.md): Cube R-CNN-style virtual depth (f_v = H_v) amounts to a virtual camera whose focal is H in original pixels. Under it, X_v = X and Y_v = Y, and only z is rescaled: z_v = z·H/f. For our K (fy ≈ 480.4, H = 448) the virtual-camera hypothesis therefore predicts z_gt ≈ 1.072·z_pred with **x and y scales ≈ 1**, a fitted uniform xyz scale would point to something else. A test with no fitting: project the predictions with K_v (fx = fy = 448, cx = cy = 224) and with our K. Under the hypothesis, K_v gives the lower pixel error, and with our K the pixels are over-scaled radially by ~1.072 about (224, 224). Decision rule: report raw outputs as the exp001 primary. Report the z×1.072 variant only as a labelled post-hoc diagnostic, and only if both tests agree. The bias is ≤ ~0.09 m at our depths, so it cannot explain a ~0.7 m error. It does not apply to SFT models (constant K is absorbed).
- status: proposed

## IDEA-025 Counterfactual impulse scaling (action-dependence test)
- source: research/literature/2026_quantiphy.md
- change: eval-only with the exp000 adapter. Keep the image fixed and multiply the impulse number in the action text by α ∈ {0.5, 2}. Read out the target's predicted displacement d̂(α) = ‖pred_B − own pred_A‖ and the median log-log slope log(d̂(2)/d̂(0.5))/log 4, split by sliders vs balls, on val_id and val_ood_forces. Flag episodes whose scaled speed leaves train [0.3, 1.6] m/s. Single change: the impulse value in the prompt.
- discriminates: slope ≈ 2 (d ∝ v² for on-table sliding) means physics-consistent use of the impulse. Slope ≈ 0 means anchoring to the training displacement prior (QuantiPhy: counterfactual priors drop MRA by about 70-80%). 0 < slope < 2 means partial use. It is a within-episode intervention, so it avoids the mass, edge-distance and mode confounds of IDEA-008's cross-episode regression. Run it in the same job as IDEA-008.
- cost: worktree prompt option plus about 0.15 GPU h. Counterfactual predictions have no GT, so log the slope to job.log and never write them to `pred/normal.jsonl`. A scored version needs re-simulated counterfactual GT (`op_gen` change, human).
- status: proposed

## IDEA-026 Counterfactual direction flip (action-dependence test)
- source: research/literature/2026_quantiphy.md
- change: eval-only with the exp000 adapter, in the same job as IDEA-025. Swap the direction phrase in the action text to its opposite (left↔right, away↔toward) and keep the image and impulse fixed. Report the fraction of targets whose predicted displacement, projected on the *stated* camera-frame direction, is positive: original vs flipped prompt.
- discriminates: ≳90% sign agreement after the flip means the model reads the action direction. Lower agreement means it infers the direction from the scene layout. That is a scene-prior shortcut, plausible because `aim_at_other_prob: 0.4` makes "toward another object" guessable from the image.
- cost: about 0.05 GPU h on top of IDEA-025
- note (planner, 2026-10-24, round 2): a pinned-visible version needs no job.log. Write the flipped-prompt predictions to `pred/cf_flip.jsonl`, which pinned code scores against the *real* GT. If the model reads the direction, Task B target error under the flip should rise well above normal on val_id/val_ood_forces, by about 2× the displacement for on-table sliders. If error(cf_flip) ≈ error(normal), the action direction is ignored. This "action-dependence gap" mirrors image_dependence_gap. The sign-agreement statistic stays a job.log extra, once exp002 confirms log visibility.
- status: proposed

## IDEA-027 Test-time intrinsics-shift eval (crop-zoom val_id, K rewritten in the prompt; diagnostic)
- source: research/literature/2026_camera_robust_3d_tooluse.md
- change: eval-only with the exp000 adapter. Centre-crop the val_id t=0 image by c ∈ {0.8, 0.67}, resize back to 448×448 (effective fx = fy scaled by 1/c, i.e. 1.25× and 1.5×; cx = cy = 224 unchanged), and rewrite fx, fy in the prompt. The GT camera-frame targets are unchanged by a crop. Single change: test image + K text. Write the results to a separate pred file (e.g. `pred/kzoom125.jsonl`), never to `normal.jsonl`.
- discriminates: Task A error close to the uncropped val_id level means the model reads K or reasons projectively. A large drop with z_pred·(1/c) ≈ z_gt means it memorised one pixel→metric map and reads depth off apparent size. Jiang et al. report RGB-only Qwen3-VL-8B keeping only 28% of its Det F1 at K×0.5 (Table 1). It is the baseline any IDEA-019 run must beat, and a direct test of whether text K is used (the supervisor's text-vs-scene concern, on the geometry side).
- cost: worktree infer option (crop + K rewrite), about 0.1 GPU h. Zoom-out (c > 1) would need re-rendering with a different fovy (op_gen config, human), so only zoom-in is proposed.
- priority note: K is fixed in every val split including val_ood_config, so this does not bear on the primary metric. Run it only together with, or just before, IDEA-019.
- status: proposed

## IDEA-028 Read-out of exp001 predictions: why swap beats normal, and what fills val_ood_config (diagnostic)
- source: review of exp001_zeroshot_qwen3d
- change: eval-only read-out of the existing `pred/normal.jsonl` and `pred/swap.jsonl`, with no new generation. Run it as a worktree infer.py option that reads the pred files and prints to job.log. Use a fresh output line, not one after tqdm (see IDEA-029). Per mode × task × split, report:
  - (a) missing_rate, and the unmatched-label rate (raw labels that `_norm` maps to no prompt name), with the 10 most frequent unmatched labels
  - (b) Task A error restricted to objects matched in *both* normal and swap
  - (c) per-axis Pearson r(pred, GT) and std(pred), camera frame
  - (d) a constant-centre prior: the mean GT camera-frame centre of `train` Task A, written as a derived `pred/const_prior.jsonl` so that pinned code scores it
  - (e) for val_ood_config, the histogram of number of objects returned per answer, and the hit-cap count (len == max_new)
- discriminates:
  - Label/format: swap has the lower missing or unmatched rate, and on matched-in-both objects normal ≤ swap.
  - Frame-convention error: r < 0 on some axis under normal (joint with IDEA-024).
  - Variance shrinkage: r ≈ 0+, std(normal) > std(swap), and the constant prior beats normal.
  - For config, (e) separates omission (few objects per answer) from truncation (hit-cap) from label mismatch (a).
- cost: < 0.05 GPU h (model load not needed); about 0.5 day of code
- note (planner, 2026-10-24, round 2): exp001's preds come from main's 48-token harness and are invalid as specified. Retarget this read-out at exp002_zeroshot_qwen3d_rerun's preds, and run it only if exp002 shows missing_rate > 5% on some split or swap ≥ normal on Task A val_id. (a) and (e) are partly answered by exp002's own stats lines.
- status: proposed (conditional on exp002)

## IDEA-029 Fix `op_log` dropping stdout lines that share a physical line with a tqdm frame (tooling)
- source: review of exp001_zeroshot_qwen3d
- change: in pinned `physwm/agent/exec.py:356`, apply the tqdm filter to the segment after the last `\r`, not to the whole line. Alternatively, surface missing_rate in results.md (exp000 review item). Interim worktree workaround: print `"\n"` before every stats line.
- discriminates: n/a. Without this fix, every pre-review logging requirement routed through job.log is unverifiable, and exp001's validity gate failed to be checkable for exactly this reason.
- status: rejected: premise wrong (human, 2026-10-24). exp001's stats lines were never printed because the job ran main's infer.py (executor bug, fixed in d4de761), so op_log was not at fault. exp002 prints its stats lines with a leading newline as a cheap guard and so tests job.log visibility directly.
