# Idea backlog

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`, `needs-human: reason`).
The planner re-ranks this file; the researcher and reviewer append to it.

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
- status: proposed

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
- status: needs-human: new validation seed ranges must be reserved in `configs/research_agent_policy.yaml`

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
- status: proposed
