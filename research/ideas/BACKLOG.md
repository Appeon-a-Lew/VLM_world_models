# Idea backlog

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`, `needs-human: reason`).
The planner re-ranks this file; the researcher and reviewer append to it.

## Ranking (planner, 2026-10-25, round 5, re-rank only, no new spec)
Inputs: IDEA-037..042 (V-JEPA, SFT-vs-RL, Thinking in Space, Depth Anything V2). exp005 (repro) is running; exp004 follows on its adapter.
Top 5:
1. IDEA-034 + 035 (planned: exp004_exp000_oracle_anchor): still the highest info per GPU-h; runs right after exp005.
2. IDEA-040 training-answer prior baseline: no GPU, runs now while exp005 trains. Bounds how much of exp000's val_id and forces gain is image-free (links to IDEA-008, 025). Risk: camera-frame prior is weak, so only a floor.
3. IDEA-026 then 025 counterfactual actions on exp000 (~0.15 GPU-h each): tests whether the forces gain (78% of aggregate) uses the action. Needs the exp005 adapter.
4. IDEA-014 (u, v, depth) output (1.4 GPU-h): first geometry lever, conditional on exp004 R1/R2.
5. IDEA-002 displacement output (1.4 GPU-h): conditional on exp004 R3 or on 025/026 showing the action is ignored.
Next: IDEA-042 (DA-V2 vs elevation, replaces the DA half of 022), IDEA-041 (blind-trained arm, after 040 shows whether needed), IDEA-037 (likelihood margin), IDEA-036 (data side done; eval needs exp005 adapter), IDEA-038 (needs a retrain, so it cannot ride on exp005 now), IDEA-039 (low, days of engineering).

## Ranking (planner, 2026-10-24, round 4; spec: exp004_exp000_oracle_anchor)
Inputs: exp003 report + review_post (AMEND adopted), CLAIMS C-005..C-008, literature notes 2025_pixels_to_principles_probing and 2024_physics_context_builders, exec.py / metrics.py (current main), exp003 worktree infer.py / derived.py.

What changed since round 3 (exp003, research/experiments/exp003_exp000_diag_eval/report.md):
- CI-robust: no single config factor reproduces exp000's config failure (C-006). Every diag split beats static.
- CI-solid: the own-perception cost on config is +0.056 [0.048, 0.065] (val_id +0.002). Perception degrades at elev (P 0.55) and corner (P 0.52), not with count (P 0.03, C-008). **IDEA-004 dropped.**
- The dynamics gain falls from −0.113 (id) to −0.028 (config); motion-normalised K goes 0.424 → 0.116. ΔG vs ΔC3 is not CI-separable, and the split depends on the decomposition path (review_post §2 Point 2).
- New planner observation that shapes the next spec: the own-static path *mechanically* shrinks G when perception is poor. With perception error e and true displacement Δ, a perfect Δ̂ = Δ still gives G = ‖e‖ − ‖e − Δ‖, and |G| < ‖Δ‖ by the triangle inequality. With e ≈ 0.13 m and Δ ≈ 0.24 m (config), perfect dynamics would already score K ≈ 0.5, not 1. So part of 0.424 → 0.116 can be a perception artefact. IDEA-035's own-path strata (isonormal/contnormal) would carry the same artefact into every stratum. They should run on the oracle-anchored path of IDEA-034, which merges 034 and 035 into one read-out.
- Routing per review_post: cheap discriminators first, and IDEA-002 is not ranked above the geometry levers on config grounds.
- Pinned constraints (unchanged): derived stems must match `[a-z]*` (exec.py:374); `compare` scores missing episodes as unparsed (exec.py:384-390), so subset read-outs need hybrid files with a neutral filler; `compare --task A … static` is refused; greedy inference is bit-reproducible (C-005).

Top 5 (for / against; effect on primary; cost; risk; information value):
1. **IDEA-034 + IDEA-035, oracle-anchored and physics-stratified → planned: exp004_exp000_oracle_anchor.**
   - For: the reviewer's alternatives 2 and 3 (exp003 review_post §3), in one eval-only job. GT_t0 + (B̂ − Â) removes the additive perception offset. Stratifying by iso / contact / off-table (from GT metadata) separates lost skill from harder physics. Each outcome picks the next 1.4 GPU-h training run (R1/R2 → geometry lever IDEA-014; R3 → IDEA-002; R4 → IDEA-025/026).
   - Against: A/B inconsistency adds noise to B̂ − Â. It biases toward "skill lost", so a gate on :oracleanchor unmoved_mean conditions only the FALSIFIED reading. n_iso on config may be small (precondition ≥ 30). The strata are selected on outcome; normalising by displacement handles magnitude but not selection.
   - Effect on primary: none (eval only). Its value is choosing between two levers worth roughly −0.01 to −0.025 on the primary.
   - Cost: ~0.25 GPU-h (exp003 took 0.214).
   - Risk: privileged GT t=0 in derived files only, never in normal.jsonl. No protected change.
   - Info: highest per GPU-h. Every outcome changes the next spec.
2. **IDEA-026, then IDEA-025: counterfactual actions on exp000 (forces).**
   - For: forces supplies −0.154 of exp000's per-split gain (78% of the aggregate; exp000 review_post §b). QuantiPhy (research/literature/2026_quantiphy.md) shows VLMs anchoring to priors. 026's `pred/cfflip.jsonl` is pinned-scored against the real GT, so it gives an action-dependence gap with a paired CI and needs no job.log. Use stem `cfflip`, not `cf_flip`: exec.py:374.
   - Against: an input intervention, so it is its own spec, not a ride-along. 025's slope is a job.log statistic.
   - Effect: none directly. If the action is ignored, C-002's forces gain is a prior and IDEA-002 is promoted regardless of exp004.
   - Cost: ~0.15 GPU-h each. Risk: none protected. Info: high, and it is the R4 branch of exp004.
3. **IDEA-014 (u, v, depth) output: first geometry improvement attempt.**
   - For: the config perception cost is the CI-solid component (C3 +0.056). Perception degrades on both elev and corner, and IDEA-016 (camera pose) can only address elev. Cube R-CNN and camera_robust_3d_tooluse support the parameterisation (research/literature/2023_omni3d_cube_rcnn.md, 2026_camera_robust_3d_tooluse.md).
   - Against: depth along low-elevation rays may stay viewpoint-tied. A parser and target change is needed (no pinned change, since scoring stays camera-frame xyz).
   - Effect: halving config's perception cost is worth ≈ −0.009 on the primary, more if it also restores the own-path G.
   - Cost: 1.4 GPU-h. Info: high, and its outcome unlocks IDEA-015.
   - Conditional on exp004 R1/R2.
4. **IDEA-002 displacement output: first dynamics improvement attempt.**
   - For: in-distribution K is only 0.424, and forces carries most of the error mass (0.510).
   - Against: on config grounds it is not yet justified (review_post §2 Point 3). The PCB evidence (research/literature/2024_physics_context_builders.md) favours end-to-end and supports 002 only as a dynamics lever. Own centre + delta compounds the Task A error.
   - Effect: −0.01 to −0.03 if a dynamics prior is the bottleneck.
   - Cost: 1.4 GPU-h. Info: high.
   - Promoted by exp004 R3 or by IDEA-025/026 showing the action is ignored.
5. **IDEA-036 two-factor elev × corner split.** For: separates (4) interaction from (5) diffuse. Super-additive perception (Σ ΔC3 0.042 vs 0.054) hints at occlusion. Against: it does not by itself pick a lever. Cost: < 0.1 GPU-h once data exist. **needs-human** (outbox sent).

Then:
6. IDEA-016 camera pose in prompt: elev-only geometry lever, runner-up to 014.
7. IDEA-030 probes (5 sites + controls per research/literature/2025_pixels_to_principles_probing.md): says *where* perception breaks; about 0.5 day of code; worth it before 020/013.
8. IDEA-031 latent floor: bounds the dynamics headroom on forces; ride-along with 025.
9. IDEA-022 Depth-Anything-V2 vs elevation.
10. IDEA-024/028/032: zero-shot read-out. Triggered, but off-objective.
11. IDEA-019 + IDEA-027. 12. IDEA-020. 13. IDEA-013. 14. IDEA-003. 15. IDEA-008, prior baseline only. 16. IDEA-018 (evidence against). 17. IDEA-021. 18. IDEA-023. 19. IDEA-005 (harmful per SD-VLM). 20. IDEA-033, which depends on IDEA-002.
- Dropped: IDEA-004 (count perception fine, C-008). IDEA-007 done via IDEA-017 (exp003).

## Ranking (planner, 2026-10-24, round 3; spec: exp003_exp000_diag_eval)
Inputs: exp002_zeroshot_qwen3d_rerun draft report (post-review pending), exec.py (current main), configs/sim_diag_*.yaml.

What changed since round 2:
- exp002 ran on the snapshot. Its job.log stats lines are visible for all 9 passes, so **job.log prints are a usable read-out channel**. IDEA-025's slope and IDEA-026's sign agreement no longer need a pinned-scored workaround.
- exp002 (zero-shot, context only): (A) is rejected, and swap beats normal on Task A val_id (−0.063, CI [−0.089, −0.036]) with missing ≤ 0.009 in both modes. That meets IDEA-024's condition and IDEA-028's condition. Both concern the zero-shot pipeline only and do not bear on the primary metric.
- Pinned facts that shape the next spec:
  - `compare --task A ... static` is DENIED, because the static baseline holds only Task B records (exec.py:522, 379-385). Per-split Task A numbers are therefore point estimates from results.md.
  - A derived pred file can be used with `compare` only if its stem matches `[a-z]*` (exec.py:374). Use `ownstatic.jsonl`, not `normal_ownstatic.jsonl`.
  - `none` is mandatory for eval_only jobs (image_dependence_gap raises on NaN, exec.py:329-334).

Top 5 (for / against; effect on primary; cost; risk; information value):
1. **IDEA-017 + IDEA-006 read-out → planned: exp003_exp000_diag_eval.** For: the top open question (config 0.269 vs static 0.241). Each diag split is sim_train with one factor changed, so val_id is a matched reference. Greedy decoding gives a reproduction check of 0.3136. Own-static as a derived file (Task A re-labelled as B) splits each factor's effect into perception and dynamics, with paired CIs through `compare exp003 exp003:ownstatic`. Against: single factors can miss an interaction; the spec's outcome (4) covers that case. The zero-shot ranking (count > elev) hints the answer may not be elevation, but that is a different model. Effect: none directly; it picks the config lever. Cost: 1.0 GPU-h budgeted, ~0.3 expected. Risk: none protected. The IDEA-006 part is instrumentation only (derived file + stats lines), with center generation and normal.jsonl unchanged. Info: highest per GPU-h.
2. **IDEA-025 + IDEA-026 counterfactual actions on exp000 (forces).** It is now cheaper, since job.log is visible (exp002), so the slope statistic can be printed. Forces is 78% of exp000's gain. Cost ~0.15 GPU-h. Two prompt interventions: split them into two specs if the reviewer insists. Info: high.
3. **Conditional on exp003's routing:** elev → IDEA-016 (then 014; IDEA-030 probes to locate the failure); count → IDEA-004; corner with dynamics broken → IDEA-002 / IDEA-031. Each is about 1.4 GPU-h, except IDEA-030 and IDEA-031, which are eval-only.
4. **IDEA-002 displacement output.** It rises to #2 if exp003 shows dynamics BROKEN on the implicated factor, or if 025/026 show the action is ignored. Otherwise it waits.
5. **IDEA-024 + IDEA-028 + IDEA-032 zero-shot read-out on exp002 preds, in one job.** Conditions met. It explains a zero-shot anomaly and does not bear on the primary metric. Cost < 0.05 GPU-h plus about 0.5-1 day of code. Low priority. IDEA-032's GT-dispersion caveat also explains why exp002's count > elev ranking is not used as evidence in exp003. In exp003 the dispersion effect is controlled: R_f uses the diff vs static on the same episodes, and exp000 is an informative predictor.

Unchanged below: IDEA-008 (prior baseline only), IDEA-023, IDEA-022, IDEA-020, IDEA-019 + IDEA-027, IDEA-003, IDEA-015, IDEA-013, IDEA-018 (evidence against), IDEA-021, IDEA-005.

## Ranking (planner, 2026-10-24, round 2; spec: exp002_zeroshot_qwen3d_rerun; superseded by round 3)
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
- note (researcher, 2026-10-24, research/literature/2024_physics_context_builders.md):
  - This is the single-model, single-pass analogue of Physics Context Builders (perception text, then reasoning). The model writes its own Task A-style perception into its context before generating the dynamics tokens.
  - PCB evidence favours keeping it end-to-end. In-distribution, the end-to-end SFT 3B model beats every PCB-augmented GPT-4o/Gemini arm (CLEVRER 92.9/98.2/88.7 vs best PCB 75.6/70.8/68.4; tower stability sim 87.6 vs ≤ 85.1; Tables 1, 2, 4, 5). On the only OOD test (20 real images) neither route wins consistently.
  - Do not build a separate two-stage pipeline in which exp000's Task A answer is pasted into its Task B prompt. That prompt is off-distribution for the exp000 adapter, so a null result would be uninterpretable.
  - Format requirement: emit the full current-centre block *before* the deltas, so the oracle-prefix diagnostic IDEA-033 can be run on the same adapter.
  - The PCB evidence is conditional. Decomposition helped mainly where the consumer lacked perception, and predictive questions were never tested. It supports 002 as a dynamics lever (forces, or a factor with SFT-B ≈ own-static in exp003), not as a config/perception fix.
- status: proposed

## IDEA-003 Task mix ablation: Task B only vs A + B
- source: Phase 1 design question. Does grounding supervision help dynamics?
- change: `tasks: [B]` in the SFT config
- status: proposed

## IDEA-004 Neutral object references (set-of-marks IDs instead of colour + shape names)
- source: Slack discussion (semantic vs geometric vs neutral references)
- change: draw numbered marks on the image; refer to objects as "object 1, 2, ..."
- status: rejected: exp003 shows exp000's perception is not degraded by 4–5 objects (Task A 0.019 vs 0.016, C3 +0.001 [−0.000, 0.003], missing 0.000; C-008). Revisit only for a different model.

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
- note (planner, round 3): name the file `pred/ownstatic.jsonl`. `compare` accepts only `exp_id:mode` with mode matching `[a-z]*` (exec.py:374), so an underscore would make it unreachable for paired CIs.
- status: done: exp003_exp000_diag_eval (2026-10-24).
  - Task B beats own-static on all 7 splits (C-007).
  - On config, own-static − static is +0.056 [0.048, 0.065], and the dynamics gain falls to −0.028 (val_id −0.113).
  - missing_rate is 0.000 everywhere for exp000.

## IDEA-007 Single-factor breakdown of val_ood_config
- source: review of exp000_sft_baseline
- change: first, zero-cost. Stratify existing exp000 val_ood_config results by object count (4 vs 5) and camera-elevation bin. If that is ambiguous, generate three small validation splits from the reserved val seed range, each changing one factor: elevation [12, 25] only, count [4, 5] only, corner cluster / min_gap 0.01 only. Then evaluate exp000 on them (eval_only).
- discriminates: camera-frame miscalibration vs object-count / format failure vs dense-cluster dynamics as the cause of the config failure.
- status: done: exp003_exp000_diag_eval (step 2, via IDEA-017).

## IDEA-008 Stratified val_ood_forces analysis + training-prior displacement baseline
- source: review of exp000_sft_baseline
- change: eval-only. Split val_ood_forces by speed mode (weak [0.08, 0.25] vs strong [2.0, 2.8] m/s) and by off-table vs on-table outcome. Regress predicted |Δ| on the impulse. Add a trivial baseline: own Task A position + mean training displacement along the stated push direction.
- discriminates: regression to the training displacement prior (prior baseline ≈ SFT; no slope with impulse) vs partial extrapolation (positive slope within modes) vs gain driven by off-table events.
- extension (researcher, 2026-10-24, research/literature/2023_physion_pp.md):
  - The cross-episode regression is confounded by mass. The action text gives the *impulse* (push.py:157-159), and speed = impulse / (ρ·V), with density ρ ∈ [400, 800] latent and volume V visible.
  - Add log V (from metadata `mass/density`) as a covariate. For sliders, physics predicts ∂log d̂/∂log I = +2 and ∂log d̂/∂log V = −2.
  - A V-coefficient ≈ 0 means the model maps the impulse number to a displacement without inferring mass from size.
  - The latent ρ and μ stay in the residual, so expect a low R² even for a physics-consistent model. IDEA-025's within-episode slope is not affected, because mass cancels there.
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
- status: done: exp003_exp000_diag_eval (2026-10-24). The hypothesis is FALSIFIED, point-estimate only: R_elev 0.417 [0.144, 0.705], R_count 0.273, R_corner 0.396.
  - All three factors are PARTIAL.
  - Outcome (5) "diffuse", point-estimate only (the sum interval [0.38, 1.83] doesn't separate it from (4)).
  - CI-robust: no single factor reproduces the failure (C-006).
  - Follow-ups: IDEA-034/035 (cheap discriminators), IDEA-036 (two-factor split, needs a human).

## IDEA-018 LoRA on the vision encoder too
- source: research/literature/2025_limits_of_finetuning.md
- change: extend `lora.target_modules` to include the ViT attention/MLP layers
- expected: possibly lower ID Task A error. The val_ood_config effect is uncertain, and it may overfit the viewpoint distribution further (the paper adapted the ViT and still failed under visual shift).
- cost: config change plus 1 SFT run (about 1.5 GPU h)
- evidence (researcher, 2026-10-24, research/literature/2026_vlm_physics_interaction.md, App. A.9.3, Fig. 22): in the same group's follow-up (Qwen3-VL-8B, LoRA r = 16), a model trained **without** vision-encoder fine-tuning "performed similarly" and failed to transfer exactly like the default. LoRA ranks 1 and 8 also made no difference. This is direct evidence that ViT adaptation is not the lever for transfer. It lowers the expected value of 018 for the OOD splits, so keep it low-priority.
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
- note (source: review of exp002_zeroshot_qwen3d_rerun): **the trigger condition is met.** exp002 reproduces swap < normal on val_id with missing ≤ 0.009 in both modes: Task A −0.063 [−0.089, −0.036], Task B −0.059 [−0.088, −0.028]. Run it on `exp002_zeroshot_qwen3d_rerun/pred/{normal,swap}.jsonl`.
  - Pure variance shrinkage predicts swap ≈ normal, so this audit's sign/offset fit (ii) is the most direct test of the leading mechanism: an anti-informative transform under the correct image. A uniform scale error does not make wrong-image predictions win.
  - Also report the fitted correction on val_ood_config and val_diag_elev. If one global sign/offset fix closes most of the config error, the config failure (C-004) is a read-out convention problem, not a perception problem.
- status: proposed (trigger met)

## IDEA-025 Counterfactual impulse scaling (action-dependence test)
- source: research/literature/2026_quantiphy.md
- change: eval-only with the exp000 adapter. Keep the image fixed and multiply the impulse number in the action text by α ∈ {0.5, 2}. Read out the target's predicted displacement d̂(α) = ‖pred_B − own pred_A‖ and the median log-log slope log(d̂(2)/d̂(0.5))/log 4, split by sliders vs balls, on val_id and val_ood_forces. Flag episodes whose scaled speed leaves train [0.3, 1.6] m/s. Single change: the impulse value in the prompt.
- discriminates: slope ≈ 2 (d ∝ v² for on-table sliding) means physics-consistent use of the impulse. Slope ≈ 0 means anchoring to the training displacement prior (QuantiPhy: counterfactual priors drop MRA by about 70-80%). 0 < slope < 2 means partial use. It is a within-episode intervention, so it avoids the mass, edge-distance and mode confounds of IDEA-008's cross-episode regression. Run it in the same job as IDEA-008.
- cost: worktree prompt option plus about 0.15 GPU h. Counterfactual predictions have no GT, so log the slope to job.log and never write them to `pred/normal.jsonl`. A scored version needs re-simulated counterfactual GT (`op_gen` change, human).
- note (researcher, 2026-10-24, research/literature/2023_physion_pp.md): the slope-2 expectation holds regardless of the latent density ρ ∈ [400, 800] and friction μ ∈ [0.35, 0.6] (both unobservable from one frame), because they cancel in d̂(2)/d̂(0.5). So 025 stays a clean test even though absolute displacement is ambiguous by up to about 0.45×-3.05× from the latents. Do not add mass covariates here; those belong to the IDEA-008 extension. Related: research/literature/2026_vlm_physics_interaction.md has no direct action-conditioning evidence (its single-step prompts state no action).
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
- note (source: review of exp002_zeroshot_qwen3d_rerun): **the trigger condition is met** (swap < normal on Task A val_id, missing ≤ 0.009 in both modes).
  - (a) and (e) are mostly answered: missing ≤ 0.017 and truncation ≤ 0.023 on every split.
  - The open items are (b), (c) and (d). Their discriminating predictions:
    - (c) r < 0 on some axis under normal → frame transform (joint with IDEA-024).
    - std(swap) < std(normal) and (d) const-prior < normal → differential hedging.
  - Write (d) for *all* val and diag splits, not only val_id, so that pinned code also scores it on val_diag_* (see IDEA-032).
- status: proposed (trigger met; retarget at exp002 preds)

## IDEA-029 Fix `op_log` dropping stdout lines that share a physical line with a tqdm frame (tooling)
- source: review of exp001_zeroshot_qwen3d
- change: in pinned `physwm/agent/exec.py:356`, apply the tqdm filter to the segment after the last `\r`, not to the whole line. Alternatively, surface missing_rate in results.md (exp000 review item). Interim worktree workaround: print `"\n"` before every stats line.
- discriminates: n/a. Without this fix, every pre-review logging requirement routed through job.log is unverifiable, and exp001's validity gate failed to be checkable for exactly this reason.
- status: rejected: premise wrong (human, 2026-10-24). exp001's stats lines were never printed because the job ran main's infer.py (executor bug, fixed in d4de761), so op_log was not at fault. exp002 prints its stats lines with a leading newline as a cheap guard and so tests job.log visibility directly.

## IDEA-030 Linear-probe diagnostic of exp000: is the config failure in the visual representation or in the LoRA readout? (diagnostic)
- source: research/literature/2026_vlm_physics_interaction.md
- background:
  - Schulze Buschoff et al. 2026 (§4.4, App. A.6) find the task variables "highly decodable" by linear probes from visual-token activations at all layers, in base and fine-tuned Qwen3-VL-8B, while behaviour fails to transfer ("competence" without "performance").
  - The NeurReps 2026 workshop paper "Selective Cross-Modal Alignment" claims the same for Qwen2.5-VL-3B in its abstract (position R² = 0.97 from visual tokens). It is abstract-only and unverified (see QUEUE).
- change: an eval-only read-out with the exp000 adapter, as a ride-along in the IDEA-017 job. No prediction change.
  - In worktree infer.py, a forward hook on the Task A pass saves the hidden state at 3 LLM layers (e.g. layers at 1/3, 2/3 and the last) of the merged visual token(s) whose patch contains each object's projected GT centre.
  - Fit closed-form ridge in torch (no sklearn, no new dependency) mapping state → GT camera-frame xyz, on about 1,000 `train` episodes.
  - Report the held-out metric error per split (val_id, val_ood_config, val_diag_elev/count/corner) next to exp000's generated Task A error on the same objects.
  - Write the outputs to job.log or a non-scored file, never to `pred/normal.jsonl`.
- discriminates:
  - Probe error stays near val_id on config/diag_elev while generated Task A degrades (0.016 → 0.136 m): the geometry is in the tokens and the learned text readout is viewpoint-specific. That favours training-side levers (IDEA-019 crops, IDEA-014 uv + depth targets, more viewpoint diversity).
  - Probe error degrades as much: the visual representation itself does not carry metric geometry at unseen elevations. That favours input-side levers (IDEA-020/013 depth PE, IDEA-016/021 camera conditioning).
  - It complements IDEA-017, which says *which factor* breaks; 030 says *where*.
- cost: about 0.5 day of code, about 0.1 GPU-h on top of IDEA-017 (one extra forward pass on about 1k train episodes for probe fitting).
- risk:
  - The token choice matters. Pool over the 2×2 merged-token neighbourhood as a robustness check.
  - Probes fit on train elevations (30-60°) may themselves extrapolate poorly to 12-25°. Also report a probe fit on half of val_diag_elev and tested on the other half: it separates "information absent" from "linear map changes with viewpoint".
- protocol refinement (researcher, 2026-10-24, research/literature/2025_pixels_to_principles_probing.md). Ballout et al. 2025 is the readable probing paper. It covers Qwen2.5-VL 3B/7B/72B (no Qwen3-VL) and binary plausibility labels only: no continuous position or depth, no layer sweep, an MLP probe despite the "linear" wording, a single 80/10/10 split, and **no controls**. Its one usable regularity: vision embeddings before projection ≥ after projection > LLM final layer, in all nine models (Insights 3, 5). Concretely:
  - **Sites (5)**:
    - ViT last block, before the merger; 2×2-pool the 16-px patches onto the merged grid (verify the patch size in the processor config);
    - merger output, i.e. the LLM input embedding;
    - the first LLM layer after the DeepStack injection layers (check `deepstack_visual_indexes`);
    - the middle LLM layer;
    - the last LLM layer.
    - Read every site at the object's merged-token position(s), with the 2×2 neighbourhood as the robustness variant.
    - **No whole-sequence mean pooling** (Ballout's choice), since it erases per-object location.
  - **Prompt invariance**: the image precedes the text (`physwm/vlm/prompts.py:82`), so under causal attention the visual-token states do not depend on the Task A vs Task B prompt or the action line. One pass serves both tasks, but these probes measure t=0 perception only.
  - **Controls** (all cheap):
    - (i) *Selection leakage*: the token is chosen by the projected GT centre, so the choice leaks (u, v). Fit a ridge on the token's grid (row, col) alone, and the main ridge on **blank-image** features at the same positions (+1 forward pass on blank images, about 0.02 GPU-h). Report the main probe's gain over both per axis, with **z as the headline axis**.
    - (ii) *Shuffled targets*: permute xyz across training objects and refit. The held-out error must match the predict-train-mean baseline; if it is lower, there is leakage.
    - (iii) Split train/test **by episode**. Choose λ by 5-fold CV inside `train` only. Use about 2× feature-dim objects (hidden size ≈ 2k for the 2B; verify) or PCA to 256 dims first.
  - **Report**: per split, held-out metric error and R² vs the predict-train-mean baseline, next to exp000's generated Task A error on the same objects.
- status: proposed

## IDEA-031 Latent-property error floor for Task B (derived baseline, no GPU)
- source: research/literature/2023_physion_pp.md
- background:
  - The action states the impulse, and speed = impulse / (ρ·V).
  - Density ρ ∈ [400, 800] and sliding friction μ ∈ [0.35, 0.6] are latent and cannot be inferred from one frame. This is Physion++'s 'w/o property' condition by construction, and every val split shares these ranges.
  - For a Coulomb slider at fixed impulse and visible volume, the true displacement relative to that under mean latents (ρ̄ = 600, μ̄ = 0.475) spans about 0.45×-3.05×. This is the researcher's derivation, not a paper number.
- change: a derived pred file scored by pinned code, with no model change.
  - On an "isolated" subset (the target stays on the table and no other object moves, defined from GT metadata), predict p_t0 + (p_t1 − p_t0)·s for the target. s = (ρ/ρ̄)²·(μ/μ̄) for sliders and s = (ρ/ρ̄)² for balls (rolling friction is constant). Non-targets stay at GT.
  - Write `pred/latent_floor.jsonl`, and write exp000's normal predictions on the same episodes to `pred/normal_isolated.jsonl`, so that `summarize` scores both on an identical subset.
- discriminates:
  - Floor ≳ half of exp000's error on the subset: the room for dynamics gains (IDEA-002 etc.) is limited, and beating the floor needs information beyond image + impulse.
  - Floor small: latent ambiguity is negligible and the residual error is learnable.
  - The floor is also the right yardstick for forces strong pushes, where displacements and thus latent-induced errors are largest.
- cost: about 0.5 day of code, no training. Natural ride-along in the IDEA-017 job, which produces the exp000 preds.
- follow-up if the floor is large: a privileged arm that states speed (m/s) instead of impulse in the action text, analogous to Physion++'s DPI-Net with latents (mass 60.1 → 75.9%). Its gain over exp000 upper-bounds what mass inference could buy. 1 SFT run; not proposed separately yet.
- risk: approximate physics (no rotation, toppling or edge effects), hence the isolated subset only. Any non-zero error of this oracle on that subset is partly the approximation itself, so report it as an approximate floor.
- status: proposed

## IDEA-032 Interpretability controls for the zero-shot factor ranking and the swap anomaly (diagnostic, no GPU)
- source: review of exp002_zeroshot_qwen3d_rerun
- background: exp002 ranks the diag splits by Task A error: count 0.887 > elev 0.771 > val_id 0.564 ≈ corner 0.519. The splits use different episodes, so there is no paired CI. The same run shows that the pipeline loses to a wrong image on val_id (swap < normal). For a near-uninformative predictor, per-split error tracks how far GT lies from the predictor's default output, not perceptual difficulty. The ranking therefore can't be read as "count hurts localisation" yet.
- change: derived read-outs on the existing exp002 preds. No generation and no pinned change.
  - (a) A constant-centre prior: the mean `train` Task A camera-frame centre per object slot. Write it as one derived pred file covering val_id, val_ood_* and val_diag_* (shared with IDEA-028 (d)), so pinned `summarize` and `compare` score it per split. Name it `pred/constprior.jsonl`: `compare` accepts only `exp_id:[a-z]*` (exec.py:374), so IDEA-028's `const_prior` would be rejected. Then run `compare exp002_zeroshot_qwen3d_rerun exp002_zeroshot_qwen3d_rerun:constprior --task A --splits <s>` on each diag split; this is paired within the split, with a CI.
  - (b) Hungarian-matched vs label-matched Task A error per mode on val_id, as a test of label-to-box misassignment.
  - (c) Within-val_id stratification of Task A error by object count. This is a paired, same-distribution count effect. The weighting gap already hints at it: object-pooled normal is 0.564, episode-weighted 0.493.
- discriminates:
  - The exp002 − prior gap stays roughly constant across diag splits → the ranking is a GT-dispersion artefact.
  - The gap grows on count/elev → a genuine factor effect for this pipeline.
  - Hungarian ≪ label-matched under normal → misassignment, which favours IDEA-004 over frame fixes.
- cost: about 0.5 day of code, < 0.05 GPU-h. It can share a job with IDEA-024/028.
- status: proposed

## IDEA-033 Oracle-perception prefix forcing on an IDEA-002 model (dynamics-only diagnostic)
- source: research/literature/2024_physics_context_builders.md
- background: Physics Context Builders never reports a GT-description oracle or a description-without-image arm, so it cannot separate perception error from reasoning error. An IDEA-002 model whose Task B answer lists the current centres before the deltas allows exactly that separation, with no extra training.
- depends on: IDEA-002 trained with the answer order [current-centre block, then deltas].
- change: eval-only, single change at decode time.
  - Pre-fill the assistant answer with the GT t=0 current-centre block (same JSON format as training), then let the model generate only the deltas. Score `GT t0 + Δ̂`.
  - Write `pred/oracleprefix.jsonl`, plus a blank-image variant `pred/oracleblank.jsonl`. Both stems match `[a-z]*` for `compare`. Never write to `normal.jsonl`; this is privileged and diagnostic only.
- discriminates (per split, paired CIs via `compare`):
  - oracleprefix vs `static`: both use oracle perception, so the difference is the learned dynamics skill alone. A null on val_ood_forces means the predicted displacements add nothing beyond staying put. Pair it with IDEA-025/026 to tell an action-blind prior from no dynamics at all.
  - normal vs oracleprefix: how much of the Task B error comes from the model's own perception. A large gap on val_ood_config means perception carries the config failure, which points to geometry levers (IDEA-016/014/020).
  - oracleblank vs oracleprefix: ≈ 0 means the dynamics step reads only the coordinates and the action, not the scene. This is a Task B-specific image-dependence test that the none/blank/swap ablations cannot isolate, because they break perception too.
- cost: about 0.5 day of code (assistant-prefix forcing in worktree infer.py), about 0.15 GPU-h after IDEA-002's SFT.
- risk: the forced prefix uses GT, so the model never has to be consistent with its own (possibly wrong) centres. Report it only as a labelled oracle, never as a model result.
- status: proposed
- note (reviewer, review of exp003_exp000_diag_eval): this diagnoses an IDEA-002 *model*, not exp000. It cannot settle whether exp000's config loss is dynamics. For exp000, use IDEA-034.

## IDEA-034 Oracle-anchored implied displacement for exp000 (second decomposition path, derived)
- source: review of exp003_exp000_diag_eval
- background:
  - exp003 split the config shift (0.139) along one path, oracle → own perception (C3) → own dynamics (G): perception 0.054, dynamics 0.085.
  - Errors are Euclidean. For example, config Task A target error is 0.133, yet it costs only 0.056 in own-static Task B error. So the shares depend on the path, and the conservative intervals overlap: ΔG [0.057, 0.116] vs ΔC3 [0.045, 0.063].
  - The other path, oracle perception plus the model's dynamics, is missing for exp000. IDEA-033 needs an IDEA-002 model.
- change: a derived pred file, `pred/oracleanchor.jsonl` (stem matches `[a-z]*`).
  - For each episode and object, the Task B position is GT_t0 + (B̂ − Â), where Â and B̂ are exp000's own Task A and Task B answers.
  - This is exp000's implied displacement, re-anchored on oracle t=0. It is privileged and diagnostic only, and must never be written to normal.jsonl.
  - It needs a job that re-runs the exp000 adapter (bit-reproducible, C-005) with the derived write added, ~0.2 GPU-h. It would be CPU-only if a derive-only job path existed.
- discriminates (paired `compare`, per split val_id / config / diag):
  - `oracleanchor` vs `static` is dynamics skill with oracle perception. If config ≈ val_id, the config loss lives in perception or its interaction with motion, which favours the geometry levers. If config stays far below val_id, it is a real dynamics loss, which favours IDEA-002.
  - `normal` vs `oracleanchor` is the perception part on the other path. Averaging the two paths gives a Shapley-style share.
- risk: if Â and B̂ have *inconsistent* perception errors (separate generations), B̂ − Â carries perception noise. Check this with unmoved objects: their implied displacement should be ≈ 0, as exp003's Task B vs own-static unmoved_mean suggests (0.132 vs 0.134 on config).
- note (planner, round 4): results.md unmoved_mean of :oracleanchor ≈ mean ‖B̂_i − Â_i‖ over unmoved objects. That is a pinned-visible upper bound on the inconsistency, used as exp004's gate. The noise biases toward "skill lost", so the gate only conditions that reading.
- status: planned: exp004_exp000_oracle_anchor (merged with IDEA-035 on the oracle path).

## IDEA-035 Contact stratification of exp000's dynamics gain: lost skill vs harder multi-body physics (derived)
- source: review of exp003_exp000_diag_eval
- background:
  - exp003's dynamics skill per metre of target motion is K = target_skill(:normal) − target_skill(:ownstatic): val_id 0.424, elev 0.332, count 0.346, corner 0.237, config 0.116.
  - Config and corner cluster objects, and config has 4–5 of them. The target may hit neighbours more often, and multi-body outcomes are harder at any viewpoint.
  - "Lost dynamics skill" and "harder physics" predict the same drop in K.
- change: derived pred files from the exp000 adapter's preds.
  - Using GT metadata, define the "isolated" subset as the episodes where no non-target moves by more than MOVED_THRESH (as in IDEA-031).
  - Write `pred/isonormal.jsonl` (normal records on isolated episodes, own-static records elsewhere). `compare … :isonormal … :ownstatic` then gives N/n_iso × G on the isolated subset with a CI. `compare` scores missing episodes as unparsed (exec.py:384-390), so subset-only files would be diluted by penalties.
  - Write the complement `pred/contnormal.jsonl` the same way.
  - Report n_iso per split. It can share a job with IDEA-034 and IDEA-031.
- discriminates:
  - Isolated-target K on config ≈ val_id's means the config dynamics loss is contact or multi-body, and IDEA-002 is unlikely to fix it.
  - Isolated K also drops on config means a viewpoint- or layout-conditioned dynamics loss.
- risk: the isolated fraction may be small on config or corner, which gives wide CIs. Report the fraction first.
- note (planner, round 4):
  - On the own-static path, perception error shrinks G mechanically (‖e‖ − ‖e − Δ‖ < ‖Δ‖ even when Δ̂ = Δ). Own-path strata would therefore carry the perception artefact into every stratum.
  - exp004 stratifies the *oracle-anchored* prediction (IDEA-034) instead, with static as filler: `isoanchor` / `contanchor` / `offanchor` vs `static`.
  - The strata are three: iso; cont (target on the table, a non-target moved > 0.01 m); off (target off the table at t=1). A merge rule applies if n < 30.
- status: planned: exp004_exp000_oracle_anchor (oracle-path variant).

## IDEA-036 Two-factor diag split elev × corner on the exp000 adapter (diffuse vs interaction)
- source: review of exp003_exp000_diag_eval
- background:
  - exp003 cannot tell outcome (5) "diffuse" (Σ R_f = 1.086) from (4) "interaction": the conservative Σ interval is [0.38, 1.83].
  - On the perception side the single-factor costs look super-additive (Σ ΔC3 0.042 vs config 0.054). On the dynamics side they look sub-additive (Σ ΔG 0.109 vs 0.085). Both are point estimates.
  - Elev × corner is the plausible occlusion pair: a low camera plus a tight cluster. It also has the two largest single R_f.
- change: a human-authored `configs/sim_diag_elevcorner.yaml` (sim_train with elevation and corner changed, count unchanged), 300 episodes from seeds outside every reserved range, then an eval-only exp000 re-run as in exp003.
  - R_{elev+corner} near R_elev + R_corner (0.81) means additive.
  - R_{elev+corner} near 1 (config-like) with count's 0.27 still missing means interaction.
  - The same comparison runs on ΔC3 for perception.
- cost: **needs-human** (a new data config on main). ~0.1 GPU-h eval.
- status: needs-human: new diag config

## IDEA-037 Teacher-forced likelihood margin: true vs perturbed Task B answer, normal vs swap image (VoE-style image-dependence test)
- source: research/literature/2025_vjepa_intuitive_physics.md
- background: violation-of-expectation compares the model's surprise on a possible vs an impossible outcome. Our none/blank/swap ablations break perception and parsing (zero-shot `none` gives no parseable output, exp002), so they conflate perception with dynamics.
- change: eval-only, one change at scoring time. Teacher-force two Task B answers and record the summed token log-likelihood: the GT answer and a perturbed answer (GT centres with the pushed object left at its t=0 position, i.e. "static"). Compute the margin m = logP(GT) - logP(static), under the normal image and under a swapped image.
- discriminates: m(normal) > m(swap) per episode means the model's Task B distribution uses the scene to prefer true motion. m near 0 on val_ood_forces means it does not separate pushed-moves from not-moving. No parsing needed, so it works on zero-shot too.
- cost: ~0.5 day code (scoring loop, no generation), ~0.1 GPU-h. Needs an exp000 adapter (exp005 repro).
- risk: digit-token likelihood is dominated by formatting tokens; compare only coordinate tokens, and report on val_id first.
- status: proposed

## IDEA-038 SFT-length sweep: evaluate intermediate checkpoints of the exp000 recipe on the OOD splits (Task A and B)
- source: research/literature/2025_sft_memorizes_rl_generalizes.md
- background: Chu et al. report that scaling SFT lowers visual recognition and OOD performance. exp000 Task A degrades only on val_ood_config (0.136 vs 0.016-0.024), and the 100-episode smoke SFT overfit (loss 0.6 to 0.02).
- change: in the already-required exp005_exp000_repro re-train, save LoRA checkpoints at ~1/4, 1/2 and the final step (single change: checkpoint saving, no recipe change), then eval each on val_ood_* and the diag splits.
- discriminates: if val_ood_config Task A / B error is lowest at an early checkpoint, SFT length (over-fitting) contributes and early stopping or a regulariser is a cheap lever. If flat or monotonically improving, the config failure is not an SFT-length effect.
- cost: nearly zero extra training; 3 extra eval-only runs ~0.2 GPU-h each. Needs a harness change to save checkpoints (planner decision on whether it counts as changing exp005).
- risk: the learning-rate schedule makes early checkpoints not equivalent to short runs.
- status: proposed

## IDEA-039 Small GRPO/PPO stage on Task B with a distance reward, from a lightly-trained SFT checkpoint
- source: research/literature/2025_sft_memorizes_rl_generalizes.md
- background: Chu et al. find RL generalises OOD where SFT does not, but on symbolic tasks with exact verifiers; 2602.06033 finds RL from interaction does not generalise on physics. Chu also finds RL cannot rescue an overfit SFT checkpoint.
- change: one change: continue from an early checkpoint (see IDEA-038) with RL, reward = negative mean Euclidean target error (parse failure = large penalty), no other change.
- discriminates: whether OOD val error improves beyond SFT for equal total steps on val_ood_config and val_ood_forces.
- cost: high (days of engineering, several GPU-h). Not before IDEA-038 and the geometry levers (IDEA-014/016).
- risk: a geometric blind spot (config) is unlikely to be fixed by reward shaping; continuous reward is closer to a loss than a pass/fail rule.
- status: proposed (low priority)

## IDEA-040 Training-answer prior baseline (Chance-Frequency analogue) as a leaderboard row, Task A and Task B
- source: research/literature/2024_thinking_in_space.md
- background: VSI-Bench scores blind models against a "Chance (Frequency)" baseline (average 34.0), and shows priors alone can look like skill (blind LongVILA-8B 47.5 on object count by answering "2"). We only have `static` (oracle t=0 perception) and the none/blank/swap ablations of an image-trained adapter, which are not a prior baseline. Related to IDEA-032 (constant-centre, zero-shot only) and IDEA-008 (displacement prior on forces).
- change: one change, no GPU: fit on the training episodes the best image-free predictor from the prompt alone (shape/colour names, K, impulse text): (a) the global mean camera-frame centre per object slot, (b) a ridge/kNN regressor on prompt features to the answer. Report Task A and Task B on val_id and all val_ood_* and diag splits with the existing paired-bootstrap `compare`.
- discriminates: how much of exp000's val_id Task B gain (0.155 vs static 0.265) and of its OOD gain on val_ood_forces (0.510 vs 0.664) is reachable without the image. If the prior beats static on forces, the exp000 forces gain is partly a text/training prior (links to IDEA-008, IDEA-025).
- cost: ~0.5 day, no GPU.
- risk: with a random camera pose the camera-frame mean prior is weak, so it may only be a floor; report it anyway.
- status: proposed

## IDEA-041 Blind-trained control arm: same exp000 recipe with no image in training and test
- source: research/literature/2024_thinking_in_space.md
- background: exp000's test-time none/blank/swap ablations (0.486/0.472/0.400 on val_id, all worse than static 0.265) break an adapter that was trained to read image tokens, so they overstate or understate dependence on the image. VSI-Bench's blind protocol is test-time only on off-the-shelf models; for a fine-tuned model the clean version is to train blind.
- change: one change relative to exp000: remove the image from the input in both training and evaluation (everything else identical, text prompt with shape/colour/K/impulse kept). Define the image-dependence gap as blind-trained error minus exp000 error per split.
- discriminates: the real value of the image to Task A and Task B per split, and whether val_ood_config's lost dynamics gain is an image problem (the blind arm would then be roughly equal to exp000 there).
- cost: one re-train, ~1.4 GPU-h (same as exp005) plus ~0.3 GPU-h eval. Needs the torchvision fix first. Lower priority than IDEA-040 which is free.
- risk: blind arm may collapse to the mean answer, which is the informative outcome; compare with IDEA-040 to check consistency.
- status: proposed

## IDEA-042 Estimated-depth variant of IDEA-015: Depth-Anything-V2 metric depth lookup at the GT 2D centre, per split
- source: research/literature/2024_depth_anything_v2.md
- background: IDEA-015 uses GT depth (oracle). SD-VLM and Jiang et al. use predicted depth; Depth Anything V2 is the standard source (metric-indoor fine-tune on Hypersim). No paper measured it on tabletop scenes at ~1 m or across 12-60 degrees elevation.
- change: one change, eval-only, no training: run DA-V2 metric (indoor) on the val RGB renders; at the GT projected object centre take the median predicted depth over the GT mask pixels, unproject with K, add a half-extent offset along the ray as in IDEA-015. Compare to exp000 Task A and to the GT-depth oracle per split (val_id, objects, forces, config, diag_elev/count/corner).
- discriminates: whether a frozen RGB-only geometry model degrades on val_ood_config like exp000 (0.136 vs 0.016-0.024). If it also degrades, depth ambiguity at low elevation or occlusion is the cause and depth injection (IDEA-020) from estimated depth will not fix it. If it stays near the id level, a depth route is promising. Subsumes the Depth-Anything half of IDEA-022 (error vs elevation).
- cost: ~0.5 day, ~0.2 GPU-h. Should not need torchvision, but this is unverified; the HF checkpoint names for the metric models are unverified.
- risk: metric depth from a model trained on real/Hypersim indoor scenes may be biased in scale on our renders; report the bias (median ratio to GT) separately from the spread.
- status: proposed

## IDEA-043 Seed-noise floor for the exp000 recipe: retrain with seed 1 (and inspect exp005 blank tail)
- source: review of exp005_exp000_repro
- background: exp005 matched exp000 within 0.0024 on the primary, but it is one same-seed repeat on a new box, so it is not a seed-noise estimate. The blank ablation differed by +0.055 and flipped the blank-vs-none order (exp005 blank-none +0.027 [+0.011, +0.043]; exp000 -0.014).
- change: (a) zero GPU: list per-episode blank errors on val_id in exp005 (tail count, top-5 episodes, Task B vs Task A). (b) one change: seed 1, otherwise exp005's spec; report primary, per-split, gap and all three ablations.
- cost: (a) none; (b) about 1.4 GPU-h.
- status: proposed

## IDEA-044 Frame ablation for the IDEA-013 xyz position embedding: camera-frame vs table-canonical xyz (output stays camera frame)
- source: research/literature/2025_sr3d.md
- background: SR-3D adds a sinusoidal+MLP encoding of a per-pixel 3D position map to vision tokens and canonicalises it "so that it does not depend on camera pose" (single-view map from DA-V2 relative depth). val_ood_config shifts camera elevation (12-25 deg vs train 30-60).
- change: conditional on IDEA-013 being built. One change relative to the IDEA-013 arm: express the PE xyz in a table-canonical frame (origin on table plane, z up, from GT camera pose) instead of camera frame. Same output format, LoRA recipe and data.
- discriminates: if the config failure is pose dependence, the canonical PE should lower Task A and Task B error on val_ood_config and diag_elev; equal results mean PE content, not frame, is what matters.
- cost: ~1.4 GPU-h for the extra arm, small data-pipeline transform.
- risk: model must learn canonical-to-camera transform for the output; could hurt val_id. SR-3D gives no formula, so our canonicalisation is our own design.
- status: proposed

## IDEA-045 VGGT single-image point-map Task A baseline per split (eval-only)
- source: research/literature/2025_vggt.md
- background: VGGT is the geometry encoder of SpaceMind (IDEA-021/022). The paper reports no quantitative single-view accuracy, and lists degradation under extreme rotation; output scale is learned in a normalised space, not metric.
- change: no training. Run VGGT on each val RGB as a single frame; median point-map xyz over the GT object mask; one global scale factor fitted on train episodes only; add half-extent offset as in IDEA-015/042. Report Task A on val_id, val_ood_*, diag_elev/count/corner via `compare`, alongside exp005 and the IDEA-042 DA-V2 arm.
- discriminates: whether a frozen RGB-only geometry model degrades on val_ood_config like exp000 (0.136 vs 0.016-0.024). If it does, depth/point-map injection from estimated geometry will not fix config; if it stays near id level, IDEA-022 (SpaceMind-style fusion) is promising. Also reports scale bias separately from spread.
- cost: ~0.5 day, ~0.2 GPU-h (1.2B params; install and checkpoint availability unverified).
- risk: scale not metric, so a single global factor may bias per split; report bias and spread separately.
- status: proposed

## IDEA-046 Can RGB-only geometry recover camera elevation? VGGT table-plane fit vs GT elevation per split (eval-only)
- source: research/literature/2025_vggt.md
- background: IDEA-016 feeds camera pose as text, but at test time on real or new data GT pose is not available. VGGT single-frame pose is identity by convention, so elevation must come from a plane fit in its point map.
- change: no training. Fit the table plane to the VGGT point map (mask out objects), derive elevation from the plane normal vs optical axis, compare with GT elevation per split including 12-25 deg.
- discriminates: whether a predicted-elevation text input is feasible (small error) or IDEA-016 can only be an oracle diagnostic (breaks at low elevation).
- cost: ~0.3 day, ~0.1 GPU-h. Shares the VGGT install with IDEA-045.
- risk: at 12-25 deg the table is foreshortened, so the plane fit is poorly conditioned exactly where it matters.
- status: proposed
