# Research state

_Maintained by the research agent; the human may edit it at any time._

> **HUMAN NOTE (2026-10-25, box moved to nc-10-4-3; new ablation modes; uvz style).**
> - nc-10-1-1's GPU was taken by another user's jobs, so the executor moved to **nc-10-4-3** (RTX 5080). `data`, `agent` (ledger, pins, baselines) and `experiments` (incl. the exp005 adapter) were copied **byte-for-byte** (checksums identical), and the venv matches nc-10-1-1's `pip freeze` exactly. Nothing was lost: paths are unchanged, the GPU budget carries over, and `compare`/`results` on exp004–007 work. Re-pinned. Our files on nc-10-1-1 were deleted (fair use).
> - If nc-10-4-3 gets busy too, report it in the outbox as before; the human will move the box again.
> - `scripts/check_spec.py` accepts `style: uvz` (exp008 unblocked).
> - **Counterfactual modes allowed (IDEA-025/026):** `ablations` may now include `cf_flip`, `cf_impulse_0.5` and `cf_impulse_2` besides none/blank/swap (exec.py `ABLATION_MODES`, re-pinned). Each runs `infer.py --image-mode <mode>` on val_id and writes `pred/<mode>.jsonl`, scored by the pinned scorer and listed in results.md. `compare` accepts `exp_id:cf_flip` etc. Main's `infer.py` only knows none/blank/swap, so the experiment's worktree must implement the cf modes (prompts.py / infer.py argparse choices). They count as the experiment's one intentional change if they are new code.
> - The old nc-11-1-2 came back online; our leftover dir there was deleted. The exp000–003 files are therefore definitely gone (they were not recoverable to the shared home because its quota is full).

> **HUMAN NOTE (2026-10-24, GPU box moved to nc-10-1-1).** nc-11-1-2 went offline ("No route to host"). The executor now runs on **nc-10-1-1** (RTX 5080, same stack: torch 2.11, transformers 5.20.2, peft 0.21.2, MuJoCo 3.17; 13/13 tests pass). Consequences:
> - **Everything stored only on the old box is unreachable**: the exp000–003 adapters, predictions, job logs and the GPU ledger. The reports in `research/experiments/` and the numbers in this file remain the record. The ledger restarts at 0, `scripts/agent.sh results/log/compare` on exp000–003 will fail, and `experiments` in `status` is empty.
> - **All data was regenerated** from the same configs and seeds. meta.json, traj.npz, depth and seg are byte-identical; the RGB PNGs differ by ±1/255 in ≤8 pixels per image (EGL render noise, also seen between two runs on the same box). Static val OOD is 0.3789670302327517, the same as before, and all per-split static/const-vel baselines match.
> - The final-test manifest is now **v2** (human-approved): RGB files enter its hash by name only. Re-pinned.
> - **exp004 cannot run as written**: its `lora:` points at the exp000 adapter, which is gone. First re-train exp000 unchanged under a new id (e.g. `exp005_exp000_repro`, ~1.4 GPU-h), then point exp004's `lora:` at that adapter. The repro also checks whether the new box reproduces exp000's numbers within noise. Report any drift; don't hide it.
> - **IDEA-036 is done on the data side**: `val_diag_elev_corner` (300 episodes, `configs/sim_diag_elev_corner.yaml`, seeds 8.8M) is in `diagnostic_splits`, inferred in every job and scorable with `compare --splits`. Static 0.279, const-vel 0.574.
> - Another lab user may be using the GPU's compute (seen at 91 % utilisation, <1 GB memory). Jobs still launch, but they may run slower than the old timings.

> **HUMAN NOTE (2026-10-24, executor bug, fixed in d4de761 and re-pinned).** Until now, every job ran **main's** `physwm` code, not the experiment snapshot. `python -m` puts the cwd (MAIN) on sys.path ahead of PYTHONPATH. Consequences:
> - exp000 is valid, since it changed no code.
> - **exp001 did not run its reviewed harness change.** It used main's infer.py, with 48 tokens per object instead of the 128-token qwen3d cap and no per-split stats print. That is why the stats lines are missing from job.log; `op_log` is not at fault. The val_ood_config Task A median of exactly 1.000 m is likely truncation under the 48-token cap.
> - exp001's numbers therefore describe main's harness, not the spec. Mark it invalid as specified, and re-run the same spec under a new id (e.g. `exp002_zeroshot_qwen3d_rerun`). The run is cheap (0.35 GPU-h). Jobs now run from the snapshot dir, so worktree code changes take effect.
> - The outbox request `exp001_joblog_missing_rate.md` is answered by this note: the lines were never printed.

## Objective
Improve the Task B target error on the validation OOD splits (`val_ood_objects`, `val_ood_forces`, `val_ood_config`) relative to `static` and `exp000_sft_baseline`, and measure the image-dependence gap.

## Leaderboard (val, metres; lower is better)
| exp | primary (val OOD target mean) | val_id target mean | Task A val_id | image gap | parse | verdict |
|---|---|---|---|---|---|---|
| static | 0.379 | 0.265 | – | – | – | trivial baseline (oracle t=0 perception) |
| exp000_sft_baseline | 0.314 (−0.065 vs static, CI [−0.075, −0.056]) | 0.155 | 0.016 | 0.331 | 1.000 | reference; gain mostly from val_ood_forces; val_ood_config worse than static |
| ~~exp001_zeroshot_qwen3d~~ | ~~0.755~~ | ~~0.600~~ | ~~0.619~~ | – | – | **invalid as specified**: ran main's infer.py (48-token cap, no stats); superseded by exp002_zeroshot_qwen3d_rerun |
| exp002_zeroshot_qwen3d_rerun | 0.735 (+0.356 vs static, CI [0.338, 0.375]) | 0.561 | 0.564 | uninterpretable (`none` missing 0.998); swap−normal A −0.063 [−0.089, −0.036] | 0.996 min | zero-shot reference; worse than static and exp000 on every split (C-004); (A) rejected: the correct image is worse than a swapped one |

| exp005_exp000_repro | 0.316 (−0.063 vs static, CI [−0.073, −0.054]) | 0.157 | 0.015 | 0.343 | 1.000 | exp000 retrained unchanged on nc-10-1-1: REPRODUCED (tentative, C-009); now the adapter-bearing reference; blank-ablation order flipped vs exp000 |

| exp004_exp000_oracle_anchor | 0.316 (bit-exact repro of exp005; eval-only, privileged diagnostic) | 0.157 | 0.015 | 0.343 | 1.000 | oracle-anchored skill K: id 0.420, config 0.152; iso rho 0.451 (point estimate only, CI crosses 0.5 and 0.75); stratum mix explains about 1% (C-010, tentative); lever choice amended, see Known facts |

Per-split Task B target mean (exp000 / static): objects 0.161 / 0.232, forces 0.510 / 0.664, config 0.269 / 0.241.

## Known facts (Phase 1 setup, 2026-10-24)
- Pushed objects mostly stop within ~0.2 s because of friction. Constant velocity extrapolated from 0.1 s is worse than static.
- Balls roll; about a third of pushed balls leave the table, which produces large errors. Look at medians as well as means.
- In a smoke test, zero-shot Qwen3-VL-2B with the `center` prompt outputs `[0,0,0]`. The native `bbox_3d` prompt gives real but poor estimates (~0.7 m error at t = 0).
- SFT overfits 100 episodes (loss 0.6 → 0.02), so the pipeline works.
- exp000: SFT Task B depends heavily on the image. no-image / blank / swap give 0.486 / 0.472 / 0.400 on val_id, all worse than static (0.265) (C-001).
- exp000: Task A perception holds on objects and forces (0.024 / 0.016) but degrades on val_ood_config (0.136).
- exp001 is invalid as specified (see HUMAN NOTE); exp002 re-ran it with the reviewed harness.
- exp002 (zero-shot Qwen3-VL-2B, qwen3d prompt, valid; gate passed, missing ≤ 0.017 on every val split):
  - Worse than static and exp000 on every val OOD split, for both tasks (C-004, pipeline-level).
  - On val_id, swap beats normal. Task A −0.063 [−0.089, −0.036], Task B −0.059 [−0.088, −0.028]. This is not a missing artefact. The candidate mechanisms are a frame flip or offset, or hedging toward a default under swap (IDEA-024/028, triggers met).
  - `none` yields no parseable output, so `image_dependence_gap` is uninterpretable zero-shot. Use swap.
  - val_ood_config Task A is 0.972 with missing 0.017: a real pipeline-level error, not truncation.
  - Raw Task A on the diag splits: count 0.887, elev 0.771, corner 0.519, against val_id 0.564. This may reflect distance from the model's default output rather than perception (IDEA-032 constant-centre baseline).
- exp003 (exp000 adapter on the diag splits; reproduces exp000 bit-exactly, C-005):
  - No single config factor reproduces exp000's config failure. Every single-factor split beats static (elev −0.053, count −0.072, corner −0.055), while config is +0.028 [0.013, 0.042] (C-006).
  - Shares of the config shift: R_elev 0.417, R_count 0.273, R_corner 0.396. All three are partial, and the result is point-estimate "diffuse".
  - exp000 Task A perception: count 0.019 (no degradation, C-008 tentative), elev 0.082, corner 0.078, config 0.136.
  - Task B beats own-perception static on all 7 splits (C-007).
  - On config, the own-perception cost is +0.056 [0.048, 0.065] (CI-solid). The dynamics gain falls from −0.113 (id) to −0.028. Motion-normalised skill K is 0.424 (id) vs 0.116 (config). Which part is larger isn't CI-separable.
  - missing_rate is 0.000 everywhere for exp000.
- `compare` weights each episode, while results.md all_mean weights each object. They can differ by several cm, e.g. exp002 Task A val_id 0.493 vs 0.564.
- Until d4de761, all jobs ran main's `physwm`, not the snapshot. Any earlier experiment with a code change must be re-run. exp000 had none, so it is valid.
- `sim_ood_config` changes 3 factors at once: camera elevation 12–25° (train 30–60°), 4–5 objects (train 1–3), corner cluster. The intrinsics in the prompt are constant across episodes.

## Queue
1. exp002_zeroshot_qwen3d_rerun: done (0.722 GPU h), reported; post-review AMEND adopted.
2. exp003_exp000_diag_eval: done (0.214 GPU h), reported; post-review AMEND adopted. Next: the planner picks between the cheap discriminators IDEA-034 (re-anchor on true t=0) / IDEA-035 (contact split) and the first improvement attempts. IDEA-036 (elev × corner split) needs a human data config.

   Older queue: IDEA-025/026, IDEA-006, IDEA-002. Zero-shot follow-ups IDEA-024/028/032 are triggered, but they are diagnostics of the zero-shot pipeline, not of the primary objective.
3. Cheap diagnostics from the exp000 review (IDEA-006/007/008), mostly analysis or tooling, not GPU. IDEA-009 done by human; IDEA-017 unblocked (diag splits exist; exp000 numbers need an eval_only spec with the exp000 adapter).

## Tooling (human, 2026-10-24)
- `compare` now reports `per_split_ci95` / `per_split_verdict`, takes `--splits` (any val_* or val_diag_* split), `--task A|B`, and `exp_id:mode` for ablations. Use it for exp001's Task A normal-vs-swap criterion in addition to the fixed 0.05 m margin.
- Diagnostic splits `val_diag_elev`, `val_diag_count`, `val_diag_corner` (300 episodes each) are inferred in every job and listed in results.md, not in the primary metric.
- Qwen3-VL bbox_3d centre/frame/units match our parser (camera frame, metres). Our qwen3d prompt is not Qwen's native template (adds K, frame sentence, multiple objects per query); exp001 measures "Qwen3-VL with our prompt" (IDEA-023).

## exp004 findings (2026-10-25; privileged oracle-anchored diagnostics on the exp005 adapter, tentative, C-010)
- Config dynamics skill falls even with GT t=0 anchoring: K_oa id 0.420, config 0.152 (Q 0.36 vs own-path 0.25). On isolated pushes K_iso 0.482 -> 0.217, rho 0.451, CI [0.14, 0.81] (point-estimate falsified). Consistency gate passed (0.010 m).
- The stratum mix explains about 1% of the loss. Shapley split of the 0.140 config shift: perception about 0.060, dynamics about 0.080 (point estimates).
- Reviewer caveats: much of the iso K drop is a smaller denominator (iso target displacement 0.137 -> 0.103 m, absolute error only 0.071 -> 0.081 m); about half of the config loss in D space is edge-fall episodes; camera-frame push mapping confound. R3's pre-registered IDEA-002 promotion is NOT adopted; run the no-GPU IDEA-050 first (left/right vs away/toward pushes, along/cross-push split). Also IDEA-051, 052.
- iso K on diag splits: elev 0.337, count 0.444, corner 0.444 (config 0.217): points to an interaction.

## exp006 findings (2026-10-25; bit-exact re-run of exp005 adapter, privileged oracle-path diagnostics, tentative)
- Camera-frame hypothesis (away/toward pushes degrade more on config) not supported: config DiD -0.0196 [-0.057, 0.017]; the R-A magnitude (+0.02) is excluded at 95%. R-C holds on the letter, but the CIs are about ±0.03 wide, so about one third of the iso K drop (about +0.010 m error) is unresolved.
- val_diag_elev: left/right error rises, away/toward does not (DiD -0.0349 [-0.071, -0.002]). Fragile (one split, CI edge, reverse order on val_diag_elev_corner). Not read as a contradiction of the camera-frame story; per-axis error (IDEA-053) is the discriminating read-out.
- Off-table recall (privileged) drops id 0.351 -> config 0.090 (drop 0.262 [0.141, 0.382]); also low on objects, count, corner; high on elev 0.42 (IDEA-054). Two readings of R-D in the spec; same route.
- Routing: IDEA-025/026 (counterfactual actions on the exp005 adapter) next; IDEA-016 not promoted; IDEA-002 not promoted.

## exp007 findings (2026-10-25; bit-exact re-run of exp005 adapter, privileged oracle-path per-axis read-out, tentative)
- A-yz (LR yz rise localised to camera y/z, AT rise exceeding LR) not supported and not refuted: did_yz negative with CI including 0 on all three pairs (CIs about ±0.04). The LR probe check was formally invalid (miscalibrated 0.01 m limit vs 0.030 m on val_id); this does not change the route. D-mixed -> IDEA-014 next; IDEA-016 not promoted.
- Descriptive: GT away/toward motion splits y:z about equally on val_id (|y| 0.097, |z| 0.098) but z-dominant on val_ood_config and val_diag_elev (e.g. config |y| 0.030, |z| 0.092); val_diag_corner (no elevation change) stays equal (0.076 / 0.074). Whether the model's predictions follow this is untested; IDEA-055 (per-direction sign-normalised error, no GPU) and the pinned-change IDEA-025/026 are the better-posed tests.
- Lesson: calibrate validity limits from val_id before pre-registering (IDEA-056).

## Unblocked (2026-10-25)
- The torchvision gap was fixed by the human; exp005 reran and finished (1.41 GPU-h). Adapter: `/var/tmp/toyi/wm/experiments/exp005_exp000_repro/model/adapter`. Read exp004 and later exp000-derived diagnostics against exp005, not exp000's recorded numbers. Next: exp004 (prepare in progress), then IDEA-040 (no-GPU prior baseline), IDEA-025/026.

## Next (planner, 2026-10-25 round 9)
exp009_exp005_cf_action planned (IDEA-025+026, eval-only on the exp005 adapter, val_id, budget 0.75 GPU-h): does the model use the action text? Pre-registered readout rules and routes R1 (used -> IDEA-016), R2 (ignored -> IDEA-040 then IDEA-002), R3 (direction only -> IDEA-002) are in the spec. The worktree must implement the cf modes in prompts.py/infer.py; real-episode check is the [cfprompt] job.log lines (no agent.sh op runs custom code). Runners-up: IDEA-040, IDEA-057, IDEA-002.

## Open questions
- Why does zero-shot exp002 do worse with the correct image than with a swapped one: a frame flip or offset, or hedging toward a default output? (IDEA-024/028/032)
- exp000's val_ood_config failure isn't explained by any single factor (exp003). Is it an interaction (elev × corner, IDEA-036) or additive? Is the lost dynamics gain lost skill, or harder multi-body contact (IDEA-035)? Does the split depend on the decomposition path (IDEA-034)?
- Does exp000 use the stated impulse on val_ood_forces, or does it predict training-typical motion? (IDEA-008)
- Are the per-split differences outside the noise? `compare` now gives per-split CIs (IDEA-009 done); exp000 per-split CIs not yet computed.
- If the camera turns out to be the cause, which geometric input fixes it: per-patch xyz (IDEA-013), uv + depth output (IDEA-014), camera pose text (IDEA-016), or K-varying crops (IDEA-019)?

## exp008 findings (2026-10-25)
uvz output (IDEA-014) not supported: primary 0.3264 (exp005 0.3160, +0.0104 [0.0041, 0.0165]); Task A worse on every shifted split, equal on val_id; routing inconclusive (R3/R4 overlap), uvz not adopted. Descriptive (uvz only): shifted-view Task A error is depth (median |dz| 0.012 m val_id -> 0.128 m config), 2D about 1 px. Caveats: real-episode pre-submit check waived; per-split missing rate and off-image fraction unverified. Open ideas: IDEA-057 (exp005 per-axis Task A), IDEA-058 (off-image/missing post hoc), IDEA-059 (center, no FRAME). Executor now nc-10-4-3; cf_flip/cf_impulse modes available, so IDEA-025/026 unblocked.

## exp009 findings (2026-10-25)
cf-action probe on the exp005 adapter, val_id only: flipping the direction raises Task B target error 0.157 -> 0.417 (+0.2595 [0.235, 0.285]); impulse x0.5/x2 raise it +0.056 / +0.037 (CIs above 0) and the median predicted displacement scales with slope 1.10 (physics 2). F-read + I-use -> R1, action used on val_id (tentative, C-012; gate-3 deviation 499/500 changed prompts). Removes "action ignored" for the exp006/007 nulls on val_id only; config/elev/forces untested (IDEA-060). Next per spec: IDEA-016; also IDEA-060 (cf on shifted splits, needs a human change to pinned ablation_split).
