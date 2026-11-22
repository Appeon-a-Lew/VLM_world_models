# Pre-registered decision criteria — M0 screen gate

**Pinned:** 2026-11-21, before any M0 training run. No model results on the M0 arms exist at
pinning time. Amendments only by dated addendum, and only before the first model result exists.

**Scope.** The M0 decision contrast is the paired 3D effect error of the **+both arm vs the
forward-only arm** on the existing push matched counterfactual families. The same rule applies
verbatim to any substituted pair (protocol §4 substitution rule: a clearly-ahead single
component replacing a flat +both). Clearing this gate authorizes Stage B (the pre-registered
tilt fold). It does not authorize anything else.

**Metric.** Paired 3D effect error as computed by `physwm/eval/counterfactual.py` (world frame,
metres): `‖(pred_variant − pred_original) − (gt_variant − gt_original)‖`, averaged over the
**action variants {reverse, half, double}**, **target scope**. `recolor`/`elevation` remain
invariance rails and are never part of the threshold. All-object scope is reported but does not
gate.

**Calibration quantity.** `E_static` = mean static-baseline effect error on the same families
and variants (`python -m physwm.eval.counterfactual --static ...`; computable before any model
runs — it uses no model).

## The threshold

The +both arm must reduce the metric relative to the forward-only arm, **per seed**:

```
R_seed = mean_effect_err(forward-only) − mean_effect_err(+both)
gate:   R_seed ≥ max(0.10 × E_static, 0.02 m)
and     bootstrap CI95 of the paired difference excludes 0   (2000 resamples, family level,
        matching the existing counterfactual.py convention)
```

- The `0.10 × E_static` term is 10 effect-skill points on the repo-native scale
  (`skill = 1 − err/err_static`): the recipe must close ten points more of the
  static→perfect action-response gap than plain fine-tuning.
- The `0.02 m` absolute floor exists precisely so that **two tiny gains can never clear**,
  whatever the family scale. Physics sanity check: with μ ∈ [0.35, 0.6] and initial speeds
  0.3–1.6 m/s, sliding distances v²/(2μg) span ~1–38 cm, so E_static is expected in the
  0.1–0.3 m range and the floor corresponds to roughly 7–20 skill points. A static-shortcut
  model gains exactly 0; only a genuine improvement in action-conditioned response can clear.

**Missing-rate guard.** The reduction counts only if +both's missing-pair rate does not exceed
forward-only's by more than 2 percentage points. Gains bought by refusing to answer are not
gains.

## How the two seeds are judged

1. **Earning seed 2.** The seed-1 screen must show a paired reduction with CI95 excluding 0
   and point estimate ≥ **half** the threshold (≥ max(0.05 × E_static, 0.01 m)) before the two
   seed-2 runs are submitted. If +both misses this bar but a single-component arm meets it on
   its own contrast, the pre-registered substitution rule fires and seed 2 replicates that
   contrast instead.
2. **The gate itself.** Stage B is authorized only if **both seeds individually** satisfy the
   full threshold with CI95 excluding 0. Sign consistency alone does **not** clear the gate: two
   same-direction but sub-threshold gains are recorded as *not cleared*, and reported as such.
3. **Split seeds.** One seed clearing and the other not = not cleared. That borderline pattern
   is exactly the pre-registered trigger for a third seed from the compute reserve — it is a
   spending decision, not an automatic gate pass.

## Reporting requirements

- Each seed's paired bootstrap (point estimate, CI95, per-variant breakdown, target and
  all-object scopes, missing rates) is reported **separately** in every results table.
- Pooled-across-seed numbers may appear only as a secondary display, never as the gate input.
- The static and const_velocity baselines appear in the same table as every arm.

## Pre-launch calibration (before the first M0 training run, on LRZ)

1. Compute E_static on the matched families with `--static`.
2. If measured `E_static < 0.08 m`, the matched-family set is too weak to test the hypothesis:
   regenerate matched families with stronger effects (larger displacement range) and re-measure.
   Recalibration is permitted only while no model result exists.
3. Record the measured E_static and the resulting metre-equivalent threshold in a dated
   addendum below.

## Addenda

- **2026-11-21 (pre-launch fixes; no M0 model results exist yet):**
  1. **Primary screen is PE-free.** The four arms train and infer with `--no-pe`; the
     GT-depth xyz PE is a diagnostic arm run in a separate `*_pe` directory
     (`DEPTH_DIAG=1 scripts/run_m0.sh`), never in the gate comparison.
  2. **Matched-steps pinning procedure.** Every arm is planned (`sft.py --plan`, no GPU)
     before any arm trains; the shared step target is the maximum natural count and is an
     *exact* target (shorter arms extend, longer cap). Cumulative consumed examples per task
     and padded/unpadded/supervised tokens are recorded in the training log.
     Forward exposure per step is *not* matched across arms (IDP examples are additive by
     design); the logs make the exposure auditable and the token-matched secondary remains
     available.
  3. **Gate enforcement.** The gate requires the two preregistered, distinct seed pairs:
     positional pairing is verified against each run's recorded training seed, identical
     files are rejected, and the missing-rate guard applies to `earned` as well as
     `cleared`. A single seed pair can at most earn seed 2; it can never clear the gate.
  4. **Calibration tooling.** Pre-flight E_static comes from the preregistered
     `physwm/eval/counterfactual.py --static`; `paired_arms.py --calibrate` prints the same
     aggregate plus the threshold arithmetic without needing arm prediction files.
  5. **Oracle diagnostic includes support.** `--oracle-context` injects GT per-object
     support alongside the action payload, and predicted support is scored separately
     (`idpmetrics.summarize_ctx`), so a flat oracle result cannot hide a
     support-prediction failure.

- **2026-11-21 (LRZ pre-launch calibration).** LRZ job `5946324` generated 100
  matched push families with the current `sim_train.yaml`, starting at seed
  `9901000`. The static effect error was **0.441371 m**, above the 0.08 m
  calibration floor. The full per-seed threshold is **0.044137 m** and the
  seed-2 spending threshold is **0.022069 m**. These were measured before the
  production screen using the `val` split in the isolated LRZ M0 workspace.
  Matched-family metadata SHA-256:
  `1b7c63beea3f74525bd88b6818b29359d0d157cd870ef60b406fb23c80c1dc02`.

- **2026-11-21 (optimizer-step correction before production).** The smoke ran four
  optimizer steps as expected. The CPU planner had counted examples instead of batches,
  leading to a pending production array configured for 15,000 steps. The array was
  canceled before training. The corrected planner uses the training loader's batch
  count. With 10,000 episodes, 3 tasks, batch size 4, gradient accumulation 4, and
  2 epochs, the shared production target is **3,750 optimizer steps**.

- **2026-11-22 (seed-1 execution backend, before training).** A disposable RTX 5080
  test completed 25 steps of the combined arm at the planned training microbatch
  of 4 and completed Task B inference at batch size 32. All four seed-1 arms are
  assigned to RTX 5080s of the same memory capacity with the same software
  environment. The lab copy carries the audited LRZ episode metadata, the RGB
  inputs needed by the PE-free screen, and the frozen counterfactual sidecars.
  The Qwen model revision is `89644892e4d85e24eaac8bacfd4f463576704203`.
  The four arm definitions, 3,750-step budget, thresholds, and validation cohorts
  are fixed. The H100 array serves as a held fallback while the lab runs.
