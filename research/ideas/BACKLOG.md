# Idea backlog — phase 2

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`,
`needs-human: reason`). The planner re-ranks this file; the researcher and reviewer append.
Phase-1 backlog (IDEA-001…062): `research/ideas/BACKLOG_phase1_archive.md`.

## Ranking (planner re-seed, 2026-11-14; after RESULTS_phase2 §1–§6)

Inputs: `research/RESULTS_phase2.md`, `research/PLAN.md` (incl. reading rules), the open
questions in `research/STATE.md`, phase-1 archive. NG = no GPU, G = GPU.
Grounding observation that orders this list: box3d without PE is the best grounding *and*
best dynamics arm (A val_id 0.0127, B val_id 0.1423) while PE-on costs 8.7 cm on A (§1) and
0.2146 vs 0.1423 on B (§2); with PE off the privileged transfer read-out is ~0 (§3), so
dynamics is the bottleneck. Therefore the PE mechanism is the highest-value question, and the no-GPU probe
that can run while the GPU is busy is ranked alongside the first training arm.

1. **IDEA-103** PE error-geometry probe (NG). Runs now; if PE-on errors are radial it also
   tells 101 what a norm fix should recover, and it sizes 102's arms.
2. **IDEA-101** PE norm-pinned training arm (G, one train arm). The direct test of the
   feature-takeover mechanism and the stage-1 recipe we omitted.
3. **IDEA-102** PE lr sweep, 50× → 5× (G, 1–2 train arms). Config-only alternative to 101;
   falsifies or supports the "PE branch moves too fast" reading.
4. **IDEA-104** `val_ood_config_nocam` generation + eval-only rescoring (G, generation ~30 min
   plus one inference pass per adapter). Required before any phase-2 configuration claim.
5. **IDEA-105** Seed pairs for the 2×2 (G, 2–4 train arms). Calibrates whether 101/102 gains
   are above the seed floor; shares the 104 rescoring pass.
6. **IDEA-106** Forecast-horizon read-out from s_f3 (NG once the queued arms land). Decide if
   3-frame input is forecasting or reconstruction.
7. **IDEA-107** n_views = 2 arm (G, corpus regeneration at 2 views + one train arm). First
   step of the views axis; expensive because v3 is a 1-view corpus.
8. **IDEA-108** Estimated metric depth at inference (G, eval-only per arm). Depth-source axis
   and the gate before any RGB-only claim.

Not ranked: anything that needs a pinned eval/agent change, a new dependency, a final-test
run or a v3 manifest is `needs-human` by construction.

---

## IDEA-101 PE norm-pinned (OneCanvas stage-1 recipe)

- source: `research/RESULTS_phase2.md` §1 (PE-harm hypothesis); `research/literature/2026_onecanvas.md`
  stage-1 "norm pinning".
- hypothesis: the PE harm is feature takeover, not bad geometry. The 136-channel xyz PE, added
  at 50× LR through a zero-init gate, grows until the branch dominates the frozen ViT features,
  so the model reads out coordinates instead of inferring body centres. Pinning the PE vector's
  norm to half the mean visual-feature norm (per forward pass, before the MLP/gate) keeps the
  signal bounded and should restore Task A toward the no-PE level while keeping whatever OOD
  benefit the PE carries on the config axis.
- cheapest test: one train arm identical to `s_f1` in data, seed, LoRA, schedule and answer
  format; the only change is the norm pin (gate stays zero-init). Infer A/B/N on the validation
  splits and score with the existing paired read-outs.
- expected signature: Task A val_id falls from 0.0996 toward 0.0127; the diagnostic is whether
  the fall is accompanied by a config/objects gain over `s_f1_nope` (0.2672 / 0.1547). If A
  recovers but config does not, the PE carries no usable signal and `s_f1_nope` remains the
  reference. If neither moves, takeover is not the mechanism (route to 102 / 103).
- GPU cost: one training arm plus val inference (same recipe as `s_f1`); set `budget_hours`
  from the recorded `s_f1` wall time. Fits 8 h on LRZ.
- status: proposed

## IDEA-102 PE lr sweep arms (50× → 5×)

- source: RESULTS_phase2 §1 ("PE lr 50x"); `2026_onecanvas`.
- hypothesis: the damage is optimization-order, not the PE itself. At 50× the base LR the
  PE+gate reach a large-norm regime before the LoRA adapts; a lower PE LR keeps the unprojection
  informative without letting it swamp the visual features.
- cheapest test: one or two config-only arms, `pe_lr = 5×` (and `10×` if the first is
  intermediate), everything else identical to `s_f1`. No code change beyond the config, so it
  is the cheapest train-side probe of the PE branch.
- expected signature: Task A val_id decreases monotonically as PE LR falls; a 5× arm at or
  below 0.0127 means LR was the whole story and 101 is unnecessary. If 5× lands between the
  two references, the PE carries a small real signal and 101/107 become the follow-ups.
  A flat curve rejects the optimization-order explanation.
- GPU cost: 1–2 training arms plus val inference, same order as `s_f1`; parallelisable on LRZ
  (6 slots), but never the same arm on both backends.
- status: proposed

## IDEA-103 No-GPU probe: are PE-on errors aligned with the view ray / surface-to-centre offset?

- source: RESULTS_phase2 §1 ("error scale matches surface-to-centre offsets" — currently a
  hypothesis, not a claim).
- hypothesis: with PE on, the model predicts the **visible surface** xyz of the target rather
  than its body centre. Per-object Task A error should then be dominated by the component
  along the camera ray, with magnitude ≈ the object's surface-to-centre offset along that ray,
  and should flip sign/scale with the mask-centre vs body-centre offset.
- cheapest test: paired per-object analysis on the existing `s_f1` and `s_f1_nope` predictions
  (same episodes, `analyze_motion`-style pairing). Decompose the error into radial (along the
  predicted/GT view ray) and tangential components; regress signed radial error on half-extent
  projected along the ray and on the mask-centroid offset; repeat per split. No generation, no
  model load.
- expected signature: PE-on radial error ≫ tangential, with a slope near 1 against the
  ray-projected half-extent; PE-off shows no such alignment and a much smaller radial term.
  Tangential dominance instead would point at 2D grounding/binding and deprioritise 101.
- GPU cost: none (CPU analysis on existing prediction files).
- status: proposed

## IDEA-104 `val_ood_config_nocam` generation + inference-only rescoring of all adapters

- source: RESULTS_phase2 §5; PLAN decision 8 and reading rule 6;
  `configs/sim_ood_config_nocam.yaml` (committed).
- hypothesis: the legacy compound config split (camera 12–25° + count + corner) measures a
  retired OOD axis. On the phase-2 axis (camera 30–60°, count/placement only) the grounding
  gap is much smaller, and any remaining PE or format effect is attributable to configuration
  rather than camera extrapolation.
- cheapest test: generate the split from the committed config with reserved seeds, then run
  inference only on every existing adapter (`s_f1`, `s_f1_nope`, `l_f1`, `c_f1` when present,
  plus PE-fix arms as they land) and score with the same paired read-outs. No training.
- expected signature: `s_f1_nope` config_nocam near its objects level (0.1547); a PE-on gap
  that survives here is a configuration effect, not a camera effect. If the legacy 0.2495 /
  0.2672 ordering does not reproduce, all phase-1 config language stays archived.
- GPU cost: generation ~30 min GPU (RESULTS §6), plus one inference pass per adapter
  (eval-only); parallelise across LRZ slots.
- status: proposed

## IDEA-105 Complete seed-1 pairs (l_f1, s_f1_nope) + seed-2 for the 2×2 winners

- source: RESULTS_phase2 §1 and §5 (only s_f1 has a repeat so far).
- hypothesis: the 2×2 (format × PE) differences and any PE-fix gain are larger than the
  training-seed floor. The s_f1 seed repeat bounds that floor at ≤0.004 m per split, but it is
  one arm, one repeat.
- cheapest test: retrain `l_f1` and `s_f1_nope` at seed 1 with identical recipe; add seed 2 for
  the winning PE-fix arm from 101/102 and for `s_f1_nope`. Same eval protocol; reuse the 104
  rescoring pass.
- expected signature: seed spread ≤0.004 m as seen for s_f1; a PE-fix gain smaller than ~2×
  the observed spread is not established and must not be reported as an improvement. A larger
  spread on the centre-style arm would demand a seed floor per format.
- GPU cost: 2–4 training arms plus inference; the largest training-only item in the queue, so
  it follows the first PE-fix read-out.
- status: proposed

## IDEA-106 Forecast-horizon analysis from s_f3 (N at 0.3 s vs own-static)

- source: RESULTS_phase2 §3–§4; PLAN reading rule 5 and sub-goal decision 6.
- hypothesis: the short-horizon motion signal is real but decays. At 0.1 s N beats holding Â
  (0.0268 vs 0.0763), while at the first unobserved frame of a 3-frame arm (0.3 s) the margin
  should shrink and may vanish; if it vanishes, multi-frame input improves B only through a
  better state estimate, not through learned forecasting.
- cheapest test: when `s_f3` / `s_f3_nope` finish, run the paired read-out of N at 0.3 s
  against own-A-static at the same horizon, scored on the first **unobserved** frame only (never
  an observed frame, which measures reconstruction). No new GPU work.
- expected signature: N − own-static remains negative but smaller in magnitude than at 0.1 s,
  with a CI that may include 0 at 0.3 s. A sign flip would say the model cannot forecast and
  that the N head is a reconstruction head.
- GPU cost: none beyond the already-queued s_f3 arms.
- status: proposed

## IDEA-107 n_views = 2 arm

- source: RESULTS_phase2 §6; PLAN decisions 7 and sequencing step 6.
- hypothesis: if the PE failure (103) is depth/ray ambiguity, a second calibrated view resolves
  it without a new mechanism: two rays triangulate the body centre, so both grounding and the
  displacement anchored on it should improve — most on objects/config, least on forces.
- cheapest test: one arm with `views: "0,1"`, otherwise identical to the chosen reference
  (`s_f1_nope`, and later the best PE-fix arm). The generator already supports `n_views`, but
  the v3 corpus is 1-view, so this needs a 2-view corpus in a new output root first.
- expected signature: Task A improves beyond the no-PE level and the radial error component
  from 103 shrinks; B improves mainly where the anchor was the bottleneck. If the gain matches
  the PE-fix gain, prefer the simpler single-view arm (policy: simpler when comparable).
- GPU cost: 2-view corpus regeneration (scales with frames × views) plus one training arm;
  the most expensive plan item and why it is last before 108.
- status: proposed

## IDEA-108 Estimated metric depth (Depth Anything v2 metric) replacing GT depth at inference only

- source: RESULTS_phase2 §6; PLAN decision 5 and "later axes".
- hypothesis: part of the PE-on harm is that exact GT surface depth makes the surface read-out
  trivially available; realistic depth noise removes some of that shortcut. Independently, any
  RGB-only claim requires the estimated-depth variant, so the arm doubles as the honesty check
  on the whole grounding pipeline.
- cheapest test: inference-only swap of the depth source (Depth Anything v2 metric, GT depth
  replaced but K/cam2world unchanged) on `s_f1_nope` and the best PE arm; score A/B/N with the
  paired read-outs and report depth bias vs GT separately from spread.
- expected signature: if 103 shows radial errors, estimated depth raises them (the PE reads
  depth); if it degrades all components evenly, depth quality is the limit. A near-flat result
  strengthens every RGB-only claim and lets 107 use estimated depth too.
- GPU cost: eval-only per arm, no training. Check that the estimator loads with the current
  `transformers` install before planning (Depth Anything v2 did in phase 1; no new dependency).
- status: proposed