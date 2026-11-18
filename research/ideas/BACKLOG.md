# Idea backlog — phase 2

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`,
`needs-human: reason`). The planner re-ranks this file; the researcher and reviewer append.
Phase-1 backlog (IDEA-001…062): `research/ideas/BACKLOG_phase1_archive.md`.

## Ranking (planner re-seed, 2026-11-18; after IDEA-103 closed and RESULTS_phase2 §1–§6)

Inputs: `research/RESULTS_phase2.md` (§1 grounding, §2 endpoint, §3 motion attribution, §5
hygiene, §6 order), `research/experiments/idea103_surface_probe/report.md`,
`research/literature/2026_onecanvas.md`, `research/PLAN.md` incl. reading rules, the open
questions in `research/STATE.md`. NG = no GPU, G = GPU. Primary metric =
`val_ood_target_mean` (Task B target, mean over the val OOD splits).
Ordering observation: 103 moved the PE question from "surface copy" to "per-episode range
scale" (`s_ep` 1.043 ± 0.110 vs 0.998 ± 0.015), so the fix arms now test *magnitude/scale
control*, not geometry. One sequencing fact changes the plan: policy v2.0 loads
`val_ood_config_nocam` in **every** job, so 104's generation step is a precondition for
101/102/105, not just an idea — and no arm can be submitted until that split exists on the
chosen backend.

1. **IDEA-101 — planned: exp011_pe_norm_pinned** (G, 1 train arm, 3 GPU-h budget).
   *For:* 103's scale error is 96 % of the PE-on L2 variance with `s_ep` = 1.043 ± 0.110, and the
   one recipe we omitted is exactly the magnitude calibration (`2026_onecanvas.md`, stage-1 norm
   pin). *Against:* OneCanvas pins the PE in a synthetic curriculum at r=256/8B; our 2-epoch,
   r=16, single-view regime may not reach the regime the pin protects, and the gate can still
   re-amplify a bounded delta. *Effect on primary:* −0.01 to −0.025 if the perception share
   (0.0457 m of B val_id, §3) returns and the PE keeps its legacy config edge (0.2495 vs nope
   0.2672); 0 if null; the 3-split mean ceiling is 0.3216 → 0.2985. *Risk:* no protected change,
   but a gate-stuck-at-0 no-op would make a null uninformative → the spec pre-registers a
   `[penorm]` bind check. *Info:* decisive both ways — success sends us to 104/105, null sends us
   to 102 and then 107/108; it also answers "does the PE carry usable in-distribution signal".
2. **IDEA-102** PE lr 50× → 5× (G, 1–2 config-only train arms, 3–6 GPU-h). *For:* cheapest
   possible train-side probe (no code change), and the only arm that separates "branch moves too
   fast" from "branch is the wrong input"; §1's harm happens at 50× lr where the branch reaches a
   large-norm regime first. *Against:* competes with 101 for the same GPU slot and does not test
   the recipe we actually omitted; a monotone-looking curve may need a third arm; 5× may simply
   undertrain in 2 epochs. *Effect on primary:* −0.01…−0.02 only if LR is the whole story; likely
   intermediate. *Risk:* one seed, and lr changes are easy to over-read as a mechanism; no
   protected change. *Info:* high *conditional on 101 being null* — as a first arm it leaves the
   magnitude question open.
3. **IDEA-104** `val_ood_config_nocam` generation + eval-only rescoring (G, ~0.5 h generation +
   ~6 adapters × ~0.3 h ≈ 2.5–3 GPU-h). *For:* every config claim since phase 1 is on a
   camera-confounded legacy split (§5, PLAN rule 6); 103 §4 shows the config column can invert
   the PE story (PE-off radial +0.103 > PE-on), so 101's config read-out is uninterpretable
   without it; the generation half is a hard precondition for all queued arms. *Against:* no
   mechanism by itself; the rescoring pass is only useful once the PE-fix arm exists. *Effect on
   primary:* none directly, but it can move the primary's split set and therefore its value —
   the reason 101's criterion is pre-registered as a paired difference vs s_f1. *Risk:* none to
   protected splits (val_* only, committed config, reserved seeds); risk is forgetting provenance.
   *Info:* highest of any non-training item; unblocks 101/102/105 and settles the phase-2 config
   question.
4. **IDEA-105** Seed-1 pairs for l_f1 / s_f1_nope + seed-2 for the 2×2 winners (G, 3–4 arms,
   ~9–12 GPU-h). *For:* the only floor we have is one repeat of s_f1 (≤0.004 m); any PE-fix gain
   below ~0.008 m cannot be reported without it, and a centre-format floor may be larger.
   *Against:* pure cost, no mechanism; premature until the 101/102 winner is known, and it would
   consume most of a day's 12 GPU-h budget. *Effect on primary:* none; it decides whether 101/102
   effects are real. *Risk:* turns a 0.005 "comparable" into a claim if run carelessly; no
   protected change. *Info:* gates adoption only — it changes what we *report*, not what we try.
5. **IDEA-106** Forecast-horizon read-out from s_f3 / s_f3_nope (NG). *For:* free once the
   3-frame matrix arms land, and it is the only test of whether the 3-frame input forecasts or
   reconstructs (PLAN rule 5); §3's N 0.0268 vs hold-Â 0.0763 shows there is motion signal to
   lose at 0.3 s. *Against:* does not bear on the primary metric or the PE question, and it is
   blocked on s_f3 landing. *Effect on primary:* none. *Risk:* scoring an observed frame would
   measure reconstruction — the spec/read-out must use the first unobserved frame. *Info:*
   moderate; it decides the frame-axis story, not the next training arm.
6. **IDEA-107** n_views = 2 (G, corpus regeneration + 1 arm, ≫10 GPU-h). *For:* triangulation is
   the principled answer to a depth/range ambiguity and would also test whether 103's scale error
   is single-view unidentifiability. *Against:* v3 is a 1-view corpus, so it needs a regeneration
   and a new output root; policy prefers the simpler arm when comparable. *Effect on primary:*
   potentially large on objects/config, unknown on forces. *Risk:* corpus/provenance work, cost.
   *Info:* high but not cheap; correctly last.
7. **IDEA-108** Estimated metric depth at inference (G, eval-only per arm, ~0.5 GPU-h). *For:*
   the honesty gate for every RGB-only claim and the depth-source axis of PLAN decision 5;
   OneCanvas's own GT→DA3 drop is only 0.4–1.3 points, so it is not expected to be an artefact.
   *Against:* cannot be interpreted before 101 says whether the PE uses depth at all.
   *Effect on primary:* small negative to flat. *Risk:* check the estimator loads with the
   current transformers before planning; no new dependency. *Info:* medium; it constrains claims.
8. **IDEA-109** `s_ep` as a standing per-arm read-out (NG). *For:* 103 shows `s_ep` separates a
   scale fix from generic error reduction, and `analyze_pe_surface.py` already emits it.
   *Against:* nothing, it is free. *Effect:* none on the metric. *Info:* makes 101/102
   interpretable; has no version of its own (it is a required read-out in exp011's spec and should
   be quoted in every later report).

Not ranked: anything needing a pinned eval/agent change, a new dependency, a final-test run or a
v3 manifest is `needs-human` by construction. IDEA-103 is closed (done, not supported).

---

## IDEA-101 PE norm-pinned (OneCanvas stage-1 recipe)

- source: `research/RESULTS_phase2.md` §1 (PE-harm hypothesis); `research/literature/2026_onecanvas.md`
  stage-1 "norm pinning".
- hypothesis: the PE harm is feature takeover, not bad geometry. The 136-channel xyz PE, added
  at 50× LR through a zero-init gate, grows until the branch dominates the frozen ViT features,
  so the model reads out coordinates instead of inferring body centres. Pinning the PE vector's
  norm to half the mean visual-feature norm (per forward pass, in feature space between the MLP
  and the gate: the reference norm is `hidden`-dimensional, so a pin in the 136-ch code would not
  be comparable) keeps the signal bounded and should restore Task A toward the no-PE level while
  keeping whatever OOD benefit the PE carries on the config axis.
- cheapest test: one train arm identical to `s_f1` in data, seed, LoRA, schedule and answer
  format; the only change is the norm pin (gate stays zero-init). Infer A/B/N on the validation
  splits and score with the existing paired read-outs.
- expected signature: Task A val_id falls from 0.0996 toward 0.0127; the diagnostic is whether
  the fall is accompanied by a config/objects gain over `s_f1_nope` (0.2672 / 0.1547). If A
  recovers but config does not, the PE carries no usable signal and `s_f1_nope` remains the
  reference. If neither moves, takeover is not the mechanism (route to 102 / 103).
- GPU cost: one training arm plus val inference (same recipe as `s_f1`); set `budget_hours`
  from the recorded `s_f1` wall time. Fits 8 h on LRZ.
- status: **planned: exp011_pe_norm_pinned** (2026-11-18). Spec pins *each* merged token's PE
  vector norm to 0.5 × the mean visual-feature norm per forward pass, in feature space after the
  MLP and before the zero-init gate (a 136-ch-space pin would not be dimensionally comparable);
  gate init, PE lr 50×, LoRA, corpus, seed, style, frames, views and tasks unchanged vs `s_f1`.
  Pre-registered decision variable is Task A val_id (A ≤ 0.031 = ≥ 80 % of the 8.7 cm gap
  closed) with a B val_id guard (≤ 0.2196), plus the IDEA-109 `s_ep` read-out and a
  "did the pin bind" check (final |gate| and the realized added-vector share vs `s_f1`).
  Submission is blocked until `val_ood_config_nocam` exists on the chosen backend (policy v2.0).

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

- status: **done 2026-11-14 (idea103_surface_probe): NOT SUPPORTED.** PE-on error is radial
  (99.3 % of squared error) but with the OPPOSITE sign of a surface read-out (rad +0.037 m
  [ +0.027, +0.047 ] vs −d_near ≈ −0.074 m), does not track d_near across objects within an
  episode (corr −0.11, slope −0.18 [−0.33,−0.03] vs −1), and is instead a per-episode
  multiplicative range/depth-scale distortion: pred ≈ s_ep·gt, s_ep = 1.043 ± 0.110
  (PE off: 0.998 ± 0.015); scale model residual 0.020 m vs 0.100 m error.
  val_ood_config is confounded by the legacy camera shift (PE-off radial +0.103 m > PE-on).
  Consequence: 101/102 target scale calibration; IDEA-109 makes s_ep a standing metric.

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
## IDEA-109 Standing metric: per-episode radial scale factor s_ep on every arm

- source: IDEA-103 verdict (scale distortion, not surface copy).
- hypothesis: any PE/config change that fixes grounding should drive s_ep -> 1.00 with
  episode-to-episode spread <= 0.02 (the PE-off level); s_ep separates scale fixes from
  generic error reductions.
- cheapest test: extend analyze_pe_surface.py with a --table mode emitting s_ep mean/spread
  per arm; call it in the loop's report step for every finished arm (no GPU).
- expected signature: norm-pinned or low-lr PE arms with A-error near PE-off AND s_ep ~ 0.998
  => scale distortion removed; A-error down but s_ep still ~1.04 => different mechanism.
- GPU cost: 0.
- status: proposed (2026-11-14).
