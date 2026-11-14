# Phase-2 results (grounding-first world model) — status 2026-11-14

Corpus: **v3** (corrected generator, 11 rendered frames @10 Hz to t=1 s, 1 view,
17 400 episodes; published at HF `Appeon-a-Lew/physwm-tabletop-pushes`, `v3/`).
Model: Qwen3-VL-2B-Instruct, LoRA r per `configs/sft_qwen3vl2b.yaml`, 2 epochs,
eff. batch 16; answer style `box3d` (centre+extents JSON) unless noted;
xyz-PE = OneCanvas-style per-patch metric embedding (136-ch sinusoid, 2-layer MLP,
zero-init gate, 50x LR) added to merged visual tokens when "PE on".
Scoring: target-object mean L2 (m) with 1 m missing penalty; omission-aware
columns (`target_missing`, `*_obs`) per the 2026-11-13 review; paired episode
bootstrap (2000 draws, seed 0) for all differences.

## 1. Headline: the grounding regression is the position embedding

Task A (t=0 grounding), val_id target mean, single seed unless noted:

| | PE off | PE on |
|---|---|---|
| centre (legacy format) | **0.0148** (l_f1) | pending (c_f1, queued) |
| box3d | **0.0127** (s_f1_nope) | 0.0996 (s_f1); seed 1: 0.1012 |

Seed repeat of s_f1 (seed 1) moves every split mean by <= 0.004 m, so the
8.7 cm PE effect is ~20x the training-seed noise. The corpus is exonerated
(l_f1 reproduces phase-1 legacy numbers on v3: B id 0.152 / forces 0.511 /
config 0.270 vs legacy 0.155 / 0.510 / 0.269). The box3d format without PE is
the best grounding *and* the best dynamics model we have (below).

Hypothesis for the PE harm (to test, not yet claimed): patch tokens carry
visible-surface xyz; the model shortcuts to reading them out instead of
inferring body centres (error scale matches surface-to-centre offsets).
OneCanvas's stage-1 PE-norm pinning, which we omitted, exists partly to prevent
feature takeover. PE-fix arms (norm-pinned / lower PE lr / centre-offset
targets) are the next training round.

## 2. Endpoint prediction (Task B, t=+1 s), target mean L2

| arm (seed 0) | val_id | ood_objects | ood_forces | ood_config* |
|---|---|---|---|---|
| s_f1_nope (box3d, PE off) | **0.1423** | **0.1547** | **0.4912** | 0.2672 |
| s_f1 (box3d, PE on) | 0.2146 (seed1 0.2109) | 0.1982 (0.1998) | 0.5170 (0.5198) | 0.2495 (0.2510) |
| l_f1 (centre, PE off) | 0.1519 | 0.1598 | 0.5109 | 0.2697 |
| legacy exp000 (v1 corpus, centre) | 0.155 | 0.161 | 0.510 | 0.269 |

IoU (body-frame, s_f1_nope): 0.65 id / 0.46 objects / 0.65 forces / 0.10 config;
extents L2 0.006 m id. missing_rate 0 except config (0.05-3.6 % depending on
token budget; 64+64/obj budget reduces it to <=0.1 %).

*config = legacy compound shift (camera elev 12-25 deg + 4-5 objects + corner
cluster; provenance-verified). It is NOT the phase-2 config axis; see §5.

## 3. Does prediction add value beyond grounding? (motion attribution)

Paired per-object read-outs (`physwm/eval/analyze_motion.py`), means [95 % CI]:

| val_id | s_f1 (PE on) | s_f1_nope (PE off) |
|---|---|---|
| B error | 0.2146 | 0.1423 |
| own-A-static (hold Â at t=1) | 0.2958 | 0.2669 |
| GT-static (hold GT0) | 0.2657 | 0.2657 |
| privileged transfer GT0+(B̂−Â) | 0.1689 | 0.1401 |
| B − own-static | −0.0812 [−0.0963,−0.0675] | −0.1246 |
| B − GT-static | −0.0510 [−0.0666,−0.0349] | −0.1234 |
| transfer − B | −0.0457 [−0.0513,−0.0398] | −0.0022 |

Reading: future prediction beats both static baselines on every split (forces
most: B 0.517 vs own-static 0.706). With PE on, ~4.6 cm of the B error is
perception (re-anchoring helps); with PE off the anchor is already perfect and
the residual is pure displacement error — i.e. **dynamics modelling is now the
bottleneck, not grounding**. Next-frame task N: 0.0268 (nope) vs hold-Â 0.0763
→ real short-horizon motion signal; still worse than GT-static-hold at 0.1 s
(+0.030), as expected at that horizon.

## 4. Frame-count and forecast axes (pending)

`s_f3` / `s_f3_nope` (frames 0,1,2; N retargeted to the first UNOBSERVED frame,
0.3 s) and `c_f1` (centre + PE on, tasks A,B,N) are queued on LRZ (partition at
capacity, reason Priority). They complete the 2x2 and test whether observing the
first 0.2 s of motion improves B beyond the model's own-static baseline.

## 5. Split hygiene and open items

- val_ood_config results are labelled legacy-compound-shift; `sim_ood_config_nocam.yaml`
  (camera 30-60 deg, same count/cluster) is committed; generation + inference-only
  rescoring of all adapters queued after the matrix.
- Omission: old 64+40/obj budget truncated 4-5-object answers (3.6 % target missing
  on config), inflating config means by ~3 cm and extents by ~4 cm; fixed budget
  64+64/obj + omission-aware columns; s_f1 re-inferred accordingly.
- Seed floor: s_f1 seed 1 done (<=0.004 m shifts); l_f1/s_f1_nope seed 1 queued.
- LRZ H100 single-GPU pace 1.5 s/step vs lab 5090 1.1 s/step: LRZ value is
  parallel arms, not per-step speed.
- Website: https://appeon-a-lew.github.io/VLM_world_models/ streams the full v3
  corpus + preds for s_f1 / s_f1_nope / l_f1 (GT boxes, pred centres, error lines,
  11-frame slider); bundle v2 + meta.json fallback deployed 2026-11-14.

## 6. Next experiments (ordered)

1. PE-fix arm: norm-pinned PE (stage-1 recipe) and/or PE lr 5x, same data/recipe.
2. val_ood_config_nocam generation (~30 min GPU) + inference-only rescoring of all arms.
3. Seed-1 completion for l_f1 / s_f1_nope (queued) -> CI-backed 2x2.
4. Only then: view-count axis (n_views 2/4) and estimated-depth variant.
