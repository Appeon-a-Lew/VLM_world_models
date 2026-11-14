# IDEA-103 — Is the PE-on grounding error a visible-surface read-out?

Status: **probe complete (GPU-free). Provisional verdict: not supported.**
Date: 2026-11-14 · Corpus v3 · Arms `s_f1` (PE on) and `s_f1_nope` (PE off) ·
Splits `val_id` (1–3 objects) and `val_ood_config` (4–5 objects, legacy
compound camera/config shift) · Task A (t = 0) per-object centres.

Code: `physwm/eval/analyze_pe_surface.py`; tests: `tests/test_pe_surface.py`.
GT `meta.json` for 1 000 episodes cached under `/tmp/opencode/metas/`
(2 null object entries in `s_f1/val_ood_config` are skipped as invalid).

```
MUJOCO_GL=egl /tmp/opencode/wm-venv/bin/python -m physwm.eval.analyze_pe_surface \
  --pred-dir /tmp/opencode/predsdl/site/v3/api/pred --meta-cache /tmp/opencode/metas \
  --splits val_id val_ood_config --arm s_f1 --arm s_f1_nope
```

## Method and sign convention

Camera at the origin, OpenCV frame. For GT centre `c` and prediction `p`:

- view ray `r = c/||c||` (points **away** from the camera),
- `e = p − c`, `rad = e·r`, `tan = ||e − rad r||`.

Moving towards the camera is therefore `rad < 0`. `d_near ≥ 0` is the distance
from the body centre to the nearest body-surface point along the ray, computed
as the box support function in the body frame,
`d_near = Σ_a h_a |r_body,a|` (exact for boxes; half-extent box proxy for the
other primitives). A pure **visible-surface** read-out predicts
`p = c − d_near r`, i.e. `rad = −d_near`, `tan = 0`, `rad/d_near = −1`
(equivalently `−rad/d_near = +1`). A **body-centre** read-out predicts
`rad ≈ 0`. Both ratios are listed because the task wording "rad ≈ −d_near,
ratio ≈ 1" mixes the two sign conventions; only one of them can equal +1.

CIs are episode-clustered percentile bootstraps (2 000 draws, seed 0); the
paired difference resamples episodes and pairs object ids present in both arms.

## 1. Per-arm error geometry

| arm | split | n_obj | n_ep | mean \|e\| (m) | mean rad (m) | mean tan (m) | mean \|rad\| (m) | mean d_near (m) | rad/d_near | −rad/d_near | slope rad~d_near | frac rad<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s_f1 (PE on) | val_id | 989 | 500 | 0.1004 [0.0947, 0.1063] | +0.0373 [+0.0270, +0.0472] | 0.0047 [0.0043, 0.0053] | 0.1000 | 0.0736 | **+0.51** | −0.51 | −0.44 [−0.80, −0.09] | 0.40 |
| s_f1 (PE on) | val_ood_config | 2084 | 500 | 0.1092 [0.1045, 0.1141] | +0.0393 [+0.0294, +0.0495] | 0.0101 [0.0093, 0.0109] | 0.1066 | 0.0609 | **+0.65** | −0.65 | −0.22 [−0.56, +0.10] | 0.39 |
| s_f1_nope (PE off) | val_id | 989 | 500 | 0.0134 [0.0126, 0.0144] | −0.0030 [−0.0045, −0.0017] | 0.0029 [0.0025, 0.0034] | 0.0126 | 0.0736 | −0.04 | +0.04 | −0.03 [−0.07, +0.02] | 0.60 |
| s_f1_nope (PE off) | val_ood_config | 2086 | 500 | 0.1369 [0.1290, 0.1450] | +0.1026 [+0.0920, +0.1136] | 0.0107 [0.0098, 0.0116] | 0.1345 | 0.0609 | +1.69 | −1.69 | −0.27 [−0.64, +0.06] | 0.22 |

Paired radial difference, same episodes and object ids:

| split | mean rad(s_f1) − mean rad(s_f1_nope) (m) | n_obj | n_ep |
|---|---|---|---|
| val_id | **+0.0404 [+0.0299, +0.0505]** | 989 | 500 |
| val_ood_config | **−0.0634 [−0.0729, −0.0535]** | 2084 | 500 |

Box-only check (exact surface proxy, not the half-extent proxy):

| arm | split | n_obj | mean rad (m) | mean d_near (m) | rad/d_near |
|---|---|---|---|---|---|
| s_f1 | val_id | 320 | +0.0343 | 0.0748 | +0.46 |
| s_f1 | val_ood_config | 733 | +0.0334 | 0.0593 | +0.56 |
| s_f1_nope | val_id | 320 | −0.0038 | 0.0748 | −0.05 |
| s_f1_nope | val_ood_config | 733 | +0.1044 | 0.0593 | +1.76 |

Per-object ratios make the same point: only **9.0 %** of PE-on val_id objects
fall in the surface band `rad/d_near ∈ [−1.2, −0.8]` (PE-off val_id: 0.5 %);
the median per-object ratio is **+0.53** (PE-off: −0.05).

## 2. The radial signature is real — but it is not surface tracking

PE-on val_id errors are **99.3 % radial** by squared error (`tan` is 4.7 % of
`|e|`, i.e. ~2 px at 1.1 m); PE-off val_id is 83 %. So the first half of the
BACKLOG signature — "radial ≫ tangential" — is confirmed. The specific
surface-copy test fails on every discriminator:

- **Sign.** `rad = −d_near` predicts −0.074 m; observed **+0.037 m**
  (+0.027 … +0.047), the wrong side of the centre (farther from the camera).
- **Ratio.** Surface predicts `rad/d_near = −1`; observed **+0.51**
  (config +0.65). `−rad/d_near` is −0.51, not the +1 that the "toward camera"
  wording would require.
- **Per-object tracking.** Surface predicts a within-episode correlation
  between `rad` and `d_near` near −1. Observed within-episode
  corr = **−0.11**, episode-fixed-effects slope **−0.18 [−0.33, −0.03]**
  (config: corr −0.07, slope −0.25 [−0.42, −0.08]) — 5× too shallow and barely
  better than the centre null (0). The pooled slope in the table (−0.44) is
  only the tiny correlation (−0.09) amplified by the 5× larger spread in `rad`.

## 3. What the radial error actually is: a per-episode range scale

Mechanism diagnostics (`mechanism_stats` in the script):

| arm | split | radial share | within corr(rad, d_near) | within slope [95 % CI] | s_ep mean | s_ep sd | sd(ep-mean pred range) | sd(ep-mean GT range) | within corr(pred, GT range) | scale-model resid mean (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| s_f1 | val_id | 0.993 | −0.11 | −0.18 [−0.33, −0.03] | 1.043 | 0.110 | 0.045 | 0.128 | 0.95 | 0.0198 |
| s_f1 | val_ood_config | 0.969 | −0.07 | −0.25 [−0.42, −0.08] | 1.048 | 0.109 | 0.019 | 0.118 | 0.84 | 0.0491 |
| s_f1_nope | val_id | 0.830 | −0.01 | +0.00 [−0.05, +0.05] | 0.998 | 0.015 | 0.128 | 0.128 | 0.97 | 0.0053 |
| s_f1_nope | val_ood_config | 0.978 | −0.08 | −0.14 [−0.24, −0.04] | 1.102 | 0.123 | 0.117 | 0.118 | 0.96 | 0.0340 |

`s_ep = mean|pred| / mean|GT|` per episode; the scale model is
`p ≈ s_ep · c` fitted per episode. For PE-on val_id that model leaves a
0.020 m residual against a 0.100 m error — it explains ~96 % of the error
variance. Within an episode the predicted range tracks the true range almost
perfectly (corr 0.95) and image-plane directions are essentially exact, but
the **whole scene is rescaled by a per-episode factor `1.043 ± 0.110`**.
The episode-mean predicted range barely varies (sd 0.045 m) while the true
episode-mean range varies with the camera draw (sd 0.128 m; corpus camera
distance 1.099 ± 0.115 m), so the model effectively re-normalises each scene
towards the corpus-canonical ~1.1 m camera distance. PE-off val_id has
`s_ep = 0.998 ± 0.015` and no such failure.

This is a **depth/range calibration** failure, not a copy of a token's
visible-surface xyz: the model gets the ray and the relative depth right and
the absolute metric scale wrong per scene.

## 4. `val_ood_config` caveat

On the legacy compound-shift config split the PE-off arm also has a large
radial term (rad +0.103 m, `rad/d_near = +1.69`, `s_ep = 1.10 ± 0.12`), and
PE-on is *less* radial there (paired diff −0.063 m). That split moves camera
elevation and object count/cluster, so its radial error is confounded with the
same scene-scale prior; it cannot be used as evidence for a PE-specific
surface signature. The `val_id` contrast is the clean one.

## 5. Verdict

**Provisional verdict: NOT SUPPORTED.** In BACKLOG language: the PE-on radial
error *is* much larger than the tangential error (99.3 % of squared error on
val_id, tangential ≈ 5 mm), but its slope against the ray-projected half-extent
is ≈ −0.2…−0.4 (sign opposite to `rad = −d_near`, magnitude far from 1), it
does not track `d_near` per object within an episode (−0.11 correlation), and
the observed mean is on the *far* side of the body centre. Instead the error is
a per-episode multiplicative range error (`s_ep = 1.043 ± 0.110`) that leaves
image-plane direction and within-scene relative depth nearly perfect. PE-off
does show a much smaller radial term on `val_id` (−0.003 m vs +0.037 m;
spread 0.016 m vs 0.112 m), but not on the confounded config split.

Implication for the PE-fix round (IDEA-101/102): this probe does **not**
confirm the visible-surface-copy mechanism, so a norm pin should be justified
as a metric-range/feature-takeover fix, and its success criterion should include
the radial term (`rad` spread and `s_ep`), not only Task-A mean L2. Tangential
dominance — the BACKLOG's alternative route to 2D grounding/binding — is
explicitly ruled out for `s_f1` on `val_id`.