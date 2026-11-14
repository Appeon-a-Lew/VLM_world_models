# Research state — phase 2

_Maintained by the orchestrator; the human may edit it at any time._

Phase-1 record (exp000–exp010, dataset v1): `research/STATE_phase1_archive.md`, frozen
2026-11-14. Those numbers stay valid only as dataset-v1 numbers (`research/PLAN.md`).
Phase-1 ideas: `research/ideas/BACKLOG_phase1_archive.md`.

## Leaderboard

`research/RESULTS_phase2.md` is the **only** leaderboard: §1 grounding (Task A), §2 endpoint
prediction (Task B), §3 motion attribution, §4 frame/forecast axes, §5 hygiene and open items,
§6 next experiments. Do not duplicate its numbers here; point at the section instead
(e.g. "A val_id 0.0996 → 0.0127, RESULTS_phase2 §1").

## What phase 2 has established (pointers, no new numbers)

- box3d **without** PE is both the best grounding and the best dynamics arm we have; PE-on
  costs ~8.7 cm on Task A val_id, ~20× the seed shift measured for s_f1 seed 1 (§1, §2).
- With PE off, the privileged transfer read-out is ~0, i.e. the residual B error is
  displacement, not anchoring: **dynamics, not grounding, is now the bottleneck** (§3).
- The legacy compound `val_ood_config` (camera 12–25°) is descriptive only. The phase-2
  configuration axis is `val_ood_config_nocam` (camera in-distribution); it is not generated
  yet (§5, PLAN reading rule 6).
- Corpus v3 is published; the Pages site streams v3 renders and the s_f1 / s_f1_nope / l_f1
  predictions (§5).

## Queue (RESULTS_phase2 §6 order — do not reorder without a planner justification)

1. **PE-fix arm(s):** norm-pinned PE (OneCanvas stage-1 recipe) and/or PE lr 5×; same data,
   recipe and corpus (IDEA-101, IDEA-102).
2. **`val_ood_config_nocam`:** generation, then inference-only rescoring of every existing
   adapter (IDEA-104).
3. **Seed floor:** complete seed-1 pairs for l_f1 and s_f1_nope, seed 2 for the 2×2 winners
   (IDEA-105).
4. **Only then:** view-count axis (n_views 2, IDEA-107) and estimated metric depth (IDEA-108).

GPU-free and always allowed in parallel: the PE error-geometry probe on existing predictions
(IDEA-103), and the forecast-horizon read-out once `s_f3` lands (IDEA-106).

## Open questions

- **PE mechanism:** is the PE read as visible-surface xyz (errors radial, along the view ray,
  at surface-to-centre scale) or does it disturb body-centre inference? IDEA-103 separates the
  radial and tangential components; the answer decides whether norm-pinning (101) should
  recover grounding.
- **Config attribution:** does the phase-2 config axis (object count + placement, camera
  in-distribution) reproduce any grounding gap, or was the old config failure entirely the
  camera? Blocked on `val_ood_config_nocam` (IDEA-104).
- **Seed floor:** only s_f1 has a repeat. Are l_f1 ↔ s_f1_nope differences, and any PE-fix
  gain, above the training-seed floor? (IDEA-105)
- **Forecast-horizon value:** does N stay better than own-A-static at 0.3 s (first unobserved
  frame), i.e. is the 3-frame input learning motion rather than reconstructing the input?
  (IDEA-106)

## Infra notes

- **Lab box churns.** Hosts are probed in policy order (`nc-9-4-1`, `nc-10-4-2`,
  `nc-10-4-3`); one job at a time, and power-offs / another user's load are normal. Long
  training goes to LRZ.
- **LRZ** partition `lrz-hgx-h100-94x4` is often at `Priority`, 6 concurrent jobs, 8 h wall,
  `--gres=gpu:1`, `nice -n 10`. Its value is parallel arms, not per-step speed.
- **Publication surface:** HF dataset `Appeon-a-Lew/physwm-tabletop-pushes` (`v3/`) plus the
  Pages site. Refresh the dashboard bundle after every finished arm, validation preds only.
- **Final test:** the manifest in the repo is legacy (v1/v2). A v3 final-test manifest, and
  any final-test or publish run, is a separate **human-approved decision**.