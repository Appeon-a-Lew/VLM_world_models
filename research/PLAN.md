# Phase 2 plan — grounding-first world model with OneCanvas-style unprojection

_Recorded 2026-11-13 with the human. Supersedes the Phase-1 loop direction
(`LOOP.md`, exp000–exp010). Legacy results stay valid only as dataset-v1
numbers; lost progress on that track is accepted._

## Decisions (human, 2026-11-13)

1. **Grounding before dynamics.** First capability: 3D bounding-box grounding
   from 2D frames. Dynamics (t = +1 s) is predicted on top of it, not instead of it.
2. **Backbone:** OneCanvas-style unprojection — per-patch metric xyz position
   embeddings added to the frozen ViT features of Qwen3-VL, LoRA on the LLM.
   Single view: the canvas degenerates to camera-frame xyz per patch
   (see `research/literature/2026_onecanvas.md`, "Relevance to us").
3. **Loss:** VLM default next-token cross-entropy on text answers. No auxiliary
   losses, no fancy objectives for the baseline.
4. **Answer format (start):** centre + extents in metres, camera frame.
   Token precision is an open question (OneCanvas used 1 dp for distances;
   cm-level boxes need finer).
5. **Depth:** ground-truth metric depth (already stored per episode) first.
   The impact of swapping in a metric depth estimator is a later experiment.
6. **Frames are an experimental axis:** baseline measurement of **1 frame vs
   1–3 frames** input. The prediction goal stays **t = +1 s**.
   **Sub-goal: next-frame accuracy** (state at the next stored frame), which
   requires **all frames of the action in the dataset** (dense RGB, not just t0/t1).
7. **Views are an experimental axis:** start with **1 view** (fine: the
   single-view canvas is a no-op remap, the PE carries the effect); vary the
   number of views later.
8. **OOD axes:** different **objects**, different **forces**, different
   **starting configurations**. The camera stays **in-distribution** — geometry
   is supplied by unprojection, so the model should not have to "learn" the
   camera and camera shift is not an OOD test anymore.
9. Legacy camera-shift splits (`val_ood_config` with elevation 12–25°,
   `val_diag_elev*`) are retired as OOD tests; keep them only as descriptive
   diagnostics if useful.

## Task definition

- **Input:** V views × N frames of RGB (N ∈ {1, 2, 3} for the baseline matrix),
  per-frame metric depth (phase A: GT), per-view K and cam2world, the object
  list with referring expressions, and the action text.
- **Primary output:** per-object 3D box (centre + extents) at **t = +1 s**.
- **Sub-goal output:** per-object 3D box at the **next stored frame**
  (t = +0.1 s under the proposed schedule) — the next-frame accuracy track.
- **Grounding sanity output:** per-object 3D box at t = 0 (perception only).
- **Metrics:** centre L2 (m), extents error, 3D IoU, next-frame centre L2,
  parse/missing rate; reported per split and vs `static`/oracle references.
  Frame-count and view-count comparisons use paired bootstrap CIs on val.

## Data generation changes

The corrected generator (audit of 2026-10-27) stays the base. Changes:

1. **Frame schedule.** Replace the hardcoded 2-frame rendering with a
   configurable schedule. Proposal: RGB + depth + seg at
   `t = 0.0, 0.1, …, 1.0 s` (11 frames, 10 Hz); `traj.npz` keeps 50 Hz poses to
   t = 2 s. 20 Hz is affordable (~14 GB); 50 Hz RGB is not (~53 GB) and adds
   nothing over `traj.npz`.
2. **Views.** Config `n_views` (start 1): sample V cameras per episode with
   known poses and render every stored time from every view. Design the
   episode schema for V ≥ 1 now so later view experiments need no migration.
3. **Storage budget (13.2k episodes):** ~0.55 MB/ep at 11 frames ≈ 7–8 GB
   (RGB 39 KB + depth 10 KB + seg ~1 KB per frame), vs 1.5 GB today.
   Render cost scales linearly with frames × views; one-off regeneration.
4. **Splits (new OOD axes, camera in-distribution everywhere):**
   - `ood_objects`: unseen shapes/sizes (as today).
   - `ood_forces`: push speeds outside the train range (as today).
   - `ood_config`: object count + placement only — **camera sampled from the
     train range** (this is the change vs Phase 1).
   - `id_test` as today. Diagnostic camera splits retired per decision 9.
5. **Provenance:** new output root; the existing provenance/index/coverage
   checks already prevent silent mixing of generator versions.

## Model and baseline matrix (no fancy stuff)

- Qwen3-VL-2B; ViT + merger frozen; LoRA on LLM attention/MLP (OneCanvas setup).
- Unprojection per merged vision token: median GT depth over the token's pixel
  block with K rescaled to the feature grid → camera-frame xyz; OneCanvas
  136-channel sinusoidal PE → 2-layer MLP → addition through a zero-initialised
  scalar gate (PE lr 50× base). Multi-frame: frame index available to the PE;
  no panoramic M-RoPE remap in the single-view baseline.
- Supervision: default CE on the answer text (centre + extents), for each
  requested time (t = 0 / next frame / t = +1 s) as separate queries.

Baseline measurements, in order:

| id | input | predicts | purpose |
|---|---|---|---|
| B1 | 1 frame (t=0), 1 view, GT depth | t=0 boxes | grounding sanity / perception floor |
| B2 | 1 frame (t=0), 1 view, GT depth | t=+1 s boxes | single-frame reference |
| B3 | 1–3 frames (t=0, 0.1, 0.2), 1 view, GT depth | t=+1 s boxes | **frame-count axis** vs B2 |
| B4 | 1 frame (t=0), 1 view, GT depth | next-frame (t=0.1) boxes | **sub-goal track** |
| B5 | B2/B3 without the xyz PE | same | isolates the unprojection contribution |

Later axes (after B1–B5): views ∈ {2, 4}; estimated metric depth instead of GT;
frame schedules; answer-format precision.

## Sequencing

1. Commit the pending audit/counterfactual working tree in reviewable chunks;
   this plan file goes in with the docs chunk.
2. Generator work: frame schedule, per-frame depth/seg, `n_views`, schema and
   tests, provenance v3. (Local, CPU-only except render smoke.)
3. Regenerate the corpus on a lab box (nc-10-4-2 currently holds the corrected
   v2 corpus and is reachable; nc-10-4-3 with legacy data + adapters is down).
4. Model surgery: PE/MLP/gate into the Qwen3-VL visual path; SFT and inference
   scripts accept V × N frames; parser/scorer for centre+extents boxes and the
   new metrics (IoU, next-frame).
5. Run B1–B5; new leaderboard keyed to dataset v3; per-axis CIs.
6. Then the view-count and depth-source experiments.

## Open questions

- Answer precision: how many decimals / which tokenisation for centre+extents?
- Ground all objects or only the action target (or both, as separate queries)?
- Multi-view output frame: camera frame of view 0 vs world frame?
- Frame schedule: 10 Hz vs 20 Hz; keep a t = 2 s tail frame as optional input?
- Segmentation per frame or t = 0 only (cheap either way)?
- Which metric depth estimator for the post-GT-depth experiment
  (Depth Anything V2 metric vs Depth Anything 3)?
- Does the next-frame sub-goal help t = +1 s as an auxiliary query, or is it
  purely a diagnostic? (Baseline: purely diagnostic.)
