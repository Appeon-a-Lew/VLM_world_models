---
license: cc-by-4.0
task_categories:
  - robotics
  - question-answering
language: en
tags:
  - simulation
  - mujoco
  - world-models
  - 3d-grounding
  - tabletop
pretty_name: physwm tabletop pushes (VLMs as physical world models)
size_categories:
  - 10K<n<100K
configs:
  - config_name: meta
    default: true
    data_files:
      - split: train
        path: tables/train.parquet
      - split: id_test
        path: tables/id_test.parquet
      - split: ood_objects
        path: tables/ood_objects.parquet
      - split: ood_forces
        path: tables/ood_forces.parquet
      - split: ood_config
        path: tables/ood_config.parquet
      - split: val_id
        path: tables/val_id.parquet
      - split: val_ood_objects
        path: tables/val_ood_objects.parquet
      - split: val_ood_forces
        path: tables/val_ood_forces.parquet
      - split: val_ood_config
        path: tables/val_ood_config.parquet
      - split: val_diag_elev
        path: tables/val_diag_elev.parquet
      - split: val_diag_count
        path: tables/val_diag_count.parquet
      - split: val_diag_corner
        path: tables/val_diag_corner.parquet
      - split: val_diag_elev_corner
        path: tables/val_diag_elev_corner.parquet
---

# physwm tabletop pushes

Synthetic tabletop push episodes for studying VLMs as physical world models:
**3D bounding-box grounding and 1 s-ahead endpoint prediction from 2D frames**
with metric depth, intrinsics and dense trajectories.

- Code, plans and experiment records: https://github.com/Appeon-a-Lew/VLM_world_models
- Viewer (GitHub Pages, streams files from this repo): see the code README once published

## Task
- **Input:** RGB frame(s) at t = 0 (optionally t = 0.1, 0.2, …), camera intrinsics K and
  cam2world, the object list with referring expressions, and an action text such as
  *"Push the red box to the left with an impulse of 0.85 N·s."*
- **Task A (perception):** 3D centre of every object at t = 0.
- **Task B (dynamics):** 3D centre of every object 1 s after the push.
- **Task N (next frame):** 3D centre of every object at the next stored frame (t = 0.1 s).
- **Coordinates:** camera frame of view 0 (OpenCV: x right, y down, z forward), metres.
  Object states in `meta.json` are world frame (z up, table top z = 0); quaternions wxyz.

## Scenes
Tabletop with 1–3 primitives (train); colour, shape/size, density and friction sampled
independently; one random camera per episode (view 0; further views in v3+); an
instantaneous impulse at the target's centre of mass in a camera-relative direction
(left / right / away / toward) with unreported ±15° direction jitter. Density, friction
and the jitter are **hidden variables**: endpoint prediction is conditional on them.

## Splits and seeds
| split | config | seeds | what differs |
|---|---|---|---|
| train | sim_train.yaml | 0–9999 | — |
| id_test / val_id | sim_train.yaml | 1 000 000+ / 5 000 000+ | nothing (ID) |
| ood_objects / val_ood_objects | sim_ood_objects.yaml | 2 000 000+ / 6 000 000+ | capsule / cone / hexagonal prism |
| ood_forces / val_ood_forces | sim_ood_forces.yaml | 3 000 000+ / 7 000 000+ | push speeds outside the train range |
| ood_config / val_ood_config | sim_ood_config.yaml | 4 000 000+ / 8 000 000+ | 4–5 objects clustered at a table corner |
| val_diag_{elev,count,corner,elev_corner} | sim_diag_*.yaml | 8 800 000+ | single-factor diagnostics (legacy OOD axes) |

Counts: train 10 000; legacy phase-1 splits id_test / ood_objects / ood_forces /
ood_config 1 000 each; their validation counterparts val_id / val_ood_* 500 each;
val_diag_* 300 each (17 400 episodes, ~11 GB in v3). Every episode is
reproducible from (config, seed); `provenance.json` per split pins the resolved config,
generator source hashes and MuJoCo/NumPy versions.
The dataset viewer shows the `meta` config: per-episode parquet tables under `tables/`
(seed, object count/shapes, action, camera, target displacement); raw episode files
live under `v3/` and stream over HTTPS.

## Files per episode (`<split>/<episode_id>/`)
- `meta.json` — objects (shape, colour, half extents, mass, density, friction), camera(s)
  (K, cam2world, fovy, azimuth/elevation/distance), action (target, direction, impulse,
  speed, text), stored states, `files` map; v3+ adds `views` and `frame_times`.
- v2 naming: `rgb_t0.png`, `rgb_t1.png` (448×448), `seg_t0.png` (object-id mask),
  `depth_t0.npz` (metric depth, float16, key `depth`), `traj.npz` (50 Hz pos/quat, 2 s).
- v3 naming (dense frames): `rgb_v{v}f{kk}.png`, `depth_v{v}f{kk}.npz`, `seg_v{v}f{kk}.png`
  for every view v and frame kk in `frame_times` (0.0–1.0 s at 10 Hz), plus `traj.npz`.
- `index.jsonl`, `provenance.json`, `generation.json` per split.

## Versions
- `v2/` — corrected corpus (2026-10-27 audit fixes: observation timing, settled aiming,
  provenance-gated resume, indexed loading), 2 rendered frames.
- `v3/` — same physics plus the dense frame schedule and multi-view support (2026-11-13).
- `site/` — static JSON API + sample files for the web viewer.
- `preds/` — prediction jsonl files from published baselines (records: episode_id, task,
  objects {id: [x,y,z] camera frame or null}, extents for box3d style, raw answer).

## Known caveats
- Pushed objects mostly stop within ~0.2 s (friction); balls roll and ~1/3 of pushed
  balls leave the table (large errors; look at medians as well as means).
- Visibility is not guaranteed: partially occluded or out-of-frame objects occur;
  stratify by the segmentation mask before attributing errors to dynamics.
- `on_table` in meta.json is a positional heuristic, not a contact predicate.
- Depth background is ~87 m (sky); valid surface depth is < ~3 m.

## Citation
Please cite the code repository and, once available, the companion paper.
Dataset generated with MuJoCo 3.17; generator and evaluation code in the GitHub repo.
