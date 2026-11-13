# v3 corpus generation (dense frames) — attempt 2026-11-13, DEFERRED

Goal: regenerate all splits with the dense frame schedule (`frame_times` 0.0–1.0 @10 Hz,
`n_views: 1`) into a new root `data_v3`, per research/PLAN.md. Code is synced to the NFS
home `~/adl4cv-worlmodels` (visible from every nc-* box); do not stage data on NFS.

## Attempt on nc-9-4-1 (RTX 5090, GPU free)
- Launch failed immediately: `/var/tmp` NVMe (790 G) is **100 % full** (750 G, not ours);
  `mkdir /var/tmp/toyi` -> ENOSPC. Leftover dir removed.
- The box boots from a live overlay (`/` = cow, 30 G): not a durable place for the corpus.
- Gen venv created there was removed (NFS quota); the synced repo copy on NFS was kept.

## Probe results 2026-11-13 (GPU mem/util, local disk free, users)
| box | GPU | disk | users |
|---|---|---|---|
| nc-10-4-2 | 15.3 G / 97 % | 310 G | gekl |
| nc-9-4-1 | free | **full** | – |
| nc-9-3-1 | 11.0 G / 96 % | 45 G | lxiaol |
| nc-11-1-2 | 8.0 G / 100 % | 31 G | baile |
| nc-9-1-2, nc-9-2-3 | busy | – | lopen/penel, dieh/… |
| nc-9-1-1, nc-9-2-2, nc-9-3-3, nc-9-4-2, nc-9-5-1, nc-10-1-1, nc-10-4-3, nc-10-3-1, nc-10-4-1, nc-10-4-4 | down | | |

No box currently offers a free GPU **and** ≥60 G free local disk. MuJoCo has no CPU render
path here (no OSMesa libs on the boxes; EGL needs a free GPU), so generation waits for a
box to free up. Re-probe at the start of every session (small batches, skill tum-gpu).

## Launch recipe once a suitable box <BOX> appears
```bash
# on <BOX>, machine-local venv with mujoco==3.17 (nc-10-4-2 already has /var/tmp/toyi/wm/sim-venv)
ssh <BOX> 'cd ~/adl4cv-worlmodels && nohup env MUJOCO_GL=egl OUT=/var/tmp/toyi/wm/data_v3 \
  WORKERS=16 PATH=$HOME/<venv>/bin:$PATH bash scripts/gen_all.sh > ~/gen_v3.log 2>&1 &'
```
Expected size ≈ 7–8 GB (11 frames/episode, ~0.55 MB/ep, 13.2k episodes). Verify with
`scripts/gen_all.sh` completion + per-split `index.jsonl` counts and `generation.json`
errors empty; record host, PID, timings and provenance hash here afterwards.
