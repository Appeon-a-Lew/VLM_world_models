# v3 corpus generation (dense frames) — 2026-11-13, COMPLETED + published

Goal: regenerate all splits with the dense frame schedule (`frame_times` 0.0–1.0 @10 Hz,
`n_views: 1`) into a new root `data_v3`, per research/PLAN.md.

## Outcome
- Host **nc-9-4-1** (RTX 5090), launcher PID 1559430, `nice -n 10`, 16 workers, EGL.
- Space: `/var/tmp` NVMe was 100 % full; with human approval (2026-11-13) our own
  `/var/tmp/toyi-vid` ComfyUI scratch (33 G) was deleted; third-party data under the
  sticky-bit `/var/tmp` was left untouched.
- **All 9 splits complete, 0 generation errors**: train 10 000, val_id/ood_* 500 each,
  val_diag_* 300 each (index counts verified; `generation.json` errors empty).
- Corpus root: `/var/tmp/toyi/wm/data_v3` on nc-9-4-1 (~8 GB, 11 frames/episode).
- Gen venv: `/var/tmp/toyi/wm/venv` on nc-9-4-1 (mujoco 3.17.0). Delete when unused.

## Publication (Hugging Face, dataset `Appeon-a-Lew/physwm-tabletop-pushes`, public)
- `README.md` = research/DATACARD_hf.md (commit d90056a).
- `v3/` = full corpus upload launched from nc-9-4-1 (`~/hf_upload_v3.log`, resumable
  `hf upload ... data_v3 v3`); token via `~/.hf_tok` (chmod 600, delete after uploads).
- `site/v3/` = static dashboard bundle (360 sample episodes with all frames, 17.2k JSON
  API files, 530 MB) exported on the box with `scripts/dashboard_export.py`
  (`~/hf_upload_site.log`).
- GitHub Pages: `gh-pages` branch of VLM_world_models holds the viewer + `config.js`
  pointing at the HF `site/v3` bundle. **Human step pending:** repo Settings → Pages →
  deploy from branch `gh-pages` / root; then https://appeon-a-lew.github.io/VLM_world_models/
- Preds: none yet; B1–B5 prediction files go to `preds/` after the runs.

## Probe table from the deferred attempt (kept for reference)
| box | GPU | disk | users |
|---|---|---|---|
| nc-10-4-2 | 15.3 G / 97 % | 310 G | gekl |
| nc-9-4-1 | free | was full → 33 G after our cleanup | – |
| nc-9-3-1 | 11.0 G / 96 % | 45 G | lxiaol |
| nc-11-1-2 | 8.0 G / 100 % | 31 G | baile |

Launch recipe for future regenerations (any suitable box):
```bash
ssh <BOX> 'cd ~/adl4cv-worlmodels && nohup nice -n 10 env MUJOCO_GL=egl OUT=/var/tmp/toyi/wm/data_vN \
  WORKERS=16 PATH=/var/tmp/toyi/wm/venv/bin:$PATH bash scripts/gen_all.sh > ~/gen_vN.log 2>&1 &'
```
