# Matrix arms on nc-9-3-1 — 2026-11-15

Reason: LRZ `lrz-hgx-h100-94x4` sat at "Priority" (saturated) for ~2 days with six of our
jobs pending; the previous lab box nc-9-4-1 is powered off. Human decision 2026-11-15:
use another lab box when LRZ does not deliver.

## Placement after this decision
| work | backend | jobs |
|---|---|---|
| `s_f3`, `s_f3_nope`, `c_f1` (matrix completion) | **nc-9-3-1** (RTX 5090, idle, no users, 45 G free) | `~/box2_pipeline.sh` (venv -> regen -> runner), log `~/box2_pipeline.log` |
| seed-1 pairs `s_f1`, `s_f1_nope`, `l_f1` | LRZ (kept) | 5924526, 5924527 pending (5924525 done) |
| `val_ood_config_nocam` generation | LRZ (kept) | 5925914 pending |
| cancelled on LRZ to avoid duplication | — | 5924789, 5924790, 5924791 (the three matrix arms) |

## Data on nc-9-3-1
The v3 splits needed by the runner (train 10k, val_id/val_ood_* 500 each) are
**regenerated locally** with the same configs and seeds (`/var/tmp/toyi/wm/data_v3`),
not copied: physics/meta are seed-deterministic; rendered RGB can differ by +-1/255 in
<=8 pixels per image (EGL render noise, same caveat as the 2026-10-24 v2 regeneration).
Cross-box comparisons (s_f3 here vs s_f1 on nc-9-4-1) therefore carry that pixel-level
difference; the exp005 reproduction showed results stable at <=0.002 m under it.

## Environment
Machine-local venv `/var/tmp/toyi/wm/venv` (torch, transformers 5.20.2, peft 0.21.2, ...),
`HF_HOME=/var/tmp/toyi/wm/hf`, `MUJOCO_GL=egl`, everything under `nice -n 10`.
Repo copy on the shared NFS home `~/adl4cv-worlmodels` (resynced 2026-11-15).

## Outputs
`/var/tmp/toyi/wm/runs_b/<arm>/{adapter,pe.pt,pred_val.jsonl,metrics.csv}` and
`runs_b/matrix.md` when the runner finishes; collect with
`ssh nc-9-3-1 'cut -d, -f1,2,3,6,12,23,24,25 /var/tmp/toyi/wm/runs_b/*/metrics.csv'`.
