# VLMs as Physical World Models — Phase 1

Can a VLM predict **where objects end up** after a push, as grounded 3D coordinates and not as multiple-choice text, including in situations it never trained on?

Phase 1 is a small, rigorously evaluated baseline. It compares trivial baselines, zero-shot Qwen3-VL-2B and plain LoRA SFT, both in-distribution and OOD. It also includes an image-dependence test, which checks whether the model reads the scene at all. Causal / counterfactual / physics-informed ideas (Miro A–G) are deferred until this table exists.

## Task
- **Input:** one RGB image (t = 0), the camera intrinsics, the list of objects, and the action text, e.g. *"Push the red box to the left with an impulse of 0.85 N·s."*
- **Task A** (perception): the 3D centre of every object now.
- **Task B** (dynamics): the 3D centre of every object 1 s after the push.
- **Coordinates:** camera frame (OpenCV: x right, y down, z forward), metres.
- **Metrics:** 3D centre L2 error (m), reported for all, target, moved and unmoved objects, plus `skill = 1 − err/err_static`. Missing answers count as 1 m.

## Data (MuJoCo, `physwm/sim`)
- **Scenes:** tabletop with 1–3 primitives. Colour, shape/size, density and friction are sampled independently.
- **Camera:** one random camera per episode.
- **Push:** an instantaneous impulse at the CoM, in a camera-relative direction (left / right / away / toward).
- **Stored per episode:** RGB at t=0 and t=1, metric depth, segmentation, K and cam2world, masses, frictions, extents, poses at t ∈ {0, 1, 2} s, and a dense 50 Hz trajectory.
- Every episode is reproducible from (config, seed).

| split | config | seeds | what differs |
|---|---|---|---|
| train | `sim_train.yaml` | 0–9999 | — |
| id_test | `sim_train.yaml` | 1 000 000+ | nothing (ID) |
| ood_objects | `sim_ood_objects.yaml` | 2 000 000+ | capsule / cone / hexagonal prism |
| ood_forces | `sim_ood_forces.yaml` | 3 000 000+ | push speeds outside the train range |
| ood_config | `sim_ood_config.yaml` | 4 000 000+ | 4–5 objects clustered at a table corner, low camera |

## Running (TUM lab machines)
The code sits in the NFS home (`~/adl4cv-worlmodels`, synced via `scripts/sync.sh`). The venv, data, HF cache and runs live on the machine's local disk under `/var/tmp/toyi/wm`. Currently that is **nc-11-1-2**.

```bash
scripts/sync.sh                                   # laptop -> halle
ssh nc-11-1-2
source ~/adl4cv-worlmodels/scripts/env.sh         # venv, HF_HOME, MUJOCO_GL=egl, cd repo
python -m pytest -q tests                         # geometry, sim sanity, parser
bash scripts/gen_all.sh                           # ~15 min, ~1.6 GB
bash scripts/run_phase1.sh                        # baselines, zero-shot, SFT, eval, ablations, table
python scripts/viz_episode.py --data $WM/data --split id_test --n 20 --out $WM/viz/id \
    --pred $WM/runs/sft_eval/normal.jsonl         # GT (o) vs prediction (x) overlays
```

## Layout
- `physwm/schema.py`: episode schema
- `physwm/geometry.py`: camera and rotation maths
- `physwm/sim/`: scene sampling, push simulation, rendering, generation CLI
- `physwm/eval/`: metrics, baselines (static, constant velocity), table
- `physwm/vlm/`: prompts and image ablations, answer parser, batched inference, LoRA SFT
- `configs/`: simulation splits and SFT hyper-parameters
