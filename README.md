# VLMs as Physical World Models

Can a vision-language model predict **where objects end up** after a push, as grounded 3D
coordinates (not multiple-choice text), including in situations it never trained on?

This repository contains the simulated data-generation pipeline and the evaluation loop
(baselines, zero-shot and LoRA-SFT Qwen3-VL inference, image-dependence ablations, scoring,
counterfactual analysis).

## Task
- **Input:** one RGB image (t = 0), the camera intrinsics, the list of objects, and the action
  text, e.g. *"Push the red box to the left with an impulse of 0.85 N·s."*
- **Task A** (perception): the 3D centre of every object now.
- **Task B** (dynamics): the 3D centre of every object 1 s after the push.
- **Coordinates:** camera frame (OpenCV: x right, y down, z forward), metres.
- **Metrics:** 3D centre L2 error (m) for all, target, moved and unmoved objects, plus
  `skill = 1 − err/err_static`. Missing or non-finite answers count as 1 m; parse and missing
  rates are reported alongside.

## Data generation (`physwm/sim`)
- **Scenes:** MuJoCo tabletop with 1–3 primitives. Colour, shape/size, density and friction are
  sampled independently.
- **Camera:** one random camera per episode (azimuth, elevation, distance).
- **Push:** an instantaneous impulse at the CoM, in a camera-relative direction
  (left / right / away / toward) with a small hidden jitter.
- **Stored per episode:** RGB at t=0 and t=1, metric depth, segmentation, K and cam2world,
  masses, frictions, extents, poses at t ∈ {0, 1, 2} s, and a dense 50 Hz trajectory.
- Every episode is reproducible from (config, seed). Each split records its resolved config,
  generator source hashes and simulator versions; resuming requires matching provenance, so
  changed code or configs must be generated into a new output root.
- `physwm/sim/counterfactual.py` generates matched families from one settled state: original,
  reversed direction, half / double impulse, target recolour, and camera-elevation change.

| split | config | seeds | what differs |
|---|---|---|---|
| train | `sim_train.yaml` | 0–9999 | — |
| val_id / id_test | `sim_train.yaml` | 5 000 000+ / 1 000 000+ | nothing (ID) |
| val_ood_objects / ood_objects | `sim_ood_objects.yaml` | 6 000 000+ / 2 000 000+ | capsule / cone / hexagonal prism |
| val_ood_forces / ood_forces | `sim_ood_forces.yaml` | 7 000 000+ / 3 000 000+ | push speeds outside the train range |
| val_ood_config / ood_config | `sim_ood_config.yaml` | 8 000 000+ / 4 000 000+ | 4–5 objects clustered at a table corner, low camera |
| val_diag_{elev,count,corner,elev_corner} | `sim_diag_*.yaml` | 8 500 000+ | one or two `ood_config` factors at a time |

Use the `val_*` splits for all iteration; keep `id_test` / `ood_*` as the held-out test set.

**Interpretation notes.** Density, friction and the ±15° direction jitter are not given to the
model, so part of the Task B error is irreducible from the inputs. The forces split shifts
initial *speed* (impulse = mass × speed), so impulse ranges can overlap. The config split changes
several factors at once (the diagnostic splits isolate them). The static baseline uses true
initial positions, i.e. it has oracle perception.

## Evaluation loop (`physwm/eval`, `physwm/vlm`)
- `eval/baselines.py`: static and constant-velocity predictions.
- `vlm/infer.py`: batched Qwen3-VL inference (base or + LoRA), `center` or `qwen3d` answer
  style, image modes `normal / none / blank / swap` for the image-dependence test.
- `vlm/sft.py`: LoRA SFT on Task A + B (`configs/sft_qwen3vl2b.yaml`).
- `eval/evaluate.py`: results table (markdown / CSV) from any set of prediction files; checks
  complete, duplicate-free episode/task coverage.
- `eval/analyze_baseline.py`: paired episode-bootstrap comparisons against GT-static and
  own-perception static, stratified descriptive summaries.
- `eval/counterfactual.py`: endpoint and paired-effect errors on counterfactual families, in
  world coordinates, with family-level bootstrap.

## Running

```bash
uv venv -p 3.12 && source .venv/bin/activate && uv pip install -e '.[dev]'
export MUJOCO_GL=egl                      # headless rendering
python -m pytest -q tests                 # geometry, simulation, provenance, parser, metrics
bash scripts/gen_all.sh                   # all splits into $WM/data (default WM=work)
bash scripts/run_phase1.sh                # baselines, zero-shot, SFT, eval, ablations, table
bash scripts/run_phase1.sh baselines table   # or individual stages
python scripts/viz_episode.py --data work/data --split val_id --n 20 --out viz/ \
    --pred work/runs/sft_eval/normal.jsonl   # GT (o) vs prediction (x) overlays
```

Counterfactual families and their evaluation:

```bash
# split must be train or val (the seed assigns each family to one); seeds start at 9 901 000,
# use a separate output root from the main splits
python -m physwm.sim.counterfactual --config configs/sim_train.yaml --split val \
    --seed-start 9901000 --n 100 --out work/cf_data
python -m physwm.vlm.infer --data work/cf_data --splits val --tasks B --lora <adapter> --out work/runs/cf/pred.jsonl
python -m physwm.eval.counterfactual --data work/cf_data --splits val --pred sft=work/runs/cf/pred.jsonl \
    --static --n-boot 2000 --seed 0 --out work/runs/cf
```

SFT needs a CUDA GPU (developed on a 16 GB RTX 5080, bf16); generation and evaluation are CPU-only.

## Layout
- `physwm/schema.py`: episode schema and split loading
- `physwm/geometry.py`: camera and rotation maths
- `physwm/sim/`: scene sampling, push simulation, rendering, generation CLI, counterfactual families
- `physwm/eval/`: metrics, baselines, results table, statistical analyses
- `physwm/vlm/`: prompts and image ablations, answer parser, batched inference, LoRA SFT
- `configs/`: simulation splits and SFT hyper-parameters
