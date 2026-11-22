# VLMs as Physical World Models

Can a VLM predict **where objects end up** after a push, as grounded 3D coordinates (camera frame, metres), starting from a minimal deterministic setup and scaling up step by step?

## The experiments in plain language

We show a model a picture of objects on a table and tell it which object we will push, in what direction, and how hard. The model predicts where each object's 3D box will be one second later. We use a simulator so we know where the objects actually end up.

We train four versions on the same set of push scenes:

| Version | What it learns |
| --- | --- |
| Forward-only (`fwd`) | Predict the starting and ending 3D boxes. |
| Inverse dynamics (`idp`) | Also watch a before-and-after pair and infer which object was pushed, in what direction, and roughly how hard. |
| Physical context (`ctx`) | Also describe what supports the objects, where the table edge is, and what's in the push's path before predicting the ending boxes. |
| Both (`both`) | Learn the inverse-dynamics task and describe the physical context. |

To test whether the model uses the action, we keep the starting scene fixed and reverse, halve, or double the push. We measure how closely the change in its prediction matches the change in the simulator. We also check what happens when we change only the object's color or the camera angle.

If an improvement is large enough, we repeat the key comparison with a different training seed. If the gain holds up, the next experiment trains on pushes, drops, and stacks, then tests on tilts the models have never trained on. The decision rules are in [the preregistration](research/PREREG.md).

## Task
- **Input:** RGB image(s), camera intrinsics, object list, and action text (e.g. *"Push the red box to the left with an impulse of 0.85 N·s."*).
- **Task A (perception):** 3D centre (and optionally extents) of every object at t = 0.
- **Task B (dynamics):** 3D centre (and optionally extents) of every object at t = 1 s after the push.
- **Task N (next frame):** 3D centre at the first stored frame after the last input frame.
- **Coordinates:** OpenCV camera frame (x right, y down, z forward), metres.
- **Metrics:** 3D centre L2 error (m) and `skill = 1 − err / err_static`. Missing/unparseable answers count as 1 m.

## Start Simple (Recommended Workflow)
1. **Generate a small deterministic dataset** (`configs/sim_simple.yaml`: 1 box, fixed density & friction, zero direction jitter):
   ```bash
   python -m physwm.sim.generate --config configs/sim_simple.yaml --split train --seed-start 0 --n 1000 --out $WM/data_simple
   python -m physwm.sim.generate --config configs/sim_simple.yaml --split val_id --seed-start 1000000 --n 200 --out $WM/data_simple
   ```
2. **Train SFT** (with or without `--no-pe` for the depth-based xyz position embedding):
   ```bash
   python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml --data $WM/data_simple --style box3d --out $WM/runs/simple
   ```
3. **Run inference & evaluate** (`infer.py` automatically loads `pe.pt` from `--lora` or its parent directory):
   ```bash
   python -m physwm.vlm.infer --data $WM/data_simple --splits val_id --tasks A B \
     --lora $WM/runs/simple/adapter --style box3d --out $WM/runs/simple/pred_val.jsonl
   python -m physwm.eval.evaluate --data $WM/data_simple --splits val_id \
     --pred simple=$WM/runs/simple/pred_val.jsonl
   ```
4. Once deterministic physics works cleanly, scale up to `configs/sim_train.yaml` (1–3 objects, hidden density/friction, direction jitter) and the OOD splits (`sim_ood_objects.yaml`, `sim_ood_forces.yaml`, `sim_ood_config_nocam.yaml`, `sim_ood_config.yaml`).

## Layout
- `physwm/schema.py`: episode schema
- `physwm/geometry.py`: camera and rotation maths
- `physwm/sim/`: MuJoCo scene sampling, push simulation, rendering, generation CLI
- `physwm/vlm/`: prompts, answer parser, xyz position embedding (`pe.py`), LoRA SFT (`sft.py`), batched inference (`infer.py`)
- `physwm/eval/`: metrics, 3D box metrics, baselines, evaluation & motion-attribution scripts
- `configs/`: simulation splits and SFT hyper-parameters
- `scripts/`: environment, generation, training, and visualization helpers
- `tests/`: unit tests (`pytest -q tests`)

## M0 screen (Tier 1, `research/PREREG.md`)
Four arms on the same push-only data — `fwd` (A,B box3d), `idp` (+ inverse dynamics on the
fixed t = 0.1 s pair), `ctx` (emit the physical context before the future boxes, `ctx3d`),
`both` — trained and inferred **without the xyz PE** (the GT-depth PE is a separate
diagnostic: `DEPTH_DIAG=1`), with the shared optimizer-step target pinned from `--plan`
before any arm trains:
```bash
source scripts/env.sh && CF_SPLIT=<matched-family split> bash scripts/run_m0.sh calibrate
source scripts/env.sh && CF_SPLIT=<...> bash scripts/run_m0.sh                     # seed 1
# then, if paired_arms.py reports 'earned' on the screen contrast:
source scripts/env.sh && CF_SPLIT=<...> SEED=2 bash scripts/run_m0.sh fwd both
python -m physwm.eval.paired_arms --data $WM/data --splits <...> \
  --a fwd=$WM/runs_m0/fwd_s1/pred_val.jsonl --a fwd=$WM/runs_m0/fwd_s2/pred_val.jsonl \
  --b both=$WM/runs_m0/both_s1/pred_val.jsonl --b both=$WM/runs_m0/both_s2/pred_val.jsonl \
  --out $WM/runs_m0/paired/m0_gate.json
```
The gate is pre-registered: reduction ≥ max(0.10 × E_static, 0.02 m) with bootstrap CI95
excluding 0, per seed, **two distinct verified seed pairs, both seeds individually** — one
seed can only earn the second. `--oracle-context` evaluates any checkpoint with
simulator-provided support + action context (the PCB diagnostic; predicted support is
scored separately). On LRZ, `scripts/lrz/m0_smoke_job.sh` runs the end-to-end smoke
before any screen submission.

The [M0 results dashboard](https://appeon-a-lew.github.io/VLM_world_models/m0/)
shows the completed screen, training curves, and inference prompt diagnostic.
To rebuild the figures and TensorBoard bundle from the saved seed-1 logs and
paired scores, first compute `paired/effect_skill.json` from the matching validation
metadata and Task B predictions with `python -m physwm.eval.effect_skill --help`
as a reference. Then run:

```bash
python scripts/export_m0_visuals.py runs/m0-visuals/source runs/m0-visuals
tensorboard --logdir runs/m0-visuals/tensorboard
```

The exporter uses `matplotlib` and `tensorboard`. TensorBoard includes loss,
learning rate, elapsed time, token and example counts, and a single final
evaluation point per run. The run logs contain no intermediate validation loss.
