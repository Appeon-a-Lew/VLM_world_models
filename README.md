# VLMs as Physical World Models

Can a VLM predict **where objects end up** after a push, as grounded 3D coordinates (camera frame, metres), starting from a minimal deterministic setup and scaling up step by step?

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
