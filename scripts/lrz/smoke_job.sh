#!/usr/bin/env bash
# LRZ smoke + speed benchmark: ~230 SFT steps on 1 GPU, tiny subset.
# Validates the offline stack and prints s/step for comparison with the lab 5090.
#SBATCH --job-name=wm_smoke
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
cd "$WM/repo"
export HF_HOME=$WM/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=$WM/venv/bin:$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml --data "$WM/data_v3" \
  --style box3d --frames 0 --views 0 --tasks A,B,N --limit 1024 --epochs 0.3 \
  --out "$WM/runs_b/smoke"
python - <<'PY'
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path(f"{__import__('os').environ['WM']}/runs_b/smoke/train_log.jsonl").read_text().splitlines()]
warm = [r for r in rows if r["step"] >= 20]
dt = (warm[-1]["elapsed_s"] - warm[0]["elapsed_s"]) / max(1, warm[-1]["step"] - warm[0]["step"])
print(f"SMOKE s/step={dt:.3f} over {len(warm)} warm steps; final loss {warm[-1]['loss']:.4f}")
PY
