#!/usr/bin/env bash
# Generate val_ood_config_nocam (camera-in-distribution config axis) on LRZ.
# EGL rendering on one H100; seeds 9,200,000+ are outside the reserved 1M-9M range.
#SBATCH --job-name=wm_gen_nocam
#SBATCH --gres=gpu:1
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
cd "$WM/repo" && git pull -q
"$WM/venv/bin/python" -c "import mujoco" 2>/dev/null || "$WM/venv/bin/pip" install -q mujoco==3.17.0
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PATH=$WM/venv/bin:$PATH
python -m physwm.sim.generate --config configs/sim_ood_config_nocam.yaml \
  --split val_ood_config_nocam --seed-start 9200000 --n 500 \
  --out "$WM/data_v3" --workers 16
python - "$WM/data_v3/val_ood_config_nocam" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
g = json.loads((d / "generation.json").read_text())
n = len((d / "index.jsonl").read_text().splitlines())
print(f"nocam split: {n} episodes, {len(g.get('errors', []))} errors")
PY
