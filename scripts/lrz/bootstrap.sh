#!/usr/bin/env bash
# LRZ bootstrap: run ON the LRZ login node (or via ssh once a key is registered).
# Creates the project dir on scratch, clones the repo, builds a venv, and
# prefetches the Qwen3-VL-2B weights so compute nodes can run offline.
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
mkdir -p "$WM"
cd "$WM"
[ -d repo ] || git clone https://github.com/Appeon-a-Lew/VLM_world_models.git repo
cd repo && git pull --ff-only || true
if [ ! -x "$WM/venv/bin/python" ]; then
  python3 -m venv "$WM/venv"
  "$WM/venv/bin/pip" install -q torch torchvision peft==0.21.2 accelerate qwen-vl-utils \
    pillow pyyaml tqdm numpy scipy pyarrow
  # pin transformers after torch so the resolver keeps the cu12 wheels
  "$WM/venv/bin/pip" install -q transformers==5.20.2
fi
"$WM/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# prefetch weights on the login node (compute nodes are offline)
export HF_HOME=$WM/hf
[ -d "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct" ] || \
  "$WM/venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-2B-Instruct")
print("weights cached")
PY
echo "bootstrap done: $WM"
