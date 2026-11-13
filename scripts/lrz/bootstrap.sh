#!/usr/bin/env bash
# LRZ bootstrap: run ON the LRZ login node (or via ssh once a key is registered).
# Creates the project dir on scratch, clones the repo, builds a venv (system venv
# module or uv fallback), and prefetches the Qwen3-VL-2B weights so compute nodes
# can run offline. Idempotent: safe to re-run.
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
mkdir -p "$WM"
cd "$WM"
if [ ! -d repo/.git ]; then
  git clone https://github.com/Appeon-a-Lew/VLM_world_models.git repo || { echo "NEED_REPO_RELAY"; exit 3; }
fi
( cd repo && git pull --ff-only || true )
export PATH=$HOME/.local/bin:$PATH
if ! "$WM/venv/bin/python" -c "import torch, transformers, peft" 2>/dev/null; then
  rm -rf "$WM/venv"
  python3 -m venv "$WM/venv" 2>/dev/null || true
  if [ ! -x "$WM/venv/bin/pip" ]; then
    rm -rf "$WM/venv"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$HOME/.local/bin:$PATH
    uv venv --python 3.12 "$WM/venv"
    PIP="uv pip install --python $WM/venv/bin/python -q"
  else
    PIP="$WM/venv/bin/pip install -q"
  fi
  $PIP torch torchvision peft==0.21.2 accelerate qwen-vl-utils \
    pillow pyyaml tqdm numpy scipy pyarrow transformers==5.20.2
fi
"$WM/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
export HF_HOME=$WM/hf
if [ ! -d "$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct" ]; then
  "$WM/venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct'); print('weights cached')" \
    || { echo "NEED_WEIGHTS_RELAY"; exit 4; }
fi
echo "bootstrap done: $WM"
