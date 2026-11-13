#!/usr/bin/env bash
# Slurm job: run ONE arm of the B-matrix on LRZ (H100/A100), offline weights.
# Submit via scripts/lrz/submit_arm.sh <arm> [partition]; ARM arrives via --export.
#SBATCH --job-name=wm_arm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
cd "$WM/repo"
export HF_HOME=$WM/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DATA=$WM/data_v3 RUNS=$WM/runs_b
export PATH=$WM/venv/bin:$PATH
mkdir -p "$RUNS" logs
echo "host $(hostname) arm ${ARM:?} gpu:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash scripts/run_b_matrix.sh "$ARM"
echo "arm $ARM finished on $(hostname)"
