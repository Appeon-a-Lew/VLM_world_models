#!/usr/bin/env bash
# Run one M0 arm after the smoke and matched-family calibration.
# Export M0_REPO, M0_DATA, CF_SPLIT, M0_STEPS, RUNS, and SEED before submitting.
#SBATCH --job-name=wm_m0_arm
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
export WM=$SCR/wm
export DATA=${M0_DATA:?set M0_DATA to the audited cohort root}
: "${M0_STEPS:?pin M0_STEPS before submitting}"
: "${RUNS:?set RUNS to the shared screen output directory}"
: "${SEED:?set SEED to the preregistered training seed}"
export HF_HOME=$WM/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=$WM/venv/bin:$PATH PYTHONUNBUFFERED=1
cd "${M0_REPO:?set M0_REPO to the code snapshot}"
arms=("$@")
index=${SLURM_ARRAY_TASK_ID:-0}
exec bash scripts/run_m0.sh "${arms[index]:?pass the arm names for this job or array}"
