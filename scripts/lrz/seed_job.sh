#!/usr/bin/env bash
# Slurm wrapper for seed_arm.sh: sbatch-style header; submit with
#   sbatch --export=ARM=s_f1,SEED=1,ALL scripts/lrz/seed_job.sh
#SBATCH --job-name=wm_seed
#SBATCH --gres=gpu:1
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
exec bash "$(dirname "$0")/seed_arm.sh" "${ARM:?}" "${SEED:?}"
