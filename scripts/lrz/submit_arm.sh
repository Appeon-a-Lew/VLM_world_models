#!/usr/bin/env bash
# Submit one B-matrix arm to LRZ: usage: submit_arm.sh <arm> [partition]
# Arms: s_f1 s_f3 s_f1_nope s_f3_nope (see scripts/run_b_matrix.sh).
set -euo pipefail
ARM=${1:?usage: submit_arm.sh <arm> [partition]}
PART=${2:-lrz-hgx-h100-94x4}
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
mkdir -p "$SCR/wm/logs"
sbatch --partition="$PART" --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 \
  --job-name="wm_$ARM" --output="$SCR/wm/logs/${ARM}_%j.log" \
  --export=ARM=$ARM,ALL "$(dirname "$0")/arm_job.sh"
