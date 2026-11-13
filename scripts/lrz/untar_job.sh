#!/usr/bin/env bash
# Extract the relayed corpus tarball on a compute node (login nodes stay for prep).
#SBATCH --job-name=wm_untar
#SBATCH --partition=lrz-cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SCRATCH:-${WORK:-$HOME/scratch}}/wm"
tar xf data_v3.tar
rm -f data_v3.tar
ls data_v3 | tr '\n' ' '; echo
du -sh data_v3
