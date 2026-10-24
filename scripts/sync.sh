#!/usr/bin/env bash
# Sync the repo (code only) to the shared NFS home on the TUM lab machines.
# Venv, data and HF cache live on the machine's local disk: /var/tmp/toyi/wm
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete --exclude .git --exclude __pycache__ --exclude data --exclude runs \
  -e "ssh -o BatchMode=yes -o LogLevel=ERROR" ./ halle:~/adl4cv-worlmodels/
