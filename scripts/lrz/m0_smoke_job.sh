#!/usr/bin/env bash
# LRZ end-to-end Tier 1 smoke: validates the M0 pipeline on the offline stack before any
# screen submission -- comma task lists into SFT, task I on the fixed 0.1 s pair, ctx3d
# answers, --no-pe primary, plan/step pinning, IDP + predicted-context scoring, and the
# paired_arms gate tooling. Tiny limits; not a result.
#   sbatch scripts/lrz/m0_smoke_job.sh           (expects CF_SPLIT in the environment)
#SBATCH --job-name=wm_m0_smoke
#SBATCH --gres=gpu:1
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
DATA=${M0_DATA:-$WM/data_v3}
CF=${CF_SPLIT:?export CF_SPLIT (matched counterfactual split name in M0_DATA)}
cd "${M0_REPO:-$WM/repo}"
export HF_HOME=$WM/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PATH=$WM/venv/bin:$PATH WM
if [ -f "$WM/data_v3.tar" ] && [ ! -d "$WM/data_v3" ]; then
  echo "extracting corpus on compute node"; tar xf "$WM/data_v3.tar" -C "$WM" && rm -f "$WM/data_v3.tar"
fi
OUT=$(mktemp -d "$WM/runs_m0_smoke.XXXXXX")
echo "Smoke outputs: $OUT"
SMOKE_LIMIT=8

echo "== plan (P0/P1: comma tasks, pinned steps before any training)"
python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml --data "$DATA" \
  --train-split train --limit 64 --style box3d --frames 0 --views 0 --tasks A,B,I --plan

echo "== calibrate (preregistered static baseline; no model files)"
python -m physwm.eval.counterfactual --data "$DATA" --splits "$CF" --static \
  --out "$OUT/calib_static"
python -m physwm.eval.paired_arms --data "$DATA" --splits "$CF" --calibrate \
  --out "$OUT/calib_static/threshold.json"

echo "== both arm, tiny (ctx3d + IDP, --no-pe primary)"
python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml --data "$DATA" \
  --train-split train --limit 64 --epochs 0.3 --seed 1 --style ctx3d --frames 0 --views 0 \
  --tasks A,B,I --no-pe --out "$OUT/both_smoke"

echo "== inference over A B I + scoring (IDP and predicted-context tables must appear)"
python -m physwm.vlm.infer --data "$DATA" --splits val_id --limit "$SMOKE_LIMIT" --tasks A B I \
  --style ctx3d --frames 0 --views 0 --no-pe --lora "$OUT/both_smoke/adapter" \
  --batch-size 8 --out "$OUT/both_smoke/pred_smoke.jsonl"
python -m physwm.eval.evaluate --data "$DATA" --splits val_id --limit "$SMOKE_LIMIT" \
  --pred smoke="$OUT/both_smoke/pred_smoke.jsonl" --csv "$OUT/both_smoke/metrics.csv"

echo "== oracle-context diagnostic (support + action payload injected)"
python -m physwm.vlm.infer --data "$DATA" --splits val_id --limit "$SMOKE_LIMIT" --tasks B \
  --style ctx3d --frames 0 --views 0 --no-pe --oracle-context \
  --lora "$OUT/both_smoke/adapter" --batch-size 8 --out "$OUT/both_smoke/pred_oracle.jsonl"
python -m physwm.eval.evaluate --data "$DATA" --splits val_id --limit "$SMOKE_LIMIT" \
  --pred oracle="$OUT/both_smoke/pred_oracle.jsonl" --csv "$OUT/both_smoke/oracle.csv"

echo "M0 smoke finished on $(hostname); inspect $OUT for the tables and threshold.json"
