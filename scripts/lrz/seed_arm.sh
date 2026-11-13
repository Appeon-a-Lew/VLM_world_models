#!/usr/bin/env bash
# Seed-repeat arm on LRZ: same recipe as run_b_matrix.sh but with --seed, for the
# training-seed noise floor. usage: seed_arm.sh <arm> <seed>
# Arms: s_f1 | s_f1_nope | l_f1
set -euo pipefail
ARM=${1:?usage: seed_arm.sh <arm> <seed>}
SEED=${2:?usage: seed_arm.sh <arm> <seed>}
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
cd "$WM/repo"
export HF_HOME=$WM/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PATH=$WM/venv/bin:$PATH
export WM
DATA=$WM/data_v3
R=$WM/runs_b
VAL="val_id val_ood_objects val_ood_forces val_ood_config"
ID=${ARM}_seed${SEED}
OUT=$R/$ID
case $ARM in
  s_f1)      FR=0; STYLE=box3d;  PE="";        TASKS=A,B,N ;;
  s_f1_nope) FR=0; STYLE=box3d;  PE="--no-pe"; TASKS=A,B,N ;;
  l_f1)      FR=0; STYLE=center; PE="--no-pe"; TASKS=A,B ;;
  *) echo "unsupported arm $ARM" >&2; exit 2 ;;
esac
if [ -f "$WM/data_v3.tar" ] && [ ! -d "$WM/data_v3" ]; then
  echo "extracting corpus on compute node"; tar xf "$WM/data_v3.tar" -C "$WM" && rm -f "$WM/data_v3.tar"
fi
mkdir -p "$OUT"
if [ ! -d "$OUT/adapter" ]; then
  python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml --data "$DATA" --style $STYLE $PE \
    --frames "$FR" --views 0 --tasks $TASKS --seed "$SEED" --out "$OUT"
fi
PRED=$OUT/pred_val.jsonl
if [ ! -f "$PRED" ]; then
  python -m physwm.vlm.infer --data "$DATA" --splits $VAL --tasks ${TASKS//,/ } \
    --lora "$OUT/adapter" --style $STYLE $PE --frames "$FR" --views 0 \
    --batch-size 32 --out "$PRED"
fi
python -m physwm.eval.evaluate --data "$DATA" --splits $VAL --pred "$ID=$PRED" --csv "$OUT/metrics.csv"
echo "seed arm $ID done"
