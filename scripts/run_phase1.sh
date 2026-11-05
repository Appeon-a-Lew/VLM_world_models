#!/usr/bin/env bash
# Phase 1 end to end (after scripts/gen_all.sh): baselines -> zero-shot -> SFT -> ID/OOD eval ->
# image-dependence test -> results table.
#   bash scripts/run_phase1.sh [stage...]   (env: WM=work, RUNS=$WM/runs)
# Stages: baselines zeroshot sft eval_sft ablate table (default: all, in this order)
set -euo pipefail
WM=${WM:-work}
R=${RUNS:-$WM/runs}
D=$WM/data
TEST="val_id val_ood_objects val_ood_forces val_ood_config"
SFT=$R/sft_qwen3vl2b
stages=${*:-baselines zeroshot sft eval_sft ablate table}

for s in $stages; do case $s in
baselines)
  python -m physwm.eval.baselines --data $D --splits $TEST --out $R/baselines ;;
zeroshot)
  for style in qwen3d center; do
    python -m physwm.vlm.infer --data $D --splits $TEST --style $style \
      --batch-size 32 --out $R/zeroshot/$style.jsonl
  done ;;
sft)
  python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml --out $SFT ;;
eval_sft)
  python -m physwm.vlm.infer --data $D --splits $TEST --lora $SFT/adapter --batch-size 32 \
    --out $R/sft_eval/normal.jsonl ;;
ablate)
  for mode in none blank swap; do
    python -m physwm.vlm.infer --data $D --splits val_id --lora $SFT/adapter --image-mode $mode \
      --batch-size 32 --out $R/sft_eval/$mode.jsonl
  done ;;
table)
  python -m physwm.eval.evaluate --data $D --splits $TEST --csv $R/phase1_results.csv --pred \
    static=$R/baselines/static.jsonl const_vel=$R/baselines/const_velocity.jsonl \
    zs_qwen3d=$R/zeroshot/qwen3d.jsonl zs_center=$R/zeroshot/center.jsonl \
    sft=$R/sft_eval/normal.jsonl sft_noimg=$R/sft_eval/none.jsonl \
    sft_blank=$R/sft_eval/blank.jsonl sft_swap=$R/sft_eval/swap.jsonl | tee $R/phase1_results.md ;;
*) echo "unknown stage $s"; exit 1 ;;
esac; done
