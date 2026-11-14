#!/usr/bin/env bash
# Slurm job body for scripts/agent_lrz.sh submit: unpack the submitted commit
# archive, re-check the spec offline, train/infer/evaluate with the archive's own
# code (snapshot discipline, same spirit as physwm/agent/exec.py), write results
# to $WM/runs_exp/<exp>/. Protected splits are never touched: inference runs on
# validation + diagnostic splits only, training split comes from the spec and is
# validated by check_spec.py.
set -euo pipefail
SCR=${SCRATCH:-${WORK:-$HOME/scratch}}
WM=$SCR/wm
EXP=${EXP:?}; COMMIT=${COMMIT:?}
WORK=$WM/exp_runs/${EXP}_${COMMIT}
OUT=$WM/runs_exp/$EXP
DATA=$WM/data_v3
rm -rf "$WORK"; mkdir -p "$WORK" "$OUT"
tar xf "$WM/exp_archives/${EXP}_${COMMIT}.tar" -C "$WORK"
cd "$WORK"
python3 scripts/check_spec.py "research/experiments/$EXP/spec.yaml" | python3 -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d["ok"] else 2)'
export HF_HOME=$WM/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK PATH=$WM/venv/bin:$PATH
SPEC="research/experiments/$EXP/spec.yaml"
get() { python3 -c "import yaml,sys; v=yaml.safe_load(open('$SPEC'))$1; print('' if v is None else v)"; }
STYLE=$(get ".get('style','center')"); KIND=$(get ".get('kind','train')")
FRAMES=$(get ".get('frames') or []" | tr -d '[]' | tr -s ' ' ',' | sed 's/^,//;s/,$//')
VIEWS=$(get ".get('views') or []" | tr -d '[]' | tr -s ' ' ',' | sed 's/^,//;s/,$//')
TASKS=$(get ".get('tasks') or []" | tr -d "[]'" | tr -s ' ' ',' | sed 's/^,//;s/,$//')
NOPE=$(get ".get('use_pe', True)")
VAL=$(python3 -c "import yaml; p=yaml.safe_load(open('configs/research_agent_policy.yaml'))['data']; print(' '.join(p['validation_splits'] + p.get('diagnostic_splits', [])))")
PEFLAG=""; [ "$NOPE" = "False" ] && PEFLAG="--no-pe"
LORA=""
if [ "$KIND" = "train" ]; then
  CFG=$(get ".get('train_config')")
  SFT="--style $STYLE"
  [ -n "$FRAMES" ] && SFT="$SFT --frames $FRAMES"
  [ -n "$VIEWS" ] && SFT="$SFT --views $VIEWS"
  [ -n "$TASKS" ] && SFT="$SFT --tasks $TASKS"
  python3 -m physwm.vlm.sft --config "$CFG" --data "$DATA" $SFT $PEFLAG --out "$OUT/model"
  LORA=$OUT/model/adapter
else
  LORA=$(get ".get('lora')")
  [ -d "$LORA" ] || { echo "eval_only spec needs an existing adapter path on this backend: $LORA" >&2; exit 2; }
fi
INF="--style $STYLE --batch-size 32 --lora $LORA"
[ -n "$FRAMES" ] && INF="$INF --frames $FRAMES"
[ -n "$VIEWS" ] && INF="$INF --views $VIEWS"
[ -n "$TASKS" ] && INF="$INF --tasks ${TASKS//,/ }"
[ -n "$PEFLAG" ] && INF="$INF $PEFLAG"
python3 -m physwm.vlm.infer --data "$DATA" --splits $VAL $INF --out "$OUT/pred_normal.jsonl"
AB=$(get ".get('ablations') or ['none']" | tr -d "[]'" | tr -s ' ' ' ')
ABL_SPLIT=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/research_agent_policy.yaml'))['data']['ablation_split'])")
for mode in $AB; do
  [ "$mode" = "none" ] && continue
  python3 -m physwm.vlm.infer --data "$DATA" --splits "$ABL_SPLIT" --image-mode "$mode" $INF --out "$OUT/pred_$mode.jsonl"
done
python3 -m physwm.eval.evaluate --data "$DATA" --splits $VAL --pred "$EXP=$OUT/pred_normal.jsonl" --csv "$OUT/metrics.csv"
{ echo "# $EXP results (LRZ, commit $COMMIT)"; cut -d, -f1,2,3,6,12,23,24,25 "$OUT/metrics.csv" | column -s, -t; } > "$OUT/results.md"
echo "exp $EXP done on $(hostname)"
