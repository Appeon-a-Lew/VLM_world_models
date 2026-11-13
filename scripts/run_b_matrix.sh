#!/usr/bin/env bash
# B1-B5 grounding baseline matrix (research/PLAN.md): style box3d, GT depth, view 0.
# Trains four arms (1 vs 3 input frames x PE on/off), infers tasks A,B,N on the val
# splits, scores them, and writes $R/matrix.md mapping plan ids to (arm, task).
# Usage on a lab box: source scripts/env.sh && bash scripts/run_b_matrix.sh [arm ...]
#   arms: s_f1 s_f3 s_f1_nope s_f3_nope (default: all); eval follows each trained arm.
# DATA must point at the v3 corpus for the 3-frame arms (legacy data has 2 frames).
set -euo pipefail
R=${RUNS:-$WM/runs_b}
D=${DATA:-$WM/data}
VAL="val_id val_ood_objects val_ood_forces val_ood_config"
CFG=${SFTCFG:-configs/sft_qwen3vl2b.yaml}
TASKS=A,B,N
mkdir -p "$R"

frames_ok() { # need
  python - "$D" "$1" <<'PY'
import json, pathlib, sys
root, need = pathlib.Path(sys.argv[1]), int(sys.argv[2])
meta = sorted((root / "train").glob("*/meta.json"))[0]
ft = json.loads(meta.read_text()).get("frame_times", [0.0, 1.0])
sys.exit(0 if len(ft) >= need else 1)
PY
}

train() { # id frames [--no-pe]
  local id=$1 frames=$2 nope=${3:-}
  if [ -d "$R/$id/adapter" ]; then echo "skip trained arm $id"; return 0; fi
  python -m physwm.vlm.sft --config "$CFG" --data "$D" --style box3d --frames "$frames" \
    --views 0 --tasks $TASKS $nope --out "$R/$id"
}

eval_arm() { # id frames [--no-pe]
  local id=$1 frames=$2 nope=${3:-}
  local pred=$R/$id/pred_val.jsonl
  if [ ! -f "$pred" ]; then
    python -m physwm.vlm.infer --data "$D" --splits $VAL --tasks A B N \
      --lora "$R/$id/adapter" --style box3d --frames "$frames" --views 0 $nope \
      --batch-size 32 --out "$pred"
  fi
  python -m physwm.eval.evaluate --data "$D" --splits $VAL --pred "$id=$pred" \
    --csv "$R/$id/metrics.csv"
}

train_c() { # id frames: centre-style reference arm (legacy answer format) on the same corpus
  local id=$1 frames=$2
  if [ -d "$R/$id/adapter" ]; then echo "skip trained arm $id"; return 0; fi
  python -m physwm.vlm.sft --config "$CFG" --data "$D" --style center --frames "$frames" \
    --views 0 --tasks A,B --out "$R/$id"
}

eval_c() { # id frames
  local id=$1 frames=$2
  local pred=$R/$id/pred_val.jsonl
  if [ ! -f "$pred" ]; then
    python -m physwm.vlm.infer --data "$D" --splits $VAL --tasks A B \
      --lora "$R/$id/adapter" --style center --frames "$frames" --views 0 \
      --batch-size 32 --out "$pred"
  fi
  python -m physwm.eval.evaluate --data "$D" --splits $VAL --pred "$id=$pred" \
    --csv "$R/$id/metrics.csv"
}

arm() {
  case $1 in
    s_f1)      train s_f1 0;     eval_arm s_f1 0 ;;
    c_f1)      train_c c_f1 0;   eval_c c_f1 0 ;;
    s_f3)      frames_ok 3 || { echo "arm s_f3 needs >=3 frame_times in $D (v3 corpus)" >&2; exit 3; }
               train s_f3 0,1,2; eval_arm s_f3 0,1,2 ;;
    s_f1_nope) train s_f1_nope 0 --no-pe;     eval_arm s_f1_nope 0 --no-pe ;;
    s_f3_nope) frames_ok 3 || { echo "arm s_f3_nope needs >=3 frame_times in $D (v3 corpus)" >&2; exit 3; }
               train s_f3_nope 0,1,2 --no-pe; eval_arm s_f3_nope 0,1,2 --no-pe ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for a in ${*:-s_f1 s_f3 s_f1_nope s_f3_nope}; do arm "$a"; done

cat > "$R/matrix.md" <<'MD'
# B-matrix mapping (research/PLAN.md)
| plan id | arm | task | input frames | PE |
|---|---|---|---|---|
| B1 | s_f1 | A | 0 | on |
| B2 | s_f1 | B | 0 | on |
| B4 | s_f1 | N | 0 | on |
| B3 | s_f3 | B | 0,1,2 | on |
| B5a | s_f1_nope | B | 0 | off |
| B5b | s_f3_nope | B | 0,1,2 | off |
| C-ref | c_f1 | A,B | 0 | n/a (centre style, legacy format) |
MD
echo "wrote $R/matrix.md"
