#!/usr/bin/env bash
# M0 screen arms (research/PREREG.md, plan v1.1) — launch-blocker fixes pinned 2026-11-21:
#   * SFT takes a comma task list (A,B,I); inference takes separate arguments (A B I).
#   * The shared optimizer-step target is pinned BEFORE any arm trains: every arm is
#     planned (--plan, no GPU), the target is the max natural count, and --max-steps is an
#     exact target (shorter arms extend, longer cap). Consumed examples per task and
#     padded/unpadded/supervised tokens are logged cumulatively. Forward
#     exposure per step is therefore NOT matched across arms (IDP examples are additive by
#     design, research/PREREG.md addendum); the logs make the exposure auditable.
#   * The primary screen trains and infers WITHOUT the xyz PE (GT-depth PE is privileged):
#     DEPTH_DIAG=1 reruns an arm with PE into a separate *_pe directory, never the gate.
#   * `calibrate` prints E_static + the threshold from the preregistered static baseline.
#
#   source scripts/env.sh && CF_SPLIT=<matched-family split> bash scripts/run_m0.sh [arm ...]
#   source scripts/env.sh && CF_SPLIT=<...> SEED=2 bash scripts/run_m0.sh fwd both
#   source scripts/env.sh && CF_SPLIT=<...> bash scripts/run_m0.sh calibrate
set -euo pipefail
R=${RUNS:-$WM/runs_m0}
D=${DATA:-$WM/data}
CF=${CF_SPLIT:?set CF_SPLIT to the matched counterfactual split name in \$DATA}
VAL="val_id val_ood_forces"
SEED=${SEED:-1}
SFTCFG=${SFTCFG:-configs/sft_qwen3vl2b.yaml}
PE_FLAGS=(--no-pe); PE_SUFFIX=""
if [ "${DEPTH_DIAG:-0}" = 1 ]; then PE_FLAGS=(); PE_SUFFIX="_pe"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$R"

style_of() { case $1 in fwd|idp) echo box3d;; ctx|both) echo ctx3d;; *) return 2;; esac; }
tasks_sft() { case $1 in fwd|ctx) echo "A,B";; idp|both) echo "A,B,I";; *) return 2;; esac; }
tasks_inf() { case $1 in fwd|ctx) echo "A B";; idp|both) echo "A B I";; *) return 2;; esac; }
arm_id() { echo "${1}_s${SEED}${PE_SUFFIX}"; }

if [ "${1:-}" = calibrate ]; then
  # Pre-registered pre-flight: E_static from the static baseline, no model files needed.
  python -m physwm.eval.counterfactual --data "$D" --splits "$CF" --static \
    --out "$R/calib_static"
  python -m physwm.eval.paired_arms --data "$D" --splits "$CF" --calibrate \
    --out "$R/calib_static/threshold.json"
  echo "if E_static < 0.08 m regenerate matched families with stronger effects (PREREG.md)"
  exit 0
fi

plan_arm() { # arm -> JSON with samples_per_task and natural_optimizer_steps
  python -m physwm.vlm.sft --config "$SFTCFG" --data "$D" --train-split train \
    --seed "$SEED" --style "$(style_of "$1")" --frames 0 --views 0 \
    --tasks "$(tasks_sft "$1")" --plan
}

natural_steps() { # arm -> the natural optimizer-step count
  plan_arm "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["natural_optimizer_steps"])'
}

training_artifacts() (
  cd "$1" || return 1
  files=(config.json adapter/adapter_config.json)
  if [ -s adapter/adapter_model.safetensors ]; then
    files+=(adapter/adapter_model.safetensors)
  else
    files+=(adapter/adapter_model.bin)
  fi
  if [ -n "$PE_SUFFIX" ]; then files+=(pe.pt); fi
  for file in "${files[@]}"; do
    [ -s "$file" ] || return 1
  done
  sha256sum -- "${files[@]}"
)

train() { # arm, training fingerprint
  local id style tasks stamp artifacts
  id=$(arm_id "$1"); style=$(style_of "$1"); tasks=$(tasks_sft "$1")
  stamp=$R/$id/train.complete
  if [ -f "$stamp" ] || [ -e "$R/$id/config.json" ] || [ -e "$R/$id/adapter" ]; then
    if [ ! -f "$stamp" ] || ! artifacts=$(training_artifacts "$R/$id") ||
       [ "$(<"$stamp")" != "$2"$'\n'"$artifacts" ]; then
      echo "$id has incomplete or incompatible training artifacts. Set RUNS to a fresh directory." >&2
      return 1
    fi
    echo "skip verified training run $id"
    return 0
  fi
  python -m physwm.vlm.sft --config "$SFTCFG" --data "$D" --train-split train \
    --seed "$SEED" --style "$style" --frames 0 --views 0 --tasks "$tasks" \
    --max-steps "$M0_STEPS" "${PE_FLAGS[@]}" --out "$R/$id"
  artifacts=$(training_artifacts "$R/$id")
  printf '%s\n' "$2" "$artifacts" > "$stamp.tmp"
  mv -f -- "$stamp.tmp" "$stamp"
}

predict() ( # output path, training fingerprint, inference arguments
  local pred=$1 fingerprint stamp pending
  fingerprint=$(printf '%s\0' "${@:2}" | sha256sum)
  shift 2
  stamp=$pred.complete
  if [ -s "$pred" ] && [ -f "$stamp" ] &&
     [ "$(<"$stamp")" = "$fingerprint $(sha256sum < "$pred")" ]; then
    echo "skip verified predictions $pred"
    return 0
  fi
  pending=$(mktemp "$pred.tmp.XXXXXX")
  trap 'rm -f -- "$pending" "$pending.complete"' EXIT
  python -m physwm.vlm.infer "$@" --out "$pending"
  if [ ! -s "$pending" ]; then
    echo "inference produced no predictions for $pred" >&2
    return 1
  fi
  printf '%s %s\n' "$fingerprint" "$(sha256sum < "$pending")" > "$pending.complete"
  mv -f -- "$pending" "$pred"
  mv -f -- "$pending.complete" "$stamp"
)

eval_arm() { # arm, training fingerprint
  local id style tasks pred mode
  id=$(arm_id "$1"); style=$(style_of "$1"); tasks=$(tasks_inf "$1")
  pred=$R/$id/pred_val.jsonl
  predict "$pred" "$2" --data "$D" --splits $VAL "$CF" --style "$style" \
    --frames 0 --views 0 --tasks $tasks "${PE_FLAGS[@]}" --lora "$R/$id/adapter" --batch-size 32
  python -m physwm.eval.evaluate --data "$D" --splits $VAL "$CF" --csv "$R/$id/metrics.csv" \
    --pred "$id=$pred"
  if [ "$SEED" = 1 ] && [ -z "$PE_SUFFIX" ]; then  # ablation battery once, screening seed, no-PE only
    for mode in none blank swap; do
      predict "$R/$id/pred_$mode.jsonl" "$2" --data "$D" --splits val_id --image-mode "$mode" \
        --style "$style" --frames 0 --views 0 --tasks A B "${PE_FLAGS[@]}" \
        --lora "$R/$id/adapter" --batch-size 32
    done
  fi
}

run_arm() (
  local id config_digest fingerprint
  id=$(arm_id "$1")
  exec 9>"$R/$id.lock"
  flock -n 9 || { echo "run $id is already active" >&2; return 1; }
  config_digest=$(sha256sum < "$SFTCFG")
  fingerprint=$(printf '%s\0' "$config_digest" "$D" "$id" "$M0_STEPS" \
    "$(style_of "$1")" "$(tasks_sft "$1")" "${PE_FLAGS[@]}" | sha256sum)
  train "$1" "$fingerprint"
  eval_arm "$1" "$(sha256sum < "$R/$id/train.complete")"
)

gate() {  # prints the pre-registered contrast; the tool verifies seed identity itself
  echo "screen contrast (decides whether +both earns seed 2; one seed can only earn):"
  echo "  python -m physwm.eval.paired_arms --data $D --splits $CF \\"
  echo "    --a fwd=$R/fwd_s1/pred_val.jsonl --b both=$R/both_s1/pred_val.jsonl \\"
  echo "    --out $R/paired/screen_s1.json"
  echo "gate (needs BOTH distinct preregistered seed pairs):"
  echo "  python -m physwm.eval.paired_arms --data $D --splits $CF \\"
  echo "    --a fwd=$R/fwd_s1/pred_val.jsonl --a fwd=$R/fwd_s2/pred_val.jsonl \\"
  echo "    --b both=$R/both_s1/pred_val.jsonl --b both=$R/both_s2/pred_val.jsonl \\"
  echo "    --out $R/paired/m0_gate.json"
}

if [ -z "${M0_STEPS:-}" ]; then
  M0_STEPS=0
  for a in fwd idp ctx both; do
    s=$(natural_steps "$a")
    [ "$s" -gt "$M0_STEPS" ] && M0_STEPS=$s
  done
  echo "pinning M0_STEPS=$M0_STEPS (max natural optimizer steps across arms; exact target"
  echo "per arm: shorter arms extend to it, longer ones cap at it)"
fi

D=$(realpath "$D")
if [ "$#" -eq 0 ]; then set -- fwd idp ctx both; fi
for a in "$@"; do run_arm "$a"; done
gate
echo "wrote $R"
