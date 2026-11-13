#!/usr/bin/env bash
# Isolated, bounded validation rerun after the data/setup audit.
# Usage: timeout --signal=TERM --kill-after=30s 6h bash scripts/run_corrected_baseline.sh RUN_ROOT
# RUN_ROOT contains a frozen `code/` snapshot; uses the existing machine-local venv/cache.
set -euo pipefail
RUN_ROOT=$(realpath "${1:?provide an isolated run root}")
export WM=${WM:-/var/tmp/toyi/wm}
export HF_HOME=$WM/hf
export XDG_CACHE_HOME=$WM/cache
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${SIM_VENV:-$WM/venv}/bin/activate"
cd "$RUN_ROOT/code"
export PYTHONPATH=$PWD
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
exec 9>"$RUN_ROOT/run.lock"
flock -n 9 || { echo 'This run is already active'; exit 1; }
if [[ -e "$RUN_ROOT/status.json" ]]; then
  echo 'Existing run state found; use a new run root rather than overwrite results.'
  exit 1
fi
STAGE=starting
status() {
  python - "$RUN_ROOT" "$1" "$STAGE" "${2:-}" <<'PY'
import datetime, json, os, pathlib, socket, sys
root, state, stage, detail = sys.argv[1:]
p = pathlib.Path(root)
record = dict(state=state, stage=stage, detail=detail, host=socket.gethostname(),
              updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getppid())
(p / 'status.tmp').write_text(json.dumps(record, indent=2) + '\n')
(p / 'status.tmp').replace(p / 'status.json')
with (p / 'events.jsonl').open('a') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record), flush=True)
PY
}
on_exit() {
  rc=$?
  if [[ $rc -ne 0 ]]; then status failed "exit $rc; see logs/$STAGE.log"; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait_for_gpu() {
  # Remote background wait is bounded by the outer six-hour timeout.
  while true; do
    if python - <<'PY'
import ctypes, subprocess, sys
cuda = ctypes.CDLL('libcuda.so.1')
if cuda.cuInit(0) != 0:
    print('CUDA initialization failed; this host cannot run training', file=sys.stderr)
    sys.exit(2)
line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0]
memory, util = [part.strip() for part in line.split(',')]
# Unknown utilization never counts as idle; CUDA must initialize successfully too.
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True)
active = any(line.strip().isdigit() for line in apps.splitlines())
free = int(memory) < 2500 and util.isdigit() and int(util) <= 5 and not active
sys.exit(0 if free else 1)
PY
    then
      break
    else
      rc=$?
      if [[ $rc -ne 1 ]]; then return "$rc"; fi
    fi
    status waiting_for_gpu 'Shared GPU busy; retrying in 60 seconds'
    sleep 60
  done
  status running
}
run_stage() {
  STAGE=$1
  shift
  status running
  "$@" > "$RUN_ROOT/logs/$STAGE.log" 2>&1
}
status running
python -m pip freeze > "$RUN_ROOT/environment.txt"
nvidia-smi > "$RUN_ROOT/gpu_start.txt"
wait_for_gpu
run_stage preflight python -m pytest -q tests
# Regenerate only train and validation/diagnostics. Frozen final-test data stays on the old host.
while read -r split config seed count; do
  STAGE="generate_$split"
  wait_for_gpu
  run_stage "$STAGE" python -m physwm.sim.generate --config "configs/$config" \
    --split "$split" --seed-start "$seed" --n "$count" --out "$RUN_ROOT/data" --workers 8
done <<'SPLITS'
train sim_train.yaml 0 10000
val_id sim_train.yaml 5000000 500
val_ood_objects sim_ood_objects.yaml 6000000 500
val_ood_forces sim_ood_forces.yaml 7000000 500
val_ood_config sim_ood_config.yaml 8000000 500
val_diag_elev sim_diag_elev.yaml 8500000 300
val_diag_count sim_diag_count.yaml 8600000 300
val_diag_corner sim_diag_corner.yaml 8700000 300
val_diag_elev_corner sim_diag_elev_corner.yaml 8800000 300
SPLITS
SPLITS=(val_id val_ood_objects val_ood_forces val_ood_config val_diag_elev val_diag_count val_diag_corner val_diag_elev_corner)
run_stage baselines python -m physwm.eval.baselines --data "$RUN_ROOT/data" \
  --splits "${SPLITS[@]}" --out "$RUN_ROOT/baselines"
python - "$RUN_ROOT" <<'PY'
import pathlib, sys, yaml
root = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(pathlib.Path('configs/sft_qwen3vl2b.yaml').read_text())
cfg['data'] = str(root / 'data')
cfg['out'] = str(root / 'model')
(root / 'sft_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
STAGE=train
while [[ ! -f "$WM/audit_environment_copy_complete" ]]; do
  status waiting_for_environment 'Waiting for the verified environment/cache transfer to finish'
  sleep 30
done
source "$WM/venv/bin/activate"
python -m pip freeze > "$RUN_ROOT/training_environment.txt"
wait_for_gpu
run_stage train timeout --signal=TERM --kill-after=30s 3h python -m physwm.vlm.sft --config "$RUN_ROOT/sft_config.yaml"
STAGE=infer_normal
wait_for_gpu
run_stage infer_normal python -m physwm.vlm.infer --data "$RUN_ROOT/data" --splits "${SPLITS[@]}" \
  --lora "$RUN_ROOT/model/adapter" --batch-size 32 --out "$RUN_ROOT/pred/normal.jsonl"
for mode in none blank swap; do
  STAGE="infer_$mode"
  wait_for_gpu
  run_stage "$STAGE" python -m physwm.vlm.infer --data "$RUN_ROOT/data" --splits val_id \
    --lora "$RUN_ROOT/model/adapter" --image-mode "$mode" --batch-size 32 --out "$RUN_ROOT/pred/$mode.jsonl"
done
run_stage evaluate python -m physwm.eval.evaluate --data "$RUN_ROOT/data" --splits "${SPLITS[@]}" \
  --csv "$RUN_ROOT/results/metrics.csv" --pred \
  "static=$RUN_ROOT/baselines/static.jsonl" "const_velocity=$RUN_ROOT/baselines/const_velocity.jsonl" \
  "sft=$RUN_ROOT/pred/normal.jsonl" "sft_none=$RUN_ROOT/pred/none.jsonl" \
  "sft_blank=$RUN_ROOT/pred/blank.jsonl" "sft_swap=$RUN_ROOT/pred/swap.jsonl"
cp "$RUN_ROOT/logs/evaluate.log" "$RUN_ROOT/results/table.md"
STAGE=complete
status done 'Corrected validation reference and image ablations finished'
