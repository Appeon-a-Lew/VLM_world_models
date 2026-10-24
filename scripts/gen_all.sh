#!/usr/bin/env bash
# Generate all Phase-1 splits. Seed ranges are disjoint per split, so every episode is reproducible
# from (config, seed). Usage (on a lab machine): source scripts/env.sh && bash scripts/gen_all.sh
set -euo pipefail
OUT=${OUT:-$WM/data}
W=${WORKERS:-20}
gen() { python -m physwm.sim.generate --config "$1" --split "$2" --seed-start "$3" --n "$4" --out "$OUT" --workers "$W"; }

[ -d "$OUT/train" ] || gen configs/sim_train.yaml train 0 10000
gen configs/sim_train.yaml       id_test      1000000  1000
gen configs/sim_ood_objects.yaml ood_objects  2000000  1000
gen configs/sim_ood_forces.yaml  ood_forces   3000000  1000
gen configs/sim_ood_config.yaml  ood_config   4000000  1000
# validation splits for the research loop (same configs, fresh seeds); the four id_test / ood_* splits above are the
# frozen final test (manifests/final_test.json) and are only evaluated with human approval
gen configs/sim_train.yaml       val_id          5000000  500
gen configs/sim_ood_objects.yaml val_ood_objects 6000000  500
gen configs/sim_ood_forces.yaml  val_ood_forces  7000000  500
gen configs/sim_ood_config.yaml  val_ood_config  8000000  500
# diagnostic validation splits (policy data.diagnostic_splits): one or two ood_config factors each
gen configs/sim_diag_elev.yaml   val_diag_elev   8500000  300
gen configs/sim_diag_count.yaml  val_diag_count  8600000  300
gen configs/sim_diag_corner.yaml val_diag_corner 8700000  300
gen configs/sim_diag_elev_corner.yaml val_diag_elev_corner 8800000  300
du -sh "$OUT"/*
