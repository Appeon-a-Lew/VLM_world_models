#!/usr/bin/env bash
# The research agent's entry point to the LRZ Slurm backend. Same op surface as
# scripts/agent.sh; jobs run the submitted commit archive on H100 nodes with
# --gres=gpu:1 and nice, capped by configs/research_agent_policy.yaml (resources.lrz).
#
#   scripts/agent_lrz.sh status
#   scripts/agent_lrz.sh submit <exp_id>          # from the experiment's worktree, clean + committed
#   scripts/agent_lrz.sh results <exp_id>
#   scripts/agent_lrz.sh log <exp_id> [n]
#   scripts/agent_lrz.sh cancel <exp_id>
#   scripts/agent_lrz.sh compare ...              # not supported on LRZ v1 (use agent.sh compare or local analysis)
set -euo pipefail
PART=${LRZ_PARTITION:-lrz-hgx-h100-94x4}
MAX_JOBS=${LRZ_MAX_JOBS:-6}
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=20 ai)
REMOTE_ROOT='~/scratch/wm'
id_ok() { [[ "$1" =~ ^exp[0-9]{3}_[a-z0-9_]{1,48}$ ]] || { echo "bad experiment id: $1" >&2; exit 2; }; }

op=${1:-}; shift || true
case "$op" in
  status)
    "${SSH[@]}" 'squeue -h -u "$USER" -o "%i %j %T %M" 2>/dev/null | head -12; echo ---; ls ~/scratch/wm/runs_exp 2>/dev/null' ;;
  submit)
    exp=$1; id_ok "$exp"
    git rev-parse --is-inside-work-tree >/dev/null
    [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ] || { echo "submit from an exp/* worktree, not main" >&2; exit 2; }
    [ -z "$(git status --porcelain)" ] || { echo "worktree not clean; commit first" >&2; exit 2; }
    git cat-file -e "HEAD:research/experiments/$exp/spec.yaml" 2>/dev/null \
      || { echo "research/experiments/$exp/spec.yaml not committed" >&2; exit 2; }
    python3 scripts/check_spec.py "research/experiments/$exp/spec.yaml" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["ok"] else print(d["errors"]))' \
      || { echo "spec check failed" >&2; exit 2; }
    running=$("${SSH[@]}" 'squeue -h -u "$USER" -r -o "%j" 2>/dev/null | grep -c "^wm_exp_" || true')
    [ "$running" -lt "$MAX_JOBS" ] || { echo "DENIED: $running LRZ jobs running (cap $MAX_JOBS)" >&2; exit 4; }
    commit=$(git rev-parse HEAD)
    short=${commit:0:8}
    budget=$(python3 -c "import yaml;print(min(8.0,float(yaml.safe_load(open('research/experiments/$exp/spec.yaml'))['budget_hours']))")
    git archive --format=tar HEAD | "${SSH[@]}" "mkdir -p $REMOTE_ROOT/exp_archives && cat > $REMOTE_ROOT/exp_archives/${exp}_${short}.tar"
    "${SSH[@]}" "cd $REMOTE_ROOT/repo && sbatch --partition=$PART --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=${budget}:00:00 --nice=10 --job-name=wm_exp_$exp --export=EXP=$exp,COMMIT=$short,ALL scripts/lrz/exp_job.sh" ;;
  results)
    id_ok "$1"
    "${SSH[@]}" "cut -d, -f1,2,3,6,12,23,24,25 $REMOTE_ROOT/runs_exp/$1/metrics.csv 2>/dev/null | column -s, -t; echo ---; tail -5 $REMOTE_ROOT/runs_exp/$1/results.md 2>/dev/null" ;;
  log)
    id_ok "$1"
    "${SSH[@]}" "f=\$(ls -t ~/scratch/wm/repo/slurm-*wm_exp_$1*.out ~/scratch/wm/repo/slurm-*.out 2>/dev/null | head -1); tail -${2:-40} \"\$f\"" ;;
  cancel)
    id_ok "$1"
    "${SSH[@]}" "squeue -h -u \"\$USER\" -o \"%i %j\" 2>/dev/null | awk -v e=wm_exp_$1 '\$2==e{print \$1}' | xargs -r scancel" ;;
  compare)
    echo "compare is not supported on the LRZ backend (v1); use scripts/agent.sh compare or physwm/eval/analyze_motion.py" >&2
    exit 5 ;;
  *)
    echo "usage: agent_lrz.sh status|submit|results|log|cancel|compare" >&2; exit 2 ;;
esac
