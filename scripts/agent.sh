#!/usr/bin/env bash
# The research agent's ONLY entry point to the GPU box. Every op is checked by physwm/agent/exec.py
# against configs/research_agent_policy.yaml.
#
#   scripts/agent.sh status
#   scripts/agent.sh submit <exp_id>             # run from the experiment's git worktree, clean + committed
#   scripts/agent.sh results <exp_id>
#   scripts/agent.sh log <exp_id> [n]
#   scripts/agent.sh cancel <exp_id>
#   scripts/agent.sh compare <a> <b>             # paired bootstrap on the primary metric (exp ids or static)
#   scripts/agent.sh gen <config> <split> <seed_start> <n>
#   scripts/agent.sh final-eval <exp_id>         # refused without a human approval token
set -euo pipefail
HOST=${AGENT_HOST:-nc-11-1-2}
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=20 "$HOST")
remote() {  # remote <exec args...>; arguments are shell-quoted
  local q
  q=$(printf ' %q' "$@")
  "${SSH[@]}" "source ~/adl4cv-worlmodels/scripts/env.sh >/dev/null 2>&1 && python -m physwm.agent.exec$q"
}
id_ok() { [[ "$1" =~ ^exp[0-9]{3}_[a-z0-9_]{1,48}$ ]] || { echo "bad experiment id: $1" >&2; exit 2; }; }

op=${1:-}; shift || true
case "$op" in
  status) remote status ;;
  submit)
    exp=$1; id_ok "$exp"
    git rev-parse --is-inside-work-tree >/dev/null
    [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ] || { echo "submit from an exp/* worktree, not main" >&2; exit 2; }
    [ -z "$(git status --porcelain)" ] || { echo "worktree not clean; commit first" >&2; exit 2; }
    git cat-file -e "HEAD:research/experiments/$exp/spec.yaml" 2>/dev/null \
      || { echo "research/experiments/$exp/spec.yaml not committed" >&2; exit 2; }
    commit=$(git rev-parse HEAD)
    git archive --format=tar HEAD | remote submit --exp "$exp" --commit "$commit" ;;
  results) id_ok "$1"; remote results --exp "$1" ;;
  log) id_ok "$1"; remote log --exp "$1" --n "${2:-40}" ;;
  cancel) id_ok "$1"; remote cancel --exp "$1" ;;
  compare) remote compare --a "$1" --b "$2" ;;
  gen) remote gen --config "$1" --split "$2" --seed-start "$3" --n "$4" ;;
  final-eval) id_ok "$1"; remote final-eval --exp "$1" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
