#!/usr/bin/env bash
# Preflight + start instructions for the phase-2 autonomous research loop (OpenCode).
#
#   scripts/start_loop.sh
#
# This replaces the phase-1 tmux + Claude Code launcher. It starts no agent process; it
# checks the prerequisites, prints the watcher pattern, and prints the exact line to send
# to a fresh orchestrator session. Stop the loop by telling the session to stop; nothing
# is left running except the watcher shells you started yourself.
#
# --- Watcher / recovery template (one per live backend) --------------------------------
# Start a watcher only while a job is queued or running. It polls the agent script -- NOT
# squeue/sacct -- once per 300 s, and prints a WM_WAKE marker when the backend no longer
# reports a running job. That marker is what resumes the orchestrator session.
#
#   nohup bash -c '
#     while :; do
#       out=$(scripts/agent.sh status 2>&1) || { echo "WM_WAKE lab status-failed"; exit 0; }
#       if ! printf "%s" "$out" | grep -qi running; then
#         echo "WM_WAKE lab idle-or-changed"; printf "%s\n" "$out"; exit 0
#       fi
#       sleep 300
#     done' >> ../wm-worktrees/watch.log 2>&1 &
#
# For LRZ, use scripts/agent_lrz.sh in the same template. Adapt the "running" match to the
# backend's status format the first time. Never wrap squeue/sacct in a loop, and never poll
# more often than every 5 minutes.
# ---------------------------------------------------------------------------------------

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
STATUS=0

ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; STATUS=1; }

printf 'repo: %s\n' "$REPO"

# --- 1. git: main, clean ---------------------------------------------------------------
branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ "$branch" = "main" ]; then
  ok "on branch main"
else
  bad "on branch '${branch:-unknown}', expected main"
fi

if [ -z "$(git -C "$REPO" status --porcelain)" ]; then
  ok "worktree clean"
else
  bad "worktree dirty; commit or stash before starting the loop"
fi

# --- 2. experiment worktrees ------------------------------------------------------------
mkdir -p "$REPO/../wm-worktrees"
ok "experiment worktrees: $REPO/../wm-worktrees"

# --- 3. at least one GPU backend --------------------------------------------------------
LAB=0
LRZ=0

if [ ! -x "$REPO/scripts/agent.sh" ]; then
  bad "scripts/agent.sh missing or not executable"
elif "$REPO/scripts/agent.sh" status >/dev/null 2>&1; then
  LAB=1
  ok "lab backend reachable (scripts/agent.sh status)"
else
  warn "lab backend unreachable (box down, power-cycled or no route)"
fi

if [ ! -x "$REPO/scripts/agent_lrz.sh" ]; then
  warn "scripts/agent_lrz.sh not present yet (LRZ backend unavailable)"
elif "$REPO/scripts/agent_lrz.sh" status >/dev/null 2>&1; then
  LRZ=1
  ok "LRZ backend reachable (scripts/agent_lrz.sh status)"
else
  warn "LRZ backend not answering (partition at Priority is normal; status must still reply)"
fi

if [ "$LAB" -eq 0 ] && [ "$LRZ" -eq 0 ]; then
  bad "no GPU backend reachable; fix one before starting the loop"
else
  ok "GPU backends: lab=$LAB lrz=$LRZ (policy caps: 1 lab job, 6 LRZ jobs, 8 h each)"
fi

# --- 4. HF auth (needed for the public dashboard bundle) --------------------------------
if command -v hf >/dev/null 2>&1; then
  if hf auth whoami >/dev/null 2>&1; then
    ok "hf authenticated"
  else
    warn "hf CLI present but not authenticated; dashboard upload will fail"
  fi
elif command -v huggingface-cli >/dev/null 2>&1; then
  if huggingface-cli whoami >/dev/null 2>&1; then
    ok "huggingface-cli authenticated"
  else
    warn "huggingface-cli present but not authenticated; dashboard upload will fail"
  fi
else
  warn "no hf CLI (hf / huggingface-cli); dashboard upload will fail"
fi

# --- 5. verdict --------------------------------------------------------------------------
printf '\n'
if [ "$STATUS" -ne 0 ]; then
  printf 'Preflight failed: fix the FAIL lines above before starting the loop.\n'
  exit 1
fi

cat <<'EOF'
Preflight passed. Start an OpenCode session in this repo and send exactly:

    Follow research/LOOP.md for one iteration.

The orchestrator checks the agent scripts first, so a job that finished while no session
was open gets reported before any new planning.

Resume after a watcher notification with:

    Follow research/LOOP.md; a background watcher reported a job state change.

Notes:
  - Watchdog shells are started by you, one per live backend (template at the top of this
    script); the loop itself never polls squeue/sacct.
  - Subagents are briefed with the role files in .claude/agents/; nothing survives a session
    except what is committed under research/.
EOF