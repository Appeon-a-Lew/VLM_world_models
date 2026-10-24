#!/usr/bin/env bash
# Start the autonomous research loop on this laptop in tmux, reachable via Remote Control.
#   scripts/start_loop.sh          # then: tmux attach -t wm-agent   (or open the session on claude.ai / phone)
# Stop: type "stop the loop" in the session (or tmux kill-session -t wm-agent).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO/../wm-worktrees"
tmux has-session -t wm-agent 2>/dev/null && { echo "already running: tmux attach -t wm-agent"; exit 0; }
tmux new-session -d -s wm-agent -c "$REPO" \
  "claude --remote-control wm-research --permission-mode acceptEdits \
     --settings research/agent_settings.json \
     '/loop Follow research/LOOP.md for one iteration.'"
echo "started: tmux attach -t wm-agent"
