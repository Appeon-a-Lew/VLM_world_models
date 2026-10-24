#!/usr/bin/env bash
# HUMAN ONLY. Approve exactly one final-test evaluation of one experiment commit.
# The agent is denied this script in .claude/settings.json.
#   scripts/approve.sh exp003_foo
set -euo pipefail
exp=$1
HOST=${AGENT_HOST:-nc-10-1-1}
ssh -o BatchMode=yes -o LogLevel=ERROR "$HOST" "set -e; W=/var/tmp/toyi/wm
  c=\$(python3 -c \"import json;print(json.load(open('\$W/experiments/$exp/status.json'))['commit'])\")
  echo \"experiment $exp at commit \$c\"; cat \$W/experiments/$exp/metrics.json
  mkdir -p \$W/agent/approvals && echo \$c > \$W/agent/approvals/$exp.ok
  echo approved: one final-test run for $exp"
