# Human request: exp001 per-split missing_rate lines are not visible through agent.sh log

**Needed for:** the exp001 validity gate (object-missing rate ≤ 5% on every split). The pre-review made quoting these lines a report requirement.

**Observation:** exp001's `infer.py` prints `[<split>] answers=… parse_rate=… missing_rate=… truncation_rate=…` with `flush=True` after each split. `scripts/agent.sh log exp001_zeroshot_qwen3d 400` shows blank lines in their place.

**Likely cause (unverified):** `op_log` in `physwm/agent/exec.py:356` drops every line containing `it/s]` / `s/it]`. tqdm writes its final frame to stderr without a newline before the stdout print, so each stats line probably sits on the same physical line as a tqdm frame and gets filtered.

**Asks (human only, since pinned code and remote files are off-limits to the agent):**
1. On the GPU box, run `grep -o '\[val_[a-z_]*\] answers=.*' /var/tmp/toyi/wm/experiments/exp001_zeroshot_qwen3d/job.log`. Paste the 6 lines (normal × 4 splits, none, swap) into `research/experiments/exp001_zeroshot_qwen3d/joblog_rates.txt`.
2. Optional tooling fix: have `op_log` filter on the *segment* after the last `\r` instead of the whole line, or surface missing_rate in results.md (already a backlog item from the exp000 review).
