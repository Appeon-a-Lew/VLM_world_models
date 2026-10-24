# Research loop: instructions for each iteration

You are the ADL4CV research agent. Every iteration, follow these steps. The policy is `configs/research_agent_policy.yaml`. It overrides anything here.

**Objective.** Improve future 3D position prediction (Task B, the pushed object, t = 1 s) on the **validation OOD splits** relative to the static baseline and `exp000_sft_baseline`, while measuring reliance on visual input. Keep the evaluation protocol fixed, record every experiment, and prefer the simpler approach when results are comparable (difference < `improvement_threshold`).

## Hard rules
- Reach the GPU box **only** through `scripts/agent.sh`. Never ssh, scp or rsync directly, never run `scripts/sync.sh` or `scripts/approve.sh`, and never edit `main`'s code.
- Each experiment lives in its own git worktree on branch `exp/<id>`, at `../wm-worktrees/<id>`. Never merge into `main`.
- One intentional change per experiment, relative to its stated `baseline`.
- Never touch the metrics, the evaluation code, the policy, `manifests/`, or the final test splits (`id_test`, `ood_*`). Model selection uses `val_*` only.
- Never add dependencies: `pyproject.toml` must stay identical to main.
- Outbound actions need the human: messages to Bart/Simon, Miro edits, publishing, Artifacts, GitHub. Draft them into `research/outbox/` instead.
- Stop and notify the human (PushNotification) in these cases:
  - preflight tests fail twice for the same idea
  - metrics come back invalid
  - `agent.sh` answers DENIED for budget or failure-count reasons
  - you believe the final test should be run
  - an idea needs a protected change

## Iteration
1. **Observe.** Run `scripts/agent.sh status`, then read `research/STATE.md` and the newest `research/experiments/*/report.md`.
2. **Running job?** Check `scripts/agent.sh log <id> 20` for errors.
   - If the job is healthy, schedule the next wakeup (ScheduleWakeup, ~1200–1800 s) and end the iteration.
3. **Finished job?** Run `scripts/agent.sh results <id>` and write `research/experiments/<id>/report.md` on main. The report must cover:
   - the hypothesis
   - the change
   - the results table excerpt
   - the primary metric vs the baseline and vs static
   - the image-dependence gap
   - the parse rate
   - a verdict: supported, refuted, or inconclusive
   Then update `research/STATE.md`: the leaderboard, what was learned, and the open questions. Commit on main, touching only `research/`.
4. **Hypothesize.** Pick ONE next experiment and justify it from the evidence. Good directions:
   - prompt / answer format
   - which tasks to mix (A vs B)
   - data amount or diversity (new `train_*` splits via `agent.sh gen`, with configs added in a worktree; note that `gen` reads configs from main, so ask the human to sync a new config)
   - LoRA rank and target modules
   - learning rate and epochs
   - adding depth or a second frame as input
   - auxiliary targets (displacement instead of absolute position)
   - multi-view input
   Cite papers in the report if you used any.
5. **Implement.**
   - `git worktree add ../wm-worktrees/<id> -b exp/<id> main`
   - make the change
   - add `research/experiments/<id>/spec.yaml` (template: `research/experiments/exp000_sft_baseline/spec.yaml`)
   - run the fast tests locally where possible
   - commit
6. **Execute.** From inside the worktree, run `scripts/agent.sh submit <id>`. If it is DENIED, read the reason and fix it, or wait for the budget to free up.
7. **Record.** Append a line to `research/experiments.log`: date, id, status. Then ScheduleWakeup.

Experiment ids are `exp###_short_name`, numbered sequentially.
