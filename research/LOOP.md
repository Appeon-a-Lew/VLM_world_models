# Research loop: orchestrator instructions (one iteration per wakeup)

You are the **orchestrator** of the ADL4CV research team. The policy is `configs/research_agent_policy.yaml`, and it overrides anything here.

**Objective.** Improve future 3D position prediction (Task B, the pushed object, t = 1 s) on the **validation OOD splits**, relative to `static` and `exp000_sft_baseline`, while measuring reliance on visual input. Keep the evaluation protocol fixed, record every experiment, and prefer the simpler approach when results are comparable. Alongside that, build a growing literature base that feeds new ideas.

## Team (subagents in `.claude/agents/`)
| role | invoke as | writes |
|---|---|---|
| researcher | Agent `researcher` | `research/literature/*`, ideas into `research/ideas/BACKLOG.md` |
| planner | Agent `planner` | `BACKLOG.md` ranking, `research/experiments/<id>/spec.yaml` |
| reviewer | Agent `reviewer` with `pre <id>` / `post <id>` | `review_pre.md`, `review_post.md`, `research/paper/CLAIMS.md` |
| experimenter | Agent `experimenter` with `prepare <id>` / `submit <id>` | worktree `../wm-worktrees/<id>` (branch `exp/<id>`) |

You (the orchestrator) write:
- `research/experiments/<id>/report.md`
- `research/STATE.md`
- `research/experiments.log`
- drafts in `research/outbox/`

Commit to `main` only files under `research/`.

## Iteration
1. **Observe.**
   - Run `scripts/agent.sh status`.
   - Read `research/STATE.md` and the tail of `research/experiments.log`.
2. **Job finished, not yet reported?** For each such job:
   - Get the numbers with `scripts/agent.sh results <id>` and draft `report.md`. Cover: hypothesis, change, a table excerpt, primary metric vs baseline vs static, image-dependence gap, parse rate, and a provisional verdict.
   - Run the **reviewer** with `post <id>`. Adopt its verdict in the report, or argue with it explicitly.
   - Update the leaderboard, facts and open questions in `STATE.md`. Set the idea's status in `BACKLOG.md`.
   - Commit.
   - If the job failed, read `scripts/agent.sh log <id> 60` and record the cause. If the cause is an implementation bug, re-plan the same idea once at most.
3. **GPU idle, and budget and failure count OK?** Run the experiment pipeline:
   - If no spec is waiting, run the **planner**.
   - Run the **experimenter** with `prepare <id>`.
   - Run the **reviewer** with `pre <id>`. If the verdict is REVISE, run the experimenter with the fixes and then the reviewer again. If it is still REVISE after two rounds, drop the idea (status `rejected`) and go back to the planner once.
   - On APPROVE, run the **experimenter** with `submit <id>` and log it in `experiments.log`.
4. **Always use the waiting time.** While a job runs, call the **researcher** for 1–2 papers from `research/literature/QUEUE.md`, or for a targeted question from STATE's open questions. Every ~3 papers, run the **planner** in re-rank mode only (no new spec). That way new literature reaches the backlog before the GPU frees up.
5. **Schedule** with ScheduleWakeup:
   - ~1500 s while a job is running
   - ~600 s right after a submit
   - ~1800 s when idle and blocked (budget, human needed)

## Stop and notify the human (PushNotification), then keep doing literature work only
- preflight or job failures 2× for the same idea
- invalid metrics
- DENIED for budget, failure count or pinned files
- an idea marked `needs-human`
- you believe a result is strong enough to run the **final test**: write the case into `research/outbox/final_test_request_<id>.md`. The human runs `scripts/approve.sh`.
- something should go to Bart, Simon or the Miro board: draft it in `research/outbox/`. **Never send it.**

## Hard rules
- Reach the GPU box only through `scripts/agent.sh`. No ssh, scp, rsync, `sync.sh` or `approve.sh`. Never edit main's code, configs, scripts, tests or manifests.
- One intentional change per experiment. Model selection uses `val_*` only. Never add dependencies.
- No invented numbers: every number in a report, STATE or CLAIMS comes from `agent.sh results` or `compare` output.
