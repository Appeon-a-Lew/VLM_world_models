# Needs human: 2026-10-24

The planner re-ranked the backlog (see the Ranking section of `research/ideas/BACKLOG.md`). Three items need a human decision. None of them blocks the next GPU job.

1. **IDEA-009: per-split confidence intervals.** A proper fix means editing the pinned file `physwm/agent/exec.py` (`compare` only bootstraps the aggregate). Until then, the agent will print an unpinned per-split bootstrap to the diagnostic job's log. **Decision:** should main's `compare` get per-split CIs?
2. **IDEA-017: single-factor validation splits** (elevation only, object count only, cluster only), to find the cause of the val_ood_config failure. **Blocker:** the new sim configs have to land in main's `configs/`, and pinned scoring covers only `val_*` splits, so `dev_*` results would not get official metrics. Seed ranges are *not* the blocker: `dev_*` already accepts seeds ≥ 9,000,000. The planner expects IDEA-007's breakdown by object count and elevation to be ambiguous, because the corner cluster is constant and every config scene's elevation is outside training. **Decision:** add these configs and allow scoring on them?
3. **IDEA-022, VGGT half.** VGGT would probably need a new pip package, and the policy forbids dependency changes. The Depth-Anything-V2 half may run with the current dependencies. **Decision:** allow the dependency, or drop the VGGT half?
