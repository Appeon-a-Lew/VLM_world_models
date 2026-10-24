# Pre-review exp005_exp000_repro

**Verdict: APPROVE**

Checks
- One change: the diff is only spec.yaml. train_config, style, seed 0, ablations and budget_hours 3 match exp000's spec. Nothing rides along.
- Baseline and budget: the comparison is against exp000's recorded numbers (0.3136, 0.161/0.510/0.269, val_id 0.155, Task A 0.016, gap 0.331). These match exp000's report.md. The budget is realistic (exp000 took about 1.4 GPU-h).
- Leakage and gaming: no code, data-config or prompt change, so none is introduced. The regenerated data has +-1/255 RGB noise. It is disclosed, and it is confounded with the box change.
- Falsifiable: the thresholds are fixed before the run, and the failure branch is specified (drift is reported and exp005 becomes the reference).

Notes (not blocking)
- Single seed, so a pass or fail cannot separate box effects from seed noise. State this in the report. A drift is a noise-floor estimate, not proof that the box is at fault.
- The 0.010 band on the mean is tight against the val_ood_config noise. exp000's CI vs static is about 0.02 wide, and config is the least stable split. A per-split miss on config alone should be flagged as marginal, not as failed reproduction.
- `compare exp005 exp000` is unavailable (exp000 predictions are gone). Report numbers without a paired CI, and do not claim significance on the exp005 vs exp000 difference.
- The adapter path must be recorded from job.log, as the spec says.
