# Corrected saved-baseline CPU analysis

Implementation: `physwm/eval/analyze_baseline.py`. A frozen minimal NumPy-only import snapshot is archived in `baseline_analysis_code.tar.gz`, with each source SHA256 recorded alongside it. No simulator, model, GPU or final-test invocation is required.

The planned cohort is all eight validation/diagnostic splits. Each expected episode must have exactly one A and one B record; incomplete coverage and duplicates fail before scoring. The primary comparisons subtract GT-static endpoint error or own-A-static endpoint error from model B endpoint error, within the same episode. The bootstrap resamples episodes within each split (10,000 draws, seed 0), then weights the three main OOD split means equally for the primary estimate. All-eight-split estimates are separately descriptive; if any main OOD split is absent, the primary estimate is explicitly unavailable. Complete-pair sensitivity uses the same main OOD grouping and equal weights. The percentile 95% interval is descriptive uncertainty, without an additive causal decomposition or multiple-testing claims.

All reported metrics concern the pushed target. A and B endpoint L2 errors follow the existing evaluator's 1 m penalty for invalid/missing coordinates. Their summary n includes penalized predictions; separate A/B missing rates preserve visibility. Axis errors and B-minus-A displacement errors require valid coordinates and therefore report observed-only counts and missing rates. Complete-pair comparison sensitivity additionally conditions on validity; it may suffer selection bias. Shape, fixed weak/in-range/strong speed, true displacement, initial centre-to-table-edge distance, and camera elevation strata are within-split descriptive summaries. Small strata receive no confidence intervals or significance claims.

Validation: six independent synthetic tests passed with `/tmp/physwm-audit-venv/bin/python -m pytest tests/test_baseline_analysis.py -q`. Tests cover paired covariance, equal split weighting with unequal cohort sizes, deterministic bootstrap, shared position offsets, missing penalties, and coverage failures.

Remote execution is pending connectivity. Two escalated SSH attempts to `nc-10-4-3` returned `No route to host` through the configured jump host. No corrected baseline measurements have been calculated by this analysis yet; no numerical claims should be inferred from the synthetic tests.

Remote command after extracting the snapshot into `/var/tmp/toyi/wm/measurements_20261029/analysis_code`:

```bash
PYTHONPATH=/var/tmp/toyi/wm/measurements_20261029/analysis_code python3 -m physwm.eval.analyze_baseline \
  --data /var/tmp/toyi/wm/audit_v2_20261027/data \
  --pred /var/tmp/toyi/wm/audit_v2_20261027/pred/normal.jsonl \
  --out /var/tmp/toyi/wm/measurements_20261029/baseline_analysis \
  --bootstrap 10000 --seed 0
```

The interpreter must provide NumPy; use the existing audit environment if system Python does not.

For newly completed training seeds, run the same frozen CPU analyzer separately for each seed's complete `pred/normal.jsonl`, with a distinct output directory such as `baseline_analysis_seed1` and `baseline_analysis_seed2`. Pass `--data` for the corrected dataset used by that seed. Analyze only completed prediction files; full A/B coverage is mandatory. Source and metadata hashes in each JSON allow checking identical definitions and datasets across runs. Single-seed episode-bootstrap intervals quantify evaluation-cohort uncertainty; they do not quantify variation from training seeds. Do not pool episode rows across seeds as independent observations. Deployment of this standalone analysis snapshot does not change the frozen SFT/inference source used for training or prediction.
