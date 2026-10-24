# Human request: two-factor diagnostic split (IDEA-036)

**Why:** exp003 showed that no single factor (elevation, count, corner) reproduces exp000's val_ood_config failure. Every single-factor split beats static, while config is worse: +0.028 [0.013, 0.042]. The shares are R_elev 0.417, R_count 0.273 and R_corner 0.396. Their sum is 1.086, but its conservative interval [0.38, 1.83] can't tell "additive" from "interaction".

**Ask:** a `configs/sim_diag_elev_corner.yaml` equal to sim_train.yaml with **both** low elevation (12–25°) and the corner cluster changed, and object count kept at 1–3. Generate it as `val_diag_elev_corner` (300 episodes, like the other diag splits) and add it to policy `data.diagnostic_splits`.

The agent can't create data configs or generate splits on main. With the split in place, an eval_only spec on the exp000 adapter (~0.1 GPU-h) gives R_{elev+corner}. If that is close to 1, the failure is a pairwise interaction. If it is close to R_elev + R_corner (0.81), the contributions are additive.

Priority: medium. The cheap discriminators IDEA-034/035 can run first without it.
