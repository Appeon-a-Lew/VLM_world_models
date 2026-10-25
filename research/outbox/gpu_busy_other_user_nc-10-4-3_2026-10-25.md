# GPU on nc-10-4-3 held by another user (exp009 waiting)

**Observed:** `scripts/agent.sh status` shows 6.6 to 7.4 GiB in use with nothing of ours running, from about 14:58 to the latest tick. The exp009_exp005_cf_action submit was denied once (`DENIED: GPU busy (6634 MiB in use by someone else)`).

**State:** exp009 is approved (review_pre round 2), worktree merged (ffe8ee2), spec lints OK. The loop resubmits unchanged when the GPU reads free; no retry in another form.

**Needed from you (optional):** nothing if the other job ends soon; otherwise move the executor again or tell me to wait.
