# GPU on nc-10-1-1 occupied by another user (blocks exp004 submit)

- Submit of exp004_exp000_oracle_anchor (pre-review APPROVE, round 2) was denied by preflight: `DENIED: GPU busy (11120 MiB in use by someone else)`.
- `agent.sh status` has since shown 11120 → 12188 MiB used with none of our jobs running, over about 1 h of wakeups.
- exp004 needs a single eval_only pass (budget 0.75 h, about 0.2 GPU-h expected).

**Needed from human (any one):** free the GPU, or tell me whether the busy check may be bypassed or relaxed (the card is 16 GB, and the process may not fit beside the other user), or point me to another box. I will keep retrying at each wakeup and do literature work meanwhile.

## Answer (human, 2026-10-25)
Resolved by moving the executor to nc-10-4-3 (idle GPU), with all state copied byte-for-byte. Our files on nc-10-1-1 were deleted. If the new box gets busy, report it the same way.
