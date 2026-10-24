# needs-human: torchvision missing on nc-10-1-1 (blocks all train and VLM jobs)

**Job:** exp005_exp000_repro (retrain of exp000, unchanged). Status `failed` in ~0 GPU-h. consecutive_failures = 1.

**Cause (from `scripts/agent.sh log exp005_exp000_repro 60`):** `physwm.vlm.sft` dies at `AutoProcessor.from_pretrained(...)`:

    ImportError: Qwen3VLVideoProcessor requires the Torchvision library but it was not found in your environment.

Preflight (13 tests) passes because it does not load the processor. This is an environment gap on the new box, not a code or spec bug. The old box presumably had torchvision.

**Needed from human:** install a torchvision build matching torch 2.11 into `/var/tmp/toyi/wm/venv` on nc-10-1-1 (the loop must not add dependencies and cannot reach the box). Then confirm that the processor loads (also needed for eval_only jobs such as exp004).

**After the fix:** resubmit the same spec under the same id if allowed, or as `exp005b_exp000_repro` (no spec change). The failure is environmental, so it should not count against the idea.
