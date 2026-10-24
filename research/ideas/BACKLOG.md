# Idea backlog

Status: `proposed` → `planned: exp###` → `done: exp###` (or `rejected: reason`, `needs-human: reason`).
The planner re-ranks this file; the researcher and reviewer append to it.

## IDEA-001 Zero-shot with the native 3D-grounding prompt
- source: Phase 1 smoke test (the `center` prompt yields `[0,0,0]`)
- change: no training; `style: qwen3d`
- status: planned: exp001_zeroshot_qwen3d

## IDEA-002 Predict the displacement instead of the absolute future position
- source: Phase 1 design. Absolute targets mix perception error and dynamics error.
- change: Task B answers both the current centre and `delta` = p(t=1) − p(t=0); the scored position is the model's own current centre + delta (never the GT t=0 position, which would leak)
- status: proposed

## IDEA-003 Task mix ablation: Task B only vs A + B
- source: Phase 1 design question. Does grounding supervision help dynamics?
- change: `tasks: [B]` in the SFT config
- status: proposed

## IDEA-004 Neutral object references (set-of-marks IDs instead of colour + shape names)
- source: Slack discussion (semantic vs geometric vs neutral references)
- change: draw numbered marks on the image; refer to objects as "object 1, 2, ..."
- status: proposed

## IDEA-005 Add metric depth as a second image (pseudo-colour)
- source: Bart, grounding via unprojection
- change: give the depth map as an extra image input
- status: proposed
