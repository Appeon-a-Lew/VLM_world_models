# Can Vision Language Models Learn Intuitive Physics from Interaction?

**Citation**: Luca M. Schulze Buschoff, Konstantinos Voudouris, Can Demircan, Eric Schulz. arXiv:2602.06033 (v1 2026-02-05, v2 2026-06-01). The arXiv comments call v2 the "accepted version for ICML'26". https://arxiv.org/abs/2602.06033, HTML: https://arxiv.org/html/2602.06033v2
**Read on**: 2026-10-24. I read the HTML v2 in full: method (§3), results (§4.1-4.5), discussion (§5-6) and appendices A.2-A.11. **Figures could not be read**, so per-layer probe numbers (Fig. 13) and the real-image bar values beyond those quoted in §4.3 are missing. This is the queue item "Can vision language models learn intuitive physics from interaction?", follow-up of `2025_limits_of_finetuning.md` by the same group.

## TL;DR
LoRA-adapted Qwen3-VL-8B trained with GRPO ("interaction", reward from a block-tower simulator) or with SFT reaches near-ceiling on its training task. Neither transfers reliably to sibling tasks with the same visuals and physics, nor to real photos. Linear probes show the task variables (tower stability, top-block x-offset) are "highly decodable" from visual-token activations in the base and fine-tuned models alike. The authors read this as competence without performance: "task-specific shortcuts".

## Setup / method
- **Input**: one 256×256 ThreeDWorld render of 2-4 cubes from a **fixed camera** with fixed block sizes, which the authors chose so the model "can learn the pixel-to-distance mapping" (§3.1). Two datasets: *top block* (top cube displaced left/right) and *side block* (a cube on the floor next to the tower).
- **Four tasks** (§3.2, App. A.5):
  - binary stability (yes/no);
  - x-only top: one integer in [−600, 600] that moves the top block to a stable spot;
  - x-only side;
  - x-y side: two integers that place the floor block on top.
  - The single-step prompts state the task and the action range only. **There is no given action to condition on**: the model *outputs* the action.
  - A multi-step "triplet" variant (§4.5.3) shows frame → action → frame → action → frame.
- **Supervision**:
  - GRPO: 16 completions per prompt, no KL term, shaped rewards such as 20·e^(−d²) for a stable result and penalties for unparseable or below-floor answers (§3.4, A.3).
  - SFT: cross-entropy on the optimal action (§3.3).
  - Both use Unsloth 4-bit, LoRA r = α = 16 in all layers, 10k training images and 10k steps, one image per step.
- **Models**: Qwen3-VL-8B (main), plus Qwen2.5-VL-7B, Qwen3-VL-32B (GRPO / GSPO / SFT), Llama-3.2-11B and Gemma-3-12B (§4.5, A.8).
- **Generalisation axes** (all visual or task shifts; no physical-parameter OOD such as force magnitude):
  - (i) cross-task on the same data (x-only top → binary top);
  - (ii) cross-data on the same task (x-only top → x-only side);
  - (iii) synthetic → 100 real Lerer et al. 2016 photos (§4.3, A.10);
  - (iv) single-step ↔ triplet (§4.5.3).
- **Probing** (§4.4, A.6):
  - Probes: L2 logistic regression for stability, Bayesian linear regression for the x-offset.
  - 600 images, 10-fold CV, "at all intermediate layers".
  - Because "the image tokens appear before the text tokens, the linear probes only have access to the visual information".
  - The exact token or pooling used is not stated in the text.

## Key results
- **In-distribution** (Table 1, Fig. 2), GRPO / SFT against a ceiling of 1 or 20:
  - binary 0.943 / 0.969;
  - x-only top 19.999 / 20;
  - x-only side 19.998 / 20;
  - x-y side 17.313 / 19.86.
  - Interaction gives no in-task advantage.
- **Transfer** (Table 1), GRPO with SFT in parentheses:
  - x-y side → x-only side 5.396 (19.316). This is the only clear transfer, and the authors attribute it to x-only being a sub-case of x-y. Restricted to legal answers, GRPO's 5.396 rises to 12.772 (§4.2).
  - x-only top → binary top: 0.788 (0.624) in Table 1 / A.2. The §4.2 text says 0.624 for GRPO, an **internal inconsistency in the paper**.
  - Most other off-diagonal cells sit near zero or negative reward, e.g. x-only side → x-y side: 3.738 (−5).
- **Real photos** (§4.3, Fig. 3; accuracy for models trained on binary top / x-only top / x-only side / x-y side):
  - GRPO 0.6 / 0.57 / 0.52 / 0.31;
  - SFT 0.59 / 0.53 / 0.55 / 0.57;
  - all below the human average.
  - A.10: no model generalises to real images, including Qwen2.5-VL-7B trained on synthetic binary stability itself.
- **Scale / algorithm** (§4.5, A.8): Qwen3-VL-32B with GRPO or GSPO shows "no meaningful generalization", and SFT-32B transfers somewhat from x-only top to binary. Qwen2.5-VL-7B transfers even less.
- **Ablations** (A.9.3, Fig. 22): LoRA rank 1 and 8, a reasoning prompt, and **no vision-encoder fine-tuning** all learn the training task and all fail to transfer in the same way.
- **Longer training** (A.9.2): up to 48k steps, GRPO overfits the reward after about 10k, and even the x-y → x-only transfer disappears.
- **Probing** (§4.4):
  - Stability is "trivially decodable" even in the **base** model. The authors suggest a pixel shortcut along the horizontal centre line.
  - The top-block x-offset is also highly decodable, with "small" differences between base, GRPO and SFT. No R² values are given in the text.
  - Conclusion (§6): "these models are not learning true physical intuitions, but rather task-specific shortcuts".
  - Proposed remedy (§5, untested): "auxiliary objectives that encourage models to ground their predictions in the underlying physical variables".

## Relevance to us
- **SFT and OOD.** This is the strongest version yet of the prior that exp000 tests. Their failures are all task or visual-domain shifts (cross-task, side ↔ top, synthetic → real), not changes to physical parameters. exp000 transfers to new shapes (val_ood_objects Task B 0.161 vs static 0.232) and on average to new forces. It fails on val_ood_config, a visual and geometric shift, which is the closest analogue to their side ↔ top and real-photo failures. Their fixed-camera design ("learn the pixel-to-distance mapping") is exactly our constant-K, 30-60° training camera. Their result therefore predicts that exp000 learned a viewpoint-specific pixel → metric map, consistent with Task A rising from 0.016 (val_id) to 0.136 (config).
- **Competence vs performance: the key new point for IDEA-017.** The probes say the needed variable is linearly present in the visual tokens even when the readout fails to transfer. The unread workshop paper (Muhtar, NeurIPS 2026 NeurReps; abstract only, see QUEUE) claims the same pattern for Qwen2.5-VL-3B: position decodable at R² = 0.97, velocity not used by the readout. IDEA-017 can say *which factor* (elevation / count / corner) breaks Task A on config. It cannot say whether the **visual representation** or the **language readout learned by LoRA** breaks. That distinction picks between two groups of levers:
  - representation fails: inject geometry (IDEA-013/020/016/021);
  - readout fails: diversify viewpoints in training, or add an auxiliary target that forces a geometric readout (IDEA-019/014).
  - This is IDEA-030 below.
- **IDEA-018 (ViT LoRA).** A.9.3 is direct evidence: with and without vision-encoder fine-tuning, the model learns the task and fails to transfer identically. It lowers the expected value of IDEA-018 for the OOD splits.
- **IDEA-025/026 (action dependence).** No direct evidence. Their single-step prompts contain **no stated action**, so they never test whether a model conditions on a given action. The indirect bearing is that their shortcut finding (stability readable from a thin pixel band) is the same family as the shortcut IDEA-026 tests: the push direction inferred from scene layout, since `aim_at_other_prob: 0.4` makes "toward another object" guessable. It raises the prior on a "scene, not text" outcome for 026 and on "slope < 2" for 025. Neither is evidence by itself.
- **Image-dependence gap.** The probes are image-only by construction (image tokens precede text). They show that an image can carry the variable while behaviour ignores it. The converse matters for us: exp000's large image gap (0.331) shows the image is *used*, but not that the right variable is read. That is the same argument as in `2025_limits_of_finetuning.md`.

## Ideas for us
- **IDEA-030: linear-probe diagnostic of exp000 (competence vs performance), as a ride-along to the IDEA-017 eval.**
  - Single change: an eval-only read-out, no prediction change.
  - In worktree infer.py, register a forward hook and save, for each Task A sample, the hidden state at 3 LLM layers (e.g. 1/3, 2/3, last) of the merged visual token(s) whose 28-px patch contains the projected GT centre of each object.
  - Fit closed-form ridge in torch (no sklearn, no new dependency) from token state to GT camera-frame xyz on about 1,000 `train` episodes. Report held-out R² and metric error on val_id, val_ood_config and val_diag_elev/count/corner.
  - Compare against the generated Task A error on the same objects.
  - Expected:
    - probe error stays near val_id on config/diag while generated Task A degrades (0.016 → 0.136): the representation is fine and the learned readout is viewpoint-specific, so prefer training-side levers (IDEA-019/014);
    - probe error degrades as much: the geometry is not in the tokens, so prefer input-side levers (IDEA-020/013/016).
  - Cost: about 0.5 day of code, about 0.1 GPU-h extra in the IDEA-017 job. Probe outputs go to a separate log or file, never `normal.jsonl`.
- **Extension to IDEA-018** (no new ID): the A.9.3 evidence above.

## Caveats / disagreements
- Their tasks are static stability and placement, with integer actions in unstated units. Ours are continuous 3D coordinates after a stated push. They have no perception/dynamics decomposition like our Task A / Task B.
- They use 4-bit QLoRA on 8B-32B models, with reward shaping specific to their task. We use bf16 LoRA on Qwen3-VL-2B with the ViT frozen. Their RL arm has no counterpart in our pipeline.
- The probe details (token, pooling, per-layer numbers) are only in Fig. 13, which I could not read. "Highly decodable" is a qualitative claim in the text.
- The 0.788 vs 0.624 inconsistency (Table 1 vs §4.2) means no conclusion should rest on that cell.
- Their generalisation claim is broader than our evidence. exp000 does transfer along two of our three OOD axes. Their OOD axes are task and visual-domain shifts, ours include physical-parameter shifts (forces), which they never test.
