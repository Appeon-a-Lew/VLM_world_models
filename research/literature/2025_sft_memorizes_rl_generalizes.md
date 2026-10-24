# SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

## Citation
- Chu, Zhai, Yang, Tong, Xie, Schuurmans, Le, Levine, Ma (as I recall the author list; the fetched page summary did not list authors, so verify). arXiv:2501.17161v2, 2025 (ICML 2025 to my recollection, unverified). https://arxiv.org/abs/2501.17161 (read via https://arxiv.org/html/2501.17161).
- **Read on**: 2026-10-24. Read the first ~100k of ~120k characters (method, main results, ablations, limitations). The last ~20k characters (likely appendix) were not read.

## TL;DR
On a card-arithmetic game and a navigation task, in text and vision variants, RL with an outcome reward improves out-of-distribution performance while SFT on the same data degrades it. SFT is still needed first to get a parseable output format.

## Setup / method
- Backbone: Llama-3.2-Vision-11B (not Qwen, not 2B). SFT, then PPO RL with outcome rewards, multi-turn with a verifier giving feedback on each attempt.
- Tasks: GeneralPoints (build 24 from four cards; text GP-L and image GP-VL) and V-IRL (real-world navigation; V-IRL-L, V-IRL-VL).
- OOD tests: rule variants (face cards = 10 in training, 11-13 at test; absolute vs relative directions) and visual variants (black vs red suits; New York routes vs worldwide).
- Supervision for SFT: expert demonstrations; for RL: verifiable outcome reward.

## Key results
- Rule OOD, RL vs SFT: GP-L 11.5 to 15.0 vs 11.5 to 3.4; V-IRL-L 80.8 to 91.8 vs 80.8 to 1.3; GP-VL 11.2 to 14.2 vs 11.2 to 5.6; V-IRL-VL 35.7 to 45.0 vs 35.7 to 2.5 (percent).
- Visual OOD: GP-VL 23.6 to 41.2 (RL) vs 23.6 to 13.7 (SFT); V-IRL-VL 16.7 to 77.8 vs 16.7 to 11.1. Reported V-IRL mini-benchmark SOTA 44.0 to 77.8.
- Scaling RL raised visual recognition accuracy; scaling SFT lowered it. The authors hypothesise SFT overfits to reasoning tokens at the expense of recognition tokens.
- End-to-end RL from the base model failed (unstructured long outputs, no reward signal), so SFT is needed as a format stabiliser.
- More verification iterations help OOD: +2.2 (3 steps), +3.0 (5), +6.0 (10) points over one-step.
- Limits: SFT did not reach RL's in-distribution level on GP-VL. RL cannot rescue an overfitted SFT checkpoint (it collapses to the training rule); it also struggles from an underfit one.

## Relevance to us
- It is the counter-claim to our other RL reference (2602.06033, `2026_vlm_physics_interaction.md`: RL from interaction does not generalise on physics) and to `2025_limits_of_finetuning.md`. Together: SFT-fails-OOD is common, RL-fixes-it is task-dependent. The tasks here are discrete symbolic answers with an exact verifier; ours is continuous metric regression where the "reward" would be distance to GT, which behaves more like a loss than a pass/fail rule.
- Direct bearing on the exp000 failure mode: Task A perception holds on objects/forces (0.024 / 0.016) but degrades on val_ood_config (0.136). The "SFT degrades recognition" finding predicts the Task A drop on OOD could grow with SFT length. exp000 overfit the 100-episode smoke test (loss 0.6 to 0.02), so over-training is plausible. This is testable cheaply with checkpoints (IDEA-038).
- Their visual OOD (suit colour, city) is an appearance shift. Our config shift is geometric (camera elevation, object count, layout). Their positive visual-OOD result for RL does not tell us about viewpoint shift. I would not expect an RL step to fix a geometric blind spot that a lever like IDEA-014/016 targets.
- Their warning that RL starting from an overfit SFT checkpoint collapses bears on any RL idea after exp000: start from an early or lightly-trained checkpoint, not the final one.
- Image-dependence gap: not studied here. Recognition degradation is the closest analogue.

## Ideas for us
- IDEA-038: save intermediate LoRA checkpoints in the exp000 re-train (exp005 repro) and evaluate them on val_ood splits for Task A and B (SFT-length sweep). Near-zero cost on top of the already required repro.
- IDEA-039: a small GRPO/PPO stage on Task B with a verifiable distance reward from a lightly-trained SFT checkpoint. High cost, low priority; only after IDEA-038 shows OOD degrades with SFT length, and the 2602.06033 result suggests low odds.

## Caveats / disagreements
- 11B Llama, PPO, symbolic tasks with exact verifiers; ours is 2B Qwen3-VL, LoRA, continuous outputs.
- The "RL" gains are small in absolute terms on several GeneralPoints rows (e.g. 11.2 to 14.2), and the baselines are low.
- OOD axes are rule and appearance changes, not geometry. SFT collapsing to 1-3 percent on V-IRL rule shift is partly format memorisation of the training rule, which does not map onto our regression targets.
- Author list and venue above are from memory, not from the fetched page; verify before citing.
- Appendix not read.
