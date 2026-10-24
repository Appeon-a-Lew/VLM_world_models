# Intuitive Physics Understanding Emerges from Self-Supervised Pretraining on Natural Videos

## Citation
- Garrido, Assran, Ballas, Bardes, Bojanowski, LeCun, et al. (Meta FAIR). arXiv:2502.11831, 2025 (preprint). https://arxiv.org/abs/2502.11831 (read via https://arxiv.org/html/2502.11831).
- **Read on**: 2026-10-24. Read the method, benchmark description, results, ablations and limitations via the HTML page as summarised by WebFetch. I did not check individual tables or figures, so numbers below are the headline ones only.

## TL;DR
V-JEPA, a video model that predicts masked regions in a learned latent space (not pixels), shows above-chance "surprise" on physically impossible videos zero-shot. Pixel-prediction video models and multimodal LLMs (Qwen2-VL, Gemini 1.5 Pro) are only marginally above untrained networks.

## Setup / method
- Input: short video clips (~3-4 s). Output: none in the usual sense. The model encodes context frames, predicts future-frame representations, and the per-step prediction error is the "surprise".
- Supervision: self-supervised masked latent prediction on natural video (VideoMix2M, K710, SSv2, HowTo100M variants). No task training and no benchmark data seen.
- Evaluation is violation-of-expectation (VoE): pairwise accuracy of "impossible" vs "possible" video by higher surprise. Benchmarks: IntPhys (dev), GRASP (10 properties), InfLevel-lab (realistic).

## Key results
- Pairwise accuracy of V-JEPA: 98% IntPhys, 66% GRASP, 62% InfLevel-lab. It is the only method significantly above untrained networks on all three.
- VideoMAEv2 (pixel prediction), Qwen2-VL and Gemini 1.5 Pro: only marginally above randomly initialised networks.
- Ablations: random masking costs only ~5 points on IntPhys (vs ~20 on standard video tasks), so latent-space prediction matters more than the mask design. SSv2 alone gives near-chance, HowTo100M is best. Even 128 h of unique video keeps >70%. A 115M model already exceeds 85% on IntPhys.
- Human comparison: V-JEPA-H matches or exceeds humans on IntPhys properties.
- Limitations stated: near chance on object-interaction violations (solidity, collision); weak where a contextualising event must be remembered; cannot condition on actions (observer only); optimal context length chosen per property, which is test-set tuning.

## Relevance to us
- It is a representation-level reference, not a method we can transplant. Our setting is one still image plus a push action, with metric 3D output; V-JEPA needs video, and has no action input.
- Bears on the open question "is the physical information in the visual features, or lost in the LoRA readout?" (IDEA-030). It supports the position that failure of a VLM to behave physically (Qwen2-VL, Gemini here) does not mean no encoder can carry the information, but it says nothing about Qwen3-VL's ViT specifically.
- Its listed weakness (collisions and solidity near chance) matches our val_ood_config suspect: 4-5 objects in a corner cluster raise multi-body contact (IDEA-035 strata). A latent-prediction model is not expected to be better there.
- The VoE evaluation idea is an image-dependence test in a different form: instead of comparing predictions under none/blank/swap, compare the model's likelihood or error on a true vs an implausible outcome. This sidesteps the exp002 problem that `none` yields unparseable output (IDEA-037).
- Scale and data results (115M suffices, data diversity beats volume, SSv2 alone fails) are weak evidence that data diversity, not model size, may drive generalisation. It does not translate directly to our 2B LoRA setting.

## Ideas for us
- IDEA-037 Teacher-forced likelihood margin (VoE-style image-dependence test), see BACKLOG. Eval-only, uses the existing exp000 adapter once one exists again (after the exp005 repro).
- I am not proposing a V-JEPA encoder swap. It would be a large change (video-pretrained encoder, no action input, retraining the projector), and the paper gives no evidence that it helps metric single-image localisation. IDEA-030 (probe on Qwen ViT) must come first; only if it shows the geometry is absent from the Qwen ViT would a frozen alternative encoder be worth a note.

## Caveats / disagreements
- Binary VoE plausibility on synthetic and lab clips, not metric regression. 98% on IntPhys is on one of the easier benchmarks; GRASP 66% and InfLevel 62% are modest.
- The VLM baselines are prompted video QA (Qwen2-VL, Gemini 1.5), not fine-tuned. Our exp000 is fine-tuned and clearly image-dependent (C-001), so the "VLMs fail" finding does not transfer to a fine-tuned setting.
- The VoE metric is an observer metric. Our task is action-conditioned prediction, where the paper says V-JEPA cannot be applied as is.
- Context length was tuned per property on the evaluation sets, so the headline numbers are somewhat optimistic.
