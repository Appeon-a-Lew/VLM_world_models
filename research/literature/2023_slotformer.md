# SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models

## Citation
Wu, Dvornik, Greff, Kipf, Garg. ICLR 2023 (per arXiv page). arXiv:2210.05861 (v2, Jan 2023). https://arxiv.org/abs/2210.05861 (HTML: https://arxiv.org/html/2210.05861).
**Read on**: 2026-10-25. Read the first ~100k of 118k characters of the HTML (method, experiments, ablations). The final ~18k (appendix, incl. Limitations in App. F) was not read.

## TL;DR
SlotFormer is a Transformer that autoregressively predicts the next-step object slots of a pretrained slot encoder (SAVi/STEVE). The rollouts help downstream predictive VQA, but only slightly on Physion. It is a non-VLM, object-centric dynamics reference with no explicit action input and no 3D handling.

## Setup / method
- Input: video frames. A pretrained slot model (SAVi on OBJ3D/CLEVRER/PHYRE, STEVE on Physion) converts each frame to object slots. Output: future slots, decoded to frames by the frozen decoder.
- Dynamics: Transformer over linearly projected slots, residual per slot, temporal positional encoding only (slots permutation-equivariant). Two-stage training: slot model on reconstruction, then SlotFormer on extracted slots.
- Loss: slot L2 plus image reconstruction through the frozen decoder (lambda 1 on OBJ3D/CLEVRER, 0 on Physion/PHYRE). Trained on its own predictions (no teacher forcing).
- Supervision is fully unsupervised (no object labels). Downstream: readout models for VQA (Aloe), Physion contact prediction, PHYRE planning.
- Actions: the dynamics model takes only slot sequences. For PHYRE the action is rendered into the image as a red ball and a success classifier scores simulated futures.

## Key results
- Video prediction CLEVRER: PSNR/SSIM/LPIPS 30.21 / 0.89 / 0.11, best LPIPS; PredRNN has higher PSNR (31.34). OBJ3D: 32.40 / 0.91 / 0.08.
- CLEVRER object dynamics: AR 53.14, FG-ARI 63.00 vs G-SWM 43.98 / 49.61.
- CLEVRER predictive VQA (Aloe): 96.5% per option / 93.3% per question with SlotFormer rollouts, vs 93.1 / 87.3 re-implemented Aloe and 94.5 / 89.2 VRDP.
- Physion contact prediction: 67.1% with rollouts vs 65.2% observation-only (+1.9 points); human 74.7%.
- PHYRE within-template AUCCESS 82.0 vs 80.7 for SAVi without rollouts; Dyn-DQN (ground-truth masks) 86.2.
- Ablations: teacher forcing is the most harmful change on OBJ3D (LPIPS 0.106 vs 0.080); on Physion it removes nearly all rollout gain; 12 layers diverge.

## Relevance to us
- Task B vs Task A: it is the non-VLM version of "perceive objects, then roll dynamics forward". The Physion result (+1.9 points from rollouts over a good observation-only readout) is consistent with our finding that own-perception static is a hard baseline and the dynamics gain is small and shrinks on val_ood_config (0.113 to 0.028).
- OOD: all of its evaluation is within-distribution (within-template PHYRE, in-distribution CLEVRER/Physion). No viewpoint or force-shift test, so it says nothing about val_ood_config. Physion randomises camera pose, but the paper does not analyse it (as read).
- Image-dependence: slot models are image-grounded by construction (reconstruction loss), unlike an LLM that can emit text priors; it is the architecture that cannot cheat the swap test.
- Teacher-forcing finding is loosely relevant: our SFT trains on GT-prefixed answers; for Task B conditioned on its own Task A output (IDEA-002/033) a train-test gap of this kind could appear.

## Ideas for us
- IDEA-049: state-only dynamics ceiling (see BACKLOG).

## Caveats / disagreements
- No force/impulse conditioning, no metric 3D output, slots are not mapped to metres; contact-yes/no and pixels, not centre error.
- Very different scale (trained from scratch on thousands of videos) and multi-frame input; we get one image.
- Unread appendix may contain limitations that qualify the above.
