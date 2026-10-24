# Testing the Limits of Fine-Tuning for Improving Visual Cognition in Vision Language Models

**Citation**: Luca M. Schulze Buschoff, Konstantinos Voudouris, Zeynep Akata, Matthias Bethge, Joshua B. Tenenbaum, Eric Schulz. ICML 2025 (PMLR vol. 267). arXiv:2502.15678. https://arxiv.org/abs/2502.15678, HTML: https://arxiv.org/html/2502.15678v2, PMLR: https://proceedings.mlr.press/v267/schulze-buschoff25a.html
**Read on**: 2026-10-24. I read the HTML v2 (method, results, discussion). This is the paper behind the queue item "Limits of Fine-Tuning". I identified it by title search, and it is also the "Schulze Buschoff et al." in our Background list.

## TL;DR
QLoRA fine-tuning of VLMs (Qwen2-VL-7B, Llama-3.2-Vision 11B/90B) on synthetic block-tower stability or causal "remove the red block" judgments lifts in-domain accuracy from chance to roughly 0.6–0.92. It does not transfer to real photos of block towers (Lerer et al. 2016) or across the physics↔causal domains. The authors conclude task-specific SFT yields "task-specific superficial shortcuts that do not generalize".

## Setup / method
- **Input**: one image (synthetic "Cubeworld" block towers of 2–7 blocks, or pyramids of 3–15 blocks with one red block) plus a yes/no question.
- **Output**: binary answer, scored via the normalised probability of the "Yes" and "No" tokens.
- **Supervision**: ground-truth simulation labels (10,000 image-text pairs, 10 epochs). One variant uses human labels instead.
- **Fine-tuning**: 4-bit QLoRA (unsloth), r = 16, α = 16, LR 2e-4, adapters in **both the ViT vision encoder and the LLM** attention/MLP layers, 3 seeds.
- **Evaluation axes**: (i) tower/pyramid size generalisation (train 2–4 blocks, test 5–7 and vice versa); (ii) visual generalisation to 100 real Lerer et al. tower photos; (iii) cross-domain (physics ↔ causal), including paired images where the physics image is the causal image with the red block removed.

## Key results
- Zero-shot base models are at about chance. Humans score 0.63 on towers and 0.72 on causal pyramids.
- **In-domain after SFT**: 11B models reach 0.6–0.92 on training block sizes, and 2–4-block tower models reach 0.78–0.83 across 7B/11B/90B. This exceeds human accuracy.
- **Size generalisation is partial**: 2–4-block models degrade on larger towers but stay above base. 5–7-block models are flat across sizes.
- **Visual-domain shift (Lerer real photos)**: the 11B tower model beats chance only on 4-block photos, and some models fall below chance.
- **Cross-domain**: no transfer in either direction, even on the paired images, which the authors read as shortcut learning. Joint training (5k + 5k) fits both synthetic tasks but still fails on some Lerer sizes (e.g. 2-block towers).
- **Human alignment (Cohen's κ, 11B)**: physics-GT SFT 0.23 on the same task, 0.15 on Lerer. Training on human labels gives 0.37 on the same task and 0.37 on Lerer, so human-label SFT transferred alignment to the real photos better.
- The proposed explanations (binding problem, shortcuts, SFT memorisation vs RL, scale/diversity) are **not tested**.

## Relevance to us
- **This is the prior that exp000 tests.** Our result partly agrees and partly disagrees. SFT generalises to new objects (val_ood_objects Task B 0.161 vs static 0.232) and new forces (mean). It fails on val_ood_config, which, like their Lerer test, is a shift in **visual characteristics / viewpoint** while the physics stays the same. Their "fails under visual shift" finding is the closest analogue to our config failure. Our Task A numbers (0.016 → 0.136 m) localise the failure in perception, which their binary-output setup cannot do. That is an argument for keeping grounded coordinates and Task A.
- **Size generalisation ↔ our object count.** val_ood_config also raises object count from 1–3 to 4–5, much like their train-2–4 → test-5–7 tower split, where they saw degradation. Our config failure therefore confounds viewpoint with count and clutter. We should separate the two before attributing it to the camera.
- **Adapter placement**: they adapt the ViT, and our `sft_qwen3vl2b.yaml` freezes it ("language model only"). Since both setups fail under visual shift, ViT adaptation is not a fix by itself. Whether it helps or hurts our config split is an open, cheap question.
- **Image-dependence test**: their paired-image result (no transfer even when the only difference is one block) is the shortcut pattern our none/blank/swap ablations guard against. exp000's image gap of 0.331 shows we are not in the pure text-prior regime, but a large gap does not rule out a viewpoint-specific shortcut (an image→coordinate lookup tied to the training elevation range).

## Ideas for us
- **IDEA-017: factorised val_ood_config diagnostics (eval only).** Generate three small validation-only splits that each change one factor of `sim_ood_config.yaml`: (a) elevation 12–25° only, (b) 4–5 objects only, (c) the corner cluster only (`region`, `region_center`, `min_gap`). Evaluate exp000 Task A/B on each. Expected: attributes the 0.136 m Task A error to the camera vs count/clutter, which decides whether IDEA-013/014/016 (geometry) or a binding/clutter fix is the right lever. Cost: about 0.2 GPU h eval plus sim generation. **needs-human** for new validation seed ranges, because `research_agent_policy.yaml` reserves seed ranges.
- **IDEA-018: LoRA on the vision encoder as well.** Extend `target_modules` to the ViT blocks (as in this paper). Expected: possibly lower ID Task A error. Effect on val_ood_config is uncertain, and it could overfit the training viewpoint distribution further, which the paper's Lerer failure hints at. Cost: config change plus 1 SFT run (about 1.5 GPU h).
- **IDEA-019: random crop/zoom augmentation with updated intrinsics.** At training time, randomly crop and resize the 448² image (scale 0.6–1.0, random offset) and rewrite fx, fy, cx, cy in the prompt to match. The camera-frame targets are unchanged by a crop. Currently K is constant, so the model can ignore it and memorise a pixel→metric map. Varying K forces the model to use projective geometry, which should transfer better to new viewpoints. Expected: a small ID cost and lower Task A error on val_ood_config. It also tests whether the model can use K at all (evaluate exp000 vs this run on centre crops with changed K). Cost: data-loader change plus 1 SFT run (about 1.4 GPU h).

## Caveats / disagreements
- Their outputs are binary judgments on static stability. Ours are continuous 3D coordinates after a push. Our metric can separate perception from dynamics, which theirs cannot.
- Their visual shift is synthetic → real photos (texture, lighting, realism). Ours is a camera/configuration shift inside the same renderer. Both are "visual characteristic" shifts, but ours is geometric.
- They use 4-bit QLoRA r = 16 on 7B–90B models. We use bf16 LoRA r = 16 on a 2B model, LLM only.
- The Lerer test set is small (100 photos), so per-size results are noisy.
- Their "SFT does not generalise" claim is broader than our evidence supports: exp000 generalises on 2 of 3 OOD axes.
