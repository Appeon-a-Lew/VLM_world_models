# SpaceMind: Camera-Guided Modality Fusion for Spatial Reasoning in Vision-Language Models

**Citation**: Ruosen Zhao, Zhikang Zhang, Jialei Xu, Jiahao Chang, Dong Chen, Lingyun Li, Weijian Sun, Zizhuang Wei. arXiv preprint, Nov 2025 (v2 Dec 2025), no venue listed. arXiv:2511.23075. https://arxiv.org/abs/2511.23075 (HTML read: https://arxiv.org/html/2511.23075v2)
**Read on**: 2026-10-24 (abstract, method, experiments and ablation table via HTML. The training loss is not stated on the page.)

## TL;DR
SpaceMind runs a frozen VGGT geometry encoder next to InternVL3-8B's ViT. A small fusion module uses VGGT's per-frame camera token as a control signal: it biases the geometry keys/values and gates the fused update to the visual tokens, so the camera is not just one more concatenated feature. Trained on video spatial-QA, it reaches 69.6 on VSI-Bench (prior best VLM-3R 60.9). Each camera-specific component adds a modest +0.9 to +1.6 points.

## Setup / method
- **Input**: RGB only (32 frames per sample in training; single images on SPBench-SI at test) plus a text question. **No GT intrinsics or poses**: the camera token is predicted by VGGT from the pixels.
- **Encoders**: InternViT-300M (visual tokens f_v) and the VGGT aggregator (spatial tokens f_s plus one camera token f_c per frame; registers discarded). Both are frozen.
- **Camera-Guided Modality Fusion (CGMF)**: cross-attention with queries from the visual tokens and keys/values from the VGGT tokens. (1) Camera-conditioned bias: B_g = MLP([f_s, f_c]) is added to K and V. (2) A query-independent token weight σ(MLP(f_s)) multiplies V. (3) The camera token is prepended to the memory: K' = [C; K]. (4) A camera-conditioned SwiGLU gate g = Swish(W1·C̄) ⊙ W2·C̄ scales the projected attention output, and the result is added residually: f_fused = P_L(f_proj) ⊙ g + f_v. The token count and shape are unchanged, so the LLM interface is untouched.
- **Training**: CGMF fully trained, LoRA rank 256 on the LLM, 2 epochs on VLM-3R-data + ViCA-322K + SQA3D. About 25 h on 64 H100s.
- **Supervision**: spatial QA answers (multiple choice and numeric). The loss is not specified, presumably next-token CE.

## Key results
- VSI-Bench average 69.6 vs VLM-3R 60.9, InternVL3-78B 48.5, Spatial-MLLM 48.4, GPT-4o 34.0. Abs. Dist. 61.4 vs 49.4, Rel. Dir. 88.4 vs 80.5, Appr. Order 70.6 vs 40.1. Route Plan is slightly worse (44.3 vs 45.4).
- SQA3D EM@1 54.1 vs Video-3D LLM 51.1, with video only (competitors use depth, point clouds or meshes).
- SPBench (not in training): overall 67.3 vs ViLaSR-7B 54.0. Single-image SPBench-SI 59.7 (Gemini-2.0-Flash 54.7). But SI multiple choice is 53.2, *below* ViLaSR 63.7 and SpaceR 61.5, so the SI lead comes from numeric questions.
- **Ablation (VSI-Bench avg)**: InternVL3-8B fine-tuned without 3D 63.07 → + VGGT tokens via shallow cross-attention 66.77 (+3.70) → + token weighting 67.17 (+0.40) → + camera-conditioned bias (geoMLP) 68.73 (+1.56) → + camera SwiGLU gate 69.58 (+0.85).
- Missing ablations: no run removes only the camera token, there is no head-to-head against plain concatenation, and there is no controlled viewpoint-shift experiment.

## Relevance to us
- **val_ood_config (camera elevation 12–25° vs train 30–60°)**: the paper's central claim is that camera information should actively condition how geometry is fused into the visual tokens. That is the IDEA-016 hypothesis (camera pose in the prompt) delivered through a different channel. The evidence is weak for our question: the gains attributable to the camera are +1.56 / +0.85 on a QA benchmark, and there is no viewpoint-OOD test. Most of the gain (+3.70) comes from adding the geometry-encoder tokens at all.
- **The geometry encoder as an RGB-only route**: the bigger lever for us is the frozen feed-forward 3D model (VGGT) that supplies the per-token geometry, the RGB-only counterpart of feeding GT depth (IDEA-013/020). Whether VGGT itself stays accurate at our low OOD elevations decides whether that route can fix val_ood_config without GT depth (IDEA-022).
- **Image-dependence**: fusion is residual on f_v with a gate, so the no-image/blank/swap ablations must also remove or swap the VGGT branch. Otherwise the gap is misattributed.
- **Task B / force conditioning**: not addressed. Static spatial QA only.

## Ideas for us
- **IDEA-021 Camera-pose FiLM/SwiGLU gate on visual tokens (non-text alternative to IDEA-016)**: encode the known `cam2world` as a small vector (gravity direction in the camera frame plus camera height above the table). Map it with an MLP to a SwiGLU gate and bias applied residually to the merged visual tokens (zero-init), à la CGMF step 4. Train the MLP plus the existing LoRA. Expected: if IDEA-016 (text) gives a partial Task A gain on val_ood_config, this tests whether a continuous embedding conditions better than digits in text. The risk is the same: 12–25° is extrapolation for the MLP too. Run only after IDEA-016. Cost: about 1 day of surgery plus 1.5 GPU h.
- **IDEA-022 Frozen geometry-model diagnostic across camera elevation (eval only)**: run a pretrained single-image geometry model (VGGT, plus Depth-Anything-V2 as a second reference) on the t=0 images of val_id and val_ood_config. Measure the depth AbsRel against `depth_t0` and the error in object-centre xyz (from the predicted point map averaged over the GT mask) by elevation bin. Expected: if the error is roughly flat across elevation while exp000 Task A jumps 0.016 → 0.136 m, a geometry-encoder route (VGGT tokens or predicted-depth PE for IDEA-013/020) can fix config RGB-only, and the failure is in the VLM's learned pixel→metric map. If VGGT also degrades at 12–25°, low elevation is intrinsically harder (foreshortening, occlusion in 4–5 object clusters), and some of the config gap is not attributable to SFT. Cost: about 0.5 day, under 0.2 GPU h, no training.

## Caveats / disagreements
- Video and multi-view training (32 frames). The camera token mainly encodes *relative* pose between frames. With one image per sample, VGGT's camera token for a single frame is essentially canonical (first-frame convention), so CGMF's camera signal may carry little information in our setting. The benefit for us would come from the geometry tokens, not the camera gating.
- The backbone is InternVL3-8B, not Qwen3-VL-2B. Compute (64 H100 × 25 h) is far beyond ours.
- Benchmarks are QA accuracy (MCA/MRA), not metric 3D coordinate regression.
- There is no ablation isolating the camera token, so the attribution "camera guidance helps viewpoint-sensitive tasks" is the authors' qualitative interpretation.
- Preprint with no venue. Code release is promised, not verified.
