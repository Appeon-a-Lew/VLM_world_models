# Towards Camera-Robust 3D Localization: Equation-Anchored Tool-Use for MLLMs

**Citation**: Xueying Jiang, Wenhao Li, Quanhao Qian, Deli Zhao, Shijian Lu, Gongjie Zhang, Ran Xu. arXiv preprint 2605.19528 (v1, 19 May 2026; no venue listed). https://arxiv.org/abs/2605.19528. Read from the HTML (https://arxiv.org/html/2605.19528): abstract, Introduction, §3.2 (back-projection), §3.4 (data), §4.1-4.2, Tables 1-5.
**Read on**: 2026-10-24.

## TL;DR
RGB-only MLLM 3D detection and grounding (Qwen3-VL-8B, VG LLM-8B) collapses when the camera intrinsics are rescaled to 0.5× or 1.5× of the canonical value. The authors call this overfitting "to a canonical training intrinsic". Their fix: give the model tools that return K and several metric depth samples, and have it write out and fill in the pinhole back-projection in its chain of thought. This raises accuracy at every scale, but the camera-shift drop stays large.

## Setup / method
- **Input**: one RGB image and a task prompt. K is not given as prompt text. It comes from a "Camera Intrinsic Tool" that returns "dataset-provided camera calibration parameters".
- **Output**: JSON 9-DoF boxes `[Xc, Yc, Zc, l, w, h, ψ, θ, ϕ]` in the camera frame, depth in metres.
- **Base model**: Qwen3-VL-8B-Instruct. Vision encoder and projector frozen. SFT only, with a masked causal-LM loss. No RL.
- **Equation (§3.2)**: X̂ = (u_c − c_x)·Z̄/f_x, Ŷ = (v_c − c_y)·Z̄/f_y, Ẑ = Z̄. (u_c, v_c) is the 2D box centre, and Z̄ is the mean of the sampled depths.
- **Depth tool**: SAM 3 turns the 2D box into an object mask. Up to N = 5 valid pixels are sampled from the mask, and depths < 0.1 m are discarded. Depth is either GT (oracle) or UniDepthV2 (monocular metric depth, conditioned on intrinsics).
- **Training data (§3.4)**: traces built from ScanNet (detection) and ScanRefer (grounding). GT boxes are projected to 2D, masks come from SAM 3, and tool outputs are simulated from GT depth. Qwen3.5 writes and verifies the referring expressions. No sample counts or hyperparameters are given.
- **Metrics**: detection Avg F1 @ IoU 0.25 (31 categories), grounding Acc @ IoU 0.25. **No metric error in metres is reported.**
- **Camera-shift protocol**: "intrinsics rescaled from 0.5× to 1.5×". **The paper does not say how the test images are made** (crop/zoom, resize, or only the K value changed), nor whether GT changes.

## Key results
- **Intrinsic overfitting (Tables 1-2, scale 0.5× / 1.0× / 1.5×)**:
  - Qwen3-VL-8B-Instruct, Det F1: 10.75 / 38.20 / 15.05. Ground Acc: 8.49 / 53.85 / 20.69.
  - VG LLM-8B, Det: 13.19 / 38.49 / 17.16. Ground: 17.51 / 48.28 / 24.67.
  - Generalist APIs are near zero at every scale on Det (e.g. GPT-5.4 2.67 / 0.63 / 0.59, Claude-Opus-4.7 0 / 0 / 0). Qwen3.6-Flash 4.03 / 25.68 / 17.88 and Seed-2.0-Pro 2.00 / 24.00 / 12.54 are the exceptions.
  - Retention at 0.5× (my arithmetic from these numbers): Qwen3-VL-8B keeps 28% of its 1.0× Det score and 16% of its Ground score.
- **The fix (Tables 1-2)**: Ours (UniDepthV2) Det 16.97 / 44.13 / 20.76 and Ground 13.26 / 55.17 / 28.12. Ours (GT depth) Det 18.93 / 44.90 / 24.29 and Ground 11.41 / 58.62 / 32.89.
  - It beats both RGB-only baselines on Det at every scale and on Ground at 1.0× and 1.5×. At 0.5× on Ground it *trails* VG LLM (13.26 and 11.41 vs 17.51).
  - Retention (my arithmetic): UniDepthV2 Det keeps 38% at 0.5× and 47% at 1.5×. **Explicit K + depth reduces the collapse but does not remove it.**
- **Tool ablation at 1.0× only (Table 3, Det Cat31 / Ground)**: no tools 38.43 / 50.13; intrinsic tool only 40.27 / 51.46; depth tool only 40.09 / 54.38; both 44.13 / 55.17.
- **Depth source at 1.0× (Table 4)**: UniDepthV2 44.13 / 55.17 vs GT 44.90 / 58.62. A small gap.
- **CoT (Table 5)**: without CoT, 38.20 / 53.85, the same as the Qwen3-VL-8B baseline numbers in Tables 1-2 (unexplained). With CoT, 44.13 / 55.17.
- **All ablations are at 1.0×**, so no ablation shows which component gives the robustness under camera shift.
- **Extrinsics / elevation / viewpoint shift: not studied.** Only K rescaling is evaluated.
- The paper does not say whether the Qwen3-VL-8B baseline is zero-shot or fine-tuned on ScanNet (it is listed under "Specialist MLLMs").

## Relevance to us
- **exp001 (zero-shot qwen3d prompt)**: the paper describes plain RGB-only Qwen3-VL as having no mechanism to use K ("bypass camera parameters entirely"). Our qwen3d prompt puts K in the text. This is further, indirect support for reading exp001 as "text K is likely ignored", in line with `2025_qwen3_vl.md` and IDEA-023. The paper does not test text K, so it does not show that text K is ignored.
- **Is intrinsic overfitting our val_ood_config problem? No.** K is constant in train and in every val split, including val_ood_config. Our shift is *extrinsic*: elevation 12–25° vs 30–60°, plus count and cluster. The paper's effect is real and large, but it is the wrong axis for our current failure. Its analogue, overfitting to a canonical *training viewpoint*, is plausible for exp000 but is not tested here. val_diag_elev / IDEA-007 / IDEA-017 are the right tools for that.
- **IDEA-014 / IDEA-015 (u, v, depth + analytic back-projection)**: the paper's equation is exactly the analytic step in IDEA-014/015. The difference is that they let the LLM do the arithmetic; we would compute it deterministically, which is strictly safer. Evidence at fixed K (our regime): giving metric depth samples alone lifts Ground 50.13 → 54.38, and depth + K lifts Det 38.43 → 44.13 (Table 3). So the analogue of IDEA-015's depth lookup is expected to help Task A even with K constant. It also explains why: the centre no longer depends on viewpoint-specific pixel→metric memorisation, provided the 2D centre is right.
- **Task B caveat**: back-projection fixes the camera-frame *centre*. Task B also needs the table plane and gravity direction in the camera frame, because pushes move objects along the table, and that geometry depends on elevation. So even a perfect IDEA-014/015 Task A at low elevation leaves the camera-frame *dynamics* tied to the viewpoint. That argues for pairing 014/015 with camera-pose information (IDEA-016/021) on val_ood_config Task B.
- **IDEA-019 (K-varying crops)**: this is the strongest evidence so far that an MLLM trained at one K memorises that K. Qwen3-VL-8B keeps only 16–28% of its score at 0.5×. Our exp000 has seen exactly one K, so it should be at least as brittle. That explains *why* IDEA-019 is needed for K-robustness claims. Since val_ood_config has fixed K, it is not evidence that 019 helps the primary metric. The modest gains under shift even *with* explicit K + depth (38% retention) suggest that crop augmentation alone, with K in text, will not make the model fully K-equivariant.

## Ideas for us
- **IDEA-027 Test-time intrinsics-shift eval (crop-zoom val_id, K rewritten in prompt)**: eval-only. Centre-crop val_id t=0 images by factor c ∈ {0.8, 0.67}, resize back to 448 (effective f scaled 1.25× / 1.5×), and rewrite fx, fy, cx, cy in the prompt. GT camera-frame targets are unchanged. Run exp000's adapter. Single change: test image + K text. Expected: a large Task A drop if exp000 memorised one pixel→metric map. The fingerprint is a predicted z that ignores the zoom, so z_pred·(f′/f) tracks z_gt. This gives the baseline any IDEA-019 run must beat, and it is a direct test of whether the model reads K. Cost: worktree infer option (crop + K rewrite), ~0.1 GPU h. Predictions go to a separate pred file (scored by pinned code like other derived files), never `normal.jsonl`. Low priority for the primary metric (K is fixed in val_ood_config); run it only alongside or before IDEA-019.
- **Extend IDEA-015** with the paper's sampling recipe: the median/mean over ≤ 5 mask pixels (we have GT masks), dropping invalid depths, instead of the single depth value at the predicted (u, v). The GT vs UniDepthV2 gap at 1.0× is small (Det 44.90 vs 44.13), which supports a later predicted-depth version.
- **Extend IDEA-014** with the evidence: Table 3, depth + K tools +5.7 Det F1 at fixed K.

## Caveats / disagreements
- The scale-shift protocol is unspecified. If only the K value changed and the image did not, the "collapse" partly measures a mismatch between the tool's K and the pixels, which would make it a different phenomenon from a real camera change.
- The setting is room-scale (ScanNet), IoU-thresholded, with an 8B model. We use 2B, small tabletop primitives and centre error in metres. No metres are reported, so the effect sizes do not transfer.
- The arithmetic is done by the LLM in the CoT, so "deterministic" describes the form of the computation, not its execution.
- Some numbers are odd: the w/o-CoT row equals the baseline exactly, and Table 3's no-tools row (38.43 / 50.13) does not match the baseline (38.20 / 53.85). The training details of the baselines (zero-shot vs ScanNet-fine-tuned) are not stated.
- It is a single-version preprint with no code released yet.
