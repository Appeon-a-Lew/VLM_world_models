# Reading queue

Top = next. The researcher marks items `[x]` when read and appends up to 3 follow-ups per paper. Titles in quotes come from the project description; find the exact paper first.

## Priority (project reading list + supervisor pointers)
- [x] OneCanvas: 3D Scene Understanding via Panoramic Reprojection (arXiv 2606.19253). Bart's suggested grounding backbone. How does it unproject patches, and how could it consume our depth + poses? → `2026_onecanvas.md`
  - [x] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded VLMs (arXiv 2509.17664, NeurIPS 2025). Single-view depth positional encoding is the closest analogue to IDEA-013 for our one-image setting. → `2025_sd_vlm.md`
    - [ ] Depth Anything V2 (Yang et al., NeurIPS 2024, arXiv 2406.09414). SD-VLM's estimated-depth source. Needed for the RGB-only follow-up to IDEA-013/020 and for the IDEA-022 diagnostic.
  - [ ] SR-3D: 3D Aware Region Prompted Vision Language Model (arXiv 2509.13317). Unifies single-view and multi-view inputs with 3D position embeddings; check how single-view xyz is encoded.
  - [x] SpaceMind: Camera-Guided Modality Fusion for Spatial Reasoning in VLMs (arXiv 2511.23075). Camera conditioning is a direct lever for the val_ood_config viewpoint shift (IDEA-016). → `2025_spacemind.md`
    - [ ] VGGT: Visual Geometry Grounded Transformer (Wang et al., CVPR 2025, arXiv 2503.11651). SpaceMind's frozen geometry encoder. Check its single-image depth/point-map accuracy and camera convention before running IDEA-022.

## Question-driven (open question in STATE.md: does the model use the stated impulse or regress to a prior? Not cited by a read paper)
- [x] QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of VLMs (Puyin et al., CVPR 2026, arXiv 2512.19526, https://arxiv.org/abs/2512.19526). Gives VLMs a numeric prior (size/velocity) and tests counterfactual priors. → `2026_quantiphy.md`
  - [ ] Thinking in Space / VSI-Bench (Yang et al., arXiv 2412.14171; verify the ID). The source of the MRA metric and of blind (no-vision) baselines for metric spatial QA. Qwen3-VL-2B scores 53.9 on it.
  - [ ] Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties (Tung et al., 2023; arXiv ID to verify). Tests mass/friction inference from dynamics. Our impulse→speed step needs mass inferred from the image.
- [ ] "Selective Cross-Modal Alignment in Vision-Language Models: Why Position Grounding Succeeds but Physical Reasoning Fails" (NeurIPS 2026, https://neurips.cc/virtual/2026/157263; arXiv ID not yet found). According to a search summary, probes Qwen2.5-VL-3B on synthetic physics. Velocity is decodable from the visual tokens but unused by the language readout. Mirrors our Task A ok / Task B weak pattern; could motivate a probing diagnostic. Verify the paper before relying on it.
- [x] "Limits of Fine-Tuning" (VLM physical reasoning). Earlier work says SFT fails OOD; this is the claim exp000 tests. → Schulze Buschoff et al., ICML 2025, arXiv 2502.15678, `2025_limits_of_finetuning.md`
  - [ ] Can vision language models learn intuitive physics from interaction? (arXiv 2602.06033). The follow-up shows that RL from interaction also fails to generalise. Probably the "Learning Physics from Interaction" item below; verify.
  - [ ] Lerer, Gross, Fergus 2016, "Learning Physical Intuition of Block Towers by Example" (arXiv 1603.01312). Their real-photo OOD test set and a non-VLM baseline that predicts from images.
- [ ] "Physics Context Builders". Small fine-tuned VLMs as context providers for a large VLM.
- [ ] "World Models from Inverse Dynamics". Inverse dynamics as a prior.
- [ ] "Learning Physics from Interaction"
- [ ] Physion: Evaluating Physical Prediction from Vision in Humans and Machines (Bear et al., 2021). Benchmark design, OOD scenarios.
- [x] Qwen3-VL technical report. The 3D-grounding output format, the camera conventions it was trained with, and what spatial data it saw. → Bai et al., arXiv 2511.21631, `2025_qwen3_vl.md`
  - [x] Omni3D / Cube R-CNN (Brazil et al., CVPR 2023, arXiv 2207.10660). Qwen3-VL "follows Omni3D" for its virtual camera. The paper defines the frame (+x right, +y down, +z in) and virtual depth z_v = z·(f_v/f)·(H/H_v), f_v = H_v = 512. Read in full: does Omni3D include tabletop data (Objectron), and how exactly is the virtual camera used? This bears on IDEA-024. → `2023_omni3d_cube_rcnn.md`
    - [ ] Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations (Ahmadyan et al., CVPR 2021; arXiv 2012.09988, verify ID). The only close-up / tabletop-scale source in Omni3D (46,644 images, 9 categories). Check the object scales, camera distances and categories to judge how close Qwen3-VL's possible 3D pretraining comes to our 5-12 cm primitives at ~1 m.
    - [ ] Metric3D: Towards Zero-shot Metric 3D Prediction from a Single Image (Yin et al., ICCV 2023, arXiv 2307.10984). A canonical-camera transform for metric depth, the depth-network analogue of virtual depth. Relevant if IDEA-022/020 move to predicted depth, and to the IDEA-019 virtual-depth control arm.
  - [x] Towards Camera-Robust 3D Localization: Equation-Anchored Tool-Use for MLLMs (Jiang et al., arXiv 2605.19528, preprint). Abstract: MLLMs "overfit to a canonical training intrinsic"; tested with K rescaled 0.5×-1.5×; back-projection with K plus sampled depth fixes it. Direct evidence for IDEA-019/014. → `2026_camera_robust_3d_tooluse.md` (only K is studied, not extrinsics/elevation)
    - [ ] VLM3: Vision Language Models Are Native 3D Learners (arXiv 2605.30561; seen only in secondary summaries, verify). Qwen3-VL-4B SFT that resizes images to a standard focal of 1000 px and uses text pixel coordinates on a [0, 2000) grid. A focal-normalisation alternative to IDEA-019, and a pixel-coordinate format relevant to IDEA-014.
    - [ ] Grounded 3D-Aware Spatial Vision-Language Modeling (GR3D; CVPR 2026, arXiv 2605.30307). Search snippet: names the coupling of object scale, depth and intrinsics as the core single-view 3D-grounding ambiguity. Check whether it addresses viewpoint/elevation as well as K, which is the gap left by Jiang et al.
    - [ ] UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler (Piccinelli et al., arXiv 2502.20110). The predicted-depth source in Jiang et al. (small GT vs UniDepthV2 gap at 1.0×). A candidate RGB-only depth for IDEA-015/020 if Depth-Anything-V2 (relative depth) is not metric enough; check whether it runs with the current install.
  - [ ] N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in VLMs (arXiv 2512.16561). Feeds estimated depth and intrinsics so that boxes come out in metric camera frame. An alternative route to IDEA-020/013 without GT depth.

## Background
- [ ] Kubric (Greff et al., 2022) / MOVi datasets. A candidate second data source with ground-truth 3D trajectories.
- [ ] CLEVRER: Collision Events for Video Representation and Reasoning (Yi et al., 2020)
- [ ] The "something something" video database (Goyal et al., 2017). The real-video transfer question.
- [ ] Shortcut Learning in Deep Neural Networks (Geirhos et al., 2020). Framing for the image-dependence test.
- [ ] Recent "do VLMs have intuitive physics" evaluations (e.g. PhysBench, IntPhys 2, Schulze Buschoff et al.). What do they find about grounding vs text priors?
- [ ] SpatialVLM / SpatialRGPT-style metric spatial fine-tuning. Does metric 3D SFT data transfer to dynamics?
