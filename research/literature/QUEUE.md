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
- [ ] QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of VLMs (Puyin et al., CVPR 2026, arXiv 2512.19526, https://arxiv.org/abs/2512.19526). Gives VLMs a numeric prior (size/velocity) and tests counterfactual priors. A search summary of the abstract (not yet read) says models "lean heavily on pre-trained world knowledge" over the supplied inputs. Directly relevant to IDEA-008 and the design of a counterfactual-impulse test.
- [ ] "Selective Cross-Modal Alignment in Vision-Language Models: Why Position Grounding Succeeds but Physical Reasoning Fails" (NeurIPS 2026, https://neurips.cc/virtual/2026/157263; arXiv ID not yet found). According to a search summary, probes Qwen2.5-VL-3B on synthetic physics. Velocity is decodable from the visual tokens but unused by the language readout. Mirrors our Task A ok / Task B weak pattern; could motivate a probing diagnostic. Verify the paper before relying on it.
- [x] "Limits of Fine-Tuning" (VLM physical reasoning). Earlier work says SFT fails OOD; this is the claim exp000 tests. → Schulze Buschoff et al., ICML 2025, arXiv 2502.15678, `2025_limits_of_finetuning.md`
  - [ ] Can vision language models learn intuitive physics from interaction? (arXiv 2602.06033). The follow-up shows that RL from interaction also fails to generalise. Probably the "Learning Physics from Interaction" item below; verify.
  - [ ] Lerer, Gross, Fergus 2016, "Learning Physical Intuition of Block Towers by Example" (arXiv 1603.01312). Their real-photo OOD test set and a non-VLM baseline that predicts from images.
- [ ] "Physics Context Builders". Small fine-tuned VLMs as context providers for a large VLM.
- [ ] "World Models from Inverse Dynamics". Inverse dynamics as a prior.
- [ ] "Learning Physics from Interaction"
- [ ] Physion: Evaluating Physical Prediction from Vision in Humans and Machines (Bear et al., 2021). Benchmark design, OOD scenarios.
- [ ] Qwen3-VL technical report. The 3D-grounding output format, the camera conventions it was trained with, and what spatial data it saw.

## Background
- [ ] Kubric (Greff et al., 2022) / MOVi datasets. A candidate second data source with ground-truth 3D trajectories.
- [ ] CLEVRER: Collision Events for Video Representation and Reasoning (Yi et al., 2020)
- [ ] The "something something" video database (Goyal et al., 2017). The real-video transfer question.
- [ ] Shortcut Learning in Deep Neural Networks (Geirhos et al., 2020). Framing for the image-dependence test.
- [ ] Recent "do VLMs have intuitive physics" evaluations (e.g. PhysBench, IntPhys 2, Schulze Buschoff et al.). What do they find about grounding vs text priors?
- [ ] SpatialVLM / SpatialRGPT-style metric spatial fine-tuning. Does metric 3D SFT data transfer to dynamics?
