# Reading queue

Top = next. The researcher marks items `[x]` when read and appends up to 3 follow-ups per paper. Titles in quotes come from the project description; find the exact paper first.

## Priority (project reading list + supervisor pointers)
- [ ] OneCanvas: 3D Scene Understanding via Panoramic Reprojection (arXiv 2606.19253). Bart's suggested grounding backbone. How does it unproject patches, and how could it consume our depth + poses?
- [ ] "Limits of Fine-Tuning" (VLM physical reasoning). Earlier work says SFT fails OOD; this is the claim exp000 tests.
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
