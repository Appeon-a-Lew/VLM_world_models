# Pixels to Principles: Probing Intuitive Physics Understanding in Multimodal Language Models

**Citation**: Mohamad Ballout, Serwan Jassim, Elia Bruni. arXiv:2507.16572 (v1, 22 Jul 2025), cs.CL. The abstract page lists no venue. https://arxiv.org/abs/2507.16572, HTML: https://arxiv.org/html/2507.16572v1
**Read on**: 2026-10-24. I read the HTML v1: §4 (behavioural evaluation), §5 (probing setup), §6 (Insights 1-6), §7-8, App. B (Table 2) and App. E (prompts). **Figures 1, 2 and 4 could not be read**, so per-model, per-embedding-site probe accuracies are only available where the text quotes them. I read this paper as the substitute for the blocked "Selective Cross-Modal Alignment" poster (see QUEUE).

## TL;DR
Open VLMs (InternVL 2.5, Qwen2.5-VL, LLaVA-OneVision; 0.5B-78B) score near chance at telling physically plausible from implausible videos (GRASP Level 2, IntPhys 2). A probe on the vision-encoder embeddings classifies plausibility better than one on the language-model embeddings in all nine models. The authors conclude the bottleneck is vision-language integration, not the vision encoder. The probe target is a scene-level binary label, and no continuous physical quantity (position, velocity, depth) is probed.

## Setup / method
- **Input**: 10-second videos sampled at 1 FPS, i.e. 10 frames (§4.1). Datasets:
  - GRASP Level 1 (basic perception: shape, colour, movement, direction);
  - GRASP Level 2 (plausible vs implausible across 16 scenes such as gravity, inertia, support);
  - IntPhys 2.
- **Behavioural output**: a plausible/implausible answer under three decoding modes: free sampling (temperature 0.6, top-p 0.9, top-k 40), greedy, and chain-of-thought (§4.1).
- **Models**: InternVL 2.5 2B/8B/78B, **Qwen2.5-VL 3B/7B/72B**, LLaVA-OneVision 0.5B/7B/72B, and Gemini 2.0 Flash Thinking (behaviour only). **No Qwen3-VL.**
- **Probing protocol** (§5.1-5.2):
  - *Sites*: three embedding sites per model.
    - "Vision - before projection": "from the last vision transformer block".
    - "Vision - after projection": "after the projection layer, at the end of the entire vision encoder".
    - "Language": "the final layer of the language model (before the output MLP)", under a simple prompt and a detailed prompt (App. E).
    - **No layer sweep**; §7 lists layer-wise analysis as future work.
  - *Pooling*: "we averaged over the sequence dimension, condensing multiple token-level embeddings into a single vector per video". For the language site the paper does not say whether only visual-token positions or all tokens are pooled.
  - *Probe*: the §5 intro says "linear probes", but §5.2 says "one-hidden-layer MLP probes (hidden size=512, ReLU activation, no dropout) for 500 epochs using cross-entropy loss". The probe is therefore **non-linear**; this is an internal inconsistency in the paper.
  - *Split and tuning*: a single "80% training, 10% validation, and 10% testing" split. "Hyperparameters were optimized via grid search" (grid not given). Results are averaged over three seeds.
  - *Dataset sizes* (§5.2): GRASP Level 1 has 128 videos per task; Level 2 has 4,096 videos (128 plausible + 128 implausible per each of 16 scenes); IntPhys 2 has 1,012 videos.
  - *Labels*: binary plausibility for Level 2 and IntPhys 2. Level 1 probes only shape and movement; colour and direction were dropped (four classes, too few samples).
  - *Dimensions* (App. B, Table 2): Qwen2.5-VL is 1280 before projection at all sizes. After projection, and for the language site, it is 2048 (3B), 3584 (7B) and 8192 (72B).
- **Controls against probe overfitting**: **none**. There are no shuffled-label, control-task, random-init or blind (no-image) probe baselines. The only reference point is chance (0.5).

## Key results
- **Behaviour** (§4, Table 1):
  - GRASP Level 1 is mostly solved by the large models. Qwen2.5-VL 72B scores 83.7 (free) / **99.7 (greedy)** / 80.4 (CoT), against LLaVA-OV 72B 62.8 / 68.1 / 76.4 and InternVL 2.5 78B 66.9 / 74.8 / 66.4.
  - GRASP Level 2: the largest Qwen2.5-VL and InternVL 2.5 only "exceed 53% accuracy" (§4.1). Gemini 2.0 Flash Thinking scores 53.7%.
  - IntPhys 2: the largest models reach about 52%.
  - Scene-level outliers exist, e.g. Qwen2.5-VL 7B scores 68.9% on Gravity-Inertia (free).
- **Probing** (§6):
  - Insight 1: GRASP Level 1 features (shape, movement) are probed at 100% for all models. IntPhys 2 is at chance for vision and language embeddings alike.
  - Insight 2: GRASP Level 2 vision embeddings reach 85.0% (InternVL 2.5 78B), 80.2% (LLaVA-OV 0.5B) and **70.8% (Qwen2.5-VL 7B)**. That is far above the ~50-53% behaviour, so "embeddings encode sufficient information to assess video plausibility".
  - Insight 3: "all nine models, irrespective of their size, perform better when probing vision embeddings" than language embeddings. The language-site accuracies are in Fig. 2 only, so I have no numbers for them.
  - Insight 4: the detailed prompt helps the language-site probe for InternVL 2.5 78B and all LLaVA-OV models, but **"all Qwen 2.5 VL models performed worse"**.
  - Insight 5: "The projection layer generally has a negative effect, with some exceptions", the exceptions being InternVL 2.5 2B and 78B.
  - Insight 6: t-SNE shows vision embeddings clustering by scene, and this clustering "gradually diminishes after embeddings pass through the projection layer". This is qualitative, and there is no plausible/implausible separation in the plots.
- **Stated limitation** (§8): the exact mechanism of the loss "requires further investigation".

## Relevance to us
- **What is decodable, and where: weak but directionally consistent evidence.**
  - The ordering (vision before projection ≥ vision after projection > language final layer) matches the "competence without performance" pattern in `2026_vlm_physics_interaction.md` §4.4, and the abstract-only claim of the blocked NeurReps poster.
  - The variable here is a scene-level binary plausibility label, decodable by a 512-unit MLP. It is not a metric quantity, and "linearly decodable" is not supported (the probe is an MLP).
  - So this paper does **not** tell us whether camera-frame xyz is linearly decodable from Qwen visual tokens. IDEA-030 still has to measure that.
- **Qwen-family coverage**: Qwen2.5-VL 3B/7B/72B, not Qwen3-VL. The one Qwen number is 70.8% (7B, vision site, Level 2), the lowest of the three quoted.
  - Insight 4 is the Qwen-specific oddity: prompt detail *lowered* language-site decodability for every Qwen2.5-VL.
  - In our pipeline the image comes before the text (`physwm/vlm/prompts.py:82`). Under causal attention, LLM hidden states *at visual-token positions* therefore do not depend on the prompt text: Task A vs Task B and the action line make no difference there.
  - Probing visual-token positions is thus immune to the Insight 4 effect. It also cannot see the action, so it measures t=0 perception only, never dynamics.
- **IDEA-030 protocol: what to copy and what not to copy.**
  - *Copy*: probe **before the merger as well**. Insight 5 says the projection step generally loses decodability. For Qwen3-VL, the matching sites are the last ViT block (pre-merger, four 16-px patches per merged token; verify the patch size in the processor config) and the merger output (= LLM input embedding).
    - Qwen3-VL also adds DeepStack ViT features into early LLM layers (`2025_qwen3_vl.md`, Setup). The first LLM layer to probe should therefore come *after* the injection layers; check `deepstack_visual_indexes` in the config.
  - *Do not copy*:
    - Whole-sequence mean pooling erases per-object location, which is fine for a scene label but wrong for per-object xyz.
    - The MLP probe.
    - The single 80/10/10 split with no control.
    - With ~4k videos, a 512-unit MLP on 1280-3584-d inputs over 500 epochs, no dropout and λ tuned on one validation split, the paper's above-chance numbers have no selectivity check.
- **Image-dependence gap**: none of the probes is a blind or blank-image control. Our supervisor's concern (answers from priors, not the scene) maps to the probe question "would the probe score as well on features of a blank image at the same token positions?" The paper never asks this.
- **Task A vs Task B / OOD split**: there is no OOD analysis (one split per dataset) and no action conditioning, so it gives nothing on val_ood_forces/config directly.

## Ideas for us
No new IDEA. The paper sharpens the protocol of the existing IDEA-030 rather than suggesting a new single change. Appended to IDEA-030 in BACKLOG as a "protocol refinement":
- **Probe sites** (5 instead of 3): ViT last block (pre-merger, 2×2-pooled onto the merged grid), merger output, the first LLM layer after the DeepStack injection, the middle LLM layer, and the last LLM layer. All are read at the object's merged-token position(s); there is no whole-sequence mean pooling. Rationale: Insight 5, where the projection reduced decodability in most models.
- **Selection-leakage control**. The token is chosen by the projected GT centre, so (u, v) leaks into the probe through the choice itself. Fit a ridge on the token's grid coordinates alone (plus a constant), and fit the main ridge on features of a **blank image** at the same token positions. Report the main probe's gain over both, per axis, with **z (depth) as the headline axis**, because u, v leakage cannot fake it.
- **Shuffled-target control** (Hewitt & Liang-style selectivity): refit with xyz targets permuted across training objects. The held-out error should equal the predict-the-mean error. If it is lower, the probe or split leaks, for example the same episode in train and test.
- **Split and regularisation**: split by **episode**, not by object. Pick λ by 5-fold CV inside `train` only. Use at least ~2× feature-dim training objects (the 2B hidden size is about 2k; verify), or PCA to 256 dims first. Report the held-out error next to a predict-train-mean baseline per split.
- **Cost**: unchanged, about 0.5 day of code and about 0.1 GPU-h, plus one extra forward on blank images (≈ +0.02 GPU-h).

## Caveats / disagreements
- The task (video plausibility, violation of expectation) and the inputs (10-frame video) differ from ours (one frame, metric coordinates, stated action).
- "Linear" vs MLP probe inconsistency: no claim about linear decodability can be cited from this paper.
- There are no numbers for the language-site probes or for Qwen2.5-VL 3B/72B; these are in Fig. 2, which I could not read.
- The pooling at the language site is unspecified. If it includes text tokens, the vision vs language comparison is not like-for-like.
- With a single split, three seeds and no controls, the 70.8-85.0% values have unknown variance and selectivity.
- No Qwen3-VL model was tested. Qwen3-VL's DeepStack injection changes where visual information enters the LLM, so the ordering of "after projection" vs "language" sites may not carry over.
