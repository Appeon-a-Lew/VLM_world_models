# Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models

**Citation**: Vahid Balazadeh, Mohammadmehdi Ataei, Hyunmin Cheong, Amir Hosein Khasahmadi, Rahul G. Krishnan. arXiv:2412.08619 (v1 11 Dec 2024, v2 10 Mar 2025, v3 28 Oct 2025), cs.CV/cs.AI. The abstract page lists no venue. **ID verified** against the abstract page: https://arxiv.org/abs/2412.08619. Read: HTML v3, https://arxiv.org/html/2412.08619v3
**Read on**: 2026-10-24. I read §2-6 and App. A.2, C and Tables 1-10 in the HTML. Some table headers in the HTML are garbled, so column labels in Tables 2, 5 and 10 are partly my reading (flagged below). Fig. 3 (data scaling) was not readable. This is the queue item "Physics Context Builders".

## TL;DR
A PaliGemma-3B, LoRA-fine-tuned on simulator annotations to *describe* a scene (a "PCB"), writes text that is given, together with the image and the question, to a frozen large VLM (GPT-4o, GPT-4o-mini, Gemini 1.5 Pro). This raises the large VLM's accuracy, by up to +25.5 points on simulated tower stability. In-distribution, however, the same 3B model fine-tuned end-to-end on the QA pairs beats every PCB-augmented large model. On the only OOD test (20 real photos) neither route consistently wins. Gains concentrate on perception-heavy questions; counterfactual gains are small, and predictive questions were not evaluated.

## Setup / method
- **Data** (§3.1):
  - CLEVRER: videos; descriptive, explanatory, predictive and counterfactual QA; 10,000 training videos.
  - *Falling Tower* (new): 4,864 simulated stack images, 72,775 QAs, 75/25 train/eval. Whether the split is by scene or by configuration is not stated. There is also a real set of 20 photos of 3D-printed objects with 100 human-written QAs (App. C).
- **PCB** (§4.1, Table 7):
  - Model: PaliGemma-3B with LoRA r = 16 (1.24% trainable), "vision and language components optimized jointly". 8 frames for CLEVRER, 1 for Falling Tower.
  - Target: a scene description generated "Using annotations from the simulator", in one of two styles (App. A.2).
    - *Human Narration (HN)*: prose, e.g. "In this scene, there are 3 collisions occurring in sequence."
    - *Structured Physics (SP)*: per-frame tags "[FRAME] [OBJECTS] [OBJ] SHAPE COLOR MATERIAL [LOC] LOC [/LOC] [VEL] VEL [/VEL] [/OBJ] ...", with "discretized and normalized" locations and velocities (bins not given) plus past collisions.
- **Consumer** (§4.2): GPT-4o / GPT-4o-mini / Gemini 1.5 Pro. They receive the **image or video, the question, and the PCB text**, so this is not a text-only reasoner. CoT is used.
- **Baselines** (Tables 1, 2, 4, 5):
  - zero-shot CoT large VLMs;
  - zero-shot PaliGemma-3B-mix;
  - **PaliGemma-3B fine-tuned end-to-end on the QA pairs** (direct SFT).
  - Not reported: a GT-description oracle, a description-without-image arm, a small model reading its own description, or PCB + SFT.

## Key results
- **Falling Tower, tower stability, sim / real accuracy** (Table 1, Table 4, HN):
  - zero-shot: GPT-4o 59.6 / 55.0, GPT-4o-mini 53.1 / 36.8, Gemini 60.5 / 60.0;
  - with PCB: GPT-4o 85.1 / 70.0, GPT-4o-mini 84.7 / 40.0, Gemini 84.9 / 57.9;
  - **end-to-end SFT PaliGemma-3B: 87.6 / 65.0**.
  - In sim, SFT ≥ every PCB arm. On real (20 images), only GPT-4o-PCB beats SFT (+5.0), and two PCB arms are below it.
- **Falling Tower, object stability, sim / real** (Tables 1, 4):
  - SFT 84.6 / 70.0;
  - with PCB: GPT-4o 76.7 / 75.0, GPT-4o-mini 75.0 / 70.0, Gemini 75.9 / 73.7;
  - zero-shot Gemini 54.6 / **80.0**. The best real-image object-stability score is the zero-shot model, which says more about the size of the real set than about any method.
- **CLEVRER** (Tables 2, 5; first value per column, which I read as per-question; headers garbled):
  - GPT-4o zero-shot → GPT-4o-PCB (HN): descriptive 62.7 → 75.6, explanatory 65.5 → 67.0, counterfactual 60.2 → 68.4;
  - end-to-end SFT PaliGemma-3B: 92.9 / 98.2 / 88.7.
  - **SFT beats the best PCB arm in each category in-distribution by about 17-27 points** (descriptive 92.9 vs 75.6, explanatory 98.2 vs 70.8 for Gemini-PCB, counterfactual 88.7 vs 68.4).
  - Predictive was not evaluated with PCBs: "our current PCBs are designed to describe observed scenes" (Remark, §4.2.2).
- **HN vs SP** (Table 5): structured physics text with discretised locations and velocities is mostly *worse* than prose, e.g. GPT-4o explanatory 61.1 (−4.4 vs zero-shot) under SP vs 67.0 under HN.
- **Average gains** (§5): "PCBs increase the average performance of GPT-4o (GPT-4o-mini) by 11.1% (11.8%) on CLEVRER and 8.2% (13.8%) on Falling Tower."
- **Ablations**:
  - Freezing PaliGemma's vision module lowers SFT accuracy in all categories (Table 10, header garbled; the largest drop is −8.9).
  - Descriptive accuracy saturates at about 10% of the training data, stability keeps improving (Fig. 3, text only).
- **Authors' conclusions** (§5-6): PCBs help most with perception and description. "the subpar performance on more involved reasoning tasks calls for additional interventions", and "this translation process does not capture all the visual cues needed for more complex reasoning."

## Relevance to us
- **Does perception→text→reasoning beat end-to-end SFT OOD? Not in this paper's evidence.**
  - In-distribution, end-to-end SFT of the small model wins everywhere (CLEVRER by about 17-27 points against the best PCB arm, tower stability 87.6 vs ≤ 85.1).
  - The only OOD test is Sim2Real on 20 images / 100 QAs. On tower stability, PCB beats SFT for one consumer out of three (70.0 vs 65.0) and loses for the other two (40.0, 57.9). On object stability, two of three PCB arms beat SFT (75.0, 73.7 vs 70.0) and one ties (70.0), but zero-shot Gemini without a PCB beats them all (80.0). On 100 QAs, a 5-point difference is about 5 questions.
  - So the paper's case for decomposition rests on modularity and on lifting a frozen large model, not on better OOD generalisation than SFT. There is no CI.
- **Mapping to our Task A / Task B split.**
  - A PCB is our **Task A** (perception to text). The consumer's job, given the description and the image, is our **Task B dynamics step**.
  - The paper's pattern: big gains on perception-limited questions (descriptive, stability where the consumer lacked perception), small gains on counterfactual, untested on predictive. This matches our situation: Task A is good ID (0.016 m) and Task B is the weaker part.
  - A description only helps a downstream reasoner when perception is the bottleneck. Their remark that a 3B SFT model beats GPT-4o + PCB also says the dynamics/reasoning step is learnable end-to-end in-domain. That is what exp000 shows on val_id and objects.
- **Implication for the next lever after exp003** (conditional on its routing):
  - **If exp003 says perception breaks** (SFT-B well below own-static, e.g. on val_diag_elev): a perception→dynamics decomposition just passes the 0.136 m Task A error on to Task B. PCB-style decomposition is *not* the config lever; use geometry inputs (IDEA-016/014/020) or locate the failure first (IDEA-030).
  - **If exp003 says dynamics breaks with good perception** (SFT-B ≈ own-static on a factor, or the forces gain proves prior-driven via IDEA-025/026): decomposing (IDEA-002, and IDEA-033 below) gives a clean dynamics-only measurement.
- **Cheap analogue with our single 2B model.**
  - Feeding exp000's own Task A answer into its Task B prompt at inference is **off-distribution for the exp000 adapter**, which never saw coordinates in a Task B prompt. A null result would be uninterpretable, so I do not recommend it as an eval-only shortcut.
  - The single-model analogue that *is* in-distribution is **IDEA-002**: Task B answers the current centre first, then the displacement. The model writes its own perception into its context before the dynamics tokens, which is PCB with producer = consumer and in one pass. At training the centre tokens are teacher-forced with GT, and at test the model reads its own. This is the same train/test asymmetry as a PCB trained on simulator descriptions.
  - IDEA-002 also allows a free oracle arm: force the current-centre prefix to the GT t=0 centres at decode time and let the model generate only the displacement (IDEA-033). This is the "GT description" oracle the paper lacks. It is directly comparable to `static`, since both use oracle perception, so it measures pure dynamics skill per split.
- **Format lesson (HN > SP)**: discretised structured physics text was worse than prose for the consumer. Our outputs are continuous JSON coordinates and the consumer is the same fine-tuned model, so this does not transfer directly. It is a mild warning against adding many numeric tokens to the prompt (relevant to IDEA-016 camera-pose text) without SFT on that format.
- **Image-dependence**: the consumer always gets the image as well, and no description-only arm exists, so the paper cannot tell whether the consumer reads the scene or only the text. In our setting the analogous question for IDEA-002/033 is whether the displacement step uses the image beyond the centre coordinates. A forced-GT-prefix run with a **blank image** answers it (IDEA-033).

## Ideas for us
- **IDEA-033: oracle-perception prefix forcing on an IDEA-002 model (dynamics-only diagnostic).**
  - Requires IDEA-002 to be trained, with an answer format that lists all current centres *before* the displacements.
  - Single change, eval-only: at decode time, pre-fill the assistant answer with the GT t=0 current-centre block, then let the model generate the displacements. Score `GT t0 + Δ̂`.
  - Write to `pred/oracleprefix.jsonl` (stem matches `[a-z]*` for `compare`), never to `normal.jsonl`. Add a blank-image variant `pred/oracleblank.jsonl`.
  - Expected:
    - (oracleprefix vs static) = pure dynamics skill at oracle perception, per split, with a paired CI from `compare`.
    - (normal vs oracleprefix) = the perception error's contribution to Task B.
    - (oracleblank vs oracleprefix) ≈ 0 means the dynamics step reads only the text coordinates and action, not the scene (neighbours, table edge). This is a Task B-specific image-dependence test that the current ablations cannot isolate.
  - Cost: about 0.5 day of code (assistant-prefix forcing in worktree infer.py) plus about 0.15 GPU-h, after IDEA-002's SFT.
- **Extension to IDEA-002** (no new ID): IDEA-002 is the single-model, single-pass analogue of PCB. The paper's in-distribution SFT > PCB result argues for keeping it end-to-end (one model, one pass) rather than building a separate two-stage pipeline. Put the current-centre block *before* the delta in the answer so that IDEA-033 is possible.

## Caveats / disagreements
- The tasks are classification and multiple-choice QA (stability, CLEVRER), not continuous metric regression. The consumers are frozen proprietary large VLMs, not the fine-tuned producer itself.
- Their "OOD" is only Sim2Real on 20 images / 100 QAs, with no CI. Gemini zero-shot is the best real-image object-stability model (80.0), so the real-set ranking is noise-dominated.
- How Falling Tower is split (by scene or by configuration) is not stated, so "in-distribution" may include near-duplicate scenes.
- The HTML table headers for Tables 2, 5 and 10 are garbled. The Table 5 "explanatory per-option" GPT-4o value (18.7) matches Table 2's predictive per-option value, not its explanatory one (30.3). Per-option columns should not be relied on. I used only the first value per category.
- No error-propagation analysis: the paper never measures how PCB perception errors affect the consumer. That is exactly the quantity IDEA-033's (normal vs oracleprefix) difference would measure for us.
