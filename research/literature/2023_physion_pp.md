# Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties

**Citation**: Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel M. Bear, Chuang Gan, Joshua B. Tenenbaum, Daniel L. K. Yamins, Judith Fan, Kevin A. Smith. NeurIPS 2023 (Datasets & Benchmarks poster). arXiv:2306.15668 (ID verified; v2 2023-11-02). https://arxiv.org/abs/2306.15668, PDF: https://arxiv.org/pdf/2306.15668, project: https://dingmyu.github.io/physion_v2/
**Read on**: 2026-10-24. I read the PDF main text pp. 1-10: benchmark design §3, experiments §4.1-4.2, Tables 1-3, Fig. 4, discussion §5. I did not read the supplementary material (stimulus details, model training details). This note substitutes for the "Selective Cross-Modal Alignment" paper, whose full text I could not access (see QUEUE).

## TL;DR
Physion++ is a ThreeDWorld benchmark of 9 scenarios in which the outcome (does the red object touch the yellow one?) depends on a latent mass, friction, elasticity or deformability that can only be inferred from an "inference phase" of the video. Video, ImageNet-encoder and object-centric models barely use that phase: overall accuracy is ~52-57% whether it is shown or not. Only DPI-Net, given ground-truth 3D state and latent properties, gains clearly on mass (60.1% → 75.9%).

## Setup / method
- **Stimuli** (§3.1, Fig. 2): single videos of 13-20 s with:
  - an inference phase, where the latent property shows itself in the dynamics;
  - for some scenarios, a curtain "transition phase" that rearranges the objects;
  - a cue that blinks the red and yellow objects;
  - a prediction phase cut off before the outcome.
- **Scenarios** (9): Mass-Dominoes / Waterpush / Collision, Elasticity-Wall / Platform, Friction-Slide / Collision / Clothslide, Deform-Roll.
- **Paired trials**: the two videos are visually identical in the first prediction frame but have different latent values and outcomes, so appearance cannot predict the label (§3.1).
- **Task**: object contact prediction (OCP), a binary yes/no.
- **Data** (§3.2), per property:
  - 2,000 unlabelled dynamics-training trials;
  - a 192-trial (96-pair) readout set;
  - a 192-trial test set.
- **Models** (§4.1):
  - MCVD (pixel video diffusion);
  - pRESNET / pVGG / pDEIT-mlp (ImageNet encoders plus a learned latent-dynamics MLP);
  - SlotFormer and ALOE (object-centric);
  - DPI-Net (particle graph network on **ground-truth 3D state and latent properties**, no vision).
- **Protocol**:
  - Pretrain the dynamics with a future-prediction loss, then fit a logistic-regression readout on features from observed plus rolled-out frames.
  - Training is either 'separate' (one model per property) or 'unified'.
  - Testing uses three video variants:
    - **'w/ property'**: the full video;
    - **'w/o property'**: the first frames, where the critical object first interacts, are removed, so no property inference is possible;
    - 'fully observed': the outcome is visible, which upper-bounds the readout.
- **Humans**: 200 Prolific participants, 50 per property, 32 trials each (§4.2).

## Key results
- **Overall, separate training** (Table 2), w/ property vs w/o property vs fully observed, accuracy %:

| model | w/ property | w/o property | fully observed |
|---|---|---|---|
| MCVD | 54.3 | 54.3 | 59.9 |
| DeiT-mlp | 55.4 | 54.1 | 63.9 |
| ResNet-mlp | 55.1 | 54.8 | 61.2 |
| VGG-mlp | 55.5 | 55.1 | 56.0 |
| ALOE | 53.4 | 52.5 | 55.5 |
| SlotFormer | 56.7 | 56.2 | 65.1 |
| DPI-Net | 60.2 | 57.5 | 92.2 |

- The authors: "Most models with property inference perform similarly to those without property inference, indicating that these models are not utilizing physical property inference" (§4.1).
- **The one clear exception**: DPI-Net on mass, 75.9 w/ vs 60.1 w/o (Table 2, §4.1). DPI-Net receives the latents and 3D state directly.
- **Unified vs separate**: unified training is generally worse (Table 2, finding (1)).
- **Humans** (§4.2, Fig. 4):
  - accuracy about 60%;
  - split-half reliability r = 0.37;
  - best model-human correlation SlotFormer, r = 0.12;
  - models underperform humans on "easy" pairs and outperform them on "hard" pairs.
- **Data scaling** (Table 3): 200 → 2,000 trials per property lifts the four models tested from ~49-52% to ~53-57%, with small gains past 1,000.
- **Discussion** (§5): no model correlates with humans, and the authors conclude current models "do not make online inferences about physical properties".

## Relevance to us
- **Our Task B has a latent-property step we have not accounted for.** The action text gives the **impulse** (`push.py:157-159`, "with an impulse of {impulse:.2f} N·s"), and the initial speed is impulse / mass. Mass = density × volume:
  - Volume is visible in the image.
  - **Density is latent**, uniform in [400, 800] kg/m³ (`sim_train.yaml:10`, comment: "narrow so mass is ~inferable from visible size").
  - **Sliding friction is latent**, uniform in [0.35, 0.6]. `scene.py:148-149` makes the object's own friction govern sliding, since the table is set low and MuJoCo takes max().
  - All val splits share these ranges (the OOD configs inherit `sim_train.yaml`).
  - Our input is a **single frame**, so there is no inference phase at all. This is Physion++'s 'w/o property' condition by construction.
- **Size of the latent spread** (my derivation, not from the paper). For a point-mass Coulomb slider, d = v₀²/(2μg) with v₀ = I/(ρV). At fixed impulse and visible volume:
  - the 2× density range spans a 4× range in sliding distance;
  - friction adds 1.7×.
  - Relative to the mean latents (ρ̄ = 600, μ̄ = 0.475), d_true/d_mean runs from about 0.45 to about 3.05.
  - So part of Task B's error is **irreducible from one image** and grows with displacement. It weighs most on val_ood_forces strong pushes and on balls that roll far. That changes how to read exp000's Task B numbers (val_id 0.155, forces 0.510): we do not know how much headroom is left. This is IDEA-031.
- **IDEA-025 is robust to this; IDEA-008 is not.**
  - IDEA-025 scales the impulse *within* an episode, so ρ, μ and V cancel in d̂(2)/d̂(0.5). Physics predicts the slope of 2 regardless of the latents. That is good news for the next planned action-dependence experiment.
  - IDEA-008 regresses displacement on impulse *across* episodes, so it is confounded by the visible volume and by the latent ρ and μ. It needs a volume covariate (extension below).
- **The image-dependence test.** Physion++'s w/ vs w/o property contrast is an "information-removal ablation", the same logic as our none/blank/swap and the IDEA-026 cf_flip. Their main lesson is that a model can score above chance with no change when the decisive information is removed. In our terms, an apparent gain on forces may come from a displacement prior rather than from the stated impulse. That is the C-002 question IDEA-025/026 address.
- **Object-centric representations.** Objectness helped somewhat (SlotFormer is best among the vision models). This marginally supports IDEA-004 (set-of-marks) for the 4-5-object config split, but the effect is small (56.7 vs ~55%).

## Ideas for us
- **IDEA-031: latent-property error floor, as a derived pred file (no GPU).**
  - Single change: none to the model. A new derived baseline scored by pinned code.
  - For each val episode whose target stays on the table and where no other object moves (an "isolated slider/roller" subset, defined from GT metadata), build the prediction of an agent that is perfect except for the latents.
  - Target: p_t0 + (p_t1 − p_t0) · s, where s is the latent-mismatch factor:
    - sliders: s = (ρ/ρ̄)² · (μ/μ̄), with ρ̄ = 600 and μ̄ = 0.475;
    - balls: s = (ρ/ρ̄)², since the rolling friction is a constant 0.006.
  - Non-targets stay at GT. Write it as `pred/latent_floor.jsonl`, plus exp000's `normal` predictions restricted to the same episodes (`pred/normal_isolated.jsonl`), so pinned `summarize` scores both on the same subset.
  - Expected: a floor that is a sizeable fraction of exp000's Task B error on that subset would mean dynamics headroom from more training (IDEA-002) is limited, and a model that beats the floor must be reading something beyond the image and impulse. A negligible floor means latent ambiguity can be ignored.
  - Cost: about 0.5 day of code, no training. It can ride along in the IDEA-017 job (the exp000 adapter preds are needed for the subset file).
  - Caveat: the scaling is approximate (no rotation, toppling or edge effects), hence the isolated subset only.
  - Follow-up if the floor is large: a Physion++-style "w/ property" arm that states speed (m/s) instead of impulse in the action text. This is privileged, and its gain over exp000 upper-bounds what mass inference can buy. It costs 1 SFT run and is not proposed separately yet.
- **Extension to IDEA-008** (no new ID): add log visible volume (GT `mass/density` from metadata) as a covariate. Physics predicts ∂log d/∂log I = +2 and ∂log d/∂log V = −2 for sliders. A volume coefficient of ≈ 0 means exp000 converts impulse to displacement without inferring mass from size: a text-prior reading of the impulse number.

## Caveats / disagreements
- Their task is binary contact prediction from long multi-phase videos. Ours is continuous 3D regression from one frame plus an action text. Their models are video-prediction networks, not VLMs, so nothing here says how a VLM handles latent properties.
- Their 'w/o property' condition removes evidence that exists. In our setup the evidence never exists, so the analogue is an irreducible-error floor, not a capability test.
- Model accuracies of ~52-57% on 192 test trials per property have wide intervals. Table 2 has no CIs, and most w/ vs w/o differences are about 1 point.
- The latent-spread numbers in "Relevance to us" are my idealised derivation from our config ranges, not the paper's.
