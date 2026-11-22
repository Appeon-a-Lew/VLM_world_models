# M0 push screen results

All four seed-1 arms completed 3,750 optimizer steps on TUM RTX 5080s. LRZ scoring job `5947386` evaluated 100 matched push families. The preregistered decision was **no replication earned**. Seed 2 and the held-out tilt experiment were not launched.

| Arm | Matched-pair effect error | Error reduction vs forward-only | Paired 95% CI |
| --- | ---: | ---: | ---: |
| Forward-only | 31.66 cm | — | — |
| Inverse dynamics | 31.14 cm | +0.52 cm | [−1.21, +2.32] cm |
| Physical context | 31.36 cm | +0.30 cm | [−0.91, +1.63] cm |
| Both | 33.17 cm | −1.51 cm | [−3.47, +0.44] cm |

Positive reductions favor the added training task. Seed 2 required a reduction of at least **2.21 cm**, a confidence interval above zero, and a missing-pair rate within two percentage points of forward-only. Every arm had zero missing target pairs. [Preregistered criteria](PREREG.md)

The additional tasks learned useful signals on the validation data. The inverse-dynamics arm answered **81%** of in-distribution pairs exactly. The physical-context arm matched the obstacle set on **89%** of in-distribution episodes. Forward-only had the lowest in-distribution endpoint error for the pushed object (**13.90 cm**). All four arms beat the static baseline on strong out-of-range pushes, and the static baseline had the lowest error on weak pushes.

These comparisons use one training seed. The confidence intervals quantify variation across scene families. The screen establishes the spending decision under the preregistered rule.

The scoring run saved the paired comparisons, force buckets, counterfactual analysis, and image ablations in its `runs/paired/` directory.

## Skill relative to a no-change baseline

This exploratory view rescales the original M0 scores. Matched-pair skill is `1 − sum(model effect error) / sum(no-change effect error)`. The baseline predicts zero change between the original and altered pushes. The 300 pairs fall into three equal groups by true endpoint change, with cutoffs of about 13.9 cm and 46.0 cm. The dashboard confidence intervals resample scene families. [M0 skill charts](https://appeon-a-lew.github.io/VLM_world_models/m0/#effect-skill)

| Arm | All pairs | Weak effects | Medium effects | Strong effects | Weak OOD pushes | Strong OOD pushes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward-only | 28.3% | 18.3% | 40.6% | 25.9% | −188.8% | 29.9% |
| Inverse dynamics | 29.5% | 25.1% | 44.1% | 26.1% | −256.8% | 27.6% |
| Physical context | 29.0% | 27.2% | 42.3% | 25.8% | −169.6% | 28.9% |
| Both | 24.9% | 17.1% | 36.8% | 22.4% | −167.5% | 25.5% |

The OOD columns use the existing push-speed buckets and score future endpoints against the starting-position baseline. The matched-pair columns score changes between two predicted endpoints. For the appearance-only controls, the predicted endpoint stayed within 2 cm of the original in 61–67% of recolor pairs and 16–21% of camera-elevation pairs. The preregistered spending decision remains based on the original effect-error comparison.

## Post-screen audit

### What the papers implemented

Qiu et al. train an inverse model separately. They use it to label additional video transitions for forward training and to rank candidate futures by the likelihood of the requested action. The released forward model predicts image tokens. M0 instead mixes an action-reconstruction task into the same LoRA training run as 3D box prediction and decodes a single future. The published repository and its data-processing branch contain the separate training, preprocessing, and verification paths. [Paper](https://arxiv.org/abs/2506.06006) · [Code](https://github.com/yfqiu-nlp/vlm-world-model) · [Preprocessing](https://github.com/yfqiu-nlp/vlm-world-model/tree/data-processing)

Balazadeh et al. train a separate perception model to describe observed scenes and feed its output to a larger reasoner. Their videos use eight input frames and the released paper reports results for question answering and stability. M0 asks one model to emit a short, action-specific context and then forecast numeric boxes. The paper links its Falling Tower datasets and documents the training procedure. A public training repository could not be verified. [PCB paper and appendix](https://arxiv.org/html/2412.08619v2) · [Falling Tower data](https://drive.google.com/file/d/1RgpIdcywqvjYYXT3xTo-r1vxpUu2giby/view?usp=sharing)

### What we verified on the saved run

- **Data integrity.** The 10,000 training and 1,000 ordinary validation episodes had the required pre-action and 0.1 s images. There was no seed overlap between splits. All 100 matched families passed the existing family checks. Replaying the original, reverse, half, and double variants with the recorded state and action reproduced all 400 target endpoints to the reported precision. In a 128-episode training sample, 509 of 510 visible object centers projected within their segmentation bounding box with a two-pixel tolerance.
- **Action information.** The prompt exposes a direction label while the simulator adds up to 15° of jitter. Replaying with the nominal named direction and otherwise exact simulator inputs produced **7.23 cm mean paired-effect error**. This measures the effect of substituting the nominal direction. Density and friction also vary independently of appearance, which makes the exact 3D forecast uncertain from a still image.
- **Context coverage.** All 20,078 starting objects in training had table support. Only 22.84% of training scenes had a labeled obstacle in the push corridor, and only 5.56% ended with another object displaced by at least 2 cm. An always-empty obstacle prediction would score 77% on in-distribution validation. The context model scored 89%. Target and direction are already in the forward prompt.
- **IDP coverage.** The 0.1 s labeling rule assigned all 250 weak out-of-range pushes to the no-clear-motion class. The reported out-of-range accuracy therefore offers no test of weak-push target, direction, or strength recovery.
- **Training exposure.** All four run logs show 3,750 optimizer steps, with PE disabled. Forward-only and context each consumed **30,000 Task B examples**. IDP and both each consumed **20,000**. In a 233-episode sample tokenized with the production tokenizer, future-box tokens accounted for approximately **49%, 45%, 28%, and 27%** of supervised tokens in forward-only, IDP, context, and both respectively. These are token shares. The run measures the practical trade-off at fixed optimizer steps, with forward exposure varying across recipes.
- **Action response.** For doubled pushes, the mean change in the simulator's target endpoint was **64.3 cm**. The four models changed their predictions by **15.5–19.3 cm** on average. They responded to the action, with substantial underestimation of the size of its effect.

The gate result stands for the executed recipes. A focused follow-up would first test how the same predictor responds to explicit direction and simulator context on the existing matched scenes. An auxiliary-training comparison should hold Task B exposure fixed. A paper-style inverse-dynamics comparison would test whether a separate verifier improves selection among genuinely different candidate futures.

## Exploratory prompt diagnostic

The saved forward-only and context checkpoints each receive four Task B prompts on the same 100 matched push families. Every condition uses the same input image, batch size, decoder, and box output format for its checkpoint.

| Prompt | Added information |
| --- | --- |
| Original | Existing action description |
| Nominal vector | The named direction expressed as a camera-frame unit vector |
| Recorded vector | The actual simulator direction in the same numeric format |
| Recorded vector + context | The actual direction plus simulator-provided support, edge distance, and obstacles |

The main diagnostic comparisons are nominal versus recorded direction and recorded direction versus recorded direction plus context. Error differences use 2,000 paired family bootstrap draws, as in the screen. The numeric prompts can expose camera geometry, so the original-versus-nominal comparison checks how the new representation affects the saved checkpoints. This follow-up uses the same validation families and tests how the existing checkpoints respond to extra information at inference.

| Input prompt | Forward-only error | Context-trained error |
| --- | ---: | ---: |
| Original | 31.18 cm | 31.42 cm |
| Nominal vector | 30.71 cm | 31.74 cm |
| Recorded vector | 31.07 cm | 33.06 cm |
| Recorded vector + context | 31.41 cm | 35.63 cm |

All eight passes covered all 100 families with zero missing target pairs. Positive reductions in the table below favor the second prompt.

| Comparison | Forward-only reduction with 95% CI | Context-trained reduction with 95% CI |
| --- | ---: | ---: |
| Nominal → recorded | −0.36 cm [−2.03, +1.40] | −1.31 cm [−4.50, +2.02] |
| Recorded → recorded + context | −0.34 cm [−1.64, +1.21] | −2.57 cm [−5.73, +0.43] |
| Original → recorded + context | −0.22 cm [−1.51, +1.28] | −4.21 cm [−8.44, +0.03] |

The extra prompt fields produced no measurable gain with the saved checkpoints. The simulator replay still shows that the exact direction can matter. Supplying the vector and context at inference also changes the prompt format these checkpoints were trained on. A controlled training follow-up would use the explicit direction during training and evaluation and hold forward supervision fixed.

Both diagnostic runs saved their raw predictions and bootstrap summaries alongside their evaluation outputs.
