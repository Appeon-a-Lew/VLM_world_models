# Thinking in Space: How Multimodal LLMs See, Remember, and Recall Spaces (VSI-Bench)

## Citation
Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, Saining Xie. arXiv:2412.14171 (v1 18 Dec 2024, v2 2 Jul 2025). The arXiv page lists no conference venue (venue unverified; some sources say CVPR 2025, not checked here).
URL: https://arxiv.org/abs/2412.14171 (HTML read: https://arxiv.org/html/2412.14171v2)
**Read on**: 2026-10-24. Read the abstract page and the HTML text (main text plus the appendix blind tables). The PDF could not be parsed. The HTML was delivered through a summariser: the last ~13k characters of the first pass were re-read via an offset call. The per-task blind tables (Tables 12-13) came back with an inconsistent "Avg." column (see Caveats); I use only the per-task cells and do not trust the Avg. column.

## TL;DR
VSI-Bench (>5,000 QA, 288 indoor videos) shows MLLMs are well below humans on metric spatial questions. Its blind (vision-disabled) and frequency-prior baselines are the standard protocol for checking whether a spatial score comes from the image or from priors, and its MRA metric is the standard relative-error score for numeric answers.

## Setup / method
- Input: video of an indoor scene (ScanNet, ScanNet++, ARKitScenes validation sets) plus a text question. Output: multiple-choice letter (4 tasks: relative distance, relative direction, route plan, appearance order) or a number (4 tasks: object count, absolute distance, object size, room size). Supervision: none. It is an evaluation-only benchmark, no fine-tuning experiments are reported (the paper lists task-specific fine-tuning as future work).
- Metric for numeric answers, MRA (Mean Relative Accuracy): a prediction y_hat counts as correct at threshold theta if |y_hat - y| / y < 1 - theta, averaged over theta in {0.50, 0.55, ..., 0.95}. Multiple choice: exact-match accuracy.
- Baselines: Chance (Random), Chance (Frequency) (answer with the most frequent training/test answer per task), a human subset, and "Vision Enabled vs Vision Disabled" (blind) evaluation of six strong models (3 open, 3 closed) in the main text, 15 models in the appendix.
- Diagnostics: self-explanation error analysis (163 wrong answers, hand-labelled), CoT / self-consistency / tree-of-thoughts, and "cognitive maps" (model emits object centres on a 10x10 grid before answering).

## Key results
- Chance (Frequency) average 34.0; human (tiny subset) 79.2; Gemini-1.5 Pro 45.4 (full) / 48.8 (tiny). Humans are at 94-100 on configurational and appearance-order tasks, but only 47.0 abs distance and 45.9 room size (humans are also poor at metric estimation). "7 of 12 open-source models fall below chance-level baselines" (paper's wording).
- Blind evaluation: vision-on minus vision-off is consistently positive on average, blind models are generally below the frequency baseline. Absolute distance, route plan and relative direction stay near chance with or without vision. Object size is well above chance even blind (common-sense size priors). Blind LongVILA-8B gets 47.5 on object count mostly by answering "2" (a frequency-prior exploit). Appendix Table 12 per-task blind cells, e.g. Gemini-1.5 Pro blind: count 32.3, abs dist 30.6, obj size 11.5, room size 51.5. Vision-enabled minus disabled on abs distance: GPT-4o +46.1, LLaVA-Video-72B +29.9, LLaVA-OneVision-72B +35.4 (Table 13), whereas on object size the gain is about 0 for most models.
- Reasoning: zero-shot CoT about 4 points below baseline, tree-of-thoughts about 4 below, self-consistency 1.1 below. Room size / object size drop 8-21 points. Text reasoning hurts metric tasks.
- Error analysis: about 71% of 163 wrong answers are attributed to spatial reasoning (not perception, language, or ego/allo transformation).
- Cognitive maps (Gemini-1.5 Pro, relative distance): 46.0 without map, 56.0 with a self-generated map (+10), 66.0 with the ground-truth map (+20). Map accuracy falls with distance (0.64 nearest bin to 0.00 farthest), read as "local world models". For LLaVA-Video-7B the self-map reduced accuracy (40.0 to 32.0).
- Format effects: video-before-question 48.8 vs question-first 46.3; repeating the video 48.8 to 50.9.

## Relevance to us
- Image-dependence test (supervisor's concern). Their blind protocol removes the image at test time on off-the-shelf models and compares to a frequency baseline. Our exp000 none/blank/swap ablations do the same for an SFT'd model, and are all worse than static (0.486/0.472/0.400 vs 0.265, C-001). The paper shows what we do not yet have: a frequency-prior baseline fit to the answer distribution (here the "answer with the training-typical value"), against which "the image helps" should be measured. exp000's blind-at-test numbers conflate "no prior knowledge" with "adapter was trained to expect image tokens". Hence two cleaner references: a non-learned prior fit on the training answers (IDEA-040) and a blind-trained control arm (IDEA-041).
- It also gives a calibration for how much text priors can buy. Object size is above chance blind. In our setting the prompt names the shape and gives K, so size/height priors and a typical-table-centre prior can already give a non-trivial Task A score without the image. Task B in forces: a "training-typical displacement" prior is IDEA-008 (already in backlog); this paper supports making it an explicit blind row in the leaderboard.
- Zero-shot exp002 anomaly (swap beats normal on val_id): Thinking in Space notes blind models can score at or above vision-on on some tasks (negative enabled-minus-disabled cells, e.g. LLaVA-OneVision-7B relative distance -13.9, LongVILA abs distance -18.2). So a negative image gain on a single task is not unheard of for weak models; it supports the "hedging toward a default" mechanism over a frame flip (IDEA-024/028/032), but it does not decide it.
- Chain-of-thought hurts metric estimation (Fig. 9): do not spend effort on a text-reasoning-before-answer variant of Task B. The cognitive map result (+10 self-generated, +20 GT) is the analogue of our Task A-then-Task B ordering (IDEA-002/033): an explicit intermediate map helps only when it is accurate. That fits the exp000 finding that Task A perception fails on val_ood_config (0.136), the same split where Task B dynamics gain drops.
- MRA: relative error. Our errors are absolute metres at ~1 m scene scale, so a relative metric is not needed; but the thresholded form is a cheap robust companion to the mean (the balls leaving the table skew means, see STATE Known facts).

## Ideas for us
- **IDEA-040** (non-GPU): fit a training-answer prior baseline (Chance-Frequency analogue) and add it to every leaderboard row; see BACKLOG.
- **IDEA-041** (1 re-train, ~1.4 GPU-h): blind-trained control arm: same exp000 recipe with the image removed in training and test; define the image-dependence gap against it. See BACKLOG.

## Caveats / disagreements
- Setting: video, real indoor scans, MC and room-scale metric answers; ours is one synthetic image, centres in metres, and fine-tuned. Their baselines are off-the-shelf models; the blind protocol does not carry over unchanged to a fine-tuned adapter.
- Frequency baseline definition in the paper is per-task most-frequent answer; I did not find the exact construction for the numeric tasks in the text I read (unverified).
- The appendix Tables 12/13 as extracted show an "Avg." column that is inconsistent with the per-task cells (e.g. a negative or tiny average beside cells of 20-40). Use per-task cells only, and treat any average I did not quote as unreliable. The text says "8%" for LLaVA-Video-72B's map gain where the table gives 6 points, also a sign that the prose numbers are loose.
- Human comparison is on a tiny subset.
