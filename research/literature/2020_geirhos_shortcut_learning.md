# Shortcut Learning in Deep Neural Networks

**Citation**: Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. Nature Machine Intelligence (perspective article; year/volume not shown on the arXiv page, unverified); arXiv:2004.07780 (v5, Nov 2023), https://arxiv.org/abs/2004.07780. **Read on**: 2026-10-25.
**Reading depth**: abstract page plus the ar5iv HTML via WebFetch summariser (text reported as complete through conclusion and appendix). It is a conceptual perspective; there are almost no benchmark numbers, and the one toy experiment below is the only quantitative item I extracted.

## TL;DR
Shortcuts are decision rules that win on i.i.d. test data but fail on o.o.d. tests, so i.i.d. success cannot be credited as the intended ability. The paper gives rules for testing: a clear shift, a well-defined intended solution, and a test models actually struggle with.

## Setup / method
Perspective, no new model. A taxonomy of decision rules: those that solve the training set, those that also pass the i.i.d. test, and the intended solution that also passes o.o.d. tests. Toy example (Appendix A): star vs moon with location as the shortcut, 4,000 train images, 1,000 test per class with randomised positions.

## Key results
- Toy: a three-layer MLP reached 100% train and about chance (51.0%) on the randomised-position test; a CNN reached 100% on both.
- Four conclusions: shortcut learning is ubiquitous; credit high-level abilities only after ruling out shortcuts ("Never attribute to high-level abilities that which can be adequately explained by shortcut learning"); test o.o.d., since i.i.d. testing "is insufficient to distinguish" intended from shortcut solutions; study what makes a solution easy (architecture, data, loss, optimiser).
- Good o.o.d. test: clear distribution shift, well-defined intended solution, a test most current models struggle with. Box I lists benchmarks, including cue-conflict stimuli and controlled-correlation sets.
- No accuracy figures for the cows-on-grass and X-ray examples in the text I saw.

## Relevance to us
- Frames the image-dependence test: exp000's val_id Task B (0.155 vs static 0.265) is an i.i.d. score; val_ood_* are the o.o.d. tests. Our three OOD splits meet the "clear shift" criterion; the "well-defined intended solution" criterion is met for Task A (GT centre) but weaker for Task B on val_ood_forces, where the intended use of the impulse is unverified (IDEA-008, 025, 026).
- Our known shortcut candidates: per-object training-typical displacement (prior) and shape/colour identity. exp000's none/blank/swap ablations show image reliance, but a swap changes identity, count and positions together, so they cannot say which cue is read. The cue-conflict idea fits this (IDEA-048).
- val_ood_config changes three factors at once (elevation, count, corner); the paper's stress on controlled single-cue shifts supports exp003's single-factor diag splits.
- No bearing on the Qwen-specific zero-shot anomaly or on Task A pixel-to-metre geometry.

## Ideas for us
- IDEA-048: identity-matched swap (same object multiset, different positions) to split the image-dependence gap into position vs identity reliance. Cost: small render job plus about 0.2 GPU-h. Appended to BACKLOG.

## Caveats / disagreements
- Conceptual, mostly classification and NLP; no physics or regression setting, no VLM evidence.
- "Easiest solution" arguments are qualitative; they do not predict which shortcut our model uses.
- Our model fine-tunes a pretrained VLM with LoRA, a case the paper does not discuss.
- The Nature MI year is not shown on the page I read; do not cite it without checking.
