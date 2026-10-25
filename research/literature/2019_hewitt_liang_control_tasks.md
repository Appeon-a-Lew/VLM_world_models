# Designing and Interpreting Probes with Control Tasks

**Citation**: John Hewitt, Percy Liang. EMNLP 2019 (per arXiv comments field). arXiv:1909.03368, https://arxiv.org/abs/1909.03368 (full text: https://ar5iv.labs.arxiv.org/html/1909.03368, ACL: https://aclanthology.org/D19-1275). **Read on**: 2026-10-25.
**Reading depth**: abstract page plus the ar5iv HTML digest (WebFetch summariser output; the PDF itself came back as binary and could not be read). Numbers below come from that digest, not from my own reading of the tables. Spot-check against the PDF before citing them in a report.

## TL;DR
A probe's accuracy does not show that the representation encodes a property, because a flexible probe can learn the task itself. They add a control task (random output per word type) and define selectivity = task accuracy - control accuracy; popular MLP probes on ELMo are not selective, linear ones are.

## Setup / method
- Input: frozen ELMo layer 1 or 2 representations of Penn Treebank words. Output: POS tag (45 tags) or dependency edge. The tasks are POS tagging and dependency edge prediction (not NER).
- Supervision: supervised probe on top of frozen features.
- Control task: each word type gets a random output sampled independently. Every token of that type gets the same output regardless of context. The mapping must be memorised, so only the probe can learn it, not the representation.
- Probes compared: linear/bilinear, MLP-1, MLP-2, with sweeps over hidden size, dropout, weight decay, early stopping and training set size.

## Key results
ELMo layer 1, PTB dev, default settings (digest):

| Probe | POS acc | POS control | POS selectivity | Dep acc | Dep control | Dep selectivity |
|---|---|---|---|---|---|---|
| Linear / bilinear | 97.2 | 71.2 | 26.0 | 89.0 | 82.4 | 6.6 |
| MLP-1 | 97.3 | 92.8 | 4.5 | 92.3 | 93.0 | -0.7 |
| MLP-2 | 97.3 | 93.2 | 4.2 | 93.9 | 92.0 | 1.9 |

- Linear matches MLP accuracy on POS and is far more selective. MLPs stay more accurate on the harder dependency task.
- Shrinking the MLP hidden state (about 10 units for POS, 50 for dependencies) raised selectivity with little accuracy loss.
- Dropout was unreliable (MLP-2 POS selectivity 4.2 to 3.4 in one setting). Weight decay helped for some values but a setting keeping both accuracy and selectivity high was hard to find. Early stopping mattered for MLP dependency probes only. Fewer training examples helped POS selectivity but hurt dependency accuracy.
- Layer comparison: linear POS accuracy ELMo1 97.2 vs ELMo2 96.6, but selectivity 26.0 vs 31.4. The authors argue ELMo1's accuracy edge may be easier access to word identity. Exception: the bilinear dependency probe was slightly less selective on ELMo2 (6.2 vs 6.7).
- Recommendations: report selectivity next to accuracy, constrain probe size rather than rely on dropout, prefer linear probes, do not compare layers on accuracy alone.

## Relevance to us
- Bears on IDEA-030 (linear-probe read-out of exp000's visual tokens, to say whether the val_ood_config failure sits in the representation or in the LoRA readout). The earlier note on Ballout et al. already added a shuffled-target control; this paper says what that control should be (type-keyed) and why layers should be compared by selectivity (Task A: which layer holds geometry; the config split is where raw accuracy could mislead).
- Our probe has an extra risk this paper highlights: object type (shape, colour) is visible in the token and in the prompt, so a probe could memorise per-type centres. Selectivity is exactly the guard.
- Image-dependence gap: indirect only. It is a probe-level analogue of the supervisor's "text prior vs reading the scene" concern, but it does not test the VLM's behaviour.
- Does not bear on Task B dynamics or the impulse question.

## Ideas for us
- IDEA-047: type-keyed control target plus selectivity (linear ridge primary) added to IDEA-030's protocol. Expected effect: makes a "probe error stays at id level on config" result interpretable; cost about 0.1 day on top of IDEA-030, no extra GPU. Appended to BACKLOG.

## Caveats / disagreements
- Text-token classification with discrete outputs; ours is continuous regression in metres, so accuracy-based selectivity has no direct unit and the control must be defined by analogy (a fixed random xyz per type).
- The control isolates memorisation of the input-to-label lookup. It does not rule out that the probe reads a confound that correlates with geometry (e.g. image position of the patch). A patch-position-only baseline would be a separate control, and it is not in this paper.
- ELMo, not a ViT/LLM hybrid. The layer result (more selective, less accurate) is about one pair of layers and one exception exists.
- Numbers come from a digest of the HTML, see above.
