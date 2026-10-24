# Depth Anything V2

## Citation
Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao. NeurIPS 2024; arXiv:2406.09414 (v1 Jun 2024, v2 Oct 2024).
URL: https://arxiv.org/abs/2406.09414 (HTML read: https://arxiv.org/html/2406.09414)
**Read on**: 2026-10-24. Read the abstract and the HTML text through a summariser (method, Tables 2-4, limitations). The PDF exceeded the fetch size limit. I did not read the appendix or the released-model cards. The mapping of DA-2K numbers to model names was inferred by the summariser from the text; marked below.

## TL;DR
A monocular depth model trained as: synthetic-only teacher (DINOv2-G), then pseudo-labelling 62M real images to train student models of 25M-1.3B parameters. It outputs relative (affine-invariant inverse) depth, with metric variants obtained by fine-tuning on synthetic indoor/outdoor data.

## Setup / method
- Input: one RGB image. Output: dense relative depth (affine-invariant inverse depth, i.e. disparity up to scale and shift) from the base models; metric-depth models are separate fine-tunes.
- Supervision: teacher trained only on 595K synthetic images (five datasets). It pseudo-labels 62M unlabeled real images from eight public datasets. Students are trained only on pseudo-labels (top 10% highest-loss regions ignored). Sizes follow DINOv2 S/B/L/G.
- Metric depth: fine-tuned with the ZoeDepth pipeline for the benchmark table; released indoor/outdoor metric models use Hypersim and Virtual KITTI 2 (synthetic). The paper gives only qualitative comparisons for the released metric models.

## Key results
- Zero-shot relative depth (AbsRel / delta1), DA V2 ViT-G vs DA V1 ViT-L vs MiDaS V3.1: KITTI 0.075/0.948 vs 0.076/0.947 vs 0.127/0.850; NYU-D 0.044/0.979 vs 0.043/0.981 vs 0.048/0.980; Sintel 0.506/0.772 vs 0.458/0.760 vs 0.587/0.699. V2 is not better than V1 on the standard benchmarks (the paper says so, arguing that they miss fine detail).
- DA-2K (the authors' new sparse relative-order benchmark): V2 ViT-S 95.3, B 97.0, L 97.1, G 97.4 percent; Marigold 86.8, GeoWizard 88.1, DepthFM 85.8, DA V1 88.5 (model-to-number mapping inferred from the text).
- Metric depth after fine-tuning (NYU-D delta1/AbsRel; KITTI delta1/AbsRel): ViT-S 0.961/0.073 and 0.973/0.053; ViT-B 0.977/0.063 and 0.979/0.048; ViT-L 0.984/0.056 and 0.983/0.045.
- Stated limitations: heavy compute for the 62M-image stage, synthetic data not diverse enough, DA-2K is sparse.

## Relevance to us
- It is the predicted-depth source used by SD-VLM (2025_sd_vlm.md) and the candidate RGB-only depth for IDEA-013/015/020/022. In our setting the camera extrinsics change across OOD splits (elevation 12-25 degrees vs train 30-60) and exp000 Task A perception drops on val_ood_config (0.136 vs 0.016-0.024). The decisive property for us is not benchmark rank but whether depth accuracy is stable across elevation and with 4-5 clustered objects (occlusion): nothing in this paper measures viewpoint robustness, so IDEA-022 stays a measurement, not an assumption.
- Relative vs metric: the base model gives disparity up to scale and shift. For an object-centre lookup to produce camera-frame metres we need the metric variant or a scale fit. The indoor metric variant was fine-tuned on Hypersim (synthetic indoor); our MuJoCo tabletop at ~1 m with 5-12 cm objects is a different range and rendering domain. The paper reports no numbers at this scale.
- A depth-from-RGB perception path would test whether the config failure is a ViT/LoRA readout problem or a missing geometry signal, without GT depth: the RGB-only counterpart of IDEA-015 (GT-depth lookup oracle).
- Not relevant to Task B dynamics directly; it only touches perception (Task A and the t=0 anchor).

## Ideas for us
- **IDEA-042**: estimated-depth variant of IDEA-015 (DA-V2 metric-indoor depth lookup at the GT 2D centre) as an RGB-only perception floor per split. See BACKLOG.

## Caveats / disagreements
- All benchmarks are natural indoor/outdoor images; none is a close-range tabletop or varying elevation. Expect scale errors on our renders (unverified).
- Relative-depth models need scale/shift recovery; do not feed raw disparity into a "metres" pipeline.
- I did not verify the exact released checkpoint names or the HF transformers class for the metric models (to be checked at implementation time).
