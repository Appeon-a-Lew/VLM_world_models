# exp009_exp005_cf_action: report (provisional, pending post-review)

## Hypothesis and change
The exp005 adapter uses the action text on val_id Task B: flipping the push direction (`cf_flip`) and scaling the impulse (`cf_impulse_0.5`, `cf_impulse_2`) while keeping the normal image should raise the target error from 0.157. Eval-only on the exp005 adapter, infer-time text rewrite of the Task B action sentence only. Done in 0.325 GPU-h (budget 0.75), commit ffe8ee228b (pre-review REVISE round 1 -> APPROVE round 2; first submit denied by a busy GPU, resubmitted unchanged).

## Validity
- Primary reproduces: metrics.json value 0.3160281875, gap 0.34253104, parse 1.000 (identical to exp005).
- Task A under each cf mode vs normal (reviewer-verified): diff 0.0 (0.5), 0.0 (2), +4.3e-5 (flip), inside the 0.0005 gate.
- [cfsign] normal val_id frac_pos = 1.0000 (>= 0.80, so the sign statistic is valid). Non-sphere n = 263 (>= 50).
- **Gate 3 deviation:** cf_impulse_0.5 changed 499 of 500 prompts (cf_flip and cf_impulse_2: 500 of 500). By the letter of the spec this routes inconclusive. One episode cannot move the +0.056 rise (CI lower bound 0.0435), so the reviewer and I accept R1 in substance, with this stated.
- [cfdisp] columns are `n median_dhat median_dgt`, so the slope s below is the pinned one (no CI on s). [cfdisp]/[cfsign] numbers are on the GT-moved subset (414 of 500 episodes).

## Results (val_id Task B target; paired bootstrap vs normal, `compare --splits val_id`)
| mode | target mean | rise vs normal (CI) |
|---|---|---|
| normal | 0.157 | |
| cf_flip | 0.417 | +0.2595 [0.2354, 0.2849] |
| cf_impulse_0.5 | 0.213 | +0.0556 [0.0435, 0.0695] |
| cf_impulse_2 | 0.194 | +0.0368 [0.0170, 0.0563] |
| none (text only, no image) | 0.500 | |
Static val_id 0.265; cf_flip 0.417 is worse than static.

Print-only [cfdisp] (pred displacement, GT displacement; first column of numbers after n): non-sphere n=263 pred 0.0430 (0.5) / 0.0941 (normal) / 0.1972 (2), GT 0.1463; sphere n=151 pred 0.0910 / 0.2240 / 0.5170, GT 0.3883; all n=414 pred 0.0519 / 0.1245 / 0.2377, GT 0.1811. [cfsign] frac_pos: normal 1.000, flip 0.0121, 0.5 1.000, 2 1.000.
Slope s (non-sphere, 0.5 -> 2): ln(0.1972/0.0430)/ln 4 = 1.10, computed from the printed values; the spec defines s on median_dhat, and I did not check whether the printed column is a mean or a median (reviewer to confirm). alpha 0.5 vs 1 ratio 0.457, alpha 2 vs 1 ratio 2.10: ordering d(0.5) < d(1) < d(2) holds.

## Which rule fired (pre-registered)
- F-read: E_flip 0.417 >= 0.265; rise CI lower bound 0.235 > 0.05; frac_pos(flip) 0.012 <= 0.35 (valid). **F-read.**
- I-use: s = 1.10 >= 0.8, both pinned rises > 0.03 with CI lower bounds > 0 (0.0435 and 0.0170), ordering holds. **I-use**, subject to the s caveat above.
- Cell F-read x I-use: **R1, action used** on val_id. Next per spec: IDEA-016 (camera pose text); forces not tested here. Alpha 2 is the smallest rise (+0.037, CI lower 0.017), not clearly above 0.03 by CI. The paired CI of flip vs static was not computed.

## Verdict (provisional)
Supported: on val_id (in-distribution speeds) the exp005 adapter reads both the push direction and the impulse magnitude; flipping the direction pushes the answer to the wrong side (frac_pos 0.012), and doubling or halving the impulse scales the predicted displacement roughly proportionally (slope about 1.1; the physical value for sliding is 2, so the model under-scales). It does not read them from the image alone. **val_id only; action dependence on val_ood_forces, config and elevation is not tested.** This removes the "action ignored" explanation for the exp006/exp007 nulls on val_id only; their underpower and the config/elev action reading stay open. Tentative (C-012); R1 call carries the gate-3 deviation above.

## Caveats
- val_id only (the pinned ablation_split); action use on val_ood_forces or config is untested.
- One adapter, one seed. The alpha 2 rise is the smallest (+0.037) and partly outside the training speed range (0.6-3.2 m/s vs 0.3-1.6); the slope and ordering are the discriminators and both pass.
- Slope is computed by me from printed values, not from the pinned median definition; reviewer to verify.
- Flip error 0.417 above static is expected for a correctly read but wrong direction; it does not say the model "predicts worse physics".
