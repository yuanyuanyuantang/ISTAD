# V10 innovation-guided BR-KAN-HGAT frozen screen

Frozen before the first V10 test run on 2026-09-11.  The four test sets have
already been inspected by earlier versions, so this is a development screen,
not independent confirmation.

## Fixed method

- Fit the existing ridge VAR(1) and residual scales on normal training points.
- Keep the V4 magnitude evidence and automatic sparse/dense pooling unchanged.
- Feed signed, train-standardized VAR innovations to the existing causal Conv,
  bounded-residual KAN projection and rank-8 HGAT-Lite.
- Train only on normal innovations with the existing four-family denoising loss.
- Pool V4 feature evidence through the learned incidence, calibrate both branches
  with normal-train ECDF, and use the existing normal-tail reliability mixture
  with maximum graph weight 0.20 and exact V4 fallback.
- No test sample or label may fit VAR, scaling, incidence, calibration or fusion.

## Seed-87 continuation gate

Run EXATHLON, PSM, SMD and SWAT once with seed 87. Continue to seeds 90 and 98
only if all structural checks pass and the following development criteria hold:

1. Macro unadjusted AP improves by at least 0.005 over frozen V4-HG.
2. At least three of four datasets have non-negative AP change.
3. No dataset loses more than 0.003 AP.
4. POT+PA and Best-F1+PA Macro do not fall more than 0.001 below V4.

If the gate fails, report the result and retain V4-HG as the paper candidate;
do not tune V10 on these test labels.
