# V9 bounded-residual KAN-HGAT seed-87 screen

Frozen on 2026-09-11 before any V9 result was generated. This candidate was
designed after the V8 test results were revealed, so it is a second development
experiment rather than independent confirmation.

## Single architectural change

Keep HGAT-Lite's original shared scalar linear projection and add one bounded,
zero-initialized shared B-spline residual:

`phi(x) = W*x + 0.1*tanh(sum_k c_k*B_k(x))`.

Grid size 5, spline order 3, relation rank 8. The spline coefficients start at
zero, so the network is exactly the linear V4-HG model at initialization. The
correction is bounded to 0.1 per relation coordinate. This adds 64 trainable
parameters; all training, scoring, dataset, threshold, and rank-safe settings
remain unchanged.

## Screen and stopping rule

- Development seed: 87 on Exathlon, PSM, SMD, and SWaT; retain all four.
- Structural requirements: 48+ tests pass, exact linear nesting at
  initialization, finite artifacts, exactly 64 added parameters, learned spline
  norm above `1e-6`, zero distinct V4-rank inversions, and non-empty final tie
  refinement on at least three datasets.
- HGAT-only AUC-PR requirements versus frozen linear V4-HG: macro delta >= 0,
  at least two datasets have positive delta, and no dataset delta < -0.02.
- POT+PA and Best-F1+PA must retain V4 within `1e-4`. These are compatibility
  checks and are not evidence of a KAN gain.
- Only if every requirement passes may seeds 90/98 be run unchanged. Otherwise
  retain V9 as a negative ablation and stop this architecture axis.

No V9 hyperparameter was selected using test labels. Because the same four test
sets have already been inspected in earlier development, even a passing result
would require untouched data or an external preregistered replication before a
confirmatory claim.
