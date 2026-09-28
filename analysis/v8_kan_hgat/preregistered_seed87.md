# V8 shared-KAN HGAT seed-87 screen

Frozen on 2026-09-11 before any V8 test result was generated.

## Single architectural change

Replace HGAT-Lite's shared scalar `Linear(1, rank)` projection with a shared
KAN edge function using the repository's existing cubic B-spline basis:

`phi_r(x) = a_r SiLU(x) + sum_k c_{r,k} B_k(x) + b_r`.

Grid size 5, spline order 3, relation rank 8, all other V4-HG training,
scoring, dataset, threshold, and rank-safe settings remain unchanged. The KAN
projection adds exactly 72 trainable parameters over the linear projection.

## Screen and stopping rule

- Development seed: 87 on Exathlon, PSM, SMD, and SWaT, with all four retained.
- Structural requirements: finite artifacts, zero distinct V4-rank inversions,
  non-empty tie refinement, and no more than 128 added parameters.
- Accuracy diagnostic: compare the HGAT-only ROC-AUC/AUC-PR with the frozen
  linear HGAT-only branch. Promotion requires mean AUC-PR not to decrease by
  more than 0.005 and at least two datasets to improve AUC-PR by 0.005.
- POT+PA and Best-F1+PA must retain V4 within 1e-4, as guaranteed by rank-safe
  fusion. These PA metrics are compatibility checks, not evidence for KAN.
- If the accuracy diagnostic fails, keep V8 as a negative ablation and do not
  run seeds 90/98 or tune the spline on these revealed test labels.

These four test sets were already inspected; this is a development screen, not
independent confirmation or current-SOTA evidence.
