# V4-HG frozen seed-87 screen

Frozen before reading any V4-HG test metric.

## Scope

- Datasets remain exactly `EXATHLON`, `PSM`, `SMD`, and `SWAT`.
- The development screen uses seed `87` on every dataset.
- Data splits, entity boundaries, point adjustment, POT/SPOT implementation,
  and adaptive Best-F1 search are unchanged from the frozen V4 protocol.
- Test labels are evaluation-only. They do not fit the model, calibration, or
  the HGAT reliability gate.

## Fixed model

- V4 branch: lag-1 ridge VAR (`ridge=0.01`) and the unchanged train-only
  sparse/dense innovation pooling rule.
- Neural branch: causal convolution, HGAT-Lite only (`rank=8`, top-20% nodes
  per edge; the existing SWAT recipe remains 20 edges/top-10), and a shared
  32-unit pointwise reconstruction decoder.
- Removed from this candidate: TCN/KAN-TCN, GRU, KANAD, RevIN, dual heads,
  and the learned anomaly-evidence head.
- Training is normal-only clean reconstruction plus the existing synthetic
  denoising task. Synthetic masks are not part of test scoring.

## Fixed score

1. V4 produces per-variable causal innovation evidence `e_t`.
2. HGAT-Lite produces a dynamic sparse incidence matrix `H_t`.
3. Hyperedge evidence is `g_t = pool_edges(H_t^T e_t)` using the same
   train-selected sparse/dense regime as V4.
4. Both V4 point evidence and hyperedge evidence are calibrated with normal
   training data only.
5. Maximum HGAT weight is fixed at `0.20`. The first 80% / last 20% split of
   each normal-training entity checks 1% tail stability. If HGAT tail inflation
   exceeds 1.25 times both the target rate and V4's observed rate, its weight
   is set to zero. A zero weight returns the exact original V4 score.
6. A non-zero mixture receives one final normal-training ECDF calibration.

No score equation or constant above may change after viewing seed-87 metrics.

## Decision rule

The candidate advances to seeds `90` and `98` only if all conditions hold:

- POT + point-adjust macro-F1 is at least frozen V4 (`0.902270`).
- Best-F1 + point-adjust macro-F1 is at least frozen V4 (`0.9451`, rounded
  published table target).
- At least two datasets select a non-zero HGAT weight; otherwise the candidate
  has not demonstrated that HGAT contributes.
- Every score artifact contains separately reproducible V4, HGAT, and fused
  components plus the train-only gate metadata.

Failure is recorded as a negative result; it is not repaired by dataset-wise
test-label tuning.
