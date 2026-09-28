# V4-HG rank-safe refinement, frozen seed-87 re-score

Frozen after the failure of the separately recorded 0.20 reliability-mixture
candidate and before reading any rank-safe score metric.

The model checkpoints, four datasets, splits, V4 innovation branch, HGAT-Lite
branch, POT settings, and Best-F1 search are unchanged. No model is retrained.

## Single change

Let `q_t` be the V4 normal-train ECDF score, `g_t` the HGAT hyperedge evidence
after its own normal-train ECDF, and `n` the V4 calibration size. Define

`epsilon = 0.5 / (n + 1)`

`score_t = (q_t + epsilon * g_t) / (1 + epsilon)`.

The reliability check from candidate 1 remains solely as a kill-switch. If it
fails, `score_t = q_t` exactly. There is no final ECDF recalibration.

Because adjacent distinct V4 empirical ranks differ by `1/(n+1)` and
`0 <= g_t <= 1`, the HGAT correction is strictly smaller than one rank step.
It may order V4 ties (including saturated tail ties) but cannot invert distinct
V4 training ranks. `epsilon` is data-size-derived, shared in form by all
datasets, and is not tuned against test labels.

## Decision rule

- POT + PA and Best-F1 + PA macro-F1 must each be no lower than V4 beyond
  `1e-4` numerical tolerance.
- At least one dataset must change a tied-score ordering or metric; otherwise
  HGAT is computationally present but empirically vacuous.
- Seeds 90 and 98 run only after the seed-87 four-dataset screen passes.

As with all previously inspected benchmarks in this repository, this is a
development result, not an untouched independent confirmation.
