# Training-only fused-score recalibration (frozen before evaluation)

Frozen: 2026-09-10, after the seed-87 four-dataset HGAT-Lite run and before
computing any recalibrated test metric.

## Motivation and fixed transformation

Both the causal innovation score and neural model score are calibrated against
normal training data, but their weighted mixture is not itself uniform on the
training distribution.  This makes the shared `-log(1-u)` POT transform less
comparable after fusion.

Keep the already frozen fusion weight `w=0.001`, form

`z = (1-w) * innovation_score + w * model_score`,

then replace both train and test `z` by its right-sided empirical CDF computed
only from the normal training `z`.  No label, validation anomaly, or test score
is used to fit the recalibration.  All POT levels, multipliers, `q=1e-5`, shared
tail margin 1.04, PA logic, and entity boundaries remain unchanged.

## Decision rule

Evaluate all four datasets once without a weight or parameter sweep.  Promote
the transform only if:

1. POT+PA macro-F1 is no lower than the unrecalibrated HGAT-Lite result;
2. no dataset loses more than 0.001 POT+PA F1;
3. SWaT POT+PA exceeds the stored TimesNet value 0.8215;
4. AUC-ROC, AUC-PR, and Best-F1+PA remain unchanged up to ECDF tie effects.

Replacing the formal V4 row requires the stronger condition that POT+PA macro
also reaches the formal V4 macro 0.9022703904.

## Frozen-run outcome

The training-only ECDF raised SWaT POT+PA from 0.820116 to 0.832272 and raised
POT+PA macro from 0.899405 to 0.902163.  It exceeded the stored external
baseline on all four datasets under both PA protocols.  However, SMD declined
by 0.001056 (slightly beyond guard 2), and POT+PA macro remained 0.000107 below
formal V4.  It is therefore retained as a baseline-SOTA candidate but does not
replace the formal V4 row.  Exact results are in
`all_datasets_s87_train_ecdf.json`.
