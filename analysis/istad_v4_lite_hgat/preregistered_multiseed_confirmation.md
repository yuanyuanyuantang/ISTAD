# HGAT-Lite + train-ECDF multi-seed confirmation

Frozen: 2026-09-10, after development seed 87 and before running seeds 90/98
on Exathlon, SMD, and SWaT or recalculating their metrics.

## Fixed candidate

- Architecture: Causal Conv + HGAT-Lite(rank=16) + KAN-TCN + KANAD.
- Score: `(1-0.001) * innovation_score + 0.001 * model_score`.
- Post-fusion calibration: right-sided ECDF fitted only on normal training
  fusion scores.
- Dataset training recipes, strict validation, entity handling, POT `q`, POT
  levels/multipliers, shared tail margin 1.04, and PA implementation are frozen.
- Development seed 87 is identified separately.  Seeds 90/98 are confirmation
  runs; no hyperparameter or threshold change is allowed after either result.

PSM seed-90/98 checkpoints already exist from the paired architecture ablation
under the identical neural recipe.  They will be rescored rather than retrained.

## Metrics and promotion gate

Report each seed and three-seed mean +/- population standard deviation for:

- POT+PA and Best-F1+PA of the fused score;
- AUC-ROC, AUC-PR, and train-p99 raw F1 of the neural `model_score`;
- parameter count and the already measured efficiency profile.

Promote the candidate to the paper configuration only if:

1. mean POT+PA exceeds the stored strongest external baseline on all datasets;
2. mean Best-F1+PA exceeds the stored strongest external baseline on all datasets;
3. at least two of three seeds exceed the corresponding external baseline for
   every dataset and PA protocol;
4. no run fails, produces a non-finite score, or crosses an SMD entity boundary;
5. mean POT+PA macro is at least the formal V4 macro 0.9022703904.

If gates 1-4 pass but gate 5 fails, retain it as an external-baseline SOTA
candidate rather than replacing formal V4.

## Frozen outcome

Completed on 2026-09-10 without changing the frozen candidate.

- Gates 1-4 passed.  Both PA protocols have a three-seed mean above every
  stored external baseline, every cell wins in at least two of three seeds,
  every run completed with finite scores, and all SMD runs retained 28-entity
  boundary-aware evaluation.
- Gate 5 failed: mean POT+PA macro is 0.9010051269568605 (population standard
  deviation 0.0015491118519494358), below formal V4 0.9022703904133883.
- Best-F1+PA macro is 0.9456452116926592 +/- 0.0002138982131650996.
- Decision: retain HGAT-Lite + train-ECDF as a multi-seed external-baseline
  SOTA candidate; do not replace formal V4.

Exact per-seed results and artifact hashes are stored in
`multiseed_confirmation_s87_s90_s98.json`.
