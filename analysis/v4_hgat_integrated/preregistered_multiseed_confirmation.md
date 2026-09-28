# V4-HG rank-safe multi-seed confirmation

Frozen after seed 87 satisfied the rank-safe decision rule and before training
seeds 90 or 98.

- Confirmatory development seeds: `90`, `98` on EXATHLON, PSM, SMD, SWAT.
- Configuration and scoring are byte-for-byte those in
  `ISTAD_v4_hgat_integrated.sh`; no dataset-specific score or threshold change.
- Primary reports: three-seed mean and population standard deviation for
  POT + point-adjust and Best-F1 + point-adjust.
- Structural reports: train-selected HGAT kill-switch, rank-safe epsilon,
  neural parameter count, and separately dumped V4/HGAT/fused scores.

Pass conditions:

1. Three-seed mean POT + PA macro-F1 is no lower than V4 `0.902270` beyond
   `1e-4` tolerance.
2. Three-seed mean Best-F1 + PA macro-F1 is no lower than V4 `0.945107` beyond
   `1e-4` tolerance.
3. No seed triggers a malformed/non-finite artifact.
4. HGAT changes tied-score ordering on at least one dataset/seed while the
   rank-safety invariant holds for all distinct V4 ranks.

The same four test sets have already been inspected during model development;
therefore this measures seed stability, not independent generalization.
