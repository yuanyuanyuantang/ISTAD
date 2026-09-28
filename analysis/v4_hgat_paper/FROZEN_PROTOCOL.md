# V4-HG paper experiment lock

Frozen on 2026-09-10 after the V4-HG three-seed development runs.

- Architecture and scoring command: `code/ISTAD/scripts/anomaly_detection/ISTAD_v4_hgat_integrated.sh`.
- Datasets: Exathlon, PSM, SMD, SWaT; no additional tuning on their test labels.
- Seeds: 87, 90, 98; all completed runs are retained.
- Primary non-oracle evidence: ROC-AUC, AUC-PR, train-derived POT raw F1, and overlap Event-F1.
- Compatibility evidence: POT+PA.
- Oracle upper bounds: exact Best raw F1 and Best-F1+PA; both must be labelled as test-label oracles.
- Component arms are computed from the same frozen artifact: V4-HG, V4 innovation only, and HGAT-routed evidence only.
- The learned rank-safe refinement is compared with 100 deterministic random tie refinements per run.
- All statistical score arrays must be stored as float64 because the rank-safe correction is smaller than float32 resolution near one.
- Efficiency uses seed-87 checkpoints, 30 warm-up passes and 100 measured passes on one named device. It excludes data loading and closed-form VAR fitting.

This lock is a retrospective development lock, not a preregistered independent confirmation. The four benchmark test sets had already been inspected.
