# V5 NASA neural-branch diagnostic (seed 87)

Frozen: 2026-09-10, before training or scoring either neural arm.

## Status and scope

MSL/SMAP labels were already revealed by the negative pure-V4 confirmation, so
this is a **development diagnostic**, not independent confirmation.  Its purpose
is to decide whether the existing compact neural representation is worth taking
to a multi-seed V5 stage.  No result from this run can restore an unbiased SOTA
claim; a later frozen model still requires a new untouched confirmation target.

## Fixed data protocol

- Exact official `thuml/Time-Series-Library` arrays, verified by SHA-256.
- Window/score step: 100/100.
- Entity-aware processing: 27 MSL and 53 SMAP sequences.  Windows, lag pairs,
  point adjustment, and the chronological split cannot cross an entity.
- Per entity, first 80% of normal training data is used for fitting and final
  20% for validation.  `StandardScaler` is fitted on the concatenated fitting
  portions only.
- Seed: 87.  Test loss is not inspected during training.

## Fixed arms and recipe

The only arm difference is the spatial branch:

1. `none`: compact KAN-TCN plus the direct causal-convolution path;
2. `lite`: the same network plus rank-16 HGAT-Lite.

Shared recipe: KANAD order 4, KAN grid 10/order 3, dropout 0.2, batch 128,
three epochs, Adam learning rate 0.01 with `type1` schedule, normal-data
synthetic denoising (four frozen corruption families), evidence-head weight
0.2, and no RevIN/dual view.  The existing pure-innovation fusion weight 0.001
and training-only fusion ECDF are retained only as a compatibility diagnostic;
they are not expected to rescue weak neural ranking.

## Metrics and decision rules

Primary neural-branch metrics: ROC-AUC, PR-AUC, train-p99 raw F1, and oracle best
raw F1.  Best-F1+PA and innovation-fused metrics are secondary diagnostics.

Literature-context targets (not controlled head-to-head gates):

| Dataset | ROC-AUC | PR-AUC | PA-F1 |
|---|---:|---:|---:|
| MSL | 0.7314 | 0.2186 | 0.9513 |
| SMAP | 0.6224 | 0.1912 | 0.9732 |

- Continue to seeds 90/98 only if one arm beats pure V4 on both ROC and AP for
  both datasets, and AP exceeds anomaly prevalence on both.
- Attribute a benefit to HGAT-Lite only if it improves at least one primary
  ranking metric by 0.01 on at least one dataset, while no ROC/AP metric falls
  by more than 0.01 relative to `none`.
- A one-seed pass creates a candidate, never a final model.
- If neither arm passes, stop this recipe.  The next V5 iteration must change
  the training objective/representation rather than tune PA, POT, fusion weight,
  window length, or test-label thresholds.

## Commands

```bash
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v5_nasa_diagnostic.sh MSL lite 0 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v5_nasa_diagnostic.sh MSL none 1 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v5_nasa_diagnostic.sh SMAP lite 0 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v5_nasa_diagnostic.sh SMAP none 1 87
```

## Frozen-plan outcome (recorded after all four runs)

All four seed-87 arms completed.  The evaluator and the complete machine-readable
output are `analysis/evaluate_v5_nasa_diagnostic.py` and
`analysis/v5_nasa_diagnostic/seed87.json`.

| Dataset | Arm | Neural ROC-AUC | Neural PR-AUC | Train-p99 raw F1 | Best raw F1 | Best-F1+PA |
|---|---|---:|---:|---:|---:|---:|
| MSL | none | **0.6216** | **0.1544** | **0.1089** | **0.2325** | 0.7832 |
| MSL | lite | 0.5492 | 0.1299 | 0.0578 | 0.1986 | **0.8033** |
| SMAP | none | **0.4816** | **0.1202** | 0.0117 | **0.2296** | 0.6874 |
| SMAP | lite | 0.3872 | 0.1083 | **0.0272** | 0.2262 | **0.6907** |

Neither arm passes the continuation gate.  In particular, SMAP neural ROC is
below 0.5 and PR-AUC is below anomaly prevalence (0.1277) for both arms.  Lite
also reduces ROC/AP relative to `none` by 0.0725/0.0245 on MSL and
0.0944/0.0119 on SMAP, so the HGAT-Lite benefit gate fails.

**Decision:** do not run seeds 90/98 and do not promote either arm.  This recipe
is closed as specified above.  The next model iteration must change the learned
objective or representation; PA/POT, fusion-weight, window-length, or
test-threshold tuning is not an admissible response to this result.
