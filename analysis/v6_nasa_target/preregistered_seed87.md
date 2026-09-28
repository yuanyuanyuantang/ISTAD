# V6 NASA target-aware diagnostic (seed 87)

Frozen: 2026-09-10, before training or scoring either V6 arm.

## Status and semantic motivation

MSL/SMAP are already revealed development datasets, not independent confirmation.
The original Telemanom format defines column 0 as the telemetry value being
predicted, while the remaining 54 MSL or 24 SMAP inputs are one-hot command
context.  Its labels describe anomalies in that target telemetry stream.  V5
instead reconstructed and pooled all columns as if they were homogeneous sensor
outputs.  V6 corrects that task mismatch without consulting test labels.

Sources:

- Hundman et al., KDD 2018: https://arxiv.org/abs/1802.04431
- Original data/code documentation: https://github.com/khundman/telemanom

## Fixed data and training protocol

- Exact official arrays and the same entity-aware 80/20 normal split as V5.
- MSL/SMAP contain 27/53 independent entities; no split, window, score, or PA
  operation may cross an entity.
- Inputs retain all 55/25 dimensions, but the decoder has one output and the
  training loss and anomaly score use target column 0 only.
- Seed 87, window/evaluation step 100/100, three epochs, batch 128, Adam 0.01
  with `type1`, KAN-TCN plus direct path, KAN grid/order 10/3, KANAD order 4,
  dropout 0.2, and no spatial branch, RevIN, dual view, evidence head, or
  innovation fusion.
- Four normal-training corruption families remain, but corruption and denoising
  supervision are restricted to target column 0.
- Test loss is hidden during training.  No test label selects a checkpoint,
  threshold, score channel, or hyperparameter.

## Fixed arms

1. `trecon`: reconstruct target column 0 while the current target is visible;
2. `forecast`: replace the current target input with its one-step lag, keep the
   current command context, and predict target column 0.  The first point of each
   non-overlapping window has no prior context and receives a fixed zero score in
   both normal-train calibration and test evaluation.

The sole arm difference is whether the current target value is visible.

## Metrics and decision gates

Primary: neural target-error ROC-AUC and PR-AUC.  Train-p99 raw F1, oracle best
raw F1, and Best-F1+PA are diagnostics.

V5 no-spatial neural reference:

| Dataset | ROC-AUC | PR-AUC | Prevalence |
|---|---:|---:|---:|
| MSL | 0.6216 | 0.1544 | 0.1032 |
| SMAP | 0.4816 | 0.1202 | 0.1277 |

Literature context (not a controlled head-to-head gate): MSL ROC/AP/PA-F1
0.7314/0.2186/0.9513; SMAP 0.6224/0.1912/0.9732.

- Continue `forecast` to seeds 90/98 only if its ROC and AP both exceed V5 on
  both datasets, AP exceeds prevalence on both, and SMAP ROC exceeds 0.5.
- Attribute benefit to causal forecasting only if it improves at least one
  ROC/AP metric over `trecon` by 0.01 and no ROC/AP metric falls by more than
  0.01.
- A seed-87 pass creates a development candidate, never a final/SOTA model.
- If the continuation gate fails, do not tune the lag, target set, PA/POT,
  threshold, or window on these labels.  Record the result and stop this route.

## Commands

```bash
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6_nasa_target.sh MSL trecon 0 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6_nasa_target.sh MSL forecast 1 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6_nasa_target.sh SMAP trecon 2 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6_nasa_target.sh SMAP forecast 0 87
```

## Outcome and post-run architecture audit

All four arms completed and are recorded in
`analysis/v6_nasa_target/seed87.json`.  Target-only scoring materially improved
MSL (target reconstruction ROC/AP 0.6758/0.2424 versus V5
0.6216/0.1544), but SMAP remained near random.  The continuation gate failed.

More importantly, a post-run code audit found that the KAN-TCN activation uses
full-window InstanceNorm and the KANAD decoder contains a full-window
`Linear(W,W)`.  Consequently the `forecast` arm's lagged target input did not
make the end-to-end network causal.  Its results are retained as a target-mask
diagnostic, but **must not be reported as evidence for causal forecasting**.
No seeds 90/98 are run.  A separate, newly frozen correction must use only
causal temporal blocks plus a pointwise decoder and must include an automated
future-perturbation invariance test.
