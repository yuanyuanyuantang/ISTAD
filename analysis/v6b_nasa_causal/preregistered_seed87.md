# V6b corrected causal target diagnostic (seed 87)

Frozen: 2026-09-10, before training or scoring either V6b arm.

## Scope

This is a development diagnostic on already revealed MSL/SMAP labels.  It
corrects the future leakage discovered by the post-run V6 architecture audit.
It cannot provide independent confirmation or establish SOTA.

## Fixed architecture and protocol

- Entity-aware NASA split/window/PA protocol: 27 MSL and 53 SMAP entities,
  per-entity 80/20 normal split, window/evaluation step 100/100.
- All 55/25 columns are input context; only official target column 0 is decoded,
  trained, corrupted, and scored.
- Strictly causal front Conv1d and standard dilated TCN; the KAN-TCN is excluded
  because its InstanceNorm uses full-window statistics.
- A 32-hidden-unit pointwise decoder applies only feature-axis LayerNorm/MLP at
  each time step; there is no temporal convolution or `Linear(W,W)` in the
  decoder.
- An automated future-perturbation test must pass before training.
- Seed 87, three epochs, batch 128, Adam 0.01/type1, dropout 0.2, four target-only
  corruption families, clean/denoise weights 1.0/0.5, no HGAT, KAN, RevIN,
  dual view, evidence head, or innovation fusion.  Test loss stays hidden.

## Fixed arms and gates

1. `trecon`: current target column 0 is visible;
2. `forecast`: current target is replaced by its one-step lag while current
   one-hot command context remains visible.  Its first point per window receives
   a fixed zero score in both train calibration and test.

Primary metrics are target-error ROC-AUC and PR-AUC.  Continue `forecast` to
seeds 90/98 only if ROC/AP both exceed V5 no-spatial on both datasets
(MSL 0.6216/0.1544; SMAP 0.4816/0.1202), AP exceeds prevalence on both, and
SMAP ROC exceeds 0.5.  Attribute causal benefit only if at least one ROC/AP
gain over `trecon` is at least 0.01 and none falls by more than 0.01.

Train-p99 raw F1, best raw F1, and Best-F1+PA are diagnostics.  Literature
context remains MSL 0.7314/0.2186/0.9513 and SMAP 0.6224/0.1912/0.9732 for
ROC/AP/PA-F1, not controlled head-to-head gates.

If the gate fails, do not tune lag, target, threshold, window, PA, or POT on
these labels.  Record the result and move to an entity-conditioned objective.

## Commands

```bash
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh MSL trecon 0 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh MSL forecast 1 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh SMAP trecon 2 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh SMAP forecast 0 87
```

## Frozen-result addendum

Added after all four seed-87 artifacts had been produced and evaluated by
`analysis/evaluate_v6b_nasa_causal.py`.

| Dataset | Arm | ROC-AUC | PR-AUC | train-p99 raw F1 | train-p99 PA-F1 | Best raw F1 | Best PA-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| MSL | target reconstruct | 0.626318 | 0.210178 | 0.184318 | 0.848325 | 0.255885 | 0.888642 |
| MSL | target forecast | 0.625820 | 0.209771 | 0.135612 | 0.903308 | 0.254270 | 0.908330 |
| SMAP | target reconstruct | 0.522969 | 0.137770 | 0.039733 | 0.839887 | 0.236536 | 0.853158 |
| SMAP | target forecast | 0.543606 | 0.134266 | 0.030423 | 0.825528 | 0.239612 | 0.844535 |

The continuation and causal-benefit gates passed exactly as frozen, so only the
forecast arm continued to seeds 90/98.  No literature-context row met all three
ROC/AP/PA targets.  The complete metrics and artifact hashes are in `seed87.json`.

V6b has 134,231/74,411 trainable parameters on MSL/SMAP, including only
3,805/1,765 decoder parameters.  Relative to the V5 no-spatial model this is an
11.8%/15.9% whole-model reduction and a 77.3%/85.2% decoder reduction.
