# V7 PSM multi-seed confirmation plan

Frozen: 2026-09-10, after the preregistered seed-87 continuation gate passed
and before any V7 seed-90/98 training.

## Purpose

This confirmation checks whether the seed-87 mechanism result is repeatable. It
does not use point-adjusted F1 as its decision statistic and does not authorize
hyperparameter tuning. PSM labels have already been revealed during development,
so even a pass remains internal development evidence rather than an independent
test-set claim.

## Fixed protocol and arms

Keep the seed-87 protocol unchanged and add seeds 90 and 98 for:

1. `full`: causal prior, directed dynamic hypergraph forecast, and relation score;
2. `no_prior`: the same hypergraph without the fitted causal prior or prior KL;
3. `temporal_only`: the same causal TCN/decoder without the hypergraph path.

The prediction-only score is reconstructed directly from every `full` artifact
by setting the already frozen relation-score weight from 0.25 to zero. It is
therefore an exact scoring ablation and requires no duplicate training run.

The primary statistics are ROC-AUC and PR-AUC over the three fixed seeds
87/90/98. Train-P99 F1 and Best-F1 with point adjustment are descriptive only.

## Frozen confirmation gates

All of the following must hold:

1. every artifact contains 87,808 aligned, finite scores and the two V7 score
   components;
2. every full run has ROC-AUC above 0.5 and PR-AUC above anomaly prevalence;
3. the three-seed full mean exceeds both temporal-only and no-prior means on
   both ROC-AUC and PR-AUC;
4. at least two of three paired seeds favor full over each training ablation on
   both ROC-AUC and PR-AUC;
5. full improves the temporal-only three-seed mean by at least 0.02 on one
   primary metric, and improves the no-prior mean by at least 0.01 on one;
6. full ROC-AUC and PR-AUC standard deviations are each at most 0.05;
7. adding relation deviation to the full prediction score does not reduce the
   three-seed mean of either primary metric by more than 0.01.

The earlier HGAT-Lite PSM three-seed development means (ROC-AUC 0.7466,
PR-AUC 0.5547) are retained as a separate paper-readiness reference. They are
not part of the mechanism gate and will not be described as an external SOTA.

## Commands

```bash
for seed in 90 98; do
  bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM full 0 "$seed"
  bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM no_prior 0 "$seed"
  bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM temporal_only 0 "$seed"
done

python analysis/evaluate_v7_psm_multiseed.py
```
