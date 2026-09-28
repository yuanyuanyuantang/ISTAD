# V7 PSM four-arm development diagnostic (seed 87)

Frozen: 2026-09-10, before V7 training or scoring on PSM.

## Scope and fixed protocol

PSM is used because all 25 variables are genuine prediction targets, allowing
the directed multi-target hypergraph to be tested rather than the one-target
NASA special case.  PSM labels were previously revealed, so this is only a
development diagnostic.

- strict first-80% normal train / last-20% normal validation split;
- window/evaluation step 64/64; all 25 targets use a one-step forecast;
- seed 87; five epochs; Adam 0.005/type1; batch 128; dropout 0.1;
- V7 defaults frozen in `preregistered_design.md`; no test loss during training;
- score calibration uses only normal training windows; no point adjustment is
  used for the primary ROC-AUC and PR-AUC decision.

## Fixed arms

1. `full`: causal prior + dynamic hypergraph forecast + relation score;
2. `no_prior`: uniform structural reference, no log-prior bias or prior KL;
3. `no_relation_score`: full forecast, prediction-error score only;
4. `temporal_only`: no hypergraph forecast path and prediction-error score only.

All other settings are identical.  No arm may be rerun with a replacement seed.

## Frozen gates

Implementation completeness requires 87,808 finite aligned scores and both raw
and calibrated V7 components for every arm.

Continue full V7 to seeds 90/98 only if:

1. full PR-AUC exceeds anomaly prevalence and full ROC-AUC exceeds 0.5;
2. full is not below temporal-only by more than 0.01 on either ROC or AP, and
   improves at least one by 0.01;
3. full is not below no-prior by more than 0.01 on either ROC or AP, and improves
   at least one by 0.005;
4. adding the relation-deviation score does not reduce ROC or AP by more than
   0.01 relative to `no_relation_score`;
5. all implementation-completeness checks pass.

The stronger paper-readiness context is PSM ROC/AP 0.7560/0.5303 from the
HGAT-Lite seed-87 development run.  Beating it is reported separately and is
not silently substituted for the mechanism gates.  Best-F1+PA is an oracle
diagnostic only.

## Commands

```bash
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM full 0 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM no_prior 1 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM no_relation_score 2 87
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh PSM temporal_only 3 87
```
