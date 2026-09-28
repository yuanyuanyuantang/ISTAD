# ISTAD V7 causal-prior hypergraph design and implementation contract

Frozen: 2026-09-10, before any V7 training or test-label scoring.

## Scientific claim under test

Normal-operation lagged dependencies provide a directed structural prior that
can constrain a dynamic hypergraph forecaster.  A useful V7 must make this
structure part of both prediction and anomaly scoring, rather than attaching an
HGAT branch to a score produced elsewhere.

This is a development programme.  MSL/SMAP and the original four test sets have
already been revealed, so none can independently establish SOTA.

## Frozen architecture

1. Fit a signed ridge causal prior using only the scaled normal training split.
   The design at time `t` exactly matches inference: selected target columns are
   replaced by their lagged values while current exogenous columns stay visible.
   Entity boundaries are never crossed.
2. Treat each predicted target as a directed hyperedge.  Source-node incidence
   is initialized from the absolute ridge coefficients; coefficient signs are
   retained for signed node-to-edge aggregation.
3. Produce dynamic incidence with a low-rank feature-state/target query and a
   log-prior bias.  Top-k source selection makes every target hyperedge sparse.
4. In parallel, use only standard dilated causal convolutions.  Fuse temporal,
   lagged-target, and signed hypergraph states with a per-time decoder.  No
   InstanceNorm, `Linear(W,W)`, bidirectional recurrence, RevIN, or test-time
   temporal normalization is allowed.
5. Score each point with two quantities that directly come from V7: target
   forecast error and Jensen-Shannon deviation of dynamic incidence from the
   normal causal prior.  Calibrate both by normal-training ECDF, per entity when
   entity metadata exist, and combine with a fixed relation weight of 0.25.
6. For MSL/SMAP/SMD, optional one-hot entity context is appended after scaling
   as context nodes.  These nodes may influence target hyperedges but are never
   prediction targets or anomaly labels.

## Fixed implementation defaults

- relation rank 16; prior ridge `1e-2`; prior top-k `max(3, ceil(0.2N))`;
- log-prior strength 1.0; prior floor 0.01; prior KL weight 0.05;
- relation score weight 0.25; pointwise decoder width 32; dropout 0.1;
- target forecast lag 1; strict chronological validation; training-only prior
  and calibration; no test loss during training.

These are mechanism defaults, not values selected from test labels.  Any later
change requires a new versioned plan.

## Implementation acceptance gates

V7 is considered implemented only if all of the following pass:

1. a future-input perturbation leaves all earlier predictions and incidences
   unchanged in evaluation mode;
2. causal-prior fitting excludes cross-entity lag pairs and depends on no test
   point or label;
3. entity ECDF calibration is invariant to appending arbitrary evaluation data;
4. V7 prediction has a non-zero gradient path through dynamic incidence and the
   causal prior is stored in checkpoints as a buffer;
5. legacy checkpoints and all existing regression tests remain compatible;
6. a CPU end-to-end smoke run writes finite forecast-error, relation-deviation,
   calibrated-component, score, label, and entity metadata artifacts.

## Experimental decision frozen before execution

The first performance experiment must compare full V7 against `prior_strength=0`,
`relation_score_weight=0`, and a true `use_hypergraph=0` temporal-only control
under identical seeds and optimization.
Primary metrics are PR-AUC and ROC-AUC without point adjustment.  PA-F1 remains
secondary and Best-F1+PA remains an oracle diagnostic.  No arm may be selected
using a single favorable seed; at least seeds 87/90/98 and a new isolated
confirmation dataset are required before a paper-level superiority claim.

## Post-freeze execution record

Implementation gates, the seed-87 four-arm diagnostic, and the seeds 87/90/98
PSM confirmation have since completed. The mechanism confirmation passed, but
the paper-readiness reference was not met. This execution note changes none of
the architecture or decision rules above; exact results are recorded in
`RESULTS.md` and `psm_multiseed_confirmation.json`.
