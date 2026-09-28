# V7.1 residual causal hypergraph: frozen development plan

Frozen: 2026-09-10, after the V7 five-dataset seed-87 stopping decision and
before any V7.1 training.

## Status and purpose

This is a **development** experiment.  EXATHLON, PSM, SMD, SWAT, MSL, and
SMAP labels have already been exposed; none of its outcomes can be described
as independent confirmation.  Its purpose is to test a structural correction
derived from the uniform V7 failure mode, not to tune thresholds or select
favorable datasets.

## Frozen changes from V7

1. Preserve the training-only signed ridge prior, low-rank directed dynamic
   incidence, entity isolation/context, standard causal TCN, target masking,
   chronological validation, optimizer, epoch count, and all data recipes.
2. Replace the coupled three-input decoder with nested prediction:

   `forecast = lagged target + temporal residual + sigmoid(g) * graph residual`.

   The temporal residual has its own pointwise head.  The graph residual has a
   separate pointwise head and a learned per-target gate initialized to 0.05,
   so full V7.1 contains the temporal control and can safely approach it.
3. Replace the hard log-prior bias with a learned per-target convex prior gate:

   `incidence = (1-sigmoid(a))*dynamic + sigmoid(a)*causal_prior`.

   The gate is initialized to 0.5.  In the `no_prior` arm it is fixed to zero.
4. Use prediction error alone as the anomaly score.  Relation JS deviation is
   still exported for structural explanation and audit, but has score weight
   exactly zero.  No test label is used to learn either gate or calibrate the
   score.
5. The prior KL term is multiplied by the mean learned prior gate.  This lets
   the model reject an unreliable prior instead of paying an unavoidable
   regularization penalty.

No hyperparameter sweep is allowed.  The graph-gate initialization 0.05,
prior-gate initialization 0.5, ridge 0.01, KL weight 0.05, relation dimension
16, auto top-k, batch 128, learning rate 0.005, five epochs, and seed 87 are
fixed for all datasets.

## Arms and datasets

Run `full`, `no_prior`, and `temporal_only` on EXATHLON, PSM, SMD, SWAT, MSL,
and SMAP.  The controls share the exact V7.1 temporal path and decoder.  Report
ROC-AUC/AP without point adjustment as primary metrics; train-p99 raw/PA and
Best-F1 raw/PA are descriptive only.

## Development gates

For each dataset:

1. all three artifacts are finite, aligned, and complete;
2. full ROC-AUC > 0.5 and AP > anomaly prevalence;
3. full minus temporal-only is at least -0.01 on both ROC/AP and positive on
   at least one;
4. full minus no-prior is at least -0.01 on both ROC/AP and positive on at
   least one;
5. learned graph and prior gates are finite and strictly inside (0, 1).

V7.1 merits a new multi-seed development run only if at least 4/6 datasets pass
all gates and at least 3/4 multi-target datasets (EXATHLON/PSM/SMD/SWAT) pass.
Seeds 90/98 must not be run otherwise.  Even if this screen passes, publication
confirmation requires a new dataset whose labels were not used in V4--V7.1
development.

