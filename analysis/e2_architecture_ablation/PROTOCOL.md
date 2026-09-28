# E2 reconstruction-path HGAT architecture ablation protocol

## Objective

E2 tests whether hypergraph message passing improves the neural reconstruction
representation. It is intentionally separate from E1, which held model outputs
fixed and changed incidence only in the final rank-safe score refinement.

## Experimental arms

All arms retain the same causal convolution, HGAT parameter tensors, concatenation
width, pointwise decoder, denoising objective, optimizer, validation split, and seed.

1. `learned_dynamic`: the current sample-dependent learned incidence.
2. `learned_static`: a trainable incidence inferred only from node and hyperedge
   embeddings; message content remains sample-dependent.
3. `fixed_random`: a deterministic balanced sparse incidence with the same node
   count, hyperedge count, per-edge top-k mechanism, and full node coverage.
4. `no_message`: the HGAT feature passed to the decoder is exactly zero while its
   dimensional slot and decoder parameters are retained.

`fixed_random` is a matched-sparsity control, not an exact match to the learned
incidence's final node-degree distribution or weights. E1 remains the exact
degree/sparsity/weight-preserving score-level counterfactual.

## Scoring and metrics

- Common score: mean per-point reconstruction squared error.
- Primary metric: average precision (AP).
- Secondary metric: ROC-AUC.
- Deployable threshold: the 99th percentile of normal-training scores.
- Thresholded metrics: strict point precision, recall, F1, and overlap Event-F1.
- Supplemental oracle: exact Best raw F1 without point adjustment.
- Point adjustment is excluded from every E2 result used for comparison.

The common reconstruction score is required because the paper-facing
`innovation_hgat` score is dominated by the closed-form innovation component and
would conceal representation differences already diagnosed by E1.

## Execution stages

### Stage A: screening smoke

- Dataset: PSM
- Seed: 87
- Runs: four arms
- Purpose: verify training, gradient flow, checkpoint separation, score generation,
  finite metrics, and non-identical reconstruction outputs.
- The smoke result is not used to select or rename the proposed model.

### Stage B: full paired experiment

- Datasets: EXATHLON, PSM, SMD, SWAT
- Seeds: 87, 90, 98
- Runs: 48
- Completed smoke runs may be reused unchanged.

## Descriptive support rule

Because the benchmark test labels have already been inspected, this rule is a
transparent developmental decision criterion rather than a confirmatory test.
The learned dynamic relation receives architecture-level support only when:

1. Its macro AP is higher than each control.
2. Its per-dataset mean AP is higher than `no_message` on at least three of four
   datasets.
3. It beats `no_message` in at least eight of 12 paired seed runs.
4. Improvements are not confined to Best raw F1 or point-adjusted metrics.

Comparisons with `learned_static` and `fixed_random` determine whether any gain is
specifically attributable to learned sample-dependent relations rather than merely
to an additional message pathway.

## Claim boundary

E2 can support or reject an architectural representation claim. It cannot provide
independent external confirmation, establish root-cause correctness, or repair the
negative score-level E1 finding. E1 and E2 must be reported as different audits.
