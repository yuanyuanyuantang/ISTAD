# Stage C v2 protocol amendment: permutation-invariant relation stability

## Status and rationale

This amendment was frozen after identifying a measurement defect in Stage C v1
and before observing any v2 result. The datasets, event-selection rule, event
intervals, seeds, checkpoints, model outputs, and label-use boundary remain
unchanged. Stage C v1 directly compared latent hyperedge IDs across independently
trained seeds. Those IDs are non-identifiable: permuting all hyperedge columns
leaves the represented hypergraph unchanged. The corresponding unaligned top-1
Jaccard and peak-edge Kendall statistics are therefore retained for provenance in
`stability.json`, but they are withdrawn as evidence about cross-seed stability.

Stage C v2 is a descriptive methodological correction, not a new confirmatory
experiment. It does not retrain a model, select a checkpoint, tune a threshold,
change a score, or reselect a case.

## Frozen inputs

- Datasets: EXATHLON, PSM, SMD, and SWAT.
- Seeds: 87, 90, and 98.
- Checkpoints: the same 12 frozen V4-HG checkpoints used by Stage C v1.
- Events: the exact start, end, and entity stored in the original
  `stability.json`; the evaluator aborts if recomputation differs.
- Variable evidence: the training-calibrated one-step residual evidence already
  used in Stages A and B.
- Test-label boundary: labels only identify the already frozen demonstration
  event. They do not determine a score, threshold, model, metric, matching rule,
  or reporting decision.

## S1: seed-independent variable evidence

The one-step residual evidence contains no model seed or checkpoint input. For
each dataset, the evaluator recomputes the event evidence with the same frozen
training-only scorer, requires a maximum absolute difference no larger than
`1e-12`, and records a canonical float64 SHA256 digest.

## S2: permutation-invariant induced relation stability

Let `H_t` have shape `(C, M)`. Each row is normalized over its `M` latent
hyperedges, and the induced variable co-membership matrix is

`P_t = H_t H_t^T`.

For each of the three unordered seed pairs and every point in the frozen event,
report:

1. adjusted Rand index (ARI) and arithmetic-mean normalized mutual information
   (NMI) between the per-variable top-1 hyperedge partitions;
2. cosine similarity between the upper triangles of the two `P_t` matrices;
3. relative Frobenius distance between those upper triangles; and
4. mean Jaccard overlap between each variable's top-5 neighbors induced by
   `P_t` (or all other variables if `C < 6`).

These measures are invariant to a global or pointwise renaming of latent
hyperedges. The report includes means, medians, 5th and 95th percentiles, minima,
and maxima. Event points are temporally dependent, so these summaries are
descriptive and are not treated as independent replicates.

## S3: event-wide aligned latent-edge audit

For interpretive continuity with Stage C v1, construct an `M x M` cosine
similarity matrix by flattening each hyperedge's incidence over all variables and
all points of the frozen event. A maximum-weight Hungarian assignment supplies
one event-wide mapping from the second seed's hyperedges to the first seed's
hyperedges. After matching, report:

1. cosine similarity of each matched incidence-column pair;
2. per-point agreement of top-1 variable-to-hyperedge assignments;
3. the Jaccard transform used in Stage C v1, now after alignment; and
4. cosine similarity and Kendall tau-b between the aligned hyperedge-evidence
   vectors at the shared residual-evidence peak.

Hungarian-aligned results are auxiliary because the matching is estimated and
evaluated on the same frozen event. The permutation-invariant S2 measures are the
primary cross-seed relation evidence.

## Reporting boundary

- Report all four datasets and all three unordered seed pairs; do not select
  favorable pairs.
- Do not perform p-values or significance tests on autocorrelated event points.
- Do not set a pass/fail threshold after seeing the results.
- The one-step residual evidence remains the primary variable-level explanation.
- Dynamic HGAT incidence is an auxiliary relation view. Stage C does not establish
  physical root-cause correctness, dynamic-topology superiority, or a detection
  gain from HGAT.
- Use the final manuscript terminology: ISTAD, one-step residual evidence,
  dynamic HGAT incidence, and rank-preserving relation refinement. Historical
  filenames may retain `V4-HG` for traceability.

## Outputs

Stage C v2 writes to a new directory and never overwrites Stage C v1:

```text
analysis/v4hg_interpretability/stage_c_v2/
  stability_v2.json
  STABILITY_V2_RESULTS.md
  VERIFICATION.json
  run.log
  SHA256SUMS.txt
```

Raw event-incidence arrays are temporary by default. Their SHA256 digests are
recorded in `stability_v2.json`; `--keep-raw` may retain the arrays for an
additional audit, but they are not needed in the normal delivery archive.
