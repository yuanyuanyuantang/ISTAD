# ISTAD Stage C v2: permutation-invariant stability

This methodological correction keeps the frozen events, seeds, checkpoints, and label-use boundary. Latent hyperedge IDs are not compared before permutation alignment. All values are descriptive; no p-values or significance claims are produced.

## Seed-independent variable evidence

| Dataset | Event evidence repeat max abs diff | Event evidence SHA256 |
|---|---:|---|
| EXA | 0.000e+00 | `c1e6a97829c11280fdb6dfe69044c957b1e8febf901dd0e237c9e44a188162fc` |

## Permutation-invariant and aligned relation stability

| Dataset | Seed pair | ARI | NMI | Co-membership cosine | Top-k neighbor Jaccard | Matched incidence cosine | Aligned assignment agreement | Aligned peak-d Kendall tau-b |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EXA | 87-90 | -0.006 | 0.617 | 0.153 | 0.238 | 0.355 | 0.186 | 0.000 |
| EXA | 87-98 | -0.003 | 0.600 | 0.178 | 0.234 | 0.303 | 0.233 | 0.167 |
| EXA | 90-98 | 0.015 | 0.603 | 0.202 | 0.199 | 0.334 | 0.251 | -0.111 |

## Interpretation boundary

The one-step residual evidence is the primary variable-level explanation. The HGAT incidence is an auxiliary relational view, and its stability is assessed through induced variable relations and explicitly aligned latent hyperedges. These results do not test physical root-cause correctness or the detection benefit of dynamic topology.
