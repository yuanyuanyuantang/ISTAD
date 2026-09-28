# ISTAD Stage C v2: permutation-invariant stability

This methodological correction keeps the frozen events, seeds, checkpoints, and label-use boundary. Latent hyperedge IDs are not compared before permutation alignment. All values are descriptive; no p-values or significance claims are produced.

## Seed-independent variable evidence

| Dataset | Event evidence repeat max abs diff | Event evidence SHA256 |
|---|---:|---|
| EXA | 0.000e+00 | `c1e6a97829c11280fdb6dfe69044c957b1e8febf901dd0e237c9e44a188162fc` |
| PSM | 0.000e+00 | `4168720ba8952602d16f2dd14d665e15d3b8a8661ed7ebc50582e69e77b1a434` |
| SMD | 0.000e+00 | `85f0e1c2e96279624abf82df928e20f09c6b81303a0fae9cd35d86b062b63c32` |
| SWAT | 0.000e+00 | `a942802066894eaabdf73d6d3acb516ba672b9bd988e6e6ac19b33b3119a2b62` |

## Permutation-invariant and aligned relation stability

| Dataset | Seed pair | ARI | NMI | Co-membership cosine | Top-k neighbor Jaccard | Matched incidence cosine | Aligned assignment agreement | Aligned peak-d Kendall tau-b |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EXA | 87-90 | -0.006 | 0.617 | 0.153 | 0.238 | 0.355 | 0.186 | 0.000 |
| EXA | 87-98 | -0.003 | 0.600 | 0.178 | 0.234 | 0.303 | 0.233 | 0.167 |
| EXA | 90-98 | 0.015 | 0.603 | 0.202 | 0.199 | 0.334 | 0.251 | -0.111 |
| PSM | 87-90 | 0.007 | 0.563 | 0.211 | 0.123 | 0.332 | 0.113 | 0.091 |
| PSM | 87-98 | -0.021 | 0.594 | 0.155 | 0.114 | 0.315 | 0.203 | 0.061 |
| PSM | 90-98 | 0.008 | 0.535 | 0.221 | 0.139 | 0.310 | 0.148 | 0.030 |
| SMD | 87-90 | -0.009 | 0.559 | 0.164 | 0.081 | 0.293 | 0.092 | 0.240 |
| SMD | 87-98 | -0.021 | 0.539 | 0.167 | 0.096 | 0.277 | 0.101 | 0.345 |
| SMD | 90-98 | -0.012 | 0.539 | 0.161 | 0.098 | 0.248 | 0.141 | 0.064 |
| SWAT | 87-90 | -0.045 | 0.429 | 0.157 | 0.052 | 0.316 | 0.162 | -0.168 |
| SWAT | 87-98 | -0.051 | 0.452 | 0.146 | 0.049 | 0.310 | 0.125 | 0.116 |
| SWAT | 90-98 | -0.064 | 0.389 | 0.149 | 0.078 | 0.295 | 0.059 | -0.011 |

## Interpretation boundary

The one-step residual evidence is the primary variable-level explanation. The HGAT incidence is an auxiliary relational view, and its stability is assessed through induced variable relations and explicitly aligned latent hyperedges. These results do not test physical root-cause correctness or the detection benefit of dynamic topology.
