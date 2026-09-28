# V4-HG development results

## Candidate 1: normal-tail reliability mixture

Frozen protocol: `preregistered_seed87.md`.

The candidate failed and is retained only as a negative result. Its non-zero
0.174--0.20 HGAT weights changed the V4 ordering too aggressively. Seed-87
POT + point-adjust / Best-F1 + point-adjust macro-F1 were `0.713947 / 0.943374`
versus V4 `0.902270 / 0.945107`. In particular, SWaT POT+PA collapsed to
`0.098180`. Seeds 90 and 98 were not run.

The failure motivated the separately frozen rank-safe refinement; the failed
numbers were not relabeled as the main result.

## Rank-safe V4-HG: passed

Frozen protocols: `preregistered_rank_safe_seed87.md` and
`preregistered_multiseed_confirmation.md`.

All 12 runs (four datasets x three seeds) completed. The independently
recomputed result is:

| Dataset | POT+PA mean±std | Best-F1+PA mean±std | Neural params | HGAT params |
|---|---:|---:|---:|---:|
| Exathlon | 0.959325±0.000000 | 0.962799±0.000000 | 4,771 | 274 |
| PSM | 0.969713±0.000000 | 0.982243±0.000000 | 7,303 | 346 |
| SMD | 0.847780±0.000000 | 0.882938±0.000000 | 14,522 | 506 |
| SWaT | 0.832263±0.000000 | 0.952449±0.000000 | 44,867 | 618 |
| **Macro** | **0.902270±0.000000** | **0.945107±0.000000** | — | — |

These F1 values exactly preserve the frozen V4 results. The zero standard
deviation is a consequence of the rank-safety constraint, not evidence that
the learned graph is seed invariant. Message gates and hyperedge scores differ
across seeds.

The audit found zero inversions between distinct V4 empirical ranks in all 12
runs, while HGAT refined thousands of tied-score groups in every run. A later
serialization audit found that the original float32 dump erased part of the
O(1e-6) refinement, so all statistical scores were regenerated as float64.
Under the corrected artifacts, mean ROC-AUC deltas versus V4 are `+0.00000459`,
`+0.00000058`, `+0.00000036`, and `+0.00008703`; mean PR-AUC deltas are
`+0.00008198`, `+0.00001896`, `+0.00004268`, and `-0.00328109` on Exathlon,
PSM, SMD, and SWaT. Thus rank safety and non-vacuous tie refinement hold, but a
consistent AP gain does not. A 100-repeat random tie-control further shows that
the small AP gains on the first three datasets are not clearly separated from
generic tie breaking.

Decision: the original frozen PA/rank-safety gates passed. V4-HG remains a
compact architecture candidate because it really routes V4 feature innovations
through a learned dynamic hypergraph and cannot overwrite distinct V4 ranks.
The accuracy evidence for HGAT is mixed and must not be described as a uniform
ranking improvement. The result remains a development result on already
inspected benchmarks, not independent evidence of current-domain SOTA.

Machine-readable PA metrics, hashes, rank audits, component scores, gates, and
weights are in `multiseed_results.json`. Corrected no-PA, event, random-control,
and efficiency results are in `analysis/v4_hgat_paper/`.
