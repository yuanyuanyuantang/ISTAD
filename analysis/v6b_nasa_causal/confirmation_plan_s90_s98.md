# V6b causal target stability confirmation (seeds 90/98)

Frozen: 2026-09-10, after the seed-87 continuation gate passed and before
training seed 90 or 98.

## Fixed scope

Run only the V6b `forecast` arm on MSL and SMAP with seeds 90 and 98.  Every
architecture, data, optimization, scoring, and threshold argument remains
identical to `preregistered_seed87.md`.  Seed 87 is the development seed;
90/98 are fixed stability seeds.  These datasets are already revealed, so the
exercise confirms reproducibility, not independent generalization.

## Stability decision

Retain V6b as a stable NASA development candidate only if all conditions hold:

1. all four new artifacts are complete, finite, entity-aware, and have the
   expected 72,600/425,400 evaluated points;
2. each of seeds 90 and 98 has AP above anomaly prevalence and ROC above 0.5 on
   both MSL and SMAP;
3. three-seed mean ROC and AP both exceed the V5 no-spatial references on both
   datasets (MSL 0.6216/0.1544; SMAP 0.4816/0.1202);
4. three-seed ROC and AP standard deviations do not exceed 0.03;
5. no individual seed falls below the corresponding V5 PR-AUC reference.

Literature-context SOTA is a separate, stronger check: three-seed mean ROC,
AP, and Best-F1+PA must all meet MSL 0.7314/0.2186/0.9513 and SMAP
0.6224/0.1912/0.9732.  Failure here does not erase a stability pass, but it
forbids a SOTA claim.

No seed may be discarded or replaced.  If stability fails, do not tune this
route on MSL/SMAP.

## Commands

```bash
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh MSL forecast 0 90
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh MSL forecast 1 98
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh SMAP forecast 2 90
bash code/ISTAD/scripts/anomaly_detection/ISTAD_v6b_nasa_causal.sh SMAP forecast 0 98
```

## Frozen-result addendum

Added after all four confirmation artifacts were complete.  No run was removed
or replaced.  `analysis/evaluate_v6b_nasa_multiseed.py` produced:

| Dataset | Metric | seed 87 | seed 90 | seed 98 | mean ± std |
|---|---|---:|---:|---:|---:|
| MSL | ROC-AUC | 0.625820 | 0.677146 | 0.649408 | 0.650791 ± 0.020977 |
| MSL | PR-AUC | 0.209771 | 0.231328 | 0.214849 | 0.218649 ± 0.009202 |
| MSL | train-p99 PA-F1 | 0.903308 | 0.871762 | 0.887349 | 0.887473 ± 0.012879 |
| MSL | Best PA-F1 | 0.908330 | 0.908049 | 0.907748 | 0.908043 ± 0.000237 |
| SMAP | ROC-AUC | 0.543606 | 0.562611 | 0.451182 | 0.519133 ± 0.048671 |
| SMAP | PR-AUC | 0.134266 | 0.143195 | 0.119650 | 0.132371 ± 0.009705 |
| SMAP | train-p99 PA-F1 | 0.825528 | 0.737832 | 0.807265 | 0.790208 ± 0.037779 |
| SMAP | Best PA-F1 | 0.844535 | 0.846719 | 0.845534 | 0.845596 ± 0.000893 |

The artifact-completeness and three-seed-mean-vs-V5 gates passed.  The frozen
decision nevertheless **failed** because SMAP seed 98 was below ROC 0.5 and the
V5 AP reference, while SMAP ROC standard deviation exceeded 0.03.  Therefore
V6b is not retained as a stable development candidate, and this route must not
be tuned further on MSL/SMAP.  It also failed the literature-context SOTA gate.
The machine-readable decision, full raw metrics, and SHA-256 hashes are in
`multiseed_s87_s90_s98.json`.
