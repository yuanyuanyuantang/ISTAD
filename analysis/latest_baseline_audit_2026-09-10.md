# Latest-baseline and protocol audit

Audited: 2026-09-10

This file distinguishes the project's frozen 11-model comparison from a claim
about the current literature.  Only primary papers, publisher pages, proceedings,
and official repositories are used for technical claims.  Numbers are not pooled
across incompatible threshold or adjustment protocols.

## Conclusion

The existing ISTAD tables establish first place only against the **stored
11-model baseline set**.  They do not establish current-field SOTA.  At least one
newer peer-reviewed method, MtsCID (WWW 2025), reports higher point-adjusted F1
than ISTAD V4 on every overlapping dataset for which its exact value is available.

| Dataset | ISTAD V4 Best-F1+PA oracle | MtsCID reported PA-F1 | ISTAD minus MtsCID |
|---|---:|---:|---:|
| SMD | 0.8829 | 0.9339 | -0.0510 |
| PSM | 0.9822 | 0.9854 | -0.0032 |
| SWaT | 0.9524 | 0.9691 | -0.0166 |

This comparison is already unfavorable to ISTAD even though its column uses an
oracle test-label threshold.  Dataset preprocessing and PA implementations still
need code-level replication before calling the gap a controlled head-to-head
result, but the published values are sufficient to reject an unqualified SOTA
claim.

The preregistered MSL/SMAP extension subsequently gave the same conclusion on
two additional official TSLib arrays:

| Dataset | ISTAD ROC/AP | ISTAD POT+PA | ISTAD Best-F1+PA oracle | MtsCID reported PA-F1 | U2AD reported ROC/AP |
|---|---:|---:|---:|---:|---:|
| MSL | 0.5024 / 0.1177 | 0.8857 | 0.8863 | 0.9513 | 0.7314 / 0.2186 |
| SMAP | 0.5235 / 0.1245 | 0.7697 | 0.7705 | 0.9732 | 0.6224 / 0.1912 |

The six local arrays serialize to the exact SHA-256 values published by THUML.
The threshold-independent columns are still literature context rather than a
controlled rerun because preprocessing and score aggregation can differ.  The
PA columns are even less interchangeable; nevertheless, ISTAD losing even with
an oracle threshold decisively rules out a current SOTA claim for pure V4.

## Audited methods

| Method | Status | Reported protocol and relevant numbers | Comparability decision |
|---|---|---|---|
| MtsCID | WWW 2025, peer reviewed | Non-overlapping windows of 100, chronological 80/20 train/validation, five runs. Point-adjusted F1: SMD 0.9339, MSL 0.9513, SMAP 0.9732, SWaT 0.9691, PSM 0.9854. It also reports Affiliation-F1 and VUS metrics. | Directly relevant to the PA claim, but exact threshold and code path must be reproduced locally for a controlled table. |
| Mixer-Transformer | JNCA 2025, peer reviewed | Adaptive local threshold; F1: SMAP 0.9749, MSL 0.9518, PSM 0.9820. | Not interchangeable with ISTAD POT or Best-F1 columns because the threshold is adaptive and method-specific. |
| TFD-CAD | JKSUCIS 2026, peer reviewed | Calls its main table point-wise adjusted; five-dataset mean F1 0.9640 and reports winning MSL, SMAP, PSM and SMD. Threshold is selected by validation grid search and a dataset-specific frequency mask is tuned on validation. | Current strong baseline, but not a label-blind fixed-threshold match to ISTAD. Exact per-dataset table should be reproduced from official code before a numeric row is added. |
| MSTDF-AD | IPM 2026, online/ahead of issue | Five-dataset mean F1 0.9647; abstract reports SMD 0.9394 and SMAP 0.9785. Official repository is public. | Strong current candidate; paper/code protocol must be reproduced before merging with ISTAD tables. |
| U2AD | arXiv 2026 preprint | Point-adjusted F1: MSL 0.9460, SMAP 0.9694. Threshold-independent results: MSL ROC/AP 0.7314/0.2186; SMAP ROC/AP 0.6224/0.1912. Also reports VUS and delay metrics. | Useful protocol-separated reference, but not peer reviewed as of the audit date. AUC/AP are the cleanest comparison once identical arrays are available. |

## Primary sources

- [MtsCID published manuscript (University of Adelaide / WWW 2025)](https://digital.library.adelaide.edu.au/server/api/core/bitstreams/7c0fdbfb-9d61-4d6b-985b-e6df3b19f18a/content), DOI `10.1145/3696410.3714941`.
- [Mixer-Transformer author-hosted manuscript](https://servicearchitecture.wp.imtbs-tsp.eu/files/2025/05/Mixer-Transformer_-Adaptive-Anomaly-Detection-with-Multivariate-Time-Series.pdf), DOI `10.1016/j.jnca.2025.104216`.
- [TFD-CAD publisher page and full text](https://link.springer.com/article/10.1007/s44443-026-00805-4).
- [MSTDF-AD publisher page](https://www.sciencedirect.com/science/article/pii/S0306457326002797) and [official code](https://github.com/Conviss/MSTDF-AD).
- [U2AD manuscript](https://arxiv.org/abs/2605.09685).
- [Formal metric analysis, AISTATS 2026](https://proceedings.mlr.press/v300/wagner26a.html).
- [THUML Time-Series-Library MSL files](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/MSL)
  and [SMAP files](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/SMAP),
  including the publisher-displayed SHA-256 values used for local verification.

## Evaluation implications for the paper

1. Replace every bare “SOTA” with “first among the stored 11 baselines under the
   exploratory protocol” until current methods are rerun.
2. Keep POT+PA and Best-F1+PA as literature-compatibility columns, not the sole
   headline evidence.  Best-F1+PA is a test-label oracle.
3. Add AUC-PR, raw train-calibrated F1, and a PR-oriented range metric under one
   implementation.  Do not treat PA-F1, adaptive-threshold F1, and POT-F1 as the
   same protocol.
4. For a controlled current comparison, prioritize MtsCID and one 2026 method
   with public code.  Use identical arrays, chronological split, evaluated points,
   score aggregation, and threshold implementation, and report at least three
   seeds or the authors' released checkpoints.
5. The preregistered MSL/SMAP confirmation is complete and negative.  Do not tune
   pure V4 on those test labels or present the result as competitive.  A post-hoc
   entity-boundary sensitivity reached the same conclusion, so the next model
   iteration needs a representation change and a new untouched confirmation
   target, not another threshold adjustment.
