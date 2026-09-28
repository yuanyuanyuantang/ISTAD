# MSL/SMAP independent confirmation

Frozen: 2026-09-10, before downloading or evaluating the MSL and SMAP arrays.

## Purpose and status

MSL and SMAP have not been used to select ISTAD V4 rules, HGAT-Lite, the fusion
weight, or the four-dataset POT table.  This run is an additional benchmark
confirmation, not a claim that the public benchmark test labels are a hidden
test set.  No MSL/SMAP result may be used to revise the configuration below.

## Frozen detector

- Detector: pure training-only causal innovation branch; no neural score and no
  dataset-specific HGAT training hyperparameter.
- Strict normal split: chronological first 80% for fitting; final 20% remains a
  normal validation split and is not used to fit the detector.
- Preprocessing: `StandardScaler` fitted on the fitting split only.
- VAR lag: 1; ridge: 0.01.
- Pool selection: `auto`; sparse/dense cutoff: 0.10.
- Residual scale floor: 0.10 of the median non-degenerate scale.
- Calibration: right-sided ECDF fitted only on normal fitting scores.
- Dataset source: the MSL/SMAP NumPy artifacts published in the
  `thuml/Time-Series-Library` Hugging Face dataset and loaded by the existing
  ISTAD loaders.
- Window/evaluation step: 100/100.  This only truncates the tail to complete
  windows; it does not duplicate points.

## Frozen metrics and thresholds

Report evaluated-point count, anomaly prevalence, AUC-ROC, AUC-PR, and:

1. train-p99 raw point F1 (primary deployable threshold diagnostic);
2. train-p99 + point-adjust F1;
3. POT + point-adjust with `q=1e-5`, level=0.99, multiplier=1.0, and the already
   frozen shared tail margin 1.04;
4. Best-F1 + point-adjust using the existing 200+500 adaptive oracle grid;
5. Best raw point F1 using the same 200+500 grid, explicitly marked oracle.

The MSL/SMAP labels may be read only after this file is written.  No threshold,
pool, lag, ridge, calibration, window, or score transformation may be changed
afterward.  Results are reported even if poor.

## Interpretation gates

- Integrity: all fit/test scores finite; aligned score/label lengths; no test
  sample used by fitting or ECDF calibration.
- Ranking sanity: AUC-ROC > 0.5 and AUC-PR > anomaly prevalence on both datasets.
- External comparison: compare only with values whose adjustment and threshold
  protocol can be established from a primary paper or official implementation.
- A current-field SOTA statement requires a protocol-matched result above every
  audited public baseline on both datasets.  Otherwise use “competitive under
  the frozen protocol” or report a negative confirmation.

## Execution status

The initial 2026-09-10 download attempt stopped before reading any label because
the host's HTTPS proxy was unavailable.  A subsequent filesystem audit found
project-archived NumPy-array pickles.  Serializing those arrays with `numpy.save`
produced SHA-256 values exactly equal to all six values published on the THUML
Hugging Face file pages at commit `ffdb671`; the conversion manifest records the
source and destination hashes.  Labels were first loaded only after this frozen
protocol existed.

The frozen primary run is complete:

| Dataset | ROC-AUC | PR-AUC | prevalence | train-p99 raw F1 | POT+PA | Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|
| MSL | 0.5024 | 0.1177 | 0.1054 | 0.0324 | 0.8857 | 0.8863 |
| SMAP | 0.5235 | 0.1245 | 0.1279 | 0.0110 | 0.7697 | 0.7705 |

Integrity passed, but the preregistered ranking gate failed because SMAP PR-AUC
is below anomaly prevalence.  Both oracle PA values are also below MtsCID's
reported PA-F1 (MSL 0.9513, SMAP 0.9732).  This is therefore a **negative
confirmation**: the pure V4 innovation detector does not generalize as a
current-field SOTA method to MSL/SMAP, and these labels must not now be used to
retune the frozen result.

After the primary run, an explicitly post-hoc entity-boundary sensitivity split
the 27 MSL and 53 SMAP entities independently for fitting, windows, lag pairs,
and PA.  It produced MSL/SMAP ROC-AUC 0.5012/0.5259, PR-AUC 0.1157/0.1248,
POT+PA 0.8823/0.6987, and Best-F1+PA 0.8839/0.7680.  Thus the negative conclusion
is not an artifact of concatenation boundaries.  This sensitivity is diagnostic,
not confirmatory, because it was specified after viewing the primary result.

Reproduce from the repository root:

```bash
python analysis/convert_archived_msl_smap.py
python analysis/evaluate_independent_confirmation.py
python analysis/evaluate_nasa_entity_sensitivity.py
```

Artifacts:

- `dataset_conversion_manifest.json`: exact official-file hash verification;
- `msl_smap_results.json`: frozen primary result and protocol-separated baseline
  context;
- `msl_smap_entity_sensitivity.json`: non-confirmatory boundary sensitivity;
- `MSL_pure_v4_scores.npz` / `SMAP_pure_v4_scores.npz`: primary scores and labels.
