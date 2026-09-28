# ISTAD E2 reconstruction-path architecture ablation

> Frozen protocol: all arms use the same reconstruction-error score,
> train-only 99th-percentile threshold, data split, decoder dimensions,
> optimization settings, and paired seeds. Best raw F1 is an oracle
> diagnostic and is not a deployable primary metric.

Status: `SMOKE_ANALYZED_FROM_4_TRAINED_MODELS`

## Per-run metrics

| Dataset | Seed | Arm | ROC-AUC | AP | Train-P99 raw F1 | Event-F1 | Best raw F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| PSM | 87 | Learned dynamic incidence | 0.773123 | 0.562639 | 0.478143 | 0.610574 | 0.595728 |
| PSM | 87 | Learned static incidence | 0.779277 | 0.566807 | 0.465592 | 0.655527 | 0.594933 |
| PSM | 87 | Fixed balanced-random incidence | 0.777372 | 0.568726 | 0.472631 | 0.641275 | 0.600839 |
| PSM | 87 | Without hypergraph message passing | 0.768954 | 0.572033 | 0.469027 | 0.656558 | 0.590282 |

## Aggregate metrics

| Arm | ROC-AUC | AP | Train-P99 raw F1 | Event-F1 | Best raw F1 |
|---|---:|---:|---:|---:|---:|
| Learned dynamic incidence | 0.773123 +/- 0.000000 | 0.562639 +/- 0.000000 | 0.478143 +/- 0.000000 | 0.610574 +/- 0.000000 | 0.595728 +/- 0.000000 |
| Learned static incidence | 0.779277 +/- 0.000000 | 0.566807 +/- 0.000000 | 0.465592 +/- 0.000000 | 0.655527 +/- 0.000000 | 0.594933 +/- 0.000000 |
| Fixed balanced-random incidence | 0.777372 +/- 0.000000 | 0.568726 +/- 0.000000 | 0.472631 +/- 0.000000 | 0.641275 +/- 0.000000 | 0.600839 +/- 0.000000 |
| Without hypergraph message passing | 0.768954 +/- 0.000000 | 0.572033 +/- 0.000000 | 0.469027 +/- 0.000000 | 0.656558 +/- 0.000000 | 0.590282 +/- 0.000000 |

## Learned dynamic incidence minus controls

| Control | Delta ROC-AUC | W/T/L | Delta AP | W/T/L |
|---|---:|---:|---:|---:|
| Learned static incidence | -0.006153 | 0/0/1 | -0.004167 | 0/0/1 |
| Fixed balanced-random incidence | -0.004249 | 0/0/1 | -0.006087 | 0/0/1 |
| Without hypergraph message passing | +0.004170 | 1/0/0 | -0.009394 | 0/0/1 |

## Interpretation boundary

- This experiment audits the neural reconstruction path, not the closed-form
  causal-innovation score used by the current paper-facing detector.
- No point adjustment is used in the reported E2 metrics.
- The 99th-percentile threshold uses normal-training scores only.
- Best raw F1 uses test labels and is reported only as an oracle diagnostic.
- These benchmark labels have already been inspected, so the experiment is
  developmental evidence rather than independent confirmation.
