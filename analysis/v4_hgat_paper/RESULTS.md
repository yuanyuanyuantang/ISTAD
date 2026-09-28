# V4-HG paper-facing evaluation

> Generated from frozen score artifacts. POT is train-derived; Best metrics are
> test-label oracle upper bounds. Event-F1 is overlap-based and never applies
> point adjustment. These benchmarks were previously inspected and are not an
> independent confirmation set.

## Main results (three seeds)

| Dataset | ROC-AUC | AUC-PR | POT raw F1 | POT event F1 | Best raw F1 | POT+PA | Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXA | 0.8275±0.0000 | 0.4263±0.0000 | 0.1028±0.0000 | 0.6835±0.0000 | 0.5438±0.0000 | 0.9593±0.0000 | 0.9628±0.0000 |
| PSM | 0.6551±0.0000 | 0.4277±0.0000 | 0.1286±0.0000 | 0.6801±0.0000 | 0.4735±0.0000 | 0.9697±0.0000 | 0.9822±0.0000 |
| SMD | 0.6815±0.0000 | 0.1172±0.0000 | 0.1242±0.0000 | 0.2927±0.0000 | 0.1677±0.0000 | 0.8478±0.0000 | 0.8829±0.0000 |
| SWAT | 0.8500±0.0000 | 0.5293±0.0004 | 0.0011±0.0000 | 0.3121±0.0000 | 0.7356±0.0000 | 0.8323±0.0000 | 0.9524±0.0000 |
| **Macro** | **0.7535** | **0.3751** | **0.0892** | **0.4921** | **0.4802** | **0.9023** | **0.9451** |

## Component ablation (macro over four datasets)

| Arm | ROC-AUC | AUC-PR | POT raw F1 | POT event F1 | Best raw F1 | POT+PA | Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4-HG | 0.7535 | 0.3751 | 0.0892 | 0.4921 | 0.4802 | 0.9023 | 0.9451 |
| V4 | 0.7535 | 0.3759 | 0.0892 | 0.4921 | 0.4801 | 0.9023 | 0.9451 |
| HGAT-only | 0.7636 | 0.4101 | 0.1010 | 0.4576 | 0.5010 | 0.8337 | — |

## Learned HGAT versus random rank-safe tie breaking

| Dataset | Learned ΔROC | Random ΔROC | Learned ROC percentile | Learned ΔAP | Random ΔAP | Learned AP percentile |
|---|---:|---:|---:|---:|---:|---:|
| EXA | +0.00000459 | +0.00000000 | 1.000 | +0.00008198 | +0.00007116 | 0.547 |
| PSM | +0.00000058 | -0.00000000 | 1.000 | +0.00001896 | +0.00001800 | 0.590 |
| SMD | +0.00000036 | -0.00000000 | 1.000 | +0.00004268 | +0.00002460 | 0.760 |
| SWAT | +0.00008703 | -0.00000010 | 1.000 | -0.00328109 | -0.00505467 | 1.000 |

## Audit decisions

- Float64 statistical artifacts: **True**
- Zero distinct-rank inversions in all runs: **True**
- Non-empty learned tie refinement in all runs: **True**
- V4-HG and V4 PA-F1 equality is expected by construction; it is not an HGAT accuracy gain.
- The random-control percentiles determine whether learned tie refinement is stronger
  than generic tie breaking and must be reported rather than hidden.
