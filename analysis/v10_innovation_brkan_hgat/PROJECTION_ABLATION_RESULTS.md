# V10 projection ablation: Linear vs BR-KAN

> Post-hoc seed-87 explanatory ablation on previously inspected test sets;
> it cannot promote V10 or reopen seeds 90/98.

| Dataset | Linear AP | BR-KAN AP | ΔAP | Linear ROC | BR-KAN ROC | ΔROC |
|---|---:|---:|---:|---:|---:|---:|
| EXA | 0.453607 | 0.454850 | +0.001243 | 0.836411 | 0.836895 | +0.000484 |
| PSM | 0.432003 | 0.428805 | -0.003198 | 0.660782 | 0.658022 | -0.002761 |
| SMD | 0.125330 | 0.127282 | +0.001952 | 0.688875 | 0.689605 | +0.000730 |
| SWAT | 0.532461 | 0.564416 | +0.031955 | 0.847506 | 0.857717 | +0.010210 |
| **Macro** | 0.385850 | 0.393838 | **+0.007988** | 0.758394 | 0.760559 | **+0.002166** |

| Dataset | Linear POT+PA | BR-KAN POT+PA | Δ | Linear Best-F1+PA | BR-KAN Best-F1+PA | Δ |
|---|---:|---:|---:|---:|---:|---:|
| EXA | 0.962009 | 0.961231 | -0.000778 | 0.963790 | 0.963512 | -0.000278 |
| PSM | 0.966810 | 0.966600 | -0.000211 | 0.981460 | 0.979426 | -0.002034 |
| SMD | 0.849997 | 0.840440 | -0.009557 | 0.883572 | 0.890056 | +0.006484 |
| SWAT | 0.815097 | 0.827348 | +0.012251 | 0.954012 | 0.953618 | -0.000394 |
| **Macro** | 0.898478 | 0.898905 | **+0.000426** | 0.945708 | 0.946653 | **+0.000944** |

## Frozen decision

- Finite and matched innovation inputs: **True**
- BR-KAN Macro AP exceeds Linear: **True**
- BR-KAN AP non-negative on at least 2/4 datasets: **True** (3/4)
- Retain BR-KAN: **True**

BR-KAN adds +64 parameters. Under the frozen rule, the smaller Linear projection is preferred when `retain_brkan` is false.
