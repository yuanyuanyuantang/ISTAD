# V10 innovation-guided BR-KAN-HGAT seed-87 screen

> Development screen on previously inspected test sets; not independent confirmation.

| Dataset | V4-HG AP | V10 AP | ΔAP | V10 HGAT AP | V4-HG ROC | V10 ROC | ΔROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXA | 0.426348 | 0.454850 | +0.028501 | 0.533383 | 0.827545 | 0.836895 | +0.009350 |
| PSM | 0.427683 | 0.428805 | +0.001122 | 0.409358 | 0.655103 | 0.658022 | +0.002918 |
| SMD | 0.117178 | 0.127282 | +0.010104 | 0.149474 | 0.681471 | 0.689605 | +0.008133 |
| SWAT | 0.528790 | 0.564416 | +0.035626 | 0.578945 | 0.850023 | 0.857717 | +0.007694 |
| **Macro** | 0.375000 | 0.393838 | **+0.018838** | 0.417790 | 0.753535 | 0.760559 | **+0.007024** |

| Dataset | V4-HG POT+PA | V10 POT+PA | Δ | V4-HG Best-F1+PA | V10 Best-F1+PA | Δ |
|---|---:|---:|---:|---:|---:|---:|
| EXA | 0.959325 | 0.961231 | +0.001905 | 0.962799 | 0.963512 | +0.000713 |
| PSM | 0.969713 | 0.966600 | -0.003113 | 0.982243 | 0.979426 | -0.002817 |
| SMD | 0.847780 | 0.840440 | -0.007340 | 0.882938 | 0.890056 | +0.007118 |
| SWAT | 0.832263 | 0.827348 | -0.004915 | 0.952449 | 0.953618 | +0.001169 |
| **Macro** | 0.902270 | 0.898905 | **-0.003366** | 0.945107 | 0.946653 | **+0.001546** |

## Frozen decision

- finite_artifacts_pass: **True**
- innovation_input_pass: **True**
- parameter_budget_pass: **True**
- learned_spline_pass: **True**
- macro_ap_pass: **True**
- dataset_ap_pass: **True**
- worst_ap_pass: **True**
- pot_pa_macro_pass: **False**
- best_pa_macro_pass: **True**
- continue_to_three_seeds: **False**
