# V9 bounded-residual KAN-HGAT seed-87 screen

> Second development experiment on previously inspected test sets; not independent confirmation.

| Dataset | Linear HGAT AP | BR-KAN HGAT AP | ΔAP | ΔROC | Spline L2 | Added params | Tie groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXA | 0.498319 | 0.490782 | -0.007538 | -0.003079 | 0.177564 | +64 | 12100 |
| PSM | 0.412556 | 0.415699 | +0.003143 | -0.001328 | 1.980919 | +64 | 21405 |
| SMD | 0.120110 | 0.118327 | -0.001783 | +0.003120 | 0.404219 | +64 | 172840 |
| SWAT | 0.557061 | 0.556064 | -0.000997 | +0.002027 | 0.528524 | +64 | 77178 |
| **Macro** | — | — | **-0.001794** | **+0.000185** | — | — | — |

| Dataset | POT+PA linear / BR-KAN | Best-F1+PA linear / BR-KAN |
|---|---:|---:|
| EXA | 0.959325 / 0.959325 | 0.962799 / 0.962799 |
| PSM | 0.969713 / 0.969713 | 0.982243 / 0.982243 |
| SMD | 0.847780 / 0.847780 | 0.882938 / 0.882938 |
| SWAT | 0.832263 / 0.832263 | 0.952449 / 0.952449 |

## Frozen decision

- initial_linear_nesting_pass: **True**
- finite_artifacts_pass: **True**
- parameter_budget_pass: **True**
- learned_spline_pass: **True**
- rank_safety_pass: **True**
- non_vacuous_pass: **True**
- pa_compatibility_pass: **True**
- hgat_ap_diagnostic_pass: **False**
- promote: **False**
