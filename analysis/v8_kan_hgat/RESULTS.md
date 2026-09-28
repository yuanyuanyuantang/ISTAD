# V8 shared-KAN HGAT seed-87 screen

> This is the preregistered development screen on previously inspected test sets;
> it is not independent confirmation and does not establish current SOTA.

| Dataset | Linear HGAT AP | KAN HGAT AP | ΔAP | Linear HGAT ROC | KAN HGAT ROC | ΔROC | Added params |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXA | 0.498319 | 0.568303 | +0.069984 | 0.846548 | 0.877868 | +0.031320 | +72 |
| PSM | 0.412556 | 0.409627 | -0.002928 | 0.649454 | 0.655455 | +0.006001 | +72 |
| SMD | 0.120110 | 0.146495 | +0.026385 | 0.690461 | 0.707561 | +0.017100 | +72 |
| SWAT | 0.557061 | 0.274347 | -0.282714 | 0.858415 | 0.779550 | -0.078865 | +72 |
| **Macro** | — | — | **-0.047318** | — | — | **-0.006111** | — |

| Dataset | POT+PA (linear) | POT+PA (KAN) | Δ | Best-F1+PA (linear) | Best-F1+PA (KAN) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| EXA | 0.959325 | 0.959325 | +0.00000000 | 0.962799 | 0.962799 | +0.00000000 |
| PSM | 0.969713 | 0.969713 | +0.00000000 | 0.982243 | 0.982243 | +0.00000000 |
| SMD | 0.847780 | 0.847780 | +0.00000000 | 0.882938 | 0.882938 | +0.00000000 |
| SWAT | 0.832263 | 0.832263 | +0.00000000 | 0.952449 | 0.952449 | +0.00000000 |

## Frozen decision

- Finite artifacts: **True**
- Added parameters <= 128 and consistently +72: **True**
- Zero distinct V4-rank inversions: **True**
- Non-empty tie refinement: **False**
- PA compatibility within 1e-4: **True**
- HGAT-only AP diagnostic: **False**
- Overall promotion: **False**
