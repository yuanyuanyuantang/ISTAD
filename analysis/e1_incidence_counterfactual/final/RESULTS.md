# ISTAD E1 incidence counterfactual audit

> Material Passport: frozen V4-HG checkpoints; no retraining; test labels
> are used only after every incidence, ECDF, gate, and fusion decision is fixed.

## Deterministic counterfactuals

Values are mean +/- population standard deviation over seeds 87, 90, and 98.

| Dataset | Arm | ROC-AUC | AP | POT raw F1 | Event-F1 |
|---|---|---:|---:|---:|---:|
| EXA | Learned dynamic incidence | 0.827546 +/- 0.000000 | 0.426345 +/- 0.000011 | 0.102784 +/- 0.000000 | 0.683509 +/- 0.000000 |
| EXA | Static train-mean incidence | 0.827546 +/- 0.000001 | 0.426331 +/- 0.000033 | 0.102784 +/- 0.000000 | 0.683509 +/- 0.000000 |
| EXA | Within-entity time-shuffled incidence | 0.827545 +/- 0.000000 | 0.426324 +/- 0.000018 | 0.102784 +/- 0.000000 | 0.683509 +/- 0.000000 |
| EXA | Fixed node-permuted incidence | 0.827546 +/- 0.000000 | 0.426308 +/- 0.000016 | 0.102784 +/- 0.000000 | 0.683509 +/- 0.000000 |
| PSM | Learned dynamic incidence | 0.655103 +/- 0.000000 | 0.427682 +/- 0.000002 | 0.128555 +/- 0.000000 | 0.680100 +/- 0.000000 |
| PSM | Static train-mean incidence | 0.655104 +/- 0.000000 | 0.427683 +/- 0.000004 | 0.128555 +/- 0.000000 | 0.680100 +/- 0.000000 |
| PSM | Within-entity time-shuffled incidence | 0.655103 +/- 0.000000 | 0.427685 +/- 0.000001 | 0.128555 +/- 0.000000 | 0.680100 +/- 0.000000 |
| PSM | Fixed node-permuted incidence | 0.655103 +/- 0.000000 | 0.427682 +/- 0.000004 | 0.128555 +/- 0.000000 | 0.680100 +/- 0.000000 |
| SMD | Learned dynamic incidence | 0.681471 +/- 0.000000 | 0.117172 +/- 0.000005 | 0.124220 +/- 0.000000 | 0.292679 +/- 0.000000 |
| SMD | Static train-mean incidence | 0.681471 +/- 0.000000 | 0.117173 +/- 0.000028 | 0.124220 +/- 0.000000 | 0.292679 +/- 0.000000 |
| SMD | Within-entity time-shuffled incidence | 0.681471 +/- 0.000000 | 0.117161 +/- 0.000011 | 0.124220 +/- 0.000000 | 0.292679 +/- 0.000000 |
| SMD | Fixed node-permuted incidence | 0.681471 +/- 0.000000 | 0.117144 +/- 0.000001 | 0.124220 +/- 0.000000 | 0.292679 +/- 0.000000 |
| SWAT | Learned dynamic incidence | 0.850022 +/- 0.000001 | 0.529264 +/- 0.000391 | 0.001134 +/- 0.000000 | 0.312057 +/- 0.000000 |
| SWAT | Static train-mean incidence | 0.850021 +/- 0.000002 | 0.529584 +/- 0.000811 | 0.001134 +/- 0.000000 | 0.312057 +/- 0.000000 |
| SWAT | Within-entity time-shuffled incidence | 0.849998 +/- 0.000008 | 0.527683 +/- 0.000025 | 0.001134 +/- 0.000000 | 0.312057 +/- 0.000000 |
| SWAT | Fixed node-permuted incidence | 0.849959 +/- 0.000080 | 0.528910 +/- 0.000616 | 0.001134 +/- 0.000000 | 0.312057 +/- 0.000000 |

## Learned incidence versus 100 degree/sparsity-exact random controls

| Dataset | ROC percentile | AP percentile | POT-F1 percentile | Event-F1 percentile |
|---|---:|---:|---:|---:|
| EXA | 0.400000 +/- 0.284371 | 0.806667 +/- 0.074087 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| PSM | 0.236667 +/- 0.069442 | 0.390000 +/- 0.127279 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| SMD | 0.336667 +/- 0.441236 | 0.686667 +/- 0.061824 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| SWAT | 0.923333 +/- 0.020548 | 0.943333 +/- 0.046428 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |

## Audit constraints

- Every counterfactual keeps feature evidence, the learned normal-train ECDF
  reference, the reliability decision, and rank-safe epsilon fixed.
- Random controls relabel node rows within each entity and preserve every learned
  incidence matrix's per-time degree profile, sparsity, weights, and column mass.
- Best-F1 and point adjustment are not used in this audit.
- Interpret empirical percentiles as randomization diagnostics, not independent
  confirmatory p-values, because the four benchmark test labels are already revealed.
