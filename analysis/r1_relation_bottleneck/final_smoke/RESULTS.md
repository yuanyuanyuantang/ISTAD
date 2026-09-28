# ISTAD R1 masked-channel relation bottleneck

> All target cells are scored only while hidden. The causal encoder and
> decoder are channelwise, so HGAT is the only cross-channel path.
> Test-label Best-F1 is retained only as a diagnostic.

Decision: **PASS**

## Runs

| Dataset | Seed | Arm | ROC-AUC | AP | P99 F1 | Event F1 | Val masked MSE | Frozen removal delta | HGAT grad |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| PSM | 87 | Learned dynamic incidence | 0.669864 | 0.428794 | 0.080971 | 0.559547 | 0.225388 | 1.730363 | 0.034802 |
| PSM | 87 | Learned static incidence | 0.682422 | 0.429832 | 0.079804 | 0.542665 | 0.214794 | 1.017541 | 0.060704 |
| PSM | 87 | Fixed balanced-random incidence | 0.686131 | 0.457260 | 0.083998 | 0.529518 | 0.256101 | 0.557040 | 0.112837 |
| PSM | 87 | Without hypergraph message passing | 0.628695 | 0.394003 | 0.071856 | 0.521059 | 0.243869 | 0.000000 | 0.000000 |

## Predeclared gate

- [x] `all_dynamic_scores_differ_from_controls`
- [x] `dynamic_hgat_receives_gradient`
- [x] `dynamic_hgat_output_is_nonconstant`
- [x] `frozen_message_removal_increases_loss`
- [x] `retrained_dynamic_improves_masked_mse_by_at_least_2_percent`

Mean retrained dynamic MSE improvement over no-message: 7.578%

The smoke gate passed; the full 48-run R1 experiment may proceed.
