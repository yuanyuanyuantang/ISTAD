# ISTAD R1 masked-channel relation bottleneck

> All target cells are scored only while hidden. The causal encoder and
> decoder are channelwise, so HGAT is the only cross-channel path.
> Test-label Best-F1 is retained only as a diagnostic.

Decision: **FAIL**

## Runs

| Dataset | Seed | Arm | ROC-AUC | AP | P99 F1 | Event F1 | Val masked MSE | Frozen removal delta | HGAT grad |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| EXATHLON | 87 | Learned dynamic incidence | 0.877078 | 0.599291 | 0.216135 | 0.718544 | 0.976442 | 0.007284 | 0.017750 |
| EXATHLON | 87 | Learned static incidence | 0.876697 | 0.598276 | 0.215885 | 0.718992 | 0.974722 | 0.007682 | 0.027333 |
| EXATHLON | 87 | Fixed balanced-random incidence | 0.877776 | 0.599876 | 0.215215 | 0.717437 | 0.973219 | 0.008803 | 0.035193 |
| EXATHLON | 87 | Without hypergraph message passing | 0.876189 | 0.598787 | 0.217000 | 0.717437 | 0.980743 | 0.000000 | 0.000000 |
| EXATHLON | 90 | Learned dynamic incidence | 0.877463 | 0.598990 | 0.214849 | 0.720093 | 0.976830 | 0.006438 | 0.017154 |
| EXATHLON | 90 | Learned static incidence | 0.878450 | 0.603731 | 0.217641 | 0.719653 | 0.990052 | 0.004088 | 0.024325 |
| EXATHLON | 90 | Fixed balanced-random incidence | 0.877340 | 0.599533 | 0.215134 | 0.716019 | 0.977642 | 0.005083 | 0.018282 |
| EXATHLON | 90 | Without hypergraph message passing | 0.878682 | 0.603760 | 0.217418 | 0.718873 | 0.994093 | 0.000000 | 0.000000 |
| EXATHLON | 98 | Learned dynamic incidence | 0.878142 | 0.602148 | 0.215242 | 0.719653 | 0.973062 | 0.008703 | 0.021560 |
| EXATHLON | 98 | Learned static incidence | 0.877583 | 0.601567 | 0.215662 | 0.719321 | 0.975854 | 0.005565 | 0.023334 |
| EXATHLON | 98 | Fixed balanced-random incidence | 0.878572 | 0.602548 | 0.215885 | 0.719321 | 0.975466 | 0.006007 | 0.032616 |
| EXATHLON | 98 | Without hypergraph message passing | 0.878592 | 0.602531 | 0.215465 | 0.720428 | 0.979332 | 0.000000 | 0.000000 |
| PSM | 87 | Learned dynamic incidence | 0.669864 | 0.428794 | 0.080971 | 0.559547 | 0.225388 | 1.730363 | 0.034802 |
| PSM | 87 | Learned static incidence | 0.682422 | 0.429832 | 0.079804 | 0.542665 | 0.214794 | 1.017541 | 0.060704 |
| PSM | 87 | Fixed balanced-random incidence | 0.686131 | 0.457260 | 0.083998 | 0.529518 | 0.256101 | 0.557040 | 0.112837 |
| PSM | 87 | Without hypergraph message passing | 0.628695 | 0.394003 | 0.071856 | 0.521059 | 0.243869 | 0.000000 | 0.000000 |
| PSM | 90 | Learned dynamic incidence | 0.624404 | 0.462687 | 0.366570 | 0.553067 | 0.406764 | 0.064686 | 0.042611 |
| PSM | 90 | Learned static incidence | 0.634629 | 0.475335 | 0.369623 | 0.554426 | 0.384667 | 0.239309 | 0.042345 |
| PSM | 90 | Fixed balanced-random incidence | 0.633320 | 0.464045 | 0.368496 | 0.553398 | 0.383737 | 0.289083 | 0.074361 |
| PSM | 90 | Without hypergraph message passing | 0.629980 | 0.468176 | 0.359797 | 0.561092 | 0.409781 | 0.000000 | 0.000000 |
| PSM | 98 | Learned dynamic incidence | 0.626289 | 0.391929 | 0.073408 | 0.517314 | 0.244145 | 3.328636 | 0.043069 |
| PSM | 98 | Learned static incidence | 0.638262 | 0.396906 | 0.073599 | 0.533051 | 0.231976 | 1.170135 | 0.056358 |
| PSM | 98 | Fixed balanced-random incidence | 0.657954 | 0.406574 | 0.077708 | 0.543531 | 0.234199 | 1.783059 | 0.056993 |
| PSM | 98 | Without hypergraph message passing | 0.597889 | 0.368993 | 0.067692 | 0.520548 | 0.244734 | 0.000000 | 0.000000 |
| SMD | 87 | Learned dynamic incidence | 0.706187 | 0.119412 | 0.116502 | 0.237707 | 0.279471 | 0.308066 | 0.064148 |
| SMD | 87 | Learned static incidence | 0.703420 | 0.117697 | 0.113953 | 0.231765 | 0.271965 | 0.219705 | 0.079330 |
| SMD | 87 | Fixed balanced-random incidence | 0.708992 | 0.118460 | 0.113563 | 0.232184 | 0.272118 | 0.198814 | 0.154677 |
| SMD | 87 | Without hypergraph message passing | 0.696733 | 0.117018 | 0.117869 | 0.237012 | 0.263458 | 0.000000 | 0.000000 |
| SMD | 90 | Learned dynamic incidence | 0.671449 | 0.117338 | 0.126200 | 0.259254 | 0.257949 | 0.312595 | 0.065152 |
| SMD | 90 | Learned static incidence | 0.672542 | 0.108829 | 0.111390 | 0.239031 | 0.287861 | 0.130813 | 0.064244 |
| SMD | 90 | Fixed balanced-random incidence | 0.685773 | 0.104534 | 0.103905 | 0.243446 | 0.285713 | 0.174500 | 0.106056 |
| SMD | 90 | Without hypergraph message passing | 0.668590 | 0.112783 | 0.126565 | 0.259956 | 0.289588 | 0.000000 | 0.000000 |
| SMD | 98 | Learned dynamic incidence | 0.695343 | 0.116741 | 0.113486 | 0.236553 | 0.225097 | 0.283163 | 0.068985 |
| SMD | 98 | Learned static incidence | 0.692127 | 0.116241 | 0.115004 | 0.238091 | 0.229858 | 0.158017 | 0.094055 |
| SMD | 98 | Fixed balanced-random incidence | 0.704336 | 0.117135 | 0.115194 | 0.239160 | 0.237346 | 0.213591 | 0.108910 |
| SMD | 98 | Without hypergraph message passing | 0.692343 | 0.113548 | 0.116891 | 0.239093 | 0.258547 | 0.000000 | 0.000000 |
| SWAT | 87 | Learned dynamic incidence | 0.797831 | 0.698781 | 0.286994 | 0.256583 | 0.060770 | 1.091349 | 0.026246 |
| SWAT | 87 | Learned static incidence | 0.796766 | 0.700359 | 0.288874 | 0.238756 | 0.063082 | 0.335456 | 0.051205 |
| SWAT | 87 | Fixed balanced-random incidence | 0.793743 | 0.693962 | 0.286420 | 0.250064 | 0.066287 | 0.308032 | 0.075417 |
| SWAT | 87 | Without hypergraph message passing | 0.842430 | 0.694801 | 0.652093 | 0.131138 | 0.064422 | 0.000000 | 0.000000 |
| SWAT | 90 | Learned dynamic incidence | 0.808313 | 0.712082 | 0.290846 | 0.220963 | 0.079926 | 0.925906 | 0.025549 |
| SWAT | 90 | Learned static incidence | 0.809446 | 0.711318 | 0.306746 | 0.155239 | 0.084706 | 0.372183 | 0.030898 |
| SWAT | 90 | Fixed balanced-random incidence | 0.808436 | 0.715735 | 0.304616 | 0.168590 | 0.087150 | 0.418699 | 0.086752 |
| SWAT | 90 | Without hypergraph message passing | 0.807574 | 0.711181 | 0.304397 | 0.181930 | 0.086087 | 0.000000 | 0.000000 |
| SWAT | 98 | Learned dynamic incidence | 0.831365 | 0.726214 | 0.674149 | 0.126459 | 0.054513 | 1.131973 | 0.022460 |
| SWAT | 98 | Learned static incidence | 0.837875 | 0.734840 | 0.674503 | 0.133035 | 0.058126 | 0.314215 | 0.042663 |
| SWAT | 98 | Fixed balanced-random incidence | 0.837227 | 0.736898 | 0.662162 | 0.126865 | 0.064761 | 0.295719 | 0.087757 |
| SWAT | 98 | Without hypergraph message passing | 0.838920 | 0.739494 | 0.680949 | 0.136688 | 0.061284 | 0.000000 | 0.000000 |

## Predeclared gate

- [x] `all_dynamic_scores_differ_from_controls`
- [x] `dynamic_hgat_receives_gradient`
- [x] `dynamic_hgat_output_is_nonconstant`
- [x] `frozen_message_removal_increases_loss`
- [ ] `retrained_dynamic_improves_masked_mse_by_at_least_2_percent`

Mean retrained dynamic MSE improvement over no-message: 4.419%

Interpret the full multi-seed evidence before changing the manuscript.
