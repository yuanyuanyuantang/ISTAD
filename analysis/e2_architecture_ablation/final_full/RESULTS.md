# ISTAD E2 reconstruction-path architecture ablation

> Frozen protocol: all arms use the same reconstruction-error score,
> train-only 99th-percentile threshold, data split, decoder dimensions,
> optimization settings, and paired seeds. Best raw F1 is an oracle
> diagnostic and is not a deployable primary metric.

Status: `ANALYZED_FROM_48_TRAINED_MODELS`

## Per-run metrics

| Dataset | Seed | Arm | ROC-AUC | AP | Train-P99 raw F1 | Event-F1 | Best raw F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| EXA | 87 | Learned dynamic incidence | 0.886937 | 0.592427 | 0.205109 | 0.714903 | 0.650208 |
| EXA | 87 | Learned static incidence | 0.884923 | 0.589838 | 0.202193 | 0.715031 | 0.648388 |
| EXA | 87 | Fixed balanced-random incidence | 0.887111 | 0.593252 | 0.206012 | 0.715354 | 0.649904 |
| EXA | 87 | Without hypergraph message passing | 0.887172 | 0.593309 | 0.206940 | 0.717222 | 0.649013 |
| EXA | 90 | Learned dynamic incidence | 0.883211 | 0.601063 | 0.214375 | 0.724280 | 0.662095 |
| EXA | 90 | Learned static incidence | 0.882784 | 0.599680 | 0.213427 | 0.723937 | 0.662020 |
| EXA | 90 | Fixed balanced-random incidence | 0.878967 | 0.596355 | 0.217027 | 0.722454 | 0.659741 |
| EXA | 90 | Without hypergraph message passing | 0.884116 | 0.599674 | 0.213981 | 0.720093 | 0.661420 |
| EXA | 98 | Learned dynamic incidence | 0.881140 | 0.597221 | 0.208014 | 0.720305 | 0.657290 |
| EXA | 98 | Learned static incidence | 0.881882 | 0.597797 | 0.206836 | 0.723198 | 0.659656 |
| EXA | 98 | Fixed balanced-random incidence | 0.880299 | 0.597215 | 0.206914 | 0.723198 | 0.656786 |
| EXA | 98 | Without hypergraph message passing | 0.881935 | 0.598627 | 0.208543 | 0.725365 | 0.661201 |
| PSM | 87 | Learned dynamic incidence | 0.773123 | 0.562639 | 0.478143 | 0.610574 | 0.595728 |
| PSM | 87 | Learned static incidence | 0.779277 | 0.566807 | 0.465592 | 0.655527 | 0.594933 |
| PSM | 87 | Fixed balanced-random incidence | 0.777372 | 0.568726 | 0.472631 | 0.641275 | 0.600839 |
| PSM | 87 | Without hypergraph message passing | 0.768954 | 0.572033 | 0.469027 | 0.656558 | 0.590282 |
| PSM | 90 | Learned dynamic incidence | 0.774410 | 0.560505 | 0.476502 | 0.613511 | 0.610736 |
| PSM | 90 | Learned static incidence | 0.762473 | 0.557733 | 0.472496 | 0.611831 | 0.586231 |
| PSM | 90 | Fixed balanced-random incidence | 0.768720 | 0.563484 | 0.443699 | 0.616509 | 0.595192 |
| PSM | 90 | Without hypergraph message passing | 0.755708 | 0.557119 | 0.465982 | 0.620949 | 0.578193 |
| PSM | 98 | Learned dynamic incidence | 0.782155 | 0.570928 | 0.464924 | 0.651788 | 0.620978 |
| PSM | 98 | Learned static incidence | 0.779053 | 0.567105 | 0.476862 | 0.614771 | 0.619012 |
| PSM | 98 | Fixed balanced-random incidence | 0.779294 | 0.571070 | 0.478269 | 0.601656 | 0.613312 |
| PSM | 98 | Without hypergraph message passing | 0.771957 | 0.575225 | 0.473604 | 0.669568 | 0.588292 |
| SMD | 87 | Learned dynamic incidence | 0.747765 | 0.147175 | 0.225237 | 0.246766 | 0.230290 |
| SMD | 87 | Learned static incidence | 0.742318 | 0.149425 | 0.239671 | 0.237776 | 0.240026 |
| SMD | 87 | Fixed balanced-random incidence | 0.745920 | 0.146795 | 0.233342 | 0.235827 | 0.234462 |
| SMD | 87 | Without hypergraph message passing | 0.742333 | 0.144870 | 0.231063 | 0.231284 | 0.231868 |
| SMD | 90 | Learned dynamic incidence | 0.752059 | 0.155223 | 0.236652 | 0.219943 | 0.238276 |
| SMD | 90 | Learned static incidence | 0.749013 | 0.155547 | 0.244284 | 0.219072 | 0.245197 |
| SMD | 90 | Fixed balanced-random incidence | 0.750764 | 0.153795 | 0.238799 | 0.225291 | 0.240436 |
| SMD | 90 | Without hypergraph message passing | 0.747965 | 0.153772 | 0.243165 | 0.206813 | 0.243313 |
| SMD | 98 | Learned dynamic incidence | 0.747037 | 0.150902 | 0.242543 | 0.202520 | 0.243242 |
| SMD | 98 | Learned static incidence | 0.749925 | 0.152458 | 0.242281 | 0.192820 | 0.243029 |
| SMD | 98 | Fixed balanced-random incidence | 0.751787 | 0.155945 | 0.246074 | 0.197045 | 0.246239 |
| SMD | 98 | Without hypergraph message passing | 0.740059 | 0.149177 | 0.237381 | 0.201343 | 0.237459 |
| SWAT | 87 | Learned dynamic incidence | 0.823151 | 0.719200 | 0.289896 | 0.262332 | 0.774307 |
| SWAT | 87 | Learned static incidence | 0.820509 | 0.717040 | 0.287885 | 0.218049 | 0.774354 |
| SWAT | 87 | Fixed balanced-random incidence | 0.822417 | 0.717985 | 0.287523 | 0.153009 | 0.774080 |
| SWAT | 87 | Without hypergraph message passing | 0.822721 | 0.720405 | 0.287881 | 0.181264 | 0.774383 |
| SWAT | 90 | Learned dynamic incidence | 0.823242 | 0.717595 | 0.289698 | 0.289855 | 0.773077 |
| SWAT | 90 | Learned static incidence | 0.824181 | 0.717016 | 0.287680 | 0.301386 | 0.774072 |
| SWAT | 90 | Fixed balanced-random incidence | 0.822080 | 0.714424 | 0.288967 | 0.329862 | 0.774099 |
| SWAT | 90 | Without hypergraph message passing | 0.822717 | 0.716778 | 0.288666 | 0.215803 | 0.774160 |
| SWAT | 98 | Learned dynamic incidence | 0.823754 | 0.716035 | 0.284053 | 0.332029 | 0.774181 |
| SWAT | 98 | Learned static incidence | 0.820233 | 0.715741 | 0.282778 | 0.263144 | 0.774316 |
| SWAT | 98 | Fixed balanced-random incidence | 0.820835 | 0.712850 | 0.285843 | 0.264917 | 0.774361 |
| SWAT | 98 | Without hypergraph message passing | 0.819700 | 0.713171 | 0.280547 | 0.223100 | 0.774389 |

## Aggregate metrics

| Arm | ROC-AUC | AP | Train-P99 raw F1 | Event-F1 | Best raw F1 |
|---|---:|---:|---:|---:|---:|
| Learned dynamic incidence | 0.808165 +/- 0.051181 | 0.507576 +/- 0.213586 | 0.301262 +/- 0.103373 | 0.465734 +/- 0.211935 | 0.569201 +/- 0.200901 |
| Learned static incidence | 0.806381 +/- 0.051967 | 0.507182 +/- 0.212590 | 0.301832 +/- 0.101991 | 0.456379 +/- 0.221796 | 0.568436 +/- 0.198400 |
| Fixed balanced-random incidence | 0.807130 +/- 0.050562 | 0.507658 +/- 0.212587 | 0.300425 +/- 0.099241 | 0.452200 +/- 0.224309 | 0.568288 +/- 0.199266 |
| Without hypergraph message passing | 0.803778 +/- 0.054753 | 0.507847 +/- 0.214441 | 0.300565 +/- 0.101339 | 0.447447 +/- 0.239368 | 0.563664 +/- 0.200025 |

## Learned dynamic incidence minus controls

| Control | Delta ROC-AUC | W/T/L | Delta AP | W/T/L |
|---|---:|---:|---:|---:|
| Learned static incidence | +0.001784 | 8/0/4 | +0.000394 | 7/0/5 |
| Fixed balanced-random incidence | +0.001035 | 9/0/3 | -0.000082 | 7/0/5 |
| Without hypergraph message passing | +0.004387 | 9/0/3 | -0.000270 | 7/0/5 |

## Interpretation boundary

- This experiment audits the neural reconstruction path, not the closed-form
  causal-innovation score used by the current paper-facing detector.
- No point adjustment is used in the reported E2 metrics.
- The 99th-percentile threshold uses normal-training scores only.
- Best raw F1 uses test labels and is reported only as an oracle diagnostic.
- These benchmark labels have already been inspected, so the experiment is
  developmental evidence rather than independent confirmation.
