# V7 causal-prior directed hypergraph: development results

Evaluated: 2026-09-10. The architecture was frozen in
`preregistered_design.md`; the seed-87 continuation rules and seed-90/98
confirmation rules were written before their respective runs.

## PSM three-seed result

Values are mean ± population standard deviation over seeds 87/90/98. ROC-AUC
and PR-AUC are the primary metrics and use no point adjustment.

| Arm | ROC-AUC | PR-AUC | Train-p99 raw F1 | Train-p99+PA | Best raw F1 | Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|
| V7 full | **0.6568±0.0255** | **0.4448±0.0323** | **0.0693±0.0183** | 0.9673±0.0099 | **0.4703±0.0149** | 0.9804±0.0017 |
| no causal prior | 0.5965±0.0107 | 0.3638±0.0048 | 0.0328±0.0052 | 0.9474±0.0144 | 0.4446±0.0059 | 0.9713±0.0096 |
| temporal only | 0.6071±0.0237 | 0.3794±0.0138 | 0.0511±0.0033 | 0.9756±0.0070 | 0.4496±0.0139 | 0.9835±0.0008 |
| no relation score¹ | 0.6208±0.0274 | 0.3947±0.0240 | 0.0554±0.0065 | **0.9840±0.0010** | 0.4549±0.0153 | **0.9850±0.0004** |

¹ Exact scoring ablation reconstructed from the full artifacts; no model was
retrained. The seed-87 reconstructed arrays are byte-for-byte equal to the
separately trained/evaluated `no_relation_score` arm.

| Full minus control | Δ ROC-AUC | Δ PR-AUC | Positive paired seeds |
|---|---:|---:|---:|
| temporal only | +0.0497 | +0.0654 | 3/3 on both |
| no causal prior | +0.0603 | +0.0810 | 3/3 on both |
| no relation score | +0.0360 | +0.0501 | 3/3 on both |

All seven frozen confirmation gates passed. Full V7 has 70,200 trainable
parameters; the active temporal-only path has 69,271, so the directed
hypergraph mechanism adds 929 parameters.

## Interpretation boundary

The result supports the internal mechanism claim: the normal-training causal
prior, hypergraph forecast path, and relation-deviation score all improve PSM
ranking consistently. It does not establish SOTA. Full V7 remains below the
earlier HGAT-Lite PSM development means of ROC-AUC 0.7466 and PR-AUC 0.5547,
and PSM labels were already exposed before V7. Point adjustment also reverses
some arm rankings, so the PA columns must not be used to claim that the V7
mechanism is superior.

Exact artifacts, hashes, per-seed metrics, and gate decisions are in
`psm_seed87.json` and `psm_multiseed_confirmation.json`.

## Five-dataset frozen seed-87 screen

After the PSM result, the unchanged V7 configuration was frozen in
`preregistered_cross_dataset_seed87.md` and run on EXATHLON, SMD, SWAT, MSL,
and SMAP.  All 15 trained arms are complete and finite.  The table reports
full V7 and the exact prediction-only scoring ablation reconstructed from the
same full artifact; ROC-AUC and PR-AUC do not use point adjustment.

| Dataset | Full ROC | Full AP | Prediction-only ROC | Prediction-only AP | Full train-p99+PA | Full Best-F1+PA |
|---|---:|---:|---:|---:|---:|---:|
| EXATHLON | 0.8576 | 0.4716 | **0.8633** | **0.4959** | 0.0656 | 0.9490 |
| SMD | 0.7172 | 0.1326 | **0.7179** | **0.1684** | 0.7156 | 0.7724 |
| SWAT | 0.6242 | 0.1492 | **0.7702** | **0.3137** | 0.4123 | 0.9222 |
| MSL | 0.4505 | 0.0897 | **0.5161** | **0.1182** | 0.4415 | 0.4919 |
| SMAP | 0.6376 | **0.3208** | **0.6687** | 0.2727 | 0.8545 | 0.8562 |

The fixed 0.25 relation-deviation score reduced at least one primary ranking
metric on every dataset, so the relation-score guard failed 5/5.  The
hypergraph gate also failed on SMD, SWAT, MSL, and SMAP; the prior gate failed
on SWAT and MSL.  Consequently no dataset passed all five preregistered gates,
the required 3/5 overall and 2/3 multi-target support conditions failed, and
seeds 90/98 were **not run**.  This is the frozen stopping decision, not a
missing experiment.

The screen falsifies the cross-dataset claim for V7 as specified.  In
particular, the favorable PSM relation-score result did not generalize.  The
next development version may retain the causal-prior hypergraph as a
prediction mechanism, but it must remove the fixed additive relation score
and make the graph correction nested with a temporal baseline.  Any such
version is a new development experiment; these revealed labels cannot be
reused as independent confirmation evidence.  Exact values, artifact hashes,
and decisions are in `cross_dataset_seed87.json`.

## V7.1 nested-residual development screen

V7.1 was frozen in `preregistered_v71_residual_seed87.md` before training.  It
removed the additive relation score, separated the temporal and graph
decoders, added a learned per-target graph residual gate, and replaced the
hard prior bias with a learned prior mixture gate.  This made the graph path a
nested correction to an otherwise identical temporal forecaster.

| Dataset | Full ROC | Full AP | Train-p99+PA | Best-F1+PA | Δ vs temporal ROC/AP | Δ vs no-prior ROC/AP | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| EXATHLON | 0.8190 | 0.4243 | 0.9274 | 0.9540 | +0.0010 / +0.0024 | -0.0045 / -0.0042 | No |
| PSM | 0.5689 | 0.3559 | 0.9846 | 0.9850 | -0.0125 / -0.0128 | -0.0098 / -0.0077 | No |
| SMD | 0.7171 | 0.1624 | 0.8177 | 0.8781 | +0.0112 / +0.0028 | -0.0030 / -0.0011 | No |
| SWAT | 0.6335 | 0.1518 | 0.9407 | 0.9555 | +0.0665 / +0.0014 | -0.0761 / -0.0470 | No |
| MSL | **0.6307** | **0.1447** | 0.5960 | 0.6838 | +0.1422 / +0.0471 | +0.0594 / +0.0166 | **Yes** |
| SMAP | 0.6510 | 0.2866 | 0.7844 | 0.8602 | +0.1732 / +0.1251 | -0.0219 / -0.0490 | No |

All 18 artifacts passed completeness and finite-value checks, and every full
model gate remained strictly inside (0, 1).  Only MSL passed all per-dataset
gates: 1/6 overall and 0/4 multi-target datasets passed, below the frozen 4/6
and 3/4 continuation thresholds.  Seeds 90/98 were therefore not run.

V7.1 shows that nested residualization prevents a graph path from universally
destroying the temporal baseline, but it does not validate the causal prior as
a cross-dataset mechanism.  High PA values on PSM and SWAT again coexist with
weak ROC/AP and are not evidence for adoption.  Exact metrics and hashes are
in `v71_seed87.json`.  Further iteration on these same revealed labels would
be benchmark fitting, so the V7/V7.1 architecture campaign is closed pending a
new data split or genuinely unseen dataset.
