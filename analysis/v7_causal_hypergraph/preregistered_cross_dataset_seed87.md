# V7 frozen cross-dataset development screen (seed 87)

Frozen: 2026-09-10, before any V7 training or scoring on Exathlon, SMD,
SWaT, MSL, or SMAP.

## Purpose and boundary

PSM established a repeatable mechanism signal. This screen asks whether the
same frozen V7 mechanism transfers across the five remaining local datasets.
All five test sets were exposed during earlier versions, so this is development
evidence only. It cannot independently confirm SOTA and it must not be used to
tune dataset-specific V7 hyperparameters.

## Unchanged protocol

- use the V7 defaults in `preregistered_design.md` without modification;
- seed 87, five epochs, Adam 0.005/type1, batch 128, chronological normal-only
  80/20 train/validation, and evaluation step equal to the window length;
- predict every sensor for Exathlon/SMD/SWaT and only official target column 0
  for MSL/SMAP;
- SMD/MSL/SMAP use entity-separated windows, priors, calibration and PA, with
  one-hot entity context appended after scaling;
- no test loss during training; all component calibration uses normal training
  scores only; primary decisions use ROC-AUC and PR-AUC without PA.

Expected evaluated points are Exathlon 52,800; SMD 706,560; SWaT 449,856;
MSL 72,600; and SMAP 425,400.

## Fixed arms

1. `full`: causal prior + dynamic directed hypergraph forecast + relation score;
2. `no_prior`: no log-prior bias and no prior KL;
3. `temporal_only`: no hypergraph prediction path and prediction-error score;
4. `no_relation_score`: exact weight-zero score reconstructed from each full
   artifact, with no duplicate training.

## Per-dataset continuation gates

A dataset continues to seeds 90/98 only if all conditions hold:

1. every arm has the expected number of finite, aligned scores and two score
   components;
2. full ROC-AUC exceeds 0.5 and full PR-AUC exceeds anomaly prevalence;
3. relative to temporal-only, full is no worse than -0.01 on either primary
   metric and improves at least one by 0.01;
4. relative to no-prior, full is no worse than -0.01 on either primary metric
   and improves at least one by 0.005;
5. adding relation deviation reduces neither primary metric by more than 0.01.

The cross-dataset mechanism signal is considered sufficiently broad to reserve
an untouched external confirmation set only if at least three of five datasets
pass and at least two of the three multi-target datasets (Exathlon/SMD/SWaT)
pass. Failures remain reported and no seed may be replaced.

## Commands

```bash
for dataset in EXATHLON SMD SWAT MSL SMAP; do
  for arm in full no_prior temporal_only; do
    bash code/ISTAD/scripts/anomaly_detection/ISTAD_v7_causal_hypergraph.sh \
      "$dataset" "$arm" 0 87
  done
done

python analysis/evaluate_v7_cross_dataset_seed87.py
```
