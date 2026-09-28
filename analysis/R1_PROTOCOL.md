# R1 masked-channel relation bottleneck

## Purpose

E2 showed that the original reconstruction route can work without HGAT. R1 is
an isolated architecture experiment that asks a narrower question: does learned
hypergraph message passing improve reconstruction when it is the only route for
cross-channel information?

R1 does not replace the frozen ISTAD model and does not change any E1/E2 output.
The new behavior is disabled unless `--istad_relation_bottleneck 1` is supplied.

## Isolation constraints

1. At every training time step, exactly about 20% of channels are replaced by
   zero in standardized input space.
2. Loss is computed only on the hidden cells.
3. The causal convolution is depthwise, so it cannot mix channels.
4. The decoder predicts channel `c` only from its own causal state and HGAT
   message for channel `c`.
5. During scoring, deterministic folds hide every time-channel cell exactly
   once. The anomaly score is the mean hidden-cell squared error across channels.
6. All four arms have the same trainable parameter shapes and optimization
   settings. They differ only in the HGAT relation control.

The four arms are learned dynamic incidence, learned static incidence, fixed
balanced-random incidence, and no hypergraph message passing.

## Leakage policy

Training, early stopping, mask generation, score calibration, and thresholding
use normal training/validation data only. Test labels are used for final metric
calculation. Best raw F1 is an oracle diagnostic and is not a primary result.

## Smoke gate

Run only `PSM / seed 87 / four arms` first. The summarizer reports `PASS` only
when all of the following hold:

- dynamic scores differ from every control score array;
- the dynamic HGAT receives non-zero gradients;
- its message output is non-constant;
- removing its message from the frozen trained model increases masked loss;
- retrained dynamic HGAT lowers normal-validation masked MSE by at least 2%
  relative to the retrained no-message arm.

Anomaly AP/F1 is reported but is not used to tune or pass the smoke gate.

## Server commands

From the release root:

```bash
mkdir -p analysis/r1_relation_bottleneck
bash analysis/run_r1_relation_bottleneck.sh smoke 0 2>&1 | tee analysis/r1_relation_bottleneck/smoke.log
```

Inspect:

```bash
cat analysis/r1_relation_bottleneck/final_smoke/RESULTS.md
sha256sum analysis/ISTAD_R1_smoke_results_20260912.tar.gz \
  analysis/r1_relation_bottleneck/smoke.log
```

Do not run the full experiment unless `RESULTS.md` says `Decision: PASS`.
If it passes, the predeclared full command is:

```bash
bash analysis/run_r1_relation_bottleneck.sh full 0 2>&1 | tee analysis/r1_relation_bottleneck/full.log
```

The full scope is 4 datasets x 3 seeds x 4 relation controls = 48 runs.
