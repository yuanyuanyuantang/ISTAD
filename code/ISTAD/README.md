# ISTAD

Reference implementation of **ISTAD: Interpretable Rank-Safe Hypergraph Refinement of Causal Innovations for Multivariate Time-Series Anomaly Detection**.

This directory contains the model, training/evaluation pipeline, frozen experiment entry points, and utility code used for the paper. The repository root [README](../../README.md) is the authoritative record of the method, the frozen protocols, the reported results, and the evidence boundaries; the method draft is [paper/ISTAD_KBS_draft.md](../../paper/ISTAD_KBS_draft.md). This file documents the code itself and deliberately does not restate performance numbers, so that the code and the paper record cannot drift apart.

## Scope

The release contains the main model plus several frozen ablations and diagnostics. Their roles are defined in the root README, not in this file: some configurations are part of the main method, while others are controls or negative results that are kept for auditability. The shell scripts under `scripts/anomaly_detection/` are the frozen experiment entry points, including the protocols used for those negative results. They are not a general-purpose hyperparameter search interface.

## What is implemented

- **Neural backbone** (`models/ISTAD.py`): causal `Conv1d`, an optional hypergraph-attention branch, an optional temporal branch (TCN or KAN-TCN), and a reconstruction head (GRU, KANAD, or pointwise).
- **Training-only causal innovation scorer (V4)** (`utils/innovation.py`): a ridge-regularized vector autoregression fitted on normal training points, scored by one-step innovations. Coefficients, scales, pooling choice, and calibration references are fitted without test values or labels.
- **Lightweight hypergraph mixer and rank-safe refinement (V4-HG)**: a low-rank dynamic hypergraph over variables (`--istad_spatial_type lite`) whose incidence is used to refine the V4 score. The `rank_tiebreak` strategy is bounded by half a V4 empirical-CDF rank step and therefore cannot invert distinct V4 ranks.
- **Released variants**: HGST2 (`--istad_arch hgst2`), causal-prior hypergraph V7/V7.1 (`--istad_arch v7|v71`), relation bottleneck (`--istad_relation_bottleneck 1`), denoising/evidence-head training (`--istad_denoise 1`), RevIN and dual-head reconstruction (`--istad_revin`, `--istad_dual`), and target reconstruction/forecast objectives (`--istad_objective`). Availability of some flags is restricted by the validations in `run.py`.
- **Evaluation**: percentile thresholds, Best-F1 threshold search, raw and point-adjusted point metrics, ROC-AUC, PR-AUC, per-entity metrics when entity metadata is available, and event-level metrics (`utils/event_metrics.py`).
- **Explainability**: vanilla or SmoothGrad saliency maps, feature and temporal contributions, and per-segment localization summaries.

## Repository layout

```text
code/ISTAD/
├── run.py                       # CLI entry point
├── models/
│   ├── ISTAD.py                 # Model wrapper and architecture dispatch
│   └── istad_layers/            # Conv, HGAT, TCN/KAN-TCN, decoders, V7 graph, ...
├── data_provider/
│   ├── data_factory.py          # Dataset registry
│   └── data_loader.py           # Per-dataset loaders and entity-aware utilities
├── exp/
│   └── exp_anomaly_detection.py # Train/test loop, scoring, metrics, explainability
├── utils/
│   ├── innovation.py            # V4 innovation scorer and HGAT pooling
│   ├── causal_prior.py          # V7 training-only causal prior
│   ├── event_metrics.py         # Entity-safe event-level metrics
│   └── tools.py                 # Early stopping, point adjustment, Best-F1 search
├── scripts/anomaly_detection/   # Frozen experiment entry points
├── tests/                       # Unit and smoke tests
├── tools/                       # Diagnostic probes
├── requirements.txt
└── result_anomaly_detection.txt # Runtime log (appended by exp/), not a paper table
```

## Installation

The pinned environment targets Python 3.11 and PyTorch 2.5.1 with CUDA 12.1.

```bash
cd <release-root>
pip install -r code/ISTAD/requirements.txt
```

`requirements.txt` is a full environment snapshot and contains more packages than the model imports. CPU-only execution is supported with `--no_use_gpu`, although the released protocols were run on GPUs.

## Data

Datasets are not redistributed in this repository. `--root_path` must point to a directory containing the files below. For PSM, MSL, SMAP, SMD, and SWAT the loaders fall back to downloading `thuml/Time-Series-Library` from Hugging Face when the local files are missing; that fallback requires network access and the `datasets`/`huggingface_hub` packages. EXATHLON is local-only and raises an error if its files are absent.

| Dataset | Files expected in `--root_path` | Notes |
|---|---|---|
| PSM | `train.csv`, `test.csv`, `test_label.csv` | The first CSV column is treated as an index and dropped. |
| MSL | `MSL_train.npy`, `MSL_test.npy`, `MSL_test_label.npy` | Flattened NASA entity arrays; entity lengths are fixed in `data_loader.py`. |
| SMAP | `SMAP_train.npy`, `SMAP_test.npy`, `SMAP_test_label.npy` | Same convention as MSL. |
| SMD | `SMD_train.npy`, `SMD_test.npy`, `SMD_test_label.npy` | Entity lengths are fixed in `data_loader.py`. |
| SWAT | `swat_train2.csv`, `swat2.csv` | `swat2.csv` must carry the binary label in its last column. |
| EXATHLON | `Exathlon_train.npy`, `Exathlon_test.npy`, `Exathlon_test_label.npy` | Must be provided locally. |

Arrays are expected as `(time, channels)` numeric arrays; labels are binary per time point. Do not redistribute datasets whose licences do not permit it; obtain them from their official sources. For MSL/SMAP/SMD, entity-aware training (`--istad_entity_aware 1`) keeps windows and point adjustment inside each entity and requires `--istad_holdout_val 1`.

## Quick start

The frozen scripts under `scripts/anomaly_detection/` are the canonical entry points. The root README maps each script to the experiment it reproduces. Examples:

```bash
# Legacy KAN-TCN configuration on PSM
bash scripts/anomaly_detection/PSM/ISTAD.sh

# V4-HG main configuration: dataset, GPU, seed, mode (train or eval)
bash scripts/anomaly_detection/ISTAD_v4_hgat_integrated.sh PSM 0 87 train

# Evaluation-only pass from matching frozen checkpoints
bash scripts/anomaly_detection/ISTAD_v4_hgat_integrated.sh PSM 0 87 eval
```

The scripts read `ISTAD_DATA_ROOT`, `ISTAD_CHECKPOINT_ROOT`, and `ISTAD_RESULT_ROOT` where applicable; check the header of each script for its defaults. Evaluation-only runs require checkpoints produced by a training run with the same arguments and seed.

A direct CLI invocation, adapted from `scripts/anomaly_detection/PSM/ISTAD.sh`:

```bash
python -u run.py \
  --task_name anomaly_detection --is_training 1 \
  --root_path ./dataset/PSM --model_id PSM --model ISTAD --data PSM \
  --features M --seq_len 64 --enc_in 25 --c_out 25 \
  --istad_recon_type kanad --istad_branch_mode hgat_kan_tcn \
  --istad_kanad_order 6 --anomaly_ratio 1 \
  --use_bestf1_threshold 1 --bestf1_search_mode adaptive \
  --bestf1_coarse_step_num 200 --bestf1_fine_step_num 500 \
  --bestf1_use_adjustment 1 --istad_dropout 0.3 \
  --learning_rate 0.01 --batch_size 64 --num_workers 4 \
  --patience 5 --train_epochs 3 --seed 87
```

Use `python run.py --help` for the full argument list. `run.py` validates flag combinations and rejects unsupported ones.

## Key options

| Option | Values (default first) | Purpose |
|---|---|---|
| `--istad_arch` | `legacy`, `hgst2`, `v7`, `v71` | Architecture family. `legacy` covers the reconstruction backbone and V4/V4-HG scoring. |
| `--istad_branch_mode` | resolved to `hgat_kan_tcn` when omitted; `hgat`, `tcn`, `kan_tcn`, `hgat_tcn`, `hgat_kan_tcn` | Branch ablation for the legacy backbone. |
| `--istad_spatial_type` | `legacy`, `lite` | Dense legacy HGAT or the low-rank dynamic hypergraph mixer. |
| `--istad_recon_type` | `kanad`, `gru`, `pointwise` | Reconstruction head. |
| `--istad_score_mode` | `base_mean`, `evidence`, `innovation`, `innovation_fused`, `innovation_hgat`, `v7_fused` | Test-time score. The innovation modes use the training-only V4 scorer. |
| `--istad_objective` | `reconstruct`, `target_reconstruct`, `target_forecast` | Reconstruct all channels, selected targets, or causally forecast targets. |
| `--istad_hgat_relation_mode` | `dynamic`, `static`, `fixed_random`, `no_message` | Architecture-level HGAT controls. |
| `--istad_hgat_input` | `raw`, `innovation` | Input domain for the lightweight HGAT branch. |
| `--istad_entity_aware` | `0`, `1` | Respect known entity boundaries (MSL/SMAP/SMD only; requires `--istad_holdout_val 1`). |
| `--istad_holdout_val` | `0`, `1` | Use a disjoint chronological 80/20 train/validation split. |
| `--istad_eval_step` | `0`, positive integer | Evaluation/scoring window stride; `0` keeps the loader default. |
| `--istad_threshold_source` | `combined`, `train` | Pool for the percentile threshold. |
| `--use_bestf1_threshold` | `1`, `0` | Search the Best-F1 threshold on test labels. |
| `--bestf1_use_adjustment` | `1`, `0` | Apply point adjustment inside the Best-F1 search. |

## Training and evaluation protocol

- **Normalisation and split.** Each loader fits a `StandardScaler` on the training portion and transforms the test split. By default the legacy split trains on the complete training array and reuses its final 20% as validation; `--istad_holdout_val 1` switches to a disjoint chronological holdout and is required for entity-aware runs.
- **Loss and stopping.** The default objective is mean-squared reconstruction error, with optional denoising, evidence-head, dual-head, relation-bottleneck, or V7 terms enabled by their flags. Early stopping monitors the validation loss, saves `checkpoint.pth` on improvement, and the best checkpoint is restored before testing.
- **Scoring.** With `--istad_score_mode base_mean`, the per-point score is the mean squared reconstruction error over the observed channels. The innovation modes replace or refine that score with the training-only V4 branch.
- **Thresholds and point adjustment.** Both threshold paths report raw point metrics. The percentile path additionally applies point adjustment (PA) before writing the headline metric; Best-F1 search uses PA by default (`--bestf1_use_adjustment 1`). PA is entity-safe when entity metadata is present. The Best-F1 threshold is selected using test labels and is an oracle upper bound, not a deployable operating point. The root README defines how these numbers may be reported.
- **Outputs.**
  - `checkpoints/<setting>/checkpoint.pth` (or the directory given by `--checkpoints`).
  - `test_results/<setting>/` (override with `ISTAD_RESULT_ROOT`): `percentile_threshold_results*.json`, `bestf1_threshold_results*.json`, `anomaly_score_threshold*.png`, optional `point_scores*.npz`, and training-time summaries.
  - `result_anomaly_detection.txt` in the working directory: an appended text log of the percentile and Best-F1 results.
  - With explainability enabled (`--istad_enable_explain 1`, default): saliency maps, reconstruction-error maps, feature/temporal contributions, the feature-localisation CSV, and the explainability/segment JSON summaries.

## Reproducing the paper experiments

Use the root [README](../../README.md), sections 6–7, as the index of frozen scripts and evaluators, and `analysis/` for the evaluation code and machine-readable results. In particular, `scripts/anomaly_detection/ISTAD_v4_hgat_paper_eval.sh` regenerates the float64 score artefacts from frozen checkpoints and then runs the paper-facing evaluators; it performs no training and no hyperparameter selection.

## Tests

From the release root:

```bash
python -m unittest discover -s code/ISTAD/tests -v
```

The tests are unit and smoke tests. A few of them reference `analysis/` and therefore expect the full release layout rather than this directory in isolation.

## License

MIT. See [../../LICENSE](../../LICENSE).

## Citation

If you use this code, please cite the ISTAD paper:

> ISTAD: Interpretable Rank-Safe Hypergraph Refinement of Causal Innovations for Multivariate Time-Series Anomaly Detection.

The manuscript is under review. Author list, venue, and DOI will be added here once they are final.