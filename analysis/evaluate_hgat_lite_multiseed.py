#!/usr/bin/env python3
"""Aggregate the frozen HGAT-Lite + train-ECDF multi-seed confirmation.

The training entry point has already run the same adaptive 200+500 Best-F1
search for every score artifact.  This script validates and reuses those JSON
records, and independently recomputes ranking metrics, train-p99 neural metrics,
and the frozen SPOT/POT + point-adjusted metrics from the dumped arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_istad_innovation_pa import (  # noqa: E402
    POT_LM,
    TABLE_BEST,
    _fixed_pa_metrics,
    _pot_threshold,
)


DATASETS = ("EXA", "PSM", "SMD", "SWAT")
SEEDS = (87, 90, 98)
FORMAL_V4_POT_MACRO = 0.9022703904133883
TAIL_SAFETY_MARGIN = 1.04
SCORE_FILE = (
    "point_scores_innovation_fused_w0p001_calibrated_prob_train_ecdf.npz"
)
BEST_FILE = (
    "bestf1_threshold_results_innovation_fused_w0p001_calibrated_prob_train_ecdf.json"
)


def _result_dir(dataset: str, seed: int) -> Path:
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    description = f"hgatabl_lite_s{seed}" if dataset == "PSM" and seed != 87 else f"v4litehg_s{seed}"
    matches = [
        path
        for path in (REPO_ROOT / "code" / "ISTAD" / "test_results").glob(
            f"anomaly_detection_{long_name}_*_{description}_0"
        )
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {dataset} seed-{seed} result, found {len(matches)}: {matches}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
    }


def _evaluate_one(dataset: str, seed: int) -> dict:
    result_dir = _result_dir(dataset, seed)
    score_path = result_dir / SCORE_FILE
    best_path = result_dir / BEST_FILE
    if not score_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {result_dir}")

    with np.load(score_path) as artifact:
        required = {"score", "train_score", "model_score", "train_model_score", "label"}
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{score_path} is missing {sorted(missing)}")
        score = artifact["score"].reshape(-1).astype(np.float64)
        train_score = artifact["train_score"].reshape(-1).astype(np.float64)
        model_score = artifact["model_score"].reshape(-1).astype(np.float64)
        train_model_score = artifact["train_model_score"].reshape(-1).astype(np.float64)
        label = artifact["label"].reshape(-1).astype(np.int8)
        entity_ids = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files
            else None
        )

    arrays = (score, train_score, model_score, train_model_score)
    finite = bool(all(np.isfinite(array).all() for array in arrays))
    lengths_match = bool(len(score) == len(model_score) == len(label))
    if not finite or not lengths_match:
        raise ValueError(
            f"Invalid arrays for {dataset} seed {seed}: finite={finite}, "
            f"lengths_match={lengths_match}"
        )

    if dataset == "SMD":
        if entity_ids is None or len(entity_ids) != len(label):
            raise ValueError("SMD result lacks aligned entity IDs")
        entity_count = int(len(np.unique(entity_ids)))
        if entity_count != 28:
            raise ValueError(f"Expected 28 SMD entities, got {entity_count}")
    else:
        entity_count = None

    # float64 clipping is intentional: float32 1.0 cannot represent 1-1e-12.
    train_pot_score = -np.log1p(-np.clip(train_score, 0.0, 1.0 - 1e-12))
    test_pot_score = -np.log1p(-np.clip(score, 0.0, 1.0 - 1e-12))
    level, multiplier = POT_LM[dataset]
    threshold, raw_threshold, effective_level = _pot_threshold(
        train_pot_score,
        test_pot_score,
        level,
        multiplier,
        tail_safety_margin=TAIL_SAFETY_MARGIN,
    )
    pot = _fixed_pa_metrics(
        test_pot_score,
        label,
        threshold,
        entity_ids=entity_ids,
    )
    pot.update(
        {
            "configured_level": level,
            "effective_level": effective_level,
            "base_multiplier": multiplier,
            "tail_safety_margin": TAIL_SAFETY_MARGIN,
            "raw_spot_threshold": raw_threshold,
        }
    )

    best = json.loads(best_path.read_text(encoding="utf-8"))
    if best.get("search_method") != "adaptive_two_stage":
        raise ValueError(f"Unexpected Best-F1 search in {best_path}")
    if best.get("coarse_step_num") != 200 or best.get("fine_step_num") != 500:
        raise ValueError(f"Unexpected Best-F1 grid in {best_path}")
    if dataset == "SMD" and not best.get("entity_aware"):
        raise ValueError("SMD Best-F1 result was not entity-aware")

    neural_threshold = float(np.percentile(train_model_score, 99.0))
    neural_prediction = (model_score > neural_threshold).astype(np.int8)
    neural_precision, neural_recall, neural_f1, _ = precision_recall_fscore_support(
        label,
        neural_prediction,
        average="binary",
        zero_division=0,
    )

    return {
        "dataset": dataset,
        "seed": seed,
        "role": "development" if seed == 87 else "confirmation",
        "artifact": str(score_path.relative_to(REPO_ROOT)),
        "artifact_sha256": _sha256(score_path),
        "evaluated_points": int(len(label)),
        "validation": {
            "all_scores_finite": finite,
            "point_lengths_match": lengths_match,
            "entity_aware": bool(entity_ids is not None),
            "entity_count": entity_count,
        },
        "fused_score": {
            "roc_auc": float(roc_auc_score(label, score)),
            "pr_auc": float(average_precision_score(label, score)),
            "POT+PA": pot,
            "Best-F1+PA": {
                key: best[key]
                for key in ("f1", "precision", "recall", "threshold", "TP", "TN", "FP", "FN")
            },
        },
        "neural_model_score": {
            "roc_auc": float(roc_auc_score(label, model_score)),
            "pr_auc": float(average_precision_score(label, model_score)),
            "train_p99_threshold": neural_threshold,
            "train_p99_raw_precision": float(neural_precision),
            "train_p99_raw_recall": float(neural_recall),
            "train_p99_raw_f1": float(neural_f1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the exact JSON report in addition to printing it.",
    )
    args = parser.parse_args()
    runs = {
        dataset: {
            f"seed_{seed}": _evaluate_one(dataset, seed) for seed in SEEDS
        }
        for dataset in DATASETS
    }

    aggregate = {}
    for dataset in DATASETS:
        dataset_runs = [runs[dataset][f"seed_{seed}"] for seed in SEEDS]
        aggregate[dataset] = {}
        for section, metrics in (
            ("fused_score", ("roc_auc", "pr_auc")),
            ("neural_model_score", ("roc_auc", "pr_auc", "train_p99_raw_f1")),
        ):
            aggregate[dataset][section] = {
                metric: _summary([run[section][metric] for run in dataset_runs])
                for metric in metrics
            }
        aggregate[dataset]["fused_score"].update(
            {
                protocol: _summary(
                    [run["fused_score"][protocol]["f1"] for run in dataset_runs]
                )
                for protocol in ("POT+PA", "Best-F1+PA")
            }
        )

    seed_macros = {
        f"seed_{seed}": {
            protocol: float(
                np.mean(
                    [
                        runs[dataset][f"seed_{seed}"]["fused_score"][protocol]["f1"]
                        for dataset in DATASETS
                    ]
                )
            )
            for protocol in ("POT+PA", "Best-F1+PA")
        }
        for seed in SEEDS
    }
    macro = {
        protocol: _summary(
            [seed_macros[f"seed_{seed}"][protocol] for seed in SEEDS]
        )
        for protocol in ("POT+PA", "Best-F1+PA")
    }

    mean_above = {
        protocol: {
            dataset: bool(
                aggregate[dataset]["fused_score"][protocol]["mean"]
                > TABLE_BEST[protocol][dataset]
            )
            for dataset in DATASETS
        }
        for protocol in ("POT+PA", "Best-F1+PA")
    }
    seed_wins = {
        protocol: {
            dataset: int(
                sum(
                    runs[dataset][f"seed_{seed}"]["fused_score"][protocol]["f1"]
                    > TABLE_BEST[protocol][dataset]
                    for seed in SEEDS
                )
            )
            for dataset in DATASETS
        }
        for protocol in ("POT+PA", "Best-F1+PA")
    }
    all_valid = bool(
        all(
            run["validation"]["all_scores_finite"]
            and run["validation"]["point_lengths_match"]
            and (dataset != "SMD" or run["validation"]["entity_count"] == 28)
            for dataset in DATASETS
            for run in runs[dataset].values()
        )
    )
    gates = {
        "mean_POT+PA_above_external_every_dataset": bool(
            all(mean_above["POT+PA"].values())
        ),
        "mean_Best-F1+PA_above_external_every_dataset": bool(
            all(mean_above["Best-F1+PA"].values())
        ),
        "at_least_two_of_three_seeds_above_external_every_cell": bool(
            all(count >= 2 for values in seed_wins.values() for count in values.values())
        ),
        "all_runs_valid_and_SMD_entity_aware": all_valid,
        "mean_POT+PA_macro_at_least_formal_V4": bool(
            macro["POT+PA"]["mean"] >= FORMAL_V4_POT_MACRO
        ),
    }
    gates["passed"] = bool(all(gates.values()))

    output = {
        "status": "completed_frozen_multiseed_confirmation",
        "architecture": "Causal Conv + HGAT-Lite(rank=16) + KAN-TCN + KANAD",
        "score": "train_ecdf((1-0.001)*innovation_score + 0.001*model_score)",
        "seeds": {"development": [87], "confirmation": [90, 98]},
        "protocol": {
            "best_f1_search": "adaptive 200+500, point-adjusted",
            "pot_tail_safety_margin": TAIL_SAFETY_MARGIN,
            "stored_external_baselines": TABLE_BEST,
            "formal_V4_POT+PA_macro": FORMAL_V4_POT_MACRO,
        },
        "runs": runs,
        "three_seed_mean_population_std": aggregate,
        "per_seed_macro": seed_macros,
        "macro_mean_population_std": macro,
        "external_baseline_mean_above": mean_above,
        "external_baseline_seed_win_count": seed_wins,
        "promotion_gate": gates,
        "efficiency_reference": {
            "PSM": {
                "legacy_spatial_parameters": 2905,
                "lite_spatial_parameters": 690,
                "spatial_parameter_reduction": 0.7624784853700517,
                "legacy_total_parameters": 86564,
                "lite_total_parameters": 84349,
            },
            "SWAT": {
                "legacy_spatial_parameters": 138513,
                "lite_spatial_parameters": 1234,
                "spatial_parameter_reduction": 0.9910910956386049,
                "legacy_total_parameters": 303330,
                "lite_total_parameters": 166051,
                "single_pilot_peak_memory_mib": {"legacy": 1650.9, "lite": 27.6},
                "single_pilot_forward_ms_per_batch": {"legacy": 9.756, "lite": 9.381},
                "warning": "Memory and latency are single-pilot figures, not multi-seed estimates.",
            },
        },
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
