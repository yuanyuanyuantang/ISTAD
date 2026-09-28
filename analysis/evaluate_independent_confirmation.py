#!/usr/bin/env python3
"""Run the preregistered pure-V4 confirmation on MSL and SMAP.

This evaluator must remain label-blind until the final metric calls.  Fitting,
residual scaling, automatic pool selection, ECDF calibration, and every fixed
threshold use only the chronological normal fitting split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ISTAD_ROOT = REPO_ROOT / "code" / "ISTAD"
sys.path.insert(0, str(ISTAD_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from data_provider.data_loader import MSLSegLoader, SMAPSegLoader  # noqa: E402
from evaluate_istad_innovation_pa import (  # noqa: E402
    _fixed_pa_metrics,
    _pot_threshold,
)
from utils.innovation import TrainingOnlyInnovationScorer  # noqa: E402
from utils.tools import bf_search_adaptive  # noqa: E402


DATASETS = {
    "MSL": (MSLSegLoader, 55),
    "SMAP": (SMAPSegLoader, 25),
}
WINDOW = 100
POT_LEVEL = 0.99
POT_MULTIPLIER = 1.0
POT_TAIL_MARGIN = 1.04
AUDITED_BASELINES = {
    "MtsCID_WWW_2025": {
        "source": "https://doi.org/10.1145/3696410.3714941",
        "protocol": "reported point-adjusted F1; exact threshold path not locally reproduced",
        "MSL": {"pa_f1": 0.9513},
        "SMAP": {"pa_f1": 0.9732},
    },
    "U2AD_arXiv_2026": {
        "source": "https://arxiv.org/abs/2605.09685",
        "protocol": "reported threshold-independent ROC-AUC/AP and point-adjusted F1",
        "MSL": {"roc_auc": 0.7314, "pr_auc": 0.2186, "pa_f1": 0.9460},
        "SMAP": {"roc_auc": 0.6224, "pr_auc": 0.1912, "pa_f1": 0.9694},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _complete_windows(values: np.ndarray, win_size: int) -> np.ndarray:
    points = ((len(values) - win_size) // win_size + 1) * win_size
    return np.asarray(values[:points])


def _external_comparison(results: dict) -> dict:
    comparisons = {}
    for dataset, result in results.items():
        comparisons[dataset] = {
            "ISTAD": {
                "roc_auc": result["ranking"]["roc_auc"],
                "pr_auc": result["ranking"]["pr_auc"],
                "POT+PA_f1": result["POT+PA"]["f1"],
                "Best-F1+PA_f1": result["Best-F1+PA"]["f1"],
            },
            "delta_to_U2AD_threshold_independent": {
                "roc_auc": result["ranking"]["roc_auc"]
                - AUDITED_BASELINES["U2AD_arXiv_2026"][dataset]["roc_auc"],
                "pr_auc": result["ranking"]["pr_auc"]
                - AUDITED_BASELINES["U2AD_arXiv_2026"][dataset]["pr_auc"],
            },
            "oracle_PA_context_only": {
                "delta_Best-F1+PA_to_MtsCID_reported_PA_f1": result[
                    "Best-F1+PA"
                ]["f1"]
                - AUDITED_BASELINES["MtsCID_WWW_2025"][dataset]["pa_f1"],
                "delta_Best-F1+PA_to_U2AD_reported_PA_f1": result[
                    "Best-F1+PA"
                ]["f1"]
                - AUDITED_BASELINES["U2AD_arXiv_2026"][dataset]["pa_f1"],
            },
        }
    return {
        "status": "protocol_separated_literature_context",
        "audited_baselines": AUDITED_BASELINES,
        "comparison": comparisons,
        "warning": (
            "AUC/AP deltas are threshold-independent but still require identical preprocessing "
            "for a controlled head-to-head result. PA deltas are context only: ISTAD Best-F1+PA "
            "is a test-label oracle and the baseline threshold implementations were not rerun."
        ),
    }


def _raw_metrics(score: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    prediction = (score > threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label,
        prediction,
        average="binary",
        zero_division=0,
    )
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
    }


def evaluate_dataset(name: str, data_root: Path, output_dir: Path) -> dict:
    loader_class, expected_features = DATASETS[name]
    loader_args = SimpleNamespace(istad_holdout_val=1, istad_entity_aware=0)
    root = data_root / name
    train_dataset = loader_class(
        loader_args,
        str(root),
        WINDOW,
        step=WINDOW,
        flag="train",
    )
    test_dataset = loader_class(
        loader_args,
        str(root),
        WINDOW,
        step=WINDOW,
        flag="test",
    )
    train_points = _complete_windows(train_dataset.train, WINDOW)
    test_points = _complete_windows(test_dataset.test, WINDOW)
    labels = _complete_windows(test_dataset.test_labels, WINDOW).reshape(-1)
    labels = (labels > 0).astype(np.int8)

    if train_points.ndim != 2 or test_points.ndim != 2:
        raise ValueError(f"{name}: expected 2-D multivariate arrays")
    if train_points.shape[1] != expected_features:
        raise ValueError(
            f"{name}: expected {expected_features} features, got {train_points.shape[1]}"
        )
    if len(test_points) != len(labels):
        raise ValueError(f"{name}: test score/label alignment failed")

    scorer = TrainingOnlyInnovationScorer(
        lag=1,
        ridge=0.01,
        pool="auto",
        degenerate_cutoff=0.10,
        scale_floor_ratio=0.10,
    ).fit(train_points)
    train_score = scorer.score(train_points).astype(np.float64)
    test_score = scorer.score(test_points).astype(np.float64)
    if not np.isfinite(train_score).all() or not np.isfinite(test_score).all():
        raise ValueError(f"{name}: non-finite innovation score")

    train_p99 = float(np.percentile(train_score, 99.0))
    p99_raw = _raw_metrics(test_score, labels, train_p99)
    p99_pa = _fixed_pa_metrics(test_score, labels, train_p99)

    train_pot = -np.log1p(-np.clip(train_score, 0.0, 1.0 - 1e-12))
    test_pot = -np.log1p(-np.clip(test_score, 0.0, 1.0 - 1e-12))
    pot_threshold, raw_pot_threshold, effective_level = _pot_threshold(
        train_pot,
        test_pot,
        POT_LEVEL,
        POT_MULTIPLIER,
        tail_safety_margin=POT_TAIL_MARGIN,
    )
    pot_pa = _fixed_pa_metrics(test_pot, labels, pot_threshold)
    pot_pa.update(
        {
            "q": 1e-5,
            "configured_level": POT_LEVEL,
            "effective_level": effective_level,
            "base_multiplier": POT_MULTIPLIER,
            "tail_safety_margin": POT_TAIL_MARGIN,
            "raw_spot_threshold": raw_pot_threshold,
            "score_transform": "-log(1-train_ecdf_score)",
        }
    )

    best_pa = bf_search_adaptive(
        score=test_score,
        label=labels,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=True,
    )
    best_raw = bf_search_adaptive(
        score=test_score,
        label=labels,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=False,
    )

    source_files = {}
    for kind in ("train", "test", "test_label"):
        filename = f"{name}_{kind}.npy"
        candidates = (root / filename, root / name / filename)
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{name}: expected one downloaded source for {filename}, got {matches}"
            )
        source_files[kind] = matches[0]
    integrity = {
        "all_scores_finite": True,
        "score_label_lengths_match": bool(len(test_score) == len(labels)),
        "expected_feature_count": expected_features,
        "observed_feature_count": int(test_points.shape[1]),
        "source_sha256": {
            key: _sha256(path) for key, path in source_files.items()
        },
    }
    prevalence = float(labels.mean())
    ranking = {
        "roc_auc": float(roc_auc_score(labels, test_score)),
        "pr_auc": float(average_precision_score(labels, test_score)),
        "anomaly_prevalence": prevalence,
        "roc_above_random": bool(roc_auc_score(labels, test_score) > 0.5),
        "pr_above_prevalence": bool(
            average_precision_score(labels, test_score) > prevalence
        ),
    }

    score_path = output_dir / f"{name}_pure_v4_scores.npz"
    np.savez_compressed(
        score_path,
        score=test_score.astype(np.float32),
        train_score=train_score.astype(np.float32),
        label=labels,
        metadata_json=np.asarray(json.dumps(scorer.metadata_dict(), sort_keys=True)),
    )
    return {
        "dataset": name,
        "fit_points": int(len(train_points)),
        "held_out_normal_validation_points": int(len(train_dataset.val)),
        "evaluated_points": int(len(labels)),
        "innovation": scorer.metadata_dict(),
        "integrity": integrity,
        "ranking": ranking,
        "train-p99_raw": p99_raw,
        "train-p99+PA": p99_pa,
        "POT+PA": pot_pa,
        "Best-raw": best_raw,
        "Best-F1+PA": best_pa,
        "score_artifact": _report_path(score_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT.parent / "ISTAD" / "dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "independent_confirmation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        name: evaluate_dataset(name, args.data_root, args.output_dir)
        for name in DATASETS
    }
    integrity_passed = bool(
        all(
            result["integrity"]["all_scores_finite"]
            and result["integrity"]["score_label_lengths_match"]
            and result["integrity"]["expected_feature_count"]
            == result["integrity"]["observed_feature_count"]
            for result in results.values()
        )
    )
    ranking_passed = bool(
        all(
            result["ranking"]["roc_above_random"]
            and result["ranking"]["pr_above_prevalence"]
            for result in results.values()
        )
    )
    report = {
        "status": "completed_preregistered_independent_benchmark_confirmation",
        "preregistration": "analysis/independent_confirmation/preregistered_msl_smap.md",
        "detector": "pure ISTAD V4 training-only causal innovation",
        "configuration": {
            "lag": 1,
            "ridge": 0.01,
            "pool": "auto",
            "degenerate_cutoff": 0.10,
            "scale_floor_ratio": 0.10,
            "window_and_step": 100,
            "POT": {
                "q": 1e-5,
                "level": POT_LEVEL,
                "multiplier": POT_MULTIPLIER,
                "tail_safety_margin": POT_TAIL_MARGIN,
            },
        },
        "results": results,
        "gates": {
            "integrity": integrity_passed,
            "ranking_sanity_both_datasets": ranking_passed,
            "passed": bool(integrity_passed and ranking_passed),
        },
        "external_comparison": _external_comparison(results),
    }
    output_path = args.output_dir / "msl_smap_results.json"
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
