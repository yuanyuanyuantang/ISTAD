#!/usr/bin/env python3
"""Evaluate the frozen seed-87 MSL/SMAP neural representation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))

from utils.tools import adjustment, bf_search_adaptive  # noqa: E402


DATASETS = ("MSL", "SMAP")
ARMS = ("none", "lite")
TARGETS = {
    "MSL": {"roc_auc": 0.7314, "pr_auc": 0.2186, "pa_f1": 0.9513},
    "SMAP": {"roc_auc": 0.6224, "pr_auc": 0.1912, "pa_f1": 0.9732},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_metrics(
    score: np.ndarray,
    train_score: np.ndarray,
    label: np.ndarray,
    entity_ids: np.ndarray,
) -> dict:
    score = np.asarray(score).reshape(-1)
    train_score = np.asarray(train_score).reshape(-1)
    label = np.asarray(label).reshape(-1).astype(np.int8)
    entity_ids = np.asarray(entity_ids).reshape(-1)
    if len(score) != len(label) or len(entity_ids) != len(label):
        raise ValueError("score, label, and entity ID lengths do not match")
    if not np.isfinite(score).all() or not np.isfinite(train_score).all():
        raise ValueError("score contains NaN or infinite values")

    threshold = float(np.percentile(train_score, 99.0))
    prediction = (score > threshold).astype(np.int8)
    precision, recall, raw_f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0
    )
    _, adjusted = adjustment(
        label.copy(), prediction.copy(), entity_ids=entity_ids
    )
    pa_precision, pa_recall, pa_f1, _ = precision_recall_fscore_support(
        label, adjusted, average="binary", zero_division=0
    )
    best_raw = bf_search_adaptive(
        score,
        label,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=False,
    )
    best_pa = bf_search_adaptive(
        score,
        label,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=True,
        entity_ids=entity_ids,
    )
    return {
        "roc_auc": float(roc_auc_score(label, score)),
        "pr_auc": float(average_precision_score(label, score)),
        "anomaly_prevalence": float(label.mean()),
        "train_p99": {
            "threshold": threshold,
            "raw_f1": float(raw_f1),
            "raw_precision": float(precision),
            "raw_recall": float(recall),
            "point_adjusted_f1": float(pa_f1),
            "point_adjusted_precision": float(pa_precision),
            "point_adjusted_recall": float(pa_recall),
        },
        "best_raw": best_raw,
        "best_point_adjusted": best_pa,
    }


def evaluate_artifact(path: Path) -> dict:
    with np.load(path) as artifact:
        required = {
            "score", "train_score", "model_score", "train_model_score",
            "innovation_score", "train_innovation_score", "label", "entity_id",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        result = {
            "artifact": str(path.relative_to(REPO_ROOT)),
            "artifact_sha256": _sha256(path),
            "evaluated_points": int(len(artifact["label"])),
            "entities": int(len(np.unique(artifact["entity_id"]))),
            "neural": _score_metrics(
                artifact["model_score"],
                artifact["train_model_score"],
                artifact["label"],
                artifact["entity_id"],
            ),
            "innovation": _score_metrics(
                artifact["innovation_score"],
                artifact["train_innovation_score"],
                artifact["label"],
                artifact["entity_id"],
            ),
            "fused_train_ecdf": _score_metrics(
                artifact["score"],
                artifact["train_score"],
                artifact["label"],
                artifact["entity_id"],
            ),
        }
    return result


def _find_artifact(score_root: Path, dataset: str, arm: str, seed: int) -> Path:
    directories = list(score_root.glob(
        f"anomaly_detection_{dataset}_ISTAD_{dataset}_ftM_sl100_*"
        f"_bs128_v5nasa_{arm}_s{seed}_0"
    ))
    if len(directories) != 1:
        raise FileNotFoundError(
            f"expected one {dataset}/{arm}/seed-{seed} result directory, got {directories}"
        )
    artifacts = list(directories[0].glob(
        "point_scores_innovation_fused_w0p001_calibrated_prob_train_ecdf.npz"
    ))
    if len(artifacts) != 1:
        raise FileNotFoundError(f"expected one score artifact under {directories[0]}")
    return artifacts[0]


def _decisions(results: dict) -> dict:
    continuation = {}
    for arm in ARMS:
        arm_passes = []
        for dataset in DATASETS:
            neural = results[dataset][arm]["neural"]
            innovation = results[dataset][arm]["innovation"]
            arm_passes.append(
                neural["roc_auc"] > innovation["roc_auc"]
                and neural["pr_auc"] > innovation["pr_auc"]
                and neural["pr_auc"] > neural["anomaly_prevalence"]
            )
        continuation[arm] = bool(all(arm_passes))

    lite_deltas = {}
    ranking_deltas = []
    for dataset in DATASETS:
        lite_deltas[dataset] = {}
        for metric in ("roc_auc", "pr_auc"):
            delta = (
                results[dataset]["lite"]["neural"][metric]
                - results[dataset]["none"]["neural"][metric]
            )
            lite_deltas[dataset][metric] = float(delta)
            ranking_deltas.append(delta)
    lite_benefit = bool(
        max(ranking_deltas) >= 0.01 and min(ranking_deltas) >= -0.01
    )

    literature_context = {}
    for dataset in DATASETS:
        literature_context[dataset] = {}
        for arm in ARMS:
            neural = results[dataset][arm]["neural"]
            literature_context[dataset][arm] = {
                "roc_target_met": bool(neural["roc_auc"] >= TARGETS[dataset]["roc_auc"]),
                "pr_target_met": bool(neural["pr_auc"] >= TARGETS[dataset]["pr_auc"]),
                "best_PA_target_met": bool(
                    neural["best_point_adjusted"]["f1"] >= TARGETS[dataset]["pa_f1"]
                ),
            }
    return {
        "continue_arm_to_seeds_90_98": continuation,
        "hgat_lite_deltas_vs_none": lite_deltas,
        "hgat_lite_benefit_gate": lite_benefit,
        "literature_context_targets": TARGETS,
        "literature_context_gate": literature_context,
        "warning": (
            "MSL/SMAP are development datasets after label reveal. Literature targets "
            "are protocol context, not controlled head-to-head comparisons."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root",
        type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "analysis" / "v5_nasa_diagnostic" / "seed87.json",
    )
    args = parser.parse_args()
    results = {
        dataset: {
            arm: evaluate_artifact(
                _find_artifact(args.score_root, dataset, arm, args.seed)
            )
            for arm in ARMS
        }
        for dataset in DATASETS
    }
    report = {
        "status": "completed_v5_nasa_seed87_development_diagnostic",
        "confirmatory": False,
        "preregistration": "analysis/v5_nasa_diagnostic/preregistered_seed87.md",
        "seed": args.seed,
        "results": results,
        "decisions": _decisions(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
