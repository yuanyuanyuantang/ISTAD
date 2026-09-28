#!/usr/bin/env python3
"""Evaluate the frozen V7 five-dataset seed-87 development screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_v6_nasa_target import _metrics  # noqa: E402
from evaluate_v7_psm_seed87 import _evaluate  # noqa: E402


DATASETS = ("EXATHLON", "SMD", "SWAT", "MSL", "SMAP")
MULTI_TARGET = {"EXATHLON", "SMD", "SWAT"}
ARMS = ("full", "no_prior", "temporal_only")
EXPECTED_POINTS = {
    "EXATHLON": 52800,
    "SMD": 706560,
    "SWAT": 449856,
    "MSL": 72600,
    "SMAP": 425400,
}
LOCAL_CONTEXT = {
    "EXATHLON": {"name": "HGAT-Lite", "roc_auc": 0.8721, "pr_auc": 0.6189},
    "SMD": {"name": "HGAT-Lite", "roc_auc": 0.7879, "pr_auc": 0.2134},
    "SWAT": {"name": "HGAT-Lite", "roc_auc": 0.8424, "pr_auc": 0.7063},
    "MSL": {"name": "V6b", "roc_auc": 0.6508, "pr_auc": 0.2186},
    "SMAP": {"name": "V6b", "roc_auc": 0.5191, "pr_auc": 0.1324},
}


def _find(score_root: Path, dataset: str, arm: str, seed: int) -> Path:
    directories = list(score_root.glob(
        f"anomaly_detection_{dataset}_ISTAD_{dataset}_ftM_*"
        f"_archv7*_tfcast_bs128_v7_{arm}_s{seed}_0"
    ))
    if len(directories) != 1:
        raise FileNotFoundError(
            f"expected one V7 {dataset}/{arm}/seed-{seed}: {directories}"
        )
    artifact = directories[0] / "point_scores_v7_fused.npz"
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    return artifact


def _prediction_only(path: Path) -> dict:
    with np.load(path) as artifact:
        calibrated = np.asarray(artifact["v7_calibrated_components"])
        train_calibrated = np.asarray(
            artifact["train_v7_calibrated_components"]
        )
        label = np.asarray(artifact["label"])
        entity = (
            np.asarray(artifact["entity_id"])
            if "entity_id" in artifact.files
            else np.zeros(len(label), dtype=np.int8)
        )
    return _metrics(
        calibrated[:, 0], train_calibrated[:, 0], label, entity
    )


def _dataset_decision(dataset: str, results: dict) -> dict:
    expected = EXPECTED_POINTS[dataset]
    metric = lambda arm, name: results[arm]["metrics"][name]
    complete = all(
        results[arm]["evaluated_points"] == expected
        and results[arm]["finite"]
        and results[arm]["aligned"]
        and results[arm]["component_shape"] == [expected, 2]
        and results[arm]["metadata"]["prior_ready"]
        for arm in ARMS
    )
    floor = (
        metric("full", "roc_auc") > 0.5
        and metric("full", "pr_auc")
        > results["full"]["metrics"]["anomaly_prevalence"]
    )
    deltas = {}
    for control in ("temporal_only", "no_prior", "no_relation_score"):
        deltas[control] = {
            name: float(metric("full", name) - metric(control, name))
            for name in ("roc_auc", "pr_auc")
        }
    temporal = (
        min(deltas["temporal_only"].values()) >= -0.01
        and max(deltas["temporal_only"].values()) >= 0.01
    )
    prior = (
        min(deltas["no_prior"].values()) >= -0.01
        and max(deltas["no_prior"].values()) >= 0.005
    )
    relation = min(deltas["no_relation_score"].values()) >= -0.01
    gates = {
        "artifact_completeness": bool(complete),
        "full_above_random_prevalence_floor": bool(floor),
        "hypergraph_mechanism_gate": bool(temporal),
        "causal_prior_mechanism_gate": bool(prior),
        "relation_score_guard": bool(relation),
    }
    context = LOCAL_CONTEXT[dataset]
    return {
        "deltas": deltas,
        "gates": gates,
        "continue_to_seeds_90_98": bool(all(gates.values())),
        "local_paper_readiness_context": {
            **context,
            "roc_met": bool(metric("full", "roc_auc") >= context["roc_auc"]),
            "pr_met": bool(metric("full", "pr_auc") >= context["pr_auc"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "v7_causal_hypergraph" /
        "cross_dataset_seed87.json",
    )
    args = parser.parse_args()

    results = {}
    decisions = {}
    for dataset in DATASETS:
        dataset_results = {
            arm: _evaluate(_find(args.score_root, dataset, arm, args.seed))
            for arm in ARMS
        }
        full_path = _find(args.score_root, dataset, "full", args.seed)
        dataset_results["no_relation_score"] = {
            **{key: value for key, value in dataset_results["full"].items()
               if key != "metrics"},
            "derived_from_full_components": True,
            "metrics": _prediction_only(full_path),
        }
        results[dataset] = dataset_results
        decisions[dataset] = _dataset_decision(dataset, dataset_results)

    passing = [
        dataset for dataset in DATASETS
        if decisions[dataset]["continue_to_seeds_90_98"]
    ]
    passing_multi = [dataset for dataset in passing if dataset in MULTI_TARGET]
    report = {
        "status": "completed_v7_cross_dataset_seed87_development_screen",
        "confirmatory": False,
        "seed": args.seed,
        "preregistration": (
            "analysis/v7_causal_hypergraph/"
            "preregistered_cross_dataset_seed87.md"
        ),
        "results": results,
        "decisions": decisions,
        "overall": {
            "passing_datasets": passing,
            "passing_multi_target_datasets": passing_multi,
            "cross_dataset_mechanism_supported": bool(
                len(passing) >= 3 and len(passing_multi) >= 2
            ),
            "warning": (
                "All local test labels were exposed before this screen; "
                "results are development evidence only."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
