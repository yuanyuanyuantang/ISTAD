#!/usr/bin/env python3
"""Evaluate the frozen six-dataset V7.1 seed-87 development screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_v7_psm_seed87 import _evaluate  # noqa: E402


DATASETS = ("EXATHLON", "PSM", "SMD", "SWAT", "MSL", "SMAP")
MULTI_TARGET = {"EXATHLON", "PSM", "SMD", "SWAT"}
ARMS = ("full", "no_prior", "temporal_only")
EXPECTED_POINTS = {
    "EXATHLON": 52800,
    "PSM": 87808,
    "SMD": 706560,
    "SWAT": 449856,
    "MSL": 72600,
    "SMAP": 425400,
}


def _find(score_root: Path, dataset: str, arm: str, seed: int) -> Path:
    directories = list(score_root.glob(
        f"anomaly_detection_{dataset}_ISTAD_{dataset}_ftM_*"
        f"_archv71*_tfcast_bs128_v71_{arm}_s{seed}_0"
    ))
    if len(directories) != 1:
        raise FileNotFoundError(
            f"expected one V7.1 {dataset}/{arm}/seed-{seed}: {directories}"
        )
    artifact = directories[0] / "point_scores_v7_fused.npz"
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    return artifact


def _decision(dataset: str, results: dict) -> dict:
    expected = EXPECTED_POINTS[dataset]
    metric = lambda arm, name: results[arm]["metrics"][name]
    complete = all(
        results[arm]["evaluated_points"] == expected
        and results[arm]["finite"]
        and results[arm]["aligned"]
        and results[arm]["component_shape"] == [expected, 2]
        and results[arm]["metadata"]["prior_ready"]
        and results[arm]["metadata"]["relation_score_weight"] == 0.0
        for arm in ARMS
    )
    floor = (
        metric("full", "roc_auc") > 0.5
        and metric("full", "pr_auc")
        > results["full"]["metrics"]["anomaly_prevalence"]
    )
    deltas = {
        control: {
            name: float(metric("full", name) - metric(control, name))
            for name in ("roc_auc", "pr_auc")
        }
        for control in ("temporal_only", "no_prior")
    }
    mechanism = {
        control: (
            min(delta.values()) >= -0.01
            and max(delta.values()) > 0.0
        )
        for control, delta in deltas.items()
    }
    gates = results["full"]["metadata"].get("learned_gates")
    learned_gates_valid = bool(
        gates
        and all(
            values
            and all(0.0 < float(value) < 1.0 for value in values)
            for values in gates.values()
        )
    )
    checks = {
        "artifact_completeness": bool(complete),
        "full_above_random_prevalence_floor": bool(floor),
        "hypergraph_mechanism_gate": bool(mechanism["temporal_only"]),
        "causal_prior_mechanism_gate": bool(mechanism["no_prior"]),
        "learned_gates_valid": learned_gates_valid,
    }
    return {
        "deltas": deltas,
        "gates": checks,
        "continue_to_seeds_90_98": bool(all(checks.values())),
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
        "v71_seed87.json",
    )
    args = parser.parse_args()

    results = {
        dataset: {
            arm: _evaluate(_find(args.score_root, dataset, arm, args.seed))
            for arm in ARMS
        }
        for dataset in DATASETS
    }
    decisions = {
        dataset: _decision(dataset, results[dataset])
        for dataset in DATASETS
    }
    passing = [
        dataset for dataset in DATASETS
        if decisions[dataset]["continue_to_seeds_90_98"]
    ]
    passing_multi = [dataset for dataset in passing if dataset in MULTI_TARGET]
    payload = {
        "status": "completed_v71_seed87_development_screen",
        "confirmatory": False,
        "seed": args.seed,
        "preregistration": (
            "analysis/v7_causal_hypergraph/"
            "preregistered_v71_residual_seed87.md"
        ),
        "results": results,
        "decisions": decisions,
        "overall": {
            "passing_datasets": passing,
            "passing_multi_target_datasets": passing_multi,
            "continue_to_seeds_90_98": bool(
                len(passing) >= 4 and len(passing_multi) >= 3
            ),
            "warning": (
                "All dataset labels were exposed before V7.1; results are "
                "development evidence only."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

