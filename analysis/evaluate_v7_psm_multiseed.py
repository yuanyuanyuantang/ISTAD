#!/usr/bin/env python3
"""Evaluate the frozen V7 PSM seed-87/90/98 confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_v6_nasa_target import _metrics  # noqa: E402
from evaluate_v7_psm_seed87 import EXPECTED_POINTS, _evaluate, _find  # noqa: E402


SEEDS = (87, 90, 98)
ARMS = ("full", "no_prior", "temporal_only")
PRIMARY = ("roc_auc", "pr_auc")
HGAT_LITE_REFERENCE = {"roc_auc": 0.7466, "pr_auc": 0.5547}


def _prediction_only(path: Path) -> dict:
    """Reconstruct the exact weight-zero scoring ablation from a full artifact."""
    with np.load(path) as artifact:
        calibrated = np.asarray(artifact["v7_calibrated_components"])
        train_calibrated = np.asarray(artifact["train_v7_calibrated_components"])
        label = np.asarray(artifact["label"])
        score = calibrated[:, 0]
        train_score = train_calibrated[:, 0]
    return _metrics(score, train_score, label, np.zeros(len(label), dtype=np.int8))


def _summary(per_seed: dict) -> dict:
    summary = {}
    for arm in (*ARMS, "no_relation_score"):
        summary[arm] = {}
        for metric in PRIMARY:
            values = np.asarray([
                per_seed[str(seed)][arm]["metrics"][metric] for seed in SEEDS
            ])
            summary[arm][metric] = {
                "values": [float(value) for value in values],
                "mean": float(values.mean()),
                "std_population": float(values.std()),
            }
    return summary


def _decisions(per_seed: dict, summary: dict) -> dict:
    complete = all(
        per_seed[str(seed)][arm]["evaluated_points"] == EXPECTED_POINTS
        and per_seed[str(seed)][arm]["finite"]
        and per_seed[str(seed)][arm]["aligned"]
        and per_seed[str(seed)][arm]["component_shape"] == [EXPECTED_POINTS, 2]
        for seed in SEEDS for arm in ARMS
    )
    full_floor = all(
        per_seed[str(seed)]["full"]["metrics"]["roc_auc"] > 0.5
        and per_seed[str(seed)]["full"]["metrics"]["pr_auc"]
        > per_seed[str(seed)]["full"]["metrics"]["anomaly_prevalence"]
        for seed in SEEDS
    )

    deltas = {}
    paired_counts = {}
    for control in ("temporal_only", "no_prior", "no_relation_score"):
        deltas[control] = {}
        paired_counts[control] = {}
        for metric in PRIMARY:
            values = [
                per_seed[str(seed)]["full"]["metrics"][metric]
                - per_seed[str(seed)][control]["metrics"][metric]
                for seed in SEEDS
            ]
            deltas[control][metric] = {
                "values": [float(value) for value in values],
                "mean": float(np.mean(values)),
            }
            paired_counts[control][metric] = int(sum(value > 0 for value in values))

    mean_superiority = all(
        deltas[control][metric]["mean"] > 0
        for control in ("temporal_only", "no_prior") for metric in PRIMARY
    )
    paired_consistency = all(
        paired_counts[control][metric] >= 2
        for control in ("temporal_only", "no_prior") for metric in PRIMARY
    )
    effect_size = (
        max(deltas["temporal_only"][metric]["mean"] for metric in PRIMARY) >= 0.02
        and max(deltas["no_prior"][metric]["mean"] for metric in PRIMARY) >= 0.01
    )
    stability = all(
        summary["full"][metric]["std_population"] <= 0.05 for metric in PRIMARY
    )
    relation_guard = all(
        deltas["no_relation_score"][metric]["mean"] >= -0.01
        for metric in PRIMARY
    )
    gates = {
        "artifact_completeness": bool(complete),
        "full_above_random_prevalence_floor_every_seed": bool(full_floor),
        "mean_superiority_over_training_ablations": bool(mean_superiority),
        "paired_consistency_two_of_three": bool(paired_consistency),
        "minimum_mean_effect_size": bool(effect_size),
        "full_stability_std_at_most_0_05": bool(stability),
        "relation_score_mean_guard": bool(relation_guard),
    }
    paper_context = {
        metric: {
            "v7_mean": summary["full"][metric]["mean"],
            "hgat_lite_reference": reference,
            "met": bool(summary["full"][metric]["mean"] >= reference),
        }
        for metric, reference in HGAT_LITE_REFERENCE.items()
    }
    return {
        "paired_full_minus_control": deltas,
        "paired_positive_seed_counts": paired_counts,
        "gates": gates,
        "confirmation_pass": bool(all(gates.values())),
        "paper_readiness_context": paper_context,
        "warning": "PSM labels were previously revealed; this is development evidence only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "v7_causal_hypergraph" /
        "psm_multiseed_confirmation.json",
    )
    args = parser.parse_args()
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            arm: _evaluate(_find(args.score_root, arm, seed)) for arm in ARMS
        }
        full_path = _find(args.score_root, "full", seed)
        per_seed[str(seed)]["no_relation_score"] = {
            **{key: value for key, value in per_seed[str(seed)]["full"].items()
               if key != "metrics"},
            "derived_from_full_components": True,
            "metrics": _prediction_only(full_path),
        }

    summary = _summary(per_seed)
    report = {
        "status": "completed_v7_psm_multiseed_development_confirmation",
        "confirmatory": False,
        "seeds": list(SEEDS),
        "preregistration": (
            "analysis/v7_causal_hypergraph/confirmation_plan_psm_s90_s98.md"
        ),
        "results": per_seed,
        "summary": summary,
        "decisions": _decisions(per_seed, summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
