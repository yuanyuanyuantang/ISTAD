#!/usr/bin/env python3
"""Aggregate frozen V6b forecast artifacts for seeds 87/90/98."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_v6_nasa_target import REPO_ROOT, TARGETS, V5_REFERENCE, _evaluate


DATASETS = ("MSL", "SMAP")
SEEDS = (87, 90, 98)
EXPECTED_POINTS = {"MSL": 72600, "SMAP": 425400}


def _find(score_root: Path, dataset: str, seed: int) -> Path:
    directories = list(score_root.glob(
        f"anomaly_detection_{dataset}_ISTAD_{dataset}_ftM_sl100_bmtcn_v3_ent_"
        f"tfcast_bs128_v6bnasa_forecast_s{seed}_0"
    ))
    if len(directories) != 1:
        raise FileNotFoundError(
            f"expected one {dataset}/forecast/seed-{seed} directory, got {directories}"
        )
    path = directories[0] / "point_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _summary(seed_results: dict) -> dict:
    fields = {
        "roc_auc": lambda m: m["roc_auc"],
        "pr_auc": lambda m: m["pr_auc"],
        "train_p99_raw_f1": lambda m: m["train_p99"]["raw_f1"],
        "train_p99_pa_f1": lambda m: m["train_p99"]["point_adjusted_f1"],
        "best_raw_f1": lambda m: m["best_raw"]["f1"],
        "best_pa_f1": lambda m: m["best_point_adjusted"]["f1"],
    }
    summary = {}
    for name, extract in fields.items():
        values = np.asarray([
            extract(seed_results[str(seed)]["target_error"]) for seed in SEEDS
        ], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": [float(value) for value in values],
        }
    return summary


def _decisions(results: dict, summaries: dict) -> dict:
    completeness = all(
        results[dataset][str(seed)]["evaluated_points"] == EXPECTED_POINTS[dataset]
        and results[dataset][str(seed)]["entities"] == {"MSL": 27, "SMAP": 53}[dataset]
        for dataset in DATASETS for seed in SEEDS
    )
    confirmation_seed_floor = all(
        results[dataset][str(seed)]["target_error"]["roc_auc"] > 0.5
        and results[dataset][str(seed)]["target_error"]["pr_auc"]
            > results[dataset][str(seed)]["target_error"]["anomaly_prevalence"]
        for dataset in DATASETS for seed in (90, 98)
    )
    mean_beats_v5 = all(
        summaries[dataset][metric]["mean"] > V5_REFERENCE[dataset][metric]
        for dataset in DATASETS for metric in ("roc_auc", "pr_auc")
    )
    variance_guard = all(
        summaries[dataset][metric]["std"] <= 0.03
        for dataset in DATASETS for metric in ("roc_auc", "pr_auc")
    )
    no_ap_regression = all(
        results[dataset][str(seed)]["target_error"]["pr_auc"]
            >= V5_REFERENCE[dataset]["pr_auc"]
        for dataset in DATASETS for seed in SEEDS
    )
    literature = {
        dataset: {
            "roc_target_met": summaries[dataset]["roc_auc"]["mean"]
                >= TARGETS[dataset]["roc_auc"],
            "pr_target_met": summaries[dataset]["pr_auc"]["mean"]
                >= TARGETS[dataset]["pr_auc"],
            "best_PA_target_met": summaries[dataset]["best_pa_f1"]["mean"]
                >= TARGETS[dataset]["pa_f1"],
        }
        for dataset in DATASETS
    }
    gates = {
        "artifact_completeness": bool(completeness),
        "confirmation_seed_floor": bool(confirmation_seed_floor),
        "three_seed_mean_beats_v5": bool(mean_beats_v5),
        "variance_guard": bool(variance_guard),
        "no_individual_ap_regression_vs_v5": bool(no_ap_regression),
    }
    return {
        "stability_gates": gates,
        "retain_as_stable_development_candidate": bool(all(gates.values())),
        "literature_context_gate": literature,
        "sota_context_met": bool(all(all(values.values()) for values in literature.values())),
        "warning": (
            "This is multi-seed stability on revealed development labels, not "
            "independent confirmation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "v6b_nasa_causal" / "multiseed_s87_s90_s98.json",
    )
    args = parser.parse_args()
    results = {
        dataset: {
            str(seed): _evaluate(_find(args.score_root, dataset, seed))
            for seed in SEEDS
        }
        for dataset in DATASETS
    }
    summaries = {dataset: _summary(results[dataset]) for dataset in DATASETS}
    report = {
        "status": "completed_v6b_nasa_multiseed_stability",
        "confirmatory": False,
        "development_seed": 87,
        "fixed_stability_seeds": [90, 98],
        "plan": "analysis/v6b_nasa_causal/confirmation_plan_s90_s98.md",
        "results": results,
        "summary": summaries,
        "decisions": _decisions(results, summaries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
