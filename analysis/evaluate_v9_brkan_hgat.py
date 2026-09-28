#!/usr/bin/env python3
"""Evaluate the preregistered V9 bounded-residual KAN-HGAT screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))

from evaluate_v8_kan_hgat import (  # noqa: E402
    DATASETS,
    _checkpoint,
    _evaluate_dataset,
)
from models.istad_layers import BoundedResidualKANProjection  # noqa: E402


def _spline_norm(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    matches = [
        value for key, value in state.items()
        if key.endswith("value_projection.spline_weight")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one spline tensor in {checkpoint}")
    return float(torch.linalg.vector_norm(matches[0]).item())


def _initial_nesting_check():
    projection = BoundedResidualKANProjection(out_features=8)
    values = torch.linspace(-4.0, 4.0, 41).reshape(1, 41, 1)
    with torch.no_grad():
        return bool(torch.equal(projection(values), projection.linear(values)))


def _markdown(report):
    lines = [
        "# V9 bounded-residual KAN-HGAT seed-87 screen",
        "",
        "> Second development experiment on previously inspected test sets; not independent confirmation.",
        "",
        "| Dataset | Linear HGAT AP | BR-KAN HGAT AP | ΔAP | ΔROC | Spline L2 | Added params | Tie groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        run = report["runs"][dataset]
        ranking = run["ranking"]
        lines.append(
            f"| {dataset} | {ranking['linear_hgat_pr_auc']:.6f} | "
            f"{ranking['kan_hgat_pr_auc']:.6f} | {ranking['delta_hgat_pr_auc']:+.6f} | "
            f"{ranking['delta_hgat_roc_auc']:+.6f} | {run['spline_l2_norm']:.6f} | "
            f"{run['parameters']['added']:+d} | {run['rank_safety']['changed_tie_groups']} |"
        )
    diagnostic = report["diagnostic"]
    lines.extend([
        f"| **Macro** | — | — | **{diagnostic['mean_hgat_pr_auc_delta']:+.6f}** | "
        f"**{diagnostic['mean_hgat_roc_auc_delta']:+.6f}** | — | — | — |",
        "",
        "| Dataset | POT+PA linear / BR-KAN | Best-F1+PA linear / BR-KAN |",
        "|---|---:|---:|",
    ])
    for dataset in DATASETS:
        pa = report["runs"][dataset]["point_adjusted"]
        lines.append(
            f"| {dataset} | {pa['POT+PA']['linear']:.6f} / {pa['POT+PA']['kan']:.6f} | "
            f"{pa['Best-F1+PA']['linear']:.6f} / {pa['Best-F1+PA']['kan']:.6f} |"
        )
    lines.extend(["", "## Frozen decision", ""])
    for key, value in report["decision"].items():
        lines.append(f"- {key}: **{value}**")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument(
        "--linear-checkpoint-root", type=Path,
        default=REPO_ROOT / "checkpoints" / "ISTAD_v4_hgat_integrated",
    )
    parser.add_argument(
        "--candidate-checkpoint-root", type=Path,
        default=REPO_ROOT / "checkpoints" / "ISTAD_v9_brkan_hgat",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "v9_brkan_hgat",
    )
    args = parser.parse_args()
    shared = SimpleNamespace(
        score_root=args.score_root,
        linear_checkpoint_root=args.linear_checkpoint_root,
        kan_checkpoint_root=args.candidate_checkpoint_root,
    )

    runs = {}
    for dataset in DATASETS:
        run = _evaluate_dataset(shared, dataset, "v9brk")
        run["spline_l2_norm"] = _spline_norm(_checkpoint(
            args.candidate_checkpoint_root, dataset, "v9brk"
        ))
        runs[dataset] = run

    ap_deltas = np.asarray([
        run["ranking"]["delta_hgat_pr_auc"] for run in runs.values()
    ])
    roc_deltas = np.asarray([
        run["ranking"]["delta_hgat_roc_auc"] for run in runs.values()
    ])
    diagnostic = {
        "mean_hgat_pr_auc_delta": float(ap_deltas.mean()),
        "mean_hgat_roc_auc_delta": float(roc_deltas.mean()),
        "positive_hgat_pr_auc_datasets": int(np.sum(ap_deltas > 0.0)),
        "worst_hgat_pr_auc_delta": float(ap_deltas.min()),
    }
    decision = {
        "initial_linear_nesting_pass": _initial_nesting_check(),
        "finite_artifacts_pass": all(run["finite"] for run in runs.values()),
        "parameter_budget_pass": all(
            run["parameters"]["added"] == 64 for run in runs.values()
        ),
        "learned_spline_pass": all(
            run["spline_l2_norm"] > 1e-6 for run in runs.values()
        ),
        "rank_safety_pass": all(
            run["rank_safety"]["distinct_rank_inversions"] == 0
            for run in runs.values()
        ),
        "non_vacuous_pass": sum(
            run["rank_safety"]["changed_tie_groups"] > 0
            for run in runs.values()
        ) >= 3,
        "pa_compatibility_pass": all(
            abs(protocol["delta"]) <= 1e-4
            for run in runs.values()
            for protocol in run["point_adjusted"].values()
        ),
        "hgat_ap_diagnostic_pass": bool(
            diagnostic["mean_hgat_pr_auc_delta"] >= 0.0
            and diagnostic["positive_hgat_pr_auc_datasets"] >= 2
            and diagnostic["worst_hgat_pr_auc_delta"] >= -0.02
        ),
    }
    decision["promote"] = bool(all(decision.values()))
    report = {
        "status": "second_development_screen_not_independent_confirmation",
        "seed": 87,
        "runs": runs,
        "diagnostic": diagnostic,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed87_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(report))
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
