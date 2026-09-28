#!/usr/bin/env python3
"""Evaluate the preregistered seed-87 shared-KAN HGAT screen.

V8 changes only HGAT-Lite's scalar relation projection.  This evaluator
compares it with the frozen linear V4-HG run and enforces the stopping rule in
``analysis/v8_kan_hgat/preregistered_seed87.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_v4_hgat_paper import (  # noqa: E402
    BEST_FILE,
    SCORE_FILE,
    _arm_metrics,
    _rank_audit,
    _sha256,
)


DATASETS = ("EXA", "PSM", "SMD", "SWAT")
SEED = 87
REQUIRED_ARRAYS = (
    "score",
    "train_score",
    "innovation_score",
    "train_innovation_score",
    "hgat_score",
    "train_hgat_score",
)


def _result_dir(score_root: Path, dataset: str, tag: str) -> Path:
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    matches = list(score_root.glob(
        f"anomaly_detection_{long_name}_*_{tag}_s{SEED}_0"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {dataset}/{tag}/seed-{SEED} result, found {matches}"
        )
    return matches[0]


def _load_run(score_root: Path, dataset: str, tag: str):
    directory = _result_dir(score_root, dataset, tag)
    score_path = directory / SCORE_FILE
    best_path = directory / BEST_FILE
    if not score_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {directory}")

    with np.load(score_path) as artifact:
        required = {*REQUIRED_ARRAYS, "label", "hgat_metadata_json"}
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{score_path} is missing {sorted(missing)}")
        arrays = {
            name: artifact[name].astype(np.float64, copy=False)
            for name in REQUIRED_ARRAYS
        }
        labels = artifact["label"].reshape(-1).astype(np.int8)
        entity_ids = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files else None
        )
        metadata = json.loads(str(artifact["hgat_metadata_json"].item()))
        dtypes = {name: str(artifact[name].dtype) for name in REQUIRED_ARRAYS}

    best = json.loads(best_path.read_text())
    full = _arm_metrics(
        dataset, labels, arrays["score"], arrays["train_score"], entity_ids
    )
    full["Best-F1+PA"] = {
        key: best[key] for key in (
            "f1", "precision", "recall", "threshold", "TP", "TN", "FP", "FN"
        )
    }
    graph = _arm_metrics(
        dataset, labels, arrays["hgat_score"],
        arrays["train_hgat_score"], entity_ids
    )
    return {
        "directory": directory,
        "score_path": score_path,
        "best_path": best_path,
        "arrays": arrays,
        "metadata": metadata,
        "dtypes": dtypes,
        "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
        "full": full,
        "graph": graph,
    }


def _checkpoint(checkpoint_root: Path, dataset: str, tag: str) -> Path:
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    matches = list(checkpoint_root.glob(
        f"anomaly_detection_{long_name}_*_{tag}_s{SEED}_0/checkpoint.pth"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {dataset}/{tag}/seed-{SEED} checkpoint, found {matches}"
        )
    return matches[0]


def _checkpoint_parameter_count(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    # BSplineBasis.grid is the only newly introduced persistent buffer.
    return int(sum(
        value.numel() for key, value in state.items()
        if torch.is_tensor(value) and not key.endswith(".basis.grid")
    ))


def _evaluate_dataset(args, dataset, candidate_tag="v8khg"):
    linear = _load_run(args.score_root, dataset, "v4hg")
    kan = _load_run(args.score_root, dataset, candidate_tag)
    linear_checkpoint = _checkpoint(args.linear_checkpoint_root, dataset, "v4hg")
    kan_checkpoint = _checkpoint(args.kan_checkpoint_root, dataset, candidate_tag)

    rank = _rank_audit(
        kan["arrays"]["innovation_score"], kan["arrays"]["score"]
    )
    graph_linear = linear["graph"]["ranking"]
    graph_kan = kan["graph"]["ranking"]
    pa = {}
    for protocol, path in (
        ("POT+PA", ("POT", "point_adjusted", "f1")),
        ("Best-F1+PA", ("Best-F1+PA", "f1")),
    ):
        linear_value = linear["full"]
        kan_value = kan["full"]
        for key in path:
            linear_value = linear_value[key]
            kan_value = kan_value[key]
        pa[protocol] = {
            "linear": float(linear_value),
            "kan": float(kan_value),
            "delta": float(kan_value - linear_value),
        }

    linear_parameters = _checkpoint_parameter_count(linear_checkpoint)
    kan_parameters = _checkpoint_parameter_count(kan_checkpoint)
    return {
        "linear_result_dir": str(linear["directory"].relative_to(REPO_ROOT)),
        "kan_result_dir": str(kan["directory"].relative_to(REPO_ROOT)),
        "linear_score_sha256": _sha256(linear["score_path"]),
        "kan_score_sha256": _sha256(kan["score_path"]),
        "kan_best_sha256": _sha256(kan["best_path"]),
        "parameters": {
            "linear": linear_parameters,
            "kan": kan_parameters,
            "added": kan_parameters - linear_parameters,
        },
        "finite": kan["finite"],
        "stored_dtypes": kan["dtypes"],
        "metadata": kan["metadata"],
        "rank_safety": rank,
        "ranking": {
            "full_roc_auc": kan["full"]["ranking"]["roc_auc"],
            "full_pr_auc": kan["full"]["ranking"]["pr_auc"],
            "linear_hgat_roc_auc": graph_linear["roc_auc"],
            "kan_hgat_roc_auc": graph_kan["roc_auc"],
            "delta_hgat_roc_auc": graph_kan["roc_auc"] - graph_linear["roc_auc"],
            "linear_hgat_pr_auc": graph_linear["pr_auc"],
            "kan_hgat_pr_auc": graph_kan["pr_auc"],
            "delta_hgat_pr_auc": graph_kan["pr_auc"] - graph_linear["pr_auc"],
        },
        "point_adjusted": pa,
    }


def _markdown(report):
    lines = [
        "# V8 shared-KAN HGAT seed-87 screen",
        "",
        "> This is the preregistered development screen on previously inspected test sets;",
        "> it is not independent confirmation and does not establish current SOTA.",
        "",
        "| Dataset | Linear HGAT AP | KAN HGAT AP | ΔAP | Linear HGAT ROC | KAN HGAT ROC | ΔROC | Added params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        run = report["runs"][dataset]
        rank = run["ranking"]
        lines.append(
            f"| {dataset} | {rank['linear_hgat_pr_auc']:.6f} | "
            f"{rank['kan_hgat_pr_auc']:.6f} | {rank['delta_hgat_pr_auc']:+.6f} | "
            f"{rank['linear_hgat_roc_auc']:.6f} | {rank['kan_hgat_roc_auc']:.6f} | "
            f"{rank['delta_hgat_roc_auc']:+.6f} | {run['parameters']['added']:+d} |"
        )
    lines.append(
        f"| **Macro** | — | — | **{report['diagnostic']['mean_hgat_pr_auc_delta']:+.6f}** | "
        f"— | — | **{report['diagnostic']['mean_hgat_roc_auc_delta']:+.6f}** | — |"
    )
    lines.extend([
        "",
        "| Dataset | POT+PA (linear) | POT+PA (KAN) | Δ | Best-F1+PA (linear) | Best-F1+PA (KAN) | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        pa = report["runs"][dataset]["point_adjusted"]
        lines.append(
            f"| {dataset} | {pa['POT+PA']['linear']:.6f} | "
            f"{pa['POT+PA']['kan']:.6f} | {pa['POT+PA']['delta']:+.8f} | "
            f"{pa['Best-F1+PA']['linear']:.6f} | "
            f"{pa['Best-F1+PA']['kan']:.6f} | {pa['Best-F1+PA']['delta']:+.8f} |"
        )
    decision = report["decision"]
    lines.extend([
        "",
        "## Frozen decision",
        "",
        f"- Finite artifacts: **{decision['finite_artifacts_pass']}**",
        f"- Added parameters <= 128 and consistently +72: **{decision['parameter_budget_pass']}**",
        f"- Zero distinct V4-rank inversions: **{decision['rank_safety_pass']}**",
        f"- Non-empty tie refinement: **{decision['non_vacuous_pass']}**",
        f"- PA compatibility within 1e-4: **{decision['pa_compatibility_pass']}**",
        f"- HGAT-only AP diagnostic: **{decision['hgat_ap_diagnostic_pass']}**",
        f"- Overall promotion: **{decision['promote']}**",
        "",
    ])
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
        "--kan-checkpoint-root", type=Path,
        default=REPO_ROOT / "checkpoints" / "ISTAD_v8_kan_hgat",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "v8_kan_hgat",
    )
    args = parser.parse_args()

    runs = {dataset: _evaluate_dataset(args, dataset) for dataset in DATASETS}
    ap_deltas = np.asarray([
        runs[dataset]["ranking"]["delta_hgat_pr_auc"] for dataset in DATASETS
    ])
    roc_deltas = np.asarray([
        runs[dataset]["ranking"]["delta_hgat_roc_auc"] for dataset in DATASETS
    ])
    diagnostic = {
        "mean_hgat_pr_auc_delta": float(ap_deltas.mean()),
        "mean_hgat_roc_auc_delta": float(roc_deltas.mean()),
        "datasets_with_hgat_pr_auc_gain_at_least_0.005": int(
            np.sum(ap_deltas >= 0.005)
        ),
    }
    decision = {
        "finite_artifacts_pass": all(run["finite"] for run in runs.values()),
        "parameter_budget_pass": all(
            run["parameters"]["added"] == 72
            and run["parameters"]["added"] <= 128
            for run in runs.values()
        ),
        "rank_safety_pass": all(
            run["rank_safety"]["distinct_rank_inversions"] == 0
            for run in runs.values()
        ),
        "non_vacuous_pass": all(
            run["rank_safety"]["changed_tie_groups"] > 0
            for run in runs.values()
        ),
        "pa_compatibility_pass": all(
            abs(protocol["delta"]) <= 1e-4
            for run in runs.values()
            for protocol in run["point_adjusted"].values()
        ),
        "hgat_ap_diagnostic_pass": bool(
            diagnostic["mean_hgat_pr_auc_delta"] >= -0.005
            and diagnostic["datasets_with_hgat_pr_auc_gain_at_least_0.005"] >= 2
        ),
    }
    decision["promote"] = bool(all(decision.values()))
    report = {
        "status": "preregistered_development_screen_not_independent_confirmation",
        "seed": SEED,
        "datasets": list(DATASETS),
        "runs": runs,
        "diagnostic": diagnostic,
        "decision": decision,
    }
    # Paths and arrays are runtime-only objects and are not part of the report.
    for run in report["runs"].values():
        run.pop("directory", None)
        run.pop("score_path", None)
        run.pop("best_path", None)
        run.pop("arrays", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "seed87_metrics.json"
    markdown_path = args.output_dir / "RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(_markdown(report))
    print(markdown_path)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
