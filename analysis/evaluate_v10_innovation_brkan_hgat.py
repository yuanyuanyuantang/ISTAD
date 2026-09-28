#!/usr/bin/env python3
"""Evaluate the frozen V10 innovation-input BR-KAN-HGAT seed-87 screen."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_v4_hgat_paper import _arm_metrics  # noqa: E402
from evaluate_v8_kan_hgat import (  # noqa: E402
    DATASETS,
    _checkpoint,
    _checkpoint_parameter_count,
)


SEED = 87
SCORE_FILE = "point_scores_innovation_hgat_w0p2_train_ecdf.npz"
BEST_FILE = "bestf1_threshold_results_innovation_hgat_w0p2_train_ecdf.json"


def _result_dir(score_root, dataset, tag="v10ibrk"):
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    matches = list(score_root.glob(
        f"anomaly_detection_{long_name}_*_{tag}_s{SEED}_0"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one V10 result for {dataset}, found {matches}")
    return matches[0]


def _spline_norm(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = [
        value for key, value in state.items()
        if key.endswith("value_projection.spline_weight")
    ]
    if len(values) != 1:
        raise RuntimeError(f"Expected one BR-KAN spline tensor in {checkpoint}")
    return float(torch.linalg.vector_norm(values[0]).item())


def _load_candidate(
    score_root, checkpoint_root, dataset, tag="v10ibrk", require_spline=True
):
    directory = _result_dir(score_root, dataset, tag)
    score_path = directory / SCORE_FILE
    best_path = directory / BEST_FILE
    with np.load(score_path) as artifact:
        required = {
            "score", "train_score", "label", "innovation_score",
            "train_innovation_score", "hgat_score", "train_hgat_score",
            "hgat_metadata_json",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{score_path} is missing {sorted(missing)}")
        arrays = {
            key: artifact[key].astype(np.float64, copy=False)
            for key in required if key not in {"label", "hgat_metadata_json"}
        }
        label = artifact["label"].reshape(-1).astype(np.int8)
        entity_ids = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files else None
        )
        metadata = json.loads(str(artifact["hgat_metadata_json"].item()))
    full = _arm_metrics(
        dataset, label, arrays["score"], arrays["train_score"], entity_ids
    )
    best = json.loads(best_path.read_text())
    full["Best-F1+PA"] = {
        key: best[key] for key in (
            "f1", "precision", "recall", "threshold", "TP", "TN", "FP", "FN"
        )
    }
    graph = _arm_metrics(
        dataset, label, arrays["hgat_score"],
        arrays["train_hgat_score"], entity_ids,
    )
    checkpoint = _checkpoint(checkpoint_root, dataset, tag)
    return {
        "result_dir": str(directory.relative_to(REPO_ROOT)),
        "full": full,
        "graph": graph,
        "metadata": metadata,
        "parameters": _checkpoint_parameter_count(checkpoint),
        "spline_l2_norm": _spline_norm(checkpoint) if require_spline else None,
        "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
    }


def _markdown(report):
    lines = [
        "# V10 innovation-guided BR-KAN-HGAT seed-87 screen",
        "",
        "> Development screen on previously inspected test sets; not independent confirmation.",
        "",
        "| Dataset | V4-HG AP | V10 AP | ΔAP | V10 HGAT AP | V4-HG ROC | V10 ROC | ΔROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        run = report["runs"][dataset]
        lines.append(
            f"| {dataset} | {run['baseline']['ap']:.6f} | {run['candidate']['ap']:.6f} | "
            f"{run['delta']['ap']:+.6f} | {run['graph_ap']:.6f} | "
            f"{run['baseline']['roc']:.6f} | {run['candidate']['roc']:.6f} | "
            f"{run['delta']['roc']:+.6f} |"
        )
    macro = report["macro"]
    lines.append(
        f"| **Macro** | {macro['baseline_ap']:.6f} | {macro['candidate_ap']:.6f} | "
        f"**{macro['delta_ap']:+.6f}** | {macro['graph_ap']:.6f} | "
        f"{macro['baseline_roc']:.6f} | {macro['candidate_roc']:.6f} | "
        f"**{macro['delta_roc']:+.6f}** |"
    )
    lines.extend([
        "",
        "| Dataset | V4-HG POT+PA | V10 POT+PA | Δ | V4-HG Best-F1+PA | V10 Best-F1+PA | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        run = report["runs"][dataset]
        lines.append(
            f"| {dataset} | {run['baseline']['pot_pa']:.6f} | "
            f"{run['candidate']['pot_pa']:.6f} | {run['delta']['pot_pa']:+.6f} | "
            f"{run['baseline']['best_pa']:.6f} | {run['candidate']['best_pa']:.6f} | "
            f"{run['delta']['best_pa']:+.6f} |"
        )
    lines.append(
        f"| **Macro** | {macro['baseline_pot_pa']:.6f} | {macro['candidate_pot_pa']:.6f} | "
        f"**{macro['delta_pot_pa']:+.6f}** | {macro['baseline_best_pa']:.6f} | "
        f"{macro['candidate_best_pa']:.6f} | **{macro['delta_best_pa']:+.6f}** |"
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
        "--checkpoint-root", type=Path,
        default=REPO_ROOT / "checkpoints" / "ISTAD_v10_innovation_brkan_hgat",
    )
    parser.add_argument(
        "--baseline-checkpoint-root", type=Path,
        default=REPO_ROOT / "checkpoints" / "ISTAD_v4_hgat_integrated",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "v10_innovation_brkan_hgat",
    )
    args = parser.parse_args()
    baseline_report = json.loads(
        (REPO_ROOT / "analysis" / "v4_hgat_paper" / "paper_metrics.json").read_text()
    )

    runs = {}
    for dataset in DATASETS:
        candidate = _load_candidate(args.score_root, args.checkpoint_root, dataset)
        baseline_arm = baseline_report["runs"][dataset][str(SEED)]["arms"]["V4-HG"]
        baseline = {
            "ap": baseline_arm["ranking"]["pr_auc"],
            "roc": baseline_arm["ranking"]["roc_auc"],
            "pot_pa": baseline_arm["POT"]["point_adjusted"]["f1"],
            "best_pa": baseline_arm["Best-F1+PA"]["f1"],
        }
        current = {
            "ap": candidate["full"]["ranking"]["pr_auc"],
            "roc": candidate["full"]["ranking"]["roc_auc"],
            "pot_pa": candidate["full"]["POT"]["point_adjusted"]["f1"],
            "best_pa": candidate["full"]["Best-F1+PA"]["f1"],
        }
        baseline_params = _checkpoint_parameter_count(_checkpoint(
            args.baseline_checkpoint_root, dataset, "v4hg"
        ))
        runs[dataset] = {
            "result_dir": candidate["result_dir"],
            "baseline": baseline,
            "candidate": current,
            "delta": {key: current[key] - baseline[key] for key in baseline},
            "graph_ap": candidate["graph"]["ranking"]["pr_auc"],
            "graph_roc": candidate["graph"]["ranking"]["roc_auc"],
            "parameters": {
                "baseline": baseline_params,
                "candidate": candidate["parameters"],
                "added": candidate["parameters"] - baseline_params,
            },
            "spline_l2_norm": candidate["spline_l2_norm"],
            "metadata": candidate["metadata"],
            "finite": candidate["finite"],
        }

    def mean(path):
        return float(np.mean([
            runs[dataset][path[0]][path[1]] for dataset in DATASETS
        ]))

    macro = {}
    for metric in ("ap", "roc", "pot_pa", "best_pa"):
        macro[f"baseline_{metric}"] = mean(("baseline", metric))
        macro[f"candidate_{metric}"] = mean(("candidate", metric))
        macro[f"delta_{metric}"] = (
            macro[f"candidate_{metric}"] - macro[f"baseline_{metric}"]
        )
    macro["graph_ap"] = float(np.mean([
        runs[dataset]["graph_ap"] for dataset in DATASETS
    ]))
    ap_deltas = np.asarray([runs[dataset]["delta"]["ap"] for dataset in DATASETS])
    decision = {
        "finite_artifacts_pass": all(run["finite"] for run in runs.values()),
        "innovation_input_pass": all(
            run["metadata"].get("relation_input")
            == "signed_standardized_var_innovation" for run in runs.values()
        ),
        "parameter_budget_pass": all(
            run["parameters"]["added"] == 64 for run in runs.values()
        ),
        "learned_spline_pass": all(
            run["spline_l2_norm"] > 1e-6 for run in runs.values()
        ),
        "macro_ap_pass": macro["delta_ap"] >= 0.005,
        "dataset_ap_pass": int(np.sum(ap_deltas >= 0.0)) >= 3,
        "worst_ap_pass": float(ap_deltas.min()) >= -0.003,
        "pot_pa_macro_pass": macro["delta_pot_pa"] >= -0.001,
        "best_pa_macro_pass": macro["delta_best_pa"] >= -0.001,
    }
    decision["continue_to_three_seeds"] = bool(all(decision.values()))
    report = {
        "status": "development_screen_not_independent_confirmation",
        "seed": SEED,
        "runs": runs,
        "macro": macro,
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
