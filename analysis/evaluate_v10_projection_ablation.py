#!/usr/bin/env python3
"""Compare innovation-input Linear-HGAT and BR-KAN-HGAT at seed 87."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_v10_innovation_brkan_hgat import DATASETS, REPO_ROOT, _load_candidate


def _metrics(run):
    full = run["full"]
    return {
        "ap": full["ranking"]["pr_auc"],
        "roc": full["ranking"]["roc_auc"],
        "pot_pa": full["POT"]["point_adjusted"]["f1"],
        "best_pa": full["Best-F1+PA"]["f1"],
    }


def _markdown(report):
    lines = [
        "# V10 projection ablation: Linear vs BR-KAN",
        "",
        "> Post-hoc seed-87 explanatory ablation on previously inspected test sets;",
        "> it cannot promote V10 or reopen seeds 90/98.",
        "",
        "| Dataset | Linear AP | BR-KAN AP | ΔAP | Linear ROC | BR-KAN ROC | ΔROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        run = report["runs"][dataset]
        lines.append(
            f"| {dataset} | {run['linear']['ap']:.6f} | {run['brkan']['ap']:.6f} | "
            f"{run['delta']['ap']:+.6f} | {run['linear']['roc']:.6f} | "
            f"{run['brkan']['roc']:.6f} | {run['delta']['roc']:+.6f} |"
        )
    macro = report["macro"]
    lines.append(
        f"| **Macro** | {macro['linear']['ap']:.6f} | {macro['brkan']['ap']:.6f} | "
        f"**{macro['delta']['ap']:+.6f}** | {macro['linear']['roc']:.6f} | "
        f"{macro['brkan']['roc']:.6f} | **{macro['delta']['roc']:+.6f}** |"
    )
    lines.extend([
        "",
        "| Dataset | Linear POT+PA | BR-KAN POT+PA | Δ | Linear Best-F1+PA | BR-KAN Best-F1+PA | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        run = report["runs"][dataset]
        lines.append(
            f"| {dataset} | {run['linear']['pot_pa']:.6f} | "
            f"{run['brkan']['pot_pa']:.6f} | {run['delta']['pot_pa']:+.6f} | "
            f"{run['linear']['best_pa']:.6f} | {run['brkan']['best_pa']:.6f} | "
            f"{run['delta']['best_pa']:+.6f} |"
        )
    lines.append(
        f"| **Macro** | {macro['linear']['pot_pa']:.6f} | "
        f"{macro['brkan']['pot_pa']:.6f} | **{macro['delta']['pot_pa']:+.6f}** | "
        f"{macro['linear']['best_pa']:.6f} | {macro['brkan']['best_pa']:.6f} | "
        f"**{macro['delta']['best_pa']:+.6f}** |"
    )
    decision = report["decision"]
    lines.extend([
        "",
        "## Frozen decision",
        "",
        f"- Finite and matched innovation inputs: **{decision['valid_comparison']}**",
        f"- BR-KAN Macro AP exceeds Linear: **{decision['macro_ap_pass']}**",
        f"- BR-KAN AP non-negative on at least 2/4 datasets: "
        f"**{decision['dataset_ap_pass']}** "
        f"({decision['datasets_with_nonnegative_ap_delta']}/4)",
        f"- Retain BR-KAN: **{decision['retain_brkan']}**",
        "",
        f"BR-KAN adds {report['parameters']['brkan_minus_linear']:+d} parameters. "
        "Under the frozen rule, the smaller Linear projection is preferred when "
        "`retain_brkan` is false.",
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
        default=REPO_ROOT / "checkpoints" / "ISTAD_v10_innovation_linear_hgat",
    )
    parser.add_argument(
        "--brkan-checkpoint-root", type=Path,
        default=REPO_ROOT / "checkpoints" / "ISTAD_v10_innovation_brkan_hgat",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "v10_innovation_brkan_hgat",
    )
    args = parser.parse_args()

    runs = {}
    parameter_deltas = []
    for dataset in DATASETS:
        linear_run = _load_candidate(
            args.score_root, args.linear_checkpoint_root, dataset,
            tag="v10ilin", require_spline=False,
        )
        brkan_run = _load_candidate(
            args.score_root, args.brkan_checkpoint_root, dataset,
        )
        linear = _metrics(linear_run)
        brkan = _metrics(brkan_run)
        delta = {key: brkan[key] - linear[key] for key in linear}
        parameter_delta = brkan_run["parameters"] - linear_run["parameters"]
        parameter_deltas.append(parameter_delta)
        runs[dataset] = {
            "linear": linear,
            "brkan": brkan,
            "delta": delta,
            "parameters": {
                "linear": linear_run["parameters"],
                "brkan": brkan_run["parameters"],
                "added": parameter_delta,
            },
            "linear_result_dir": linear_run["result_dir"],
            "brkan_result_dir": brkan_run["result_dir"],
            "finite": linear_run["finite"] and brkan_run["finite"],
            "relation_inputs": {
                "linear": linear_run["metadata"].get("relation_input"),
                "brkan": brkan_run["metadata"].get("relation_input"),
            },
        }

    macro = {arm: {} for arm in ("linear", "brkan", "delta")}
    for metric in ("ap", "roc", "pot_pa", "best_pa"):
        for arm in macro:
            macro[arm][metric] = float(np.mean([
                runs[dataset][arm][metric] for dataset in DATASETS
            ]))

    ap_deltas = np.asarray([runs[d]["delta"]["ap"] for d in DATASETS])
    expected_input = "signed_standardized_var_innovation"
    valid = all(
        run["finite"]
        and run["relation_inputs"]["linear"] == expected_input
        and run["relation_inputs"]["brkan"] == expected_input
        for run in runs.values()
    )
    decision = {
        "valid_comparison": valid,
        "macro_ap_pass": macro["delta"]["ap"] > 0.0,
        "datasets_with_nonnegative_ap_delta": int(np.sum(ap_deltas >= 0.0)),
        "dataset_ap_pass": int(np.sum(ap_deltas >= 0.0)) >= 2,
    }
    decision["retain_brkan"] = bool(
        decision["valid_comparison"]
        and decision["macro_ap_pass"]
        and decision["dataset_ap_pass"]
    )
    if len(set(parameter_deltas)) != 1:
        raise RuntimeError(f"Inconsistent parameter deltas: {parameter_deltas}")
    report = {
        "status": "post_hoc_explanatory_ablation_not_independent_confirmation",
        "seed": 87,
        "changed_factor": "HGAT scalar projection: linear -> bounded-residual KAN",
        "runs": runs,
        "macro": macro,
        "parameters": {"brkan_minus_linear": parameter_deltas[0]},
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "projection_ablation_seed87.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "PROJECTION_ABLATION_RESULTS.md").write_text(
        _markdown(report)
    )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
