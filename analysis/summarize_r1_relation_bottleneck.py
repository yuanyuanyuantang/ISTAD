#!/usr/bin/env python3
"""Validate and summarize the masked-channel relation bottleneck experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "code" / "ISTAD"))

from utils.event_metrics import event_overlap_metrics  # noqa: E402


DATASETS = ("EXATHLON", "PSM", "SMD", "SWAT")
SEEDS = (87, 90, 98)
ARMS = (
    "learned_dynamic",
    "learned_static",
    "fixed_random",
    "no_message",
)
ARM_LABELS = {
    "learned_dynamic": "Learned dynamic incidence",
    "learned_static": "Learned static incidence",
    "fixed_random": "Fixed balanced-random incidence",
    "no_message": "Without hypergraph message passing",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "population_std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _expected_runs(scope):
    if scope == "smoke":
        return [("PSM", 87, arm) for arm in ARMS]
    return [
        (dataset, seed, arm)
        for dataset in DATASETS
        for seed in SEEDS
        for arm in ARMS
    ]


def _find_result(result_root: Path, dataset: str, seed: int, arm: str) -> Path:
    prefix = f"anomaly_detection_{dataset}_"
    suffix = f"_r1_{arm}_s{seed}_0"
    matches = [
        path.parent
        for path in result_root.rglob("point_scores.npz")
        if path.parent.name.startswith(prefix) and path.parent.name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one R1 artifact for {dataset}/seed-{seed}/{arm}, found {matches}"
        )
    return matches[0]


def _point_metrics(label, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(prediction.sum()),
    }


def _best_raw(label, score):
    precision, recall, thresholds = precision_recall_curve(label, score)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    index = int(np.argmax(f1))
    return {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "threshold": float(np.nextafter(thresholds[index], -np.inf)),
        "selection": "test_label_oracle_diagnostic_without_point_adjustment",
    }


def _evaluate_one(result_dir: Path, checkpoint_root: Path, percentile: float):
    score_path = result_dir / "point_scores.npz"
    diagnostic_path = result_dir / "relation_bottleneck_diagnostics.json"
    checkpoint_path = checkpoint_root / result_dir.name / "checkpoint.pth"
    for required in (score_path, diagnostic_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing R1 artifact: {required}")

    with np.load(score_path) as artifact:
        missing = {"score", "train_score", "label"}.difference(artifact.files)
        if missing:
            raise KeyError(f"{score_path} is missing {sorted(missing)}")
        score = artifact["score"].reshape(-1).astype(np.float64)
        train_score = artifact["train_score"].reshape(-1).astype(np.float64)
        label = artifact["label"].reshape(-1).astype(np.int8)
        entity_ids = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files else None
        )
        stored_dtypes = {name: str(artifact[name].dtype) for name in artifact.files}

    if len(score) != len(label) or not len(score) or not len(train_score):
        raise ValueError(f"invalid score lengths in {score_path}")
    if entity_ids is not None and len(entity_ids) != len(label):
        raise ValueError(f"entity/label length mismatch in {score_path}")
    if not np.isfinite(score).all() or not np.isfinite(train_score).all():
        raise ValueError(f"non-finite score in {score_path}")
    if not np.isin(label, (0, 1)).all() or np.unique(label).size != 2:
        raise ValueError(f"labels must contain both binary classes in {score_path}")

    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic.get("protocol") != "masked_channel_relation_bottleneck_v1":
        raise ValueError(f"unexpected diagnostic protocol in {diagnostic_path}")
    required_diagnostics = {
        "masked_validation_mse",
        "frozen_no_message_masked_validation_mse",
        "relative_loss_increase_without_message",
        "hgat_output_variance_mean",
        "incidence_entropy_normalized_mean",
        "message_gate_mean",
        "training",
    }
    missing = required_diagnostics.difference(diagnostic)
    if missing:
        raise KeyError(f"{diagnostic_path} is missing {sorted(missing)}")

    threshold = float(np.percentile(train_score, percentile))
    prediction = (score > threshold).astype(np.int8)
    point = _point_metrics(label, prediction)
    event = event_overlap_metrics(label, prediction, entity_ids)
    return {
        "setting": result_dir.name,
        "score_artifact": {
            "path": str(score_path),
            "sha256": _sha256(score_path),
            "bytes": score_path.stat().st_size,
            "stored_dtypes": stored_dtypes,
        },
        "diagnostic_artifact": {
            "path": str(diagnostic_path),
            "sha256": _sha256(diagnostic_path),
            "bytes": diagnostic_path.stat().st_size,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "n_train": int(len(train_score)),
        "n_test": int(len(score)),
        "roc_auc": float(roc_auc_score(label, score)),
        "average_precision": float(average_precision_score(label, score)),
        "train_threshold": {
            "percentile": float(percentile),
            "value": threshold,
            "point": point,
            "event": event,
        },
        "best_raw": _best_raw(label, score),
        "diagnostics": diagnostic,
    }


def _score_equal(left, right):
    with np.load(left["score_artifact"]["path"]) as item:
        left_score = item["score"].reshape(-1)
    with np.load(right["score_artifact"]["path"]) as item:
        right_score = item["score"].reshape(-1)
    return bool(np.array_equal(left_score, right_score))


def _aggregate(runs, scope):
    datasets = ("PSM",) if scope == "smoke" else DATASETS
    seeds = (87,) if scope == "smoke" else SEEDS
    report = {}
    for dataset in datasets:
        report[dataset] = {"arms": {}, "dynamic_minus_no_message": {}}
        for arm in ARMS:
            selected = [runs[(dataset, seed, arm)] for seed in seeds]
            report[dataset]["arms"][arm] = {
                "roc_auc": _summary([run["roc_auc"] for run in selected]),
                "average_precision": _summary([
                    run["average_precision"] for run in selected
                ]),
                "train_p99_point_f1": _summary([
                    run["train_threshold"]["point"]["f1"] for run in selected
                ]),
                "train_p99_event_f1": _summary([
                    run["train_threshold"]["event"]["event_f1"] for run in selected
                ]),
                "masked_validation_mse": _summary([
                    run["diagnostics"]["masked_validation_mse"] for run in selected
                ]),
            }
        dynamic = report[dataset]["arms"]["learned_dynamic"]
        no_message = report[dataset]["arms"]["no_message"]
        report[dataset]["dynamic_minus_no_message"] = {
            "roc_auc": dynamic["roc_auc"]["mean"] - no_message["roc_auc"]["mean"],
            "average_precision": (
                dynamic["average_precision"]["mean"]
                - no_message["average_precision"]["mean"]
            ),
            "relative_masked_mse_improvement": (
                no_message["masked_validation_mse"]["mean"]
                - dynamic["masked_validation_mse"]["mean"]
            ) / max(no_message["masked_validation_mse"]["mean"], np.finfo(float).eps),
        }
    return report


def _decision(runs, scope):
    expected = _expected_runs(scope)
    dynamic_runs = [runs[key] for key in expected if key[2] == "learned_dynamic"]
    no_message_runs = [runs[key] for key in expected if key[2] == "no_message"]
    relative_improvements = []
    for dynamic, no_message in zip(dynamic_runs, no_message_runs):
        dynamic_mse = dynamic["diagnostics"]["masked_validation_mse"]
        no_message_mse = no_message["diagnostics"]["masked_validation_mse"]
        relative_improvements.append(
            (no_message_mse - dynamic_mse)
            / max(no_message_mse, np.finfo(float).eps)
        )
    distinct_scores = all(
        not _score_equal(
            runs[(dataset, seed, "learned_dynamic")],
            runs[(dataset, seed, control)],
        )
        for dataset, seed, _ in expected
        if _ == "learned_dynamic"
        for control in ARMS[1:]
    )
    dynamic_gradients = [
        run["diagnostics"]["training"]["mean_epoch_hgat_gradient_norm"]
        for run in dynamic_runs
    ]
    output_variances = [
        run["diagnostics"]["hgat_output_variance_mean"]
        for run in dynamic_runs
    ]
    frozen_increases = [
        run["diagnostics"]["relative_loss_increase_without_message"]
        for run in dynamic_runs
    ]
    checks = {
        "all_dynamic_scores_differ_from_controls": distinct_scores,
        "dynamic_hgat_receives_gradient": bool(all(value > 0.0 for value in dynamic_gradients)),
        "dynamic_hgat_output_is_nonconstant": bool(all(value > 1e-12 for value in output_variances)),
        "frozen_message_removal_increases_loss": bool(all(value > 0.0 for value in frozen_increases)),
        "retrained_dynamic_improves_masked_mse_by_at_least_2_percent": bool(
            all(value >= 0.02 for value in relative_improvements)
        ),
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "paired_relative_masked_mse_improvements": relative_improvements,
        "mean_relative_masked_mse_improvement": float(np.mean(relative_improvements)),
        "recommendation": (
            "The smoke gate passed; the full 48-run R1 experiment may proceed."
            if passed and scope == "smoke"
            else "Do not launch the full R1 experiment; inspect the failed checks first."
            if not passed and scope == "smoke"
            else "Interpret the full multi-seed evidence before changing the manuscript."
        ),
    }


def _markdown(report):
    lines = [
        "# ISTAD R1 masked-channel relation bottleneck",
        "",
        "> All target cells are scored only while hidden. The causal encoder and",
        "> decoder are channelwise, so HGAT is the only cross-channel path.",
        "> Test-label Best-F1 is retained only as a diagnostic.",
        "",
        f"Decision: **{report['decision']['status'].upper()}**",
        "",
        "## Runs",
        "",
        "| Dataset | Seed | Arm | ROC-AUC | AP | P99 F1 | Event F1 | Val masked MSE | Frozen removal delta | HGAT grad |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, seed, arm in report["expected_runs"]:
        run = report["runs"][dataset][str(seed)][arm]
        diagnostics = run["diagnostics"]
        lines.append(
            f"| {dataset} | {seed} | {ARM_LABELS[arm]} | "
            f"{run['roc_auc']:.6f} | {run['average_precision']:.6f} | "
            f"{run['train_threshold']['point']['f1']:.6f} | "
            f"{run['train_threshold']['event']['event_f1']:.6f} | "
            f"{diagnostics['masked_validation_mse']:.6f} | "
            f"{diagnostics['relative_loss_increase_without_message']:.6f} | "
            f"{diagnostics['training']['mean_epoch_hgat_gradient_norm']:.6f} |"
        )
    lines.extend(["", "## Predeclared gate", ""])
    for name, passed in report["decision"]["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend([
        "",
        f"Mean retrained dynamic MSE improvement over no-message: "
        f"{100.0 * report['decision']['mean_relative_masked_mse_improvement']:.3f}%",
        "",
        report["decision"]["recommendation"],
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    args = parser.parse_args()

    expected = _expected_runs(args.scope)
    runs = {}
    for dataset, seed, arm in expected:
        result_dir = _find_result(args.result_root, dataset, seed, arm)
        runs[(dataset, seed, arm)] = _evaluate_one(
            result_dir, args.checkpoint_root, args.threshold_percentile
        )

    report = {
        "protocol": "masked_channel_relation_bottleneck_v1",
        "scope": args.scope,
        "expected_runs": expected,
        "runs": {},
        "aggregate": _aggregate(runs, args.scope),
        "decision": _decision(runs, args.scope),
    }
    for dataset, seed, arm in expected:
        report["runs"].setdefault(dataset, {}).setdefault(str(seed), {})[arm] = (
            runs[(dataset, seed, arm)]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metrics_summary.json"
    results_path = args.output_dir / "RESULTS.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path.write_text(_markdown(report), encoding="utf-8")

    manifest = {
        "protocol": report["protocol"],
        "scope": args.scope,
        "artifacts": [
            {
                "dataset": dataset,
                "seed": seed,
                "arm": arm,
                "score_sha256": runs[(dataset, seed, arm)]["score_artifact"]["sha256"],
                "diagnostic_sha256": runs[(dataset, seed, arm)]["diagnostic_artifact"]["sha256"],
                "checkpoint_sha256": runs[(dataset, seed, arm)]["checkpoint"]["sha256"],
            }
            for dataset, seed, arm in expected
        ],
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output_files = (results_path, summary_path, manifest_path)
    (args.output_dir / "sha256sum.txt").write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n" for path in output_files
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
