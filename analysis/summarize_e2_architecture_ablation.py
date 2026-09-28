#!/usr/bin/env python3
"""Summarize the E2 reconstruction-path HGAT architecture ablation."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))

from utils.event_metrics import event_overlap_metrics  # noqa: E402


DATASETS = ("EXA", "PSM", "SMD", "SWAT")
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
METRICS = (
    "roc_auc",
    "average_precision",
    "train_p99_point_f1",
    "train_p99_event_f1",
    "best_raw_f1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _point_metrics(label, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(np.sum(prediction)),
    }


def _exact_best_raw(label, score):
    precision, recall, thresholds = precision_recall_curve(label, score)
    if len(thresholds) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": None}
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
        "selection": "exact_test_label_oracle_no_point_adjustment",
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


def _long_dataset(dataset):
    return "EXATHLON" if dataset == "EXA" else dataset


def _find_result(result_root: Path, dataset: str, seed: int, arm: str) -> Path:
    suffix = f"_e2_{arm}_s{seed}_0"
    prefix = f"anomaly_detection_{_long_dataset(dataset)}_"
    matches = [
        path.parent
        for path in result_root.rglob("point_scores.npz")
        if path.parent.name.startswith(prefix) and path.parent.name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one E2 score artifact for {dataset}/seed-{seed}/{arm}, "
            f"found {matches}"
        )
    return matches[0]


def _evaluate_one(result_dir: Path, checkpoint_root: Path, percentile: float):
    score_path = result_dir / "point_scores.npz"
    with np.load(score_path) as artifact:
        required = {"score", "train_score", "label"}
        missing = required.difference(artifact.files)
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

    if len(score) != len(label):
        raise ValueError(f"score/label length mismatch in {score_path}")
    if entity_ids is not None and len(entity_ids) != len(label):
        raise ValueError(f"entity/label length mismatch in {score_path}")
    if not np.isfinite(score).all() or not np.isfinite(train_score).all():
        raise ValueError(f"non-finite score in {score_path}")
    if not np.isin(label, (0, 1)).all() or np.unique(label).size != 2:
        raise ValueError(f"labels must be binary with two classes in {score_path}")

    threshold = float(np.percentile(train_score, percentile))
    prediction = (score > threshold).astype(np.int8)
    point = _point_metrics(label, prediction)
    event = event_overlap_metrics(label, prediction, entity_ids)
    best = _exact_best_raw(label, score)

    checkpoint_path = checkpoint_root / result_dir.name / "checkpoint.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    train_time_path = result_dir / "train_time_summary.json"
    train_time = (
        json.loads(train_time_path.read_text())
        if train_time_path.is_file() else None
    )

    return {
        "setting": result_dir.name,
        "score_artifact": {
            "path": str(score_path),
            "sha256": _sha256(score_path),
            "bytes": score_path.stat().st_size,
            "stored_dtypes": stored_dtypes,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "n_train": int(len(train_score)),
        "n_test": int(len(score)),
        "anomaly_rate": float(label.mean()),
        "roc_auc": float(roc_auc_score(label, score)),
        "average_precision": float(average_precision_score(label, score)),
        "train_threshold": {
            "percentile": float(percentile),
            "value": threshold,
            "point": point,
            "event": event,
        },
        "best_raw": best,
        "train_time": train_time,
        "score_statistics": {
            "train_mean": float(train_score.mean()),
            "train_std": float(train_score.std(ddof=0)),
            "test_mean": float(score.mean()),
            "test_std": float(score.std(ddof=0)),
        },
    }


def _metric(run, name):
    if name == "train_p99_point_f1":
        return run["train_threshold"]["point"]["f1"]
    if name == "train_p99_event_f1":
        return run["train_threshold"]["event"]["event_f1"]
    if name == "best_raw_f1":
        return run["best_raw"]["f1"]
    return run[name]


def _score_difference(left, right):
    with np.load(left["score_artifact"]["path"]) as artifact:
        left_score = artifact["score"].reshape(-1).astype(np.float64)
    with np.load(right["score_artifact"]["path"]) as artifact:
        right_score = artifact["score"].reshape(-1).astype(np.float64)
    if left_score.shape != right_score.shape:
        raise ValueError("paired E2 score arrays must have equal shape")
    difference = np.abs(left_score - right_score)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "exactly_equal": bool(np.array_equal(left_score, right_score)),
    }


def _aggregate(runs, scope):
    datasets = ("PSM",) if scope == "smoke" else DATASETS
    seeds = (87,) if scope == "smoke" else SEEDS
    per_dataset = {}
    for dataset in datasets:
        per_dataset[dataset] = {
            "arms": {},
            "dynamic_deltas": {},
            "dynamic_score_differences": {},
        }
        for arm in ARMS:
            arm_runs = [runs[(dataset, seed, arm)] for seed in seeds]
            per_dataset[dataset]["arms"][arm] = {
                metric: _summary([_metric(run, metric) for run in arm_runs])
                for metric in METRICS
            }
        dynamic = [runs[(dataset, seed, "learned_dynamic")] for seed in seeds]
        for control in ARMS[1:]:
            control_runs = [runs[(dataset, seed, control)] for seed in seeds]
            per_dataset[dataset]["dynamic_deltas"][control] = {}
            score_differences = [
                _score_difference(left, right)
                for left, right in zip(dynamic, control_runs)
            ]
            per_dataset[dataset]["dynamic_score_differences"][control] = {
                "max_abs": _summary([item["max_abs"] for item in score_differences]),
                "mean_abs": _summary([item["mean_abs"] for item in score_differences]),
                "exactly_equal_runs": int(sum(
                    item["exactly_equal"] for item in score_differences
                )),
            }
            for metric in METRICS:
                deltas = [
                    _metric(left, metric) - _metric(right, metric)
                    for left, right in zip(dynamic, control_runs)
                ]
                tolerance = 1e-12
                per_dataset[dataset]["dynamic_deltas"][control][metric] = {
                    **_summary(deltas),
                    "wins": int(sum(value > tolerance for value in deltas)),
                    "ties": int(sum(abs(value) <= tolerance for value in deltas)),
                    "losses": int(sum(value < -tolerance for value in deltas)),
                }

    macro = {"arms": {}, "dynamic_deltas": {}}
    expected = _expected_runs(scope)
    for arm in ARMS:
        arm_runs = [runs[key] for key in expected if key[2] == arm]
        macro["arms"][arm] = {
            metric: _summary([_metric(run, metric) for run in arm_runs])
            for metric in METRICS
        }
    dynamic_keys = [key for key in expected if key[2] == "learned_dynamic"]
    for control in ARMS[1:]:
        macro["dynamic_deltas"][control] = {}
        for metric in METRICS:
            deltas = []
            for dataset, seed, _ in dynamic_keys:
                deltas.append(
                    _metric(runs[(dataset, seed, "learned_dynamic")], metric)
                    - _metric(runs[(dataset, seed, control)], metric)
                )
            tolerance = 1e-12
            macro["dynamic_deltas"][control][metric] = {
                **_summary(deltas),
                "wins": int(sum(value > tolerance for value in deltas)),
                "ties": int(sum(abs(value) <= tolerance for value in deltas)),
                "losses": int(sum(value < -tolerance for value in deltas)),
            }
    return per_dataset, macro


def _fmt(summary):
    return f"{summary['mean']:.6f} +/- {summary['population_std']:.6f}"


def _markdown(report):
    lines = [
        "# ISTAD E2 reconstruction-path architecture ablation",
        "",
        "> Frozen protocol: all arms use the same reconstruction-error score,",
        "> train-only 99th-percentile threshold, data split, decoder dimensions,",
        "> optimization settings, and paired seeds. Best raw F1 is an oracle",
        "> diagnostic and is not a deployable primary metric.",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Per-run metrics",
        "",
        "| Dataset | Seed | Arm | ROC-AUC | AP | Train-P99 raw F1 | Event-F1 | Best raw F1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key in report["expected_runs"]:
        dataset, seed, arm = key
        run = report["runs"][dataset][str(seed)][arm]
        lines.append(
            f"| {dataset} | {seed} | {ARM_LABELS[arm]} | "
            f"{run['roc_auc']:.6f} | {run['average_precision']:.6f} | "
            f"{run['train_threshold']['point']['f1']:.6f} | "
            f"{run['train_threshold']['event']['event_f1']:.6f} | "
            f"{run['best_raw']['f1']:.6f} |"
        )

    lines.extend([
        "",
        "## Aggregate metrics",
        "",
        "| Arm | ROC-AUC | AP | Train-P99 raw F1 | Event-F1 | Best raw F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        values = report["macro"]["arms"][arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {_fmt(values['roc_auc'])} | "
            f"{_fmt(values['average_precision'])} | "
            f"{_fmt(values['train_p99_point_f1'])} | "
            f"{_fmt(values['train_p99_event_f1'])} | "
            f"{_fmt(values['best_raw_f1'])} |"
        )

    lines.extend([
        "",
        "## Learned dynamic incidence minus controls",
        "",
        "| Control | Delta ROC-AUC | W/T/L | Delta AP | W/T/L |",
        "|---|---:|---:|---:|---:|",
    ])
    for control in ARMS[1:]:
        values = report["macro"]["dynamic_deltas"][control]
        roc = values["roc_auc"]
        ap = values["average_precision"]
        lines.append(
            f"| {ARM_LABELS[control]} | {roc['mean']:+.6f} | "
            f"{roc['wins']}/{roc['ties']}/{roc['losses']} | "
            f"{ap['mean']:+.6f} | {ap['wins']}/{ap['ties']}/{ap['losses']} |"
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- This experiment audits the neural reconstruction path, not the closed-form",
        "  causal-innovation score used by the current paper-facing detector.",
        "- No point adjustment is used in the reported E2 metrics.",
        "- The 99th-percentile threshold uses normal-training scores only.",
        "- Best raw F1 uses test labels and is reported only as an oracle diagnostic.",
        "- These benchmark labels have already been inspected, so the experiment is",
        "  developmental evidence rather than independent confirmation.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--train-percentile", type=float, default=99.0)
    args = parser.parse_args()
    if not 0.0 < args.train_percentile < 100.0:
        parser.error("--train-percentile must lie in (0, 100)")

    expected = _expected_runs(args.scope)
    runs = {}
    sources = []
    for dataset, seed, arm in expected:
        result_dir = _find_result(args.result_root, dataset, seed, arm)
        run = _evaluate_one(result_dir, args.checkpoint_root, args.train_percentile)
        runs[(dataset, seed, arm)] = run
        sources.append({
            "dataset": dataset,
            "seed": seed,
            "arm": arm,
            "setting": run["setting"],
            "score_sha256": run["score_artifact"]["sha256"],
            "checkpoint_sha256": run["checkpoint"]["sha256"],
        })

    per_dataset, macro = _aggregate(runs, args.scope)
    if args.scope == "smoke":
        degenerate = [
            control
            for control, values in per_dataset["PSM"][
                "dynamic_score_differences"
            ].items()
            if values["exactly_equal_runs"] > 0
        ]
        if degenerate:
            raise RuntimeError(
                "E2 smoke produced an identical dynamic/control score array for "
                f"{degenerate}"
            )
    nested_runs = {}
    for (dataset, seed, arm), run in runs.items():
        nested_runs.setdefault(dataset, {}).setdefault(str(seed), {})[arm] = run
    report = {
        "schema_version": "istad-e2-architecture-ablation-v1",
        "status": (
            "SMOKE_ANALYZED_FROM_4_TRAINED_MODELS"
            if args.scope == "smoke"
            else "ANALYZED_FROM_48_TRAINED_MODELS"
        ),
        "scope": args.scope,
        "protocol": {
            "score": "mean per-point reconstruction squared error",
            "primary_metric": "average_precision",
            "secondary_metric": "roc_auc",
            "deployable_threshold": f"normal-train percentile {args.train_percentile}",
            "point_adjustment": False,
            "paired_seeds": list(SEEDS if args.scope == "full" else (87,)),
            "decoder_shape_control": "identical across all arms",
        },
        "expected_runs": [list(key) for key in expected],
        "runs": nested_runs,
        "per_dataset": per_dataset,
        "macro": macro,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = args.output_dir / "per_run"
    per_run_dir.mkdir(exist_ok=True)
    for (dataset, seed, arm), run in runs.items():
        path = per_run_dir / f"{dataset}_s{seed}_{arm}.json"
        path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    summary_path = args.output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": report["status"],
        "expected_runs": len(expected),
        "found_runs": len(runs),
        "sources": sources,
    }, indent=2) + "\n", encoding="utf-8")
    results_path = args.output_dir / "RESULTS.md"
    results_path.write_text(_markdown(report), encoding="utf-8")

    files = sorted(
        path for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "sha256sum.txt"
    )
    hash_path = args.output_dir / "sha256sum.txt"
    hash_path.write_text("".join(
        f"{_sha256(path)}  {path.relative_to(args.output_dir).as_posix()}\n"
        for path in files
    ), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
