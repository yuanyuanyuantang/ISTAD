#!/usr/bin/env python3
"""Aggregate the 12 frozen-checkpoint E1 incidence audit reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


DATASETS = ("EXA", "PSM", "SMD", "SWAT")
SEEDS = (87, 90, 98)
ARMS = (
    "learned_dynamic",
    "static_train_mean",
    "within_entity_time_shuffled",
    "fixed_node_permuted",
)
METRICS = ("roc_auc", "average_precision", "pot_raw_f1", "event_f1")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(arm, name):
    if name == "pot_raw_f1":
        return float(arm["POT"]["point"]["f1"])
    if name == "event_f1":
        return float(arm["POT"]["event"]["event_f1"])
    return float(arm[name])


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _load_reports(runtime_root):
    reports = {}
    sources = []
    for path in sorted(Path(runtime_root).rglob("e1_incidence_counterfactual.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (report["dataset"], int(report["seed"]))
        if key in reports:
            raise RuntimeError(f"duplicate E1 report for {key}")
        reports[key] = report
        sources.append({
            "dataset": key[0],
            "seed": key[1],
            "path": str(path),
            "sha256": _sha256(path),
        })
    return reports, sources


def _aggregate(reports):
    per_dataset = {}
    for dataset in DATASETS:
        rows = [reports[(dataset, seed)] for seed in SEEDS]
        per_dataset[dataset] = {
            "arms": {
                arm: {
                    metric: _summary([
                        _metric(row["arms"][arm], metric) for row in rows
                    ])
                    for metric in METRICS
                }
                for arm in ARMS
            },
            "learned_percentile_among_random": {
                metric: _summary([
                    row["random_controls"]["learned_percentile_among_random"][metric]
                    for row in rows
                ])
                for metric in METRICS
            },
        }

    all_rows = [reports[(dataset, seed)] for dataset in DATASETS for seed in SEEDS]
    macro = {
        "arms": {
            arm: {
                metric: _summary([
                    _metric(row["arms"][arm], metric) for row in all_rows
                ])
                for metric in METRICS
            }
            for arm in ARMS
        },
        "learned_percentile_among_random": {
            metric: _summary([
                row["random_controls"]["learned_percentile_among_random"][metric]
                for row in all_rows
            ])
            for metric in METRICS
        },
    }
    return per_dataset, macro


def _fmt(item):
    return f'{item["mean"]:.6f} +/- {item["population_std"]:.6f}'


def _markdown(summary):
    lines = [
        "# ISTAD E1 incidence counterfactual audit",
        "",
        "> Material Passport: frozen V4-HG checkpoints; no retraining; test labels",
        "> are used only after every incidence, ECDF, gate, and fusion decision is fixed.",
        "",
        "## Deterministic counterfactuals",
        "",
        "Values are mean +/- population standard deviation over seeds 87, 90, and 98.",
        "",
        "| Dataset | Arm | ROC-AUC | AP | POT raw F1 | Event-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "learned_dynamic": "Learned dynamic incidence",
        "static_train_mean": "Static train-mean incidence",
        "within_entity_time_shuffled": "Within-entity time-shuffled incidence",
        "fixed_node_permuted": "Fixed node-permuted incidence",
    }
    for dataset in DATASETS:
        for arm in ARMS:
            values = summary["per_dataset"][dataset]["arms"][arm]
            lines.append(
                f'| {dataset} | {labels[arm]} | {_fmt(values["roc_auc"])} | '
                f'{_fmt(values["average_precision"])} | {_fmt(values["pot_raw_f1"])} | '
                f'{_fmt(values["event_f1"])} |'
            )
    lines.extend([
        "",
        "## Learned incidence versus 100 degree/sparsity-exact random controls",
        "",
        "| Dataset | ROC percentile | AP percentile | POT-F1 percentile | Event-F1 percentile |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        values = summary["per_dataset"][dataset]["learned_percentile_among_random"]
        lines.append(
            f'| {dataset} | {_fmt(values["roc_auc"])} | '
            f'{_fmt(values["average_precision"])} | {_fmt(values["pot_raw_f1"])} | '
            f'{_fmt(values["event_f1"])} |'
        )
    lines.extend([
        "",
        "## Audit constraints",
        "",
        "- Every counterfactual keeps feature evidence, the learned normal-train ECDF",
        "  reference, the reliability decision, and rank-safe epsilon fixed.",
        "- Random controls relabel node rows within each entity and preserve every learned",
        "  incidence matrix's per-time degree profile, sparsity, weights, and column mass.",
        "- Best-F1 and point adjustment are not used in this audit.",
        "- Interpret empirical percentiles as randomization diagnostics, not independent",
        "  confirmatory p-values, because the four benchmark test labels are already revealed.",
        "",
    ])
    return "\n".join(lines)


def main():
    release_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root", type=Path,
        default=release_root / "analysis" / "e1_incidence_counterfactual" / "runtime",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=release_root / "analysis" / "e1_incidence_counterfactual" / "final",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    reports, sources = _load_reports(args.runtime_root)
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    found = set(reports)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if (missing or extra) and not args.allow_incomplete:
        raise RuntimeError(f"expected 12 formal reports; missing={missing}, extra={extra}")
    if missing:
        raise RuntimeError("incomplete reports cannot be aggregated into the paper summary")

    per_dataset, macro = _aggregate(reports)
    summary = {
        "schema_version": "istad-e1-incidence-counterfactual-summary-v1",
        "status": "ANALYZED_FROM_12_FROZEN_CHECKPOINTS",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "per_dataset": per_dataset,
        "macro_over_12_runs": macro,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = args.output_dir / "per_run"
    per_run_dir.mkdir(exist_ok=True)
    for source in sources:
        source_path = Path(source["path"])
        target = per_run_dir / f'{source["dataset"]}_s{source["seed"]}.json'
        shutil.copy2(source_path, target)

    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps({
            "status": summary["status"],
            "expected_runs": 12,
            "found_runs": len(reports),
            "sources": sources,
        }, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    hash_lines = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256sum.txt":
            hash_lines.append(f"{_sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "sha256sum.txt").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
