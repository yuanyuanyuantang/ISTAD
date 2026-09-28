#!/usr/bin/env python3
"""Evaluate external baseline score arrays under the frozen ISTAD protocol.

The input manifest points to NPZ files containing ``score``, ``train_score``
and ``label``.  Scores must increase with anomaly likelihood.  Labels and
entity boundaries are checked against the frozen V4-HG arrays so different
preprocessing cannot silently enter the controlled comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from evaluate_v4_hgat_paper import _arm_metrics  # noqa: E402


VALID_DATASETS = ("EXA", "PSM", "SMD", "SWAT")


def _array_sha256(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalize_dataset(name):
    value = str(name).upper()
    if value == "EXATHLON":
        value = "EXA"
    if value not in VALID_DATASETS:
        raise ValueError(f"Unsupported dataset {name}; expected {VALID_DATASETS}")
    return value


def _reference(dataset):
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    matches = list((REPO_ROOT / "code" / "ISTAD" / "test_results").glob(
        f"anomaly_detection_{long_name}_*_v4hg_s87_0/"
        "point_scores_innovation_hgat_rank_tiebreak.npz"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one V4-HG label reference for {dataset}: {matches}")
    with np.load(matches[0]) as artifact:
        label = artifact["label"].reshape(-1).astype(np.int8)
        entity = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files else None
        )
    return label, entity, matches[0]


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _evaluate_entry(entry, manifest_dir):
    for key in ("model", "dataset", "seed", "artifact"):
        if key not in entry:
            raise KeyError(f"Manifest entry is missing {key}: {entry}")
    dataset = _normalize_dataset(entry["dataset"])
    path = _resolve(manifest_dir, entry["artifact"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as artifact:
        required = {"score", "train_score", "label"}
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        score = artifact["score"].reshape(-1).astype(np.float64)
        train_score = artifact["train_score"].reshape(-1).astype(np.float64)
        label = artifact["label"].reshape(-1).astype(np.int8)
        supplied_entity = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files else None
        )
    if not np.isfinite(score).all() or not np.isfinite(train_score).all():
        raise ValueError(f"Non-finite score in {path}")
    if len(score) != len(label):
        raise ValueError(f"score/label length mismatch in {path}")

    reference_label, reference_entity, reference_path = _reference(dataset)
    if not np.array_equal(label, reference_label):
        raise ValueError(
            f"{path} labels do not exactly match frozen {dataset} evaluated points"
        )
    if supplied_entity is not None and reference_entity is not None:
        if not np.array_equal(supplied_entity, reference_entity):
            raise ValueError(f"{path} entity IDs do not match frozen {dataset} boundaries")
    entity_ids = reference_entity if reference_entity is not None else supplied_entity
    metrics = _arm_metrics(dataset, label, score, train_score, entity_ids)
    return {
        "model": str(entry["model"]),
        "dataset": dataset,
        "seed": int(entry["seed"]),
        "artifact": str(path),
        "score_sha256": _array_sha256(score),
        "train_score_sha256": _array_sha256(train_score),
        "label_sha256": _array_sha256(label),
        "label_reference": str(reference_path.relative_to(REPO_ROOT)),
        "evaluated_points": int(len(label)),
        "metrics": metrics,
    }


def _path_value(entry, path):
    value = entry["metrics"]
    for key in path.split("."):
        value = value[key]
    return float(value)


METRICS = {
    "ROC-AUC": "ranking.roc_auc",
    "AUC-PR": "ranking.pr_auc",
    "POT raw F1": "POT.point.f1",
    "POT event F1": "POT.event.event_f1",
    "Best raw F1": "Best-point.f1",
    "POT+PA": "POT.point_adjusted.f1",
}


def _aggregate(entries):
    groups = {}
    for entry in entries:
        groups.setdefault((entry["model"], entry["dataset"]), []).append(entry)
    result = []
    for (model, dataset), values in sorted(groups.items()):
        item = {
            "model": model,
            "dataset": dataset,
            "seeds": sorted(value["seed"] for value in values),
            "metrics": {},
        }
        for label, path in METRICS.items():
            array = np.asarray([_path_value(value, path) for value in values])
            item["metrics"][label] = {
                "mean": float(array.mean()),
                "population_std": float(array.std()),
            }
        result.append(item)
    return result


def _markdown(report):
    lines = [
        "# Controlled external-baseline evaluation",
        "",
        "> Every label array was checked byte-for-byte against the frozen V4-HG",
        "> evaluated points. Best raw F1 is a test-label oracle; POT uses training",
        "> scores. No point adjustment is used for ROC, AP, raw F1, or Event-F1.",
        "",
        "| Model | Dataset | Seeds | ROC-AUC | AUC-PR | POT raw F1 | POT Event-F1 | Best raw F1 | POT+PA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        metric = row["metrics"]
        fmt = lambda key: (
            f"{metric[key]['mean']:.4f}±{metric[key]['population_std']:.4f}"
        )
        lines.append("| " + " | ".join([
            row["model"], row["dataset"], ",".join(map(str, row["seeds"])),
            fmt("ROC-AUC"), fmt("AUC-PR"), fmt("POT raw F1"),
            fmt("POT event F1"), fmt("Best raw F1"), fmt("POT+PA"),
        ]) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "external_baselines"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    raw_entries = manifest.get("runs")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("manifest must contain a non-empty 'runs' list")
    entries = [
        _evaluate_entry(entry, args.manifest.resolve().parent)
        for entry in raw_entries
    ]
    report = {
        "protocol": "ISTAD frozen score-array protocol v1",
        "manifest": str(args.manifest.resolve()),
        "runs": entries,
        "aggregate": _aggregate(entries),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(report))
    print(args.output_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
