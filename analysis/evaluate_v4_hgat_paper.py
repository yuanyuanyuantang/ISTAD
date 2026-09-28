#!/usr/bin/env python3
"""Generate paper-facing V4-HG metrics from the frozen score artifacts.

The evaluator keeps deployable, oracle, point-adjusted, and event metrics in
separate fields.  It also compares learned HGAT tie refinement with a random
within-rank control; this prevents a generic tie-breaking effect from being
misattributed to the learned graph.
"""

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
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))

from evaluate_istad_innovation_pa import (  # noqa: E402
    POT_LM,
    _fixed_pa_metrics,
    _pot_threshold,
)
from utils.event_metrics import event_overlap_metrics  # noqa: E402


DATASETS = ("EXA", "PSM", "SMD", "SWAT")
SEEDS = (87, 90, 98)
SCORE_FILE = "point_scores_innovation_hgat_rank_tiebreak.npz"
BEST_FILE = "bestf1_threshold_results_innovation_hgat_rank_tiebreak.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_dir(score_root: Path, dataset: str, seed: int) -> Path:
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    matches = list(score_root.glob(
        f"anomaly_detection_{long_name}_*_v4hg_s{seed}_0"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one result for {dataset}/seed-{seed}, found {matches}"
        )
    return matches[0]


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
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
    """Exact test-label oracle over every distinct score threshold."""
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
    # sklearn's PR curve includes values equal to its threshold.  The project
    # convention is score > threshold, so step one float below it.
    threshold = np.nextafter(float(thresholds[index]), -np.inf)
    return {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "threshold": threshold,
        "selection": "exact_test_label_oracle_no_point_adjustment",
    }


def _pot_metrics(dataset, label, score, train_score, entity_ids):
    cap = 1.0 - np.finfo(np.float32).eps
    train_tail = -np.log1p(-np.clip(train_score, 0.0, cap))
    test_tail = -np.log1p(-np.clip(score, 0.0, cap))
    level, multiplier = POT_LM[dataset]
    threshold, raw_threshold, active_level = _pot_threshold(
        train_tail, test_tail, level, multiplier
    )
    prediction = (test_tail > threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "raw_spot_threshold": float(raw_threshold),
        "effective_level": float(active_level),
        "point": _point_metrics(label, prediction),
        "point_adjusted": _fixed_pa_metrics(
            test_tail, label, threshold, entity_ids
        ),
        "event": event_overlap_metrics(label, prediction, entity_ids),
    }


def _arm_metrics(dataset, label, score, train_score, entity_ids):
    return {
        "ranking": {
            "roc_auc": float(roc_auc_score(label, score)),
            "pr_auc": float(average_precision_score(label, score)),
        },
        "POT": _pot_metrics(dataset, label, score, train_score, entity_ids),
        "Best-point": _exact_best_raw(label, score),
    }


def _rank_audit(base, refined):
    order = np.argsort(base, kind="stable")
    sorted_base = base[order]
    sorted_refined = refined[order]
    _, starts, counts = np.unique(
        sorted_base, return_index=True, return_counts=True
    )
    group_min = np.minimum.reduceat(sorted_refined, starts)
    group_max = np.maximum.reduceat(sorted_refined, starts)
    return {
        "distinct_rank_inversions": int(np.sum(group_max[:-1] >= group_min[1:])),
        "changed_tie_groups": int(np.sum(
            (counts > 1) & ((group_max - group_min) > 0.0)
        )),
        "changed_values": int(np.count_nonzero(refined != base)),
        "score_dtype": str(refined.dtype),
    }


def _random_tie_control(label, base, learned, epsilon, repeats, random_seed):
    base_roc = roc_auc_score(label, base)
    base_ap = average_precision_score(label, base)
    learned_roc = roc_auc_score(label, learned)
    learned_ap = average_precision_score(label, learned)
    rng = np.random.default_rng(random_seed)
    values = np.empty((repeats, 2), dtype=np.float64)
    for index in range(repeats):
        random_graph = rng.random(len(base))
        random_refined = (base + epsilon * random_graph) / (1.0 + epsilon)
        values[index] = (
            roc_auc_score(label, random_refined),
            average_precision_score(label, random_refined),
        )
    return {
        "repeats": int(repeats),
        "random_seed": int(random_seed),
        "learned_delta_roc_auc": float(learned_roc - base_roc),
        "learned_delta_pr_auc": float(learned_ap - base_ap),
        "random_delta_roc_auc": _summary(values[:, 0] - base_roc),
        "random_delta_pr_auc": _summary(values[:, 1] - base_ap),
        "learned_roc_percentile_among_random": float(np.mean(values[:, 0] < learned_roc)),
        "learned_pr_percentile_among_random": float(np.mean(values[:, 1] < learned_ap)),
    }


def _evaluate_one(score_root, dataset, seed, random_tie_repeats):
    directory = _result_dir(score_root, dataset, seed)
    score_path = directory / SCORE_FILE
    best_path = directory / BEST_FILE
    if not score_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {directory}")
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
            name: artifact[name].astype(np.float64, copy=False)
            for name in (
                "score", "train_score", "innovation_score",
                "train_innovation_score", "hgat_score", "train_hgat_score",
            )
        }
        label = artifact["label"].reshape(-1).astype(np.int8)
        entity_ids = (
            artifact["entity_id"].reshape(-1)
            if "entity_id" in artifact.files else None
        )
        metadata = json.loads(str(artifact["hgat_metadata_json"].item()))
        stored_dtypes = {name: str(artifact[name].dtype) for name in required if name != "hgat_metadata_json"}

    best = json.loads(best_path.read_text())
    full = _arm_metrics(
        dataset, label, arrays["score"], arrays["train_score"], entity_ids
    )
    full["Best-F1+PA"] = {
        key: best[key] for key in (
            "f1", "precision", "recall", "threshold", "TP", "TN", "FP", "FN"
        )
    }
    base = _arm_metrics(
        dataset, label, arrays["innovation_score"],
        arrays["train_innovation_score"], entity_ids
    )
    # Rank safety makes the V4 and full Best-F1+PA search spaces order-equivalent.
    base["Best-F1+PA"] = {**full["Best-F1+PA"], "source": "rank_equivalent_to_full"}
    graph = _arm_metrics(
        dataset, label, arrays["hgat_score"], arrays["train_hgat_score"], entity_ids
    )
    epsilon = float(metadata["selected_weight"])
    return {
        "result_dir": str(directory.relative_to(REPO_ROOT)),
        "score_sha256": _sha256(score_path),
        "best_sha256": _sha256(best_path),
        "stored_dtypes": stored_dtypes,
        "metadata": metadata,
        "rank_safety": _rank_audit(arrays["innovation_score"], arrays["score"]),
        "arms": {"V4-HG": full, "V4": base, "HGAT-only": graph},
        "random_tie_control": _random_tie_control(
            label, arrays["innovation_score"], arrays["score"], epsilon,
            random_tie_repeats, 20260910 + 1000 * DATASETS.index(dataset) + seed,
        ),
    }


def _get(run, arm, metric):
    value = run["arms"][arm]
    for key in metric.split("."):
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


def _aggregate(runs):
    per_dataset = {}
    for dataset in DATASETS:
        per_dataset[dataset] = {}
        for arm in ("V4-HG", "V4", "HGAT-only"):
            per_dataset[dataset][arm] = {
                label: _summary([
                    _get(runs[dataset][str(seed)], arm, path) for seed in SEEDS
                ])
                for label, path in METRICS.items()
            }
            if arm != "HGAT-only":
                per_dataset[dataset][arm]["Best-F1+PA"] = _summary([
                    _get(runs[dataset][str(seed)], arm, "Best-F1+PA.f1")
                    for seed in SEEDS
                ])

    macro = {}
    for arm in ("V4-HG", "V4", "HGAT-only"):
        macro[arm] = {}
        labels = list(METRICS)
        if arm != "HGAT-only":
            labels.append("Best-F1+PA")
        for label in labels:
            seed_macro = [
                np.mean([
                    per_dataset[dataset][arm][label]["mean"]
                    for dataset in DATASETS
                ])
            ]
            # Dataset-level macro is deterministic after per-dataset seed means;
            # retain dispersion across datasets rather than mislabel it seed std.
            macro[arm][label] = {
                "mean": float(seed_macro[0]),
                "dataset_population_std": float(np.std([
                    per_dataset[dataset][arm][label]["mean"]
                    for dataset in DATASETS
                ])),
            }
    return per_dataset, macro


def _fmt(summary, digits=4):
    return f"{summary['mean']:.{digits}f}±{summary['population_std']:.{digits}f}"


def _markdown(report):
    aggregate = report["aggregate"]
    macro = report["macro"]
    lines = [
        "# V4-HG paper-facing evaluation",
        "",
        "> Generated from frozen score artifacts. POT is train-derived; Best metrics are",
        "> test-label oracle upper bounds. Event-F1 is overlap-based and never applies",
        "> point adjustment. These benchmarks were previously inspected and are not an",
        "> independent confirmation set.",
        "",
        "## Main results (three seeds)",
        "",
        "| Dataset | ROC-AUC | AUC-PR | POT raw F1 | POT event F1 | Best raw F1 | POT+PA | Best-F1+PA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        values = aggregate[dataset]["V4-HG"]
        lines.append("| " + " | ".join([
            dataset,
            _fmt(values["ROC-AUC"]), _fmt(values["AUC-PR"]),
            _fmt(values["POT raw F1"]), _fmt(values["POT event F1"]),
            _fmt(values["Best raw F1"]), _fmt(values["POT+PA"]),
            _fmt(values["Best-F1+PA"]),
        ]) + " |")
    lines.append("| **Macro** | " + " | ".join(
        f"**{macro['V4-HG'][label]['mean']:.4f}**" for label in (
            "ROC-AUC", "AUC-PR", "POT raw F1", "POT event F1",
            "Best raw F1", "POT+PA", "Best-F1+PA"
        )
    ) + " |")

    lines.extend([
        "", "## Component ablation (macro over four datasets)", "",
        "| Arm | ROC-AUC | AUC-PR | POT raw F1 | POT event F1 | Best raw F1 | POT+PA | Best-F1+PA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("V4-HG", "V4", "HGAT-only"):
        values = macro[arm]
        row = [arm] + [f"{values[label]['mean']:.4f}" for label in (
            "ROC-AUC", "AUC-PR", "POT raw F1", "POT event F1", "Best raw F1", "POT+PA"
        )]
        row.append(f"{values['Best-F1+PA']['mean']:.4f}" if arm != "HGAT-only" else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "", "## Learned HGAT versus random rank-safe tie breaking", "",
        "| Dataset | Learned ΔROC | Random ΔROC | Learned ROC percentile | Learned ΔAP | Random ΔAP | Learned AP percentile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        controls = [report["runs"][dataset][str(seed)]["random_tie_control"] for seed in SEEDS]
        learned_roc = np.mean([item["learned_delta_roc_auc"] for item in controls])
        learned_ap = np.mean([item["learned_delta_pr_auc"] for item in controls])
        random_roc = np.mean([item["random_delta_roc_auc"]["mean"] for item in controls])
        random_ap = np.mean([item["random_delta_pr_auc"]["mean"] for item in controls])
        roc_pct = np.mean([item["learned_roc_percentile_among_random"] for item in controls])
        ap_pct = np.mean([item["learned_pr_percentile_among_random"] for item in controls])
        lines.append(
            f"| {dataset} | {learned_roc:+.8f} | {random_roc:+.8f} | "
            f"{roc_pct:.3f} | {learned_ap:+.8f} | {random_ap:+.8f} | {ap_pct:.3f} |"
        )

    decision = report["decision"]
    lines.extend([
        "", "## Audit decisions", "",
        f"- Float64 statistical artifacts: **{decision['float64_scores_pass']}**",
        f"- Zero distinct-rank inversions in all runs: **{decision['rank_safety_pass']}**",
        f"- Non-empty learned tie refinement in all runs: **{decision['non_vacuous_pass']}**",
        "- V4-HG and V4 PA-F1 equality is expected by construction; it is not an HGAT accuracy gain.",
        "- The random-control percentiles determine whether learned tie refinement is stronger",
        "  than generic tie breaking and must be reported rather than hidden.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results"
    )
    parser.add_argument("--random-tie-repeats", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "analysis" / "v4_hgat_paper"
    )
    args = parser.parse_args()
    if args.random_tie_repeats < 1:
        parser.error("--random-tie-repeats must be positive")

    runs = {
        dataset: {
            str(seed): _evaluate_one(
                args.score_root, dataset, seed, args.random_tie_repeats
            ) for seed in SEEDS
        } for dataset in DATASETS
    }
    aggregate, macro = _aggregate(runs)
    all_runs = [runs[dataset][str(seed)] for dataset in DATASETS for seed in SEEDS]
    decision = {
        "float64_scores_pass": all(
            run["stored_dtypes"]["score"] == "float64" for run in all_runs
        ),
        "rank_safety_pass": all(
            run["rank_safety"]["distinct_rank_inversions"] == 0 for run in all_runs
        ),
        "non_vacuous_pass": all(
            run["rank_safety"]["changed_tie_groups"] > 0 for run in all_runs
        ),
    }
    report = {
        "status": "frozen_development_benchmarks_not_independent_confirmation",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "metric_protocol": {
            "POT": "training-score-derived threshold",
            "Best-point": "exact test-label oracle without point adjustment",
            "Best-F1+PA": "adaptive test-label oracle with point adjustment",
            "event": "overlap event precision/recall; entity-safe; no point adjustment",
            "random_tie_control_repeats": args.random_tie_repeats,
        },
        "runs": runs,
        "aggregate": aggregate,
        "macro": macro,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "paper_metrics.json"
    md_path = args.output_dir / "RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))
    print(md_path)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
