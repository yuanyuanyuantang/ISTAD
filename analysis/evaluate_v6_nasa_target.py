#!/usr/bin/env python3
"""Evaluate the frozen seed-87 target-aware NASA development diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))

from utils.tools import adjustment, bf_search_adaptive  # noqa: E402
from models.ISTAD import Model  # noqa: E402


DATASETS = ("MSL", "SMAP")
ARMS = ("trecon", "forecast")
V5_REFERENCE = {
    "MSL": {"roc_auc": 0.6216457678578993, "pr_auc": 0.1544116865292471},
    "SMAP": {"roc_auc": 0.4815785219037676, "pr_auc": 0.12019568095622207},
}
TARGETS = {
    "MSL": {"roc_auc": 0.7314, "pr_auc": 0.2186, "pa_f1": 0.9513},
    "SMAP": {"roc_auc": 0.6224, "pr_auc": 0.1912, "pa_f1": 0.9732},
}


def _parameter_count(dataset: str) -> dict:
    channels = {"MSL": 55, "SMAP": 25}[dataset]

    def build(c_out: int, evidence: int, objective: str, targets: str) -> Model:
        return Model(SimpleNamespace(
            task_name="anomaly_detection", seq_len=100, enc_in=channels, c_out=c_out,
            istad_arch="legacy", istad_branch_mode="kan_tcn", istad_tcn_type="kan",
            istad_spatial_type="legacy", istad_recon_type="kanad", istad_kanad_order=4,
            istad_kan_grid_size=10, istad_kan_spline_order=3, istad_dropout=0.2,
            istad_dual=0, istad_revin=0, istad_evidence_head=evidence,
            istad_objective=objective, istad_target_features=targets,
        ))

    v5 = build(channels, 1, "reconstruct", "")
    v6 = build(1, 0, "target_forecast", "0")

    def count(module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()
                       if parameter.requires_grad))

    v5_total, v6_total = count(v5), count(v6)
    v5_decoder = count(v5.backbone.recon_model)
    v6_decoder = count(v6.backbone.recon_model)
    return {
        "v5_none_total": v5_total,
        "v6_total": v6_total,
        "total_reduction_fraction": float((v5_total - v6_total) / v5_total),
        "v5_decoder": v5_decoder,
        "v6_decoder": v6_decoder,
        "decoder_reduction_fraction": float((v5_decoder - v6_decoder) / v5_decoder),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(score, train_score, label, entity_ids) -> dict:
    score = np.asarray(score).reshape(-1)
    train_score = np.asarray(train_score).reshape(-1)
    label = np.asarray(label).reshape(-1).astype(np.int8)
    entity_ids = np.asarray(entity_ids).reshape(-1)
    if not (len(score) == len(label) == len(entity_ids)):
        raise ValueError("score, label, and entity ID lengths do not match")
    if not np.isfinite(score).all() or not np.isfinite(train_score).all():
        raise ValueError("score contains NaN or infinite values")

    threshold = float(np.percentile(train_score, 99.0))
    prediction = (score > threshold).astype(np.int8)
    precision, recall, raw_f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0
    )
    _, adjusted = adjustment(label.copy(), prediction.copy(), entity_ids=entity_ids)
    pa_precision, pa_recall, pa_f1, _ = precision_recall_fscore_support(
        label, adjusted, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(label, score)),
        "pr_auc": float(average_precision_score(label, score)),
        "anomaly_prevalence": float(label.mean()),
        "train_p99": {
            "threshold": threshold,
            "raw_f1": float(raw_f1),
            "raw_precision": float(precision),
            "raw_recall": float(recall),
            "point_adjusted_f1": float(pa_f1),
            "point_adjusted_precision": float(pa_precision),
            "point_adjusted_recall": float(pa_recall),
        },
        "best_raw": bf_search_adaptive(
            score, label, coarse_step_num=200, fine_step_num=500,
            verbose=False, use_adjustment=False,
        ),
        "best_point_adjusted": bf_search_adaptive(
            score, label, coarse_step_num=200, fine_step_num=500,
            verbose=False, use_adjustment=True, entity_ids=entity_ids,
        ),
    }


def _find_artifact(score_root: Path, dataset: str, arm: str, seed: int) -> Path:
    objective_tag = "trecon" if arm == "trecon" else "tfcast"
    directories = list(score_root.glob(
        f"anomaly_detection_{dataset}_ISTAD_{dataset}_ftM_sl100_bmkan_tcn_v3_ent_"
        f"{objective_tag}_bs128_v6nasa_{arm}_s{seed}_0"
    ))
    if len(directories) != 1:
        raise FileNotFoundError(
            f"expected one {dataset}/{arm}/seed-{seed} result directory, got {directories}"
        )
    path = directories[0] / "point_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _evaluate(path: Path) -> dict:
    with np.load(path) as artifact:
        required = {"score", "train_score", "label", "entity_id"}
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        return {
            "artifact": str(path.relative_to(REPO_ROOT)),
            "artifact_sha256": _sha256(path),
            "evaluated_points": int(len(artifact["label"])),
            "entities": int(len(np.unique(artifact["entity_id"]))),
            "target_error": _metrics(
                artifact["score"], artifact["train_score"],
                artifact["label"], artifact["entity_id"],
            ),
        }


def _decisions(results: dict) -> dict:
    forecast_passes = []
    for dataset in DATASETS:
        metrics = results[dataset]["forecast"]["target_error"]
        reference = V5_REFERENCE[dataset]
        forecast_passes.extend([
            metrics["roc_auc"] > reference["roc_auc"],
            metrics["pr_auc"] > reference["pr_auc"],
            metrics["pr_auc"] > metrics["anomaly_prevalence"],
        ])
    forecast_passes.append(
        results["SMAP"]["forecast"]["target_error"]["roc_auc"] > 0.5
    )

    deltas = {}
    flat_deltas = []
    for dataset in DATASETS:
        deltas[dataset] = {}
        for metric in ("roc_auc", "pr_auc"):
            delta = (
                results[dataset]["forecast"]["target_error"][metric]
                - results[dataset]["trecon"]["target_error"][metric]
            )
            deltas[dataset][metric] = float(delta)
            flat_deltas.append(delta)

    literature = {}
    for dataset in DATASETS:
        literature[dataset] = {}
        for arm in ARMS:
            metric = results[dataset][arm]["target_error"]
            literature[dataset][arm] = {
                "roc_target_met": bool(metric["roc_auc"] >= TARGETS[dataset]["roc_auc"]),
                "pr_target_met": bool(metric["pr_auc"] >= TARGETS[dataset]["pr_auc"]),
                "best_PA_target_met": bool(
                    metric["best_point_adjusted"]["f1"] >= TARGETS[dataset]["pa_f1"]
                ),
            }
    return {
        "continue_forecast_to_seeds_90_98": bool(all(forecast_passes)),
        "forecast_deltas_vs_target_reconstruct": deltas,
        "causal_forecast_benefit_gate": bool(
            max(flat_deltas) >= 0.01 and min(flat_deltas) >= -0.01
        ),
        "v5_neural_reference": V5_REFERENCE,
        "literature_context_targets": TARGETS,
        "literature_context_gate": literature,
        "warning": (
            "MSL/SMAP labels were revealed before V6. This is a development "
            "diagnostic and cannot support an independent SOTA claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "v6_nasa_target" / "seed87.json",
    )
    args = parser.parse_args()
    results = {
        dataset: {
            arm: _evaluate(_find_artifact(args.score_root, dataset, arm, args.seed))
            for arm in ARMS
        }
        for dataset in DATASETS
    }
    report = {
        "status": "completed_v6_nasa_seed87_development_diagnostic",
        "confirmatory": False,
        "preregistration": "analysis/v6_nasa_target/preregistered_seed87.md",
        "seed": args.seed,
        "results": results,
        "complexity": {dataset: _parameter_count(dataset) for dataset in DATASETS},
        "decisions": _decisions(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
