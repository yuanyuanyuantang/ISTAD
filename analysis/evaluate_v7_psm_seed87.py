#!/usr/bin/env python3
"""Evaluate the frozen V7 PSM four-arm seed-87 diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_v6_nasa_target import _metrics, _sha256  # noqa: E402
from models.ISTAD import Model  # noqa: E402


ARMS = ("full", "no_prior", "no_relation_score", "temporal_only")
EXPECTED_POINTS = 87808


def _find(score_root: Path, arm: str, seed: int) -> Path:
    pattern = (
        "anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmtcn_archv7_tfcast_"
        f"bs128_v7_{arm}_s{seed}_0"
    )
    directories = list(score_root.glob(pattern))
    if len(directories) != 1:
        raise FileNotFoundError(f"expected one PSM/{arm}/seed-{seed}: {directories}")
    artifact = directories[0] / "point_scores_v7_fused.npz"
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    return artifact


def _evaluate(path: Path) -> dict:
    with np.load(path) as artifact:
        required = {
            "score", "train_score", "label", "v7_components",
            "train_v7_components", "v7_calibrated_components",
            "train_v7_calibrated_components", "v7_metadata_json",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        score = artifact["score"]
        train_score = artifact["train_score"]
        label = artifact["label"]
        components = artifact["v7_components"]
        train_components = artifact["train_v7_components"]
        calibrated = artifact["v7_calibrated_components"]
        train_calibrated = artifact["train_v7_calibrated_components"]
        entity_ids = (
            np.asarray(artifact["entity_id"]).reshape(-1)
            if "entity_id" in artifact.files
            else np.zeros(len(label), dtype=np.int16)
        )
        arrays = (score, train_score, components, train_components,
                  calibrated, train_calibrated)
        finite = bool(all(np.isfinite(values).all() for values in arrays))
        aligned = bool(
            len(score) == len(label) == len(components) == len(calibrated)
            and len(train_score) == len(train_components) == len(train_calibrated)
            and len(entity_ids) == len(label)
        )
        return {
            "artifact": str(path.relative_to(REPO_ROOT)),
            "artifact_sha256": _sha256(path),
            "evaluated_points": int(len(label)),
            "finite": finite,
            "aligned": aligned,
            "component_shape": list(components.shape),
            "metadata": json.loads(str(artifact["v7_metadata_json"].item())),
            "metrics": _metrics(score, train_score, label, entity_ids),
        }


def _complexity() -> dict:
    common = dict(
        task_name="anomaly_detection", seq_len=64, enc_in=25, c_out=25,
        istad_arch="v7", istad_objective="target_forecast",
        istad_target_features=",".join(str(index) for index in range(25)),
        istad_dual=0, istad_revin=0, istad_evidence_head=0,
        istad_v7_relation_dim=16, istad_v7_prior_topk=-1,
        istad_v7_prior_strength=1.0, istad_v7_prior_floor=0.01,
        istad_v7_use_hypergraph=1, istad_recon_hid_dim=32,
        istad_dropout=0.1,
    )
    full = Model(SimpleNamespace(**common))
    temporal = Model(SimpleNamespace(**{
        **common, "istad_v7_use_hypergraph": 0,
        "istad_v7_prior_strength": 0.0,
    }))
    count = lambda model: int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    ))
    active_temporal = (
        temporal.backbone.temporal,
        temporal.backbone.decoder_in,
        temporal.backbone.decoder_norm,
        temporal.backbone.decoder_out,
    )
    active_temporal_count = int(sum(
        parameter.numel() for module in active_temporal
        for parameter in module.parameters()
    ) + temporal.backbone.target_decoder_embedding.numel())
    return {
        "full_trainable": count(full),
        "temporal_only_active": active_temporal_count,
        "hypergraph_increment": count(full) - active_temporal_count,
    }


def _decisions(results: dict) -> dict:
    metric = lambda arm, name: results[arm]["metrics"][name]
    completeness = all(
        result["evaluated_points"] == EXPECTED_POINTS
        and result["finite"] and result["aligned"]
        and result["component_shape"] == [EXPECTED_POINTS, 2]
        and result["metadata"]["prior_ready"]
        for result in results.values()
    )
    full_floor = (
        metric("full", "roc_auc") > 0.5
        and metric("full", "pr_auc")
            > results["full"]["metrics"]["anomaly_prevalence"]
    )
    temporal_delta = {
        name: float(metric("full", name) - metric("temporal_only", name))
        for name in ("roc_auc", "pr_auc")
    }
    prior_delta = {
        name: float(metric("full", name) - metric("no_prior", name))
        for name in ("roc_auc", "pr_auc")
    }
    relation_delta = {
        name: float(metric("full", name) - metric("no_relation_score", name))
        for name in ("roc_auc", "pr_auc")
    }
    temporal_gate = min(temporal_delta.values()) >= -0.01 and max(
        temporal_delta.values()
    ) >= 0.01
    prior_gate = min(prior_delta.values()) >= -0.01 and max(
        prior_delta.values()
    ) >= 0.005
    relation_guard = min(relation_delta.values()) >= -0.01
    gates = {
        "artifact_completeness": bool(completeness),
        "full_above_random_prevalence_floor": bool(full_floor),
        "hypergraph_mechanism_gate": bool(temporal_gate),
        "causal_prior_mechanism_gate": bool(prior_gate),
        "relation_score_guard": bool(relation_guard),
    }
    return {
        "deltas": {
            "full_minus_temporal_only": temporal_delta,
            "full_minus_no_prior": prior_delta,
            "full_minus_no_relation_score": relation_delta,
        },
        "gates": gates,
        "continue_full_to_seeds_90_98": bool(all(gates.values())),
        "paper_readiness_context": {
            "reference": {"roc_auc": 0.7560, "pr_auc": 0.5303},
            "roc_met": bool(metric("full", "roc_auc") >= 0.7560),
            "pr_met": bool(metric("full", "pr_auc") >= 0.5303),
        },
        "warning": "PSM labels were revealed before V7; this is development only.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path,
        default=REPO_ROOT / "code" / "ISTAD" / "test_results",
    )
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "v7_causal_hypergraph" / "psm_seed87.json",
    )
    args = parser.parse_args()
    results = {
        arm: _evaluate(_find(args.score_root, arm, args.seed)) for arm in ARMS
    }
    report = {
        "status": "completed_v7_psm_seed87_development_diagnostic",
        "confirmatory": False,
        "seed": args.seed,
        "preregistration": "analysis/v7_causal_hypergraph/preregistered_psm_seed87.md",
        "results": results,
        "complexity": _complexity(),
        "decisions": _decisions(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
