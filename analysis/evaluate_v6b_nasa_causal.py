#!/usr/bin/env python3
"""Evaluate the frozen V6b corrected causal NASA diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate_v6_nasa_target import (
    ARMS,
    DATASETS,
    REPO_ROOT,
    _decisions,
    _evaluate,
)
from models.ISTAD import Model


def _find_artifact(score_root: Path, dataset: str, arm: str, seed: int) -> Path:
    objective_tag = "trecon" if arm == "trecon" else "tfcast"
    directories = list(score_root.glob(
        f"anomaly_detection_{dataset}_ISTAD_{dataset}_ftM_sl100_bmtcn_v3_ent_"
        f"{objective_tag}_bs128_v6bnasa_{arm}_s{seed}_0"
    ))
    if len(directories) != 1:
        raise FileNotFoundError(
            f"expected one {dataset}/{arm}/seed-{seed} result directory, got {directories}"
        )
    path = directories[0] / "point_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _parameter_count(dataset: str) -> dict:
    channels = {"MSL": 55, "SMAP": 25}[dataset]
    model = Model(SimpleNamespace(
        task_name="anomaly_detection", seq_len=100, enc_in=channels, c_out=1,
        istad_arch="legacy", istad_branch_mode="tcn", istad_tcn_type="standard",
        istad_spatial_type="legacy", istad_recon_type="pointwise",
        istad_recon_hid_dim=32, istad_dropout=0.2, istad_dual=0, istad_revin=0,
        istad_evidence_head=0, istad_objective="target_forecast",
        istad_target_features="0",
    ))
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad)),
        "decoder": int(sum(parameter.numel()
                           for parameter in model.backbone.recon_model.parameters()
                           if parameter.requires_grad)),
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
        default=REPO_ROOT / "analysis" / "v6b_nasa_causal" / "seed87.json",
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
        "status": "completed_v6b_nasa_seed87_development_diagnostic",
        "confirmatory": False,
        "causality_test": "passed_before_training",
        "preregistration": "analysis/v6b_nasa_causal/preregistered_seed87.md",
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
