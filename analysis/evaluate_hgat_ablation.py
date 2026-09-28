#!/usr/bin/env python3
"""Evaluate the neural branch of the preregistered HGAT ablation.

The training command uses ``innovation_fused`` so that the paper-facing score
is produced in the same run.  This evaluator deliberately reads
``model_score`` and ``train_model_score`` from the dump: otherwise the 0.001
fusion weight makes the three architecture arms look almost identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))

from utils.tools import adjustment, bf_search_adaptive  # noqa: E402


def evaluate(score_path: Path) -> dict[str, float | str]:
    artifact = np.load(score_path)
    required = {"model_score", "train_model_score", "label"}
    missing = required.difference(artifact.files)
    if missing:
        raise KeyError(f"{score_path} is missing {sorted(missing)}")

    score = artifact["model_score"].reshape(-1)
    train_score = artifact["train_model_score"].reshape(-1)
    label = artifact["label"].reshape(-1).astype(int)
    entity_ids = (
        artifact["entity_id"].reshape(-1) if "entity_id" in artifact.files else None
    )
    threshold = float(np.percentile(train_score, 99.0))
    prediction = (score > threshold).astype(int)

    precision, recall, raw_f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0
    )
    _, adjusted = adjustment(
        label.copy(), prediction.copy(), entity_ids=entity_ids
    )
    _, _, adjusted_f1, _ = precision_recall_fscore_support(
        label, adjusted, average="binary", zero_division=0
    )
    best_pa = bf_search_adaptive(
        score,
        label,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=True,
        entity_ids=entity_ids,
    )
    best_raw = bf_search_adaptive(
        score,
        label,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=False,
    )

    return {
        "score_path": str(score_path),
        "roc_auc": float(roc_auc_score(label, score)),
        "pr_auc": float(average_precision_score(label, score)),
        "train_p99_threshold": threshold,
        "train_p99_raw_precision": float(precision),
        "train_p99_raw_recall": float(recall),
        "train_p99_raw_f1": float(raw_f1),
        "train_p99_point_adjusted_f1": float(adjusted_f1),
        "best_raw_f1": float(best_raw["f1"]),
        "best_point_adjusted_f1": float(best_pa["f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("score_paths", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([evaluate(path) for path in args.score_paths], indent=2))


if __name__ == "__main__":
    main()
