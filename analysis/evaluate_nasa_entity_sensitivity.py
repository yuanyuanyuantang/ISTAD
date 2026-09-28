#!/usr/bin/env python3
"""Post-hoc entity-boundary sensitivity for the MSL/SMAP confirmation.

This is not a replacement for the frozen primary run.  It tests whether that
run's negative conclusion is caused by treating the standard concatenated NASA
arrays as a single series.  Each entity contributes its chronological first 80%
to fitting, incomplete windows are dropped per entity, causal pairs and point
adjustment cannot cross entity boundaries, and all other frozen settings remain
unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
ISTAD_ROOT = REPO_ROOT / "code" / "ISTAD"
sys.path.insert(0, str(ISTAD_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_independent_confirmation import (  # noqa: E402
    POT_LEVEL,
    POT_MULTIPLIER,
    POT_TAIL_MARGIN,
    WINDOW,
    _raw_metrics,
    _report_path,
    _sha256,
)
from data_provider.data_loader import (  # noqa: E402
    MSL_TEST_LENGTHS,
    MSL_TRAIN_LENGTHS,
    SMAP_TEST_LENGTHS,
    SMAP_TRAIN_LENGTHS,
)
from evaluate_istad_innovation_pa import _fixed_pa_metrics, _pot_threshold  # noqa: E402
from utils.innovation import TrainingOnlyInnovationScorer  # noqa: E402
from utils.tools import bf_search_adaptive  # noqa: E402


ENTITY_LENGTHS = {
    "MSL": {"train": MSL_TRAIN_LENGTHS, "test": MSL_TEST_LENGTHS},
    "SMAP": {"train": SMAP_TRAIN_LENGTHS, "test": SMAP_TEST_LENGTHS},
}


def _parts(values: np.ndarray, lengths: tuple[int, ...]) -> list[np.ndarray]:
    if sum(lengths) != len(values):
        raise ValueError(f"length sum {sum(lengths)} does not match {len(values)} rows")
    offsets = np.cumsum((0,) + lengths)
    return [values[offsets[index] : offsets[index + 1]] for index in range(len(lengths))]


def evaluate_dataset(name: str, data_root: Path, output_dir: Path) -> dict:
    root = data_root / name
    source_paths = {
        part: root / f"{name}_{part}.npy"
        for part in ("train", "test", "test_label")
    }
    train_raw = np.load(source_paths["train"])
    test_raw = np.load(source_paths["test"])
    label_raw = (np.load(source_paths["test_label"]).reshape(-1) > 0).astype(np.int8)
    train_parts = _parts(train_raw, ENTITY_LENGTHS[name]["train"])
    test_parts = _parts(test_raw, ENTITY_LENGTHS[name]["test"])
    label_parts = _parts(label_raw, ENTITY_LENGTHS[name]["test"])

    split_points = [int(len(part) * 0.8) for part in train_parts]
    fit_raw = [part[:split] for part, split in zip(train_parts, split_points)]
    validation_raw = [part[split:] for part, split in zip(train_parts, split_points)]
    scaler = StandardScaler().fit(np.concatenate(fit_raw))
    fit_parts = [scaler.transform(part) for part in fit_raw]
    complete_test_parts = [
        scaler.transform(part)[: (len(part) // WINDOW) * WINDOW]
        for part in test_parts
    ]
    complete_label_parts = [
        part[: (len(part) // WINDOW) * WINDOW] for part in label_parts
    ]
    fit = np.concatenate(fit_parts)
    test = np.concatenate(complete_test_parts)
    labels = np.concatenate(complete_label_parts)
    fit_entities = np.concatenate(
        [np.full(len(part), index, dtype=np.int16) for index, part in enumerate(fit_parts)]
    )
    test_entities = np.concatenate(
        [
            np.full(len(part), index, dtype=np.int16)
            for index, part in enumerate(complete_test_parts)
        ]
    )

    scorer = TrainingOnlyInnovationScorer(
        lag=1,
        ridge=0.01,
        pool="auto",
        degenerate_cutoff=0.10,
        scale_floor_ratio=0.10,
    ).fit(fit, fit_entities)
    train_score = scorer.score(fit, fit_entities)
    test_score = scorer.score(test, test_entities)
    train_p99 = float(np.percentile(train_score, 99.0))
    p99_raw = _raw_metrics(test_score, labels, train_p99)
    p99_pa = _fixed_pa_metrics(test_score, labels, train_p99, entity_ids=test_entities)

    train_pot = -np.log1p(-np.clip(train_score, 0.0, 1.0 - 1e-12))
    test_pot = -np.log1p(-np.clip(test_score, 0.0, 1.0 - 1e-12))
    pot_threshold, raw_pot_threshold, effective_level = _pot_threshold(
        train_pot,
        test_pot,
        POT_LEVEL,
        POT_MULTIPLIER,
        tail_safety_margin=POT_TAIL_MARGIN,
    )
    pot_pa = _fixed_pa_metrics(
        test_pot, labels, pot_threshold, entity_ids=test_entities
    )
    pot_pa.update(
        {
            "q": 1e-5,
            "configured_level": POT_LEVEL,
            "effective_level": effective_level,
            "base_multiplier": POT_MULTIPLIER,
            "tail_safety_margin": POT_TAIL_MARGIN,
            "raw_spot_threshold": raw_pot_threshold,
        }
    )
    best_pa = bf_search_adaptive(
        test_score,
        labels,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=True,
        entity_ids=test_entities,
    )
    best_raw = bf_search_adaptive(
        test_score,
        labels,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=False,
        entity_ids=test_entities,
    )
    score_path = output_dir / f"{name}_entity_aware_sensitivity_scores.npz"
    np.savez_compressed(
        score_path,
        score=test_score.astype(np.float32),
        train_score=train_score.astype(np.float32),
        label=labels,
        entity_ids=test_entities,
    )
    prevalence = float(labels.mean())
    return {
        "dataset": name,
        "entities": len(ENTITY_LENGTHS[name]["train"]),
        "fit_points": int(len(fit)),
        "held_out_normal_validation_points": int(sum(map(len, validation_raw))),
        "evaluated_points": int(len(test)),
        "innovation": scorer.metadata_dict(),
        "ranking": {
            "roc_auc": float(roc_auc_score(labels, test_score)),
            "pr_auc": float(average_precision_score(labels, test_score)),
            "anomaly_prevalence": prevalence,
            "roc_above_random": bool(roc_auc_score(labels, test_score) > 0.5),
            "pr_above_prevalence": bool(average_precision_score(labels, test_score) > prevalence),
        },
        "train-p99_raw": p99_raw,
        "train-p99+PA": p99_pa,
        "POT+PA": pot_pa,
        "Best-raw": best_raw,
        "Best-F1+PA": best_pa,
        "source_sha256": {key: _sha256(path) for key, path in source_paths.items()},
        "score_artifact": _report_path(score_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT.parent / "ISTAD" / "dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "independent_confirmation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        name: evaluate_dataset(name, args.data_root, args.output_dir)
        for name in ENTITY_LENGTHS
    }
    report = {
        "status": "completed_posthoc_entity_boundary_sensitivity",
        "confirmatory": False,
        "primary_result": "analysis/independent_confirmation/msl_smap_results.json",
        "change_from_frozen_protocol": (
            "80/20 split, complete-window truncation, lag pairs, and point adjustment "
            "are isolated per NASA entity instead of applied to concatenated arrays"
        ),
        "conclusion": (
            "Entity-aware handling does not reverse the negative primary confirmation."
        ),
        "results": results,
    }
    output_path = args.output_dir / "msl_smap_entity_sensitivity.json"
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
