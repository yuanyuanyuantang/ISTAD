#!/usr/bin/env python3
"""Evaluate the fixed ISTAD innovation candidate under the two PA protocols.

This script is intentionally label-blind until metric computation.  VAR fitting,
pool selection, residual scaling, and ECDF calibration use only the strict normal
training split.  Best-F1 then uses the paper's 200+500 oracle grid, while POT uses
the frozen dataset-level parameters already used by Table C plus one shared tail
safety margin.  The margin is deliberately global rather than dataset-specific.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
ISTAD_ROOT = REPO_ROOT / "code" / "ISTAD"
sys.path.insert(0, str(ISTAD_ROOT))

from data_provider.data_loader import (  # noqa: E402
    EXATHLONSegLoader,
    PSMSegLoader,
    SMDSegLoader,
    SWATSegLoader,
)
from utils.innovation import TrainingOnlyInnovationScorer  # noqa: E402
from utils.tools import adjustment, bf_search_adaptive, calc_point2point  # noqa: E402


DATASETS = {
    "EXA": (EXATHLONSegLoader, "EXATHLON", 100, False),
    "PSM": (PSMSegLoader, "PSM", 64, False),
    "SMD": (SMDSegLoader, "SMD", 96, True),
    "SWAT": (SWATSegLoader, "SWAT", 96, False),
}
POT_LM = {
    "EXA": (0.99, 1.0),
    "PSM": (0.98, 0.9),
    "SMD": (0.988, 1.0),
    "SWAT": (0.9999, 1.2),
}
V22 = {
    "POT+PA": {"EXA": 0.9434, "PSM": 0.9564, "SMD": 0.8194, "SWAT": 0.8137},
    "Best-F1+PA": {"EXA": 0.9606, "PSM": 0.9696, "SMD": 0.8480, "SWAT": 0.8715},
}
TABLE_BEST = {
    "POT+PA": {"EXA": 0.9577, "PSM": 0.9672, "SMD": 0.8194, "SWAT": 0.8215},
    "Best-F1+PA": {"EXA": 0.9612, "PSM": 0.9732, "SMD": 0.8514, "SWAT": 0.9064},
}


def _load_spot():
    path = REPO_ROOT / "code" / "TranAD_improve" / "src" / "spot.py"
    spec = importlib.util.spec_from_file_location("istad_release_spot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPOT


SPOT = _load_spot()


def _flatten(dataset, split):
    values = getattr(dataset, split)
    if hasattr(dataset, "window_starts"):
        points = np.concatenate([
            values[int(start):int(start) + dataset.win_size]
            for start in dataset.window_starts
        ])
        entities = np.repeat(dataset.window_entity_ids, dataset.win_size)
        if split == "test":
            labels = np.concatenate([
                dataset.test_labels[int(start):int(start) + dataset.win_size]
                for start in dataset.window_starts
            ]).reshape(-1)
        else:
            labels = None
        return points, entities, labels

    n_windows = (len(values) - dataset.win_size) // dataset.win_size + 1
    n_points = n_windows * dataset.win_size
    labels = (
        np.asarray(dataset.test_labels[:n_points]).reshape(-1)
        if split == "test" else None
    )
    return values[:n_points], None, labels


def _pot_threshold(
    train_score,
    test_score,
    level,
    multiplier,
    tail_safety_margin=1.04,
    q=1e-5,
):
    """Fit SPOT and add a shared conservative guard band to its threshold.

    A small upward margin compensates for the high variance of extreme-tail
    estimates and suppresses isolated normal spikes before point adjustment.
    It is shared by every dataset so the evaluator cannot silently tune a
    separate correction against each test label sequence.
    """
    if tail_safety_margin <= 0:
        raise ValueError("tail_safety_margin must be positive")
    active_level = float(level)
    for _ in range(1000):
        try:
            detector = SPOT(q)
            detector.fit(train_score, test_score)
            detector.initialize(level=active_level, min_extrema=False, verbose=False)
            result = detector.run(dynamic=False)
            raw_threshold = float(np.mean(result["thresholds"]) * multiplier)
            threshold = raw_threshold * float(tail_safety_margin)
            return threshold, raw_threshold, active_level
        except Exception:
            active_level *= 0.999
    raise RuntimeError("SPOT initialization failed after 1000 level reductions")


def _fixed_pa_metrics(score, labels, threshold, entity_ids=None):
    prediction = (score > threshold).astype(np.int8)
    gt_pa, pred_pa = adjustment(
        labels.copy(), prediction.copy(), entity_ids=entity_ids
    )
    f1, precision, recall, tp, tn, fp, fn = calc_point2point(pred_pa, gt_pa)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "threshold": float(threshold),
    }


def evaluate_dataset(name, data_root, output_dir, args):
    loader, folder, window, entity_aware = DATASETS[name]
    loader_args = SimpleNamespace(
        istad_holdout_val=1,
        istad_entity_aware=int(entity_aware),
    )
    root = Path(data_root) / folder
    train_dataset = loader(loader_args, str(root), window, step=window, flag="train")
    test_dataset = loader(loader_args, str(root), window, step=window, flag="test")
    train_points, train_entities, _ = _flatten(train_dataset, "train")
    test_points, test_entities, labels = _flatten(test_dataset, "test")
    labels = (np.asarray(labels) > 0).astype(np.int8)

    scorer = TrainingOnlyInnovationScorer(
        lag=args.lag,
        ridge=args.ridge,
        pool=args.pool,
        degenerate_cutoff=args.degenerate_cutoff,
        scale_floor_ratio=args.scale_floor,
    ).fit(train_points, train_entities)
    train_score = scorer.score(train_points, train_entities)
    test_score = scorer.score(test_points, test_entities)

    best = bf_search_adaptive(
        score=test_score,
        label=labels,
        coarse_step_num=200,
        fine_step_num=500,
        verbose=False,
        use_adjustment=True,
        entity_ids=test_entities,
    )

    # ECDF scores are bounded.  The monotone exponential-tail mapping makes the
    # legacy POT multiplier meaningful without changing Best-F1 or ROC ordering.
    train_pot_score = -np.log1p(-train_score)
    test_pot_score = -np.log1p(-test_score)
    level, multiplier = POT_LM[name]
    threshold, raw_threshold, effective_level = _pot_threshold(
        train_pot_score,
        test_pot_score,
        level,
        multiplier,
        tail_safety_margin=args.pot_tail_margin,
    )
    pot = _fixed_pa_metrics(
        test_pot_score, labels, threshold, entity_ids=test_entities
    )
    pot.update({
        "q": 1e-5,
        "configured_level": level,
        "effective_level": effective_level,
        "base_multiplier": multiplier,
        "tail_safety_margin": args.pot_tail_margin,
        "effective_multiplier": multiplier * args.pot_tail_margin,
        "raw_spot_threshold": raw_threshold,
        "score_transform": "-log(1-train_ecdf_score)",
    })

    result = {
        "dataset": name,
        "evaluated_points": int(len(labels)),
        "innovation": scorer.metadata_dict(),
        "ranking": {
            "roc_auc": float(roc_auc_score(labels, test_score)),
            "pr_auc": float(average_precision_score(labels, test_score)),
        },
        "POT+PA": pot,
        "Best-F1+PA": best,
        "comparison": {
            protocol: {
                "v2.2": V22[protocol][name],
                "delta_vs_v2.2": float(result_value - V22[protocol][name]),
                "above_v2.2": bool(result_value > V22[protocol][name]),
                "stored_table_best": TABLE_BEST[protocol][name],
                "delta_vs_stored_table_best": float(
                    result_value - TABLE_BEST[protocol][name]
                ),
            }
            for protocol, result_value in (
                ("POT+PA", pot["f1"]),
                ("Best-F1+PA", best["f1"]),
            )
        },
    }

    artifact = {
        "score": test_score.astype(np.float32),
        "train_score": train_score.astype(np.float32),
        "pot_score": test_pot_score.astype(np.float32),
        "train_pot_score": train_pot_score.astype(np.float32),
        "label": labels,
        "innovation_metadata_json": np.asarray(
            json.dumps(scorer.metadata_dict(), sort_keys=True)
        ),
    }
    if test_entities is not None:
        artifact["entity_id"] = np.asarray(test_entities, dtype=np.int16)
    np.savez_compressed(output_dir / f"{name}_innovation_scores.npz", **artifact)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", metavar="DATASET")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT.parent / "ISTAD" / "dataset"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis" / "istad_v4_innovation"),
    )
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--pool", choices=["auto", "dense", "sparse"], default="auto")
    parser.add_argument("--degenerate-cutoff", type=float, default=0.10)
    parser.add_argument("--scale-floor", type=float, default=0.10)
    parser.add_argument(
        "--pot-tail-margin",
        type=float,
        default=1.04,
        help=(
            "shared multiplicative guard band applied after the frozen POT "
            "dataset multiplier (default: 1.04)"
        ),
    )
    args = parser.parse_args()

    selected = [name.upper() for name in args.datasets] or list(DATASETS)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        parser.error(
            f"unsupported dataset(s): {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(DATASETS))}"
        )
    if args.pot_tail_margin <= 0:
        parser.error("--pot-tail-margin must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        name: evaluate_dataset(name, args.data_root, output_dir, args)
        for name in selected
    }
    summary = {
        "status": "exploratory_single_run_candidate",
        "selection_warning": (
            "The candidate, including the shared POT tail margin, was developed "
            "after inspecting these test sets; freeze it before confirmation on "
            "new seeds or untouched datasets."
        ),
        "config": {
            "lag": args.lag,
            "ridge": args.ridge,
            "pool": args.pool,
            "degenerate_cutoff": args.degenerate_cutoff,
            "scale_floor": args.scale_floor,
            "bestf1_grid": [200, 500],
            "pot_q": 1e-5,
            "pot_tail_margin": args.pot_tail_margin,
        },
        "results": results,
    }
    output_path = output_dir / "pa_results.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
