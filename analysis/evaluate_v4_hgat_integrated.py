#!/usr/bin/env python3
"""Audit the frozen three-seed V4-HG rank-safe confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from evaluate_istad_innovation_pa import (  # noqa: E402
    POT_LM,
    _fixed_pa_metrics,
    _pot_threshold,
)


DATASETS = ("EXA", "PSM", "SMD", "SWAT")
SEEDS = (87, 90, 98)
FORMAL_V4 = {
    "POT+PA": {
        "EXA": 0.9593254934529327,
        "PSM": 0.9697128396590672,
        "SMD": 0.8477799567574127,
        "SWAT": 0.8322632717841407,
    },
    "Best-F1+PA": {
        "EXA": 0.9627991102045644,
        "PSM": 0.9822430345933931,
        "SMD": 0.8829375784876803,
        "SWAT": 0.9524487504947322,
    },
}
PARAMETERS = {
    "EXA": {"neural": 4771, "hgat": 274},
    "PSM": {"neural": 7303, "hgat": 346},
    "SMD": {"neural": 14522, "hgat": 506},
    "SWAT": {"neural": 44867, "hgat": 618},
}
SCORE_FILE = "point_scores_innovation_hgat_rank_tiebreak.npz"
BEST_FILE = "bestf1_threshold_results_innovation_hgat_rank_tiebreak.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_dir(dataset: str, seed: int) -> Path:
    long_name = "EXATHLON" if dataset == "EXA" else dataset
    matches = list((REPO_ROOT / "code" / "ISTAD" / "test_results").glob(
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


def _rank_audit(base, refined):
    order = np.argsort(base, kind="stable")
    sorted_base = base[order]
    sorted_refined = refined[order]
    _, starts, counts = np.unique(
        sorted_base, return_index=True, return_counts=True
    )
    group_min = np.minimum.reduceat(sorted_refined, starts)
    group_max = np.maximum.reduceat(sorted_refined, starts)
    distinct_rank_inversions = int(np.sum(group_max[:-1] >= group_min[1:]))
    tie_spans = group_max - group_min
    changed_tie_groups = int(np.sum((counts > 1) & (tie_spans > 0.0)))
    changed_values = int(np.count_nonzero(refined != base))
    return {
        "distinct_rank_inversions": distinct_rank_inversions,
        "changed_tie_groups": changed_tie_groups,
        "changed_values": changed_values,
        "max_absolute_correction": float(np.max(np.abs(refined - base))),
    }


def _evaluate_one(dataset: str, seed: int):
    directory = _result_dir(dataset, seed)
    score_path = directory / SCORE_FILE
    best_path = directory / BEST_FILE
    if not score_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {directory}")

    best = json.loads(best_path.read_text())
    with np.load(score_path) as artifact:
        score = artifact["score"].astype(np.float64)
        train_score = artifact["train_score"].astype(np.float64)
        base = artifact["innovation_score"].astype(np.float64)
        graph = artifact["hgat_score"].astype(np.float64)
        labels = artifact["label"].astype(np.int8)
        entities = (
            artifact["entity_id"].astype(np.int64)
            if "entity_id" in artifact.files else None
        )
        metadata = json.loads(str(artifact["hgat_metadata_json"].item()))

    cap = 1.0 - np.finfo(np.float32).eps
    train_tail = -np.log1p(-np.clip(train_score, 0.0, cap))
    test_tail = -np.log1p(-np.clip(score, 0.0, cap))
    level, multiplier = POT_LM[dataset]
    threshold, raw_threshold, active_level = _pot_threshold(
        train_tail, test_tail, level, multiplier
    )
    pot = _fixed_pa_metrics(test_tail, labels, threshold, entities)
    pot.update({
        "raw_spot_threshold": raw_threshold,
        "effective_level": active_level,
        "score_transform": "-log(1-score)",
    })

    return {
        "result_dir": str(directory.relative_to(REPO_ROOT)),
        "score_sha256": _sha256(score_path),
        "best_sha256": _sha256(best_path),
        "parameters": PARAMETERS[dataset],
        "selected_weight": float(metadata["selected_weight"]),
        "message_gate": float(metadata["message_gate"]),
        "kill_switch": bool(metadata["reliability_gate"]["kill_switch"]),
        "ranking": {
            "fused_roc_auc": float(roc_auc_score(labels, score)),
            "fused_pr_auc": float(average_precision_score(labels, score)),
            "v4_roc_auc": float(roc_auc_score(labels, base)),
            "v4_pr_auc": float(average_precision_score(labels, base)),
            "hgat_roc_auc": float(roc_auc_score(labels, graph)),
            "hgat_pr_auc": float(average_precision_score(labels, graph)),
        },
        "rank_safety": _rank_audit(base, score),
        "POT+PA": pot,
        "Best-F1+PA": {
            key: best[key]
            for key in (
                "f1", "precision", "recall", "TP", "TN", "FP", "FN",
                "threshold", "search_method", "coarse_step_num", "fine_step_num",
            )
        },
    }


def main():
    runs = {
        dataset: {
            str(seed): _evaluate_one(dataset, seed) for seed in SEEDS
        }
        for dataset in DATASETS
    }
    aggregate = {}
    for dataset in DATASETS:
        aggregate[dataset] = {
            protocol: _summary([
                runs[dataset][str(seed)][protocol]["f1"] for seed in SEEDS
            ])
            for protocol in ("POT+PA", "Best-F1+PA")
        }
    macro = {
        protocol: _summary([
            np.mean([
                runs[dataset][str(seed)][protocol]["f1"]
                for dataset in DATASETS
            ])
            for seed in SEEDS
        ])
        for protocol in ("POT+PA", "Best-F1+PA")
    }
    formal_macro = {
        protocol: float(np.mean(list(FORMAL_V4[protocol].values())))
        for protocol in FORMAL_V4
    }
    invariant_pass = all(
        runs[dataset][str(seed)]["rank_safety"]["distinct_rank_inversions"] == 0
        for dataset in DATASETS for seed in SEEDS
    )
    non_vacuous = any(
        runs[dataset][str(seed)]["rank_safety"]["changed_tie_groups"] > 0
        for dataset in DATASETS for seed in SEEDS
    )
    decision = {
        "pot_macro_pass": bool(
            macro["POT+PA"]["mean"] >= formal_macro["POT+PA"] - 1e-4
        ),
        "best_macro_pass": bool(
            macro["Best-F1+PA"]["mean"]
            >= formal_macro["Best-F1+PA"] - 1e-4
        ),
        "rank_safety_pass": invariant_pass,
        "non_vacuous_tie_refinement": non_vacuous,
    }
    decision["overall_pass"] = bool(all(decision.values()))

    report = {
        "status": "development_seed_stability_not_independent_confirmation",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "formal_v4": FORMAL_V4,
        "formal_v4_macro": formal_macro,
        "runs": runs,
        "aggregate": aggregate,
        "macro": macro,
        "decision": decision,
    }
    output = REPO_ROOT / "analysis" / "v4_hgat_integrated" / "multiseed_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("dataset       POT+PA mean±std       Best-F1+PA mean±std")
    for dataset in DATASETS:
        pot = aggregate[dataset]["POT+PA"]
        best = aggregate[dataset]["Best-F1+PA"]
        print(
            f"{dataset:<8} {pot['mean']:.6f}±{pot['population_std']:.6f}   "
            f"{best['mean']:.6f}±{best['population_std']:.6f}"
        )
    print(
        f"Macro    {macro['POT+PA']['mean']:.6f}±{macro['POT+PA']['population_std']:.6f}   "
        f"{macro['Best-F1+PA']['mean']:.6f}±{macro['Best-F1+PA']['population_std']:.6f}"
    )
    print(json.dumps(decision, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
