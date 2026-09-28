"""Inference-only incidence counterfactual audit for frozen V4-HG models."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from utils.event_metrics import event_overlap_metrics


def _sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entity_codes(train_entities, test_entities, n_train, n_test):
    def normalized(values, length):
        if values is None:
            return np.zeros(length, dtype=np.int64)
        array = np.asarray(values).reshape(-1)
        if len(array) != length:
            raise ValueError("entity IDs must align with feature evidence")
        return array

    train = normalized(train_entities, n_train)
    test = normalized(test_entities, n_test)
    values = np.unique(np.concatenate([train, test]))
    lookup = {value.item() if hasattr(value, "item") else value: i
              for i, value in enumerate(values)}

    def encode(array):
        return np.asarray([
            lookup[value.item() if hasattr(value, "item") else value]
            for value in array
        ], dtype=np.int32)

    return train, test, encode(train), encode(test), values


def _rng(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, parts)]))


def _non_identity_permutation(generator, size):
    permutation = generator.permutation(size)
    if size > 1 and np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int32, copy=False)


def _shuffle_target_map(codes, seed, salt):
    mapping = np.empty(len(codes), dtype=np.int64)
    for code in np.unique(codes):
        indices = np.flatnonzero(codes == code)
        targets = indices.copy()
        _rng(seed, salt, int(code)).shuffle(targets)
        if len(indices) > 1 and np.array_equal(indices, targets):
            targets = np.roll(targets, 1)
        mapping[indices] = targets
    return mapping


def _pool(evidence, incidence, pool):
    evidence = np.asarray(evidence, dtype=np.float64)
    incidence = np.asarray(incidence, dtype=np.float64)
    mass = incidence.sum(axis=1, keepdims=True)
    normalized = incidence / np.maximum(mass, np.finfo(np.float64).eps)
    edge_evidence = np.einsum("tnm,tn->tm", normalized, evidence, optimize=True)
    if pool == "dense":
        return edge_evidence.mean(axis=1)
    if pool == "sparse":
        return edge_evidence.max(axis=1)
    raise ValueError("pool must be 'dense' or 'sparse'")


def _static_pool(evidence, entity_codes, static_incidence, pool, chunk_size=65536):
    output = np.empty(len(evidence), dtype=np.float64)
    for start in range(0, len(evidence), chunk_size):
        end = min(len(evidence), start + chunk_size)
        for code in np.unique(entity_codes[start:end]):
            local = np.flatnonzero(entity_codes[start:end] == code)
            points = evidence[start:end][local]
            relation = static_incidence[int(code)]
            edge_evidence = points @ relation
            output[start + local] = (
                edge_evidence.mean(axis=1)
                if pool == "dense" else edge_evidence.max(axis=1)
            )
    return output


def _point_metrics(label, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, prediction, average="binary", zero_division=0
    )
    prediction = np.asarray(prediction, dtype=np.int8)
    label = np.asarray(label, dtype=np.int8)
    false_positive = int(np.sum((prediction == 1) & (label == 0)))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(prediction.sum()),
        "false_positive": false_positive,
        "false_alarms_per_1000_points": 1000.0 * false_positive / len(label),
    }


def _load_pot_helpers():
    release_root = Path(__file__).resolve().parents[3]
    analysis_dir = release_root / "analysis"
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    from evaluate_istad_innovation_pa import POT_LM, _pot_threshold
    return POT_LM, _pot_threshold


def _pot_metrics(dataset, label, score, train_score, entity_ids):
    pot_lm, pot_threshold = _load_pot_helpers()
    cap = 1.0 - np.finfo(np.float32).eps
    train_tail = -np.log1p(-np.clip(train_score, 0.0, cap))
    test_tail = -np.log1p(-np.clip(score, 0.0, cap))
    level, multiplier = pot_lm[dataset]
    threshold, raw_threshold, active_level = pot_threshold(
        train_tail, test_tail, level, multiplier
    )
    prediction = (test_tail > threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "raw_spot_threshold": float(raw_threshold),
        "effective_level": float(active_level),
        "point": _point_metrics(label, prediction),
        "event": event_overlap_metrics(label, prediction, entity_ids),
    }


def _tie_audit(base, refined, label):
    base = np.asarray(base, dtype=np.float64)
    refined = np.asarray(refined, dtype=np.float64)
    order = np.argsort(base, kind="stable")
    sorted_base = base[order]
    sorted_refined = refined[order]
    _, starts, counts = np.unique(
        sorted_base, return_index=True, return_counts=True
    )
    group_min = np.minimum.reduceat(sorted_refined, starts)
    group_max = np.maximum.reduceat(sorted_refined, starts)
    changed_groups = (counts > 1) & (group_max > group_min)
    affected_sorted = np.zeros(len(base), dtype=bool)
    for start, count, changed in zip(starts, counts, changed_groups):
        if changed:
            affected_sorted[start:start + count] = True
    affected = np.zeros(len(base), dtype=bool)
    affected[order] = affected_sorted
    overall_rate = float(np.mean(label))
    affected_rate = float(np.mean(np.asarray(label)[affected])) if affected.any() else None
    enrichment = (
        affected_rate / overall_rate
        if affected_rate is not None and overall_rate > 0.0 else None
    )
    return {
        "distinct_rank_inversions": int(np.sum(group_max[:-1] >= group_min[1:])),
        "changed_tie_groups": int(changed_groups.sum()),
        "affected_samples": int(affected.sum()),
        "affected_fraction": float(affected.mean()),
        "overall_anomaly_rate": overall_rate,
        "affected_anomaly_rate": affected_rate,
        "anomaly_enrichment": enrichment,
        "changed_values": int(np.count_nonzero(refined != base)),
        "max_abs_s_minus_q": float(np.max(np.abs(refined - base))),
    }


def _arm_metrics(dataset, label, score, train_score, entity_ids, base_score):
    return {
        "roc_auc": float(roc_auc_score(label, score)),
        "average_precision": float(average_precision_score(label, score)),
        "POT": _pot_metrics(dataset, label, score, train_score, entity_ids),
        "rank_audit": _tie_audit(base_score, score, label),
    }


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


class E1IncidenceCounterfactualCollector:
    """Collect counterfactual scores while the frozen model is evaluated once."""

    def __init__(
        self,
        train_feature_evidence,
        test_feature_evidence,
        train_entity_ids,
        test_entity_ids,
        pool,
        repeats,
        seed,
        work_dir,
    ):
        self.train_evidence = np.asarray(train_feature_evidence, dtype=np.float64)
        self.test_evidence = np.asarray(test_feature_evidence, dtype=np.float64)
        if self.train_evidence.ndim != 2 or self.test_evidence.ndim != 2:
            raise ValueError("feature evidence must have shape (time, nodes)")
        if self.train_evidence.shape[1] != self.test_evidence.shape[1]:
            raise ValueError("train and test evidence must use the same nodes")
        if pool not in {"dense", "sparse"}:
            raise ValueError("pool must be 'dense' or 'sparse'")
        if int(repeats) < 1:
            raise ValueError("repeats must be positive")

        self.pool = pool
        self.repeats = int(repeats)
        self.seed = int(seed)
        self.n_nodes = int(self.train_evidence.shape[1])
        (self.train_entities, self.test_entities,
         self.train_codes, self.test_codes, self.entity_values) = _entity_codes(
            train_entity_ids, test_entity_ids,
            len(self.train_evidence), len(self.test_evidence),
        )
        self.n_entities = len(self.entity_values)
        self.shuffle_targets = {
            "train": _shuffle_target_map(self.train_codes, self.seed, 101),
            "test": _shuffle_target_map(self.test_codes, self.seed, 102),
        }
        self.fixed_permutations = np.stack([
            _non_identity_permutation(_rng(self.seed, 201, code), self.n_nodes)
            for code in range(self.n_entities)
        ])
        self.random_permutations = np.empty(
            (self.repeats, self.n_entities, self.n_nodes), dtype=np.int32
        )
        for repeat in range(self.repeats):
            for code in range(self.n_entities):
                self.random_permutations[repeat, code] = _non_identity_permutation(
                    _rng(self.seed, 301, repeat, code), self.n_nodes
                )

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cursors = {"train": 0, "test": 0}
        self.time_shuffled = {
            "train": np.empty(len(self.train_evidence), dtype=np.float64),
            "test": np.empty(len(self.test_evidence), dtype=np.float64),
        }
        self.node_permuted = {
            "train": np.empty(len(self.train_evidence), dtype=np.float64),
            "test": np.empty(len(self.test_evidence), dtype=np.float64),
        }
        self.random_paths = {
            split: self.work_dir / f"e1_random_{split}.float64.mmap"
            for split in ("train", "test")
        }
        self.random_scores = {
            "train": np.memmap(
                self.random_paths["train"], mode="w+", dtype=np.float64,
                shape=(self.repeats, len(self.train_evidence)),
            ),
            "test": np.memmap(
                self.random_paths["test"], mode="w+", dtype=np.float64,
                shape=(self.repeats, len(self.test_evidence)),
            ),
        }
        self.static_sum = None
        self.static_count = np.zeros(self.n_entities, dtype=np.int64)
        self.n_hyperedges = None

    def _initialize_relation_shape(self, incidence):
        if incidence.ndim != 3 or incidence.shape[1] != self.n_nodes:
            raise ValueError("incidence must have shape (time, nodes, edges)")
        if self.n_hyperedges is None:
            self.n_hyperedges = int(incidence.shape[2])
            self.static_sum = np.zeros(
                (self.n_entities, self.n_nodes, self.n_hyperedges), dtype=np.float64
            )
        elif incidence.shape[2] != self.n_hyperedges:
            raise ValueError("the number of hyperedges changed during evaluation")

    def consume(self, split, incidence, start):
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        relation = np.asarray(incidence, dtype=np.float64)
        self._initialize_relation_shape(relation)
        start = int(start)
        end = start + len(relation)
        evidence_all = self.train_evidence if split == "train" else self.test_evidence
        codes_all = self.train_codes if split == "train" else self.test_codes
        if start != self.cursors[split] or end > len(evidence_all):
            raise RuntimeError(f"non-sequential {split} incidence batch")
        self.cursors[split] = end

        codes = codes_all[start:end]
        evidence = evidence_all[start:end]
        if split == "train":
            for code in np.unique(codes):
                keep = codes == code
                self.static_sum[int(code)] += relation[keep].sum(axis=0)
                self.static_count[int(code)] += int(keep.sum())

        targets = self.shuffle_targets[split][start:end]
        shuffled_score = _pool(evidence_all[targets], relation, self.pool)
        self.time_shuffled[split][targets] = shuffled_score

        permuted_evidence = np.empty_like(evidence)
        for code in np.unique(codes):
            keep = codes == code
            inverse = np.argsort(self.fixed_permutations[int(code)])
            permuted_evidence[keep] = evidence[keep][:, inverse]
        self.node_permuted[split][start:end] = _pool(
            permuted_evidence, relation, self.pool
        )

        for repeat in range(self.repeats):
            randomized_evidence = np.empty_like(evidence)
            for code in np.unique(codes):
                keep = codes == code
                inverse = np.argsort(self.random_permutations[repeat, int(code)])
                randomized_evidence[keep] = evidence[keep][:, inverse]
            self.random_scores[split][repeat, start:end] = _pool(
                randomized_evidence, relation, self.pool
            )

    def complete_split(self, split):
        expected = len(self.train_evidence if split == "train" else self.test_evidence)
        if self.cursors[split] != expected:
            raise RuntimeError(
                f"{split} incidence consumed {self.cursors[split]} points, expected {expected}"
            )
        self.random_scores[split].flush()

    def _static_scores(self):
        if np.any(self.static_count == 0):
            raise RuntimeError("at least one test entity has no normal-training incidence")
        static = self.static_sum / self.static_count[:, None, None]
        return {
            "train": _static_pool(
                self.train_evidence, self.train_codes, static, self.pool
            ),
            "test": _static_pool(
                self.test_evidence, self.test_codes, static, self.pool
            ),
        }

    @staticmethod
    def _calibrator(train_reference):
        reference = np.sort(np.asarray(train_reference, dtype=np.float64))
        return lambda values: np.searchsorted(
            reference, np.asarray(values, dtype=np.float64), side="right"
        ) / float(len(reference))

    @staticmethod
    def _refine(base, graph, epsilon):
        if epsilon == 0.0:
            return np.asarray(base, dtype=np.float64).copy()
        return (
            np.asarray(base, dtype=np.float64) + epsilon * np.asarray(graph, dtype=np.float64)
        ) / (1.0 + epsilon)

    def evaluate_and_write(
        self,
        dataset,
        setting,
        label,
        train_base_score,
        test_base_score,
        train_learned_graph_raw,
        test_learned_graph_raw,
        expected_train_graph_score,
        expected_test_graph_score,
        expected_train_final_score,
        expected_test_final_score,
        epsilon,
        hgat_metadata,
        checkpoint_path,
        output_path,
    ):
        self.complete_split("train")
        self.complete_split("test")
        label = np.asarray(label, dtype=np.int8).reshape(-1)
        if len(label) != len(self.test_evidence):
            raise ValueError("labels must align with test evidence")
        dataset_key = "EXA" if str(dataset).upper() == "EXATHLON" else str(dataset).upper()
        calibrate = self._calibrator(train_learned_graph_raw)
        learned_graph = {
            "train": calibrate(train_learned_graph_raw),
            "test": calibrate(test_learned_graph_raw),
        }
        epsilon = float(epsilon)
        learned_final = {
            "train": self._refine(train_base_score, learned_graph["train"], epsilon),
            "test": self._refine(test_base_score, learned_graph["test"], epsilon),
        }
        equivalence = {
            "train_graph_max_abs_diff": float(np.max(np.abs(
                learned_graph["train"] - np.asarray(expected_train_graph_score)
            ))),
            "test_graph_max_abs_diff": float(np.max(np.abs(
                learned_graph["test"] - np.asarray(expected_test_graph_score)
            ))),
            "train_final_max_abs_diff": float(np.max(np.abs(
                learned_final["train"] - np.asarray(expected_train_final_score)
            ))),
            "test_final_max_abs_diff": float(np.max(np.abs(
                learned_final["test"] - np.asarray(expected_test_final_score)
            ))),
        }
        if max(equivalence.values()) > 1e-12:
            raise RuntimeError(f"frozen learned-arm reconstruction failed: {equivalence}")

        raw_arms = {
            "learned_dynamic": {
                "train": np.asarray(train_learned_graph_raw, dtype=np.float64),
                "test": np.asarray(test_learned_graph_raw, dtype=np.float64),
            },
            "static_train_mean": self._static_scores(),
            "within_entity_time_shuffled": self.time_shuffled,
            "fixed_node_permuted": self.node_permuted,
        }
        arms = {}
        final_scores = {}
        for name, raw in raw_arms.items():
            graph_train = calibrate(raw["train"])
            graph_test = calibrate(raw["test"])
            final_train = self._refine(train_base_score, graph_train, epsilon)
            final_test = self._refine(test_base_score, graph_test, epsilon)
            final_scores[name] = final_test
            arms[name] = _arm_metrics(
                dataset_key, label, final_test, final_train,
                self.test_entities, test_base_score,
            )

        random_runs = []
        for repeat in range(self.repeats):
            graph_train = calibrate(self.random_scores["train"][repeat])
            graph_test = calibrate(self.random_scores["test"][repeat])
            final_train = self._refine(train_base_score, graph_train, epsilon)
            final_test = self._refine(test_base_score, graph_test, epsilon)
            metrics = _arm_metrics(
                dataset_key, label, final_test, final_train,
                self.test_entities, test_base_score,
            )
            random_runs.append({
                "repeat": repeat,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "pot_raw_f1": metrics["POT"]["point"]["f1"],
                "event_f1": metrics["POT"]["event"]["event_f1"],
                "changed_tie_groups": metrics["rank_audit"]["changed_tie_groups"],
                "affected_samples": metrics["rank_audit"]["affected_samples"],
                "anomaly_enrichment": metrics["rank_audit"]["anomaly_enrichment"],
            })

        metric_names = ("roc_auc", "average_precision", "pot_raw_f1", "event_f1")
        random_summary = {
            metric: _summary([run[metric] for run in random_runs])
            for metric in metric_names
        }
        learned = arms["learned_dynamic"]
        learned_values = {
            "roc_auc": learned["roc_auc"],
            "average_precision": learned["average_precision"],
            "pot_raw_f1": learned["POT"]["point"]["f1"],
            "event_f1": learned["POT"]["event"]["event_f1"],
        }
        learned_percentiles = {
            metric: float(np.mean([
                run[metric] < learned_values[metric] for run in random_runs
            ]))
            for metric in metric_names
        }

        checkpoint_path = Path(checkpoint_path)
        report = {
            "schema_version": "istad-e1-incidence-counterfactual-v1",
            "material_passport": {
                "verification_status": "ANALYZED_FROM_FROZEN_CHECKPOINT",
                "selection": "V4-HG formal setting; no retraining",
                "test_label_use": "metrics only; never incidence, ECDF, gate, or fusion",
                "fixed_components": [
                    "feature evidence", "learned train ECDF reference",
                    "reliability gate", "rank-safe epsilon", "POT protocol",
                ],
            },
            "dataset": dataset_key,
            "seed": self.seed,
            "setting": setting,
            "pool": self.pool,
            "n_train": len(self.train_evidence),
            "n_test": len(self.test_evidence),
            "n_nodes": self.n_nodes,
            "n_hyperedges": self.n_hyperedges,
            "n_entities": self.n_entities,
            "random_repeats": self.repeats,
            "random_control": (
                "independent fixed within-entity node permutations of every learned "
                "dynamic incidence; exact per-time degree, sparsity, weight, and column-mass preservation"
            ),
            "epsilon": epsilon,
            "hgat_metadata": hgat_metadata,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "frozen_equivalence": equivalence,
            "arms": arms,
            "deltas_vs_learned": {
                name: {
                    "roc_auc": values["roc_auc"] - learned["roc_auc"],
                    "average_precision": (
                        values["average_precision"] - learned["average_precision"]
                    ),
                    "pot_raw_f1": (
                        values["POT"]["point"]["f1"]
                        - learned["POT"]["point"]["f1"]
                    ),
                    "event_f1": (
                        values["POT"]["event"]["event_f1"]
                        - learned["POT"]["event"]["event_f1"]
                    ),
                }
                for name, values in arms.items() if name != "learned_dynamic"
            },
            "random_controls": {
                "summary": random_summary,
                "learned_percentile_among_random": learned_percentiles,
                "runs": random_runs,
            },
        }
        output_path = Path(output_path)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        self.close(remove_temporary=True)
        return report

    def close(self, remove_temporary=False):
        for split in ("train", "test"):
            mmap = self.random_scores.get(split)
            if mmap is not None:
                mmap.flush()
                del mmap
                self.random_scores[split] = None
        if remove_temporary:
            for path in self.random_paths.values():
                Path(path).unlink(missing_ok=True)

