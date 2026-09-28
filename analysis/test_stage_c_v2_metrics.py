#!/usr/bin/env python3
"""Synthetic regression tests for the Stage C v2 metric implementation."""

import unittest

import numpy as np

from stage_c_v2_metrics import (
    aligned_hyperedge_metrics,
    hungarian_maximize,
    partition_scores,
    relation_metrics,
)


class StageCV2MetricTests(unittest.TestCase):
    @staticmethod
    def _incidence(seed=2026):
        rng = np.random.default_rng(seed)
        values = rng.uniform(0.05, 1.0, size=(13, 9, 5))
        return values / values.sum(axis=1, keepdims=True)

    def test_partition_metrics_ignore_label_names(self):
        labels_a = np.array([0, 0, 2, 2, 4, 4, 4, 1])
        labels_b = np.array([3, 3, 0, 0, 2, 2, 2, 4])
        ari, nmi = partition_scores(labels_a, labels_b)
        self.assertAlmostEqual(ari, 1.0, places=12)
        self.assertAlmostEqual(nmi, 1.0, places=12)

    def test_induced_relations_ignore_hyperedge_permutation(self):
        incidence_a = self._incidence()
        permutation = np.array([2, 4, 0, 1, 3])
        incidence_b = incidence_a[:, :, permutation]
        metrics = relation_metrics(incidence_a, incidence_b, neighbor_k=3)
        self.assertAlmostEqual(metrics["top1_partition_ari"]["mean"], 1.0, places=12)
        self.assertAlmostEqual(metrics["top1_partition_nmi"]["mean"], 1.0, places=12)
        self.assertAlmostEqual(metrics["co_membership_cosine"]["mean"], 1.0, places=12)
        self.assertAlmostEqual(
            metrics["co_membership_relative_frobenius"]["max"], 0.0, places=12
        )
        self.assertAlmostEqual(metrics["top3_neighbor_jaccard"]["mean"], 1.0, places=12)

    def test_hungarian_alignment_recovers_permutation(self):
        incidence_a = self._incidence()
        permutation = np.array([2, 4, 0, 1, 3])
        incidence_b = incidence_a[:, :, permutation]
        evidence_a = np.array([0.1, 0.3, 0.2, 0.9, 0.4])
        evidence_b = evidence_a[permutation]
        metrics = aligned_hyperedge_metrics(
            incidence_a, incidence_b, evidence_a, evidence_b
        )
        self.assertAlmostEqual(
            metrics["matched_incidence_column_cosine"]["min"], 1.0, places=12
        )
        self.assertAlmostEqual(
            metrics["aligned_top1_assignment_agreement"]["mean"], 1.0, places=12
        )
        self.assertAlmostEqual(
            metrics["aligned_peak_edge_evidence_cosine"], 1.0, places=12
        )
        self.assertAlmostEqual(
            metrics["aligned_peak_edge_evidence_kendall_tau_b"], 1.0, places=12
        )

    def test_hungarian_finds_known_optimum(self):
        similarity = np.array([
            [0.1, 0.9, 0.2],
            [0.8, 0.2, 0.1],
            [0.2, 0.1, 0.95],
        ])
        self.assertEqual(hungarian_maximize(similarity).tolist(), [1, 0, 2])

    def test_non_permutation_change_is_detected(self):
        incidence_a = self._incidence()
        incidence_b = incidence_a.copy()
        incidence_b[:, [0, 1]] = incidence_b[:, [1, 0]]
        metrics = relation_metrics(incidence_a, incidence_b, neighbor_k=3)
        self.assertLess(metrics["co_membership_cosine"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
