import tempfile
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from utils.e1_incidence_audit import (
    E1IncidenceCounterfactualCollector,
    _pool,
    _tie_audit,
)


class E1IncidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.train_evidence = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 5.0, 1.0],
            [4.0, 2.0, 2.0],
        ])
        self.test_evidence = self.train_evidence + 0.5
        self.entities = np.array([0, 0, 1, 1])
        self.incidence = np.array([
            [[0.7, 0.0], [0.3, 0.4], [0.0, 0.6]],
            [[0.6, 0.1], [0.4, 0.0], [0.0, 0.9]],
            [[0.2, 0.8], [0.8, 0.0], [0.0, 0.2]],
            [[0.1, 0.7], [0.9, 0.0], [0.0, 0.3]],
        ])
        self.temp = tempfile.TemporaryDirectory()
        self.collector = E1IncidenceCounterfactualCollector(
            self.train_evidence,
            self.test_evidence,
            self.entities,
            self.entities,
            pool="sparse",
            repeats=3,
            seed=87,
            work_dir=self.temp.name,
        )

    def tearDown(self):
        self.collector.close(remove_temporary=True)
        self.temp.cleanup()

    def test_batches_are_sequential_and_counterfactuals_are_finite(self):
        self.collector.consume("train", self.incidence[:2], 0)
        self.collector.consume("train", self.incidence[2:], 2)
        self.collector.consume("test", self.incidence[:1], 0)
        self.collector.consume("test", self.incidence[1:], 1)
        self.collector.complete_split("train")
        self.collector.complete_split("test")

        static = self.collector._static_scores()
        for split in ("train", "test"):
            self.assertTrue(np.isfinite(static[split]).all())
            self.assertTrue(np.isfinite(self.collector.time_shuffled[split]).all())
            self.assertTrue(np.isfinite(self.collector.node_permuted[split]).all())
            self.assertTrue(np.isfinite(self.collector.random_scores[split]).all())

        for split, codes in (("train", self.collector.train_codes),
                             ("test", self.collector.test_codes)):
            targets = self.collector.shuffle_targets[split]
            np.testing.assert_array_equal(codes, codes[targets])
            np.testing.assert_array_equal(np.sort(targets), np.arange(len(targets)))

    def test_fixed_node_permutation_matches_manual_row_relabeling(self):
        self.collector.consume("train", self.incidence, 0)
        expected = np.empty(len(self.train_evidence))
        for code in np.unique(self.collector.train_codes):
            keep = self.collector.train_codes == code
            permutation = self.collector.fixed_permutations[int(code)]
            permuted_relation = self.incidence[keep][:, permutation, :]
            expected[keep] = _pool(
                self.train_evidence[keep], permuted_relation, "sparse"
            )
        np.testing.assert_allclose(
            self.collector.node_permuted["train"], expected, rtol=0.0, atol=1e-12
        )

    def test_random_controls_are_nonidentity_permutations(self):
        identity = np.arange(self.collector.n_nodes)
        for permutation in self.collector.random_permutations.reshape(
            -1, self.collector.n_nodes
        ):
            np.testing.assert_array_equal(np.sort(permutation), identity)
            self.assertFalse(np.array_equal(permutation, identity))

    def test_tie_audit_only_marks_nonconstant_tied_groups(self):
        base = np.array([0.1, 0.1, 0.2, 0.2, 0.3])
        refined = np.array([0.100, 0.101, 0.200, 0.200, 0.300])
        label = np.array([0, 1, 0, 1, 0])
        audit = _tie_audit(base, refined, label)
        self.assertEqual(audit["changed_tie_groups"], 1)
        self.assertEqual(audit["affected_samples"], 2)
        self.assertAlmostEqual(audit["affected_anomaly_rate"], 0.5)

    def test_end_to_end_report_uses_fixed_learned_ecdf_and_cleans_mmaps(self):
        self.collector.consume("train", self.incidence, 0)
        self.collector.consume("test", self.incidence, 0)
        train_raw = _pool(self.train_evidence, self.incidence, "sparse")
        test_raw = _pool(self.test_evidence, self.incidence, "sparse")
        reference = np.sort(train_raw)
        calibrate = lambda values: np.searchsorted(
            reference, values, side="right"
        ) / float(len(reference))
        train_graph = calibrate(train_raw)
        test_graph = calibrate(test_raw)
        train_base = np.array([0.1, 0.2, 0.3, 0.4])
        test_base = np.array([0.1, 0.2, 0.3, 0.4])
        epsilon = 0.01
        train_final = (train_base + epsilon * train_graph) / (1.0 + epsilon)
        test_final = (test_base + epsilon * test_graph) / (1.0 + epsilon)

        helper = types.ModuleType("evaluate_istad_innovation_pa")
        helper.POT_LM = {"SYNTH": (0.98, 1.0)}
        helper._pot_threshold = lambda train, test, level, multiplier: (
            float(np.quantile(train, 0.75)),
            float(np.quantile(train, 0.75)),
            float(level),
        )
        previous = sys.modules.get("evaluate_istad_innovation_pa")
        sys.modules["evaluate_istad_innovation_pa"] = helper
        try:
            checkpoint = Path(self.temp.name) / "checkpoint.pth"
            checkpoint.write_bytes(b"frozen checkpoint")
            output = Path(self.temp.name) / "e1.json"
            report = self.collector.evaluate_and_write(
                dataset="SYNTH",
                setting="synthetic_s87",
                label=np.array([0, 0, 1, 1]),
                train_base_score=train_base,
                test_base_score=test_base,
                train_learned_graph_raw=train_raw,
                test_learned_graph_raw=test_raw,
                expected_train_graph_score=train_graph,
                expected_test_graph_score=test_graph,
                expected_train_final_score=train_final,
                expected_test_final_score=test_final,
                epsilon=epsilon,
                hgat_metadata={"selected_weight": epsilon},
                checkpoint_path=checkpoint,
                output_path=output,
            )
        finally:
            if previous is None:
                sys.modules.pop("evaluate_istad_innovation_pa", None)
            else:
                sys.modules["evaluate_istad_innovation_pa"] = previous

        self.assertTrue(output.is_file())
        self.assertEqual(report["random_repeats"], 3)
        self.assertEqual(set(report["arms"]), {
            "learned_dynamic", "static_train_mean",
            "within_entity_time_shuffled", "fixed_node_permuted",
        })
        self.assertLessEqual(max(report["frozen_equivalence"].values()), 1e-12)
        for path in self.collector.random_paths.values():
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
