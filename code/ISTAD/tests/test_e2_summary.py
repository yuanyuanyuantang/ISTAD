"""Tests for the E2 result discovery and metric summarizer."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RELEASE_ROOT / "analysis"))

import summarize_e2_architecture_ablation as summary


class E2SummaryTests(unittest.TestCase):
    def test_smoke_artifacts_are_discovered_and_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "runtime"
            checkpoint_root = root / "checkpoints"
            runs = {}
            for index, arm in enumerate(summary.ARMS):
                setting = (
                    "anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_"
                    f"v3_hglite_bs128_e2_{arm}_s87_0"
                )
                result_dir = result_root / setting
                checkpoint_dir = checkpoint_root / setting
                result_dir.mkdir(parents=True)
                checkpoint_dir.mkdir(parents=True)
                train_score = np.linspace(0.0, 0.8, 40) + index * 1e-3
                score = np.asarray([0.05, 0.10, 0.20, 0.85, 0.90, 1.00])
                score = score + index * np.linspace(0.0, 0.003, len(score))
                label = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
                np.savez_compressed(
                    result_dir / "point_scores.npz",
                    score=score,
                    train_score=train_score,
                    label=label,
                    entity_id=np.zeros(len(label), dtype=np.int16),
                )
                (checkpoint_dir / "checkpoint.pth").write_bytes(
                    f"checkpoint-{arm}".encode()
                )
                (result_dir / "train_time_summary.json").write_text(
                    json.dumps({"avg_epoch_train_time": 1.0 + index})
                )

                found = summary._find_result(result_root, "PSM", 87, arm)
                self.assertEqual(found, result_dir)
                run = summary._evaluate_one(found, checkpoint_root, 99.0)
                self.assertEqual(run["n_test"], 6)
                self.assertAlmostEqual(run["roc_auc"], 1.0)
                self.assertAlmostEqual(run["average_precision"], 1.0)
                runs[("PSM", 87, arm)] = run

            per_dataset, macro = summary._aggregate(runs, "smoke")
            self.assertEqual(set(per_dataset["PSM"]["arms"]), set(summary.ARMS))
            self.assertEqual(set(macro["arms"]), set(summary.ARMS))
            self.assertEqual(
                macro["dynamic_deltas"]["no_message"]["roc_auc"]["ties"], 1
            )
            self.assertGreater(
                per_dataset["PSM"]["dynamic_score_differences"]["no_message"]
                ["max_abs"]["mean"],
                0.0,
            )

            output_dir = root / "final_smoke"
            argv = [
                "summarize_e2_architecture_ablation.py",
                "--result-root", str(result_root),
                "--checkpoint-root", str(checkpoint_root),
                "--output-dir", str(output_dir),
                "--scope", "smoke",
            ]
            with mock.patch.object(sys, "argv", argv):
                summary.main()
            self.assertTrue((output_dir / "RESULTS.md").is_file())
            self.assertTrue((output_dir / "metrics_summary.json").is_file())
            self.assertTrue((output_dir / "run_manifest.json").is_file())
            self.assertTrue((output_dir / "sha256sum.txt").is_file())


if __name__ == "__main__":
    unittest.main()
