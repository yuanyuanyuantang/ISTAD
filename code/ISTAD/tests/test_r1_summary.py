"""Tests for R1 result validation and its predeclared smoke gate."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RELEASE_ROOT / "analysis"))

import summarize_r1_relation_bottleneck as summary


class R1SummaryTests(unittest.TestCase):
    def test_smoke_summary_passes_complete_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "runtime"
            checkpoint_root = root / "checkpoints"
            for index, arm in enumerate(summary.ARMS):
                setting = (
                    "anomaly_detection_PSM_ISTAD_PSM_ftM_sl64_bmhgat_"
                    f"hglite_rb_bs128_r1_{arm}_s87_0"
                )
                result_dir = result_root / setting
                checkpoint_dir = checkpoint_root / setting
                result_dir.mkdir(parents=True)
                checkpoint_dir.mkdir(parents=True)
                score = np.asarray([0.05, 0.10, 0.20, 0.80, 0.90, 1.00])
                score = score + index * np.linspace(0.0, 0.004, score.size)
                np.savez_compressed(
                    result_dir / "point_scores.npz",
                    score=score,
                    train_score=np.linspace(0.0, 0.7, 40) + index * 0.001,
                    label=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8),
                )
                validation_mse = 0.90 if arm == "learned_dynamic" else 1.00
                diagnostic = {
                    "protocol": "masked_channel_relation_bottleneck_v1",
                    "relation_mode": arm,
                    "masked_validation_mse": validation_mse,
                    "frozen_no_message_masked_validation_mse": 1.0,
                    "relative_loss_increase_without_message": (
                        0.10 if arm == "learned_dynamic" else 0.0
                    ),
                    "hgat_output_variance_mean": (
                        0.02 if arm == "learned_dynamic" else 0.0
                    ),
                    "incidence_entropy_normalized_mean": 0.5,
                    "message_gate_mean": 0.25,
                    "training": {
                        "mean_epoch_hgat_gradient_norm": (
                            0.03 if arm == "learned_dynamic" else 0.0
                        )
                    },
                }
                (result_dir / "relation_bottleneck_diagnostics.json").write_text(
                    json.dumps(diagnostic)
                )
                (checkpoint_dir / "checkpoint.pth").write_bytes(
                    f"checkpoint-{arm}".encode()
                )

            output_dir = root / "final_smoke"
            argv = [
                "summarize_r1_relation_bottleneck.py",
                "--result-root", str(result_root),
                "--checkpoint-root", str(checkpoint_root),
                "--output-dir", str(output_dir),
                "--scope", "smoke",
            ]
            with mock.patch.object(sys, "argv", argv):
                summary.main()

            report = json.loads((output_dir / "metrics_summary.json").read_text())
            self.assertEqual(report["decision"]["status"], "pass")
            self.assertGreater(
                report["decision"]["mean_relative_masked_mse_improvement"],
                0.02,
            )
            self.assertTrue((output_dir / "RESULTS.md").is_file())
            self.assertTrue((output_dir / "run_manifest.json").is_file())
            self.assertTrue((output_dir / "sha256sum.txt").is_file())


if __name__ == "__main__":
    unittest.main()
