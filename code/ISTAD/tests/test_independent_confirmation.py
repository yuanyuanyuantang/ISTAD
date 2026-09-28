import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from analysis import evaluate_independent_confirmation as confirmation  # noqa: E402


class _SyntheticLoader:
    train_raw = None
    test_raw = None
    labels = None

    def __init__(self, args, root_path, win_size, step, flag):
        del args, root_path, win_size, step, flag
        self.train = self.train_raw[:800]
        self.val = self.train_raw[800:]
        self.test = self.test_raw
        self.test_labels = self.labels


class IndependentConfirmationTests(unittest.TestCase):
    def test_complete_windows_truncates_without_duplication(self):
        values = np.arange(205 * 2).reshape(205, 2)
        actual = confirmation._complete_windows(values, 100)
        self.assertEqual(actual.shape, (200, 2))
        np.testing.assert_array_equal(actual, values[:200])

    def test_synthetic_end_to_end_report_is_finite_and_aligned(self):
        rng = np.random.default_rng(19)
        _SyntheticLoader.train_raw = rng.normal(size=(1000, 2))
        _SyntheticLoader.test_raw = rng.normal(size=(1000, 2))
        _SyntheticLoader.labels = np.zeros(1000, dtype=np.int8)
        _SyntheticLoader.labels[500:600] = 1
        _SyntheticLoader.test_raw[500:600, 0] += 5.0

        original = confirmation.DATASETS.copy()
        try:
            confirmation.DATASETS["SYNTH"] = (_SyntheticLoader, 2)
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "data"
                dataset_root = root / "SYNTH"
                output = Path(temp_dir) / "output"
                dataset_root.mkdir(parents=True)
                output.mkdir()
                for kind, values in (
                    ("train", _SyntheticLoader.train_raw),
                    ("test", _SyntheticLoader.test_raw),
                    ("test_label", _SyntheticLoader.labels),
                ):
                    np.save(dataset_root / f"SYNTH_{kind}.npy", values)

                result = confirmation.evaluate_dataset("SYNTH", root, output)

                self.assertTrue(result["integrity"]["all_scores_finite"])
                self.assertTrue(result["integrity"]["score_label_lengths_match"])
                self.assertEqual(result["evaluated_points"], 1000)
                self.assertGreater(result["ranking"]["roc_auc"], 0.5)
                self.assertTrue((output / "SYNTH_pure_v4_scores.npz").is_file())
        finally:
            confirmation.DATASETS.clear()
            confirmation.DATASETS.update(original)


if __name__ == "__main__":
    unittest.main()
