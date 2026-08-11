import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import classification_metrics


class ClassificationMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = classification_metrics(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        )
        self.assertEqual(metrics.sample_count, 3)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.macro_f1, 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(np.array([0, 1]), np.array([0]))


if __name__ == "__main__":
    unittest.main()

