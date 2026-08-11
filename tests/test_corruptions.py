import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corruptions import azimuth_sector_dropout, point_dropout


class PointDropoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.arange(100, dtype=np.float64).reshape(20, 5)

    def test_clean_condition_returns_copy(self) -> None:
        output = point_dropout(self.points, drop_rate=0.0, seed=42)
        np.testing.assert_array_equal(output, self.points)
        self.assertIsNot(output, self.points)

    def test_half_dropout_keeps_expected_count(self) -> None:
        output = point_dropout(self.points, drop_rate=0.5, seed=42)
        self.assertEqual(output.shape, (10, 5))

    def test_same_seed_is_reproducible(self) -> None:
        first = point_dropout(self.points, drop_rate=0.5, seed=7)
        second = point_dropout(self.points, drop_rate=0.5, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_more_severe_corruption_is_nested(self) -> None:
        mild = point_dropout(self.points, drop_rate=0.25, seed=7)
        severe = point_dropout(self.points, drop_rate=0.75, seed=7)
        mild_rows = {tuple(row) for row in mild}
        severe_rows = {tuple(row) for row in severe}
        self.assertTrue(severe_rows <= mild_rows)

    def test_non_complete_dropout_keeps_at_least_one_point(self) -> None:
        one_point = self.points[:1]
        output = point_dropout(one_point, drop_rate=0.9, seed=42)
        self.assertEqual(output.shape, (1, 5))

    def test_complete_dropout_returns_an_empty_point_cloud(self) -> None:
        output = point_dropout(self.points, drop_rate=1.0, seed=42)
        self.assertEqual(output.shape, (0, 5))

    def test_invalid_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            point_dropout(self.points, drop_rate=1.1, seed=42)


class AzimuthSectorDropoutTest(unittest.TestCase):
    def setUp(self) -> None:
        angles = np.linspace(-np.pi, np.pi, 20, endpoint=False)
        self.points = np.column_stack(
            [
                np.cos(angles),
                np.sin(angles),
                np.zeros(20),
                np.ones(20),
                np.arange(20),
            ]
        )

    def test_sector_dropout_keeps_expected_count(self) -> None:
        output = azimuth_sector_dropout(self.points, drop_rate=0.5, seed=42)
        self.assertEqual(output.shape, (10, 5))

    def test_sector_dropout_is_reproducible(self) -> None:
        first = azimuth_sector_dropout(self.points, drop_rate=0.5, seed=7)
        second = azimuth_sector_dropout(self.points, drop_rate=0.5, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_sector_dropout_is_nested(self) -> None:
        mild = azimuth_sector_dropout(self.points, drop_rate=0.25, seed=7)
        severe = azimuth_sector_dropout(self.points, drop_rate=0.75, seed=7)
        mild_rows = {tuple(row) for row in mild}
        severe_rows = {tuple(row) for row in severe}
        self.assertTrue(severe_rows <= mild_rows)

    def test_sector_dropout_requires_planar_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            azimuth_sector_dropout(self.points[:, :1], drop_rate=0.5, seed=7)


if __name__ == "__main__":
    unittest.main()
