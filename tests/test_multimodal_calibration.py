import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibration import negative_log_likelihood
from multimodal_calibration import (
    build_quality_features,
    fit_quality_temperature_model,
    jensen_shannon_disagreement,
)
from analyze_multimodal_calibration import (
    compute_robustness_auc_rows,
    holm_adjust,
    is_primary_degraded_fusion_condition,
    load_condition,
    load_condition_bundle,
    paired_cluster_sign_flip_test,
)
from multimodal_protocol import ExperimentCondition


class MultimodalCalibrationTest(unittest.TestCase):
    @staticmethod
    def _write_condition(path: Path, condition: ExperimentCondition) -> None:
        import json

        sample_count = 4
        logits = np.tile(np.arange(27, dtype=np.float32), (sample_count, 1))
        np.savez_compressed(
            path,
            condition_json=np.asarray(json.dumps(condition.to_dict())),
            logits=logits,
            targets=np.asarray([5, 8, 9, 11], dtype=np.int16),
            sample_ids=np.asarray([f"sample-{index}" for index in range(sample_count)]),
            subjects=np.asarray(["S01", "S02", "S03", "S04"]),
            actions=np.asarray(["A06", "A09", "A10", "A12"]),
            partitions=np.asarray(["calibration", "calibration", "test", "test"]),
            lidar_point_counts=np.asarray([10, 11, 12, 13]),
            mmwave_point_counts=np.zeros(sample_count, dtype=np.int32),
            lidar_azimuth_occupancy=np.full(sample_count, 0.5),
            mmwave_azimuth_occupancy=np.zeros(sample_count),
            lidar_range_occupancy=np.full(sample_count, 0.5),
            mmwave_range_occupancy=np.zeros(sample_count),
        )

    def test_missing_modality_feature_is_explicit(self) -> None:
        import tempfile

        condition = ExperimentCondition("lidar_only", "clean", 0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"{condition.condition_id}.npz"
            self._write_condition(path, condition)
            bundle = load_condition_bundle(path, root, "strict27")
        np.testing.assert_array_equal(bundle["features"][:, 9], 0.0)
        np.testing.assert_array_equal(bundle["features"][:, 10], 1.0)

    def test_constant_performance_has_constant_robustness_auc(self) -> None:
        rows = []
        for geometry, seed, lidar_rate, mmwave_rate in [
            ("clean", 0, 0.0, 0.0),
            ("uniform", 7, 0.25, 0.0),
            ("uniform", 7, 0.5, 0.0),
            ("uniform", 7, 0.75, 0.0),
            ("uniform", 7, 0.9, 0.0),
        ]:
            rows.append(
                {
                    "condition_id": f"condition-{len(rows)}",
                    "mask_name": "lidar_mmwave",
                    "geometry": geometry,
                    "corruption_seed": seed,
                    "lidar_drop_rate": lidar_rate,
                    "mmwave_drop_rate": mmwave_rate,
                    "method": "uncalibrated",
                    "accuracy": 0.8,
                    "macro_f1": 0.7,
                }
            )
        auc_rows = compute_robustness_auc_rows(pd.DataFrame(rows))
        self.assertEqual(len(auc_rows), 1)
        self.assertAlmostEqual(auc_rows[0]["accuracy_robustness_auc"], 0.8)
        self.assertAlmostEqual(auc_rows[0]["macro_f1_robustness_auc"], 0.7)

    def test_condition_metadata_round_trip(self) -> None:
        import json
        import tempfile

        condition = ExperimentCondition("lidar_mmwave", "uniform", 7, 0.5, 0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.npz"
            np.savez_compressed(
                path,
                condition_json=np.asarray(json.dumps(condition.to_dict())),
                logits=np.zeros((2, 3)),
            )
            loaded, arrays = load_condition(path)
        self.assertEqual(loaded, condition)
        self.assertEqual(arrays["logits"].shape, (2, 3))

    def test_primary_comparison_excludes_clean_and_unimodal_conditions(self) -> None:
        self.assertTrue(
            is_primary_degraded_fusion_condition(
                ExperimentCondition("lidar_mmwave", "uniform", 7, 0.5, 0.25)
            )
        )
        self.assertTrue(
            is_primary_degraded_fusion_condition(
                ExperimentCondition(
                    "lidar_mmwave", "azimuth_sector", 7, 0.5, 0.25
                )
            )
        )
        self.assertFalse(
            is_primary_degraded_fusion_condition(
                ExperimentCondition("lidar_mmwave", "clean", 0, 0.0, 0.0)
            )
        )
        self.assertFalse(
            is_primary_degraded_fusion_condition(
                ExperimentCondition("lidar_only", "uniform", 7, 0.5, 0.0)
            )
        )

    def test_holm_adjustment_is_monotonic_in_rank_order(self) -> None:
        raw = [0.04, 0.01, 0.03]
        adjusted = holm_adjust(raw)
        ordered = sorted(zip(raw, adjusted))
        self.assertLessEqual(ordered[0][1], ordered[1][1])
        self.assertLessEqual(ordered[1][1], ordered[2][1])
        np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])

    def test_paired_sign_flip_is_reproducible_and_detects_large_delta(self) -> None:
        sums = {f"cluster-{index}": -10.0 for index in range(12)}
        counts = {key: 10 for key in sums}
        first = paired_cluster_sign_flip_test(sums, counts, 20_000, seed=7)
        second = paired_cluster_sign_flip_test(sums, counts, 20_000, seed=7)
        self.assertEqual(first, second)
        self.assertLess(first["sign_flip_p"], 0.01)

    def test_js_disagreement_is_zero_for_equal_logits(self) -> None:
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_allclose(
            jensen_shannon_disagreement(logits, logits), 0.0, atol=1e-12
        )

    def test_js_disagreement_stays_finite_for_extreme_logits(self) -> None:
        first = np.array([[1000.0, -1000.0, -1000.0]])
        second = np.array([[-1000.0, 1000.0, -1000.0]])

        disagreement = jensen_shannon_disagreement(first, second)

        self.assertTrue(np.isfinite(disagreement).all())
        np.testing.assert_allclose(disagreement, 1.0, atol=1e-12)

    def test_quality_feature_shape(self) -> None:
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        values = np.array([10.0, 5.0, 1.0])
        occupancy = np.array([1.0, 0.5, 0.1])
        features = build_quality_features(
            values,
            values,
            occupancy,
            occupancy,
            occupancy,
            occupancy,
            logits,
            logits,
        )
        self.assertEqual(features.shape, (3, 11))
        self.assertTrue(np.isfinite(features).all())

    def test_fitted_temperatures_are_positive_and_preserve_argmax(self) -> None:
        logits = np.array(
            [
                [5.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 5.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ]
        )
        targets = np.array([0, 1, 1, 1, 0, 0])
        quality = np.linspace(0.0, 1.0, logits.shape[0])[:, None]
        model = fit_quality_temperature_model(
            logits, targets, quality, regularization=1e-3
        )
        temperatures = model.temperatures(quality)
        self.assertTrue(np.all(temperatures > 0.0))
        np.testing.assert_array_equal(
            logits.argmax(axis=1), (logits / temperatures[:, None]).argmax(axis=1)
        )
        self.assertLessEqual(
            negative_log_likelihood(logits, targets, temperatures),
            negative_log_likelihood(logits, targets) + 1e-8,
        )


if __name__ == "__main__":
    unittest.main()
