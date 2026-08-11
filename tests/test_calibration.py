import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibration import (
    apply_vector_scaling,
    calibration_metrics,
    density_degradation_score,
    fit_polynomial_quality_temperature,
    fit_quality_temperature,
    fit_temperature,
    fit_vector_scaling,
    negative_log_likelihood,
    polynomial_quality_temperatures,
    quality_temperatures,
)


class CalibrationTest(unittest.TestCase):
    def test_temperature_scaling_reduces_overconfident_nll(self) -> None:
        logits = np.array(
            [
                [8.0, 0.0],
                [8.0, 0.0],
                [0.0, 8.0],
                [0.0, 8.0],
            ]
        )
        targets = np.array([0, 1, 1, 0])
        temperature = fit_temperature(logits, targets)
        self.assertGreater(temperature, 1.0)
        self.assertLess(
            negative_log_likelihood(logits, targets, temperature),
            negative_log_likelihood(logits, targets),
        )

    def test_quality_temperature_produces_one_temperature_per_sample(self) -> None:
        logits = np.array(
            [
                [3.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
            ]
        )
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.2, 0.8, 1.0])
        parameters = fit_quality_temperature(logits, targets, scores)
        temperatures = quality_temperatures(scores, parameters)
        scalar_temperature = fit_temperature(logits, targets)
        self.assertEqual(temperatures.shape, (4,))
        self.assertTrue(np.all(temperatures > 0.0))
        self.assertLess(
            negative_log_likelihood(logits, targets, temperatures),
            negative_log_likelihood(logits, targets, scalar_temperature),
        )

    def test_metrics_are_perfect_for_correct_confident_logits(self) -> None:
        logits = np.array([[8.0, 0.0], [0.0, 8.0]])
        targets = np.array([0, 1])
        metrics = calibration_metrics(logits, targets)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.macro_f1, 1.0)
        self.assertLess(metrics.nll, 0.001)
        self.assertLess(metrics.brier, 0.001)

    def test_macro_f1_averages_only_over_target_classes(self) -> None:
        logits = np.asarray(
            [
                [0.0, 8.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )
        targets = np.asarray([1, 2])
        metrics = calibration_metrics(logits, targets)
        self.assertAlmostEqual(metrics.macro_f1, 0.5)

    def test_density_score_is_bounded(self) -> None:
        scores = density_degradation_score(np.array([100, 50, 0]), 100.0)
        np.testing.assert_allclose(scores, np.array([0.0, 0.5, 1.0]))

    def test_polynomial_quality_model_contains_linear_model(self) -> None:
        logits = np.array(
            [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]
        )
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.25, 0.75, 1.0])
        linear = fit_quality_temperature(logits, targets, scores)
        quadratic = fit_polynomial_quality_temperature(logits, targets, scores)
        self.assertEqual(quadratic.shape, (3,))
        self.assertLessEqual(
            negative_log_likelihood(
                logits,
                targets,
                polynomial_quality_temperatures(scores, quadratic),
            ),
            negative_log_likelihood(
                logits, targets, quality_temperatures(scores, linear)
            )
            + 1e-8,
        )

    def test_vector_scaling_returns_finite_class_parameters(self) -> None:
        logits = np.array(
            [[5.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
        )
        targets = np.array([0, 1, 1, 2])
        log_scales, biases = fit_vector_scaling(logits, targets)
        transformed = apply_vector_scaling(logits, log_scales, biases)
        self.assertEqual(log_scales.shape, (3,))
        self.assertEqual(biases.shape, (3,))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertLess(
            negative_log_likelihood(transformed, targets),
            negative_log_likelihood(logits, targets),
        )


if __name__ == "__main__":
    unittest.main()
