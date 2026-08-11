"""Observable-quality-aware scalar calibration for multimodal X-Fi logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

try:
    from .calibration import (
        fit_temperature,
        negative_log_likelihood,
        probabilities_from_logits,
        validate_logits_targets,
    )
except ImportError:
    from calibration import (
        fit_temperature,
        negative_log_likelihood,
        probabilities_from_logits,
        validate_logits_targets,
    )


QUALITY_FEATURE_NAMES = (
    "log1p_lidar_point_count",
    "log1p_mmwave_point_count",
    "lidar_azimuth_occupancy",
    "mmwave_azimuth_occupancy",
    "lidar_range_occupancy",
    "mmwave_range_occupancy",
    "lidar_predictive_entropy",
    "mmwave_predictive_entropy",
    "unimodal_js_disagreement",
    "lidar_missing",
    "mmwave_missing",
)


def normalized_predictive_entropy(logits: np.ndarray) -> np.ndarray:
    probabilities = probabilities_from_logits(logits)
    class_count = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    return entropy / np.log(class_count)


def jensen_shannon_disagreement(
    first_logits: np.ndarray,
    second_logits: np.ndarray,
) -> np.ndarray:
    first = probabilities_from_logits(first_logits)
    second = probabilities_from_logits(second_logits)
    if first.shape != second.shape:
        raise ValueError("unimodal logit matrices must have the same shape")
    mixture = np.clip(0.5 * (first + second), 1e-12, 1.0)
    first_kl = np.sum(
        first * (np.log(np.clip(first, 1e-12, 1.0)) - np.log(mixture)), axis=1
    )
    second_kl = np.sum(
        second * (np.log(np.clip(second, 1e-12, 1.0)) - np.log(mixture)), axis=1
    )
    return 0.5 * (first_kl + second_kl) / np.log(2.0)


def build_quality_features(
    lidar_point_counts: np.ndarray,
    mmwave_point_counts: np.ndarray,
    lidar_azimuth_occupancy: np.ndarray,
    mmwave_azimuth_occupancy: np.ndarray,
    lidar_range_occupancy: np.ndarray,
    mmwave_range_occupancy: np.ndarray,
    lidar_logits: np.ndarray,
    mmwave_logits: np.ndarray,
    lidar_missing: np.ndarray | None = None,
    mmwave_missing: np.ndarray | None = None,
) -> np.ndarray:
    """Construct deployable features without using corruption labels or targets."""
    lidar_logits = np.asarray(lidar_logits, dtype=np.float64)
    mmwave_logits = np.asarray(mmwave_logits, dtype=np.float64)
    if lidar_logits.ndim != 2 or lidar_logits.shape != mmwave_logits.shape:
        raise ValueError("lidar_logits and mmwave_logits must share [samples, classes]")
    sample_count = lidar_logits.shape[0]

    def vector(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (sample_count,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite [samples] vector")
        return result

    lidar_counts = vector(lidar_point_counts, "lidar_point_counts")
    mmwave_counts = vector(mmwave_point_counts, "mmwave_point_counts")
    if np.any(lidar_counts < 0) or np.any(mmwave_counts < 0):
        raise ValueError("point counts must be non-negative")
    lidar_missing_values = (
        np.zeros(sample_count, dtype=np.float64)
        if lidar_missing is None
        else vector(lidar_missing, "lidar_missing")
    )
    mmwave_missing_values = (
        np.zeros(sample_count, dtype=np.float64)
        if mmwave_missing is None
        else vector(mmwave_missing, "mmwave_missing")
    )

    features = np.column_stack(
        [
            np.log1p(lidar_counts),
            np.log1p(mmwave_counts),
            vector(lidar_azimuth_occupancy, "lidar_azimuth_occupancy"),
            vector(mmwave_azimuth_occupancy, "mmwave_azimuth_occupancy"),
            vector(lidar_range_occupancy, "lidar_range_occupancy"),
            vector(mmwave_range_occupancy, "mmwave_range_occupancy"),
            normalized_predictive_entropy(lidar_logits),
            normalized_predictive_entropy(mmwave_logits),
            jensen_shannon_disagreement(lidar_logits, mmwave_logits),
            lidar_missing_values,
            mmwave_missing_values,
        ]
    )
    if features.shape != (sample_count, len(QUALITY_FEATURE_NAMES)):
        raise RuntimeError("quality feature matrix has an unexpected shape")
    return features


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return float(np.log(np.expm1(value))) if value < 20.0 else value


@dataclass(frozen=True)
class QualityTemperatureModel:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    minimum_temperature: float
    regularization: float

    def temperatures(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.weights.size:
            raise ValueError("quality features do not match the fitted model")
        standardized = (features - self.feature_mean) / self.feature_scale
        linear = np.clip(standardized @ self.weights + self.bias, -20.0, 20.0)
        return self.minimum_temperature + np.logaddexp(0.0, linear)

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("feature_names", "feature_mean", "feature_scale", "weights"):
            payload[key] = np.asarray(payload[key]).tolist()
        return payload


def fit_quality_temperature_model(
    logits: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray,
    regularization: float,
    minimum_temperature: float = 0.05,
) -> QualityTemperatureModel:
    logits, targets = validate_logits_targets(logits, targets)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != targets.size:
        raise ValueError("features must have shape [samples, quality_features]")
    if not np.isfinite(features).all():
        raise ValueError("quality features must be finite")
    if regularization < 0.0 or minimum_temperature <= 0.0:
        raise ValueError("regularization must be non-negative and T_min positive")

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    standardized = (features - feature_mean) / feature_scale
    global_temperature = fit_temperature(logits, targets)
    initial_bias = _inverse_softplus(global_temperature - minimum_temperature)
    initial = np.concatenate(
        [np.zeros(features.shape[1], dtype=np.float64), [initial_bias]]
    )

    def objective(parameters: np.ndarray) -> float:
        weights = parameters[:-1]
        linear = np.clip(standardized @ weights + parameters[-1], -20.0, 20.0)
        temperatures = minimum_temperature + np.logaddexp(0.0, linear)
        penalty = regularization * float(np.mean(np.square(weights)))
        return negative_log_likelihood(logits, targets, temperatures) + penalty

    result = minimize(
        objective,
        x0=initial,
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0)] * initial.size,
        options={"ftol": 1e-11, "maxiter": 500, "maxls": 100},
    )
    if not result.success:
        raise RuntimeError(f"Quality-aware calibration failed: {result.message}")
    return QualityTemperatureModel(
        feature_names=tuple(QUALITY_FEATURE_NAMES[: features.shape[1]]),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=np.asarray(result.x[:-1], dtype=np.float64),
        bias=float(result.x[-1]),
        minimum_temperature=float(minimum_temperature),
        regularization=float(regularization),
    )


def fit_quality_temperature_group_cv(
    logits: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray,
    groups: np.ndarray,
    regularization_candidates: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    fold_count: int = 5,
) -> tuple[QualityTemperatureModel, list[dict]]:
    """Select L2 strength with subject-grouped calibration-only folds."""
    logits, targets = validate_logits_targets(logits, targets)
    features = np.asarray(features, dtype=np.float64)
    groups = np.asarray(groups)
    if features.shape[0] != targets.size or groups.shape != (targets.size,):
        raise ValueError("features and groups must align with logits")
    unique_group_count = np.unique(groups).size
    if unique_group_count < 2:
        raise ValueError("at least two calibration subject groups are required")
    split_count = min(int(fold_count), unique_group_count)
    splitter = GroupKFold(n_splits=split_count)
    rows: list[dict] = []

    for regularization in regularization_candidates:
        fold_scores = []
        for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(features, targets, groups)
        ):
            model = fit_quality_temperature_model(
                logits[train_indices],
                targets[train_indices],
                features[train_indices],
                regularization=float(regularization),
            )
            score = negative_log_likelihood(
                logits[validation_indices],
                targets[validation_indices],
                model.temperatures(features[validation_indices]),
            )
            fold_scores.append(score)
            rows.append(
                {
                    "regularization": float(regularization),
                    "fold": int(fold_index),
                    "validation_nll": float(score),
                    "validation_subject_count": int(
                        np.unique(groups[validation_indices]).size
                    ),
                }
            )
        for row in rows[-split_count:]:
            row["mean_validation_nll"] = float(np.mean(fold_scores))

    mean_scores = {
        candidate: np.mean(
            [
                row["validation_nll"]
                for row in rows
                if row["regularization"] == float(candidate)
            ]
        )
        for candidate in regularization_candidates
    }
    selected = min(mean_scores, key=lambda candidate: (mean_scores[candidate], candidate))
    final_model = fit_quality_temperature_model(
        logits,
        targets,
        features,
        regularization=float(selected),
    )
    return final_model, rows
