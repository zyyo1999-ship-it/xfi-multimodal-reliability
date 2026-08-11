"""Post-hoc confidence calibration and probabilistic evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score


TEMPERATURE_BOUNDS = (0.05, 20.0)


@dataclass(frozen=True)
class CalibrationMetrics:
    sample_count: int
    accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece: float
    adaptive_ece: float
    mce: float
    mean_confidence: float
    confidence_accuracy_gap: float
    aurc: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_logits_targets(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [samples]")
    if logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("at least one sample and two classes are required")
    if targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("targets contain an invalid class index")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return logits, targets


def temperatures_to_vector(
    temperature: float | np.ndarray,
    sample_count: int,
) -> np.ndarray:
    values = np.asarray(temperature, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(sample_count, float(values), dtype=np.float64)
    if values.shape != (sample_count,):
        raise ValueError("temperature must be scalar or have shape [samples]")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("all temperatures must be finite and positive")
    return values


def log_probabilities(
    logits: np.ndarray,
    temperature: float | np.ndarray = 1.0,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    temperatures = temperatures_to_vector(temperature, logits.shape[0])
    scaled = logits / temperatures[:, None]
    return scaled - logsumexp(scaled, axis=1, keepdims=True)


def probabilities_from_logits(
    logits: np.ndarray,
    temperature: float | np.ndarray = 1.0,
) -> np.ndarray:
    return np.exp(log_probabilities(logits, temperature))


def negative_log_likelihood(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float | np.ndarray = 1.0,
) -> float:
    logits, targets = validate_logits_targets(logits, targets)
    log_probs = log_probabilities(logits, temperature)
    return float(-np.mean(log_probs[np.arange(targets.size), targets]))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit one positive temperature by minimizing calibration-set NLL."""
    logits, targets = validate_logits_targets(logits, targets)
    log_bounds = tuple(np.log(TEMPERATURE_BOUNDS))
    result = minimize_scalar(
        lambda log_temperature: negative_log_likelihood(
            logits, targets, float(np.exp(log_temperature))
        ),
        bounds=log_bounds,
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def density_degradation_score(
    remaining_point_counts: np.ndarray,
    clean_calibration_median: float,
) -> np.ndarray:
    """Convert observed point density into a bounded degradation proxy.

    A score of zero means that the observed frame has at least the median clean
    calibration density. A score approaching one means very few points remain.
    This proxy uses only the current point count and one calibration statistic.
    """
    counts = np.asarray(remaining_point_counts, dtype=np.float64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("remaining_point_counts must be a non-negative vector")
    if not np.isfinite(clean_calibration_median) or clean_calibration_median <= 0:
        raise ValueError("clean_calibration_median must be positive")
    return np.clip(1.0 - counts / clean_calibration_median, 0.0, 1.0)


def quality_temperatures(
    degradation_scores: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(degradation_scores, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if scores.ndim != 1 or parameters.shape != (2,):
        raise ValueError("scores must be [samples] and parameters must be [2]")
    log_temperature = np.clip(parameters[0] + parameters[1] * scores, -3.0, 3.0)
    return np.exp(log_temperature)


def fit_quality_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    degradation_scores: np.ndarray,
) -> np.ndarray:
    """Fit log(T)=a+b*q, where q is an observed degradation score."""
    logits, targets = validate_logits_targets(logits, targets)
    scores = np.asarray(degradation_scores, dtype=np.float64)
    if scores.shape != (targets.size,) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("degradation_scores must be in [0, 1] for every sample")

    initial_temperature = fit_temperature(logits, targets)
    initial = np.array([np.log(initial_temperature), 0.0], dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        return negative_log_likelihood(
            logits,
            targets,
            quality_temperatures(scores, parameters),
        )

    result = minimize(
        objective,
        x0=initial,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (-6.0, 6.0)],
        # The clipped quality model can require more than SciPy's default 20
        # line-search steps near a parameter boundary. A larger deterministic
        # limit prevents a valid optimum from being reported as an abnormal
        # line-search termination on x86 builds.
        options={"ftol": 1e-12, "maxiter": 500, "maxls": 100},
    )
    if not result.success:
        raise RuntimeError(f"Quality-aware optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def polynomial_quality_temperatures(
    degradation_scores: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate log(T)=a+b*q+c*q^2 for a bounded quality score q."""
    scores = np.asarray(degradation_scores, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if scores.ndim != 1 or parameters.shape != (3,):
        raise ValueError("scores must be [samples] and parameters must be [3]")
    log_temperature = np.clip(
        parameters[0]
        + parameters[1] * scores
        + parameters[2] * np.square(scores),
        -3.0,
        3.0,
    )
    return np.exp(log_temperature)


def fit_polynomial_quality_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    degradation_scores: np.ndarray,
) -> np.ndarray:
    """Fit a quadratic quality-to-temperature mapping by calibration NLL."""
    logits, targets = validate_logits_targets(logits, targets)
    scores = np.asarray(degradation_scores, dtype=np.float64)
    if scores.shape != (targets.size,) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("degradation_scores must be in [0, 1] for every sample")

    linear_parameters = fit_quality_temperature(logits, targets, scores)
    initial = np.array(
        [linear_parameters[0], linear_parameters[1], 0.0], dtype=np.float64
    )

    def objective(parameters: np.ndarray) -> float:
        return negative_log_likelihood(
            logits,
            targets,
            polynomial_quality_temperatures(scores, parameters),
        )

    result = minimize(
        objective,
        x0=initial,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (-8.0, 8.0), (-8.0, 8.0)],
        options={"ftol": 1e-12, "maxiter": 1000, "maxls": 100},
    )
    if not result.success:
        raise RuntimeError(
            f"Polynomial quality optimization failed: {result.message}"
        )
    return np.asarray(result.x, dtype=np.float64)


def apply_vector_scaling(
    logits: np.ndarray,
    log_scales: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """Apply class-wise positive scales and centred biases to logits."""
    logits = np.asarray(logits, dtype=np.float64)
    log_scales = np.asarray(log_scales, dtype=np.float64)
    biases = np.asarray(biases, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    class_count = logits.shape[1]
    if log_scales.shape != (class_count,) or biases.shape != (class_count,):
        raise ValueError("vector-scaling parameters must match the class count")
    centred_biases = biases - biases.mean()
    return logits * np.exp(log_scales)[None, :] + centred_biases[None, :]


def fit_vector_scaling(
    logits: np.ndarray,
    targets: np.ndarray,
    regularization: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit diagonal vector scaling with light identity regularization.

    Unlike scalar temperature scaling, vector scaling can change the predicted
    class. The regularizer discourages large class-wise distortions and is fixed
    before test-set evaluation.
    """
    logits, targets = validate_logits_targets(logits, targets)
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    class_count = logits.shape[1]
    initial = np.zeros(class_count * 2, dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        log_scales = parameters[:class_count]
        biases = parameters[class_count:]
        transformed = apply_vector_scaling(logits, log_scales, biases)
        penalty = regularization * (
            np.mean(np.square(log_scales))
            + np.mean(np.square(biases - biases.mean()))
        )
        return negative_log_likelihood(transformed, targets) + penalty

    result = minimize(
        objective,
        x0=initial,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0)] * class_count + [(-8.0, 8.0)] * class_count,
        options={"ftol": 1e-12, "maxiter": 1000, "maxls": 100},
    )
    if not result.success:
        raise RuntimeError(f"Vector scaling optimization failed: {result.message}")
    return (
        np.asarray(result.x[:class_count], dtype=np.float64),
        np.asarray(result.x[class_count:], dtype=np.float64),
    )


def reliability_bins(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_count: int = 15,
) -> list[dict]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or targets.shape != (probabilities.shape[0],):
        raise ValueError("probabilities/targets have incompatible shapes")
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    assignments = np.minimum(np.digitize(confidences, edges[1:-1]), bin_count - 1)
    rows = []
    for index in range(bin_count):
        mask = assignments == index
        count = int(mask.sum())
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "fraction": float(count / targets.size),
                "mean_confidence": float(confidences[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return rows


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_count: int = 15,
) -> tuple[float, float]:
    rows = reliability_bins(probabilities, targets, bin_count)
    weighted_gaps = [
        row["fraction"] * abs(row["accuracy"] - row["mean_confidence"])
        for row in rows
        if row["count"]
    ]
    gaps = [
        abs(row["accuracy"] - row["mean_confidence"])
        for row in rows
        if row["count"]
    ]
    return float(sum(weighted_gaps)), float(max(gaps, default=0.0))


def adaptive_expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_count: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == targets
    ordered_indices = np.argsort(confidences)
    bins = np.array_split(ordered_indices, min(bin_count, targets.size))
    return float(
        sum(
            (indices.size / targets.size)
            * abs(correct[indices].mean() - confidences[indices].mean())
            for indices in bins
            if indices.size
        )
    )


def risk_coverage_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    confidences = probabilities.max(axis=1)
    errors = (probabilities.argmax(axis=1) != targets).astype(np.float64)
    order = np.argsort(-confidences, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, targets.size + 1)
    coverage = np.arange(1, targets.size + 1, dtype=np.float64) / targets.size
    return coverage, cumulative_risk, float(cumulative_risk.mean())


def calibration_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float | np.ndarray = 1.0,
    bin_count: int = 15,
) -> CalibrationMetrics:
    logits, targets = validate_logits_targets(logits, targets)
    probabilities = probabilities_from_logits(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accuracy = float(accuracy_score(targets, predictions))
    ece, mce = expected_calibration_error(probabilities, targets, bin_count)
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[targets]
    _, _, aurc = risk_coverage_curve(probabilities, targets)
    return CalibrationMetrics(
        sample_count=int(targets.size),
        accuracy=accuracy,
        macro_f1=float(
            f1_score(
                targets,
                predictions,
                labels=np.unique(targets),
                average="macro",
                zero_division=0,
            )
        ),
        nll=negative_log_likelihood(logits, targets, temperature),
        brier=float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        ece=ece,
        adaptive_ece=adaptive_expected_calibration_error(
            probabilities, targets, bin_count
        ),
        mce=mce,
        mean_confidence=float(confidence.mean()),
        confidence_accuracy_gap=float(confidence.mean() - accuracy),
        aurc=aurc,
    )
