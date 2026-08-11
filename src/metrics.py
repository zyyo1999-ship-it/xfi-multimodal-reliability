"""Evaluation metrics used by the future full X-Fi robustness runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    sample_count: int
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> ClassificationMetrics:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional")
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.size == 0:
        raise ValueError("at least one prediction is required")

    return ClassificationMetrics(
        sample_count=int(targets.size),
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    )

