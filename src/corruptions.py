"""Deterministic sensor degradation operators for the pilot experiment."""

from __future__ import annotations

import numpy as np


def _validate_point_loss_inputs(
    points: np.ndarray,
    drop_rate: float,
) -> int:
    if points.ndim != 2:
        raise ValueError("points must have shape N x D")
    if not 0.0 <= drop_rate <= 1.0:
        raise ValueError("drop_rate must be in [0.0, 1.0]")
    if points.shape[0] == 0 or drop_rate == 1.0:
        return 0
    keep_count = int(round(points.shape[0] * (1.0 - drop_rate)))
    return max(1, min(points.shape[0], keep_count))


def point_dropout(
    points: np.ndarray,
    drop_rate: float,
    seed: int,
) -> np.ndarray:
    """Drop a controlled fraction of points while preserving their order."""
    keep_count = _validate_point_loss_inputs(points, drop_rate)
    if keep_count == 0:
        return np.empty((0, points.shape[1]), dtype=points.dtype)
    if drop_rate == 0.0:
        return points.copy()

    generator = np.random.default_rng(seed)
    # One fixed random ranking creates nested corruptions: for the same seed,
    # every more severe condition retains a subset of the milder condition.
    kept_indices = np.sort(generator.permutation(points.shape[0])[:keep_count])
    return points[kept_indices].copy()


def azimuth_sector_dropout(
    points: np.ndarray,
    drop_rate: float,
    seed: int,
) -> np.ndarray:
    """Remove one contiguous azimuth sector while preserving row order.

    The first two columns are treated as planar coordinates. A seed-controlled
    sector centre is sampled, and the points closest to that direction are
    removed until the requested count is reached. For a fixed seed, increasingly
    severe conditions are nested because they extend the same removal ranking.
    This is a controlled structured-occlusion proxy, not a physical radar model.
    """
    keep_count = _validate_point_loss_inputs(points, drop_rate)
    if points.shape[1] < 2:
        raise ValueError("azimuth sector dropout requires at least two coordinates")
    if keep_count == 0:
        return np.empty((0, points.shape[1]), dtype=points.dtype)
    if drop_rate == 0.0:
        return points.copy()

    generator = np.random.default_rng(seed)
    sector_centre = generator.uniform(-np.pi, np.pi)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    circular_distance = np.abs(
        np.arctan2(
            np.sin(azimuth - sector_centre),
            np.cos(azimuth - sector_centre),
        )
    )
    removal_count = points.shape[0] - keep_count
    removal_ranking = np.argsort(circular_distance, kind="stable")
    keep_mask = np.ones(points.shape[0], dtype=bool)
    keep_mask[removal_ranking[:removal_count]] = False
    return points[keep_mask].copy()
