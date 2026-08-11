#!/usr/bin/env python3
"""Fit and evaluate formal multimodal confidence-calibration methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from calibration import (
    calibration_metrics,
    fit_temperature,
    log_probabilities,
    reliability_bins,
)
from lower_limb_protocol import LOWER_LIMB_ACTIONS, LOWER_LIMB_FULL_LABELS
from multimodal_calibration import (
    build_quality_features,
    fit_quality_temperature_group_cv,
)
from multimodal_protocol import ExperimentCondition


def load_condition(path: Path) -> tuple[ExperimentCondition, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    encoded_metadata = arrays.pop("condition_json")
    if encoded_metadata.size != 1:
        raise ValueError(f"condition_json must contain one value: {path}")
    metadata = json.loads(str(encoded_metadata.reshape(()).item()))
    condition = ExperimentCondition(
        mask_name=metadata["mask_name"],
        geometry=metadata["geometry"],
        corruption_seed=int(metadata["corruption_seed"]),
        lidar_drop_rate=float(metadata["lidar_drop_rate"]),
        mmwave_drop_rate=float(metadata["mmwave_drop_rate"]),
    )
    return condition, arrays


def condition_path(
    condition_dir: Path,
    mask_name: str,
    geometry: str,
    seed: int,
    lidar_rate: float,
    mmwave_rate: float,
) -> Path:
    if lidar_rate == 0.0 and mmwave_rate == 0.0:
        condition = ExperimentCondition(mask_name, "clean", 0, 0.0, 0.0)
    else:
        condition = ExperimentCondition(
            mask_name, geometry, seed, lidar_rate, mmwave_rate
        )
    return condition_dir / f"{condition.condition_id}.npz"


def assert_same_samples(reference: dict, candidate: dict, description: str) -> None:
    for key in ("sample_ids", "targets", "subjects", "actions", "partitions"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"Sample mismatch for {description}: {key}")


def convert_output_space(
    logits: np.ndarray,
    targets: np.ndarray,
    output_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    if output_space == "strict27":
        return logits.astype(np.float64), targets.astype(np.int64)
    columns = np.asarray(LOWER_LIMB_FULL_LABELS, dtype=np.int64)
    label_map = {full_label: index for index, full_label in enumerate(columns)}
    try:
        conditioned_targets = np.asarray(
            [label_map[int(target)] for target in targets], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("conditioned7 received a non-target action") from error
    return logits[:, columns].astype(np.float64), conditioned_targets


def load_condition_bundle(
    result_path: Path,
    condition_dir: Path,
    output_space: str,
) -> dict:
    condition, result = load_condition(result_path)
    result_logits, targets = convert_output_space(
        result["logits"], result["targets"], output_space
    )
    sample_count = targets.size
    lidar_missing = np.full(
        sample_count, condition.mask_name == "mmwave_only", dtype=np.float64
    )
    mmwave_missing = np.full(
        sample_count, condition.mask_name == "lidar_only", dtype=np.float64
    )

    if condition.mask_name == "lidar_mmwave":
        lidar_path = condition_path(
            condition_dir,
            "lidar_only",
            condition.geometry,
            condition.corruption_seed,
            condition.lidar_drop_rate,
            0.0,
        )
        mmwave_path = condition_path(
            condition_dir,
            "mmwave_only",
            condition.geometry,
            condition.corruption_seed,
            0.0,
            condition.mmwave_drop_rate,
        )
        if not lidar_path.is_file() or not mmwave_path.is_file():
            raise FileNotFoundError(
                f"Missing matching unimodal result for {condition.condition_id}"
            )
        _, lidar = load_condition(lidar_path)
        _, mmwave = load_condition(mmwave_path)
        assert_same_samples(result, lidar, "LiDAR control")
        assert_same_samples(result, mmwave, "mmWave control")
        lidar_logits, _ = convert_output_space(
            lidar["logits"], lidar["targets"], output_space
        )
        mmwave_logits, _ = convert_output_space(
            mmwave["logits"], mmwave["targets"], output_space
        )
    elif condition.mask_name == "lidar_only":
        lidar_logits = result_logits
        # Uniform logits encode unavailable evidence without fabricating a
        # prediction from the absent sensor.
        mmwave_logits = np.zeros_like(result_logits)
    elif condition.mask_name == "mmwave_only":
        lidar_logits = np.zeros_like(result_logits)
        mmwave_logits = result_logits
    else:
        raise ValueError(f"Unsupported modality mask: {condition.mask_name}")

    features = build_quality_features(
        result["lidar_point_counts"],
        result["mmwave_point_counts"],
        result["lidar_azimuth_occupancy"],
        result["mmwave_azimuth_occupancy"],
        result["lidar_range_occupancy"],
        result["mmwave_range_occupancy"],
        lidar_logits,
        mmwave_logits,
        lidar_missing=lidar_missing,
        mmwave_missing=mmwave_missing,
    )
    return {
        "condition": condition,
        "logits": result_logits,
        "targets": targets,
        "features": features,
        "sample_ids": result["sample_ids"],
        "subjects": result["subjects"],
        "actions": result["actions"],
        "partitions": result["partitions"],
    }


def load_fusion_bundle(
    fusion_path: Path,
    condition_dir: Path,
    output_space: str,
) -> dict:
    """Backward-compatible wrapper used by earlier teaching material."""
    bundle = load_condition_bundle(fusion_path, condition_dir, output_space)
    if bundle["condition"].mask_name != "lidar_mmwave":
        raise ValueError("load_fusion_bundle requires a lidar_mmwave condition")
    return bundle


def deterministic_cap_indices(
    indices: np.ndarray,
    cap: int,
    condition_id: str,
) -> np.ndarray:
    if cap <= 0 or indices.size <= cap:
        return indices
    digest = hashlib.blake2b(condition_id.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % (2**32)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(indices, size=cap, replace=False))


def is_primary_degraded_fusion_condition(
    condition: ExperimentCondition,
) -> bool:
    """Select only preregistered non-clean fused conditions for inference."""
    return (
        condition.mask_name == "lidar_mmwave"
        and condition.geometry in {"uniform", "azimuth_sector"}
    )


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_bootstrap_nll_delta(
    cluster_sums: dict[str, float],
    cluster_counts: dict[str, int],
    repetitions: int,
    seed: int = 20260803,
) -> dict:
    clusters = sorted(cluster_sums)
    if len(clusters) < 2 or repetitions <= 0:
        raise ValueError("cluster bootstrap needs >=2 clusters and positive repetitions")
    sums = np.asarray([cluster_sums[cluster] for cluster in clusters], dtype=np.float64)
    counts = np.asarray([cluster_counts[cluster] for cluster in clusters], dtype=np.float64)
    observed = float(sums.sum() / counts.sum())
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = generator.integers(0, len(clusters), size=len(clusters))
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "cluster_count": len(clusters),
        "repetitions": repetitions,
        "qa_minus_pooled_nll": observed,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_tail_probability": float(
            min(1.0, 2.0 * min(np.mean(estimates <= 0.0), np.mean(estimates >= 0.0)))
        ),
    }


def paired_cluster_sign_flip_test(
    cluster_sums: dict[str, float],
    cluster_counts: dict[str, int],
    repetitions: int,
    seed: int = 20260803,
) -> dict:
    """Test a paired clustered mean difference with Rademacher sign flips."""
    clusters = sorted(cluster_sums)
    if len(clusters) < 2 or repetitions <= 0:
        raise ValueError("cluster sign-flip test needs >=2 clusters and positive repetitions")
    sums = np.asarray([cluster_sums[cluster] for cluster in clusters], dtype=np.float64)
    counts = np.asarray([cluster_counts[cluster] for cluster in clusters], dtype=np.float64)
    denominator = float(counts.sum())
    observed = float(sums.sum() / denominator)
    generator = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    # Fixed-size blocks avoid allocating a large repetitions-by-clusters matrix.
    while completed < repetitions:
        block_size = min(2048, repetitions - completed)
        signs = generator.integers(
            0, 2, size=(block_size, len(clusters)), dtype=np.int8
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = (signs @ sums) / denominator
        extreme += int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
        completed += block_size
    return {
        "sign_flip_repetitions": int(repetitions),
        "sign_flip_seed": int(seed),
        "sign_flip_p": float((extreme + 1) / (repetitions + 1)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Control family-wise error with Holm's step-down procedure."""
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be finite values in [0, 1]")
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(count, dtype=np.float64)
    running_maximum = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(original_index)])
        running_maximum = max(running_maximum, candidate)
        adjusted[int(original_index)] = running_maximum
    return adjusted.tolist()


def compute_robustness_auc_rows(frame: pd.DataFrame) -> list[dict]:
    """Integrate preregistered 1-D degradation paths for every random seed."""
    required = {
        "condition_id",
        "mask_name",
        "geometry",
        "corruption_seed",
        "lidar_drop_rate",
        "mmwave_drop_rate",
        "method",
        "accuracy",
        "macro_f1",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot compute robustness AUC; missing columns: {missing}")
    frame = frame[frame["mask_name"] == "lidar_mmwave"].copy()
    clean = frame[
        (frame["lidar_drop_rate"] == 0.0)
        & (frame["mmwave_drop_rate"] == 0.0)
    ]
    if clean.empty:
        raise ValueError("A clean fusion condition is required for robustness AUC")

    path_definitions = {
        "lidar_degraded_mmwave_clean": lambda data: (
            (data["lidar_drop_rate"] > 0.0)
            & (data["mmwave_drop_rate"] == 0.0)
        ),
        "lidar_clean_mmwave_degraded": lambda data: (
            (data["lidar_drop_rate"] == 0.0)
            & (data["mmwave_drop_rate"] > 0.0)
        ),
        "joint_equal_degradation": lambda data: (
            (data["lidar_drop_rate"] > 0.0)
            & (data["lidar_drop_rate"] == data["mmwave_drop_rate"])
        ),
    }
    rows: list[dict] = []
    methods = tuple(frame["method"].drop_duplicates())
    geometries = tuple(
        geometry for geometry in frame["geometry"].drop_duplicates() if geometry != "clean"
    )
    seeds = sorted(
        int(seed) for seed in frame.loc[frame["geometry"] != "clean", "corruption_seed"].unique()
    )
    for geometry in geometries:
        geometry_frame = frame[frame["geometry"] == geometry]
        for method in methods:
            clean_method = clean[clean["method"] == method]
            if len(clean_method) != 1:
                raise ValueError(f"Expected one clean row for method {method}")
            clean_row = clean_method.iloc[0]
            for seed in seeds:
                seeded = geometry_frame[
                    (geometry_frame["method"] == method)
                    & (geometry_frame["corruption_seed"] == seed)
                ]
                for path_name, selector in path_definitions.items():
                    path = seeded[selector(seeded)].copy()
                    if path.empty:
                        continue
                    if path_name == "lidar_clean_mmwave_degraded":
                        severity_column = "mmwave_drop_rate"
                    else:
                        severity_column = "lidar_drop_rate"
                    points = [
                        {
                            "severity": 0.0,
                            "accuracy": float(clean_row["accuracy"]),
                            "macro_f1": float(clean_row["macro_f1"]),
                        }
                    ]
                    points.extend(
                        {
                            "severity": float(row[severity_column]),
                            "accuracy": float(row["accuracy"]),
                            "macro_f1": float(row["macro_f1"]),
                        }
                        for _, row in path.iterrows()
                    )
                    curve = pd.DataFrame(points).groupby("severity", as_index=False).mean()
                    curve = curve.sort_values("severity")
                    span = float(curve["severity"].iloc[-1] - curve["severity"].iloc[0])
                    if span <= 0.0:
                        continue
                    rows.append(
                        {
                            "geometry": geometry,
                            "corruption_seed": seed,
                            "method": method,
                            "path": path_name,
                            "severity_max": float(curve["severity"].iloc[-1]),
                            "point_count": int(len(curve)),
                            "accuracy_robustness_auc": float(
                                np.trapz(curve["accuracy"], curve["severity"]) / span
                            ),
                            "macro_f1_robustness_auc": float(
                                np.trapz(curve["macro_f1"], curve["severity"]) / span
                            ),
                        }
                    )
    return rows


def plot_target_confusion(
    logits: np.ndarray,
    targets: np.ndarray,
    output_space: str,
    output_path: Path,
) -> None:
    predictions = logits.argmax(axis=1)
    action_names = list(LOWER_LIMB_ACTIONS)
    if output_space == "strict27":
        target_labels = list(LOWER_LIMB_FULL_LABELS)
        target_to_row = {label: index for index, label in enumerate(target_labels)}
        mapped_targets = np.asarray([target_to_row[int(value)] for value in targets])
        mapped_predictions = np.asarray(
            [target_to_row.get(int(value), len(action_names)) for value in predictions]
        )
        prediction_names = action_names + ["OTHER"]
    else:
        mapped_targets = targets
        mapped_predictions = predictions
        prediction_names = action_names
    matrix = confusion_matrix(
        mapped_targets,
        mapped_predictions,
        labels=np.arange(len(prediction_names)),
    )[: len(action_names)]
    denominators = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized = matrix / denominators
    figure, axis = plt.subplots(figsize=(8.4, 6.2))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(prediction_names)), prediction_names, rotation=35, ha="right")
    axis.set_yticks(range(len(action_names)), action_names)
    axis.set_xlabel("Predicted action")
    axis.set_ylabel("True action")
    axis.set_title("Row-normalized target-action confusion")
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[row, column] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axis, label="Row fraction")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_method_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    summary = frame.groupby("method", sort=False)[["nll", "ece"]].mean()
    mask_titles = {
        "calibration_method_comparison_lidar_mmwave": "Fused LiDAR + mmWave",
        "calibration_method_comparison_lidar_only": "LiDAR-only",
        "calibration_method_comparison_mmwave_only": "mmWave-only",
    }
    mask_title = mask_titles.get(output_path.stem, "Recognition")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, label in zip(axes, ("nll", "ece"), ("NLL", "ECE")):
        axis.bar(summary.index, summary[metric], color="#2d6a4f")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"{mask_title} calibration under degradation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_heatmaps(frame: pd.DataFrame, output_dir: Path) -> None:
    source = frame[
        (frame["method"] == "uncalibrated")
        & (frame["mask_name"] == "lidar_mmwave")
    ]
    for geometry in ("uniform", "azimuth_sector"):
        subset = source[source["geometry"] == geometry]
        if subset.empty:
            continue
        for metric in ("accuracy", "nll", "ece"):
            matrix = subset.pivot_table(
                index="lidar_drop_rate",
                columns="mmwave_drop_rate",
                values=metric,
                aggfunc="mean",
            ).sort_index(ascending=False)
            figure, axis = plt.subplots(figsize=(6.2, 5.2))
            image = axis.imshow(matrix.values, aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(matrix.columns)), matrix.columns)
            axis.set_yticks(range(len(matrix.index)), matrix.index)
            axis.set_xlabel("mmWave point-loss rate")
            axis.set_ylabel("LiDAR point-loss rate")
            axis.set_title(f"{geometry}: {metric}")
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix.iloc[row, column]
                    if not np.isfinite(value):
                        axis.text(
                            column,
                            row,
                            "clean\n(separate)",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=7,
                        )
                        continue
                    axis.text(
                        column,
                        row,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white" if value < np.nanmean(matrix.values) else "black",
                        fontsize=8,
                    )
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(
                output_dir / f"heatmap_{geometry}_{metric}.png", dpi=180
            )
            plt.close(figure)


def plot_reliability(
    logits: np.ndarray,
    targets: np.ndarray,
    temperatures: dict[str, float | np.ndarray],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 5.3))
    axis.plot([0, 1], [0, 1], "--", color="black", label="ideal")
    for method, temperature in temperatures.items():
        from calibration import probabilities_from_logits

        probabilities = probabilities_from_logits(logits, temperature)
        rows = reliability_bins(probabilities, targets, bin_count=15)
        valid = [row for row in rows if row["count"]]
        axis.plot(
            [row["mean_confidence"] for row in valid],
            [row["accuracy"] for row in valid],
            marker="o",
            label=method,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-space", choices=("strict27", "conditioned7"), default="strict27")
    parser.add_argument("--calibration-cap-per-condition", type=int, default=1000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--sign-flip-repetitions", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    condition_dir = arguments.inference_dir / "conditions"
    condition_paths = []
    for path in sorted(condition_dir.glob("*.npz")):
        condition, _ = load_condition(path)
        if condition.mask_name in {"lidar_mmwave", "lidar_only", "mmwave_only"}:
            condition_paths.append(path)
    if not condition_paths:
        raise FileNotFoundError("No multimodal condition results were found")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = condition_path(
        condition_dir, "lidar_mmwave", "clean", 0, 0.0, 0.0
    )
    clean_bundles = {}
    clean_temperatures = {}
    for mask_name in ("lidar_mmwave", "lidar_only", "mmwave_only"):
        path = condition_path(condition_dir, mask_name, "clean", 0, 0.0, 0.0)
        bundle = load_condition_bundle(path, condition_dir, arguments.output_space)
        clean_bundles[mask_name] = bundle
        calibration_mask = bundle["partitions"] == "calibration"
        clean_temperatures[mask_name] = fit_temperature(
            bundle["logits"][calibration_mask],
            bundle["targets"][calibration_mask],
        )
    clean_bundle = clean_bundles["lidar_mmwave"]

    calibration_logits = []
    calibration_targets = []
    calibration_features = []
    calibration_groups = []
    for path in condition_paths:
        bundle = load_condition_bundle(path, condition_dir, arguments.output_space)
        indices = np.flatnonzero(bundle["partitions"] == "calibration")
        indices = deterministic_cap_indices(
            indices,
            arguments.calibration_cap_per_condition,
            bundle["condition"].condition_id,
        )
        calibration_logits.append(bundle["logits"][indices])
        calibration_targets.append(bundle["targets"][indices])
        calibration_features.append(bundle["features"][indices])
        calibration_groups.append(bundle["subjects"][indices])
    stacked_logits = np.concatenate(calibration_logits)
    stacked_targets = np.concatenate(calibration_targets)
    stacked_features = np.concatenate(calibration_features)
    stacked_groups = np.concatenate(calibration_groups)

    pooled_temperature = fit_temperature(stacked_logits, stacked_targets)
    quality_model, cross_validation_rows = fit_quality_temperature_group_cv(
        stacked_logits,
        stacked_targets,
        stacked_features,
        stacked_groups,
    )

    metric_rows: list[dict] = []
    cluster_sums: dict[str, float] = defaultdict(float)
    cluster_counts: dict[str, int] = defaultdict(int)
    geometry_cluster_sums: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    geometry_cluster_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    subject_cluster_sums: dict[str, float] = defaultdict(float)
    subject_cluster_counts: dict[str, int] = defaultdict(int)
    geometry_subject_cluster_sums: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    geometry_subject_cluster_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    recall_rows: list[dict] = []
    confirmatory_condition_ids: set[str] = set()
    worst_bundle = None
    worst_accuracy = float("inf")
    for path in condition_paths:
        bundle = load_condition_bundle(path, condition_dir, arguments.output_space)
        condition = bundle["condition"]
        if is_primary_degraded_fusion_condition(condition):
            confirmatory_condition_ids.add(condition.condition_id)
        calibration_mask = bundle["partitions"] == "calibration"
        test_mask = bundle["partitions"] == "test"
        condition_temperature = fit_temperature(
            bundle["logits"][calibration_mask], bundle["targets"][calibration_mask]
        )
        test_logits = bundle["logits"][test_mask]
        test_targets = bundle["targets"][test_mask]
        quality_temperatures = quality_model.temperatures(bundle["features"][test_mask])
        methods = {
            "uncalibrated": 1.0,
            "clean_global_ts": clean_temperatures[condition.mask_name],
            "pooled_global_ts": pooled_temperature,
            "severity_oracle_ts": condition_temperature,
            "quality_aware_ts": quality_temperatures,
        }
        condition_metrics = {}
        for method, temperature in methods.items():
            metrics = calibration_metrics(test_logits, test_targets, temperature)
            condition_metrics[method] = metrics
            metric_rows.append(
                {
                    "condition_id": condition.condition_id,
                    "mask_name": condition.mask_name,
                    "geometry": condition.geometry,
                    "corruption_seed": condition.corruption_seed,
                    "lidar_drop_rate": condition.lidar_drop_rate,
                    "mmwave_drop_rate": condition.mmwave_drop_rate,
                    "output_space": arguments.output_space,
                    "method": method,
                    **asdict(metrics),
                }
            )

        predictions = test_logits.argmax(axis=1)
        for action in LOWER_LIMB_ACTIONS:
            action_mask = bundle["actions"][test_mask] == action
            if not np.any(action_mask):
                continue
            recall_rows.append(
                {
                    "condition_id": condition.condition_id,
                    "mask_name": condition.mask_name,
                    "geometry": condition.geometry,
                    "corruption_seed": condition.corruption_seed,
                    "lidar_drop_rate": condition.lidar_drop_rate,
                    "mmwave_drop_rate": condition.mmwave_drop_rate,
                    "output_space": arguments.output_space,
                    "action": action,
                    "sample_count": int(action_mask.sum()),
                    "recall": float(
                        np.mean(predictions[action_mask] == test_targets[action_mask])
                    ),
                }
            )

        pooled_losses = -log_probabilities(test_logits, pooled_temperature)[
            np.arange(test_targets.size), test_targets
        ]
        quality_losses = -log_probabilities(test_logits, quality_temperatures)[
            np.arange(test_targets.size), test_targets
        ]
        for subject, action, delta in zip(
            bundle["subjects"][test_mask],
            bundle["actions"][test_mask],
            quality_losses - pooled_losses,
        ):
            cluster = f"{subject}/{action}"
            if is_primary_degraded_fusion_condition(condition):
                cluster_sums[cluster] += float(delta)
                cluster_counts[cluster] += 1
                subject_cluster_sums[str(subject)] += float(delta)
                subject_cluster_counts[str(subject)] += 1
                geometry_cluster_sums[condition.geometry][cluster] += float(delta)
                geometry_cluster_counts[condition.geometry][cluster] += 1
                geometry_subject_cluster_sums[condition.geometry][str(subject)] += float(delta)
                geometry_subject_cluster_counts[condition.geometry][str(subject)] += 1

        uncalibrated_accuracy = condition_metrics["uncalibrated"].accuracy
        if (
            condition.mask_name == "lidar_mmwave"
            and uncalibrated_accuracy < worst_accuracy
        ):
            worst_accuracy = uncalibrated_accuracy
            worst_bundle = (
                test_logits,
                test_targets,
                quality_temperatures,
                condition.condition_id,
            )

    save_csv(arguments.output_dir / "metrics_by_condition_method.csv", metric_rows)
    save_csv(arguments.output_dir / "quality_group_cv.csv", cross_validation_rows)
    save_csv(arguments.output_dir / "per_action_recall.csv", recall_rows)
    frame = pd.DataFrame(metric_rows)
    aggregate = (
        frame.groupby(
            [
                "mask_name",
                "geometry",
                "lidar_drop_rate",
                "mmwave_drop_rate",
                "method",
            ],
            dropna=False,
        )[["accuracy", "macro_f1", "nll", "brier", "ece", "adaptive_ece", "aurc"]]
        .agg(["mean", "std", "count"])
    )
    aggregate.to_csv(arguments.output_dir / "aggregate_metrics.csv")

    robustness_rows = compute_robustness_auc_rows(frame)
    save_csv(arguments.output_dir / "robustness_auc_by_seed.csv", robustness_rows)
    robustness_frame = pd.DataFrame(robustness_rows)
    robustness_frame.groupby(["geometry", "method", "path"])[
        ["accuracy_robustness_auc", "macro_f1_robustness_auc"]
    ].agg(["mean", "std", "count"]).to_csv(
        arguments.output_dir / "robustness_auc_summary.csv"
    )

    bootstrap = cluster_bootstrap_nll_delta(
        cluster_sums, cluster_counts, arguments.bootstrap_repetitions
    )
    bootstrap.update(
        paired_cluster_sign_flip_test(
            cluster_sums,
            cluster_counts,
            arguments.sign_flip_repetitions,
        )
    )
    geometry_bootstraps = []
    for offset, geometry in enumerate(("uniform", "azimuth_sector")):
        result = cluster_bootstrap_nll_delta(
            geometry_cluster_sums[geometry],
            geometry_cluster_counts[geometry],
            arguments.bootstrap_repetitions,
            seed=20260804 + offset,
        )
        result.update(
            paired_cluster_sign_flip_test(
                geometry_cluster_sums[geometry],
                geometry_cluster_counts[geometry],
                arguments.sign_flip_repetitions,
                seed=20260804 + offset,
            )
        )
        result["geometry"] = geometry
        geometry_bootstraps.append(result)
    adjusted = holm_adjust([row["sign_flip_p"] for row in geometry_bootstraps])
    for row, adjusted_p in zip(geometry_bootstraps, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
    save_csv(
        arguments.output_dir / "paired_bootstrap_by_geometry.csv",
        geometry_bootstraps,
    )
    subject_sensitivity = cluster_bootstrap_nll_delta(
        subject_cluster_sums,
        subject_cluster_counts,
        arguments.bootstrap_repetitions,
        seed=20260806,
    )
    subject_sensitivity.update(
        paired_cluster_sign_flip_test(
            subject_cluster_sums,
            subject_cluster_counts,
            arguments.sign_flip_repetitions,
            seed=20260806,
        )
    )
    subject_geometry_sensitivities = []
    for offset, geometry in enumerate(("uniform", "azimuth_sector")):
        result = cluster_bootstrap_nll_delta(
            geometry_subject_cluster_sums[geometry],
            geometry_subject_cluster_counts[geometry],
            arguments.bootstrap_repetitions,
            seed=20260807 + offset,
        )
        result.update(
            paired_cluster_sign_flip_test(
                geometry_subject_cluster_sums[geometry],
                geometry_subject_cluster_counts[geometry],
                arguments.sign_flip_repetitions,
                seed=20260807 + offset,
            )
        )
        result["geometry"] = geometry
        subject_geometry_sensitivities.append(result)
    subject_adjusted = holm_adjust(
        [row["sign_flip_p"] for row in subject_geometry_sensitivities]
    )
    for row, adjusted_p in zip(subject_geometry_sensitivities, subject_adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
    save_csv(
        arguments.output_dir / "paired_subject_cluster_sensitivity_by_geometry.csv",
        subject_geometry_sensitivities,
    )
    model_payload = {
        "output_space": arguments.output_space,
        "target_actions": list(LOWER_LIMB_ACTIONS),
        "clean_global_temperatures_by_mask": clean_temperatures,
        "pooled_global_temperature": pooled_temperature,
        "quality_aware_model": quality_model.to_dict(),
        "calibration_sample_count": int(stacked_targets.size),
        "calibration_subject_count": int(np.unique(stacked_groups).size),
        "calibration_cap_per_condition": arguments.calibration_cap_per_condition,
        "confirmatory_scope": {
            "mask_name": "lidar_mmwave",
            "geometries": ["uniform", "azimuth_sector"],
            "clean_excluded": True,
            "condition_count": len(confirmatory_condition_ids),
            "condition_ids_sha256": hashlib.sha256(
                "\n".join(sorted(confirmatory_condition_ids)).encode("utf-8")
            ).hexdigest(),
            "recording_cluster_test_observation_count": int(
                sum(cluster_counts.values())
            ),
        },
        "paired_cluster_bootstrap": bootstrap,
        "geometry_stratified_bootstrap_holm": geometry_bootstraps,
        "paired_subject_cluster_sensitivity": subject_sensitivity,
        "geometry_stratified_subject_cluster_sensitivity_holm": subject_geometry_sensitivities,
    }
    (arguments.output_dir / "calibration_models_and_statistics.json").write_text(
        json.dumps(model_payload, indent=2), encoding="utf-8"
    )

    for mask_name in ("lidar_mmwave", "lidar_only", "mmwave_only"):
        plot_method_comparison(
            frame[frame["mask_name"] == mask_name],
            arguments.output_dir / f"calibration_method_comparison_{mask_name}.png",
        )
    plot_heatmaps(frame, arguments.output_dir)
    clean_test = clean_bundle["partitions"] == "test"
    plot_target_confusion(
        clean_bundle["logits"][clean_test],
        clean_bundle["targets"][clean_test],
        arguments.output_space,
        arguments.output_dir / "confusion_clean.png",
    )
    plot_reliability(
        clean_bundle["logits"][clean_test],
        clean_bundle["targets"][clean_test],
        {
            "uncalibrated": 1.0,
            "clean_global_ts": clean_temperatures["lidar_mmwave"],
            "quality_aware_ts": quality_model.temperatures(
                clean_bundle["features"][clean_test]
            ),
        },
        arguments.output_dir / "reliability_clean.png",
    )
    if worst_bundle is not None:
        logits, targets, temperatures, condition_id = worst_bundle
        plot_reliability(
            logits,
            targets,
            {
                "uncalibrated": 1.0,
                "pooled_global_ts": pooled_temperature,
                "quality_aware_ts": temperatures,
            },
            arguments.output_dir / "reliability_worst.png",
        )
        (arguments.output_dir / "worst_condition.txt").write_text(
            condition_id + "\n", encoding="utf-8"
        )
        plot_target_confusion(
            logits,
            targets,
            arguments.output_space,
            arguments.output_dir / "confusion_worst.png",
        )
    print(json.dumps(model_payload, indent=2))


if __name__ == "__main__":
    main()
