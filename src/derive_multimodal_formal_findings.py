#!/usr/bin/env python3
"""Derive decision-oriented findings from audited multimodal metric tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("accuracy", "macro_f1", "nll", "brier", "ece", "adaptive_ece", "aurc")
SEVERITIES = (0.0, 0.25, 0.5, 0.75, 0.9)
GEOMETRIES = ("uniform", "azimuth_sector")


def load_metric_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "condition_id",
        "mask_name",
        "geometry",
        "corruption_seed",
        "lidar_drop_rate",
        "mmwave_drop_rate",
        "method",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metric table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Metric table is empty")
    return frame


def _single_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected one metric row for {description}, found {len(frame)}")
    return frame.iloc[0]


def _uncalibrated_condition(
    frame: pd.DataFrame,
    mask_name: str,
    geometry: str,
    seed: int,
    lidar_rate: float,
    mmwave_rate: float,
) -> pd.Series:
    source = frame[
        (frame["method"] == "uncalibrated")
        & (frame["mask_name"] == mask_name)
    ]
    if lidar_rate == 0.0 and mmwave_rate == 0.0:
        source = source[source["geometry"] == "clean"]
    else:
        source = source[
            (source["geometry"] == geometry)
            & (source["corruption_seed"] == seed)
            & np.isclose(source["lidar_drop_rate"], lidar_rate)
            & np.isclose(source["mmwave_drop_rate"], mmwave_rate)
        ]
    return _single_row(
        source,
        f"{mask_name}/{geometry}/seed={seed}/lidar={lidar_rate}/mmwave={mmwave_rate}",
    )


def build_fusion_vs_unimodal(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare every degraded fused condition with matched unimodal controls."""
    fusion = frame[
        (frame["method"] == "uncalibrated")
        & (frame["mask_name"] == "lidar_mmwave")
        & (frame["geometry"].isin(GEOMETRIES))
    ]
    rows: list[dict] = []
    for _, fused in fusion.iterrows():
        geometry = str(fused["geometry"])
        seed = int(fused["corruption_seed"])
        lidar_rate = float(fused["lidar_drop_rate"])
        mmwave_rate = float(fused["mmwave_drop_rate"])
        lidar = _uncalibrated_condition(
            frame, "lidar_only", geometry, seed, lidar_rate, 0.0
        )
        mmwave = _uncalibrated_condition(
            frame, "mmwave_only", geometry, seed, 0.0, mmwave_rate
        )
        row = {
            "condition_id": fused["condition_id"],
            "geometry": geometry,
            "corruption_seed": seed,
            "lidar_drop_rate": lidar_rate,
            "mmwave_drop_rate": mmwave_rate,
        }
        for metric in METRICS:
            fused_value = float(fused[metric])
            lidar_value = float(lidar[metric])
            mmwave_value = float(mmwave[metric])
            row[f"fusion_{metric}"] = fused_value
            row[f"lidar_only_{metric}"] = lidar_value
            row[f"mmwave_only_{metric}"] = mmwave_value
            if metric in {"accuracy", "macro_f1"}:
                comparator = max(lidar_value, mmwave_value)
            else:
                comparator = min(lidar_value, mmwave_value)
            row[f"fusion_minus_best_unimodal_{metric}"] = fused_value - comparator
        rows.append(row)
    if not rows:
        raise ValueError("No degraded fused conditions were found")
    return pd.DataFrame(rows)


def build_keep_degraded_vs_drop(frame: pd.DataFrame) -> pd.DataFrame:
    """Test whether retaining one degraded stream beats dropping that stream."""
    clean_lidar = _uncalibrated_condition(frame, "lidar_only", "clean", 0, 0.0, 0.0)
    clean_mmwave = _uncalibrated_condition(frame, "mmwave_only", "clean", 0, 0.0, 0.0)
    rows: list[dict] = []
    for geometry in GEOMETRIES:
        seeds = sorted(
            int(value)
            for value in frame.loc[
                (frame["geometry"] == geometry)
                & (frame["mask_name"] == "lidar_mmwave"),
                "corruption_seed",
            ].unique()
        )
        for seed in seeds:
            for severity in SEVERITIES[1:]:
                comparisons = (
                    (
                        "lidar",
                        _uncalibrated_condition(
                            frame,
                            "lidar_mmwave",
                            geometry,
                            seed,
                            severity,
                            0.0,
                        ),
                        clean_mmwave,
                    ),
                    (
                        "mmwave",
                        _uncalibrated_condition(
                            frame,
                            "lidar_mmwave",
                            geometry,
                            seed,
                            0.0,
                            severity,
                        ),
                        clean_lidar,
                    ),
                )
                for degraded_sensor, keep_row, drop_row in comparisons:
                    output = {
                        "geometry": geometry,
                        "corruption_seed": seed,
                        "degraded_sensor": degraded_sensor,
                        "drop_rate": severity,
                    }
                    for metric in METRICS:
                        output[f"keep_degraded_{metric}"] = float(keep_row[metric])
                        output[f"drop_sensor_{metric}"] = float(drop_row[metric])
                        output[f"keep_minus_drop_{metric}"] = float(
                            keep_row[metric] - drop_row[metric]
                        )
                    rows.append(output)
    if not rows:
        raise ValueError("No keep-versus-drop comparisons were produced")
    return pd.DataFrame(rows)


def build_quality_vs_pooled(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame[
        (frame["mask_name"] == "lidar_mmwave")
        & (frame["geometry"].isin(GEOMETRIES))
        & (frame["method"].isin(("pooled_global_ts", "quality_aware_ts")))
    ]
    index_columns = (
        "condition_id",
        "geometry",
        "corruption_seed",
        "lidar_drop_rate",
        "mmwave_drop_rate",
    )
    pivot = source.pivot(index=list(index_columns), columns="method", values=list(METRICS))
    if pivot.empty:
        raise ValueError("No paired quality-aware and pooled rows were found")
    rows: list[dict] = []
    for index, values in pivot.iterrows():
        row = dict(zip(index_columns, index))
        for metric in METRICS:
            pooled = float(values[(metric, "pooled_global_ts")])
            quality = float(values[(metric, "quality_aware_ts")])
            row[f"pooled_{metric}"] = pooled
            row[f"quality_aware_{metric}"] = quality
            row[f"quality_minus_pooled_{metric}"] = quality - pooled
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_seed(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {*group_columns, "condition_id", "corruption_seed"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return (
        frame.groupby(group_columns, dropna=False)[numeric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def plot_fusion_gain_heatmaps(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)
    for axis, geometry in zip(axes, GEOMETRIES):
        subset = frame[frame["geometry"] == geometry]
        matrix = subset.pivot_table(
            index="lidar_drop_rate",
            columns="mmwave_drop_rate",
            values="fusion_minus_best_unimodal_accuracy",
            aggfunc="mean",
        ).reindex(index=SEVERITIES[::-1], columns=SEVERITIES)
        limit = max(0.01, float(np.nanmax(np.abs(matrix.values))))
        image = axis.imshow(
            matrix.values,
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(geometry.replace("_", " "))
        axis.set_xticks(range(len(SEVERITIES)), SEVERITIES)
        axis.set_yticks(range(len(SEVERITIES)), SEVERITIES[::-1])
        axis.set_xlabel("mmWave point-loss rate")
        axis.set_ylabel("LiDAR point-loss rate")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="Fusion accuracy - best unimodal accuracy")
    figure.suptitle("Does fusion outperform the best matched unimodal branch?")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_keep_vs_drop(frame: pd.DataFrame, output_path: Path) -> None:
    summary = (
        frame.groupby(["geometry", "degraded_sensor", "drop_rate"])[
            "keep_minus_drop_accuracy"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    colors = {"uniform": "#2a9d8f", "azimuth_sector": "#e76f51"}
    markers = {"lidar": "o", "mmwave": "s"}
    for (geometry, sensor), group in summary.groupby(["geometry", "degraded_sensor"]):
        group = group.sort_values("drop_rate")
        axis.errorbar(
            group["drop_rate"],
            group["mean"],
            yerr=group["std"].fillna(0.0),
            label=f"{geometry}: degraded {sensor}",
            color=colors[geometry],
            marker=markers[sensor],
            capsize=3,
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Point-loss rate of retained degraded sensor")
    axis.set_ylabel("Accuracy: retain degraded sensor - drop it")
    axis.set_title("When does a degraded modality become harmful?")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_quality_nll_heatmaps(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)
    for axis, geometry in zip(axes, GEOMETRIES):
        subset = frame[frame["geometry"] == geometry]
        matrix = subset.pivot_table(
            index="lidar_drop_rate",
            columns="mmwave_drop_rate",
            values="quality_minus_pooled_nll",
            aggfunc="mean",
        ).reindex(index=SEVERITIES[::-1], columns=SEVERITIES)
        limit = max(0.01, float(np.nanmax(np.abs(matrix.values))))
        image = axis.imshow(
            matrix.values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(geometry.replace("_", " "))
        axis.set_xticks(range(len(SEVERITIES)), SEVERITIES)
        axis.set_yticks(range(len(SEVERITIES)), SEVERITIES[::-1])
        axis.set_xlabel("mmWave point-loss rate")
        axis.set_ylabel("LiDAR point-loss rate")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="Quality-aware NLL - pooled NLL")
    figure.suptitle("Where does quality-aware calibration improve NLL?")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_matched_robustness_curves(frame: pd.DataFrame, output_path: Path) -> None:
    source = frame[
        (frame["method"] == "uncalibrated")
        & (frame["mask_name"] == "lidar_mmwave")
    ]
    clean = _single_row(source[source["geometry"] == "clean"], "clean fused condition")
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    metric_labels = (("accuracy", "Accuracy"), ("nll", "NLL"), ("ece", "ECE"))
    for geometry, color in (("uniform", "#2a9d8f"), ("azimuth_sector", "#e76f51")):
        subset = source[
            (source["geometry"] == geometry)
            & np.isclose(source["lidar_drop_rate"], source["mmwave_drop_rate"])
        ]
        for axis, (metric, label) in zip(axes, metric_labels):
            grouped = subset.groupby("lidar_drop_rate")[metric].agg(["mean", "std"])
            x = np.asarray(SEVERITIES)
            means = [float(clean[metric])] + [float(grouped.loc[value, "mean"]) for value in SEVERITIES[1:]]
            stds = [0.0] + [float(grouped.loc[value, "std"]) for value in SEVERITIES[1:]]
            axis.errorbar(x, means, yerr=stds, color=color, marker="o", capsize=3, label=geometry)
            axis.set_xlabel("Matched point-loss rate")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Fused recognition and confidence under matched sensor degradation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_key_findings(
    fusion: pd.DataFrame,
    keep_drop: pd.DataFrame,
    quality: pd.DataFrame,
) -> dict:
    qa_delta = quality["quality_minus_pooled_nll"]
    keep_delta = keep_drop["keep_minus_drop_accuracy"]
    fusion_delta = fusion["fusion_minus_best_unimodal_accuracy"]
    return {
        "derived_schema_version": 1,
        "fusion_condition_count": int(len(fusion)),
        "keep_vs_drop_comparison_count": int(len(keep_drop)),
        "quality_vs_pooled_condition_count": int(len(quality)),
        "fusion_accuracy_gain_mean": float(fusion_delta.mean()),
        "fusion_accuracy_gain_min": float(fusion_delta.min()),
        "fusion_accuracy_gain_positive_fraction": float(np.mean(fusion_delta > 0.0)),
        "retain_degraded_accuracy_delta_mean": float(keep_delta.mean()),
        "retain_degraded_harmful_fraction": float(np.mean(keep_delta < 0.0)),
        "quality_minus_pooled_nll_mean": float(qa_delta.mean()),
        "quality_aware_nll_win_fraction": float(np.mean(qa_delta < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    frame = load_metric_table(arguments.metrics)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    fusion = build_fusion_vs_unimodal(frame)
    keep_drop = build_keep_degraded_vs_drop(frame)
    quality = build_quality_vs_pooled(frame)

    fusion.to_csv(arguments.output_dir / "fusion_vs_unimodal_by_condition.csv", index=False)
    summarize_by_seed(
        fusion, ["geometry", "lidar_drop_rate", "mmwave_drop_rate"]
    ).to_csv(arguments.output_dir / "fusion_vs_unimodal_summary.csv", index=False)
    keep_drop.to_csv(arguments.output_dir / "keep_degraded_vs_drop_by_seed.csv", index=False)
    summarize_by_seed(
        keep_drop, ["geometry", "degraded_sensor", "drop_rate"]
    ).to_csv(arguments.output_dir / "keep_degraded_vs_drop_summary.csv", index=False)
    quality.to_csv(arguments.output_dir / "quality_vs_pooled_by_condition.csv", index=False)
    summarize_by_seed(
        quality, ["geometry", "lidar_drop_rate", "mmwave_drop_rate"]
    ).to_csv(arguments.output_dir / "quality_vs_pooled_summary.csv", index=False)

    plot_fusion_gain_heatmaps(
        fusion, arguments.output_dir / "fusion_gain_over_best_unimodal.png"
    )
    plot_keep_vs_drop(
        keep_drop, arguments.output_dir / "retain_degraded_vs_drop_sensor.png"
    )
    plot_quality_nll_heatmaps(
        quality, arguments.output_dir / "quality_aware_nll_gain_heatmap.png"
    )
    plot_matched_robustness_curves(
        frame, arguments.output_dir / "matched_degradation_robustness_curves.png"
    )

    findings = build_key_findings(fusion, keep_drop, quality)
    (arguments.output_dir / "derived_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **findings}, indent=2))


if __name__ == "__main__":
    main()
