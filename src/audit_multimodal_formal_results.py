#!/usr/bin/env python3
"""Audit the complete formal LiDAR+mmWave inference and analysis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lower_limb_protocol import LOWER_LIMB_ACTIONS
from multimodal_protocol import formal_conditions


EXPECTED_METHODS = {
    "uncalibrated",
    "clean_global_ts",
    "pooled_global_ts",
    "severity_oracle_ts",
    "quality_aware_ts",
}
EXPECTED_OUTPUT_SPACES = ("strict27", "conditioned7")
EXPECTED_CLEAN_MASKS = {"lidar_mmwave", "lidar_only", "mmwave_only"}
EXPECTED_CONFIRMATORY_GEOMETRIES = {"uniform", "azimuth_sector"}


def expected_confirmatory_condition_ids() -> list[str]:
    return sorted(
        condition.condition_id
        for condition in formal_conditions()
        if condition.mask_name == "lidar_mmwave"
        and condition.geometry in EXPECTED_CONFIRMATORY_GEOMETRIES
    )


def condition_id_bundle_sha256(condition_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(condition_ids).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def expected_count(clean_counts: np.ndarray, rate: float, active: bool) -> np.ndarray:
    if not active:
        return np.zeros_like(clean_counts)
    retained = np.rint(clean_counts.astype(np.float64) * (1.0 - rate))
    return np.clip(retained, 1, clean_counts).astype(clean_counts.dtype)


def audit_clean_baseline(baseline_dir: Path) -> tuple[dict, list[str]]:
    """Verify that the preregistered full-27-class checkpoint gate passed."""
    gate_path = baseline_dir / "clean_baseline_gate.json"
    manifest_path = baseline_dir / "inference_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Clean-baseline gate report or inference manifest is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("Full-27-class clean-baseline gate did not pass")
    if gate.get("scope") != "full_27_class_clean_validation":
        raise RuntimeError("Unexpected clean-baseline scope")
    if int(gate.get("sample_count", -1)) != 54_433:
        raise RuntimeError("Clean-baseline frame count is not 54,433")
    if int(gate.get("action_count", -1)) != 27:
        raise RuntimeError("Clean-baseline action count is not 27")
    if int(gate.get("class_count", -1)) != 27:
        raise RuntimeError("Clean-baseline class count is not 27")
    rows = gate.get("rows", [])
    if {row.get("mask_name") for row in rows} != EXPECTED_CLEAN_MASKS:
        raise RuntimeError("Clean-baseline modality masks are incomplete")
    if any(not row.get("within_tolerance", False) for row in rows):
        raise RuntimeError("A clean-baseline modality is outside tolerance")
    if int(manifest.get("expected_sample_count", -1)) != 54_433:
        raise RuntimeError("Clean-baseline manifest frame count is inconsistent")
    completed = manifest.get("completed", {})
    if len(completed) != 3:
        raise RuntimeError("Clean-baseline manifest must contain exactly three masks")
    condition_dir = baseline_dir / "conditions"
    for condition_id, metadata in completed.items():
        artifact = condition_dir / f"{condition_id}.npz"
        if not artifact.is_file():
            raise RuntimeError(f"Missing clean-baseline artifact: {artifact.name}")
        if sha256_file(artifact) != metadata.get("sha256"):
            raise RuntimeError(f"Clean-baseline artifact hash mismatch: {artifact.name}")
    return gate, [
        "full_27_class_clean_baseline_gate_passed",
        "clean_baseline_three_masks_complete",
        "clean_baseline_artifact_sha256_matches_manifest",
    ]


def audit_inference(inference_dir: Path) -> tuple[dict, list[str]]:
    manifest_path = inference_dir / "inference_manifest.json"
    alignment_path = inference_dir / "data_alignment_audit.json"
    if not manifest_path.is_file() or not alignment_path.is_file():
        raise RuntimeError("Inference manifest or alignment audit is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    expected = {condition.condition_id: condition for condition in formal_conditions()}
    completed = manifest.get("completed", {})
    if set(completed) != set(expected):
        raise RuntimeError(
            f"Condition mismatch: missing={len(set(expected) - set(completed))}, "
            f"extra={len(set(completed) - set(expected))}"
        )
    sample_count = int(manifest["expected_sample_count"])
    if sample_count <= 0 or sample_count != int(alignment["effective_frame_count"]):
        raise RuntimeError("Invalid or inconsistent formal sample count")

    condition_dir = inference_dir / "conditions"
    clean_arrays = {}
    reference_ids = None
    reference_targets = None
    reference_subjects = None
    reference_partitions = None
    artifact_hashes = {}
    for condition_id, condition in expected.items():
        path = condition_dir / f"{condition_id}.npz"
        if not path.is_file():
            raise RuntimeError(f"Missing condition artifact: {path.name}")
        actual_hash = sha256_file(path)
        if actual_hash != completed[condition_id].get("sha256"):
            raise RuntimeError(f"Manifest hash mismatch: {path.name}")
        artifact_hashes[path.name] = actual_hash
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        if arrays["logits"].shape != (sample_count, 27):
            raise RuntimeError(f"Unexpected logits shape: {path.name}")
        if not np.isfinite(arrays["logits"]).all():
            raise RuntimeError(f"Non-finite logits: {path.name}")
        if reference_ids is None:
            reference_ids = arrays["sample_ids"]
            reference_targets = arrays["targets"]
            reference_subjects = arrays["subjects"]
            reference_partitions = arrays["partitions"]
            if len(set(reference_ids.tolist())) != sample_count:
                raise RuntimeError("Sample IDs are not unique")
        else:
            for name, reference in (
                ("sample_ids", reference_ids),
                ("targets", reference_targets),
                ("subjects", reference_subjects),
                ("partitions", reference_partitions),
            ):
                if not np.array_equal(arrays[name], reference):
                    raise RuntimeError(f"Cross-condition {name} mismatch: {path.name}")

        if condition.geometry == "clean":
            clean_arrays[condition.mask_name] = arrays

    if set(clean_arrays) != {"lidar_mmwave", "lidar_only", "mmwave_only"}:
        raise RuntimeError("The three clean modality masks are incomplete")
    clean_fusion = clean_arrays["lidar_mmwave"]
    for condition_id, condition in expected.items():
        path = condition_dir / f"{condition_id}.npz"
        with np.load(path, allow_pickle=False) as payload:
            lidar_counts = payload["lidar_point_counts"]
            mmwave_counts = payload["mmwave_point_counts"]
        expected_lidar = expected_count(
            clean_fusion["lidar_point_counts"],
            condition.lidar_drop_rate,
            condition.mask_name != "mmwave_only",
        )
        expected_mmwave = expected_count(
            clean_fusion["mmwave_point_counts"],
            condition.mmwave_drop_rate,
            condition.mask_name != "lidar_only",
        )
        if not np.array_equal(lidar_counts, expected_lidar):
            raise RuntimeError(f"LiDAR point-count mismatch: {condition_id}")
        if not np.array_equal(mmwave_counts, expected_mmwave):
            raise RuntimeError(f"mmWave point-count mismatch: {condition_id}")

    calibration_subjects = set(reference_subjects[reference_partitions == "calibration"])
    test_subjects = set(reference_subjects[reference_partitions == "test"])
    if not calibration_subjects or not test_subjects or calibration_subjects & test_subjects:
        raise RuntimeError("Post-hoc calibration/test subject split is invalid")
    checks = [
        "all_323_conditions_present",
        "artifact_sha256_matches_manifest",
        "finite_27_class_logits",
        "cross_condition_sample_alignment",
        "exact_corruption_point_counts",
        "subject_disjoint_posthoc_partition",
    ]
    return {
        "manifest": manifest,
        "alignment": alignment,
        "artifact_hashes": artifact_hashes,
        "calibration_subject_count": len(calibration_subjects),
        "test_subject_count": len(test_subjects),
    }, checks


def audit_analysis(analysis_dir: Path, sample_count: int) -> tuple[dict, list[str]]:
    summaries = {}
    required = {
        "metrics_by_condition_method.csv",
        "aggregate_metrics.csv",
        "per_action_recall.csv",
        "robustness_auc_by_seed.csv",
        "robustness_auc_summary.csv",
        "paired_bootstrap_by_geometry.csv",
        "paired_subject_cluster_sensitivity_by_geometry.csv",
        "quality_group_cv.csv",
        "calibration_models_and_statistics.json",
        "confusion_clean.png",
        "confusion_worst.png",
        "reliability_clean.png",
        "reliability_worst.png",
    }
    required_derived = {
        "fusion_vs_unimodal_by_condition.csv",
        "fusion_vs_unimodal_summary.csv",
        "keep_degraded_vs_drop_by_seed.csv",
        "keep_degraded_vs_drop_summary.csv",
        "quality_vs_pooled_by_condition.csv",
        "quality_vs_pooled_summary.csv",
        "fusion_gain_over_best_unimodal.png",
        "retain_degraded_vs_drop_sensor.png",
        "quality_aware_nll_gain_heatmap.png",
        "matched_degradation_robustness_curves.png",
        "derived_findings.json",
    }
    for output_space in EXPECTED_OUTPUT_SPACES:
        directory = analysis_dir / output_space
        missing = sorted(name for name in required if not (directory / name).is_file())
        if missing:
            raise RuntimeError(f"Missing {output_space} analysis artifacts: {missing}")
        derived_directory = directory / "derived"
        missing_derived = sorted(
            name for name in required_derived if not (derived_directory / name).is_file()
        )
        if missing_derived:
            raise RuntimeError(
                f"Missing {output_space} derived artifacts: {missing_derived}"
            )
        metrics = read_csv(directory / "metrics_by_condition_method.csv")
        recalls = read_csv(directory / "per_action_recall.csv")
        auc_rows = read_csv(directory / "robustness_auc_by_seed.csv")
        bootstraps = read_csv(directory / "paired_bootstrap_by_geometry.csv")
        subject_sensitivities = read_csv(
            directory / "paired_subject_cluster_sensitivity_by_geometry.csv"
        )
        if len(metrics) != len(formal_conditions()) * len(EXPECTED_METHODS):
            raise RuntimeError(f"Unexpected metric row count for {output_space}")
        if len(recalls) != len(formal_conditions()) * len(LOWER_LIMB_ACTIONS):
            raise RuntimeError(f"Unexpected recall row count for {output_space}")
        if len(auc_rows) != 2 * 5 * len(EXPECTED_METHODS) * 3:
            raise RuntimeError(f"Unexpected robustness AUC rows for {output_space}")
        if {row["method"] for row in metrics} != EXPECTED_METHODS:
            raise RuntimeError(f"Calibration method set mismatch for {output_space}")
        grouped = {}
        for row in metrics:
            grouped[(row["condition_id"], row["method"])] = row
        for condition in formal_conditions():
            baseline = grouped[(condition.condition_id, "uncalibrated")]
            for method in EXPECTED_METHODS - {"uncalibrated"}:
                calibrated = grouped[(condition.condition_id, method)]
                for metric in ("accuracy", "macro_f1"):
                    if not np.isclose(
                        float(baseline[metric]), float(calibrated[metric]), atol=1e-12
                    ):
                        raise RuntimeError(
                            f"Positive temperature changed {metric}: "
                            f"{condition.condition_id}/{method}"
                        )
        if len(bootstraps) != 2 or any(
            int(row["repetitions"]) < 2000 for row in bootstraps
        ):
            raise RuntimeError(f"Bootstrap protocol mismatch for {output_space}")
        if {row["geometry"] for row in bootstraps} != EXPECTED_CONFIRMATORY_GEOMETRIES:
            raise RuntimeError(
                f"Confirmatory geometry set mismatch for {output_space}"
            )
        if any(int(row["sign_flip_repetitions"]) < 20_000 for row in bootstraps):
            raise RuntimeError(f"Sign-flip protocol mismatch for {output_space}")
        if any(not 0.0 <= float(row["sign_flip_p"]) <= 1.0 for row in bootstraps):
            raise RuntimeError(f"Invalid sign-flip p-value for {output_space}")
        if any(not 0.0 <= float(row["holm_adjusted_p"]) <= 1.0 for row in bootstraps):
            raise RuntimeError(f"Invalid Holm-adjusted p-value for {output_space}")
        if len(subject_sensitivities) != 2 or any(
            int(row["repetitions"]) < 2000 for row in subject_sensitivities
        ):
            raise RuntimeError(
                f"Subject-cluster sensitivity protocol mismatch for {output_space}"
            )
        if {
            row["geometry"] for row in subject_sensitivities
        } != EXPECTED_CONFIRMATORY_GEOMETRIES:
            raise RuntimeError(
                f"Subject-cluster geometry set mismatch for {output_space}"
            )
        if any(
            int(row["sign_flip_repetitions"]) < 20_000
            for row in subject_sensitivities
        ):
            raise RuntimeError(
                f"Subject-cluster sign-flip protocol mismatch for {output_space}"
            )
        if any(
            not 0.0 <= float(row["holm_adjusted_p"]) <= 1.0
            for row in subject_sensitivities
        ):
            raise RuntimeError(
                f"Invalid subject-cluster Holm p-value for {output_space}"
            )
        model = json.loads(
            (directory / "calibration_models_and_statistics.json").read_text(
                encoding="utf-8"
            )
        )
        if model["output_space"] != output_space:
            raise RuntimeError(f"Output-space metadata mismatch: {output_space}")
        if model["calibration_sample_count"] <= 0:
            raise RuntimeError("Calibration set is empty")
        if len(model["quality_aware_model"]["feature_names"]) != 11:
            raise RuntimeError("Quality-aware feature schema is incomplete")
        expected_ids = expected_confirmatory_condition_ids()
        scope = model.get("confirmatory_scope") or {}
        expected_scope = {
            "mask_name": "lidar_mmwave",
            "clean_excluded": True,
            "condition_count": len(expected_ids),
            "condition_ids_sha256": condition_id_bundle_sha256(expected_ids),
        }
        for key, expected_value in expected_scope.items():
            if scope.get(key) != expected_value:
                raise RuntimeError(
                    f"Confirmatory scope mismatch for {output_space}: {key}"
                )
        if set(scope.get("geometries") or ()) != EXPECTED_CONFIRMATORY_GEOMETRIES:
            raise RuntimeError(
                f"Confirmatory scope geometry mismatch for {output_space}"
            )
        if int(scope.get("recording_cluster_test_observation_count", 0)) <= 0:
            raise RuntimeError(
                f"Confirmatory test observations are missing for {output_space}"
            )
        subject_sensitivity = model.get("paired_subject_cluster_sensitivity") or {}
        if int(subject_sensitivity.get("cluster_count", 0)) < 2:
            raise RuntimeError("Subject-cluster sensitivity analysis is incomplete")
        if int(subject_sensitivity.get("repetitions", 0)) < 2000:
            raise RuntimeError("Subject-cluster bootstrap repetitions are incomplete")
        if int(subject_sensitivity.get("sign_flip_repetitions", 0)) < 20_000:
            raise RuntimeError("Subject-cluster sign-flip repetitions are incomplete")
        derived = json.loads(
            (derived_directory / "derived_findings.json").read_text(encoding="utf-8")
        )
        if derived.get("fusion_condition_count") != 240:
            raise RuntimeError(f"Fusion comparison count mismatch: {output_space}")
        if derived.get("keep_vs_drop_comparison_count") != 80:
            raise RuntimeError(f"Keep/drop comparison count mismatch: {output_space}")
        if derived.get("quality_vs_pooled_condition_count") != 240:
            raise RuntimeError(f"Calibration comparison count mismatch: {output_space}")
        for image_name in (
            "fusion_gain_over_best_unimodal.png",
            "retain_degraded_vs_drop_sensor.png",
            "quality_aware_nll_gain_heatmap.png",
            "matched_degradation_robustness_curves.png",
        ):
            if (derived_directory / image_name).stat().st_size < 10_000:
                raise RuntimeError(
                    f"Derived figure is unexpectedly small: {output_space}/{image_name}"
                )
        summaries[output_space] = {
            "metric_rows": len(metrics),
            "per_action_rows": len(recalls),
            "robustness_auc_rows": len(auc_rows),
            "bootstrap_rows": len(bootstraps),
            "subject_sensitivity_rows": len(subject_sensitivities),
            "calibration_sample_count": model["calibration_sample_count"],
            "confirmatory_condition_count": scope["condition_count"],
            "formal_frame_count": sample_count,
            "derived_fusion_comparisons": derived["fusion_condition_count"],
            "derived_keep_drop_comparisons": derived[
                "keep_vs_drop_comparison_count"
            ],
        }
    return summaries, [
        "strict27_and_conditioned7_complete",
        "analysis_table_dimensions",
        "temperature_scaling_preserves_predictions",
        "paired_cluster_bootstrap_2000_repetitions",
        "paired_cluster_sign_flip_20000_repetitions",
        "subject_cluster_sensitivity_analysis",
        "holm_adjusted_p_values_valid",
        "quality_feature_schema_complete",
        "decision_oriented_derived_tables_and_figures",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    baseline = None
    baseline_checks = []
    if arguments.baseline_dir is not None:
        baseline, baseline_checks = audit_clean_baseline(arguments.baseline_dir)
    inference, inference_checks = audit_inference(arguments.inference_dir)
    sample_count = int(inference["manifest"]["expected_sample_count"])
    analysis, analysis_checks = audit_analysis(arguments.analysis_dir, sample_count)
    report = {
        "status": "PASS",
        "experiment": "formal_multimodal_lower_limb_calibration",
        "condition_count": len(formal_conditions()),
        "sample_count": sample_count,
        "calibration_subject_count": inference["calibration_subject_count"],
        "test_subject_count": inference["test_subject_count"],
        "checks": baseline_checks + inference_checks + analysis_checks,
        "clean_baseline_gate_included": baseline is not None,
        "clean_baseline_gate": baseline,
        "analysis": analysis,
        "checkpoint_sha256": inference["manifest"]["checkpoint_sha256"],
    }
    output = arguments.output or arguments.analysis_dir / "formal_multimodal_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
