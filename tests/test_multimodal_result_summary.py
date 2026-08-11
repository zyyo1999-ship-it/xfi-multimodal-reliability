import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from summarize_multimodal_formal_results import build_summary, render_markdown


METHODS = (
    "uncalibrated",
    "clean_global_ts",
    "pooled_global_ts",
    "severity_oracle_ts",
    "quality_aware_ts",
)


class MultimodalResultSummaryTests(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def make_result_tree(self, root: Path) -> None:
        analysis = root / "analysis"
        analysis.mkdir(parents=True)
        audit = {
            "status": "PASS",
            "experiment": "formal_multimodal_lower_limb_calibration",
            "condition_count": 323,
            "sample_count": 12345,
            "calibration_subject_count": 4,
            "test_subject_count": 4,
            "checkpoint_sha256": "a" * 64,
            "clean_baseline_gate_included": True,
            "clean_baseline_gate": {"status": "PASS", "sample_count": 54433},
        }
        (analysis / "formal_multimodal_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        inference = root / "inference"
        inference.mkdir(parents=True)
        (inference / "data_alignment_audit.json").write_text(
            json.dumps(
                {
                    "recording_count": 216,
                    "subject_count": 40,
                    "action_count": 7,
                }
            ),
            encoding="utf-8",
        )
        (inference / "inference_manifest.json").write_text(
            json.dumps(
                {
                    "data_audit_identity": {
                        "sha256": "b" * 64,
                        "aligned_pair_content_sha256": "c" * 64,
                        "archive_provenance_bundle_sha256": "d" * 64,
                    }
                }
            ),
            encoding="utf-8",
        )
        for output_space in ("strict27", "conditioned7"):
            directory = analysis / output_space
            rows = []
            for mask in ("lidar_mmwave", "lidar_only", "mmwave_only"):
                for method in METHODS:
                    rows.append(self.metric_row("clean", mask, "clean", method, output_space, 0.8, 0.4))
            for condition_id, accuracy, base_nll in (
                ("degraded_a", 0.7, 0.8),
                ("degraded_b", 0.5, 1.2),
            ):
                for method in METHODS:
                    nll = base_nll
                    if method == "pooled_global_ts":
                        nll -= 0.1
                    if method == "quality_aware_ts":
                        nll -= 0.2
                    rows.append(
                        self.metric_row(
                            condition_id,
                            "lidar_mmwave",
                            "uniform",
                            method,
                            output_space,
                            accuracy,
                            nll,
                        )
                    )
            self.write_csv(directory / "metrics_by_condition_method.csv", rows)
            self.write_csv(
                directory / "robustness_auc_by_seed.csv",
                [
                    {
                        "geometry": "uniform",
                        "corruption_seed": 7,
                        "method": method,
                        "path": "joint_equal_degradation",
                        "severity_max": 0.9,
                        "point_count": 5,
                        "accuracy_robustness_auc": 0.7,
                        "macro_f1_robustness_auc": 0.68,
                    }
                    for method in METHODS
                ],
            )
            statistics = {
                "paired_cluster_bootstrap": {
                    "qa_minus_pooled_nll": -0.1,
                    "sign_flip_p": 0.01,
                },
                "geometry_stratified_bootstrap_holm": [
                    {"geometry": "uniform", "holm_adjusted_p": 0.02}
                ],
                "paired_subject_cluster_sensitivity": {
                    "qa_minus_pooled_nll": -0.09,
                    "sign_flip_p": 0.02,
                    "cluster_count": 16,
                },
                "geometry_stratified_subject_cluster_sensitivity_holm": [
                    {"geometry": "uniform", "holm_adjusted_p": 0.04}
                ],
            }
            (directory / "calibration_models_and_statistics.json").write_text(
                json.dumps(statistics), encoding="utf-8"
            )

    @staticmethod
    def metric_row(
        condition_id: str,
        mask: str,
        geometry: str,
        method: str,
        output_space: str,
        accuracy: float,
        nll: float,
    ) -> dict:
        return {
            "condition_id": condition_id,
            "mask_name": mask,
            "geometry": geometry,
            "corruption_seed": 7,
            "lidar_drop_rate": 0.5 if geometry != "clean" else 0.0,
            "mmwave_drop_rate": 0.5 if geometry != "clean" else 0.0,
            "output_space": output_space,
            "method": method,
            "sample_count": 100,
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.01,
            "nll": nll,
            "brier": nll / 2,
            "ece": nll / 4,
            "adaptive_ece": nll / 5,
            "mce": nll / 3,
            "mean_confidence": 0.7,
            "confidence_accuracy_gap": 0.1,
            "aurc": 0.2,
        }

    def test_builds_formal_summary_and_marks_quality_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_result_tree(root)
            summary = build_summary(root)
            self.assertTrue(summary["claim_permitted"])
            self.assertEqual(summary["target_recording_count"], 216)
            self.assertEqual(summary["target_subject_count"], 40)
            self.assertEqual(summary["target_action_count"], 7)
            self.assertEqual(summary["aligned_pair_content_sha256"], "c" * 64)
            self.assertIn(
                "aurc",
                summary["output_spaces"]["strict27"][
                    "clean_uncalibrated_by_mask"
                ]["lidar_mmwave"],
            )
            strict = summary["output_spaces"]["strict27"]
            self.assertEqual(
                strict["worst_uncalibrated_fusion_condition"]["condition_id"],
                "degraded_b",
            )
            comparison = strict["quality_aware_vs_pooled"]
            self.assertAlmostEqual(comparison["mean_quality_minus_pooled"]["nll"], -0.1)
            self.assertEqual(comparison["quality_aware_wins"]["nll"], 2)
            markdown = render_markdown(summary)
            self.assertIn("真实正式实验", markdown)
            self.assertIn("Claim boundaries", markdown)

    def test_accepts_inference_provenance_key_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_result_tree(root)
            manifest_path = root / "inference" / "inference_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = manifest["data_audit_identity"]
            identity["provenance_bundle_sha256"] = identity.pop(
                "archive_provenance_bundle_sha256"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = build_summary(root)

            self.assertEqual(summary["archive_provenance_bundle_sha256"], "d" * 64)

    def test_nonformal_audit_is_never_promoted_to_research_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_result_tree(root)
            audit_path = root / "analysis" / "formal_multimodal_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["sample_count"] = 28
            audit["clean_baseline_gate_included"] = False
            audit["clean_baseline_gate"] = None
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            summary = build_summary(root)
            self.assertFalse(summary["claim_permitted"])
            self.assertEqual(summary["evidence_status"], "software_smoke_only")
            self.assertIn("严禁作为实证结论", render_markdown(summary))


if __name__ == "__main__":
    unittest.main()
