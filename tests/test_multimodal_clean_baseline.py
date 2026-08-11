import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_multimodal_clean_baseline import evaluate_clean_baselines
from multimodal_protocol import clean_baseline_conditions


class MultimodalCleanBaselineTest(unittest.TestCase):
    def test_perfect_aligned_baselines_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            inference_dir = Path(directory)
            condition_dir = inference_dir / "conditions"
            condition_dir.mkdir(parents=True)
            targets = np.arange(27, dtype=np.int16)
            logits = np.full((27, 27), -5.0, dtype=np.float32)
            logits[np.arange(27), targets] = 5.0
            sample_ids = np.asarray([f"sample-{index}" for index in range(27)])
            actions = np.asarray([f"A{index + 1:02d}" for index in range(27)])
            for condition in clean_baseline_conditions():
                np.savez_compressed(
                    condition_dir / f"{condition.condition_id}.npz",
                    logits=logits,
                    targets=targets,
                    sample_ids=sample_ids,
                    actions=actions,
                )
            report = evaluate_clean_baselines(
                inference_dir,
                expected_accuracies={
                    "lidar_mmwave": 1.0,
                    "lidar_only": 1.0,
                    "mmwave_only": 1.0,
                },
                tolerance=0.0,
                expected_frame_count=27,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["class_count"], 27)

    def test_cross_mask_sample_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            inference_dir = Path(directory)
            condition_dir = inference_dir / "conditions"
            condition_dir.mkdir(parents=True)
            targets = np.arange(27, dtype=np.int16)
            logits = np.eye(27, dtype=np.float32)
            actions = np.asarray([f"A{index + 1:02d}" for index in range(27)])
            for index, condition in enumerate(clean_baseline_conditions()):
                sample_ids = np.asarray(
                    [f"sample-{item + (1 if index == 1 else 0)}" for item in range(27)]
                )
                np.savez_compressed(
                    condition_dir / f"{condition.condition_id}.npz",
                    logits=logits,
                    targets=targets,
                    sample_ids=sample_ids,
                    actions=actions,
                )
            with self.assertRaises(RuntimeError):
                evaluate_clean_baselines(
                    inference_dir,
                    expected_frame_count=27,
                )


if __name__ == "__main__":
    unittest.main()
