import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lower_limb_protocol import (
    CALIBRATION_SUBJECTS,
    LOWER_LIMB_ACTIONS,
    TEST_SUBJECTS,
    lower_limb_label,
    partition_for_subject,
    summarize_protocol,
)
from mmwave_dataset import discover_validation_frames


class LowerLimbProtocolTest(unittest.TestCase):
    def test_frozen_subject_partitions_are_disjoint_and_complete(self) -> None:
        self.assertEqual(len(CALIBRATION_SUBJECTS), 20)
        self.assertEqual(len(TEST_SUBJECTS), 20)
        self.assertFalse(set(CALIBRATION_SUBJECTS) & set(TEST_SUBJECTS))
        self.assertEqual(partition_for_subject("S01"), "calibration")
        self.assertEqual(partition_for_subject("S40"), "test")

    def test_lower_limb_labels_are_contiguous(self) -> None:
        self.assertEqual(
            [lower_limb_label(action) for action in LOWER_LIMB_ACTIONS],
            list(range(7)),
        )

    def test_real_dataset_protocol_counts(self) -> None:
        data_root = PROJECT_ROOT / "data/full/filtered_mmwave"
        if not data_root.is_dir():
            self.skipTest("Full filtered-mmWave data are not available")
        summary = summarize_protocol(discover_validation_frames(data_root))
        self.assertEqual(summary.selected_frame_count, 15315)
        self.assertEqual(summary.calibration_frame_count, 8312)
        self.assertEqual(summary.test_frame_count, 7003)
        self.assertTrue(
            all(value >= 3 for value in summary.test_recordings_per_action.values())
        )


if __name__ == "__main__":
    unittest.main()
