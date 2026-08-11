import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodal_protocol import formal_conditions
from audit_multimodal_formal_results import (
    condition_id_bundle_sha256,
    expected_confirmatory_condition_ids,
)


class MultimodalProtocolTest(unittest.TestCase):
    def test_formal_matrix_is_unique_and_has_expected_size(self) -> None:
        conditions = formal_conditions()
        self.assertEqual(len(conditions), 323)
        self.assertEqual(len({condition.condition_id for condition in conditions}), 323)

    def test_clean_baselines_are_not_repeated_across_seeds(self) -> None:
        conditions = formal_conditions()
        clean_fusion = [
            condition
            for condition in conditions
            if condition.mask_name == "lidar_mmwave"
            and condition.lidar_drop_rate == 0.0
            and condition.mmwave_drop_rate == 0.0
        ]
        self.assertEqual(len(clean_fusion), 1)

    def test_confirmatory_scope_contains_only_240_degraded_fusion_conditions(self) -> None:
        condition_ids = expected_confirmatory_condition_ids()
        self.assertEqual(len(condition_ids), 240)
        self.assertEqual(len(condition_ids), len(set(condition_ids)))
        self.assertEqual(len(condition_id_bundle_sha256(condition_ids)), 64)
        self.assertFalse(any("__clean__" in value for value in condition_ids))


if __name__ == "__main__":
    unittest.main()
