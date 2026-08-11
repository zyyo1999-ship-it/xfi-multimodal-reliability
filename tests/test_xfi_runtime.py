import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xfi_runtime import (
    MMWAVE_ONLY_SPECIALIZATION,
    require_multimodal_checkpoint,
)


class XFiRuntimeTest(unittest.TestCase):
    def test_complete_checkpoint_is_allowed_for_multimodal_inference(self):
        require_multimodal_checkpoint({"specialization": "official-full-checkpoint"})
        require_multimodal_checkpoint({})

    def test_mmwave_specialization_is_rejected_for_multimodal_inference(self):
        with self.assertRaises(RuntimeError):
            require_multimodal_checkpoint(
                {"specialization": MMWAVE_ONLY_SPECIALIZATION}
            )


if __name__ == "__main__":
    unittest.main()
