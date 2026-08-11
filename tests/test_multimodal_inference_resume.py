import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_multimodal_inference import (
    RUN_SCHEMA_VERSION,
    aligned_data_fingerprint,
    validate_resume_manifest,
)


class MultimodalInferenceResumeTest(unittest.TestCase):
    def test_resume_requires_exact_run_signature(self):
        signature = {"checkpoint_sha256": "abc", "batch_size": 8}
        validate_resume_manifest(
            {"schema_version": RUN_SCHEMA_VERSION, "run_signature": signature},
            signature,
        )
        with self.assertRaises(RuntimeError):
            validate_resume_manifest(
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "run_signature": {"checkpoint_sha256": "different", "batch_size": 8},
                },
                signature,
            )

    def test_alignment_fingerprint_changes_with_sample_or_file_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lidar = root / "lidar.bin"
            mmwave = root / "mmwave.bin"
            lidar.write_bytes(b"123")
            mmwave.write_bytes(b"4567")
            frame = SimpleNamespace(
                key=SimpleNamespace(sample_id="E01/S01/A01/frame001"),
                lidar_path=lidar,
                mmwave_path=mmwave,
            )
            first = aligned_data_fingerprint([frame])
            lidar.write_bytes(b"12345")
            second = aligned_data_fingerprint([frame])
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
