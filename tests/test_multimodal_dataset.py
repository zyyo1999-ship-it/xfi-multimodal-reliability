import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodal_dataset import (
    AlignedPointDataset,
    discover_aligned_point_frames,
    frame_index_from_path,
    point_quality,
    stable_sample_seed,
)


class MultimodalDatasetTest(unittest.TestCase):
    def _write_recording(self, root: Path, lidar_ids=(1, 2), mmwave_ids=(1, 2)) -> None:
        recording = root / "E01" / "S01" / "A06"
        (recording / "lidar").mkdir(parents=True)
        (recording / "mmwave").mkdir(parents=True)
        for frame_id in lidar_ids:
            np.arange(36, dtype=np.float64).reshape(12, 3).tofile(
                recording / "lidar" / f"frame{frame_id:03d}.bin"
            )
        for frame_id in mmwave_ids:
            np.arange(60, dtype=np.float64).reshape(12, 5).tofile(
                recording / "mmwave" / f"frame{frame_id:03d}.bin"
            )

    def test_explicit_frame_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_recording(root)
            frames = discover_aligned_point_frames(
                root, data_form={"S01": ["A06"]}
            )
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].key.sample_id, "E01/S01/A06/frame001")

    def test_frame_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_recording(root, lidar_ids=(1, 2), mmwave_ids=(1,))
            with self.assertRaisesRegex(ValueError, "Frame-ID mismatch"):
                discover_aligned_point_frames(root, data_form={"S01": ["A06"]})

    def test_condition_corruption_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_recording(root)
            frames = discover_aligned_point_frames(
                root, data_form={"S01": ["A06"]}
            )
            first = AlignedPointDataset(
                frames, "uniform", 7, 0.5, 0.5, {"lidar", "mmwave"}
            )[0]
            second = AlignedPointDataset(
                frames, "uniform", 7, 0.5, 0.5, {"lidar", "mmwave"}
            )[0]
            np.testing.assert_array_equal(first["lidar"], second["lidar"])
            np.testing.assert_array_equal(first["mmwave"], second["mmwave"])
            self.assertEqual(first["lidar"].shape[0], 6)
            self.assertEqual(first["mmwave"].shape[0], 6)

    def test_quality_occupancy_is_bounded(self) -> None:
        angles = np.linspace(-np.pi, np.pi, 24, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(24)])
        quality = point_quality(points, max_range=5.0)
        self.assertEqual(quality.point_count, 24)
        self.assertGreater(quality.azimuth_occupancy, 0.9)
        self.assertGreaterEqual(quality.range_occupancy, 0.0)
        self.assertLessEqual(quality.range_occupancy, 1.0)

    def test_stable_seed_depends_on_modality(self) -> None:
        lidar_seed = stable_sample_seed("sample", "lidar", 7)
        mmwave_seed = stable_sample_seed("sample", "mmwave", 7)
        self.assertNotEqual(lidar_seed, mmwave_seed)
        self.assertEqual(lidar_seed, stable_sample_seed("sample", "lidar", 7))

    def test_frame_parser(self) -> None:
        self.assertEqual(frame_index_from_path(Path("frame001.bin")), 1)


if __name__ == "__main__":
    unittest.main()
