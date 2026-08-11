"""Auditable synchronized LiDAR+mmWave loading for the formal MM-Fi study."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .corruptions import azimuth_sector_dropout, point_dropout
    from .lower_limb_protocol import partition_for_subject
    from .mmwave_dataset import official_validation_form, subject_environment
except ImportError:
    from corruptions import azimuth_sector_dropout, point_dropout
    from lower_limb_protocol import partition_for_subject
    from mmwave_dataset import official_validation_form, subject_environment


FRAME_PATTERN = re.compile(r"frame(\d+)", re.IGNORECASE)
POINT_WIDTHS = {"lidar": 3, "mmwave": 5}
FULL_CLASS_COUNT = 27


@dataclass(frozen=True, order=True)
class FrameKey:
    scene: str
    subject: str
    action: str
    frame_index: int

    @property
    def sample_id(self) -> str:
        return (
            f"{self.scene}/{self.subject}/{self.action}/"
            f"frame{self.frame_index:03d}"
        )


@dataclass(frozen=True)
class AlignedPointFrame:
    key: FrameKey
    lidar_path: Path
    mmwave_path: Path

    @property
    def full_label(self) -> int:
        label = int(self.key.action[1:]) - 1
        if not 0 <= label < FULL_CLASS_COUNT:
            raise ValueError(f"Invalid MM-Fi action: {self.key.action}")
        return label


@dataclass(frozen=True)
class PointQuality:
    point_count: int
    azimuth_occupancy: float
    range_occupancy: float


def frame_index_from_path(path: Path) -> int:
    match = FRAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Frame filename does not contain frame<number>: {path.name}")
    return int(match.group(1))


def index_frame_directory(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing modality directory: {directory}")
    indexed: dict[int, Path] = {}
    for path in sorted(candidate for candidate in directory.iterdir() if candidate.is_file()):
        index = frame_index_from_path(path)
        if index in indexed:
            raise ValueError(f"Duplicate frame index {index} in {directory}")
        indexed[index] = path
    if not indexed:
        raise ValueError(f"No frame files found in {directory}")
    return indexed


def discover_aligned_point_frames(
    data_root: Path,
    data_form: Mapping[str, Sequence[str]] | None = None,
    actions: set[str] | None = None,
    strict_alignment: bool = True,
) -> list[AlignedPointFrame]:
    """Build a synchronized manifest using explicit frame IDs.

    The official random-split validation form is the default. A strict audit
    rejects recordings whose LiDAR and mmWave frame-ID sets differ, preventing
    silent positional misalignment.
    """
    data_root = Path(data_root)
    if data_form is None:
        data_form = official_validation_form(random_seed=0, train_ratio=0.8)

    frames: list[AlignedPointFrame] = []
    for subject, subject_actions in data_form.items():
        scene = subject_environment(subject)
        for action in subject_actions:
            if actions is not None and action not in actions:
                continue
            recording_root = data_root / scene / subject / action
            lidar = index_frame_directory(recording_root / "lidar")
            mmwave = index_frame_directory(recording_root / "mmwave")
            lidar_ids = set(lidar)
            mmwave_ids = set(mmwave)
            if strict_alignment and lidar_ids != mmwave_ids:
                only_lidar = sorted(lidar_ids - mmwave_ids)[:8]
                only_mmwave = sorted(mmwave_ids - lidar_ids)[:8]
                raise ValueError(
                    f"Frame-ID mismatch in {scene}/{subject}/{action}: "
                    f"lidar={len(lidar_ids)}, mmwave={len(mmwave_ids)}, "
                    f"only_lidar={only_lidar}, only_mmwave={only_mmwave}"
                )
            for frame_index in sorted(lidar_ids & mmwave_ids):
                frames.append(
                    AlignedPointFrame(
                        key=FrameKey(scene, subject, action, frame_index),
                        lidar_path=lidar[frame_index],
                        mmwave_path=mmwave[frame_index],
                    )
                )
    if not frames:
        raise ValueError("No synchronized LiDAR+mmWave frames were discovered")
    sample_ids = [frame.key.sample_id for frame in frames]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("The synchronized manifest contains duplicate sample IDs")
    return frames


def read_point_cloud(path: Path, width: int) -> np.ndarray:
    path = Path(path)
    if width <= 0:
        raise ValueError("point width must be positive")
    byte_count = path.stat().st_size
    row_bytes = np.dtype(np.float64).itemsize * width
    if byte_count == 0 or byte_count % row_bytes:
        raise ValueError(
            f"Invalid point-cloud byte count for width {width}: {path} ({byte_count})"
        )
    points = np.fromfile(path, dtype=np.float64).reshape(-1, width)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite point-cloud value in {path}")
    return points


class PointCloudCache:
    """Small LRU cache shared across repeated corruption conditions."""

    def __init__(self, max_bytes: int = 0) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = int(max_bytes)
        self.current_bytes = 0
        self._items: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()

    def get(self, path: Path, width: int) -> np.ndarray:
        if self.max_bytes == 0:
            return read_point_cloud(path, width)
        key = (Path(path), width)
        cached = self._items.pop(key, None)
        if cached is not None:
            self._items[key] = cached
            return cached
        points = read_point_cloud(path, width)
        if points.nbytes <= self.max_bytes:
            while self._items and self.current_bytes + points.nbytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False)
                self.current_bytes -= evicted.nbytes
            self._items[key] = points
            self.current_bytes += points.nbytes
        return points


def stable_sample_seed(
    sample_id: str,
    modality: str,
    experiment_seed: int,
) -> int:
    payload = f"{sample_id}|{modality}|{experiment_seed}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def apply_point_corruption(
    points: np.ndarray,
    geometry: str,
    drop_rate: float,
    seed: int,
) -> np.ndarray:
    if geometry == "clean" or drop_rate == 0.0:
        return points.copy()
    if geometry == "uniform":
        return point_dropout(points, drop_rate, seed)
    if geometry == "azimuth_sector":
        return azimuth_sector_dropout(points, drop_rate, seed)
    raise ValueError(f"Unknown corruption geometry: {geometry}")


def point_quality(
    points: np.ndarray,
    max_range: float,
    azimuth_bin_count: int = 12,
    range_bin_count: int = 8,
) -> PointQuality:
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape [N, D] with D >= 2")
    if max_range <= 0.0 or azimuth_bin_count <= 0 or range_bin_count <= 0:
        raise ValueError("quality histogram settings must be positive")
    if points.shape[0] == 0:
        return PointQuality(0, 0.0, 0.0)

    azimuth = np.arctan2(points[:, 1], points[:, 0])
    azimuth_histogram, _ = np.histogram(
        azimuth, bins=azimuth_bin_count, range=(-np.pi, np.pi)
    )
    planar_range = np.linalg.norm(points[:, :2], axis=1)
    range_histogram, _ = np.histogram(
        np.clip(planar_range, 0.0, max_range),
        bins=range_bin_count,
        range=(0.0, max_range),
    )
    return PointQuality(
        point_count=int(points.shape[0]),
        azimuth_occupancy=float(np.count_nonzero(azimuth_histogram) / azimuth_bin_count),
        range_occupancy=float(np.count_nonzero(range_histogram) / range_bin_count),
    )


class AlignedPointDataset:
    """Load and deterministically corrupt one aligned point-sensor condition."""

    def __init__(
        self,
        frames: Sequence[AlignedPointFrame],
        geometry: str,
        experiment_seed: int,
        lidar_drop_rate: float,
        mmwave_drop_rate: float,
        active_modalities: Iterable[str],
        cache: PointCloudCache | None = None,
    ) -> None:
        self.frames = tuple(frames)
        self.geometry = geometry
        self.experiment_seed = int(experiment_seed)
        self.lidar_drop_rate = float(lidar_drop_rate)
        self.mmwave_drop_rate = float(mmwave_drop_rate)
        self.active_modalities = frozenset(active_modalities)
        if not self.active_modalities or not self.active_modalities <= set(POINT_WIDTHS):
            raise ValueError("active_modalities must contain lidar and/or mmwave")
        for rate in (self.lidar_drop_rate, self.mmwave_drop_rate):
            if not 0.0 <= rate < 1.0:
                raise ValueError("formal drop rates must be in [0, 1)")
        self.cache = cache or PointCloudCache(max_bytes=0)

    def __len__(self) -> int:
        return len(self.frames)

    def _load_modality(
        self,
        frame: AlignedPointFrame,
        modality: str,
        drop_rate: float,
    ) -> np.ndarray:
        if modality not in self.active_modalities:
            return np.zeros((1, POINT_WIDTHS[modality]), dtype=np.float32)
        path = frame.lidar_path if modality == "lidar" else frame.mmwave_path
        points = self.cache.get(path, POINT_WIDTHS[modality])
        seed = stable_sample_seed(
            frame.key.sample_id, modality, self.experiment_seed
        )
        return apply_point_corruption(
            points, self.geometry, drop_rate, seed
        ).astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        lidar = self._load_modality(frame, "lidar", self.lidar_drop_rate)
        mmwave = self._load_modality(frame, "mmwave", self.mmwave_drop_rate)
        lidar_quality = (
            point_quality(lidar, max_range=15.0)
            if "lidar" in self.active_modalities
            else PointQuality(0, 0.0, 0.0)
        )
        mmwave_quality = (
            point_quality(mmwave, max_range=10.0)
            if "mmwave" in self.active_modalities
            else PointQuality(0, 0.0, 0.0)
        )
        return {
            "sample_id": frame.key.sample_id,
            "scene": frame.key.scene,
            "subject": frame.key.subject,
            "action": frame.key.action,
            "frame_index": frame.key.frame_index,
            "partition": partition_for_subject(frame.key.subject),
            "target": frame.full_label,
            "lidar": lidar,
            "mmwave": mmwave,
            "lidar_quality": lidar_quality,
            "mmwave_quality": mmwave_quality,
        }


def collate_aligned_points(batch: Sequence[dict]) -> dict:
    """Match X-Fi's batch-first zero-padding contract deterministically."""
    import torch

    if not batch:
        raise ValueError("cannot collate an empty batch")
    lidar = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(item["lidar"]) for item in batch], batch_first=True
    )
    mmwave = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(item["mmwave"]) for item in batch], batch_first=True
    )
    return {
        "lidar": lidar,
        "mmwave": mmwave,
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
        "scenes": [item["scene"] for item in batch],
        "subjects": [item["subject"] for item in batch],
        "actions": [item["action"] for item in batch],
        "frame_indices": np.asarray(
            [item["frame_index"] for item in batch], dtype=np.int32
        ),
        "partitions": [item["partition"] for item in batch],
        "lidar_point_counts": np.asarray(
            [item["lidar_quality"].point_count for item in batch], dtype=np.int32
        ),
        "mmwave_point_counts": np.asarray(
            [item["mmwave_quality"].point_count for item in batch], dtype=np.int32
        ),
        "lidar_azimuth_occupancy": np.asarray(
            [item["lidar_quality"].azimuth_occupancy for item in batch],
            dtype=np.float32,
        ),
        "mmwave_azimuth_occupancy": np.asarray(
            [item["mmwave_quality"].azimuth_occupancy for item in batch],
            dtype=np.float32,
        ),
        "lidar_range_occupancy": np.asarray(
            [item["lidar_quality"].range_occupancy for item in batch],
            dtype=np.float32,
        ),
        "mmwave_range_occupancy": np.asarray(
            [item["mmwave_quality"].range_occupancy for item in batch],
            dtype=np.float32,
        ),
    }
