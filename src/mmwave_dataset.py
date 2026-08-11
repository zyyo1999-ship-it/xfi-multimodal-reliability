"""MM-Fi filtered-mmWave dataset utilities for the robustness pilot.

The split logic mirrors the official X-Fi MMFi-HAR ``random_split`` protocol:
for each action, 32 subjects are assigned to training and 8 to validation,
using a deterministic action-specific random seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from corruptions import azimuth_sector_dropout, point_dropout


ACTIONS = tuple(f"A{index:02d}" for index in range(1, 28))
SUBJECTS = tuple(f"S{index:02d}" for index in range(1, 41))
ACTION_TO_LABEL = {action: index for index, action in enumerate(ACTIONS)}
ACTION_NAMES = {
    "A01": "Stretching and relaxing",
    "A02": "Chest expansion (horizontal)",
    "A03": "Chest expansion (vertical)",
    "A04": "Twist (left)",
    "A05": "Twist (right)",
    "A06": "Mark time",
    "A07": "Limb extension (left)",
    "A08": "Limb extension (right)",
    "A09": "Lunge (toward left-front)",
    "A10": "Lunge (toward right-front)",
    "A11": "Limb extension (both)",
    "A12": "Squat",
    "A13": "Raising hand (left)",
    "A14": "Raising hand (right)",
    "A15": "Lunge (toward left side)",
    "A16": "Lunge (toward right side)",
    "A17": "Waving hand (left)",
    "A18": "Waving hand (right)",
    "A19": "Picking up things",
    "A20": "Throwing (toward left side)",
    "A21": "Throwing (toward right side)",
    "A22": "Kicking (toward left side)",
    "A23": "Kicking (toward right side)",
    "A24": "Body extension (left)",
    "A25": "Body extension (right)",
    "A26": "Jumping up",
    "A27": "Bowing",
}
VALUES_PER_POINT = 5


def subject_environment(subject: str) -> str:
    """Map an MM-Fi subject identifier to its recording environment."""
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown MM-Fi subject: {subject}")
    subject_number = int(subject[1:])
    return f"E{((subject_number - 1) // 10) + 1:02d}"


def official_random_split(
    random_seed: int = 0,
    train_ratio: float = 0.8,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Reproduce the action-wise subject split used by official X-Fi code."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    train_subjects: dict[str, tuple[str, ...]] = {}
    validation_subjects: dict[str, tuple[str, ...]] = {}
    split_index = floor(train_ratio * len(SUBJECTS))

    for action_offset, action in enumerate(ACTIONS):
        generator = np.random.RandomState(random_seed + action_offset)
        permutation = generator.permutation(len(SUBJECTS))
        train_subjects[action] = tuple(SUBJECTS[index] for index in permutation[:split_index])
        validation_subjects[action] = tuple(SUBJECTS[index] for index in permutation[split_index:])

    return train_subjects, validation_subjects


def official_validation_form(
    random_seed: int = 0,
    train_ratio: float = 0.8,
) -> dict[str, tuple[str, ...]]:
    """Reproduce the validation ``data_form`` insertion order from X-Fi."""
    _, validation_subjects = official_random_split(random_seed, train_ratio)
    validation_sets = {
        action: set(subjects) for action, subjects in validation_subjects.items()
    }
    actions_by_subject: dict[str, list[str]] = {}

    # The official decoder loops over actions first and subjects second. A
    # subject key is inserted when it first appears in a validation subset.
    for action in ACTIONS:
        for subject in SUBJECTS:
            if subject in validation_sets[action]:
                actions_by_subject.setdefault(subject, []).append(action)

    return {
        subject: tuple(actions) for subject, actions in actions_by_subject.items()
    }


@dataclass(frozen=True)
class MmWaveFrame:
    path: Path
    environment: str
    subject: str
    action: str
    label: int

    @property
    def sample_id(self) -> str:
        return f"{self.environment}/{self.subject}/{self.action}/{self.path.name}"


def discover_validation_frames(
    data_root: Path,
    random_seed: int = 0,
    train_ratio: float = 0.8,
    frame_stride: int = 1,
    max_samples: int | None = None,
) -> list[MmWaveFrame]:
    """Enumerate validation frames from an extracted filtered-mmWave archive."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")

    validation_form = official_validation_form(random_seed, train_ratio)
    frames: list[MmWaveFrame] = []

    for subject, actions in validation_form.items():
        environment = subject_environment(subject)
        for action in actions:
            label = ACTION_TO_LABEL[action]
            action_directory = data_root / environment / subject / action
            if not action_directory.is_dir():
                raise FileNotFoundError(
                    f"Missing MM-Fi action directory: {action_directory}"
                )

            paths = sorted(action_directory.glob("frame*.bin"))[::frame_stride]
            if not paths:
                raise FileNotFoundError(f"No frame*.bin files in {action_directory}")

            frames.extend(
                MmWaveFrame(
                    path=path,
                    environment=environment,
                    subject=subject,
                    action=action,
                    label=label,
                )
                for path in paths
            )

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        frames = frames[:max_samples]

    return frames


def load_mmwave_frame(path: Path) -> np.ndarray:
    """Read one official MM-Fi filtered-mmWave frame as an N x 5 array."""
    values = np.fromfile(path, dtype=np.float64)
    if values.size == 0 or values.size % VALUES_PER_POINT != 0:
        raise ValueError(f"Invalid MM-Fi filtered-mmWave frame: {path}")
    return values.reshape(-1, VALUES_PER_POINT).astype(np.float32)


def sample_seed(experiment_seed: int, sample_id: str) -> int:
    """Derive a stable dropout seed from the experiment and sample identity."""
    payload = f"{experiment_seed}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class FilteredMmWaveDataset(Dataset):
    """Validation dataset with deterministic point-dropout corruption."""

    def __init__(
        self,
        frames: Iterable[MmWaveFrame],
        drop_rate: float = 0.0,
        experiment_seed: int = 7,
        corruption: str = "uniform",
    ) -> None:
        self.frames = tuple(frames)
        self.drop_rate = drop_rate
        self.experiment_seed = experiment_seed
        self.corruption = corruption
        if not self.frames:
            raise ValueError("At least one frame is required")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(
                "drop_rate must be in [0.0, 1.0); complete sensor absence "
                "requires a separate missing-modality control"
            )
        if corruption not in {"uniform", "azimuth_sector"}:
            raise ValueError(f"Unknown corruption: {corruption}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        clean_points = load_mmwave_frame(frame.path)
        corruption_function = (
            point_dropout
            if self.corruption == "uniform"
            else azimuth_sector_dropout
        )
        degraded_points = corruption_function(
            clean_points,
            drop_rate=self.drop_rate,
            seed=sample_seed(self.experiment_seed, frame.sample_id),
        )
        return {
            "points": torch.from_numpy(degraded_points),
            "label": frame.label,
            "sample_id": frame.sample_id,
            "clean_point_count": int(clean_points.shape[0]),
            "remaining_point_count": int(degraded_points.shape[0]),
        }


def collate_mmwave_batch(samples: list[dict]) -> dict:
    """Pad variable-length point clouds exactly as the official loader does."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    points = pad_sequence(
        [sample["points"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    )
    return {
        "points": points,
        "labels": torch.tensor(
            [sample["label"] for sample in samples], dtype=torch.long
        ),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "clean_point_counts": [sample["clean_point_count"] for sample in samples],
        "remaining_point_counts": [
            sample["remaining_point_count"] for sample in samples
        ],
    }
