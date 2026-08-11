"""Frozen protocol for the lower-limb rehabilitation calibration study.

The recognition model uses the official X-Fi validation recordings. We then
split those recordings globally by subject for post-hoc calibration and final
testing. This split prevents the calibration parameters from seeing test
subjects, but it does not redefine the original X-Fi training protocol.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from mmwave_dataset import ACTION_NAMES, SUBJECTS, MmWaveFrame


LOWER_LIMB_ACTIONS = (
    "A06",
    "A09",
    "A10",
    "A12",
    "A15",
    "A16",
    "A26",
)
LOWER_LIMB_ACTION_TO_LABEL = {
    action: index for index, action in enumerate(LOWER_LIMB_ACTIONS)
}
LOWER_LIMB_FULL_LABELS = tuple(int(action[1:]) - 1 for action in LOWER_LIMB_ACTIONS)

# Five subjects were sampled inside each of the four recording environments
# with NumPy default_rng(seed=20260802). The explicit list freezes the result
# against future random-number-library changes.
CALIBRATION_SUBJECTS = (
    "S01",
    "S02",
    "S03",
    "S06",
    "S09",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S22",
    "S23",
    "S24",
    "S27",
    "S29",
    "S33",
    "S34",
    "S35",
    "S38",
    "S39",
)
TEST_SUBJECTS = tuple(
    subject for subject in SUBJECTS if subject not in set(CALIBRATION_SUBJECTS)
)


@dataclass(frozen=True)
class ProtocolSummary:
    selected_frame_count: int
    calibration_frame_count: int
    test_frame_count: int
    observed_calibration_subject_count: int
    observed_test_subject_count: int
    frames_per_action: dict[str, int]
    calibration_recordings_per_action: dict[str, int]
    test_recordings_per_action: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "selected_frame_count": self.selected_frame_count,
            "calibration_frame_count": self.calibration_frame_count,
            "test_frame_count": self.test_frame_count,
            "observed_calibration_subject_count": (
                self.observed_calibration_subject_count
            ),
            "observed_test_subject_count": self.observed_test_subject_count,
            "frames_per_action": self.frames_per_action,
            "calibration_recordings_per_action": (
                self.calibration_recordings_per_action
            ),
            "test_recordings_per_action": self.test_recordings_per_action,
        }


def is_lower_limb_action(action: str) -> bool:
    return action in LOWER_LIMB_ACTION_TO_LABEL


def lower_limb_label(action: str) -> int:
    try:
        return LOWER_LIMB_ACTION_TO_LABEL[action]
    except KeyError as error:
        raise ValueError(f"Not a frozen lower-limb action: {action}") from error


def partition_for_subject(subject: str) -> str:
    if subject in CALIBRATION_SUBJECTS:
        return "calibration"
    if subject in TEST_SUBJECTS:
        return "test"
    raise ValueError(f"Unknown MM-Fi subject: {subject}")


def select_lower_limb_frames(frames: Iterable[MmWaveFrame]) -> list[MmWaveFrame]:
    return [frame for frame in frames if is_lower_limb_action(frame.action)]


def summarize_protocol(frames: Iterable[MmWaveFrame]) -> ProtocolSummary:
    selected = select_lower_limb_frames(frames)
    if not selected:
        raise ValueError("No lower-limb frames were found")

    calibration = [
        frame
        for frame in selected
        if partition_for_subject(frame.subject) == "calibration"
    ]
    test = [
        frame for frame in selected if partition_for_subject(frame.subject) == "test"
    ]

    def recording_counts(records: list[MmWaveFrame]) -> dict[str, int]:
        unique = {(frame.subject, frame.action) for frame in records}
        counts = Counter(action for _, action in unique)
        return {action: counts[action] for action in LOWER_LIMB_ACTIONS}

    summary = ProtocolSummary(
        selected_frame_count=len(selected),
        calibration_frame_count=len(calibration),
        test_frame_count=len(test),
        observed_calibration_subject_count=len(
            {frame.subject for frame in calibration}
        ),
        observed_test_subject_count=len({frame.subject for frame in test}),
        frames_per_action={
            action: sum(frame.action == action for frame in selected)
            for action in LOWER_LIMB_ACTIONS
        },
        calibration_recordings_per_action=recording_counts(calibration),
        test_recordings_per_action=recording_counts(test),
    )
    validate_protocol(summary)
    return summary


def validate_protocol(summary: ProtocolSummary) -> None:
    overlap = set(CALIBRATION_SUBJECTS) & set(TEST_SUBJECTS)
    if overlap:
        raise RuntimeError(f"Calibration/test subject leakage: {sorted(overlap)}")
    if set(CALIBRATION_SUBJECTS) | set(TEST_SUBJECTS) != set(SUBJECTS):
        raise RuntimeError("Frozen subject partitions do not cover all MM-Fi subjects")
    for action in LOWER_LIMB_ACTIONS:
        if summary.calibration_recordings_per_action[action] < 3:
            raise RuntimeError(f"Too few calibration recordings for {action}")
        if summary.test_recordings_per_action[action] < 3:
            raise RuntimeError(f"Too few test recordings for {action}")


def protocol_action_metadata() -> list[dict]:
    return [
        {
            "lower_limb_label": LOWER_LIMB_ACTION_TO_LABEL[action],
            "full_xfi_label": int(action[1:]) - 1,
            "action": action,
            "name": ACTION_NAMES[action],
        }
        for action in LOWER_LIMB_ACTIONS
    ]
