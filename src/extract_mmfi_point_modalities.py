#!/usr/bin/env python3
"""Selectively extract official validation LiDAR/mmWave files from MM-Fi zips."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from lower_limb_protocol import LOWER_LIMB_ACTIONS
from mmwave_dataset import SUBJECTS, official_validation_form, subject_environment


FRAME_FILE_PATTERN = re.compile(r"frame\d+\.bin", re.IGNORECASE)


def parse_member(name: str) -> tuple[str, str, str, str, str] | None:
    parts = PurePosixPath(name).parts
    if "__MACOSX" in parts:
        return None
    for index, part in enumerate(parts):
        if part not in SUBJECTS or index + 1 >= len(parts):
            continue
        subject = part
        action = parts[index + 1]
        if not re.fullmatch(r"A\d{2}", action):
            continue
        scene = subject_environment(subject)
        if index > 0 and re.fullmatch(r"E\d{2}", parts[index - 1]):
            if parts[index - 1] != scene:
                return None
        remainder = parts[index + 2 :]
        if remainder == ("ground_truth.npy",):
            return scene, subject, action, "ground_truth", remainder[0]
        if len(remainder) != 2:
            continue
        modality, filename = remainder
        if (
            modality in {"lidar", "mmwave"}
            and FRAME_FILE_PATTERN.fullmatch(filename)
        ):
            return scene, subject, action, modality, filename
    return None


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("target", "all"), default="all")
    parser.add_argument("--environment", choices=("E01", "E02", "E03", "E04"))
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--reference-mmwave-root", type=Path)
    arguments = parser.parse_args()

    if (
        arguments.environment is not None
        and arguments.subject is not None
        and subject_environment(arguments.subject) != arguments.environment
    ):
        parser.error(
            f"{arguments.subject} belongs to "
            f"{subject_environment(arguments.subject)}, not {arguments.environment}"
        )

    validation_form = official_validation_form(random_seed=0, train_ratio=0.8)
    allowed_actions = set(LOWER_LIMB_ACTIONS) if arguments.scope == "target" else None
    allowed_pairs = {
        (subject_environment(subject), subject, action)
        for subject, actions in validation_form.items()
        for action in actions
        if allowed_actions is None or action in allowed_actions
    }
    if arguments.environment is not None:
        allowed_pairs = {
            pair for pair in allowed_pairs if pair[0] == arguments.environment
        }
    if arguments.subject is not None:
        allowed_pairs = {
            pair for pair in allowed_pairs if pair[1] == arguments.subject
        }

    allowed_frame_names: dict[tuple[str, str, str], set[str]] | None = None
    if arguments.reference_mmwave_root is not None:
        allowed_frame_names = {}
        for scene, subject, action in allowed_pairs:
            reference_directory = (
                arguments.reference_mmwave_root / scene / subject / action
            )
            if not reference_directory.is_dir():
                raise FileNotFoundError(
                    f"Missing reference mmWave directory: {reference_directory}"
                )
            names = {
                path.name
                for path in reference_directory.glob("frame*.bin")
                if path.is_file()
            }
            if not names:
                raise FileNotFoundError(
                    f"No reference frame*.bin files in {reference_directory}"
                )
            allowed_frame_names[(scene, subject, action)] = names

    arguments.output_root.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    bytes_written = 0
    with zipfile.ZipFile(arguments.archive) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parsed = parse_member(info.filename)
            if parsed is None:
                continue
            scene, subject, action, modality, filename = parsed
            if (scene, subject, action) not in allowed_pairs:
                continue
            if (
                modality in {"lidar", "mmwave"}
                and allowed_frame_names is not None
                and filename
                not in allowed_frame_names[(scene, subject, action)]
            ):
                continue
            destination = (
                arguments.output_root / scene / subject / action / modality / filename
                if modality in {"lidar", "mmwave"}
                else arguments.output_root / scene / subject / action / filename
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size == info.file_size:
                counts[f"{modality}_skipped"] += 1
                continue
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError(f"Incomplete extraction: {info.filename}")
            os.replace(temporary, destination)
            counts[modality] += 1
            bytes_written += info.file_size

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(arguments.archive.resolve()),
        "output_root": str(arguments.output_root.resolve()),
        "scope": arguments.scope,
        "environment": arguments.environment,
        "subject": arguments.subject,
        "reference_mmwave_root": (
            str(arguments.reference_mmwave_root.resolve())
            if arguments.reference_mmwave_root is not None
            else None
        ),
        "allowed_recording_count": len(allowed_pairs),
        "counts": dict(counts),
        "bytes_written": bytes_written,
    }
    selection = arguments.subject or arguments.environment or "all"
    audit_name = f"extraction_audit_{selection}_{arguments.scope}.json"
    atomic_json(arguments.output_root / audit_name, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
