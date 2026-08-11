#!/usr/bin/env python3
"""Audit selectively extracted official MM-Fi LiDAR/mmWave validation data."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from lower_limb_protocol import LOWER_LIMB_ACTIONS
from mmwave_dataset import SUBJECTS, official_validation_form, subject_environment
from multimodal_dataset import POINT_WIDTHS, index_frame_directory


ENVIRONMENTS = ("E01", "E02", "E03", "E04")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_content_digest(
    digest: "hashlib._Hash", sample_id: str, content_sha256: str
) -> None:
    digest.update(sample_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(content_sha256))
    digest.update(b"\n")


def hash_provenance_bundle(provenance_root: Path) -> dict:
    expected = {f"S{index:02d}.zip.provenance.txt" for index in range(1, 41)}
    observed = {path.name for path in Path(provenance_root).glob("S??.zip.provenance.txt")}
    if observed != expected:
        raise RuntimeError(
            "Official archive provenance is incomplete: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    digest = hashlib.sha256()
    archives = []
    for name in sorted(expected):
        path = Path(provenance_root) / name
        content = path.read_bytes()
        fields = {}
        for line in content.decode("utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
        subject = name[:3]
        archive_sha256 = fields.get("observed_sha256", "")
        try:
            bytes.fromhex(archive_sha256)
            size_bytes = int(fields.get("observed_size_bytes", "0"))
        except ValueError as error:
            raise RuntimeError(f"Invalid provenance record: {path}") from error
        if (
            fields.get("subject") != subject
            or fields.get("source_kind") != "official_baidu_share"
            or len(archive_sha256) != 64
            or size_bytes <= 0
        ):
            raise RuntimeError(f"Invalid provenance fields: {path}")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\n")
        archives.append(
            {
                "subject": subject,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": size_bytes,
            }
        )
    return {
        "provenance_record_count": len(archives),
        "provenance_bundle_sha256": digest.hexdigest(),
        "archives": archives,
    }


def validate_binary_layout(path: Path, width: int) -> None:
    row_bytes = np.dtype(np.float64).itemsize * width
    byte_count = path.stat().st_size
    if byte_count == 0 or byte_count % row_bytes:
        raise RuntimeError(
            f"Invalid {width}-column float64 point cloud: {path} ({byte_count} bytes)"
        )


def audit_extracted_point_modalities(
    data_root: Path,
    scope: str = "all",
    environment: str | None = None,
    subject: str | None = None,
    expected_frame_count: int | None = None,
    reference_mmwave_root: Path | None = None,
    expected_reference_frame_count: int | None = None,
    data_form: Mapping[str, Sequence[str]] | None = None,
    hash_content: bool = False,
    provenance_root: Path | None = None,
) -> dict:
    if scope not in {"all", "target"}:
        raise ValueError("scope must be 'all' or 'target'")
    if environment is not None and environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")
    if subject is not None and subject not in SUBJECTS:
        raise ValueError(f"Unknown subject: {subject}")
    if (
        environment is not None
        and subject is not None
        and subject_environment(subject) != environment
    ):
        raise ValueError(
            f"{subject} belongs to {subject_environment(subject)}, not {environment}"
        )
    if expected_frame_count is not None and expected_frame_count <= 0:
        raise ValueError("expected_frame_count must be positive")
    if (
        expected_reference_frame_count is not None
        and expected_reference_frame_count <= 0
    ):
        raise ValueError("expected_reference_frame_count must be positive")
    if expected_reference_frame_count is not None and reference_mmwave_root is None:
        raise ValueError(
            "expected_reference_frame_count requires reference_mmwave_root"
        )

    validation_form = data_form or official_validation_form(
        random_seed=0, train_ratio=0.8
    )
    allowed_actions = set(LOWER_LIMB_ACTIONS) if scope == "target" else None
    expected_recordings = sorted(
        (subject_environment(subject_name), subject_name, action)
        for subject_name, actions in validation_form.items()
        for action in actions
        if (allowed_actions is None or action in allowed_actions)
        and (
            environment is None
            or subject_environment(subject_name) == environment
        )
        and (subject is None or subject_name == subject)
    )
    if not expected_recordings:
        raise RuntimeError("The requested protocol contains no recordings")

    aligned_frame_count = 0
    reference_frame_count = 0
    raw_only_frame_count = 0
    mmwave_reference_byte_match_count = 0
    ground_truth_unselected_frame_count = 0
    lidar_bytes = 0
    mmwave_bytes = 0
    frames_per_environment = {name: 0 for name in ENVIRONMENTS}
    observed_subjects: set[str] = set()
    observed_actions: set[str] = set()
    lidar_digest = hashlib.sha256() if hash_content else None
    mmwave_digest = hashlib.sha256() if hash_content else None
    aligned_pair_digest = hashlib.sha256() if hash_content else None

    for scene, subject, action in expected_recordings:
        recording_root = Path(data_root) / scene / subject / action
        ground_truth = recording_root / "ground_truth.npy"
        if not ground_truth.is_file():
            raise FileNotFoundError(f"Missing ground truth: {ground_truth}")
        try:
            ground_truth_values = np.load(
                ground_truth, mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Unreadable ground truth: {ground_truth}") from error
        if ground_truth_values.ndim < 1 or ground_truth_values.shape[0] == 0:
            raise RuntimeError(f"Invalid ground-truth shape: {ground_truth}")
        if not np.isfinite(ground_truth_values).all():
            raise RuntimeError(f"Non-finite ground truth: {ground_truth}")

        lidar = index_frame_directory(recording_root / "lidar")
        mmwave = index_frame_directory(recording_root / "mmwave")
        lidar_ids = set(lidar)
        mmwave_ids = set(mmwave)
        if lidar_ids != mmwave_ids:
            raise RuntimeError(
                f"Frame-ID mismatch in {scene}/{subject}/{action}: "
                f"only_lidar={sorted(lidar_ids - mmwave_ids)[:8]}, "
                f"only_mmwave={sorted(mmwave_ids - lidar_ids)[:8]}"
            )
        expected_ground_truth_ids = set(range(1, ground_truth_values.shape[0] + 1))
        if not lidar_ids <= expected_ground_truth_ids:
            raise RuntimeError(
                f"Ground-truth/frame mismatch in {scene}/{subject}/{action}: "
                f"ground_truth_rows={ground_truth_values.shape[0]}, "
                f"only_frames={sorted(lidar_ids - expected_ground_truth_ids)[:8]}, "
                f"only_ground_truth={sorted(expected_ground_truth_ids - lidar_ids)[:8]}"
            )
        ground_truth_unselected_frame_count += len(
            expected_ground_truth_ids - lidar_ids
        )
        if reference_mmwave_root is not None:
            reference = index_frame_directory(
                Path(reference_mmwave_root) / scene / subject / action
            )
            reference_ids = set(reference)
            if lidar_ids != reference_ids:
                raise RuntimeError(
                    f"Reference frame-ID mismatch in {scene}/{subject}/{action}: "
                    f"only_extracted={sorted(lidar_ids - reference_ids)[:8]}, "
                    f"only_reference={sorted(reference_ids - lidar_ids)[:8]}"
                )
            reference_frame_count += len(reference_ids)
            raw_only_frame_count += len(lidar_ids - reference_ids)
            reference_checked_hashes = {}
            for frame_index, reference_path in reference.items():
                mmwave_path = mmwave[frame_index]
                if hash_content:
                    mmwave_hash = sha256_file(mmwave_path)
                    reference_hash = sha256_file(reference_path)
                    matched = mmwave_hash == reference_hash
                    reference_checked_hashes[frame_index] = mmwave_hash
                else:
                    matched = filecmp.cmp(mmwave_path, reference_path, shallow=False)
                if not matched:
                    raise RuntimeError(
                        f"Raw/filtered mmWave byte mismatch: {mmwave_path} "
                        f"!= {reference_path}"
                    )
                mmwave_reference_byte_match_count += 1
        else:
            reference_checked_hashes = {}

        for frame_index in sorted(lidar_ids):
            lidar_path = lidar[frame_index]
            mmwave_path = mmwave[frame_index]
            validate_binary_layout(lidar_path, POINT_WIDTHS["lidar"])
            validate_binary_layout(mmwave_path, POINT_WIDTHS["mmwave"])
            lidar_bytes += lidar_path.stat().st_size
            mmwave_bytes += mmwave_path.stat().st_size
            if hash_content:
                sample_id = f"{scene}/{subject}/{action}/{frame_index:06d}"
                lidar_hash = sha256_file(lidar_path)
                mmwave_hash = reference_checked_hashes.get(frame_index)
                if mmwave_hash is None:
                    mmwave_hash = sha256_file(mmwave_path)
                update_content_digest(lidar_digest, sample_id, lidar_hash)
                update_content_digest(mmwave_digest, sample_id, mmwave_hash)
                aligned_pair_digest.update(sample_id.encode("utf-8"))
                aligned_pair_digest.update(b"\0")
                aligned_pair_digest.update(bytes.fromhex(lidar_hash))
                aligned_pair_digest.update(bytes.fromhex(mmwave_hash))
                aligned_pair_digest.update(b"\n")

        frame_count = len(lidar_ids)
        aligned_frame_count += frame_count
        frames_per_environment[scene] += frame_count
        observed_subjects.add(subject)
        observed_actions.add(action)

    if (
        expected_frame_count is not None
        and aligned_frame_count != expected_frame_count
    ):
        raise RuntimeError(
            f"Expected {expected_frame_count} aligned frames, found "
            f"{aligned_frame_count}"
        )
    if (
        expected_reference_frame_count is not None
        and reference_frame_count != expected_reference_frame_count
    ):
        raise RuntimeError(
            f"Expected {expected_reference_frame_count} reference frames, found "
            f"{reference_frame_count}"
        )

    provenance = (
        hash_provenance_bundle(provenance_root)
        if provenance_root is not None
        else None
    )
    return {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(Path(data_root).resolve()),
        "scope": scope,
        "environment": environment,
        "subject": subject,
        "expected_recording_count": len(expected_recordings),
        "observed_subject_count": len(observed_subjects),
        "observed_action_count": len(observed_actions),
        "aligned_frame_count": aligned_frame_count,
        "reference_frame_count": (
            reference_frame_count if reference_mmwave_root is not None else None
        ),
        "raw_only_frame_count": (
            raw_only_frame_count if reference_mmwave_root is not None else None
        ),
        "mmwave_reference_byte_match_count": (
            mmwave_reference_byte_match_count
            if reference_mmwave_root is not None
            else None
        ),
        "ground_truth_unselected_frame_count": (
            ground_truth_unselected_frame_count
        ),
        "reference_mmwave_root": (
            str(Path(reference_mmwave_root).resolve())
            if reference_mmwave_root is not None
            else None
        ),
        "frames_per_environment": frames_per_environment,
        "lidar_bytes": lidar_bytes,
        "mmwave_bytes": mmwave_bytes,
        "expected_frame_count": expected_frame_count,
        "expected_reference_frame_count": expected_reference_frame_count,
        "content_hashing_enabled": hash_content,
        "lidar_content_sha256": lidar_digest.hexdigest() if hash_content else None,
        "mmwave_content_sha256": (
            mmwave_digest.hexdigest() if hash_content else None
        ),
        "aligned_pair_content_sha256": (
            aligned_pair_digest.hexdigest() if hash_content else None
        ),
        "archive_provenance": provenance,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "target"), default="all")
    parser.add_argument("--environment", choices=ENVIRONMENTS)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--reference-mmwave-root", type=Path)
    parser.add_argument("--expected-reference-frame-count", type=int)
    parser.add_argument("--hash-content", action="store_true")
    parser.add_argument("--provenance-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    report = audit_extracted_point_modalities(
        data_root=arguments.data_root,
        scope=arguments.scope,
        environment=arguments.environment,
        subject=arguments.subject,
        expected_frame_count=arguments.expected_frame_count,
        reference_mmwave_root=arguments.reference_mmwave_root,
        expected_reference_frame_count=arguments.expected_reference_frame_count,
        hash_content=arguments.hash_content,
        provenance_root=arguments.provenance_root,
    )
    selection = arguments.subject or arguments.environment or "all"
    output = arguments.output or (
        arguments.data_root
        / f"point_modality_audit_{selection}_{arguments.scope}.json"
    )
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
