#!/usr/bin/env python3
"""Resumable frozen-X-Fi inference for the formal LiDAR+mmWave matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lower_limb_protocol import LOWER_LIMB_ACTIONS
from multimodal_dataset import (
    AlignedPointDataset,
    PointCloudCache,
    collate_aligned_points,
    discover_aligned_point_frames,
)
from multimodal_protocol import (
    ExperimentCondition,
    clean_baseline_conditions,
    formal_conditions,
    smoke_test_conditions,
)


RUN_SCHEMA_VERSION = 3
PIPELINE_SOURCE_FILES = (
    "src/run_multimodal_inference.py",
    "src/multimodal_dataset.py",
    "src/multimodal_protocol.py",
    "src/corruptions.py",
    "src/lower_limb_protocol.py",
    "src/mmwave_dataset.py",
    "src/xfi_runtime.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(project_root: Path) -> str:
    """Fingerprint every local source file that can alter formal logits."""
    digest = hashlib.sha256()
    for relative_path in PIPELINE_SOURCE_FILES:
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing inference source file: {path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def aligned_data_fingerprint(frames: list, max_samples: int | None = None) -> str:
    """Fingerprint ordered sample IDs and point-file sizes without loading all data."""
    selected = frames if max_samples is None else frames[:max_samples]
    digest = hashlib.sha256()
    for frame in selected:
        digest.update(frame.key.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(frame.lidar_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(frame.mmwave_path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_resume_manifest(prior: dict, run_signature: dict) -> None:
    """Reject artifacts produced under any different immutable run identity."""
    if prior.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError(
            "Existing inference manifest uses an incompatible schema; choose a new "
            "output directory or rerun with --overwrite."
        )
    if prior.get("run_signature") != run_signature:
        raise RuntimeError(
            "Existing inference artifacts were produced by a different checkpoint, "
            "dataset, code bundle, condition matrix, or batch configuration; choose "
            "a new output directory or rerun with --overwrite."
        )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def active_modalities(condition: ExperimentCondition) -> set[str]:
    mask = condition.modality_mask
    active = set()
    if mask[2]:
        active.add("mmwave")
    if mask[3]:
        active.add("lidar")
    return active


def load_frozen_xfi(
    project_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    official_root = project_root / "third_party" / "X-Fi" / "MMFi_HAR"
    if not official_root.is_dir():
        raise FileNotFoundError(f"Official X-Fi source is missing: {official_root}")
    sys.path.insert(0, str(official_root))

    previous_directory = Path.cwd()
    try:
        os.chdir(official_root)
        from xfi_runtime import (
            build_xfi_for_checkpoint,
            load_xfi_state_dict,
            require_multimodal_checkpoint,
        )

        model = build_xfi_for_checkpoint(model_depth=2)
    finally:
        os.chdir(previous_directory)

    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = load_xfi_state_dict(model, payload)
    require_multimodal_checkpoint(metadata)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def condition_result_is_valid(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                payload["logits"].shape == (expected_count, 27)
                and payload["targets"].shape == (expected_count,)
                and payload["sample_ids"].shape == (expected_count,)
                and np.isfinite(payload["logits"]).all()
            )
    except (OSError, KeyError, ValueError):
        return False


def infer_condition(
    model: torch.nn.Module,
    frames: list,
    condition: ExperimentCondition,
    device: torch.device,
    batch_size: int,
    cache: PointCloudCache,
    max_samples: int | None,
    model_seed: int,
) -> dict[str, np.ndarray]:
    selected_frames = frames if max_samples is None else frames[:max_samples]
    dataset = AlignedPointDataset(
        selected_frames,
        geometry=condition.geometry,
        experiment_seed=condition.corruption_seed,
        lidar_drop_rate=condition.lidar_drop_rate,
        mmwave_drop_rate=condition.mmwave_drop_rate,
        active_modalities=active_modalities(condition),
        cache=cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_aligned_points,
    )

    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    collected: dict[str, list] = {
        "logits": [],
        "targets": [],
        "sample_ids": [],
        "scenes": [],
        "subjects": [],
        "actions": [],
        "frame_indices": [],
        "partitions": [],
        "lidar_point_counts": [],
        "mmwave_point_counts": [],
        "lidar_azimuth_occupancy": [],
        "mmwave_azimuth_occupancy": [],
        "lidar_range_occupancy": [],
        "mmwave_range_occupancy": [],
    }

    with torch.inference_mode():
        for batch in loader:
            lidar = batch["lidar"].to(device, non_blocking=True)
            mmwave = batch["mmwave"].to(device, non_blocking=True)
            batch_count = lidar.shape[0]
            # RGB and depth are inactive for every Track B condition. These
            # placeholders satisfy the fixed four-argument X-Fi interface.
            rgb = torch.zeros((batch_count, 3, 1, 1), device=device)
            depth = torch.zeros((batch_count, 3, 1, 1), device=device)
            logits = model(
                rgb,
                depth,
                mmwave,
                lidar,
                list(condition.modality_mask),
            )
            collected["logits"].append(logits.detach().cpu().numpy().astype(np.float32))
            collected["targets"].append(batch["targets"].numpy().astype(np.int16))
            for key in (
                "sample_ids",
                "scenes",
                "subjects",
                "actions",
                "partitions",
            ):
                collected[key].extend(batch[key])
            for key in (
                "frame_indices",
                "lidar_point_counts",
                "mmwave_point_counts",
                "lidar_azimuth_occupancy",
                "mmwave_azimuth_occupancy",
                "lidar_range_occupancy",
                "mmwave_range_occupancy",
            ):
                collected[key].append(batch[key])

    arrays = {
        "logits": np.concatenate(collected["logits"], axis=0),
        "targets": np.concatenate(collected["targets"], axis=0),
    }
    for key in ("sample_ids", "scenes", "subjects", "actions", "partitions"):
        arrays[key] = np.asarray(collected[key])
    for key in (
        "frame_indices",
        "lidar_point_counts",
        "mmwave_point_counts",
        "lidar_azimuth_occupancy",
        "mmwave_azimuth_occupancy",
        "lidar_range_occupancy",
        "mmwave_range_occupancy",
    ):
        arrays[key] = np.concatenate(collected[key], axis=0)
    arrays["condition_json"] = np.asarray(json.dumps(condition.to_dict(), sort_keys=True))
    return arrays


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", choices=("audit", "baseline", "smoke", "formal"), default="smoke")
    parser.add_argument("--scope", choices=("target", "all"), default="target")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-gib", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--condition-id", action="append", default=[])
    parser.add_argument("--model-seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    actions = set(LOWER_LIMB_ACTIONS) if arguments.scope == "target" else None
    frames = discover_aligned_point_frames(arguments.data_root, actions=actions)
    if arguments.max_samples is not None:
        if arguments.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        expected_count = min(arguments.max_samples, len(frames))
    else:
        expected_count = len(frames)

    audit = {
        "created_at_utc": utc_now(),
        "data_root": str(arguments.data_root.resolve()),
        "scope": arguments.scope,
        "aligned_frame_count": len(frames),
        "effective_frame_count": expected_count,
        "recording_count": len(
            {(frame.key.scene, frame.key.subject, frame.key.action) for frame in frames}
        ),
        "subject_count": len({frame.key.subject for frame in frames}),
        "action_count": len({frame.key.action for frame in frames}),
        "first_sample_id": frames[0].key.sample_id,
        "last_sample_id": frames[-1].key.sample_id,
    }
    atomic_json(arguments.output_dir / "data_alignment_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if arguments.plan == "audit":
        return
    if arguments.checkpoint is None or not arguments.checkpoint.is_file():
        raise FileNotFoundError("A valid --checkpoint is required for inference")
    if arguments.batch_size <= 0 or arguments.cache_gib < 0.0:
        raise ValueError("batch size must be positive and cache size non-negative")

    data_audit_identity = None
    if arguments.data_audit is not None:
        if not arguments.data_audit.is_file():
            raise FileNotFoundError(f"Missing data audit: {arguments.data_audit}")
        data_audit = json.loads(arguments.data_audit.read_text(encoding="utf-8"))
        if data_audit.get("status") != "PASS":
            raise RuntimeError("The supplied data audit did not pass")
        if not data_audit.get("content_hashing_enabled"):
            raise RuntimeError("The supplied data audit has no content hashes")
        if not data_audit.get("aligned_pair_content_sha256"):
            raise RuntimeError("The supplied data audit has no aligned-pair hash")
        audited_root = Path(data_audit["data_root"]).resolve()
        if audited_root != arguments.data_root.resolve():
            raise RuntimeError(
                f"Data-audit root mismatch: {audited_root} != {arguments.data_root.resolve()}"
            )
        if arguments.scope == "all" and int(data_audit["aligned_frame_count"]) != len(frames):
            raise RuntimeError("Full-scope frame count differs from the data audit")
        if arguments.scope == "target" and int(data_audit["aligned_frame_count"]) < len(frames):
            raise RuntimeError("Target frames exceed the audited full cohort")
        provenance = data_audit.get("archive_provenance") or {}
        if int(provenance.get("provenance_record_count", -1)) != 40:
            raise RuntimeError("The data audit is not bound to all 40 official archives")
        data_audit_identity = {
            "path": str(arguments.data_audit.resolve()),
            "sha256": sha256_file(arguments.data_audit),
            "aligned_pair_content_sha256": data_audit["aligned_pair_content_sha256"],
            "provenance_bundle_sha256": provenance["provenance_bundle_sha256"],
            "audited_full_frame_count": int(data_audit["aligned_frame_count"]),
        }

    if arguments.plan == "baseline":
        conditions = clean_baseline_conditions()
    elif arguments.plan == "smoke":
        conditions = smoke_test_conditions()
    else:
        conditions = formal_conditions()
    if arguments.condition_id:
        selected_ids = set(arguments.condition_id)
        conditions = [c for c in conditions if c.condition_id in selected_ids]
        missing_ids = selected_ids - {condition.condition_id for condition in conditions}
        if missing_ids:
            raise ValueError(f"Unknown requested condition IDs: {sorted(missing_ids)}")
    if arguments.max_conditions is not None:
        if arguments.max_conditions <= 0:
            raise ValueError("--max-conditions must be positive")
        conditions = conditions[: arguments.max_conditions]

    checkpoint_sha256 = sha256_file(arguments.checkpoint)
    run_signature = {
        "plan": arguments.plan,
        "scope": arguments.scope,
        "checkpoint_sha256": checkpoint_sha256,
        "source_bundle_sha256": source_bundle_sha256(project_root),
        "aligned_data_fingerprint": aligned_data_fingerprint(
            frames, arguments.max_samples
        ),
        "model_seed": arguments.model_seed,
        "batch_size": arguments.batch_size,
        "expected_sample_count": expected_count,
        "condition_ids": [condition.condition_id for condition in conditions],
        "data_audit_identity": data_audit_identity,
    }
    manifest_path = arguments.output_dir / "inference_manifest.json"
    prior = None
    if manifest_path.is_file() and not arguments.overwrite:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(prior, run_signature)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: inference is running without CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, checkpoint_metadata = load_frozen_xfi(
        project_root, arguments.checkpoint, device
    )
    cache = PointCloudCache(
        max_bytes=int(arguments.cache_gib * 1024**3)
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "plan": arguments.plan,
        "scope": arguments.scope,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": checkpoint_metadata,
        "model_seed": arguments.model_seed,
        "batch_size": arguments.batch_size,
        "expected_sample_count": expected_count,
        "data_audit_identity": data_audit_identity,
        "condition_count": len(conditions),
        "run_signature": run_signature,
        "completed": {},
    }
    if prior is not None:
        manifest["created_at_utc"] = prior.get("created_at_utc", utc_now())
        manifest["completed"] = prior.get("completed", {})

    for index, condition in enumerate(conditions, start=1):
        result_path = arguments.output_dir / "conditions" / f"{condition.condition_id}.npz"
        prior_result = manifest["completed"].get(condition.condition_id)
        can_resume = (
            not arguments.overwrite
            and prior_result is not None
            and prior_result.get("condition") == condition.to_dict()
            and condition_result_is_valid(result_path, expected_count)
            and sha256_file(result_path) == prior_result.get("sha256")
        )
        if can_resume:
            print(f"[{index}/{len(conditions)}] SKIP {condition.condition_id}")
            manifest["updated_at_utc"] = utc_now()
            atomic_json(manifest_path, manifest)
            continue
        print(f"[{index}/{len(conditions)}] RUN  {condition.condition_id}")
        arrays = infer_condition(
            model=model,
            frames=frames,
            condition=condition,
            device=device,
            batch_size=arguments.batch_size,
            cache=cache,
            max_samples=arguments.max_samples,
            model_seed=arguments.model_seed,
        )
        atomic_npz(result_path, arrays)
        if not condition_result_is_valid(result_path, expected_count):
            raise RuntimeError(f"Post-write audit failed for {result_path}")
        manifest["completed"][condition.condition_id] = {
            "condition": condition.to_dict(),
            "path": str(result_path.resolve()),
            "sha256": sha256_file(result_path),
            "sample_count": expected_count,
            "completed_at_utc": utc_now(),
        }
        manifest["updated_at_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Completed {len(manifest['completed'])}/{len(conditions)} conditions")


if __name__ == "__main__":
    main()
