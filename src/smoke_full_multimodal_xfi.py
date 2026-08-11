#!/usr/bin/env python3
"""Exercise every Track B X-Fi branch with deterministic synthetic point clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from multimodal_protocol import MODALITY_MASKS
from run_multimodal_inference import load_frozen_xfi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full multimodal smoke test")

    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda:0")
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    model, checkpoint_metadata = load_frozen_xfi(
        project_root, arguments.checkpoint, device
    )

    batch_size = arguments.batch_size
    rgb = torch.zeros((batch_size, 3, 1, 1), device=device)
    depth = torch.zeros((batch_size, 3, 1, 1), device=device)
    mmwave = torch.randn((batch_size, 128, 5), device=device)
    lidar = torch.randn((batch_size, 1024, 3), device=device)

    rows = []
    with torch.inference_mode():
        for mask_name in ("lidar_only", "mmwave_only", "lidar_mmwave"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(
                rgb,
                depth,
                mmwave,
                lidar,
                list(MODALITY_MASKS[mask_name]),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            values = logits.detach().cpu().numpy().astype(np.float32)
            if values.shape != (batch_size, 27) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid {mask_name} logits: {values.shape}")
            rows.append(
                {
                    "mask_name": mask_name,
                    "output_shape": list(values.shape),
                    "finite": True,
                    "elapsed_seconds": elapsed,
                    "logits_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            )

    report = {
        "status": "PASS",
        "fixture_only": True,
        "scientific_result": False,
        "purpose": "Model branch/interface verification only",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_metadata": checkpoint_metadata,
        "batch_size": batch_size,
        "synthetic_shapes": {
            "lidar": [batch_size, 1024, 3],
            "mmwave": [batch_size, 128, 5],
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
