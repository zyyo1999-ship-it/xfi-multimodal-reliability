"""Verify that the official pretrained X-Fi MMFi-HAR model loads correctly."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from xfi_runtime import build_xfi_for_checkpoint, load_xfi_state_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XFI_HAR_DIRECTORY = PROJECT_ROOT / "third_party/X-Fi/MMFi_HAR"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "assets/xfi_weights/MMFi_HAR/mmfi_har_mmwave_only_checkpoint.pt"
)
OUTPUT_PATH = PROJECT_ROOT / "results/xfi_model_load_check.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an X-Fi checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous_directory = Path.cwd()
    sys.path.insert(0, str(XFI_HAR_DIRECTORY))
    os.chdir(XFI_HAR_DIRECTORY)
    try:
        torch.manual_seed(3407)
        model = build_xfi_for_checkpoint(model_depth=2)
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_metadata = load_xfi_state_dict(model, checkpoint_payload)
    finally:
        os.chdir(previous_directory)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    report = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "status": "passed",
        "scope": "Model construction and weight loading only; no inference",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{report['status'].upper()}] X-Fi loaded with "
        f"{parameter_count:,} parameters"
    )
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
