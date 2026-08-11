#!/usr/bin/env python3
"""Audit the released X-Fi checkpoint against full 27-class clean references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from multimodal_protocol import clean_baseline_conditions


OFFICIAL_XFI_ACCURACIES = {
    "lidar_only": 0.527,
    "mmwave_only": 0.857,
    "lidar_mmwave": 0.887,
}
REFERENCE = {
    "paper": "X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing",
    "venue": "ICLR 2025",
    "location": "Table 2, MM-Fi HAR, S1 Random Split",
    "url": "https://arxiv.org/abs/2410.10167",
    "note": "The paper reports averages from multiple experiments; the released checkpoint is one realization.",
}


def load_clean_artifact(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing clean baseline artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    required = {"logits", "targets", "sample_ids", "actions"}
    missing = required - set(arrays)
    if missing:
        raise RuntimeError(f"Missing arrays in {path.name}: {sorted(missing)}")
    sample_count = arrays["targets"].size
    if arrays["logits"].shape != (sample_count, 27):
        raise RuntimeError(f"Expected [samples, 27] logits: {path.name}")
    if arrays["sample_ids"].shape != (sample_count,):
        raise RuntimeError(f"Sample IDs do not align: {path.name}")
    if not np.isfinite(arrays["logits"]).all():
        raise RuntimeError(f"Non-finite logits: {path.name}")
    return arrays


def evaluate_clean_baselines(
    inference_dir: Path,
    expected_accuracies: dict[str, float] | None = None,
    tolerance: float = 0.03,
    expected_frame_count: int | None = 54_433,
) -> dict:
    """Return a machine-readable clean-baseline gate report."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    references = expected_accuracies or OFFICIAL_XFI_ACCURACIES
    condition_dir = Path(inference_dir) / "conditions"
    rows = []
    reference_samples = None
    reference_targets = None
    reference_actions = None
    for condition in clean_baseline_conditions():
        path = condition_dir / f"{condition.condition_id}.npz"
        arrays = load_clean_artifact(path)
        if reference_samples is None:
            reference_samples = arrays["sample_ids"]
            reference_targets = arrays["targets"]
            reference_actions = arrays["actions"]
        else:
            for name, expected, observed in (
                ("sample_ids", reference_samples, arrays["sample_ids"]),
                ("targets", reference_targets, arrays["targets"]),
                ("actions", reference_actions, arrays["actions"]),
            ):
                if not np.array_equal(expected, observed):
                    raise RuntimeError(
                        f"Cross-mask clean baseline {name} mismatch: {condition.mask_name}"
                    )
        predictions = arrays["logits"].argmax(axis=1)
        accuracy = float(np.mean(predictions == arrays["targets"]))
        macro_f1 = float(
            f1_score(
                arrays["targets"],
                predictions,
                labels=np.arange(27),
                average="macro",
                zero_division=0,
            )
        )
        expected = float(references[condition.mask_name])
        delta = accuracy - expected
        rows.append(
            {
                "mask_name": condition.mask_name,
                "sample_count": int(arrays["targets"].size),
                "action_count": int(np.unique(arrays["actions"]).size),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "official_accuracy_reference": expected,
                "accuracy_delta": delta,
                "absolute_tolerance": tolerance,
                "within_tolerance": abs(delta) <= tolerance,
            }
        )

    sample_count = int(reference_targets.size)
    if expected_frame_count is not None and sample_count != expected_frame_count:
        raise RuntimeError(
            f"Expected {expected_frame_count} full-validation frames, found {sample_count}"
        )
    if np.unique(reference_targets).size != 27:
        raise RuntimeError("The clean baseline does not contain all 27 target labels")
    status = "PASS" if all(row["within_tolerance"] for row in rows) else "FAIL"
    return {
        "status": status,
        "scope": "full_27_class_clean_validation",
        "sample_count": sample_count,
        "action_count": int(np.unique(reference_actions).size),
        "class_count": int(np.unique(reference_targets).size),
        "reference": REFERENCE,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--expected-frame-count", type=int, default=54_433)
    arguments = parser.parse_args()
    report = evaluate_clean_baselines(
        arguments.inference_dir,
        tolerance=arguments.tolerance,
        expected_frame_count=arguments.expected_frame_count,
    )
    output = arguments.output or arguments.inference_dir / "clean_baseline_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output.with_suffix(".csv"), report["rows"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Clean baseline is outside the preregistered tolerance")


if __name__ == "__main__":
    main()
