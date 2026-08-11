"""Create SHA-256 provenance records for code, data, weights, and results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ASSETS = (
    "assets/xfi_weights/MMFi_HAR/mmfi_har_mmwave_only_checkpoint.pt",
    "data/derived/filtered_mmwave_validation.zip",
    "data/derived/validation_archive_manifest.json",
    "third_party/X-Fi/MMFi_HAR/baseline2/baseline_results/all_label.npy",
    "configs/pilot_robustness.yaml",
)
CODE_DIRECTORIES = ("src", "tests", "scripts")
TRACKED_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
DOCUMENT_DIRECTORIES = ("docs",)
DOCUMENT_SUFFIXES = {".md", ".bib"}
ROOT_DOCUMENTS = ("README.md", "requirements-research.txt")
RESULT_SUFFIXES = {".json", ".csv", ".md", ".png"}
PACKAGES = (
    "einops",
    "matplotlib",
    "numpy",
    "PyYAML",
    "scikit-learn",
    "torch",
    "torchvision",
    "tqdm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create provenance manifest.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/mmwave_robustness"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path) -> dict:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> None:
    args = parse_args()
    results_directory = (
        args.results
        if args.results.is_absolute()
        else PROJECT_ROOT / args.results
    ).resolve()
    if not (results_directory / "summary.json").is_file():
        raise FileNotFoundError(
            f"Complete results are required: {results_directory / 'summary.json'}"
        )

    files = []
    for relative_path in CORE_ASSETS:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    for directory_name in CODE_DIRECTORIES:
        files.extend(
            path
            for path in (PROJECT_ROOT / directory_name).rglob("*")
            if path.is_file() and path.suffix in TRACKED_SUFFIXES
        )
    for directory_name in DOCUMENT_DIRECTORIES:
        files.extend(
            path
            for path in (PROJECT_ROOT / directory_name).rglob("*")
            if path.is_file() and path.suffix in DOCUMENT_SUFFIXES
        )
    for relative_path in ROOT_DOCUMENTS:
        path = PROJECT_ROOT / relative_path
        if path.is_file():
            files.append(path)
    files.extend(
        path
        for path in results_directory.rglob("*")
        if path.is_file()
        and path.suffix in RESULT_SUFFIXES
        and path.name != "reproducibility_manifest.json"
    )
    files = sorted(set(files), key=lambda path: str(path.relative_to(PROJECT_ROOT)))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_scope": (
            "Pretrained X-Fi filtered-mmWave HAR under controlled per-frame "
            "point sparsification"
        ),
        "platform": platform.platform(),
        "python_version": sys.version,
        "package_versions": package_versions(),
        "file_count": len(files),
        "files": [describe_file(path) for path in files],
    }
    output_path = results_directory / "reproducibility_manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[SAVED] {output_path} ({len(files)} hashed files)")


if __name__ == "__main__":
    main()
