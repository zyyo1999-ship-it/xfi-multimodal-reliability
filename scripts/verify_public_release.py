#!/usr/bin/env python3
"""Fail if tracked files contain common private-release hazards."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth"}
FORBIDDEN_PATH_PARTS = {"data", "third_party", ".venv", "__pycache__"}
SENSITIVE_PATTERNS = {
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "credential assignment": re.compile(
        r"(?i)(password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s$<{][^\s]*"
    ),
    "cloud rental host": re.compile(r"(?i)jq\d*\.\d+gpu\.com"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode("utf-8")
    return [ROOT / path for path in output.split("\0") if path]


def main() -> None:
    failures: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden binary asset: {relative}")
        if FORBIDDEN_PATH_PARTS.intersection(relative.parts):
            failures.append(f"forbidden path: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file larger than 10 MiB: {relative}")
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if relative == Path("scripts/verify_public_release.py"):
                continue
            if pattern.search(text):
                failures.append(f"possible {label}: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(tracked_files())} tracked files are ready for public release.")


if __name__ == "__main__":
    main()
