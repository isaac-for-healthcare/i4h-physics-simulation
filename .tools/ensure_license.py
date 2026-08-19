# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that source files declare copyright and SPDX license metadata."""

from __future__ import annotations

import os
from pathlib import Path

FILE_EXTENSIONS = {".py", ".sh", ".ipynb", ".slurm", ".h", ".hpp", ".cu", ".cpp", ".txt"}
IGNORED_FILES = {"NOTICE.txt"}
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "build-ci",
    "build-release",
    "dist",
    ".eggs",
    "assets",
    "tools",
    "_deps",
    "cosmos_h_dreams",
}

COPYRIGHT_MARKER = "Copyright"
LICENSE_MARKER = "SPDX-License-Identifier:"


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith("build") or name.endswith(".egg-info")


def has_license_metadata(path: Path) -> bool:
    """Return whether *path* contains both required SPDX declarations."""
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return True
    return COPYRIGHT_MARKER in content and LICENSE_MARKER in content


def find_files_without_license(directory: Path) -> list[Path]:
    """Return source files below *directory* that lack SPDX metadata."""
    missing = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [name for name in dirs if not _should_skip_dir(name)]
        for name in files:
            path = Path(root, name)
            if name not in IGNORED_FILES and path.suffix in FILE_EXTENSIONS and not has_license_metadata(path):
                missing.append(path)
    return missing


if __name__ == "__main__":
    missing_files = find_files_without_license(Path("."))
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Copyright or SPDX license metadata is missing in:\n{missing}")
    print("All source files have copyright and SPDX license metadata.")
