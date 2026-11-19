# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Local package paths must be set before importing simulation modules.
"""Run this checkout's numerical qualification, using an explicitly selected interpreter."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for package in (
    "physics",
    "surgical",
    "isaaclab",
    "assessment",
    "standalone",
    "endoluminal",
):
    sys.path.insert(0, str(root / package / "src"))

from i4h_surgical.grasp_qualification import main

if __name__ == "__main__":
    main()
