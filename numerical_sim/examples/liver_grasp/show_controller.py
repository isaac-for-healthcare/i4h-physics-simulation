# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive Slang controller; shares the 30 mm start and 20 mm lift training adapter."""

import importlib.util
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
for package in (
    "physics",
    "surgical",
    "standalone",
    "assessment",
    "isaaclab",
    "endoluminal",
):
    sys.path.insert(0, str(root / package / "src"))
if importlib.util.find_spec("rsl_rl") is None:
    training_python = Path(__file__).with_name(".venv-train") / "bin/python"
    if not training_python.exists():
        raise SystemExit("Run examples/liver_grasp/setup_training.sh first")
    os.execv(str(training_python), [str(training_python), __file__, *sys.argv[1:]])

from train import main  # noqa: E402

if __name__ == "__main__":
    sys.argv.insert(1, "play")
    sys.argv.extend(["--worlds", "1"])
    main()
