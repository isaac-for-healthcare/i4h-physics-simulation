#!/usr/bin/env python3
"""Run with isolated dependencies; never sync a qualification environment."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = Path(
    os.environ.get(
        "I4H_VERSIUS_LAB", "/home/pkorzeniowsk/Projects/isaaclab/isaaclab-main"
    )
)
NEWTON = Path(
    os.environ.get(
        "I4H_VERSIUS_NEWTON",
        "/home/pkorzeniowsk/Projects/newton/newton-1.9-worktrees/i4h-v2-soft-solvers",
    )
)
ENV = HERE / ".venv"
if not (ENV / "bin/python").exists():
    raise SystemExit(
        "Create the isolated environment with: python3 examples/versius/bootstrap.py"
    )
paths = [
    HERE / "src",
    NEWTON,
    *[
        ROOT / p / "src"
        for p in (
            "physics",
            "surgical",
            "endoluminal",
            "isaaclab",
            "assessment",
            "standalone",
        )
    ],
    *sorted(p for p in (LAB / "source").iterdir() if p.is_dir()),
]
env = dict(
    os.environ,
    PYTHONPATH=":".join(map(str, paths)),
    PYTHONDONTWRITEBYTECODE="1",
    I4H_DATA=str(ROOT / "data"),
)
args = sys.argv[1:]
command = args[1:] if args[:1] == ["python"] else ["-m", "i4h_versius.launch", *args]
receipt = ROOT / ".tmp_tests/versius/commands"
receipt.mkdir(parents=True, exist_ok=True)
# Preserve an explicitly activated environment, including uv's --with overlay.
# A system-Python invocation continues to use the isolated example environment.
interpreter = sys.executable if sys.prefix != sys.base_prefix else str(ENV / "bin/python")
record = {
    "command": [interpreter, *command],
    "cwd": str(ROOT),
    "PYTHONPATH": env["PYTHONPATH"],
}
code = subprocess.call(record["command"], env=env, cwd=ROOT)
record["exit_code"] = code
(receipt / f"{time.time_ns()}.json").write_text(json.dumps(record, indent=2) + "\n")
raise SystemExit(code)
