"""Clone retained third-party dependencies into a task-owned environment using uv."""

import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
retained = root / ".tmp_tests/stage-c30-20261109/main-env"
if (here / ".venv").exists():
    raise SystemExit("Task environment already exists; left unchanged")
subprocess.run(
    ["uv", "venv", "--python", str(retained / "bin/python"), str(here / ".venv")],
    check=True,
)
# Copy rather than linking: installing task dependencies cannot change retained packages.
site = "lib/python3.12/site-packages"
subprocess.run(
    [
        "cp",
        "-a",
        "--reflink=auto",
        str(retained / site) + "/.",
        str(here / ".venv" / site),
    ],
    check=True,
)
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--python",
        str(here / ".venv/bin/python"),
        "--no-deps",
        "-r",
        str(here / "requirements.lock"),
    ],
    check=True,
)
