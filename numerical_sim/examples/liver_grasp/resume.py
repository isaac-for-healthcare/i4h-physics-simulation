# SPDX-License-Identifier: Apache-2.0
"""Resume PPO while preserving and merging prior curves, validation and episode evidence."""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training_env import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/liver-policy-20261119"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--data", default=str(ROOT.parent.parent / "i4h-isaaclab/numerical_sim/data")
    )
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--seed", type=int, default=7341)
    args = parser.parse_args()
    folder = args.output.resolve()
    checkpoint = (args.checkpoint or folder / "ppo_latest.pt").resolve()
    archive = (
        folder / "sessions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    archive.mkdir(parents=True)
    previous = {}
    files = (
        "ppo_curves.json",
        "ppo_validation_history.json",
        "ppo_training_episodes.json",
    )
    for name in files:
        path = folder / name
        previous[name] = json.loads(path.read_text()) if path.exists() else []
        if path.exists():
            shutil.copyfile(path, archive / name)
    shutil.copyfile(checkpoint, archive / "starting_checkpoint.pt")
    offset = (
        previous["ppo_curves.json"][-1]["elapsed_s"]
        if previous["ppo_curves.json"]
        else 0.0
    )
    command = [
        str(ROOT / "examples/liver_grasp/train.sh"),
        "ppo",
        "--output",
        str(folder),
        "--checkpoint",
        str(checkpoint),
        "--data",
        args.data,
        "--worlds",
        str(args.worlds),
        "--minutes",
        str(args.minutes),
        "--seed",
        str(args.seed),
    ]
    write_json(archive / "resume.json", dict(command=command, curve_offset_s=offset))
    result = subprocess.run(command)
    for name in files:
        path = folder / name
        current = json.loads(path.read_text()) if path.exists() else []
        old = previous[name]
        if name == "ppo_curves.json":
            # Also handles an early construction failure before a new file was written.
            old_iterations = {r["iteration"] for r in old}
            current = [r for r in current if r["iteration"] not in old_iterations]
            for row in current:
                row["session_elapsed_s"] = row["elapsed_s"]
                row["elapsed_s"] += offset
        elif name == "ppo_validation_history.json":
            old_iterations = {r["iteration"] for r in old}
            current = [r for r in current if r["iteration"] not in old_iterations]
            for row in current:
                row["ppo_elapsed_s"] += offset
        elif result.returncode:
            # The terminal episode file is only written at clean completion.
            current = []
        write_json(path, old + current)
    write_json(archive / "completion.json", dict(returncode=result.returncode))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
