# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run DLV-010-29 sequentially, one installed-wheel pytest process per matrix cell.

Use the qualification environment's Python. No package installation or dependency
resolution happens here. A failed or skipped cell fails the campaign; all logs survive.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TASKS = ("I4h-Chole-Direct", "I4h-Aorta-Direct", "I4h-ReachTarget-Direct")
PACKAGES = ("physics", "host", "isaaclab", "endoluminal", "surgical", "assessment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = root.parent
    test = root / "tests/runtime/test_direct_acceptance.py"
    origins = {}
    for name in PACKAGES:
        module = importlib.import_module("i4h_" + name)
        path = Path(module.__file__).absolute()
        assert "site-packages" in path.parts, f"candidate must be installed: {path}"
        assert not path.is_relative_to(root / name / "src"), path
        origins[module.__name__] = str(path)
    data = Path(os.environ["I4H_DATA"]).resolve()
    assert data.is_dir(), data
    manifest = {
        "task": args.task, "python": sys.executable, "origins": origins, "assets": str(data),
        "base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "staged_patch_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "--cached", "--binary"], cwd=source)).hexdigest(),
        "unstaged_patch_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "--binary"], cwd=source)).hexdigest(),
        "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("warp-lang", "torch", "newton", "mujoco-core", "mujoco-warp", "rsl-rl-lib")},
        "cells": [],
    }
    assert manifest["dependencies"]["warp-lang"] == "1.17.0"
    newton = importlib.metadata.distribution("newton")
    manifest["newton_origin"] = json.loads(newton.read_text("direct_url.json"))
    assert manifest["newton_origin"]["vcs_info"]["commit_id"] == (
        "b70ad54cfcaf7570907d3a3aad6c60acc7443c33")
    manifest["qualification_source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (test, Path(__file__).resolve(), root / "scripts/setup_direct_qualification.py")}
    env = dict(os.environ, I4H_DIRECT_RESULTS=str(output / "measurements"))
    # Source tests are explicit; the working directory is outside every package source tree.
    env.pop("PYTHONPATH", None)
    for captured in ("eager", "graph"):
        for count in (1, 8):
            for kind in ("lifecycle", "ppo"):
                label = f"{kind}-{captured}-{count}"
                xml = output / (label + ".xml")
                node = f"{test}::test_direct_{kind}[{captured}-{count}-{args.task}]"
                command = [sys.executable, "-m", "pytest", node, "-q", f"--junitxml={xml}"]
                print("START", args.task, label, flush=True)
                cell_env = dict(env)
                if captured == "graph" and count == 1 and kind == "lifecycle":
                    cell_env["I4H_DIRECT_VIDEO"] = str(output / "video")
                with (output / (label + ".log")).open("w") as log:
                    result = subprocess.run(command, cwd=output, env=cell_env, stdout=log, stderr=subprocess.STDOUT)
                cases = ET.parse(xml).findall(".//testcase") if xml.exists() else []
                passed = result.returncode == 0 and len(cases) == 1 and not any(
                    c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
                manifest["cells"].append({"label": label, "command": command,
                                          "returncode": result.returncode, "passed": passed})
                (output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print("PASS" if passed else "FAIL", label, flush=True)
    passed = all(cell["passed"] for cell in manifest["cells"])
    manifest["passed"] = passed
    (output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
