# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build wheels and create an isolated, read-only dependency view of PR #292's arena environment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PACKAGES = ("physics", "host", "isaaclab", "endoluminal", "surgical", "assessment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workflows, output = args.workflows.resolve(), args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    # The public PR #292 squash has a different ancestry from its reviewed branch.
    # Accept either baseline, then verify the exact dependency stack below.
    workflows_bases = ("2988beb4cb358affcac3eb48a9fa91b245e71a00",
                       "442e495b2a7cda44f00554009481ce48054e756f")
    workflows_base = next((base for base in workflows_bases if subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=workflows).returncode == 0), None)
    assert workflows_base is not None, "Workflows must descend from a qualified dependency baseline"
    workflows_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workflows, text=True).strip()
    lab = workflows / "third_party/IsaacLab-3939445"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=lab, text=True).strip() == (
        "39394453ceaac256349ad822e675a93485373667")
    for patch in ("metadata", "deformables", "actuators", "vbd", "contact_observables"):
        path = workflows / f"third_party/isaaclab_newton19_{patch}.patch"
        subprocess.run(["git", "apply", "--reverse", "--check", str(path)], cwd=lab, check=True)
    output.mkdir(parents=True, exist_ok=False)
    seed, target = workflows / "arena/.venv", output / "env"
    subprocess.run(["uv", "venv", "--python", str(seed / "bin/python"), str(target)], check=True)
    site = target / "lib/python3.12/site-packages"
    # Dependencies are only read through links. Never run sync/upgrade/uninstall here.
    for path in (seed / "lib/python3.12/site-packages").iterdir():
        if any("i4h_" + name in path.name for name in PACKAGES) or (site / path.name).exists():
            continue
        (site / path.name).symlink_to(path, target_is_directory=path.is_dir())
    subprocess.run(["uv", "build", "--all-packages", "--out-dir", str(output / "dist")], cwd=root, check=True)
    wheels = sorted((output / "dist").glob("*.whl"))
    assert len(wheels) == 6
    subprocess.run(["uv", "pip", "install", "--python", str(target / "bin/python"), "--no-deps",
                    *map(str, wheels)], check=True)
    inventory = subprocess.check_output([str(target / "bin/python"), "-c",
        "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']: "
        "{'version':d.version,'origin':json.loads(d.read_text('direct_url.json') or 'null')} "
        "for d in m.distributions()},indent=2))"], text=True)
    (output / "dependencies.json").write_text(inventory)
    dependencies = json.loads(inventory)
    assert dependencies["newton"]["origin"]["vcs_info"]["commit_id"] == (
        "b70ad54cfcaf7570907d3a3aad6c60acc7443c33")
    assert dependencies["warp-lang"]["version"] == "1.17.0"
    for name, version in {"torch": "2.11.0+cu130", "mujoco-core": "4.0.1", "mujoco-warp": "4.0.1",
                          "rsl-rl-lib": "5.5.1"}.items():
        assert dependencies[name]["version"] == version, (name, dependencies[name]["version"], version)
    provenance = {"workflows_base": workflows_base, "workflows_head": workflows_head,
                  "physics_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (output / "dist").iterdir() if path.is_file()}
    (output / "artifact-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (output / "lab.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=lab))
    print(target / "bin/python")


if __name__ == "__main__":
    main()
