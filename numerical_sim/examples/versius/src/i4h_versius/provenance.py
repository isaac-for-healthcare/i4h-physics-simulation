"""Actual imports, source revisions, dirty-file hashes and read-only asset hashes."""

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path, usd):
    imports, repos = {}, {}
    for name in (
        "newton",
        "isaaclab",
        "isaaclab_newton",
        "i4h_physics",
        "i4h_surgical",
        "i4h_isaaclab",
        "warp",
        "torch",
        "mujoco",
        "mujoco_warp",
    ):
        m = importlib.import_module(name)
        imports[name] = {"path": m.__file__, "version": getattr(m, "__version__", None)}
        root = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(m.__file__).parent),
                    "rev-parse",
                    "--show-toplevel",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if name in ("newton", "isaaclab", "i4h_physics")
            else None
        )
        if root:
            root = Path(root.strip())

            def git(*a):
                return subprocess.check_output(["git", "-C", str(root), *a], text=True)

            changed = set(
                git("diff", "--name-only", "HEAD").splitlines()
                + git("ls-files", "--others", "--exclude-standard").splitlines()
            )
            repos[name] = {
                "path": str(root),
                "commit": git("rev-parse", "HEAD").strip(),
                "dirty_sha256": {
                    f: sha(root / f) for f in sorted(changed) if (root / f).is_file()
                },
            }
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd))
    assets = {
        str(Path(layer.realPath)): sha(layer.realPath)
        for layer in stage.GetUsedLayers()
        if layer.realPath
    }
    data = Path(__import__("os").environ["I4H_DATA"])
    for folder in (data / "choly-mid", data / "choly/cavity"):
        assets.update({str(f): sha(f) for f in folder.rglob("*") if f.is_file()})
    warp = Path(importlib.import_module("warp").__file__).parent / "bin/warp.so"
    result = {
        "python": sys.executable,
        "imports": imports,
        "repositories": repos,
        "assets_sha256": assets,
        "warp_native_sha256": sha(warp),
        "rsl_rl": importlib.metadata.version("rsl-rl-lib"),
    }
    if result["rsl_rl"] != "5.5.1":
        raise RuntimeError("RSL-RL 5.5.1 required")
    Path(path).write_text(json.dumps(result, indent=2) + "\n")
