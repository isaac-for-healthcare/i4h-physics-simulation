# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch the production assets listed in ``docs/asset-manifest.json`` into a data directory, checking each file's
SHA-256 and size.

The assets are not in Git. ``--source`` is a local folder holding the data tree, or a base URL. Once the assets are
published through the Isaac for Healthcare asset catalog, that is the catalog's
``https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/<version>/<id>/...``
folder for this tree. Files already present with the right hash are kept. A missing or mismatching file fails.

    uv run python scripts/fetch_assets.py --source /path/to/numerical_sim_data   # into ./data (or $I4H_DATA)
    uv run python scripts/fetch_assets.py --check                                  # verify ./data only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # numerical_sim/
MANIFEST = ROOT / "docs" / "asset-manifest.json"
PREFIX = "numerical_sim/data/"  # manifest paths of the data tree; package data entries are not fetched


def entries(manifest: Path = MANIFEST) -> list[dict]:
    """The manifest entries of the data tree, with ``rel``: the path below the data directory."""
    return [dict(e, rel=e["path"][len(PREFIX):]) for e in json.loads(manifest.read_text()) if e["path"].startswith(PREFIX)]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def valid(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["size"] and digest(path) == entry["sha256"]


def fetch(source: str | None, dest: Path, manifest: Path = MANIFEST, check_only: bool = False) -> list[str]:
    """Fetch every data entry into ``dest`` (or only verify it); returns the paths fetched. Raises ``RuntimeError``
    listing every file that is missing or does not match the manifest."""
    fetched, problems = [], []
    for e in entries(manifest):
        target = dest / e["rel"]
        if valid(target, e):
            continue
        if check_only or source is None:
            problems.append(f"{e['rel']}: {'missing' if not target.exists() else 'sha256/size mismatch'}")
            continue
        try:
            with tempfile.NamedTemporaryFile(dir=dest if dest.exists() else None, delete=False) as tmp:
                if "://" in source:
                    with urllib.request.urlopen(f"{source.rstrip('/')}/{e['rel']}") as response:
                        shutil.copyfileobj(response, tmp)
                else:
                    with open(Path(source) / e["rel"], "rb") as f:
                        shutil.copyfileobj(f, tmp)
            if not valid(Path(tmp.name), e):
                problems.append(f"{e['rel']}: the source file does not match the manifest's sha256/size")
                os.unlink(tmp.name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp.name, target)
            fetched.append(e["rel"])
        except OSError as error:  # urllib.error.URLError is an OSError
            problems.append(f"{e['rel']}: {error}")
    if problems:
        raise RuntimeError(f"{len(problems)} asset(s) unavailable in {dest}:\n  " + "\n  ".join(problems))
    return fetched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="a local data tree or a base URL (the asset catalog folder once published)")
    parser.add_argument("--dest", default=os.environ.get("I4H_DATA") or str(ROOT / "data"),
                        help="the data directory (default: $I4H_DATA, else numerical_sim/data)")
    parser.add_argument("--check", action="store_true", help="only verify --dest against the manifest")
    args = parser.parse_args(argv)
    if not args.check and not args.source:
        parser.error("--source is required unless --check")
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        fetched = fetch(args.source, dest, check_only=args.check)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{dest}: {len(entries())} assets verified ({len(fetched)} fetched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
