# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check the reviewed source inventory without importing simulation packages.

This validates coverage and provenance, not scratch semantics or continuation.
New allocations or changed source/contracts require an explicit review update.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from inventory_stage_c import PACKAGES, census


def verify(root, inventory):
    errors = []
    for name, digest in inventory["review_documents"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"review document changed: {name}")
    paths = sorted(p for package in PACKAGES for p in (root / package / "src").rglob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in sorted(hashes.keys() | inventory["source_sha256"].keys()):
        if hashes.get(name) != inventory["source_sha256"].get(name):
            errors.append(f"source changed: {name}")
    sites = [site for path in paths for site in census(path, root)]

    def identity(site):
        return site["path"], site["line"], site["scope"], site["expression"]

    actual = Counter(identity(site) for site in sites)
    reviewed = Counter(identity(site) for site in inventory["sites"])
    if actual != reviewed:
        errors.append(f"allocation coverage differs: {sum((actual - reviewed).values())} missing, "
                      f"{sum((reviewed - actual).values())} stale")
    for site in inventory["sites"]:
        refs = site["contracts"]
        if not refs or any(ref not in inventory["contracts"] for ref in refs):
            errors.append(f"missing field contract: {site['path']}:{site['line']}")
    if inventory.get("unresolved"):
        errors.append("unresolved allocation sites remain")
    return errors, len(sites), len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    path = args.inventory or args.root / "docs/stage-c-inventory.json"
    inventory = json.loads(path.read_text())
    errors, sites, files = verify(args.root, inventory)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {sites} allocation/helper sites in {files} source files; contracts and source hashes match")


if __name__ == "__main__":
    main()
