# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/fetch_assets.py: the production assets come from a folder or a URL, checked against the manifest."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_assets.py"


def fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_assets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(tmp_path):
    """A source tree of two assets and a manifest describing them (plus a package-data entry that is not fetched)."""
    files = {"textures/a.png": b"png bytes", "meshes/b.vtu": b"vtu bytes" * 100}
    source = tmp_path / "source"
    manifest = []
    for rel, data in files.items():
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_bytes(data)
        manifest.append({"path": "numerical_sim/data/" + rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    manifest.append({"path": "numerical_sim/endoluminal/src/i4h_endoluminal/data/x.yaml", "sha256": "0" * 64, "size": 1})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return source, path, files


@pytest.mark.parametrize("as_url", [False, True], ids=["folder", "url"])
def test_fetch_verifies_every_file(tmp_path, as_url):
    fa = fetch_module()
    source, manifest, files = bundle(tmp_path)
    dest = tmp_path / "data"
    dest.mkdir()
    src = source.as_uri() if as_url else str(source)
    assert sorted(fa.fetch(src, dest, manifest)) == sorted(files)
    assert all((dest / rel).read_bytes() == data for rel, data in files.items())
    assert fa.fetch(src, dest, manifest) == [], "files already present with the right hash are kept"
    assert fa.fetch(None, dest, manifest, check_only=True) == []


def test_fetch_rejects_missing_and_corrupt_files(tmp_path):
    fa = fetch_module()
    source, manifest, _ = bundle(tmp_path)
    dest = tmp_path / "data"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="2 asset"):
        fa.fetch(None, dest, manifest, check_only=True)
    (source / "textures/a.png").write_bytes(b"tampered")
    (source / "meshes/b.vtu").unlink()
    with pytest.raises(RuntimeError) as error:
        fa.fetch(str(source), dest, manifest)
    assert "a.png: the source file does not match" in str(error.value) and "b.vtu" in str(error.value)
    assert not any(dest.rglob("*.png")), "a mismatching file is never installed"


def test_manifest_covers_the_data_tree():
    fa = fetch_module()
    rels = [e["rel"] for e in fa.entries()]
    assert len(rels) == len(set(rels)) > 0 and all(not r.startswith("/") for r in rels)
