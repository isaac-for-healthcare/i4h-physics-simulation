# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect every wheel/sdist for package isolation and large-asset exclusion."""
import sys
import tarfile
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
packages = ("physics", "isaaclab", "standalone", "endoluminal", "surgical", "assessment")
small = {"i4h_endoluminal/data/vascular_templates.yaml", "i4h_endoluminal/data/instrument_regions.yaml"}
for name in packages:
    package = f"i4h_{name}"
    for artifact in (root / f"{package}-0.9.0-py3-none-any.whl", root / f"{package}-0.9.0.tar.gz"):
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                files = [(i.filename, i.file_size) for i in archive.infolist() if not i.is_dir()]
            roots = {path.split('/')[0] for path, _ in files}
            assert roots == {package, f"{package}-0.9.0.dist-info"}, (artifact, roots)
        else:
            with tarfile.open(artifact) as archive:
                files = [(i.name, i.size) for i in archive.getmembers() if i.isfile()]
            source_roots = {path.split("/src/", 1)[1].split("/")[0] for path, _ in files if "/src/" in path}
            assert source_roots <= {package, f"{package}.egg-info"}, (artifact, source_roots)
        for path, size in files:
            assert "i4h_sim/" not in path and "numerical_sim/data/" not in path, (artifact, path)
            if "/data/" in path:
                assert any(path.endswith(allowed) for allowed in small), (artifact, path)
            assert size < 5_000_000, (artifact, path, size)
        print(f"{artifact.name}: {len(files)} files; package and asset boundaries passed")
