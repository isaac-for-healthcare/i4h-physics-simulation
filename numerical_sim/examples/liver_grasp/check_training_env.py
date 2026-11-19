# SPDX-License-Identifier: Apache-2.0
"""Validate the overlay together with its inherited, preserved numerical packages."""

from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

requirements = Path(__file__).with_name("requirements-training.txt")
checked = []
for line in requirements.read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    pinned = Requirement(line)
    assert metadata.version(pinned.name) in pinned.specifier, line
    for raw in metadata.requires(pinned.name) or []:
        dependency = Requirement(raw)
        if dependency.marker and not dependency.marker.evaluate():
            continue
        version = metadata.version(dependency.name)
        assert version in dependency.specifier, (
            f"{pinned.name}: {dependency} (installed {version})"
        )
    checked.append(line)
print(
    f"{len(checked)} added dependency pins and their direct requirements passed with inherited packages"
)
