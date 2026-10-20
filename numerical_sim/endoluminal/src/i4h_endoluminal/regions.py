# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import yaml
from i4h_physics.regions import RegionMembership


def region_sidecar(value, kind):
    data = value if isinstance(value, dict) else yaml.safe_load(Path(value).read_text())
    key = 'branches' if kind == 'vessel' else 'regions'
    if not isinstance(data, dict) or set(data) - {'schema_version', 'default', key} or data.get('schema_version') != 1:
        raise ValueError('regions: expected schema_version: 1 and a valid region mapping')
    default = data.get('default', 'unassigned')
    if not isinstance(default, str) or not default.strip():
        raise ValueError('regions.default must be a nonempty name')
    return data, default


def instrument_regions(value, lengths):
    data, default = region_sidecar(value, 'instrument')
    names, intervals = [default], []
    for row in data.get('regions', []):
        if not isinstance(row, dict) or set(row) != {'name', 'interval'}:
            raise ValueError('instrument region needs name and interval')
        name = row['name']
        bounds = np.asarray(row['interval'], dtype=float)
        if not isinstance(name, str) or not name.strip() or name in names or bounds.shape != (2,) or not np.isfinite(bounds).all():
            raise ValueError('invalid or duplicate instrument region')
        a, b = bounds
        if not 0 <= a < b <= 1 or any(a < d and c < b for c, d, _ in intervals):
            raise ValueError('instrument region bounds invalid or overlapping')
        names.append(name)
        intervals.append((a, b, len(names) - 1))
    lengths = np.asarray(lengths, dtype=np.float64)
    midpoint = (np.cumsum(lengths) - 0.5 * lengths) / lengths.sum()
    edge = np.zeros(len(lengths), np.int32)
    for a, b, index in intervals:
        edge[(midpoint >= a) & ((midpoint < b) | ((b == 1) & (midpoint == 1)))] = index
    edge.flags.writeable = False
    return RegionMembership(tuple(names), edge)


def vessel_regions(value, known):
    data, default = region_sidecar(value, 'vessel')
    mapping = data.get('branches', {})
    if not isinstance(mapping, dict) or set(mapping) - set(known):
        raise ValueError('regions: unknown branch IDs')
    if any(not isinstance(n, str) or not n.strip() for n in mapping.values()):
        raise ValueError('regions: region names must be nonempty strings')
    names = tuple(dict.fromkeys([default, *mapping.values()]))
    return names, {b: names.index(mapping.get(b, default)) for b in known}
