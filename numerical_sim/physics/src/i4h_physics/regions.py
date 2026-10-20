# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable material membership and persistent region tables, separate from control products."""
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class RegionMembership:
    names: tuple[str, ...]
    edge: np.ndarray
    sources: tuple = ()  # source branch IDs per welded edge; empty for artificial stitching


class RodControls:
    def __init__(self, scene, segments, pairs):
        self.stretch_scale = wp.ones((scene.envs, segments), dtype=float, device=scene.device)
        self.bend_scale = wp.ones((scene.envs, pairs), dtype=float, device=scene.device)
        self.bend_offset = wp.zeros((scene.envs, pairs), dtype=wp.vec3, device=scene.device)


class RegionProperties:
    def __init__(self, scene, membership, fields, rest_len=None, rest_darboux=None):
        self.membership = membership
        self.edge = wp.array(membership.edge, dtype=wp.int32, device=scene.device)
        self.tables = {name: wp.full((scene.envs, len(membership.names)), default, dtype=float, device=scene.device)
                       for name, default in fields.items()}
        self.defaults = fields.copy()
        self.rest_len, self.rest_darboux = rest_len, rest_darboux

    def validate(self, values):
        for name, table in values.items():
            if name not in self.tables or table.shape != self.tables[name].shape or not np.isfinite(table).all():
                raise ValueError(f'invalid regional property {name}')
            if not name.startswith('curvature') and np.any((table < 0) | (table > 10)):
                raise ValueError(f'{name}: regional multipliers must be in [0, 10]')
        if 'curvature_x' in self.tables:
            cx = (values['curvature_x'] if 'curvature_x' in values else self.tables['curvature_x'].numpy())[:, self.membership.edge]
            cy = (values['curvature_y'] if 'curvature_y' in values else self.tables['curvature_y'].numpy())[:, self.membership.edge]
            angles = np.hypot(cx, cy) * self.rest_len[None]
            if np.any(angles >= np.pi):
                raise ValueError('curvature edge rotation must be below pi')
            # Check the composed quaternion, including curvature that cancels an authored bend.
            rd = self.rest_darboux.astype(np.float64)
            w = np.sqrt(np.maximum(0, 1 - np.sum(rd * rd, axis=1)))
            factor = np.divide(np.sin(angles / 2), angles, out=np.full_like(angles, .5), where=angles != 0)
            composed_w = w[None] * np.cos(angles / 2) - (rd[None, :, 0] * cx + rd[None, :, 1] * cy) * self.rest_len[None] * factor
            if np.any(composed_w <= 0):
                raise ValueError('combined authored and curvature edge rotation must be below pi')

    def edit(self, changes, region, envs=None):
        """Between steps only. Validate the entire edit before writing any table."""
        index = self.membership.names.index(region)
        E = next(iter(self.tables.values())).shape[0]
        ids = np.arange(E) if envs is None else np.asarray(envs)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any((ids < 0) | (ids >= E)):
            raise ValueError('invalid environment selection')
        values = {}
        for name, value in changes.items():
            if name not in self.tables:
                raise ValueError(f'unknown regional property {name}')
            values[name] = self.tables[name].numpy().copy()
            values[name][ids, index] = value
        self.validate(values)
        for name, value in values.items():
            source = wp.array(value, dtype=float, device='cpu')
            width = value.shape[1]
            for env in ids:
                offset = int(env) * width + index
                wp.copy(self.tables[name], source, dest_offset=offset, src_offset=offset, count=1)

    def restore(self, envs=None):
        for region in self.membership.names:
            self.edit(self.defaults, region, envs)


def material_system(sim, sid):
    """Resolve a material ID or the legacy solver ID for the same body."""
    from i4h_physics.systems.rods.material import RodDirectMaterial, RodPbdMaterial
    from i4h_physics.systems.rods.rod_direct import RodDirect
    from i4h_physics.systems.rods.rod_pbd import RodPbd
    system = next((s for s in sim.systems if s.id == sid), None)
    if isinstance(system, (RodPbdMaterial, RodDirectMaterial)):
        return system
    if isinstance(system, (RodPbd, RodDirect)):
        return next((s for s in sim.systems if isinstance(s, (RodPbdMaterial, RodDirectMaterial))
                     and s.body is system.body), None)
    return None


def region_override_tables(sim, declaration, mirrors=None):
    """Validate and merge legacy/canonical references without partial application."""
    result, seen = {}, {}
    for body_name, systems in (declaration or {}).items():
        body = sim.scene.body(body_name)
        props = body.region_properties
        if props is None:
            raise ValueError(f'{body_name}: no regional properties')
        for sid, regions in systems.items():
            provider = material_system(sim, sid)
            if provider is None or provider.body is not body:
                raise ValueError(f'{body_name}: invalid regional system {sid}')
            if body_name not in result:
                result[body_name] = (props, {n: (mirrors[(provider.id, 'region_' + n)].copy()
                                               if mirrors is not None else a.numpy().copy())
                                            for n, a in props.tables.items()})
            tables = result[body_name][1]
            for region, changes in regions.items():
                if region not in props.membership.names:
                    raise ValueError(f'unknown region {region}')
                index = props.membership.names.index(region)
                for name, value in changes.items():
                    arr = np.asarray(value, np.float64)
                    if name not in tables or arr.shape not in ((), (sim.scene.envs,)) or not np.isfinite(arr).all():
                        raise ValueError(f'{sid}/{region}/{name}: expected a scalar or {sim.scene.envs} finite values')
                    key = body_name, region, name
                    values = np.broadcast_to(arr, (sim.scene.envs,))
                    if key in seen and not np.array_equal(seen[key], values):
                        raise ValueError(f'conflicting regional declarations for {body_name}/{region}/{name}')
                    seen[key] = values.copy()
                    tables[name][:, index] = arr
    for props, tables in result.values():
        props.validate(tables)
    return list(result.values())
