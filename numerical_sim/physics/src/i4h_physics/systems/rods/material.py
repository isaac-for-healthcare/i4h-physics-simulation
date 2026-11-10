# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit regional providers. Tables persist across episode resets and disabling."""
from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.regions import RegionMembership
from i4h_physics.systems.base import System
from i4h_physics.systems.rods.region_kernels import direct_regions, pbd_regions


def _membership(body, edges):
    membership = body.regions
    if membership is None:
        edge = np.zeros(edges, np.int32)
        edge.flags.writeable = False
        body.regions = membership = RegionMembership(('unassigned',), edge)
    if membership.edge.shape != (edges,) or np.any((membership.edge < 0) | (membership.edge >= len(membership.names))):
        raise ValueError('rod material: invalid membership dimensions or region indices')
    return membership


@dataclass(eq=False)
class RodPbdMaterial(System):
    body: str
    radius_weighting: bool = False

    structural = System.structural + ('radius_weighting',)

    @classmethod
    def declare_storage(cls, authoring, declarations, spec, records):
        from i4h_physics.rod_storage import publish_material
        return publish_material(authoring.scene, declarations, spec.get("id", spec["type"]),
                                body=spec["body"], backend="pbd", settings=spec,
                                solver_specs=[dict(body=s['body'], backend=s['type'][4:])
                                              for s in authoring.cfg['systems'] if s['type'] in ('rod_pbd', 'rod_direct')])

    def __post_init__(self):
        from i4h_physics.rod_storage import bind_material
        self.body = self.scene.body(self.body)
        if self.body.rod_material is not None or self.body.rod_controls is not None:
            raise ValueError('rod material: duplicate provider or material must precede solver')
        bind_material(self, "pbd")

    def prepare(self, scene):
        scene.storage.check()
        super().prepare(scene)
        self.body.rod_material.active = self.enabled


    def substep_start(self, scene, dt, s, substeps):
        if s != 0:
            return
        m, p, c = self.body.rod_material, self.body.region_properties, self.body.rod_controls
        wp.launch(pbd_regions, dim=(scene.envs, max(m.S, m.K)),
                  inputs=[p.edge, m.pair, p.tables['stretch'], p.tables['bend'], p.tables['twist'],
                          m.radius_baseline, int(self.radius_weighting), c.stretch_scale, c.bend_scale, m.S, m.K],
                  outputs=[m.stretch, m.bend, m.twist], device=scene.device)


@dataclass(eq=False)
class RodDirectMaterial(System):
    body: str

    @classmethod
    def declare_storage(cls, authoring, declarations, spec, records):
        from i4h_physics.rod_storage import publish_material
        return publish_material(authoring.scene, declarations, spec.get("id", spec["type"]),
                                body=spec["body"], backend="direct", settings=spec,
                                solver_specs=[dict(body=s['body'], backend=s['type'][4:])
                                              for s in authoring.cfg['systems'] if s['type'] in ('rod_pbd', 'rod_direct')])

    def __post_init__(self):
        from i4h_physics.rod_storage import bind_material
        self.body = self.scene.body(self.body)
        if self.body.rod_material is not None or self.body.rod_controls is not None:
            raise ValueError('rod material: duplicate provider or material must precede solver')
        bind_material(self, "direct")

    def prepare(self, scene):
        scene.storage.check()
        super().prepare(scene)
        self.body.rod_material.active = self.enabled


    def substep_start(self, scene, dt, s, substeps):
        if s != 0:
            return
        m, p = self.body.rod_material, self.body.region_properties
        wp.launch(direct_regions, dim=(scene.envs, m.Ed),
                  inputs=[p.edge, m.rest_len, m.rest_darboux,
                          *[p.tables[n] for n in ('young_multiplier', 'torsion_multiplier', 'bend_multiplier',
                                                 'twist_multiplier', 'curvature_x', 'curvature_y')]],
                  outputs=[m.young_multiplier, m.torsion_multiplier, m.bend_multiplier, m.twist_multiplier, m.preferred_rest],
                  device=scene.device)


def validate_material_systems(scene, systems):
    """Check the explicit list and shared channel dimensions before any capture."""
    from i4h_physics.systems.rods.rod_direct import RodDirect
    from i4h_physics.systems.rods.rod_pbd import RodPbd
    providers, solvers = {}, {}
    for i, system in enumerate(systems):
        if isinstance(system, (RodPbdMaterial, RodDirectMaterial)):
            name = system.body.name
            if name in providers:
                raise ValueError('rod material: duplicate provider')
            providers[name] = (i, system)
        elif isinstance(system, (RodPbd, RodDirect)):
            name = system.body.name
            if name in solvers:
                raise ValueError('rod material: duplicate solver for shared channels')
            solvers[name] = system
            provider = providers.get(name)
            if system.body.regions is not None and provider is None:
                raise ValueError('rod material: matching material must precede solver')
            if provider is not None:
                expected = RodPbdMaterial if isinstance(system, RodPbd) else RodDirectMaterial
                if not isinstance(provider[1], expected):
                    raise ValueError('rod material: mismatched backend')
    for name, (_, provider) in providers.items():
        if name not in solvers:
            raise ValueError('rod material: missing matching solver')
        b, rod = provider.body, solvers[name]
        m, c, p = b.rod_material, b.rod_controls, b.region_properties
        S, K = (rod.S, rod.K) if isinstance(rod, RodPbd) else (rod.Ed, rod.Ed)
        arrays = [(c.stretch_scale, (scene.envs, S)), (c.bend_scale, (scene.envs, K)),
                  (c.bend_offset, (scene.envs, K)), (p.edge, (S,))]
        arrays += [(a, (scene.envs, len(p.membership.names))) for a in p.tables.values()]
        if isinstance(rod, RodPbd):
            arrays += [(m.stretch, (scene.envs, S)), (m.bend, (scene.envs, K)), (m.twist, (scene.envs, K)),
                       (m.pair, (1, K)), (m.radius_baseline, (S,))]
        else:
            arrays += [(getattr(m, n), (scene.envs, S)) for n in
                       ('young_multiplier', 'torsion_multiplier', 'bend_multiplier', 'twist_multiplier', 'preferred_rest')]
            arrays += [(m.rest_len, (1, S)), (m.rest_darboux, (1, S))]
        if any(a.shape != shape or a.device != scene.device for a, shape in arrays):
            raise ValueError('rod material: invalid array dimensions or device')
