# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import warp as wp
from i4h_physics.systems.mechanics.surface import Surface as PhysicalSurface


class Appearance:
    """Colour per particle (``visual``, ``(E * count,)`` vec4): Newton State on the surface's particle rows."""

    def __init__(self, system):
        from i4h_surgical.interaction_storage import bind_interaction
        self.count = system.body.count
        bind_interaction(system, "appearance", target=self, body=system.body)

    def reset(self, system, mask):
        system.clear(mask, (self.visual.reshape((system.scene.envs, self.count)), wp.vec4f(0.0)))

@dataclass(eq=False)
class Surface(PhysicalSurface):
    @classmethod
    def declare_storage(cls, authoring, declarations, spec, records):
        from i4h_surgical.interaction_storage import publish_appearance
        record = super().declare_storage(authoring, declarations, spec, records)
        publish_appearance(authoring.scene, declarations, spec.get("id", spec["type"]))
        return record

    def __post_init__(self):
        from i4h_physics.systems.mechanics.tet import Tet
        if not isinstance(self.tet, Tet):
            raise TypeError("surface.tet must name a tet system")
        super().__post_init__()
        self.appearance = Appearance(self)

    def reset(self, scene, mask):
        super().reset(scene, mask)
        self.appearance.reset(self, mask)
