# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import warp as wp
from i4h_physics.systems.mechanics.surface import Surface as PhysicalSurface


class Appearance:
    def __init__(self, scene, count):
        scene.check_memory({"appearance": scene.envs * count * 16})
        self.visual = wp.zeros(scene.envs * count, dtype=wp.vec4f, device=scene.device)
        self.count = count

    def reset(self, system, mask):
        system.clear(mask, (self.visual.reshape((system.scene.envs, self.count)), wp.vec4f(0.0)))

@dataclass(eq=False)
class Surface(PhysicalSurface):
    def __post_init__(self):
        from i4h_physics.systems.mechanics.tet import Tet
        if not isinstance(self.tet, Tet):
            raise TypeError("surface.tet must name a tet system")
        self.appearance = Appearance(self.scene, self.tet.body.count)
        super().__post_init__()

    def reset(self, scene, mask):
        super().reset(scene, mask)
        self.appearance.reset(self, mask)
