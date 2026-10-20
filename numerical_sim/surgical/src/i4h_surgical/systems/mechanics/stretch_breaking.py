# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import warp as wp
from i4h_physics.systems.mechanics.stretch_breaking import StretchBreaking as PhysicalBreaking


@dataclass(eq=False)
class StretchBreaking(PhysicalBreaking):
    damage_start: float = 4.0
    damage_full: float = 5.5
    blood: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self._live("damage_start", "damage_full", "blood")

    def step_end(self, scene, dt):
        b = self.body
        wp.launch(paint_damage, dim=(scene.envs, b.count),
                  inputs=[self.threshold, self.damage_start, self.damage_full, self.blood, b.count],
                  outputs=[self.stretch, self.surface.appearance.visual, self.max_stretch], device=scene.device)


# Ported from omnisurg/tet/kernels/tet.py:856-903; launch order preserved.
@wp.kernel
def paint_damage(
    threshold: wp.array(dtype=wp.float32),
    damage_start: wp.array(dtype=wp.float32),
    damage_full: wp.array(dtype=wp.float32),
    blood: wp.array(dtype=wp.float32),
    count: wp.int32,
    stretch_p: wp.array(dtype=wp.float32),
    visual: wp.array(dtype=wp.vec4f),
    max_stretch: wp.array(dtype=wp.float32),
):
    env, local = wp.tid()
    flat_index = env * count + local
    s = stretch_p[flat_index]
    stretch_p[flat_index] = 0.0
    if s <= 0.0:
        return
    wp.atomic_max(max_stretch, env, s)
    damage = wp.clamp((s - damage_start[env]) / wp.max(damage_full[env] - damage_start[env], 1.0e-6), 0.0, 1.0)
    state = visual[flat_index]
    next_damage = wp.max(state[0], damage)
    next_blood = state[2]
    if s > threshold[env]:
        next_blood = wp.clamp(next_blood + wp.max(blood[env], 0.0), 0.0, 1.0)
    if next_damage > state[0] + 1.0e-6 or next_blood > state[2] + 1.0e-6:
        visual[flat_index] = wp.vec4f(next_damage, state[1], next_blood, state[3])
