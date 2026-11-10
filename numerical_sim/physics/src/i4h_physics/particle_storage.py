# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exact native particle defaults and history, authored before finalization."""
import newton
import warp as wp
from i4h_physics.newton_model import main_rows
from i4h_physics.storage import Domain, Field, Owner


def declare_particles(declarations, scene):
    domain = Domain("scene", "particle", newton.Model.AttributeFrequency.PARTICLE,
                    (scene.envs, scene.main.width))
    declarations.domain(domain)
    declarations.add(Field(domain, "q0", Owner.MODEL, wp.vec3, domain.shape,
                           "world, main particle", "native Model rest positions", native_attribute="particle_q"))
    for key, source, owner, dtype in (
        ("inv_mass", "inv_mass", Owner.MODEL, wp.float32),
        ("flags0", "flags", Owner.MODEL, wp.int32),
        ("q_prev", "q0", Owner.STATE, wp.vec3),
        ("flags", "flags", Owner.STATE, wp.int32),
    ):
        declarations.add(Field(domain, key, owner, dtype, domain.shape,
                               "world, main particle", "selected-world authored defaults"), main_rows(scene, source))
    return domain
