# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Particle prediction and the substep finalize (XPBD outer loop).

One live ``max_velocity`` bounds both ends of the substep: ``predict`` clips the predicted
velocity, so constraints and contacts see a bounded position, and ``finalize`` clips the
final displacement and derives the velocity from it (omnisurg's separate motion-limit pass
did the latter right after the velocity update, overwriting it). Clipping only at the end
would let a contact push-out be undone by the clip and leave the particle inside the obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.checks import check_bool
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System


@dataclass(eq=False)
class Particles(System):
    damping: float = 0.97  # live, per env
    max_velocity: float = 1000.0  # live, per env; bounds the predicted velocity and the substep displacement

    body: object = None  # None retains the whole main block; otherwise one body
    gravity_scale: float = 1.0  # live per env, e.g. a neutrally buoyant wall
    damping_at_finalize: bool = False  # apply damping to the projected velocity

    structural = System.structural + ("body", "damping_at_finalize")

    def __post_init__(self):
        self.body = None if self.body is None else self.scene.body(self.body)
        if self.body is not None and self.body.pool:
            raise ValueError("particles.body must belong to the main block")
        self.damping_at_finalize = check_bool("particles.damping_at_finalize", self.damping_at_finalize)
        self.freeze("body", "damping_at_finalize")
        self._live("gravity_scale")
        self._live("damping")
        self._live("max_velocity")


    def substep_start(self, scene, dt, s, substeps):
        count = scene.main.width if self.body is None else self.body.count
        if count:  # the whole main block in one launch (fluid pools are the fluid system's, in their own block)
            f = scene.block_field if self.body is None else lambda name: scene.field(name, self.body)
            wp.launch(
                predict,
                dim=(scene.envs, count),
                inputs=[f("qd"), f("inv_mass"), f("flags"), scene.gravity, self.gravity_scale, self.damping, int(self.damping_at_finalize), self.max_velocity, dt],
                outputs=[f("q"), f("q_prev")],
                device=scene.device,
            )

    def substep_end(self, scene, dt):
        count = scene.main.width if self.body is None else self.body.count
        if count:
            f = scene.block_field if self.body is None else lambda name: scene.field(name, self.body)
            wp.launch(
                finalize,
                dim=(scene.envs, count),
                inputs=[f("q_prev"), f("inv_mass"), f("flags"), self.damping, int(self.damping_at_finalize), self.max_velocity, dt],
                outputs=[f("q"), f("qd")],
                device=scene.device,
            )


# Ported from omnisurg/tet/kernels/tet.py:1473-1503 (integrate_particles_range).
# Edits: (env, local) prologue; force array dropped; 0.97 -> damping[env]; gravity[env];
# v_max -> max_velocity[env] (the same live value finalize uses); qd_out dropped because
# finalize recomputes velocity from positions; the substep-start position is saved to
# q_prev here instead of by a separate copy (omnisurg's engine snapshotted state_prev
# before the launch).
@wp.kernel
def predict(
    qd: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    gravity_scale: wp.array(dtype=wp.float32),
    damping: wp.array(dtype=wp.float32),
    damping_at_finalize: wp.int32,
    max_velocity: wp.array(dtype=wp.float32),
    dt: wp.float32,
    q: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
):
    env, i = wp.tid()
    q0 = q[env, i]
    q_prev[env, i] = q0
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        return

    factor = float(1.0)
    if damping_at_finalize == 0:
        factor = damping[env]
    v1 = (qd[env, i] + gravity[env] * gravity_scale[env] * dt) * factor
    v_max = max_velocity[env]
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 = v1 * (v_max / v1_mag)

    q[env, i] = q0 + v1 * dt


# Ported from omnisurg/tet/kernels/tet.py:1898-1917 (update_velocities_from_positions_range)
# merged with omnisurg/physics/solver.py:10-40 (clip_particle_motion_from_anchors), which ran
# right after it and overwrote its velocity. Edits: (env, local) prologue; max_velocity[env];
# inactive particles get zero velocity.
@wp.kernel
def finalize(
    q_prev: wp.array2d(dtype=wp.vec3),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    damping: wp.array(dtype=wp.float32),
    damping_at_finalize: wp.int32,
    max_velocity: wp.array(dtype=wp.float32),
    dt: wp.float32,
    q: wp.array2d(dtype=wp.vec3),
    qd: wp.array2d(dtype=wp.vec3),
):
    env, i = wp.tid()
    if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
        qd[env, i] = wp.vec3(0.0, 0.0, 0.0)
        return

    displacement = q[env, i] - q_prev[env, i]
    distance = wp.length(displacement)
    max_delta = max_velocity[env] * dt
    if max_delta > 0.0 and distance > max_delta:
        displacement = displacement * (max_delta / distance)
        q[env, i] = q_prev[env, i] + displacement
    velocity = displacement / dt
    if damping_at_finalize != 0:
        velocity *= damping[env]
    qd[env, i] = velocity
