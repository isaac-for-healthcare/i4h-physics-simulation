# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Suction at an instrument tip: removal inside ``removal_radius``, attraction inside ``radius``.

omnisurg apply_fluid_suction_range (fluids/kernels/pbf.py:1090-1149) with one tool per system over every
pool of the fluid: while the instrument is energised (its energy column, Shift on the keyboard, the
``secondary`` button on a device; omnisurg's suction button), a particle within ``removal_radius`` of the
tip is killed (ACTIVE cleared, velocity zeroed, parked back at its rest slot ``q0`` for the ring to reuse)
and a particle within ``radius`` gets the impulse ``unit(tip - p) * power * (1 - d / radius) * dt`` on its
velocity and position. Runs in ``substep_end`` after the fluid's finalize (list order), so the impulse
survives into the next predict. The tip is ``instrument_control.root`` (the tool's controlled point, the
sphere centre of a ``kind: sphere`` probe). ``removed`` (E, nf) counts kills per pool since reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.scene import ACTIVE, INACTIVE
from i4h_physics.systems.base import System
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_physics.systems.fluids.fluid_kernels import pool_slot
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


@dataclass(eq=False)
class Suction(System):
    state_products = ("removed",)  # per-env products a host observes (System.state_products)
    fluid: Fluid
    instruments: InstrumentControl
    instrument: str  # structural: the sucking tool
    radius: float = 0.10  # live, per env: attraction radius
    removal_radius: float = 0.01  # live, per env
    power: float = 2.0  # live, per env: attraction impulse scale

    refs = ("fluid", "instruments")
    structural = System.structural + ("instrument",)

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.fluid, Fluid):
            raise TypeError("suction.fluid must name a fluid system")
        if not isinstance(self.instruments, InstrumentControl):
            raise TypeError("suction.instruments must name the instrument_control system")
        if self.instrument not in self.instruments.names:
            raise ValueError(f"suction.instrument {self.instrument!r}: known {self.instruments.names}")
        self.k = self.instruments.names.index(self.instrument)
        self.freeze("instrument")
        self._live("radius", "removal_radius", "power")
        self.removed = wp.zeros((scene.envs, self.fluid.nf), dtype=wp.int32, device=scene.device)


    def substep_end(self, scene, dt):
        ic, fl, c = self.instruments, self.fluid, self.fluid.const
        wp.launch(suck, dim=(scene.envs, fl.total),
                  inputs=[ic.root, ic.energized, self.k, self.radius, self.removal_radius, self.power,
                          fl.q0_span, c["prefix"], c["counts"], c["base"], fl.nf, dt],
                  outputs=[self.removed, fl.q_span, fl.q_prev_span, fl.qd_span, fl.flags_span], device=scene.device)

    def reset(self, scene, mask):
        self.clear(mask, (self.removed, wp.int32(0)))


@wp.kernel
def suck(
    root: wp.array2d(dtype=wp.vec3),
    energized: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    radius: wp.array(dtype=wp.float32),
    removal_radius: wp.array(dtype=wp.float32),
    power: wp.array(dtype=wp.float32),
    q0: wp.array(dtype=wp.vec3),
    prefix: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    nf: wp.int32,
    dt: wp.float32,
    removed: wp.array2d(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    if energized[env, k] == 0:
        return
    ps = pool_slot(local, prefix, nf)
    f = ps[0]
    li = base[f] + env * counts[f] + ps[1]
    if (flags[li] & ACTIVE) == 0:
        return
    position = q[li]
    to_tip = root[env, k] - position
    distance = wp.length(to_tip)
    if distance <= removal_radius[env]:
        flags[li] = flags[li] & INACTIVE
        qd[li] = wp.vec3(0.0, 0.0, 0.0)
        q[li] = q0[li]
        q_prev[li] = q0[li]
        wp.atomic_add(removed, env, f, 1)
        return
    r = radius[env]
    if distance > 1.0e-8 and distance < r:
        impulse = to_tip / distance * power[env] * (1.0 - distance / r) * dt
        qd[li] = qd[li] + impulse
        q[li] = position + impulse * dt
