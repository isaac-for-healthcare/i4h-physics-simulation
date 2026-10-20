# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cautery smoke: ballistic billboard particles born where the energised tool touches hot tissue.

omnisurg kept smoke in the renderer (rendering/particles.py ``GpuSmokeParticleSystem``, 670-1052): a
device ring buffer of ``max_particles`` with three vec4 arrays (position + age, velocity + lifetime,
start size + end size + active), spawned from a thermal event counter and updated by a captured kernel.
Here two arrays: a particle is alive while ``age < lifetime`` (a dead slot has lifetime 0) and the sizes
are system-wide values the renderer takes as uniforms.
It touches no ``Scene`` array, but it runs inside the step with per-env state and a masked reset, so
here it is a system. The trigger follows omnisurg's (tet/kernels/tet.py:594-596): while the instrument
is energised, a tip contact record whose tet has reached ``thermal.coag_heat`` is a smoke source; its
point is the record's barycentric point on the tissue (records carry no position). ``rate`` particles
per second are released from the env's ring with omnisurg's spawn distribution (particles.py:330-365:
a ball offset, a random direction times ``speed``, an ``upward`` component against gravity, a lifetime
in range); ``update`` (particles.py:299-327) advects and expires them. Everything runs in ``step_end``,
after the last substep's contact event and thermal deposit, so the source is this step's contact (the
records are cleared at every ``substep_start``). ``texture``, the sizes and the colours are read by the
renderer only. Not ported: the camera-heading bias of the spawn velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_float, check_int
from i4h_physics.systems.base import System
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.fluids.emitter import take
from i4h_surgical.systems.instruments.instrument_control import PRIM_KINDS, InstrumentControl
from i4h_surgical.systems.instruments.thermal import Thermal


@dataclass(eq=False)
class Smoke(System):
    state_products = ("alive_count",)  # per-env products a host observes (System.state_products)
    instruments: InstrumentControl
    contacts: InstrumentContact
    thermal: Thermal
    instrument: str  # structural: the cautery tool
    count: int = 512  # structural: particles per env (omnisurg max_particles)
    max_spawn: int = 16  # structural: particles per env and step at most
    rate: float = 600.0  # live, per env: particles per second while the tool smokes
    lifetime: tuple = (0.1, 0.2)  # live, per env: seconds, uniform in range (omnisurg examples/smoke.yaml)
    speed: tuple = (0.02, 0.10)  # live, per env
    upward: tuple = (0.05, 0.18)  # live, per env: added against gravity
    spawn_radius: float = 0.025  # live, per env
    texture: str | None = None  # sprite (omnisurg examples/smoke.yaml: smoke1.png); the renderer reads the five below
    start_size: float = 0.01
    end_size: float = 0.2
    start_color: tuple = (0.80, 0.78, 0.72, 0.005)
    end_color: tuple = (0.80, 0.78, 0.72, 0.0)

    refs = ("instruments", "contacts", "thermal")
    structural = System.structural + ("instrument", "count", "max_spawn")

    def __post_init__(self):
        scene = self.scene
        ic, cc, th = self.instruments, self.contacts, self.thermal
        if not isinstance(ic, InstrumentControl) or not isinstance(cc, InstrumentContact) or not isinstance(th, Thermal):
            raise TypeError("smoke.instruments, .contacts and .thermal must name an instrument_control, an instrument_contact and a thermal system")
        if self.instrument not in ic.names:
            raise ValueError(f"smoke: unknown instrument {self.instrument!r}; known: {ic.names}")
        self.count = check_int("smoke.count", self.count, 1)
        self.max_spawn = check_int("smoke.max_spawn", self.max_spawn, 1)
        self.k = k = ic.names.index(self.instrument)
        kinds, owners = ic.host_prim_kind, ic.host_prim_instrument
        chain = (kinds == PRIM_KINDS["wrist"]) | (kinds == PRIM_KINDS["shaft"])
        self.source_prim = wp.array(np.int32((owners == k) & ~chain), dtype=wp.int32, device=scene.device)  # tip capsules
        self.tissue = cc.surface.body
        self.start_size = check_float("smoke.start_size", self.start_size)
        self.end_size = check_float("smoke.end_size", self.end_size)
        self.start_color, self.end_color = tuple(float(c) for c in self.start_color), tuple(float(c) for c in self.end_color)
        self.up = wp.vec3(*(-scene.down).tolist())
        self._live("rate", "lifetime", "speed", "upward", "spawn_radius")
        E, n, dev = scene.envs, int(self.count), scene.device
        scene.check_memory({"smoke": E * n * 32})
        self.pos_age = wp.zeros(E * n, dtype=wp.vec4f, device=dev)
        self.vel_life = wp.zeros(E * n, dtype=wp.vec4f, device=dev)  # lifetime 0: a dead slot
        self.cursor = wp.zeros(E, dtype=wp.int32, device=dev)
        self.acc = wp.zeros(E, dtype=wp.float32, device=dev)
        self.serial = wp.zeros(E, dtype=wp.int32, device=dev)
        self.hot = wp.zeros(E, dtype=wp.int32, device=dev)  # a source exists this step
        self.centre = wp.zeros(E, dtype=wp.vec3, device=dev)
        self.alive_count = wp.zeros(E, dtype=wp.int32, device=dev)
        self.freeze("count", "max_spawn", "instrument")


    def step_end(self, scene, dt):
        E, dev = scene.envs, scene.device
        ic, cc, th, t = self.instruments, self.contacts, self.thermal, self.tissue
        wp.launch(spawn_smoke, dim=E,
                  inputs=[ic.energized, self.k, self.source_prim, cc.record_count, cc.rec_tet, cc.rec_verts, cc.rec_bary,
                          th.heat, th.coag_heat, scene.field("q", t), self.rate, self.lifetime, self.speed, self.upward,
                          self.spawn_radius, self.up, int(self.count), int(self.max_spawn), dt],
                  outputs=[self.hot, self.centre, self.acc, self.serial, self.cursor, self.pos_age, self.vel_life, self.alive_count],
                  device=dev)
        wp.launch(update_smoke, dim=(E, int(self.count)), inputs=[self.vel_life, int(self.count), dt],
                  outputs=[self.pos_age, self.alive_count], device=dev)

    def reset(self, scene, mask):
        zero4 = wp.vec4f(0.0, 0.0, 0.0, 0.0)
        for dst in (self.pos_age, self.vel_life):
            self.clear(mask, (dst, zero4))
        for dst in (self.cursor, self.serial, self.hot, self.alive_count):
            self.clear(mask, (dst, wp.int32(0)))
        self.clear(mask, (self.acc, wp.float32(0.0)))


# omnisurg tet/kernels/tet.py:594-596 (the smoke event inside deposit_tet_heat_from_contacts) as a scan: the first tip
# record whose tet is at coagulation heat is the source; its point is rebuilt from the record. Returns (x, y, z, found).
# The count is clamped to the record capacity: contacts beyond it are counted but not stored (instrument_contact.py).
@wp.func
def source_point(
    env: wp.int32,
    source_prim: wp.array(dtype=wp.int32),
    record_count: wp.array2d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_bary: wp.array3d(dtype=wp.vec3),
    heat: wp.array2d(dtype=wp.float32),
    coag_heat: wp.array(dtype=wp.float32),
    q: wp.array2d(dtype=wp.vec3),
) -> wp.vec4:
    for p in range(record_count.shape[1]):
        if source_prim[p] == 0:
            continue
        for r in range(wp.min(record_count[env, p], rec_tet.shape[2])):
            t = rec_tet[env, p, r]
            if t < 0 or heat[env, t] < coag_heat[env]:
                continue
            v = rec_verts[env, p, r]
            b = rec_bary[env, p, r]
            c = q[env, v[0]] * b[0] + q[env, v[1]] * b[1] + q[env, v[2]] * b[2]
            return wp.vec4(c[0], c[1], c[2], 1.0)
    return wp.vec4(0.0, 0.0, 0.0, 0.0)


# omnisurg rendering/particles.py:330-365 (_gpu_smoke_spawn_kernel) with the emitter's rate accumulator and ring, one
# thread per env, wp.rand_init in place of the hashed RNG; the camera-heading bias is dropped. The source scan runs in
# the same thread (``hot`` and ``centre`` are kept as products for tests and the renderer).
@wp.kernel
def spawn_smoke(
    energized: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    source_prim: wp.array(dtype=wp.int32),
    record_count: wp.array2d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_bary: wp.array3d(dtype=wp.vec3),
    heat: wp.array2d(dtype=wp.float32),
    coag_heat: wp.array(dtype=wp.float32),
    q: wp.array2d(dtype=wp.vec3),
    rate: wp.array(dtype=wp.float32),
    lifetime: wp.array2d(dtype=wp.float32),
    speed: wp.array2d(dtype=wp.float32),
    upward: wp.array2d(dtype=wp.float32),
    spawn_radius: wp.array(dtype=wp.float32),
    up: wp.vec3,
    count: wp.int32,
    max_spawn: wp.int32,
    dt: wp.float32,
    hot: wp.array(dtype=wp.int32),
    centre: wp.array(dtype=wp.vec3),
    acc: wp.array(dtype=wp.float32),
    serial: wp.array(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),
    pos_age: wp.array(dtype=wp.vec4f),
    vel_life: wp.array(dtype=wp.vec4f),
    alive: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    serial[env] = serial[env] + 1
    hot[env] = 0
    alive[env] = 0  # update_smoke counts into it next
    if energized[env, k] == 0:
        acc[env] = 0.0
        return
    s = source_point(env, source_prim, record_count, rec_tet, rec_verts, rec_bary, heat, coag_heat, q)
    if s[3] == 0.0:
        acc[env] = 0.0
        return
    c = wp.vec3(s[0], s[1], s[2])
    hot[env] = 1
    centre[env] = c
    t = take(acc[env], rate[env] * dt, max_spawn)
    n = int(t[0])
    acc[env] = t[1]
    base = cursor[env]
    cursor[env] = (base + n) % count  # kept below count so it never wraps int32
    state = wp.rand_init(serial[env] * 7919 + 13, env * count + base)
    for j in range(n):
        li = env * count + (base + j) % count
        p = c + wp.sample_unit_sphere(state) * spawn_radius[env]
        direction = wp.sample_unit_sphere_surface(state)
        sp = speed[env, 0] + (speed[env, 1] - speed[env, 0]) * wp.randf(state)
        upw = upward[env, 0] + (upward[env, 1] - upward[env, 0]) * wp.randf(state)
        life = lifetime[env, 0] + (lifetime[env, 1] - lifetime[env, 0]) * wp.randf(state)
        v = direction * sp + up * upw
        pos_age[li] = wp.vec4f(p[0], p[1], p[2], 0.0)
        vel_life[li] = wp.vec4f(v[0], v[1], v[2], wp.max(life, 1.0e-6))


# omnisurg rendering/particles.py:299-327 (_gpu_smoke_update_kernel) with the (env, local) prologue and an alive count;
# a particle is alive while its age is below its lifetime (a dead slot has lifetime 0).
@wp.kernel
def update_smoke(
    vel_life: wp.array(dtype=wp.vec4f),
    count: wp.int32,
    dt: wp.float32,
    pos_age: wp.array(dtype=wp.vec4f),
    alive: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    li = env * count + local
    pa = pos_age[li]
    vl = vel_life[li]
    if pa[3] >= vl[3]:
        return
    age = pa[3] + dt
    pos_age[li] = wp.vec4f(pa[0] + vl[0] * dt, pa[1] + vl[1] * dt, pa[2] + vl[2] * dt, age)
    if age < vl[3]:
        wp.atomic_add(alive, env, 1)
