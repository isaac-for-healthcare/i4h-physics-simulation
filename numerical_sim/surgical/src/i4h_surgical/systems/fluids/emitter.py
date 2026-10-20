# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fluid emitter: spawns pool particles at a world point or an instrument tip, on device.

Semantics from omnisurg's ``PositionBasedFluidEmitterRuntime`` (fluids/systems.py:2864-2964): a per-env
accumulator ``acc += rate * dt`` releases ``floor(acc)`` particles per step (capped by ``max_spawn``; the excess is
dropped, the carry stays below one particle),
each at the centre plus a uniform ball offset of ``spawn_radius`` with velocity ``axis * speed``; slots
come from the pool's ring ``cursor`` (oldest overwritten, no free check). The instrument anchor is
omnisurg's irrigation (``:2900-2929``): centre = the tool's ``root``, axis = along the shaft away from
the trocar (the live ``direction`` for a free tool), emission scaled by ``(grip - threshold) / (1 -
threshold)`` and suppressed while the tool is energised (suction wins). Everything omnisurg did on the
host (numpy round trips, Python spawn loops, host cursor and RNG) is one kernel per step with one
thread per env; the RNG is ``wp.rand_init`` seeded from a per-env step serial and the slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System, live
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


@dataclass(eq=False)
class Emitter(System):
    fluid: Fluid
    body: str  # fluid asset the particles come from
    position: tuple | None = None  # world anchor (live, per env) ...
    instrument: str | None = None  # ... or the tip of this instrument (needs ``instruments``)
    instruments: InstrumentControl | None = None
    rate: float = 256.0  # live, per env: particles per second
    speed: float = 0.2  # live, per env: initial speed along the axis
    spawn_radius: float = 0.01  # live, per env
    direction: tuple = (0.0, 1.0, 0.0)  # live, per env: world axis (a free tool's axis too)
    threshold: float = 0.35  # live, per env: grip level where irrigation starts
    enabled_env: float = 1.0  # live, per env: 0 silences an env
    max_spawn: int = 128  # structural: particles per env and step at most (4096/s at 60 Hz needs 69)

    refs = ("fluid", "instruments")
    structural = System.structural + ("body", "instrument", "max_spawn")

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        if not isinstance(self.fluid, Fluid):
            raise TypeError("emitter.fluid must name a fluid system")
        if self.body not in self.fluid.index:
            raise ValueError(f"emitter.body {self.body!r}: not a fluid pool; known: {sorted(self.fluid.index)}")
        if (self.position is None) == (self.instrument is None):
            raise ValueError("emitter: give exactly one of position or instrument")
        if self.instrument is not None:
            if not isinstance(self.instruments, InstrumentControl):
                raise TypeError("emitter.instruments must name the instrument_control system")
            if self.instrument not in self.instruments.names:
                raise ValueError(f"emitter.instrument {self.instrument!r}: known {self.instruments.names}")
            self.k = self.instruments.names.index(self.instrument)
        self.max_spawn = check_int("emitter.max_spawn", self.max_spawn, 1)
        self.f = self.fluid.index[self.body]
        self.pool = self.fluid.bodies[self.f]
        self.freeze("body", "instrument")
        self.position = live(scene, self.position if self.position is not None else (0.0, 0.0, 0.0))
        self.register_live("position")
        self._live("direction")
        self._live("rate", "speed", "spawn_radius", "threshold", "enabled_env")
        self.acc = wp.zeros(E, dtype=wp.float32, device=dev)
        self.serial = wp.zeros(E, dtype=wp.int32, device=dev)
        self.spawned = wp.zeros(E, dtype=wp.int32, device=dev)  # particles born this step


    def step_start(self, scene, dt):
        f, b, E = self.f, self.pool, scene.envs
        fl = self.fluid
        outputs = [self.acc, self.serial, self.spawned, fl.cursor, fl.q_b[f], fl.q_prev_b[f],
                   fl.qd_b[f], fl.flags_b[f]]
        if self.instrument is None:
            wp.launch(emit_point, dim=E,
                      inputs=[self.position, self.direction, self.rate, self.speed, self.spawn_radius, self.enabled_env,
                              f, b.count, int(self.max_spawn), dt],
                      outputs=outputs, device=scene.device)
        else:
            ic = self.instruments
            wp.launch(emit_tool, dim=E,
                      inputs=[ic.root, ic.trocar, ic.has_trocar, ic.actions, ic.energized, self.k, self.direction,
                              self.rate, self.speed, self.spawn_radius, self.threshold, self.enabled_env,
                              f, b.count, int(self.max_spawn), dt],
                      outputs=outputs, device=scene.device)

    def reset(self, scene, mask):
        self.clear(mask, (self.acc, wp.float32(0.0)))
        for dst in (self.serial, self.spawned):
            self.clear(mask, (dst, wp.int32(0)))


@wp.func
def release(
    env: wp.int32,
    count: wp.int32,
    centre: wp.vec3,
    velocity: wp.vec3,
    spawn_radius: wp.float32,
    base: wp.int32,
    n: wp.int32,
    seed: wp.int32,
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
):
    """Bring ``n`` particles of env's pool ``f`` to life in ring slots ``base ..`` (oldest overwritten) around ``centre``."""
    state = wp.rand_init(seed, env * count + base)
    for j in range(n):
        li = env * count + (base + j) % count
        p = centre + wp.sample_unit_sphere(state) * spawn_radius
        q[li] = p
        q_prev[li] = p
        qd[li] = velocity
        flags[li] = ACTIVE


@wp.func
def take(acc: wp.float32, amount: wp.float32, max_spawn: wp.int32) -> wp.vec2:
    """Accumulate ``amount`` particles; returns (particles to release now, remaining fraction). What ``max_spawn`` cuts
    off is dropped, not carried: the carry stays below one particle, so a rate above the cap never builds a backlog
    that keeps emitting after the source stops."""
    a = acc + amount
    n = wp.min(int(wp.floor(a)), max_spawn)
    return wp.vec2(wp.float32(n), wp.min(a - wp.float32(n), 1.0 - 1.0e-6))


@wp.func
def emit(
    env: wp.int32,
    f: wp.int32,
    count: wp.int32,
    centre: wp.vec3,
    velocity: wp.vec3,
    spawn_radius: wp.float32,
    amount: wp.float32,
    max_spawn: wp.int32,
    acc: wp.array(dtype=wp.float32),
    cursor: wp.array2d(dtype=wp.int32),
    spawned: wp.array(dtype=wp.int32),
    serial: wp.array(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
):
    """The emit kernels' shared tail: accumulate ``amount`` particles, release the whole ones from env's ring for
    pool ``f`` and advance it (one thread per env owns the ring; kept below ``count`` so it never wraps int32)."""
    t = take(acc[env], amount, max_spawn)
    n = int(t[0])
    acc[env] = t[1]
    base = cursor[env, f]
    cursor[env, f] = (base + n) % count
    release(env, count, centre, velocity, spawn_radius, base, n, serial[env] * 7919 + f, q, q_prev, qd, flags)
    spawned[env] = n
    serial[env] = serial[env] + 1


@wp.kernel
def emit_point(
    position: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    rate: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    spawn_radius: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.float32),
    f: wp.int32,
    count: wp.int32,
    max_spawn: wp.int32,
    dt: wp.float32,
    acc: wp.array(dtype=wp.float32),
    serial: wp.array(dtype=wp.int32),
    spawned: wp.array(dtype=wp.int32),
    cursor: wp.array2d(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    centre = wp.vec3(position[env, 0], position[env, 1], position[env, 2])
    axis = wp.vec3(direction[env, 0], direction[env, 1], direction[env, 2])
    scale = enabled[env]
    emit(env, f, count, centre, axis * (speed[env] * scale), spawn_radius[env], rate[env] * scale * dt, max_spawn,
         acc, cursor, spawned, serial, q, q_prev, qd, flags)


@wp.kernel
def emit_tool(
    root: wp.array2d(dtype=wp.vec3),
    trocar: wp.array(dtype=wp.vec3),
    has_trocar: wp.array(dtype=wp.int32),
    actions: wp.array3d(dtype=wp.float32),
    energized: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    direction: wp.array2d(dtype=wp.float32),
    rate: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    spawn_radius: wp.array(dtype=wp.float32),
    threshold: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.float32),
    f: wp.int32,
    count: wp.int32,
    max_spawn: wp.int32,
    dt: wp.float32,
    acc: wp.array(dtype=wp.float32),
    serial: wp.array(dtype=wp.int32),
    spawned: wp.array(dtype=wp.int32),
    cursor: wp.array2d(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
):
    """omnisurg fluids/systems.py:2900-2939: irrigation scales with the grip above ``threshold`` and stops
    while the tool sucks; the jet leaves the tool tip along its shaft."""
    env = wp.tid()
    centre = root[env, k]
    axis = wp.vec3(direction[env, 0], direction[env, 1], direction[env, 2])
    if has_trocar[k] != 0:
        shaft = centre - trocar[k]
        length = wp.length(shaft)
        if length > 1.0e-6:
            axis = shaft / length
    th = threshold[env]
    scale = wp.clamp((actions[env, k, 7] - th) / wp.max(1.0 - th, 1.0e-6), 0.0, 1.0) * enabled[env]
    if energized[env, k] != 0:
        scale = 0.0
    emit(env, f, count, centre, axis * (speed[env] * scale), spawn_radius[env], rate[env] * scale * dt, max_spawn,
         acc, cursor, spawned, serial, q, q_prev, qd, flags)
