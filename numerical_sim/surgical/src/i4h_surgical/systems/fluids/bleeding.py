# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bleeding from a cut vessel centreline: blood spawns at the boundaries of the cut run until clips seal it.

omnisurg's cholecystectomy procedure (procedures/cholecystectomy/system.py:144-237) read the artery's
centreline cut flags and clip states back to the host each frame: a node was *cut* when an active contact
sphere of the cutting tool reached it while it fired (tet/runtime/centreline.py:296-342, latched), the bleed
sites were the nodes where the cut flag changes along the line (``cut[i] != cut[i-1]``), and the vessel was
*sealed* once a clipped node existed strictly before the first cut and strictly after the last, at which
point nothing bleeds. Each site fed an external emitter at ``rate`` particles per second along a fixed world
``direction``. Here that is three launches per step on device: ``mark_cuts`` (E, K) at ``step_end`` latches
the nodes under the cutting tool's spheres while it cuts (with a ``cutting`` ref: that system's cut chains and
its ``cut_on`` product, so a node is marked in the same step as the tets the scissors removed; without one:
the named instrument's fixed chains while it fires) and the nodes whose clamp ring died (fewer than two live
clamp particles, the predicate ``clipping.hold_clips`` releases on: a ring torn by stretch or burnt away is
a cut too — ours, the source never linked deletion to the cut flags); ``find_leaks`` (E,) at ``step_start``
walks the line for the cut run, the seal and the sources, and ``bleed`` (E, K) runs the emitter's
accumulator and ring release per source node. ``sources`` (E,) and ``sealed`` (E,) are per-env products;
``clipping`` is optional (no clips: never sealed). List it after ``centreline`` and ``cutting``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.systems.base import System
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_surgical.loaders.instrument_usd import CHAIN_FIXED
from i4h_surgical.systems.fluids.emitter import release, take
from i4h_surgical.systems.instruments.clipping import Clipping
from i4h_surgical.systems.instruments.cutting import Cutting
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.centreline import Centreline


@dataclass(eq=False)
class Bleeding(System):
    state_products = ("sources", "sealed")  # per-env products a host observes (System.state_products)
    fluid: Fluid
    centreline: Centreline
    instruments: InstrumentControl
    body: str  # fluid pool the blood comes from
    line: str  # the vessel (chole: cystic_artery)
    instrument: str | None = None  # structural: the firing tool whose fixed-chain spheres sever a node (without a cutting ref)
    clipping: Clipping | None = None
    cutting: Cutting | None = None  # ref: the cutting system whose cut chains and cut_on mark the nodes (then instrument is its tool)
    margin: float = 0.004  # live, per env: added to a sphere's radius for the cut test (contact holds tissue 0.002 off the tool)
    rate: float = 256.0  # live, per env: particles per second per source
    speed: float = 0.2  # live, per env
    direction: tuple = (0.0, 1.0, 0.0)  # live, per env: world spawn direction
    spawn_radius: float = 0.01  # live, per env
    max_spawn: int = 128  # structural: particles per source and step at most

    refs = ("fluid", "centreline", "instruments", "clipping", "cutting")
    structural = System.structural + ("body", "instrument", "line", "max_spawn")

    def __post_init__(self):
        scene = self.scene
        self.max_spawn = check_int("bleeding.max_spawn", self.max_spawn)
        E, dev = scene.envs, scene.device
        fl, cl, ic = self.fluid, self.centreline, self.instruments
        if not isinstance(fl, Fluid) or not isinstance(cl, Centreline) or not isinstance(ic, InstrumentControl):
            raise TypeError("bleeding.fluid, .centreline and .instruments must name a fluid, a centreline and an instrument_control system")
        if self.clipping is not None and not isinstance(self.clipping, Clipping):
            raise TypeError("bleeding.clipping must name a clipping system")
        if self.cutting is not None:
            if not isinstance(self.cutting, Cutting) or self.cutting.instruments is not ic:
                raise TypeError("bleeding.cutting must name a cutting system on the same instrument_control")
            if self.instrument is not None and self.instrument != self.cutting.instrument:
                raise ValueError(f"bleeding.instrument {self.instrument!r} differs from cutting.instrument {self.cutting.instrument!r}")
            self.instrument = self.cutting.instrument
        elif self.instrument is None:
            raise ValueError("bleeding needs a cutting ref or an instrument")
        if self.body not in fl.index:
            raise ValueError(f"bleeding.body {self.body!r}: not a fluid pool; known: {sorted(fl.index)}")
        if self.line not in cl.names:
            raise ValueError(f"bleeding.line {self.line!r}: known {cl.names}")
        if self.instrument not in ic.names:
            raise ValueError(f"bleeding.instrument {self.instrument!r}: known {ic.names}")
        self.f = fl.index[self.body]
        self.pool = fl.bodies[self.f]
        self.k = ic.names.index(self.instrument)
        nodes = np.flatnonzero(cl.line == cl.names.index(self.line))  # a line's nodes are contiguous, in order
        self.node_lo, self.node_hi = int(nodes.min()), int(nodes.max()) + 1
        if self.cutting is not None:  # the cutting system's spheres (the scissors' cut chains) and its per-step cut signal
            self.sphere_start, self.sphere_count, self.gate = self.cutting.sphere_start, self.cutting.sphere_count, self.cutting.cut_on
        else:
            self.sphere_start = int(ic.host_sphere_start[self.k])
            own = ic.host_sphere_chain[self.sphere_start : self.sphere_start + int(ic.host_sphere_counts[self.k])]
            self.sphere_count = int((own == CHAIN_FIXED).sum())  # fixed chains come first (instrument_control.py:74)
            self.gate = ic.fire_on
        self.freeze("body", "line", "instrument")
        if self.sphere_count == 0:
            raise ValueError(f"bleeding.instrument {self.instrument!r} has no cutting spheres, so it could never cut a node")
        K = cl.positions.shape[1]
        self._live("direction")
        self._live("margin", "rate", "speed", "spawn_radius")
        self.cut = wp.zeros((E, K), dtype=wp.int32, device=dev)
        self.source = wp.zeros((E, K), dtype=wp.int32, device=dev)
        self.acc = wp.zeros((E, K), dtype=wp.float32, device=dev)
        self.sealed = wp.zeros(E, dtype=wp.int32, device=dev)
        self.sources = wp.zeros(E, dtype=wp.int32, device=dev)
        self.serial = wp.zeros(E, dtype=wp.int32, device=dev)
        self.clipped = self.clipping.clipped if self.clipping is not None else wp.zeros((E, K), dtype=wp.int32, device=dev)


    def step_end(self, scene, dt):
        """Mark the cuts after the step's deletions, the centreline update and the cut signal of the same step."""
        E, dev = scene.envs, scene.device
        cl, ic = self.centreline, self.instruments
        K = cl.positions.shape[1]
        wp.launch(mark_cuts, dim=(E, K),
                  inputs=[self.gate, self.k, cl.positions, cl.active_count, cl.clamp_count, cl.variant, ic.sphere_q, ic.sphere_radius,
                          ic.sphere_active, self.margin, self.sphere_start, self.sphere_count, self.node_lo, self.node_hi],
                  outputs=[self.cut], device=dev)

    def step_start(self, scene, dt):
        E, dev = scene.envs, scene.device
        cl, fl, b, f = self.centreline, self.fluid, self.pool, self.f
        K = cl.positions.shape[1]
        wp.launch(find_leaks, dim=E, inputs=[self.cut, self.clipped, self.node_lo, self.node_hi, f, b.count],
                  outputs=[self.source, self.sealed, self.sources, self.serial, fl.cursor], device=dev)
        wp.launch(bleed, dim=(E, K),
                  inputs=[self.source, cl.positions, self.rate, self.speed, self.spawn_radius, self.direction, self.serial,
                          f, b.count, int(self.max_spawn), dt],
                  outputs=[self.acc, fl.cursor, fl.q_b[f], fl.q_prev_b[f], fl.qd_b[f],
                           fl.flags_b[f]], device=dev)

    def reset(self, scene, mask):
        for dst in (self.cut, self.source):
            self.clear(mask, (dst, wp.int32(0)))
        self.clear(mask, (self.acc, wp.float32(0.0)))
        for dst in (self.sealed, self.sources, self.serial):
            self.clear(mask, (dst, wp.int32(0)))


# omnisurg tet/runtime/centreline.py:329-338 (cut_centrelines_intersecting_contact_spheres) as a kernel: a node under
# an active cutting sphere of the tool while it cuts (``gate``: cutting.cut_on or fire_on) is cut for good. Edits: a node
# whose clamp ring has fewer than two live particles (clipping.hold_clips' release predicate; a ring with fewer than two
# authored particles is a smaller variant's padding and never counts) is cut as well.
@wp.kernel
def mark_cuts(
    gate: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    positions: wp.array2d(dtype=wp.vec3),
    active_count: wp.array2d(dtype=wp.int32),
    clamp_count: wp.array2d(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_active: wp.array2d(dtype=wp.int32),
    margin: wp.array(dtype=wp.float32),
    sphere_start: wp.int32,
    sphere_count: wp.int32,
    node_lo: wp.int32,
    node_hi: wp.int32,
    cut: wp.array2d(dtype=wp.int32),
):
    env, n = wp.tid()
    if n < node_lo or n >= node_hi or cut[env, n] != 0:
        return
    if active_count[env, n] < 2 and clamp_count[variant[env], n] >= 2:  # the ring died: torn, burnt or cut away
        cut[env, n] = 1
        return
    if gate[env, k] == 0:
        return
    p = positions[env, n]
    for s in range(sphere_start, sphere_start + sphere_count):
        if sphere_active[env, s] != 0 and wp.length(sphere_q[env, s] - p) <= sphere_radius[s] + margin[env] + 1.0e-6:
            cut[env, n] = 1
            return


# omnisurg procedures/cholecystectomy/system.py:29-68 (_evaluate_leak) and 206-237 (_artery_leak_sources), one thread
# per env: the cut run's bounds, the seal test and the boundary sources.
@wp.kernel
def find_leaks(
    cut: wp.array2d(dtype=wp.int32),
    clipped: wp.array2d(dtype=wp.int32),
    node_lo: wp.int32,
    node_hi: wp.int32,
    f: wp.int32,
    count: wp.int32,
    source: wp.array2d(dtype=wp.int32),
    sealed: wp.array(dtype=wp.int32),
    sources: wp.array(dtype=wp.int32),
    serial: wp.array(dtype=wp.int32),
    cursor: wp.array2d(dtype=wp.int32),
):
    env = wp.tid()
    cursor[env, f] = cursor[env, f] % count  # the sources add to the ring by atomics; keep it from wrapping int32
    first = wp.int32(-1)
    last = wp.int32(-1)
    for n in range(node_lo, node_hi):
        if cut[env, n] != 0:
            if first < 0:
                first = n
            last = n
    before = wp.int32(0)
    after = wp.int32(0)
    if first >= 0:
        for n in range(node_lo, first):
            if clipped[env, n] != 0:
                before = 1
        for n in range(last + 1, node_hi):
            if clipped[env, n] != 0:
                after = 1
    done = wp.int32(0)
    if first >= 0 and before != 0 and after != 0:
        done = 1
    sealed[env] = done
    total = wp.int32(0)
    for n in range(node_lo, node_hi):
        s = wp.int32(0)
        if done == 0 and n > node_lo and cut[env, n] != cut[env, n - 1]:
            s = 1
        source[env, n] = s
        total += s
    sources[env] = total
    serial[env] = serial[env] + 1


@wp.kernel
def bleed(
    source: wp.array2d(dtype=wp.int32),
    positions: wp.array2d(dtype=wp.vec3),
    rate: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    spawn_radius: wp.array(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    serial: wp.array(dtype=wp.int32),
    f: wp.int32,
    count: wp.int32,
    max_spawn: wp.int32,
    dt: wp.float32,
    acc: wp.array2d(dtype=wp.float32),
    cursor: wp.array2d(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
):
    env, n = wp.tid()
    if source[env, n] == 0:
        acc[env, n] = 0.0  # omnisurg pruned the accumulator of a source that went away
        return
    t = take(acc[env, n], rate[env] * dt, max_spawn)
    acc[env, n] = t[1]
    n_new = int(t[0])
    if n_new <= 0:
        return
    axis = wp.vec3(direction[env, 0], direction[env, 1], direction[env, 2])
    base = wp.atomic_add(cursor, env, f, n_new)  # several sources per env share the ring
    release(env, count, positions[env, n], axis * speed[env], spawn_radius[env], base, n_new, serial[env] * 7919 + n,
            q, q_prev, qd, flags)
