# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Position-based fluid over every fluid body (omnisurg ``PositionBasedFluidSolver``, fluids/systems.py:999-2291).

One system runs all fluid bodies (assets of ``type: fluid``); their parameters are asset properties on
``Body.fluid`` like tet materials, so per-pool radii and iteration counts are baked at construction and
``relaxation`` / ``viscosity`` / ``cohesion`` are live ``(E, nf)``. Every stage is one launch over all
pools, ``dim=(E, sum of pool counts)`` (fluid_kernels.py ``pool_slot``), so the launch count does not grow
with the number of pools. One grouped ``wp.HashGrid`` over the pools' private block (``scene.pool``, pools one after another)
serves every pool and env: a particle's group is ``env * nf + f`` while alive and -1 while dead, so
neighbour queries never cross an env, a pool or a parked slot. One grid is also a requirement: Warp
uploads a grid's device descriptor from one static host struct per dtype with a pageable async memcpy
(native/hashgrid.cpp:324-356), so two grids built inside one captured graph would replay each other's
descriptors. As in omnisurg's ``substep`` mode the grid is queried once per substep into a slot-major
neighbour list of ``max_neighbors`` body-local ids (a grid traversal is a latency floor of ~50 us per
launch at a few hundred particles, so the λ, Δp and XSPH launches read the list instead); the
radix-sorted grid makes the list order, and so every sum, run-to-run stable. Per substep: ``predict_fluid``
(the ``particles`` predict with this system's own ``damping``, default 1.0 as omnisurg's fluid predict had
none, ``max_velocity`` and the grid group written alongside), the grid build and the neighbour list,
``iterations_f`` density solves per pool scheduled with ``due()`` (a bit mask selects the pools due on
an iteration), ``finalize_fluid`` writing the XSPH scratch and ``xsph`` writing ``scene.qd``.
Emitters take pool slots from the live ``cursor`` (E, nf); ``alive_count`` (E, nf) is recounted each step;
``overflow`` (E, nf) counts particles whose neighbours did not fit since reset. Every launch runs whether
or not a pool has live particles: skipping empty pools with ``wp.capture_if`` conditional graph nodes was
measured at no gain (the idle cost is the graph's node count, and a conditional is a node too). Not
ported: particle-particle rigid contacts, ``velocity_update_blend``, the surface-tension model, the
static-mesh and bounds projections (``static_contact`` and ``bounds`` bind to a fluid body unchanged),
the ``uniform`` grid backend, the per-particle neighbour diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.systems.base import System, due, live
from i4h_physics.systems.fluids.fluid_kernels import (
    apply,
    count_alive,
    deltas,
    finalize_fluid,
    lambdas,
    neighbors,
    predict_fluid,
    xsph,
)


# omnisurg/fluids/systems.py:701-739 (host estimate of the rest density on a rest_distance lattice); the FleX spiky
# kernel variant (use_spiky_kernels) is not carried.
def _poly6_weight(distance: float, radius: float) -> float:
    if distance >= radius:
        return 0.0
    value = radius * radius - distance * distance
    return 315.0 / (math.pi * 64.0 * radius**9) * value**3


def estimate_rest_density(radius: float, rest_distance: float) -> float:
    cells = max(1, int(math.ceil(radius / rest_distance)))
    density = 0.0
    for k in range(-cells, cells + 1):
        for j in range(-cells, cells + 1):
            for i in range(-cells, cells + 1):
                distance = math.sqrt(i * i + j * j + k * k) * rest_distance
                density += _poly6_weight(distance, radius)
    return max(density, 1.0e-12)


@dataclass(eq=False)
class Fluid(System):
    state_products = ("alive_count",)  # per-env products a host observes (System.state_products)
    damping: float = 1.0  # live, per env: velocity damping per substep of every fluid body
    max_velocity: float = 1000.0  # live, per env: bounds the predicted velocity and the substep displacement
    grid_dim: int = 0  # structural: hash table dimensions (a hash, not an extent); 0 = from the pool sizes

    structural = System.structural + ("grid_dim",)

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        self.bodies = [b for b in scene.pools if b.fluid is not None]
        if not self.bodies:
            raise ValueError("fluid: the scene has no asset of type fluid")
        self.index = {b.name: f for f, b in enumerate(self.bodies)}
        self.nf = nf = len(self.bodies)
        if nf > 31:
            raise ValueError("fluid: at most 31 pools (the iteration schedule is a bit mask)")
        recs = [b.fluid for b in self.bodies]
        self.radius = [r.interaction_radius for r in recs]
        self.rest_distance = [r.rest_distance for r in recs]
        self.iterations = [r.iterations for r in recs]
        self.max_neighbors = [r.max_neighbors for r in recs]
        self.density_epsilon = [r.density_epsilon for r in recs]
        self.rest_density = [estimate_rest_density(r.interaction_radius, r.rest_distance) for r in recs]
        self.cell = min(self.radius)
        # The pool block keeps each pool's envs one after another (body-major; it never becomes world-major), so the
        # span arithmetic below uses the offsets in that block directly.
        self.span_start = min(b.offset for b in self.bodies)
        self.span_end = max(b.offset + E * b.count for b in self.bodies)
        span = self.span_end - self.span_start
        counts = [b.count for b in self.bodies]
        self.total = sum(counts)  # the launch's second dimension: every pool's slots, one after another
        base = [b.offset - self.span_start for b in self.bodies]  # each pool's offset in the span
        nbr_base = np.cumsum([0] + [E * n * k for n, k in zip(counts, self.max_neighbors)])
        scene.check_memory({"span": span * 40, "neighbours": int(nbr_base[-1]) * 4, "grid": span * 24})
        def ints(values):
            return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=dev)
        def floats(values):
            return wp.array(np.asarray(values, np.float32), dtype=wp.float32, device=dev)
        self.const = dict(  # per-pool constants the span-wide kernels index by pool
            prefix=ints(np.cumsum([0] + counts)), counts=ints(counts), base=ints(base), nbr_base=ints(nbr_base[:-1]),
            max_neighbors=ints(self.max_neighbors), radius=floats(self.radius),
            rest_density=floats(self.rest_density), rest_distance=floats(self.rest_distance),
            density_epsilon=floats(self.density_epsilon),
        )
        sl, pool = slice(self.span_start, self.span_end), scene.pool  # the pools' private block
        self.q_span, self.qd_span, self.q_prev_span = pool.q[sl], pool.qd[sl], pool.q_prev[sl]
        self.inv_mass_span, self.flags_span, self.radius_span = pool.inv_mass[sl], pool.flags[sl], pool.radius[sl]
        self.q0_span = pool.q0[sl]
        self.group = wp.full(span, -1, dtype=wp.int32, device=dev)  # -1 stays on any non-fluid particle in the span
        self.lam_span = wp.zeros(span, dtype=wp.float32, device=dev)
        self.density_span = wp.zeros(span, dtype=wp.float32, device=dev)
        self.delta = wp.zeros(span, dtype=wp.vec3, device=dev)
        self.qd_scratch = wp.zeros(span, dtype=wp.vec3, device=dev)
        self.nbr = wp.zeros(int(nbr_base[-1]), dtype=wp.int32, device=dev)
        self.nbr_count = wp.zeros(span, dtype=wp.int32, device=dev)
        # Per-pool views for the systems and tests that work on one pool (emitter, bleeding, contact, suction).
        self.q_b = [pool.q[b.offset : b.offset + E * b.count] for b in self.bodies]
        self.qd_b = [pool.qd[b.offset : b.offset + E * b.count] for b in self.bodies]
        self.q_prev_b = [pool.q_prev[b.offset : b.offset + E * b.count] for b in self.bodies]
        self.flags_b = [pool.flags[b.offset : b.offset + E * b.count] for b in self.bodies]
        self.density = [self.density_span[o : o + E * n] for o, n in zip(base, counts)]
        self.cursor = wp.zeros((E, nf), dtype=wp.int32, device=dev)  # ring cursor per pool (emitters' live input)
        self.alive_count = wp.zeros((E, nf), dtype=wp.int32, device=dev)
        self.overflow = wp.zeros((E, nf), dtype=wp.int32, device=dev)  # neighbour-list overflows since reset
        self.relaxation = live(scene, [r.relaxation for r in recs])
        self.register_live("relaxation")
        self.viscosity = live(scene, [r.viscosity for r in recs])
        self.register_live("viscosity")
        self.cohesion = live(scene, [r.cohesion for r in recs])
        self.register_live("cohesion")
        self._live("damping")
        self._live("max_velocity")
        # Warp memsets 2 * dim^3 ints per build, so the table tracks the particle count, not 128^3 (16 MB per substep).
        self.grid_dim = check_int("fluid.grid_dim", self.grid_dim, 0)
        dim = self.grid_dim or min(128, max(32, 1 << math.ceil(math.log2(2.0 * span ** (1.0 / 3.0)))))
        self.grid = wp.HashGrid(dim, dim, dim, device=dev)
        self.grid.reserve(span, with_groups=True)  # allocation-free builds inside the capture
        self.freeze("grid_dim")


    def _launch(self, kernel, inputs, outputs):
        wp.launch(kernel, dim=(self.scene.envs, self.total), inputs=inputs, outputs=outputs, device=self.scene.device)

    def substep_start(self, scene, dt, s, substeps):
        c = self.const
        self._launch(predict_fluid,
                     [self.qd_span, self.inv_mass_span, self.flags_span, scene.gravity, self.damping, self.max_velocity,
                      c["prefix"], c["counts"], c["base"], self.nf, dt],
                     [self.q_span, self.q_prev_span, self.group])
        self.grid.build(self.q_span, self.cell, self.group)
        self._launch(neighbors,
                     [self.grid.id, self.q_span, self.flags_span, c["prefix"], c["counts"], c["base"], c["nbr_base"],
                      c["max_neighbors"], c["radius"], scene.envs, self.nf],
                     [self.nbr, self.nbr_count, self.overflow])

    def iteration(self, scene, dt, k, iterations):
        mask = sum(1 << f for f in range(self.nf) if due(k, self.iterations[f], iterations)[0] >= 0)
        if not mask:
            return
        c, E = self.const, scene.envs
        self._launch(lambdas,
                     [self.q_span, self.flags_span, self.nbr, self.nbr_count, c["prefix"], c["counts"], c["base"], c["nbr_base"],
                      c["radius"], c["rest_density"], c["density_epsilon"], E, self.nf, mask],
                     [self.density_span, self.lam_span])
        self._launch(deltas,
                     [self.q_span, self.flags_span, self.lam_span, self.nbr, self.nbr_count, self.cohesion, c["prefix"], c["counts"],
                      c["base"], c["nbr_base"], c["radius"], c["rest_density"], c["rest_distance"], E, self.nf, mask],
                     [self.delta])
        self._launch(apply, [self.flags_span, self.delta, self.relaxation, c["prefix"], c["counts"], c["base"], self.nf, mask],
                     [self.q_span])

    def substep_end(self, scene, dt):
        c = self.const
        self._launch(finalize_fluid,
                     [self.q_prev_span, self.inv_mass_span, self.flags_span, self.max_velocity, c["prefix"], c["counts"], c["base"],
                      self.nf, dt],
                     [self.q_span, self.qd_scratch])
        self._launch(xsph,
                     [self.q_span, self.qd_scratch, self.flags_span, self.nbr, self.nbr_count, self.viscosity, self.max_velocity,
                      c["prefix"], c["counts"], c["base"], c["nbr_base"], c["radius"], c["rest_density"], scene.envs, self.nf],
                     [self.qd_span])

    def step_end(self, scene, dt):
        c = self.const
        self.alive_count.zero_()
        self._launch(count_alive, [self.flags_span, c["prefix"], c["counts"], c["base"], self.nf], [self.alive_count])

    def reset(self, scene, mask):
        """Ring, counts and overflow; the solver scratch (neighbour list, λ, density) is rewritten before it is read."""
        for dst in (self.cursor, self.alive_count, self.overflow):
            self.clear(mask, (dst, wp.int32(0)))
