# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clip applier: a fired instrument clips the nearest node of a vessel centreline; the clip pinches
the node's cross-section and rides with the tissue.

omnisurg only flagged a node (tet/runtime/centreline.py:192-248, a host-side nearest search
consumed by the bleeding logic; no instrument, no constraint, no key). The bookkeeping is kept
(``clipped`` per node, fixed slots per env, nearest unclipped node within ``reach``) as one
device kernel on the applier's ``fired`` edge; the hold is new: every iteration the clamp
particles of a clipped node are pulled toward their live centroid until they sit at ``lumen``
from it. One thread per env walks its slots and writes the corrections in place (adjacent
nodes may share particles, so the slots are applied in order rather than averaged). Both
run for all envs in one launch each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.centreline import Centreline


@dataclass(eq=False)
class Clipping(System):
    state_products = ("clipped", "count")  # per-env products a host observes (System.state_products)
    centreline: Centreline
    instruments: InstrumentControl
    instrument: str  # the applier (kind clip_applier; chole: clip_applier); its fire channel places a clip
    line: str | tuple  # the centreline(s) to clip, a name or a list (chole: cystic_duct and cystic_artery)
    reach: float = 0.1  # live, per env: a node within this distance of the applier's root can be clipped
    lumen: float = 0.0  # live, per env: distance of the clamp ring from the node once clipped
    stiffness: float = 1.0  # live, per env: fraction of the pinch applied per iteration
    slots: int = 4  # structural: clips per env

    refs = ("centreline", "instruments")
    structural = System.structural + ("slots", "instrument", "line")  # the applier and line indices are baked into the launches

    def __post_init__(self):
        scene = self.scene
        cl, ic = self.centreline, self.instruments
        if not isinstance(cl, Centreline) or not isinstance(ic, InstrumentControl):
            raise TypeError("clipping.centreline must name a centreline system and clipping.instruments an instrument_control system")
        if self.instrument not in ic.names:
            raise ValueError(f"clipping: unknown instrument {self.instrument!r}; known: {ic.names}")
        if scene.instruments[self.instrument].kind != "clip_applier":
            raise ValueError(f"clipping.instrument {self.instrument!r} must be a clip_applier (kind: clip_applier); a grasper's fire does nothing")
        lines = (self.line,) if isinstance(self.line, str) else tuple(self.line)
        unknown = [name for name in lines if name not in cl.names]
        if unknown or not lines:
            raise ValueError(f"clipping: unknown centreline(s) {unknown}; known: {cl.names}")
        self.slots = check_int("clipping.slots", self.slots, 1)
        self.k = ic.names.index(self.instrument)
        self.line = lines  # always a tuple (structural)
        self.body = cl.body
        E, dev = scene.envs, scene.device
        K, G = cl.positions.shape[1], int(self.slots)
        indices = [cl.names.index(name) for name in lines]
        self.eligible = wp.array(np.isin(cl.line, indices).astype(np.int32), dtype=wp.int32, device=dev)  # (K,)
        self._live("reach", "lumen", "stiffness")
        self.node = wp.full((E, G), -1, dtype=wp.int32, device=dev)  # clipped node per slot
        self.active = wp.zeros((E, G), dtype=wp.int32, device=dev)
        self.count = wp.zeros(E, dtype=wp.int32, device=dev)  # clips placed; a clip torn off (its ring cut away) keeps its slot
        self.clipped = wp.zeros((E, K), dtype=wp.int32, device=dev)  # per node; stays latched after a tear (bleeding sees the node as cut)
        self.freeze("slots", "instrument", "line")


    def step_start(self, scene, dt):
        cl, ic = self.centreline, self.instruments
        if cl.positions.shape[1] == 0:
            return
        wp.launch(attach_clip, dim=scene.envs,
                  inputs=[ic.fired, self.k, ic.root, cl.positions, cl.active_count, self.eligible, self.reach, self.node.shape[1]],
                  outputs=[self.node, self.active, self.count, self.clipped], device=scene.device)

    def iteration(self, scene, dt, k, iterations):
        cl, b = self.centreline, self.body
        wp.launch(hold_clips, dim=scene.envs,
                  inputs=[scene.field("inv_mass", b), scene.field("flags", b), cl.variant, cl.clamp_start, cl.clamp_count, cl.clamp,
                          self.node, self.lumen, self.stiffness, self.node.shape[1]],
                  outputs=[self.active, scene.field("q", b)], device=scene.device)

    def reset(self, scene, mask):
        E = scene.envs
        for value, dst in ((wp.int32(-1), self.node), (wp.int32(0), self.active), (wp.int32(0), self.count.reshape((E, 1))),
                           (wp.int32(0), self.clipped)):
            if dst.shape[1]:
                self.clear(mask, (dst, value))


# centreline.py:227-247 as a kernel: the nearest unclipped live node of the listed lines within reach takes a slot.
@wp.kernel
def attach_clip(
    fired: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    root: wp.array2d(dtype=wp.vec3),
    positions: wp.array2d(dtype=wp.vec3),
    active_count: wp.array2d(dtype=wp.int32),
    eligible: wp.array(dtype=wp.int32),
    reach: wp.array(dtype=wp.float32),
    slots: wp.int32,
    node: wp.array2d(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    clipped: wp.array2d(dtype=wp.int32),
):
    env = wp.tid()
    if fired[env, k] == 0 or count[env] >= slots:
        return
    tip = root[env, k]
    best = wp.int32(-1)
    best_distance = reach[env]
    for n in range(positions.shape[1]):
        if eligible[n] == 0 or clipped[env, n] != 0 or active_count[env, n] == 0:
            continue
        distance = wp.length(positions[env, n] - tip)
        if distance <= best_distance:
            best_distance = distance
            best = n
    if best < 0:
        return
    g = count[env]
    node[env, g] = best
    active[env, g] = 1
    count[env] = g + 1
    clipped[env, best] = 1


@wp.kernel
def hold_clips(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    variant: wp.array(dtype=wp.int32),
    clamp_start: wp.array2d(dtype=wp.int32),
    clamp_count: wp.array2d(dtype=wp.int32),
    clamp: wp.array2d(dtype=wp.int32),
    node: wp.array2d(dtype=wp.int32),
    lumen: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    slots: wp.int32,
    active: wp.array2d(dtype=wp.int32),
    q: wp.array2d(dtype=wp.vec3),
):
    """Pinch: pull each live clamp particle of every clipped node toward the ring's live centroid until it sits at
    ``lumen``; a ring reduced to fewer than two live particles (cut away) releases the clip. One thread per env."""
    env = wp.tid()
    v = variant[env]
    target = wp.max(lumen[env], 0.0)
    k = wp.clamp(stiffness[env], 0.0, 1.0)
    for g in range(slots):
        if active[env, g] == 0:
            continue
        n = node[env, g]
        s = clamp_start[v, n]
        m = clamp_count[v, n]
        centroid = wp.vec3(0.0, 0.0, 0.0)
        live = wp.int32(0)
        for j in range(m):
            p = clamp[v, s + j]
            if p < 0:
                continue
            i = p
            if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
                continue
            centroid = centroid + q[env, i]
            live = live + 1
        if live < 2:
            active[env, g] = 0
            continue
        centroid = centroid / wp.float32(live)
        for j in range(m):
            p = clamp[v, s + j]
            if p < 0:
                continue
            i = p
            if (flags[env, i] & ACTIVE) == 0 or inv_mass[env, i] <= 0.0:
                continue
            offset = q[env, i] - centroid
            distance = wp.length(offset)
            if distance <= target or distance <= 1.0e-8:
                continue
            q[env, i] = q[env, i] + offset * (-(distance - target) / distance * k)
