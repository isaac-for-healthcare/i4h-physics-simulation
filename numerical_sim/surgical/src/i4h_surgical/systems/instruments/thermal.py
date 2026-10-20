# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cautery: heat from an energised instrument's contacts, diffused through the tets, burnt into
deletion and coagulation paint.

Ported from omnisurg/tet/kernels/tet.py: deposit from contact records 536-596 and from the
hook's anchor 599-652, Jacobi diffusion over face neighbours 736-770, burn integration and
deletion 773-819, the per-particle burn maximum 964-985 and the coagulation paint 1014-1031;
omnisurg/tet/thermal.py:201-219 ran them in this order once per substep. Edits: (env, tet)
prologue with per-env heat and burn ``(E, T)``; the source is the instrument's fixed-chain
capsules and the trigger its ``energized`` level (omnisurg: a sphere range of the hook, and
grip >= 0.5 or a keyboard key); records carry the owning tet (``rec_tet``) so no triangle ->
tet table is needed; deletion goes through ``surface.deactivate_tet``; the particle burn
maximum is an ``atomic_max`` into ``coag`` (omnisurg walked a particle -> tet list) and is
painted into the green channel of ``surface.appearance.visual`` once per step (it is monotone, so the
frame the renderer reads is the same as painting every substep). Smoke is its own system (``smoke``, which reads ``heat``); dropped: the
direct-contact-deletion mode, particle smoothing (off in the shipped case).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.surface import Surface, deactivate_tet
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.hook import Hook
from i4h_surgical.systems.instruments.instrument_control import PRIM_KINDS, InstrumentControl


@dataclass(eq=False)
class Thermal(System):
    surface: Surface
    contacts: InstrumentContact
    instruments: InstrumentControl
    instrument: str  # the energised instrument (chole: hook)
    hook: Hook | None = None  # while its anchor is engaged all heat goes into the anchor's tet (tet.py:599-652)
    power: float = 100.0  # live, per env: heat per second per contact record (instruments.yaml diathermy_power_per_second)
    diffusion: float = 0.1  # live, per env: Jacobi rate between face neighbours
    cooling: float = 0.0  # live, per env: fraction of heat lost per second
    burn_rate: float = 1.0  # live, per env: burn per second per unit of heat above coag_heat
    coag_heat: float = 1.0  # live, per env: heat above which a tet burns (coag_heat_threshold)
    delete_heat: float = 2.0  # live, per env: heat needed for removal (delete_heat_threshold)
    delete_burn: float = 1.0  # live, per env: burn needed for removal; also full coagulation (delete_burn_threshold)
    diffusion_substeps: int = 1  # structural: Jacobi sweeps per substep

    refs = ("surface", "contacts", "instruments", "hook")
    structural = System.structural + ("diffusion_substeps", "instrument")  # the instrument index is baked into the launches

    def __post_init__(self):
        scene = self.scene
        sf, cc, ic = self.surface, self.contacts, self.instruments
        if not isinstance(sf, Surface) or not isinstance(cc, InstrumentContact) or not isinstance(ic, InstrumentControl):
            raise TypeError("thermal.surface, thermal.contacts and thermal.instruments must name a surface, an instrument_contact "
                            "and an instrument_control system")
        if self.hook is not None and not isinstance(self.hook, Hook):
            raise TypeError("thermal.hook must name a hook system")
        if cc.surface is not sf:
            raise ValueError("thermal.contacts must record contacts on thermal.surface")
        if self.instrument not in ic.names:
            raise ValueError(f"thermal: unknown instrument {self.instrument!r}; known: {ic.names}")
        self.diffusion_substeps = check_int("thermal.diffusion_substeps", self.diffusion_substeps, 1)
        self.k = k = ic.names.index(self.instrument)
        kinds, owners = ic.host_prim_kind, ic.host_prim_instrument
        # The fixed-chain capsules (jaw or hook piece; the probe's sphere) deposit; wrist and shaft chains do not.
        chain = (kinds == PRIM_KINDS["wrist"]) | (kinds == PRIM_KINDS["shaft"])
        self.source_prim = wp.array(np.int32((owners == k) & ~chain), dtype=wp.int32, device=scene.device)
        self.freeze("instrument")
        self.body = b = sf.body
        E, dev = scene.envs, scene.device
        T, n = sf.tets.shape[1], E * b.count
        scene.check_memory({"heat": 3 * E * T * 4, "coag": n * 4})
        self._live("power", "diffusion", "cooling", "burn_rate", "coag_heat", "delete_heat", "delete_burn")
        self.heat = wp.zeros((E, T), dtype=wp.float32, device=dev)
        self.heat_next = wp.zeros((E, T), dtype=wp.float32, device=dev)
        self.burn = wp.zeros((E, T), dtype=wp.float32, device=dev)
        self.coag = wp.zeros(n, dtype=wp.float32, device=dev)  # body-local per particle: max burn / delete_burn seen
        self._no_hook = wp.zeros(E, dtype=wp.int32, device=dev)


    def substep_end(self, scene, dt):
        sf, cc, ic, b = self.surface, self.contacts, self.instruments, self.body
        E, dev = scene.envs, scene.device
        T = sf.tets.shape[1]
        if T == 0:
            return
        hook_active = self.hook.entry_active if self.hook is not None else self._no_hook
        P, R = cc.rec_tet.shape[1], cc.rec_tet.shape[2]
        if P:
            wp.launch(deposit_heat, dim=(E, P, R),
                      inputs=[ic.energized, self.k, self.source_prim, hook_active, cc.record_count, cc.rec_tet, sf.active,
                              sf.tet_part, sf.heat_enabled, self.power, dt],
                      outputs=[self.heat], device=dev)
        if self.hook is not None:
            wp.launch(deposit_hook_heat, dim=E,
                      inputs=[ic.energized, self.k, self.hook.entry_active, self.hook.entry_tet, sf.active, sf.tet_part,
                              sf.heat_enabled, self.power, dt],
                      outputs=[self.heat], device=dev)
        src, dst = self.heat, self.heat_next
        sweeps = int(self.diffusion_substeps)
        for _ in range(sweeps):
            wp.launch(diffuse_heat, dim=(E, T),
                      inputs=[src, sf.variant, sf.active, sf.tet_part, sf.heat_enabled, sf.face_neighbor, self.diffusion,
                              self.cooling, dt / sweeps],
                      outputs=[dst], device=dev)
            src, dst = dst, src
        if src is self.heat:  # an even sweep count ended in ``heat``; the burn reads ``heat_next`` and writes ``heat``
            wp.copy(self.heat_next, self.heat)
        wp.launch(burn_and_delete, dim=(E, T),
                  inputs=[self.heat_next, sf.variant, sf.tets, sf.tet_part, sf.heat_enabled, sf.deletable, sf.tet_edges, self.coag_heat,
                          self.delete_heat, self.delete_burn, self.burn_rate, dt, b.count],
                  outputs=[self.heat, self.burn, self.coag, sf.active, sf.edge_active, sf.support, sf.edge_support, scene.field("flags", b),
                           sf.deleted, sf.dirty],
                  device=dev)

    def step_end(self, scene, dt):
        b = self.body
        wp.launch(paint_coag, dim=(scene.envs, b.count), inputs=[self.coag, b.count], outputs=[self.surface.appearance.visual], device=scene.device)

    def reset(self, scene, mask):
        E, count = scene.envs, self.body.count
        for dst in (self.heat, self.burn, self.coag.reshape((E, count))):  # heat_next is rewritten before it is read
            if dst.shape[1]:
                self.clear(mask, (dst, wp.float32(0.0)))


# tet.py:536-596 (deposit_tet_heat_from_surface_contacts). Edits: (env, capsule, record) prologue; the
# owning tet comes from the record; energised and source tests per instrument; no smoke.
@wp.kernel
def deposit_heat(
    energized: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    source_prim: wp.array(dtype=wp.int32),
    hook_active: wp.array(dtype=wp.int32),
    record_count: wp.array2d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    tet_part: wp.array(dtype=wp.int32),
    heat_enabled: wp.array(dtype=wp.int32),
    power: wp.array(dtype=wp.float32),
    dt: wp.float32,
    heat: wp.array2d(dtype=wp.float32),
):
    env, p, r = wp.tid()
    if energized[env, k] == 0 or source_prim[p] == 0 or hook_active[env] != 0:
        return
    if r >= record_count[env, p]:
        return
    amount = power[env] * dt
    if amount <= 0.0:
        return
    t = rec_tet[env, p, r]
    if t < 0 or active[env, t] == 0 or heat_enabled[tet_part[t]] == 0:
        return
    wp.atomic_add(heat, env, t, amount)


# tet.py:599-652 (deposit_tet_heat_from_hook_engagement): the engaged hook heats the anchor's tet only.
@wp.kernel
def deposit_hook_heat(
    energized: wp.array2d(dtype=wp.int32),
    k: wp.int32,
    hook_active: wp.array(dtype=wp.int32),
    hook_tet: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    tet_part: wp.array(dtype=wp.int32),
    heat_enabled: wp.array(dtype=wp.int32),
    power: wp.array(dtype=wp.float32),
    dt: wp.float32,
    heat: wp.array2d(dtype=wp.float32),
):
    env = wp.tid()
    if energized[env, k] == 0 or hook_active[env] == 0:
        return
    amount = power[env] * dt
    t = hook_tet[env]
    if amount <= 0.0 or t < 0 or active[env, t] == 0 or heat_enabled[tet_part[t]] == 0:
        return
    heat[env, t] = heat[env, t] + amount


# tet.py:736-770 (diffuse_tet_heat_jacobi). Edits: (env, tet) prologue, per-variant neighbours, live rates.
@wp.kernel
def diffuse_heat(
    heat_in: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    tet_part: wp.array(dtype=wp.int32),
    heat_enabled: wp.array(dtype=wp.int32),
    face_neighbor: wp.array3d(dtype=wp.int32),
    diffusion: wp.array(dtype=wp.float32),
    cooling: wp.array(dtype=wp.float32),
    dt: wp.float32,
    heat_out: wp.array2d(dtype=wp.float32),
):
    env, t = wp.tid()
    if active[env, t] == 0 or heat_enabled[tet_part[t]] == 0:
        heat_out[env, t] = 0.0
        return
    v = variant[env]
    center = heat_in[env, t]
    delta = wp.float32(0.0)
    for face in range(4):
        n = face_neighbor[v, t, face]
        if n >= 0 and active[env, n] != 0 and heat_enabled[tet_part[n]] != 0:
            delta = delta + heat_in[env, n] - center
    next_heat = center + diffusion[env] * dt * delta
    if cooling[env] > 0.0:
        next_heat = next_heat * wp.max(0.0, 1.0 - cooling[env] * dt)
    heat_out[env, t] = wp.max(next_heat, 0.0)


# tet.py:773-819 (accumulate_tet_burn_and_delete) with 964-985 and 1014-1031 folded in: the tet's burn,
# normalised by delete_burn and clamped, is pushed onto its four particles with atomic_max (monotone
# coagulation, as omnisurg's max over incident tets).
@wp.kernel
def burn_and_delete(
    heat_in: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    tets: wp.array2d(dtype=wp.vec4i),
    tet_part: wp.array(dtype=wp.int32),
    heat_enabled: wp.array(dtype=wp.int32),
    deletable: wp.array(dtype=wp.int32),
    tet_edges: wp.array3d(dtype=wp.int32),
    coag_heat: wp.array(dtype=wp.float32),
    delete_heat: wp.array(dtype=wp.float32),
    delete_burn: wp.array(dtype=wp.float32),
    burn_rate: wp.array(dtype=wp.float32),
    dt: wp.float32,
    count: wp.int32,
    heat: wp.array2d(dtype=wp.float32),
    burn: wp.array2d(dtype=wp.float32),
    coag: wp.array(dtype=wp.float32),
    active: wp.array2d(dtype=wp.int32),
    edge_active: wp.array2d(dtype=wp.int32),
    support: wp.array(dtype=wp.int32),
    edge_support: wp.array2d(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    deleted: wp.array(dtype=wp.int32),
    dirty: wp.array(dtype=wp.int32),
):
    env, t = wp.tid()
    if active[env, t] == 0:
        heat[env, t] = 0.0
        return
    part = tet_part[t]
    if heat_enabled[part] == 0:
        heat[env, t] = 0.0
        burn[env, t] = 0.0
        return
    h = heat_in[env, t]  # the last diffusion sweep's output; every thread writes ``heat``, so nothing is copied back
    heat[env, t] = h
    b = burn[env, t]
    if h > coag_heat[env] and burn_rate[env] > 0.0 and dt > 0.0:
        b = b + burn_rate[env] * (h - coag_heat[env]) * dt
        burn[env, t] = b
    if b > 0.0:
        c = wp.clamp(b / wp.max(delete_burn[env], 1.0e-6), 0.0, 1.0)
        v = variant[env]
        ids = tets[v, t]
        for k in range(4):
            wp.atomic_max(coag, env * count + ids[k], c)
    if deletable[part] != 0 and h >= delete_heat[env] and b >= delete_burn[env]:
        deactivate_tet(env, t, variant[env], count, tets, tet_edges, active, edge_active, support, edge_support, flags,
                       deleted, dirty)
        heat[env, t] = 0.0


@wp.kernel
def paint_coag(coag: wp.array(dtype=wp.float32), count: wp.int32, visual: wp.array(dtype=wp.vec4f)):
    """tet.py:1014-1031: coagulation into the green channel (r damage and b blood belong to stretch breaking)."""
    env, local = wp.tid()
    flat_index = env * count + local
    state = visual[flat_index]
    if coag[flat_index] <= state[1]:  # coagulation only grows (atomic_max): nothing to paint for an unburnt particle
        return
    visual[flat_index] = wp.vec4f(state[0], coag[flat_index], state[2], state[3])
