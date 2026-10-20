# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact paint: held keys tint the tissue the instruments touch (damage, coagulation or blood).

Ported from omnisurg/instruments/kernels.py:3447-3516 (``paint_tissue_visual_from_surface_contacts``: per contact
record, ``mask * strength * dt`` spread by the barycentric weights onto the record triangle's vertices) and
:3518-3546 (``clamp_tissue_visual_rgb_after_paint``: clamp the painted vertices to [0, 1]); the mask came from the
window keys (system.py:3436-3467, M/N/B) and ``tissue_paint {enabled, strength_per_second 3.0}`` (schema.py:199-202).
Edits: ``(env, capsule, record)`` prologue on ``instrument_contact``'s records; the mask is a live ``(E, 3)`` value the
app's keyboard writes (M damage, V coagulation, B blood: N is the window's step-once key); runs once per step with
the frame dt instead of once per substep with the substep dt (from the last substep's records only, so it differs from
omnisurg's sum when contact changes within the step); a dead record tet paints
nothing. A debug tool: list it after ``instrument_contact``; cases opt in.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.systems.base import System, live
from i4h_physics.systems.mechanics.surface import Surface
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact


@dataclass(eq=False)
class ContactPaint(System):
    contacts: InstrumentContact
    surface: Surface
    paint: tuple = (0.0, 0.0, 0.0)  # live (E, 3): damage, coagulation, blood painted per second of contact (the keyboard's mask)
    rate: float = 3.0  # live, per env: omnisurg strength_per_second

    refs = ("contacts", "surface")

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.contacts, InstrumentContact) or not isinstance(self.surface, Surface):
            raise TypeError("contact_paint.contacts must name an instrument_contact system and contact_paint.surface a surface system")
        if self.contacts.surface is not self.surface:
            raise ValueError("contact_paint.surface must be the surface its contacts collide with")
        if len(tuple(self.paint)) != 3:
            raise ValueError("contact_paint.paint needs three values (damage, coagulation, blood)")
        E, dev = scene.envs, scene.device
        self.body = self.surface.body
        self.paint = live(scene, tuple(float(v) for v in self.paint))  # (E, 3)
        self.register_live("paint")
        self._live("rate")
        self.painted = wp.zeros(E * self.body.count, dtype=wp.int32, device=dev)  # body-local, self-clearing


    def step_end(self, scene, dt):
        cc, sf, b = self.contacts, self.surface, self.body
        E, P, R = scene.envs, cc.record_count.shape[1], cc.rec_verts.shape[2]
        wp.launch(paint_contacts, dim=(E, P, R),
                  inputs=[cc.record_count, cc.rec_tet, cc.rec_verts, cc.rec_bary, sf.active, self.paint, self.rate, dt, b.count],
                  outputs=[sf.appearance.visual, self.painted], device=scene.device)
        wp.launch(clamp_paint, dim=(E, b.count), inputs=[b.count], outputs=[sf.appearance.visual, self.painted], device=scene.device)

    def reset(self, scene, mask):
        self.clear(mask, (self.painted, wp.int32(0)))


# kernels.py:3447-3516 (paint_tissue_visual_from_surface_contacts). Edits: (env, capsule, record) prologue, live mask.
@wp.kernel
def paint_contacts(
    record_count: wp.array2d(dtype=wp.int32),
    rec_tet: wp.array3d(dtype=wp.int32),
    rec_verts: wp.array3d(dtype=wp.vec3i),
    rec_bary: wp.array3d(dtype=wp.vec3),
    tet_active: wp.array2d(dtype=wp.int32),
    paint: wp.array2d(dtype=wp.float32),
    rate: wp.array(dtype=wp.float32),
    dt: wp.float32,
    count: wp.int32,
    visual: wp.array(dtype=wp.vec4f),
    painted: wp.array(dtype=wp.int32),
):
    env, p, r = wp.tid()
    if r >= record_count[env, p]:
        return
    amount = rate[env] * dt
    if amount <= 0.0:
        return
    m0 = paint[env, 0]
    m1 = paint[env, 1]
    m2 = paint[env, 2]
    if m0 <= 0.0 and m1 <= 0.0 and m2 <= 0.0:
        return
    t = rec_tet[env, p, r]
    if t < 0 or tet_active[env, t] == 0:
        return
    verts = rec_verts[env, p, r]
    bary = rec_bary[env, p, r]
    delta = wp.vec4f(m0, m1, m2, 0.0) * amount
    for c in range(3):
        if bary[c] > 0.0:
            flat_index = env * count + verts[c]
            wp.atomic_add(visual, flat_index, delta * bary[c])
            wp.atomic_add(painted, flat_index, wp.int32(1))


# kernels.py:3518-3546 (clamp_tissue_visual_rgb_after_paint). Edits: (env, particle) prologue.
@wp.kernel
def clamp_paint(
    count: wp.int32,
    visual: wp.array(dtype=wp.vec4f),
    painted: wp.array(dtype=wp.int32),
):
    env, local = wp.tid()
    flat_index = env * count + local
    if painted[flat_index] == 0:
        return
    v = visual[flat_index]
    visual[flat_index] = wp.vec4f(wp.clamp(v[0], 0.0, 1.0), wp.clamp(v[1], 0.0, 1.0), wp.clamp(v[2], 0.0, 1.0), v[3])
    painted[flat_index] = 0
