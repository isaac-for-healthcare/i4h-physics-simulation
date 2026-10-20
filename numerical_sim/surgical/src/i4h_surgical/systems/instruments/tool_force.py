# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool force: the reaction each instrument feels from the tissue, per env and tool (M11 haptics).

New integration, not a port: omnisurg's force path (input/haptics.py:79-135, instruments/system.py:1084-1126)
ran on the excluded unified contact backend. Here every system that moves tissue on a tool's behalf publishes
the mass-weighted displacement it actually applied over the step, ``impulse (E, n) vec3`` = sum of m_i * dx_i
(``instrument_contact`` and both grasps through the per-tool ``delta_k`` channel reduced before their
``apply_average``; the hook from its in-place corrections). The tissue received that displacement over the
frame's substeps, so the average force it exerted on the tool is the negated sum over ``sdt^2 * substeps``
(momentum sum(m dx) / sdt per substep, averaged over the frame: / (sdt * dt)). Sim units: the asset masses
times sim length over s^2; the app's ``haptics.newtons_per_unit`` converts for a device.

Products: ``force (E, n)`` (on the tool, sim frame), ``count (E, n)`` contact pairs this step, and ``jaw_load (E, n)``:
the squeeze the tissue exerts against the closing jaws, from each source's ``jaw_impulse`` (the left jaw's share minus
the right jaw's; on tissue held between the jaws the two shares point opposite ways and add up where the net force
cancels), ``|left − right| / 2`` over the same ``sdt² · substeps``. A single jaw grazing tissue reports half its
reaction; a closed jaw pressed tip-first, both jaws pushing the same way, reports 0, as intended. The hook has no jaw
sign and reports 0. Live per-env gains weight the sources; ``max_force`` caps the force norm and the jaw load (0 =
uncapped). Optional sources pass a shared zero array so the launch set never depends on the case. Listed after its
sources; reads only their products.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from i4h_physics.systems.base import System
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.grasp import Grasp
from i4h_surgical.systems.instruments.hook import Hook
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp


@dataclass(eq=False)
class ToolForce(System):
    state_products = ("force", "count", "jaw_load")  # per-env products a host observes (System.state_products)
    instruments: InstrumentControl
    contacts: InstrumentContact
    grasp: Grasp | None = None
    volumetric_grasp: VolumetricGrasp | None = None
    hook: Hook | None = None
    contact_gain: float = 1.0  # live, per env
    grasp_gain: float = 1.0  # live, per env: surface and volumetric grasp holds
    hook_gain: float = 1.0  # live, per env
    max_force: float = 0.0  # live, per env; norm cap in sim units (0 = uncapped)

    refs = ("instruments", "contacts", "grasp", "volumetric_grasp", "hook")

    def __post_init__(self):
        scene = self.scene
        if not isinstance(self.instruments, InstrumentControl) or not isinstance(self.contacts, InstrumentContact):
            raise TypeError("tool_force.instruments must name an instrument_control system and tool_force.contacts an instrument_contact system")
        for name, cls in (("grasp", Grasp), ("volumetric_grasp", VolumetricGrasp), ("hook", Hook)):
            ref = getattr(self, name)
            if ref is not None and not isinstance(ref, cls):
                raise TypeError(f"tool_force.{name} must name a {name} system")
        E, dev, n = scene.envs, scene.device, max(self.instruments.count, 1)
        self._live("contact_gain", "grasp_gain", "hook_gain", "max_force")
        self._zero = wp.zeros((E, n), dtype=wp.vec3, device=dev)  # stands in for a missing optional source
        self.force = wp.zeros((E, n), dtype=wp.vec3, device=dev)  # product: force on the tool, sim frame and units
        self.count = wp.zeros((E, n), dtype=wp.int32, device=dev)  # product: contact pairs this step
        self.jaw_load = wp.zeros((E, n), dtype=wp.float32, device=dev)  # product: the squeeze against the closing jaws (sim force units)
        self._substeps = 1


    def _sources(self):
        return [s for s in (self.contacts, self.grasp, self.volumetric_grasp, self.hook) if s is not None]

    def prepare(self, scene):
        """A disabled ``tool_force`` would keep publishing its last force: cleared here. (A disabled source clears its own
        impulse products in its ``prepare``.)"""
        super().prepare(scene)
        if not self.enabled:
            self.force.zero_()
            self.count.zero_()
            self.jaw_load.zero_()

    def substep_start(self, scene, dt, s, substeps):
        self._substeps = substeps

    def step_end(self, scene, dt):
        if self.instruments.count == 0:
            return
        sdt = dt / float(self._substeps)
        grasp = self.grasp.impulse if self.grasp is not None else self._zero
        grasp_jaw = self.grasp.jaw_impulse if self.grasp is not None else self._zero
        volumetric = self.volumetric_grasp.impulse if self.volumetric_grasp is not None else self._zero
        volumetric_jaw = self.volumetric_grasp.jaw_impulse if self.volumetric_grasp is not None else self._zero
        hook = self.hook.impulse if self.hook is not None else self._zero
        wp.launch(tool_force, dim=(scene.envs, self.instruments.count),
                  inputs=[self.contacts.impulse, self.contacts.impulse_count, grasp, volumetric, hook, self.contacts.jaw_impulse, grasp_jaw,
                          volumetric_jaw, self.contact_gain, self.grasp_gain, self.hook_gain, self.max_force, 1.0 / (sdt * sdt * float(self._substeps))],
                  outputs=[self.force, self.count, self.jaw_load], device=scene.device)

    def reset(self, scene, mask):
        self.zero(mask, "force", "count", "jaw_load")


@wp.kernel
def tool_force(
    contact: wp.array2d(dtype=wp.vec3),
    contact_count: wp.array2d(dtype=wp.int32),
    grasp: wp.array2d(dtype=wp.vec3),
    volumetric: wp.array2d(dtype=wp.vec3),
    hook: wp.array2d(dtype=wp.vec3),
    contact_jaw: wp.array2d(dtype=wp.vec3),
    grasp_jaw: wp.array2d(dtype=wp.vec3),
    volumetric_jaw: wp.array2d(dtype=wp.vec3),
    contact_gain: wp.array(dtype=wp.float32),
    grasp_gain: wp.array(dtype=wp.float32),
    hook_gain: wp.array(dtype=wp.float32),
    max_force: wp.array(dtype=wp.float32),
    per_mass_dx: wp.float32,
    force: wp.array2d(dtype=wp.vec3),
    count: wp.array2d(dtype=wp.int32),
    jaw_load: wp.array2d(dtype=wp.float32),
):
    env, k = wp.tid()
    imposed = contact[env, k] * contact_gain[env] + (grasp[env, k] + volumetric[env, k]) * grasp_gain[env] + hook[env, k] * hook_gain[env]
    f = -imposed * per_mass_dx  # the tissue pushes back on the tool
    cap = max_force[env]
    norm = wp.length(f)
    if cap > 0.0 and norm > cap:
        f = f * (cap / norm)
    force[env, k] = f
    count[env, k] = contact_count[env, k]
    squeeze = contact_jaw[env, k] * contact_gain[env] + (grasp_jaw[env, k] + volumetric_jaw[env, k]) * grasp_gain[env]
    load = 0.5 * wp.length(squeeze) * per_mass_dx  # left minus right: twice the squeeze on tissue held between the jaws
    if cap > 0.0 and load > cap:
        load = cap
    jaw_load[env, k] = load
