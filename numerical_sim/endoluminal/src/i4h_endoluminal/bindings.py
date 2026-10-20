# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_endoluminal.systems.rods.catheter_control import CatheterControl


class CatheterBinding:
    def __init__(self, sim, body="catheter"):
        self._sim = sim
        self.owner = sim.physics
        self.evidence = None
        self.owner.check_valid()
        self.body = sim.scene.body(body)
        control = next(
            s
            for s in sim.systems
            if isinstance(s, CatheterControl) and s.body is self.body
        )
        self.envs, self.device = sim.scene.envs, sim.scene.device
        self.q = sim.scene.field("q", self.body)
        self.qd = sim.scene.field("qd", self.body)
        self.flags = sim.scene.field("flags", self.body)
        self.distal_node = self.body.count - 1
        self.commands, self.targets = control.actions, control.targets
        from i4h_physics.systems.rods.rod_pbd import RodPbd

        self.material_backend = "pbd" if isinstance(control.rod, RodPbd) else "direct"
        names = ("stretch", "bend", "twist") if self.material_backend == "pbd" else ("young", "torsion", "bend", "twist")
        self.materials = {name: getattr(control.rod, name) for name in names}

    @property
    def valid(self):
        return self.owner.valid

    def configure_tip_evidence(self, catheter, capacity):
        """Bind before final prepare/capture; changing capacity requires reconstruction."""
        from i4h_endoluminal.systems.tip_evidence import TipEvidence

        self.owner.check_valid()
        if catheter.owner is not self.owner:
            raise ValueError("catheter must belong to this physics binding")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("evidence capacity must be a positive integer")
        existing = next((system for system in self._sim.systems
                         if isinstance(system, TipEvidence) and system.owner is self.owner), None)
        if existing is not None:
            self.evidence = existing
        if self.evidence is not None:
            if self.evidence.body is not catheter.body or self.evidence.capacity != capacity:
                raise ValueError("evidence body/capacity is fixed; reconstruct the runtime")
            return self.evidence
        self.evidence = TipEvidence(self.owner, catheter, capacity)
        self._sim.systems.append(self.evidence)
        return self.evidence
