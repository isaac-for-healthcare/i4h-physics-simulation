# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Simulation-level lifetime and physical reset ownership.

A binding keeps its simulation alive, rejects use after explicit invalidation,
and delegates masked reset to the physical stepper. It stores no assessment or
domain-specific views. Domain bindings retain this owner while exposing their
own arrays; hard rebuild or close invalidates the owner before consumers rebind.
"""



class PhysicsBinding:
    """One simulation-level owner for physical reset and the lifetime of public views."""

    def __init__(self, sim):
        self.sim = sim
        self.valid = True

    def check_valid(self):
        if not self.valid:
            raise RuntimeError("Physics binding invalidated by hard rebuild or close; bind again")

    def invalidate(self):
        self.valid = False

    def reset(self, env_ids=None, *, mask=None):
        self.check_valid()
        self.sim.reset_physics(env_ids, mask=mask)
