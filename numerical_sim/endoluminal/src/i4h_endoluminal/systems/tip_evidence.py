# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity, physics-rate distal position/active evidence. No task policy."""

import warp as wp
from i4h_physics.systems.base import System


@wp.kernel
def record_tip(q: wp.array2d(dtype=wp.vec3), flags: wp.array2d(dtype=int), node: int,
               positions: wp.array2d(dtype=wp.vec3), active: wp.array2d(dtype=int),
               count: wp.array(dtype=int), overflow: wp.array(dtype=int)):
    e = wp.tid()
    slot = count[e]
    if slot < positions.shape[1]:
        positions[e, slot] = q[e, node]
        active[e, slot] = flags[e, node] & 1
    else:
        overflow[e] = 1
    count[e] = slot + 1


class TipEvidence(System):
    structural = ("enabled", "capacity", "node")

    def __init__(self, owner, catheter, capacity):
        super().__init__(owner.sim.scene)
        self.owner = owner
        self.body = catheter.body
        self.node = catheter.distal_node
        self.capacity = capacity
        self.q, self.flags = catheter.q, catheter.flags
        shape, dev = (catheter.envs, capacity), catheter.device
        self.positions = wp.zeros(shape, dtype=wp.vec3, device=dev)
        self.active = wp.zeros(shape, dtype=int, device=dev)
        self.count = wp.zeros(catheter.envs, dtype=int, device=dev)
        self.overflow = wp.zeros_like(self.count)
        self.freeze("capacity", "node")

    @property
    def valid(self):
        return self.owner.valid

    def begin_action(self):
        self.owner.check_valid()
        self.count.zero_()
        self.overflow.zero_()

    def step_end(self, scene, dt):
        wp.launch(record_tip, scene.envs, inputs=[self.q, self.flags, self.node],
                  outputs=[self.positions, self.active, self.count, self.overflow], device=scene.device)

    def reset(self, scene, mask):
        self.zero(mask, "positions", "active", "count", "overflow")
