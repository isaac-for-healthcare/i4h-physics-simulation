# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from i4h_assessment.executor import TargetMeasurement
from i4h_physics.scene import ACTIVE_INT


class ReachTarget(TargetMeasurement):
    is_reach = True
    def _bind_tool(self):
        if self.physics_binding is not None:
            binding = self.physics_binding
            self.tool_body = binding.body
            self.tool_index = binding.distal_node
            self.tool_q, self.tool_flags = binding.q, binding.flags
            if not np.all(self.tool_body.rest.flags[:, self.tool_index] & ACTIVE_INT):
                raise ValueError("reach_target tool tip is inactive or padded")
            return
        self.tool_body = self.scene.body(self.cfg["tool"])
        if self.tool_body.rod is None:
            raise ValueError("reach_target tool must be a rod")
        self.tool_index = self.tool_body.count - 1
        if not np.all(self.tool_body.rest.flags[:, self.tool_index] & ACTIVE_INT):
            raise ValueError("reach_target tool tip is inactive or padded")
        self.tool_q = self.scene.field("q", self.tool_body)
        self.tool_flags = self.scene.field("flags", self.tool_body)

    def _qualify(self):
        pass



def single_target_assessor(physics, *, target, radius, hold, timeout, horizon_s=None):
    from i4h_assessment.executor import Procedure
    physics.owner.check_valid()
    spec = dict(type="reach_target", tool=physics.body.name, target={"position": target}, radius=radius, hold=hold, timeout=timeout)
    return Procedure(spec, physics._sim, measurements={"reach_target": ReachTarget}, physics_binding=physics, horizon_s=horizon_s)
