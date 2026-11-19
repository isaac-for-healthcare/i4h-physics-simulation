# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock loop timing: physics includes manager work and required synchronization."""


class LoopTiming:
    def __init__(self):
        self.steps = 0
        self.physics = self.viewer = self.loop = 0.0

    def add(self, start, physics_end, render_end, *, warmup=False):
        if not warmup:
            self.steps += 1
            self.physics += physics_end - start
            self.viewer += render_end - physics_end
            self.loop += render_end - start

    def report(self):
        if not self.steps:
            return 'timing: no samples after warmup'
        ms = 1000.0 / self.steps
        return ('mean ms: physics %.3f, viewer %.3f, total loop %.3f'
                % (self.physics * ms, self.viewer * ms, self.loop * ms))
