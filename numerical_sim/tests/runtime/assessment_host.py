# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class AssessmentHost:
    """Test host evaluates one sample after physics and resets assessment explicitly."""
    def __init__(self, simulation, procedure=None):
        self.simulation = simulation
        self.procedure = procedure

    def step(self):
        self.simulation.step()
        if self.procedure is not None:
            self.procedure.update(self.dt)

    def reset(self, env_ids=None, *, mask=None):
        if not self.simulation.reset_physics(env_ids, mask=mask):
            return
        if self.procedure is not None:
            self.procedure.reset(mask=self.simulation._mask)

    def prepare(self):
        self.simulation.prepare()

    @property
    def scene(self):
        return self.simulation.scene

    @property
    def systems(self):
        return self.simulation.systems

    @property
    def dt(self):
        return self.simulation.dt

    @property
    def physics(self):
        return self.simulation.physics

    @property
    def graph(self):
        return self.simulation.graph

    @property
    def reset_graph(self):
        return self.simulation.reset_graph

    @property
    def solver(self):
        return self.simulation.solver

    @property
    def model(self):
        return self.simulation.model

    @property
    def state(self):
        return self.simulation.state

    @property
    def _mask(self):
        return self.simulation._mask

    @property
    def cuda_graph(self):
        return self.simulation.cuda_graph

    @cuda_graph.setter
    def cuda_graph(self, value):
        self.simulation.cuda_graph = value
