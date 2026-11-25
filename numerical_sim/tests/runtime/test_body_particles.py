# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independent rod and wall prediction without integrating either twice."""
import copy

import numpy as np
import pytest
from i4h_host.composition import load_case


@pytest.mark.parametrize('captured', [False, True])
def test_independent_body_gravity_and_projected_velocity_damping(captured):
    rod = dict(type='rod', start=[0, 0, 0], direction=[1, 0, 0], count=3,
               segment_length=.01, radius=.001, lock_root=False)
    cfg = dict(sim=dict(device='cuda:0', envs=2, dt=.01, substeps=1, iterations=1,
                        gravity=[0, 0, -10], cuda_graph=captured),
               assets=dict(wire=rod, wall=copy.deepcopy(rod)),
               systems=[dict(type='particles', id='wire', body='wire', damping=.5),
                        dict(type='particles', id='wall', body='wall', gravity_scale=0,
                             damping=.25, damping_at_finalize=True)])
    sim = load_case(cfg)
    for body in sim.scene.bodies:
        sim.scene.field('qd', body).fill_((0, 1, 0))
    sim.step()
    wire, wall = (sim.scene.body(name) for name in ('wire', 'wall'))
    np.testing.assert_allclose(sim.scene.field('q', wire).numpy()[..., 1], .005)
    np.testing.assert_allclose(sim.scene.field('q', wire).numpy()[..., 2], -.0005)
    np.testing.assert_allclose(sim.scene.field('q', wall).numpy()[..., 1], .01)
    np.testing.assert_allclose(sim.scene.field('q', wall).numpy()[..., 2], 0)
    np.testing.assert_allclose(sim.scene.field('qd', wall).numpy()[..., 1], .25)
    before = sim.scene.q.numpy().copy()
    sim.reset([0])
    np.testing.assert_array_equal(sim.scene.q.numpy().reshape(2, -1, 3)[1], before.reshape(2, -1, 3)[1])
    for systems in ([cfg['systems'][0]], [cfg['systems'][0], dict(cfg['systems'][0], id='again')]):
        bad = copy.deepcopy(cfg)
        bad['systems'] = systems
        with pytest.raises(ValueError, match='each main body exactly once'):
            load_case(bad)
