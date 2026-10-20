# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tissue against the static cavity through the Warp mesh query: a block dropped inside stays inside."""

import os

import numpy as np
import pytest
import warp as wp
from i4h_isaaclab.composition import data_dir
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_static_mesh, add_tet_grid
from i4h_physics.systems.contact.static_contact import StaticContact
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.mechanics.tet import Tet

DATA = data_dir() / "choly" / "cavity"
pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not DATA.is_dir(), reason=f"cavity data not found at {DATA}")


@wp.kernel
def signed_distance(mesh: wp.uint64, q: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    r = wp.mesh_query_point_sign_normal(mesh, q[i], 10.0)
    out[i] = wp.length(q[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v)) * r.sign


def test_block_stays_inside_the_cavity():
    scene = Scene("cuda:0", 2, (0.0, -200.0, 0.0))  # strong gravity: the block hits the floor of the cavity
    add_static_mesh(scene, "cavity", str(DATA), translation=(0.8, -3.3, -0.4))
    centre = np.float32([0.53, 1.32, -3.71])  # the chole gallbladder centroid: 13 cm from the nearest wall
    add_tet_grid(scene, "block", dim=(2, 2, 2), cell=0.02, origin=tuple(centre - 0.02), density=1000.0, radius=0.005,
                 parts={"block": {"spring": 1.0, "volume": 1.0}})
    scene.allocate()
    static = StaticContact(scene, body="block", mesh="cavity", margin=0.01)
    sim = Stepper.create(scene, [Particles(scene, max_velocity=5.0), Tet(scene, body="block"), static], dt=1 / 60, substeps=4, iterations=8)
    out = wp.zeros(scene.particle_count, dtype=wp.float32, device=scene.device)
    wp.launch(signed_distance, dim=scene.particle_count, inputs=[static.wp_mesh.id, scene.q], outputs=[out], device=scene.device)
    assert out.numpy().min() > 0.05  # starts well inside
    lowest = np.inf
    for _ in range(600):
        sim.step()
        wp.launch(signed_distance, dim=scene.particle_count, inputs=[static.wp_mesh.id, scene.q], outputs=[out], device=scene.device)
        lowest = min(lowest, float(out.numpy().min()))
    assert np.isfinite(scene.q.numpy()).all()
    assert lowest > 0.0, lowest  # never crossed the wall (the impact transient dips inside the margin, not the wall)
    final = out.numpy().min()
    assert abs(final - (0.005 + 0.01)) < 3e-3, final  # rests on the floor at radius + margin
    q = scene.view(scene.q, scene.body("block")).numpy()
    assert (q[:, :, 1].max(axis=1) < centre[1]).all()  # both envs fell (the bounce amplifies atomic noise, so no bitwise check)
