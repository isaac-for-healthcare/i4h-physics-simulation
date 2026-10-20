# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M1 check: chole tissue falls under gravity, hangs from the pin and settles inside the bounds."""

import os

import numpy as np
import pytest
from helpers import find, tet_volumes
from i4h_isaaclab.composition import data_dir, load_case
from i4h_physics.systems.mechanics.attachments import Attachments
from i4h_physics.systems.mechanics.bounds import Bounds
from i4h_physics.systems.mechanics.connectors import Connectors
from i4h_physics.systems.mechanics.tet import Tet

DATA = data_dir() / "choly-mid"
pytestmark = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not DATA.is_dir(), reason=f"chole data not found at {DATA}")


def connector_residuals(sim):
    """Mean |distance - rest| per connector group."""
    scene, c = sim.scene, find(sim, Connectors)
    q = scene.view(scene.q, scene.body('tissue')).numpy()[0]
    p, t, b, r = (a.numpy()[0] for a in (c.particle, c.tri, c.bar, c.rest))
    g = c.group.numpy()
    tri_pos = np.einsum("ijk,ij->ik", q[t], b)
    d = np.abs(np.linalg.norm(q[p] - tri_pos, axis=1) - r)
    return [float(d[g == k].mean()) for k in range(g.max() + 1)]


def test_connectors_hold_parts():
    """With connectors the fat sits on the liver and the gallbladder on the fat (measured
    residuals 4e-4 and 9e-4); without them the fat falls a metre away."""
    residual = {}
    for enabled in (True, False):
        sim = load_case("chole_tet_med")
        find(sim, Connectors).enabled = enabled
        for _ in range(200):
            sim.step()
        residual[enabled] = connector_residuals(sim)
    assert max(residual[True]) < 0.005, residual
    assert min(residual[False]) > 0.02, residual


def test_tissue_hangs():
    sim = load_case("chole_tet_med")
    scene = sim.scene
    body = scene.body('tissue')
    q0 = scene.view(scene.q0, body).numpy()[0]
    active = scene.view(scene.flags, body).numpy()[0] != 0
    pins = find(sim, Attachments).index.numpy()[0]
    pins = pins[pins >= 0]
    liver = body.parts["liver"]
    liver_slice = slice(liver.particle_start, liver.particle_start + liver.particle_count)
    bounds = find(sim, Bounds)

    energy = []
    for step in range(600):
        sim.step()
        if step % 10 == 9 or step >= 540:
            qd = scene.view(scene.qd, body).numpy()[0]
            energy.append((step, float(0.5 * 0.1 * np.sum(qd[active] ** 2))))
    assert sim.graph is not None
    q = scene.view(scene.q, body).numpy()[0]
    assert np.isfinite(q).all()

    lower, upper = np.array(bounds.lower), np.array(bounds.upper)
    assert np.all(q[active] >= lower - 1e-5) and np.all(q[active] <= upper + 1e-5), "tissue left the bounds box"
    assert q[active][:, 1].min() > 0.05, "tissue fell to the floor"
    # Calibrated on 2026-09-17: the 0.0005 pin lerp lets the pinned rim sag 0.19 and the
    # liver centroid drop 0.19 (fat 0.27, gallbladder 0.28) before it settles by step ~120.
    pin_drift = np.linalg.norm(q[pins] - q0[pins], axis=1).max()
    assert pin_drift < 0.25, f"pinned particles drifted {pin_drift:.3f}"
    drop = q0[liver_slice][:, 1].mean() - q[liver_slice][:, 1].mean()
    assert 0.05 < drop < 0.3, f"liver centroid moved {drop:.3f} in y"

    energy = np.array(energy)
    peak = energy[:, 1].max()
    tail = energy[energy[:, 0] >= 540][:, 1].mean()
    assert tail < 0.1 * peak, f"tissue did not settle: tail {tail:.3e} vs peak {peak:.3e}"

    # Volume constraint at work per part. Measured totals 1.000 / 1.058 / 1.010; individual
    # sliver tets (rest volume 50-2000x below the part median) do invert, so totals are tested.
    tet = find(sim, Tet)
    tets, rest, part = tet.tets.numpy()[0], tet.rest_volume.numpy()[0], tet.tet_part.numpy()
    volume = tet_volumes(q, tets)
    for index, (name, limit) in enumerate((("liver", 0.01), ("fat", 0.1), ("gallbladder", 0.03))):
        sel = part == index
        ratio = volume[sel].sum() / rest[sel].sum()
        assert abs(ratio - 1.0) < limit, f"{name} total volume ratio {ratio:.3f}"


def test_alias_convention_holds():
    """Every M1 kernel (tet, attachments, connectors, motion limits) keeps inputs and outputs disjoint."""
    sim = load_case("chole_tet_med", debug=True)
    sim.cuda_graph = False
    for _ in range(3):
        sim.step()
    assert np.isfinite(sim.scene.q.numpy()).all()


def test_identical_envs_agree():
    """Four identical envs settle to the same state. Float atomics make the fall itself
    non-deterministic (~3 mm measured at M1, occasionally more on a sliver tet), so compare after
    settling and with a tolerance far below the metre-scale error an env-indexing leak would produce."""
    sim = load_case("chole_tet_med", envs=4)
    scene = sim.scene
    body = scene.body('tissue')
    for _ in range(600):
        sim.step()
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all()
    for env in range(1, 4):
        assert np.allclose(q[env], q[0], rtol=1e-3, atol=3e-2), f"env {env} diverged from env 0"
