# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9b contact paint: a held mask tints the vertices of the triangles the tools touch, per env, clamped to [0, 1]."""


import numpy as np
from helpers import find, push_actions, run
from i4h_host.composition import load_case
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.contact_paint import ContactPaint
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface


def test_paint_follows_the_contacts_env_locally():
    sim = load_case("two_worlds")
    scene, body = sim.scene, sim.scene.bodies[0]
    control, contact, surface = find(sim, InstrumentControl), find(sim, InstrumentContact), find(sim, Surface)
    paint = ContactPaint(scene, contacts=contact, surface=surface)
    sim.systems.append(paint)
    assert paint.paint.shape == (2, 3) and float(paint.rate.numpy()[0]) == 3.0
    run(sim, 20, push_actions(control, scene.envs))  # the probe presses into env 0's block: live records, mask zero
    assert not surface.appearance.visual.numpy().any() and contact.record_count.numpy()[0].sum() > 0
    paint.paint.assign(np.float32([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    run(sim, 10, push_actions(control, scene.envs))
    visual = surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)
    painted = np.flatnonzero(visual[0][:, 0] > 0.0)
    assert painted.size and not visual[0][:, 1:3].any() and not visual[1].any()
    q = scene.view(scene.q, body).numpy()[0]
    probe = control.sphere_q.numpy()[0, 0]
    assert np.linalg.norm(q[painted] - probe, axis=1).max() < 0.015 + 0.02 + 0.03  # the touched triangles' vertices
    assert not paint.painted.numpy().any()  # the counter clears every step
    assert visual[0][:, 0].max() <= 1.0
    run(sim, 100, push_actions(control, scene.envs))
    visual = surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)
    assert visual[0][:, 0].max() == 1.0 and visual[0][:, 0].min() >= 0.0  # 3 per second saturates within a second, clamped
    paint.paint.assign(np.float32([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    run(sim, 10, push_actions(control, scene.envs))
    visual = surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)
    assert (visual[0][:, 2] > 0).any() and not visual[0][:, 1].any() and not visual[1].any()
    assert np.isfinite(scene.q.numpy()).all()
    sim.reset([0])
    assert not surface.appearance.visual.numpy().any() and not paint.painted.numpy().any()
