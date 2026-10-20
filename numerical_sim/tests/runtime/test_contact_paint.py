# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9b contact paint: a held mask tints the vertices of the triangles the tools touch, per env, clamped to [0, 1]."""

from types import SimpleNamespace

import numpy as np
import warp as wp
from helpers import find, push_actions, run
from i4h_isaaclab.composition import load_case
from i4h_standalone.input.keyboard import PAINT_KEYS, PaintKeys
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.contact_paint import ContactPaint
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface


class _Key:
    def __init__(self, name, press, release=False):
        self.key, self._press, self._release = SimpleNamespace(name=name), press, release

    def is_key_press(self):
        return self._press

    def is_key_release(self):
        return self._release

    def is_key_repeat(self):
        return False


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


def test_paint_keys_write_the_drawn_env():
    from i4h_isaaclab.camera.camera import Camera
    from i4h_standalone.app.instruments import InstrumentDriver

    keys = PaintKeys()
    assert keys.mask().tolist() == [0.0, 0.0, 0.0] and list(PAINT_KEYS) == ["m", "v", "b"]
    keys.handle_key(_Key("m", True))
    keys.handle_key(_Key("b", True))
    keys.handle_key(_Key("n", True))  # step-once, not a paint key
    assert keys.mask().tolist() == [1.0, 0.0, 1.0]
    keys.handle_key(_Key("m", False, release=True))
    assert keys.mask().tolist() == [0.0, 0.0, 1.0]

    sim = load_case("two_worlds")
    control = find(sim, InstrumentControl)
    mask = wp.zeros((2, 3), dtype=wp.float32, device=sim.scene.device)
    renderer = SimpleNamespace(env=0, camera=Camera(), key_handlers=[])
    driver = InstrumentDriver(sim, control, None, "keyboard", renderer=renderer, paint=mask)
    assert driver.paint_keys is not None and driver.paint_keys.handle_key in renderer.key_handlers
    driver.paint_keys.handle_key(_Key("v", True))
    driver.update(0.0)
    assert mask.numpy().tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    renderer.env = 1
    driver.update(sim.dt)
    assert mask.numpy().tolist() == [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # the left env stops painting
    scripted = InstrumentDriver(sim, control, None, "scripted", renderer=renderer, paint=mask)
    assert scripted.paint_keys is None  # no window keys in scripted mode
