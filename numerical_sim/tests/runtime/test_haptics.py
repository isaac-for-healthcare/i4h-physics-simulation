# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M11 haptics: the ``tool_force`` product on two_worlds and chole."""


import os


import numpy as np


import pytest


import warp as wp


from i4h_host.composition import data_dir, load_case


from i4h_surgical.systems.contact.instrument_contact import InstrumentContact


from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


from i4h_surgical.systems.instruments.tool_force import ToolForce


CHOLE = data_dir() / "choly-mid"


needs_chole = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not CHOLE.is_dir(), reason=f"chole data not found at {CHOLE}")


def find(sim, cls):
    return next(s for s in sim.systems if isinstance(s, cls))


def press(sim, z, envs=(0,), steps=60):
    """Hold every probe at rest and the listed envs' probes at height ``z`` (pushing the block's bottom face up)."""
    control = find(sim, InstrumentControl)
    actions = np.zeros((sim.scene.envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    for env in envs:
        actions[env, :, 2] = z
    control.actions.assign(actions)
    forces = []
    for _ in range(steps):
        sim.step()
        forces.append(find(sim, ToolForce).force.numpy().copy())
    return np.array(forces)  # (steps, E, n, 3)


def test_probe_force_direction_penetration_gain_cap_and_release():
    sim = load_case("two_worlds")
    tf, contact = find(sim, ToolForce), find(sim, InstrumentContact)
    E = sim.scene.envs

    settled = press(sim, 0.0, envs=(), steps=60)  # probes parked below the block: no contact anywhere
    assert np.all(settled == 0.0) and tf.count.numpy().max() == 0

    shallow = press(sim, 0.045, steps=90)
    f = shallow[-30:, 0, 0].mean(axis=0)
    assert np.isfinite(shallow).all() and tf.count.numpy()[0, 0] > 0
    assert f[2] < 0.0 and abs(f[2]) > 5.0 * np.hypot(f[0], f[1]), f  # the block pushes the probe back down
    assert np.all(shallow[:, 1] == 0.0) and tf.count.numpy()[1, 0] == 0  # env 1's probe is parked: isolation
    assert np.all(tf.jaw_load.numpy() == 0.0)  # a probe has no jaws: the signed channel stays empty while the force is not
    sim.reset()
    deeper = press(sim, 0.05, steps=90)[-30:, 0, 0].mean(axis=0)
    assert deeper[2] < f[2], (deeper, f)  # deeper penetration, larger reaction

    tf.contact_gain.fill_(0.0)  # live: no recapture
    assert np.all(press(sim, 0.05, steps=5)[-1] == 0.0) and tf.count.numpy()[0, 0] > 0
    tf.contact_gain.fill_(1.0)
    cap = 0.25 * abs(deeper[2])
    tf.max_force.fill_(cap)
    capped = press(sim, 0.05, steps=10)[-1, 0, 0]
    assert np.isclose(np.linalg.norm(capped), cap, rtol=1e-4), (capped, cap)
    tf.max_force.fill_(0.0)

    released = press(sim, 0.0, envs=(), steps=3)[-1]
    assert np.all(released == 0.0) and tf.count.numpy().max() == 0  # the product is per step, not a latch
    assert contact.impulse.numpy().shape == (E, 1, 3) and contact.delta_k.numpy().shape == (E * contact.body.count, 1, 3)


def test_channel_matches_the_applied_displacement():
    """One tool: the reduced channel must equal the mass-weighted displacement ``apply_average`` applies, so a
    contact event's impulse increment is ``sum m_i * delta_i / count_i`` over the particles it moved."""

    sim = load_case("two_worlds")
    sim.cuda_graph = False
    scene, contact = sim.scene, find(sim, InstrumentContact)
    press(sim, 0.05, steps=20)
    # Re-run one contact event by hand: the contact kernel scatters, the reduce sums, the apply moves.
    control = find(sim, InstrumentControl)
    control.step_start(scene, sim.dt)
    contact.substep_start(scene, sim.dt / sim.substeps, 0, sim.substeps)
    contact.impulse.zero_()
    q_before = scene.q.numpy().copy()
    contact.iteration(scene, sim.dt / sim.substeps, sim.iterations - 1, sim.iterations)  # the last turn always carries an event
    q_after = scene.q.numpy()
    dx = q_after - q_before
    m = np.where(scene.inv_mass.numpy() > 0.0, 1.0 / np.maximum(scene.inv_mass.numpy(), 1e-30), 0.0)
    expected = (dx * m[:, None]).reshape(scene.envs, -1, 3).sum(axis=1)  # only the block moved (one body touched)
    got = contact.impulse.numpy()[:, 0]
    assert np.linalg.norm(expected[0]) > 0.0
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-7 * np.abs(expected).max()), (got, expected)
    assert np.all(contact.delta_k.numpy() == 0.0)  # the reduce clears the channel


def test_masked_reset_captured_and_batching():
    sim = load_case("two_worlds")
    tf = find(sim, ToolForce)
    press(sim, 0.05, envs=(0, 1), steps=30)
    f0 = tf.force.numpy()[0].copy()
    assert (tf.force.numpy() != 0.0).any(axis=(1, 2)).all()
    sim.reset([1])
    assert np.array_equal(tf.force.numpy()[0], f0) and np.all(tf.force.numpy()[1] == 0.0) and tf.count.numpy()[1].max() == 0
    assert np.all(tf.jaw_load.numpy()[1] == 0.0)

    captured, eager = load_case("two_worlds"), load_case("two_worlds")
    eager.cuda_graph = False
    a = press(captured, 0.05, envs=(0, 1), steps=40)
    b = press(eager, 0.05, envs=(0, 1), steps=40)
    scale = np.abs(a).max()
    assert np.allclose(a[-1], b[-1], rtol=1e-3, atol=2e-3 * scale), (a[-1], b[-1])  # float atomics: tolerance, not bitwise

    launches = []
    orig = wp.launch

    def spy(kernel, dim, *args, **kwargs):
        launches.append((kernel.key, dim))
        return orig(kernel, dim, *args, **kwargs)

    wp.launch = spy
    try:
        eager.step()
    finally:
        wp.launch = orig
    ours = [d for k, d in launches if k in ("tool_force", "reduce_tool_delta")]
    assert ours and all((d[0] if isinstance(d, (tuple, list)) else d) == eager.scene.envs for d in ours)


def test_state_views_expose_force_and_count():
    torch = pytest.importorskip("torch")
    from i4h_isaaclab.views import state_views

    sim = load_case("two_worlds")
    tf = find(sim, ToolForce)
    views = state_views(sim)
    force = next(v for v in views if v.data_ptr() == tf.force.ptr)
    count = next(v for v in views if v.data_ptr() == tf.count.ptr)
    jaw = next(v for v in views if v.data_ptr() == tf.jaw_load.ptr)
    assert tuple(force.shape) == (2, 3) and tuple(count.shape) == (2, 1) and tuple(jaw.shape) == (2, 1) and force.dtype == torch.float32


def test_disabled_sources_and_a_disabled_tool_force_publish_no_stale_force():
    """A source disabled through prepare() stops zeroing its impulse; the force must not keep its last contribution."""
    sim = load_case("two_worlds")
    tf, contact = find(sim, ToolForce), find(sim, InstrumentContact)
    press(sim, 0.05, steps=30)
    assert np.abs(tf.force.numpy()[0, 0]).max() > 0.0
    contact.record_count.fill_(1)
    contact.enabled = False
    sim.prepare()
    assert not contact.record_count.numpy().any()
    press(sim, 0.0, envs=(), steps=3)  # the probe retreats: nothing touches
    assert np.all(tf.force.numpy() == 0.0) and tf.count.numpy().max() == 0 and np.all(contact.impulse.numpy() == 0.0)
    assert np.all(contact.jaw_impulse.numpy() == 0.0) and np.all(tf.jaw_load.numpy() == 0.0)
    contact.enabled = True
    sim.prepare()
    press(sim, 0.05, steps=30)
    assert np.abs(tf.force.numpy()[0, 0]).max() > 0.0
    contact.enabled = False
    sim.prepare()
    press(sim, 0.05, steps=2)
    sim.reset([0])  # a reset with a disabled source: its stale impulse goes too
    press(sim, 0.05, steps=2)
    assert np.all(tf.force.numpy()[0] == 0.0)
    contact.enabled = True
    tf.enabled = False
    sim.prepare()
    press(sim, 0.05, steps=5)
    assert np.all(tf.force.numpy() == 0.0)  # a disabled tool_force publishes nothing


@needs_chole
def test_disabled_contact_and_hook_clear_transients_preserve_damage():
    from i4h_surgical.systems.instruments.hook import Hook
    sim = load_case("chole_tet_med", device="cpu", envs=2)
    contact = find(sim, InstrumentContact)
    hook = find(sim, Hook)
    contact.record_count.fill_(1)
    hook.active.fill_(1)
    contact.impulse.fill_((1, 2, 3))
    hook.impulse.fill_((1, 2, 3))
    contact.surface.appearance.visual.fill_((.2, .3, .4, .5))
    before = contact.surface.appearance.visual.numpy().copy()
    contact.enabled = hook.enabled = False
    sim.prepare()
    assert not contact.record_count.numpy().any()
    assert not contact.suppress_active.numpy().any()
    assert not contact.impulse.numpy().any() and not hook.impulse.numpy().any()
    np.testing.assert_array_equal(contact.surface.appearance.visual.numpy(), before)
    sim.step()  # enabled tool-force/thermal consumers see cleared producers
    assert not find(sim, ToolForce).force.numpy().any()
