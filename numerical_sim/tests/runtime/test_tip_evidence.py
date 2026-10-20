# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physics-rate evidence versus the same live executor, including within-action transitions."""

import numpy as np
import pytest
import warp as wp
from i4h_endoluminal.assessment import single_target_assessor
from i4h_endoluminal.bindings import CatheterBinding
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_physics.loaders import add_rod
from i4h_physics.scene import Scene
from i4h_physics.stepper import Stepper
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.rod_direct import RodDirect


def small_rod(device="cuda:0", envs=2, young=1e3):
    scene = Scene(device, envs, (0.0, 0.0, -9.81))
    add_rod(scene, "rod", [0, 0, 0.2], [1, 0, 0], 9, 0.02, 0.004, mass=0.01)
    scene.allocate()
    rod = RodDirect(scene, body="rod", young=young)
    control = CatheterControl(scene, rod=rod, track_length=0.0)
    return Stepper.create(scene, [Particles(scene, max_velocity=5.0), rod, control],
                          dt=1 / 60, substeps=4, iterations=8), rod


@pytest.mark.parametrize("device,captured", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
@pytest.mark.parametrize("points,flags,hold,timeout,result,elapsed", [
    ([0, 1, 0, 0], [1, 1, 1, 1], 0.5, 2.0, 1, 1.0),  # exit and re-entry interrupt hold
    ([0, 0, 0, 0], [1, 0, 1, 1], 0.5, 2.0, 1, 1.0),  # inactive tip interrupts hold
    ([0, 0, 1, 1], [1, 1, 1, 1], 0.5, 2.0, 1, 0.5),  # terminal remains latched
    ([0, 0, 0, 0], [1, 1, 1, 1], 0.5, 0.5, 1, 0.5),  # completion exactly at deadline
    ([0, 0, 0, 0], [1, 1, 1, 1], 0.5, 0.4, 2, 0.4),  # capped elapsed cannot complete
])
def test_replay_matches_live(device, captured, points, flags, hold, timeout, result, elapsed):
    sim, _ = small_rod(device)
    binding = CatheterBinding(sim, "rod")
    evidence = binding.configure_tip_evidence(binding, len(points))
    live, replay = [single_target_assessor(binding, target=[0, 0, 0], radius=0.1,
                                          hold=hold, timeout=timeout) for _ in range(2)]
    publish = None
    if captured:
        with wp.ScopedCapture(device=device) as capture:
            evidence.step_end(sim.scene, 0.25)
        publish = capture.graph
    evidence.begin_action()
    live.begin_action()
    replay.begin_action()
    positions, active = binding.q.numpy(), binding.flags.numpy()
    per_sample = []
    incremental = single_target_assessor(binding, target=[0, 0, 0], radius=0.1, hold=hold, timeout=timeout)
    incremental.begin_action()
    for x, flag in zip(points, flags):
        positions[:, -1] = [x, 0, 0]
        active[:, -1] = flag
        binding.q.assign(positions)
        binding.flags.assign(active)
        if publish is None:
            evidence.step_end(sim.scene, 0.25)
        else:
            wp.capture_launch(publish)
        live.update(0.25)
        incremental.consume(evidence, .25)
        for name in ("reward", "completion_pulses", "action_reward", "action_completions", "history", "elapsed"):
            np.testing.assert_array_equal(getattr(live, name).numpy(), getattr(incremental, name).numpy())
        per_sample.append(live.reward.numpy().copy())
    np.testing.assert_array_equal(np.sum(per_sample, axis=0), live.action_reward.numpy())
    replay.consume(evidence, 0.25, validate=True)
    for name in ("rows", "history", "active_hold", "elapsed", "result", "reward", "completion_pulses", "action_completions", "action_reward"):
        assert np.array_equal(getattr(live, name).numpy(), getattr(replay, name).numpy()), name
    assert np.all(replay.result.numpy() == result)
    assert np.allclose(replay.elapsed.numpy(), elapsed)
    assert np.array_equal(evidence.positions.numpy()[0, :, 0], points)
    assert np.array_equal(evidence.active.numpy()[0], flags)
    assert np.all(evidence.count.numpy() == 4) and not evidence.overflow.numpy().any()
    names = ("rows", "history", "reward", "completion_pulses", "action_reward", "action_completions", "elapsed")
    before = {name: getattr(replay, name).numpy().copy() for name in names}
    replay.consume(evidence, 0.25, validate=True)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(replay, name).numpy(), value)
    # Explicit overflow accounting never overwrites the recorded samples.
    saved = evidence.positions.numpy().copy()
    evidence.step_end(sim.scene, 0.25)
    assert np.all(evidence.count.numpy() == 5) and evidence.overflow.numpy().all()
    assert np.array_equal(evidence.positions.numpy(), saved)
    evidence.begin_action()
    assert not evidence.count.numpy().any() and not evidence.overflow.numpy().any()


@pytest.mark.parametrize("envs", [1, 2])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("decimation", [1, 2, 4, 8])
def test_captured_physics_sample_order(envs, captured, decimation):
    sim, _ = small_rod(envs=envs)
    sim.cuda_graph = captured
    view = CatheterBinding(sim, "rod")
    evidence = view.configure_tip_evidence(view, decimation)
    sim.prepare()
    ptrs = [a.ptr for a in (evidence.positions, evidence.active, evidence.count)]
    for _ in range(2):
        evidence.begin_action()
        view.commands.fill_(0.2)
        expected = []
        for _ in range(decimation):
            sim.step()
            expected.append(view.q.numpy()[:, -1].copy())
        assert np.array_equal(evidence.positions.numpy(), np.stack(expected, axis=1))
        assert np.all(evidence.count.numpy() == decimation)
        assert not evidence.overflow.numpy().any()
    sim.prepare()
    assert view.valid and evidence.valid
    assert ptrs == [a.ptr for a in (evidence.positions, evidence.active, evidence.count)]
    with pytest.raises(ValueError, match="reconstruct"):
        view.configure_tip_evidence(view, decimation + 1)


def test_live_material_write_changes_only_selected_dynamics_after_capture():
    sim, rod = small_rod(young=1e6)
    control, _ = small_rod(young=1e6)
    view = CatheterBinding(sim, "rod")
    sim.prepare()
    control.prepare()
    graph = sim.graph
    young = view.materials["young"].numpy()
    young[0] = 1e2
    view.materials["young"].assign(young)
    for _ in range(300):
        sim.step()
        control.step()
    q, baseline = view.q.numpy(), control.scene.field("q", control.scene.body("rod")).numpy()
    assert np.isfinite(q).all()
    assert baseline[0, -1, 2] - q[0, -1, 2] > 5e-4  # existing cantilever droop tolerance
    assert np.allclose(q[1], baseline[1], atol=1e-6, rtol=0)
    assert np.abs(np.linalg.norm(np.diff(q, axis=1), axis=2) - 0.02).max() < 0.05 * 0.02
    stepped = q[1].copy()
    sim.physics.reset([0])
    assert np.array_equal(view.q.numpy()[1], stepped)
    assert np.array_equal(view.materials["young"].numpy(), young)
    assert sim.graph is graph


@pytest.mark.parametrize("device,captured", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
@pytest.mark.parametrize("fault", ["underfilled", "overflow", "count_only", "overflow_only"])
@pytest.mark.parametrize("completed", [False, True])
def test_validation_is_atomic_and_reset_recovers(device, captured, fault, completed):
    sim, _ = small_rod(device)
    sim.cuda_graph = captured
    binding = CatheterBinding(sim, "rod")
    evidence = binding.configure_tip_evidence(binding, 4)
    hold = 2 * sim.dt if completed else 0.5
    assessor = single_target_assessor(binding, target=[0, 0, 0], radius=100, hold=hold, timeout=2)
    sim.prepare()
    evidence.begin_action()
    assessor.begin_action()
    if completed:
        for _ in range(4):
            sim.step()
        assessor.consume(evidence, sim.dt, validate=True)
        assert np.all(assessor.action_reward.numpy() == 1)
        assert np.all(assessor.consumed.numpy() == 4)
        # Revalidation must also preserve nonzero rewards, terminal history and cursors.
        evidence.begin_action()
    for _ in range(3 if fault == "underfilled" else 5 if fault == "overflow" else 4):
        sim.step()
    # A valid first environment must not be consumed if the second is malformed.
    counts, overflow = evidence.count.numpy(), evidence.overflow.numpy()
    counts[0], overflow[0] = 4, 0
    if fault == "count_only":
        counts[1] = 5
    elif fault == "overflow_only":
        overflow[1] = 1
    evidence.count.assign(counts)
    evidence.overflow.assign(overflow)
    arrays = [a for a in vars(assessor).values() if isinstance(a, wp.array)]
    before = [a.numpy().copy() for a in arrays]
    with pytest.raises(RuntimeError, match=rf"env 1: count={counts[1]}, overflow={overflow[1]}"):
        assessor.consume(evidence, sim.dt, validate=True)
    for a, saved in zip(arrays, before):
        assert np.array_equal(a.numpy(), saved)
    sim.physics.reset()
    assessor.reset()
    evidence.begin_action()
    assessor.begin_action()
    for _ in range(4):
        sim.step()
    assessor.consume(evidence, sim.dt, validate=True)
    assert np.all(assessor.consumed.numpy() == 4)
    assert np.allclose(assessor.elapsed.numpy(), (2 if completed else 4) * sim.dt)
    assert np.all(assessor.result.numpy() == int(completed))
    assert np.all(assessor.action_reward.numpy() == int(completed))


def test_default_consume_does_not_read_device_counters(monkeypatch):
    sim, _ = small_rod("cpu")
    binding = CatheterBinding(sim, "rod")
    evidence = binding.configure_tip_evidence(binding, 1)
    assessor = single_target_assessor(binding, target=[0, 0, 0], radius=100, hold=0.5, timeout=2)
    evidence.step_end(sim.scene, sim.dt)
    def unexpected_read(*args, **kwargs):
        pytest.fail("default consumption must not read device arrays")
    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", unexpected_read)
        assessor.consume(evidence, sim.dt)
    assert np.all(assessor.consumed.numpy() == 1)
