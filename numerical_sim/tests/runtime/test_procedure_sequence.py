import copy
import os

import numpy as np
import pytest
from assessment_host import AssessmentHost
from i4h_isaaclab.composition import build_procedure
from test_procedures import rod_task

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0





def sequence(count=3, hold=0.1, timeout=1, device="cpu", graph=False):
    sim, cfg = rod_task(device, graph)
    step = cfg["procedure"] | {"hold": hold}
    cfg["procedure"] = {
        "steps": [copy.deepcopy(step) | {"id": str(i)} for i in range(count)],
        "timeout": timeout,
        "metrics": {"tip_path_length": {"tool": "wire"}},
    }
    task = sim.procedure = build_procedure(cfg, sim)
    return sim, task, cfg


@pytest.mark.parametrize(
    "device,graph", [("cpu", False), ("cuda:0", False), ("cuda:0", True)]
)
def test_overlapping_holds_action_and_partial_reset(device, graph):
    sim, t, _ = sequence(hold=1 / 60, device=device, graph=graph)
    sim.prepare()
    t.begin_action()
    for i in range(3):
        sim.step()
        assert (t.history.numpy()[0, :, 1] >= 0).sum() == i + 1
        assert t.completion_pulses.numpy()[0].sum() == 1
        assert t.action_completions.numpy()[0].sum() == i + 1
        if i < 2:
            assert t.active_hold.numpy()[0] == 0
    assert t.results()[0]["result"] == "success"
    before = [a.numpy().copy() for a in t._state()]
    t.results()
    t.measurements()
    t.step_history()
    for a, b in zip(t._state(), before):
        np.testing.assert_array_equal(a.numpy(), b)
    sim.reset([0])
    for a, b in zip(t._state(), before):
        np.testing.assert_array_equal(a.numpy()[1], b[1])
    assert t.history.numpy()[0, 0, 0] == 0 and np.all(t.history.numpy()[0, 1:] == -1)
    assert not t.action_completions.numpy()[0].any()
    t.begin_action()
    assert not t.action_completions.numpy().any()


def test_interruption_and_permanence():
    sim, t, _ = sequence(count=2, hold=0.2)
    t.update(0.1)
    q = sim.scene.field("q", sim.scene.body("wire"))
    rest = q.numpy().copy()
    away = rest.copy()
    away[:, -1, 0] += 1
    q.assign(away)
    t.update(0.1)
    assert not t.active_hold.numpy().any()
    q.assign(rest)
    t.update(0.2)
    assert np.all(t.history.numpy()[:, 0, 1] >= 0)
    q.assign(away)
    t.update(0.2)
    assert np.all(t.history.numpy()[:, 0, 1] >= 0) and not t.active_hold.numpy().any()


@pytest.mark.parametrize("restore", [False, True])
def test_future_liveness_and_activation(restore):
    sim, _, cfg = sequence(count=2)
    cfg["procedure"]["steps"][1]["target"] = {"body": "wire", "index": 6}
    flags = sim.scene.field("flags", sim.scene.body("wire"))
    live = flags.numpy().copy()
    dead = live.copy()
    dead[0, 6] = 0
    flags.assign(dead)
    t = build_procedure(cfg, sim)
    t.update(0.05)
    assert t.result.numpy().tolist() == [0, 0]  # future invalidity has no effect
    if restore:
        flags.assign(live)
    t.update(0.05)
    assert t.result.numpy().tolist() == ([0, 0] if restore else [3, 0])
    assert t.active_hold.numpy().tolist() == [0, 0]
    assert np.all(t.history.numpy()[:, 0, 1] >= 0)


@pytest.mark.parametrize(
    "count,invalid,expected", [(1, False, 1), (2, False, 2), (1, True, 3), (2, True, 3)]
)
def test_deadline_precedence(count, invalid, expected):
    sim, _, cfg = sequence(count=count, hold=0.15, timeout=0.15)
    cfg["procedure"]["steps"][0]["target"] = {"body": "wire", "index": 7}
    t = build_procedure(cfg, sim)
    t.update(0.1)
    if invalid:
        sim.scene.field("flags", sim.scene.body("wire")).zero_()
    t.update(0.1)
    assert np.all(t.result.numpy() == expected)
    np.testing.assert_allclose(t.elapsed.numpy(), 0.15)
    assert np.all(t.completion_pulses.numpy()[:, 0] == (0 if invalid else 1))
    if count == 2:
        assert np.all(t.history.numpy()[:, 1] == -1)  # no successor at timeout


def test_path_partial_interval_freeze_and_reset():
    sim, _, cfg = sequence(count=1, hold=1, timeout=0.15)
    cfg["procedure"]["steps"][0]["target"] = {"position": [100, 0, 0]}
    t = sim.procedure = build_procedure(cfg, sim)
    q = sim.scene.field("q", sim.scene.body("wire"))
    values = q.numpy()
    values[:, -1, 1] += 1
    q.assign(values)
    t.update(0.1)
    values[:, -1, 1] += 2
    q.assign(values)
    t.update(0.1)
    np.testing.assert_allclose(t.path_length.numpy(), 2, atol=1e-6)
    before = [
        a.numpy().copy() for a in (t.rows, t.history, t.path_length, t.previous_tip)
    ]
    values[:, -1, 1] += 10
    q.assign(values)
    t.update(0.1)
    for a, b in zip((t.rows, t.history, t.path_length, t.previous_tip), before):
        np.testing.assert_array_equal(a.numpy(), b)
    sim.reset([0])
    assert t.path_length.numpy().tolist() == [0, 2]
    np.testing.assert_array_equal(t.previous_tip.numpy()[0], q.numpy()[0, -1])


@pytest.mark.parametrize(
    "change",
    ["duplicate", "type", "timeout", "nan", "index", "metric", "padded", "overflow"],
)
def test_sequence_validation(change):
    sim, _, cfg = sequence(count=2)
    spec = cfg["procedure"]
    if change == "duplicate":
        spec["steps"][1]["id"] = "0"
    if change == "type":
        spec["type"] = "reach_target"
    if change == "timeout":
        spec["steps"][1]["timeout"] = 1
    if change == "nan":
        spec["steps"][1]["hold"] = float("nan")
    if change == "index":
        spec["steps"][1]["target"] = {"body": "wire", "index": 8}
    if change == "overflow":
        spec["steps"][1]["radius"] = 1e300
    if change == "metric":
        spec["metrics"]["unknown"] = {}
    if change == "padded":
        spec["steps"][1]["target"] = {"body": "wire", "index": 7}
        sim.scene.body("wire").rest.flags[1, 7] = 0
    with pytest.raises(ValueError):
        build_procedure(cfg, sim)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.skipif(not os.environ.get("I4H_DATA"), reason="external case requires I4H_DATA")
def test_real_grasp_and_mixed_sequence(mixed):
    from i4h_physics import Scene, Stepper
    from i4h_physics.loaders import add_rod, add_tet_grid
    from i4h_physics.systems.mechanics.tet import Tet
    from i4h_surgical.loaders.instrument_usd import add_instrument
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
    from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp
    from test_volumetric_grasp import GRASPER, servo

    scene = Scene("cuda:0", 2, gravity=(0, 0, 0))
    add_tet_grid(
        scene, "block", [6, 6, 4], 0.02, origin=(-0.06, -0.06, 0), fix_top=True
    )
    add_rod(scene, "wire", [1, 0, 0], [1, 0, 0], 3, 0.05, 0.01)
    add_instrument(scene, "grasper", **GRASPER)
    scene.allocate()
    control = InstrumentControl(scene)
    grasp = VolumetricGrasp(scene, instruments=control, body="block", capacity=64)
    sim = AssessmentHost(Stepper.create(
        scene,
        [control, Tet(scene, body="block"), grasp],
        dt=1 / 60,
        substeps=2,
        iterations=4,
    ))
    actions = control.actions.numpy()
    servo(sim, control, actions, int(grasp.jaw_start.numpy()[0]) + 2, np.zeros(3))
    actions[0, 0, 7] = 1
    control.actions.assign(actions)
    sim.step()
    assert grasp.count.numpy()[0, 0] > 0
    g = dict(
        type="grasp_target",
        tool="grasper",
        body="block",
        target={"position": [0, 0, 0]},
        radius=0.2,
        hold=1 / 60,
    )
    reach = dict(
        type="reach_target",
        tool="wire",
        target={"position": [1.1, 0, 0]},
        radius=0.01,
        hold=1 / 60,
    )
    steps = [
        g | {"id": "grasp"},
        (reach if mixed else g) | {"id": "next"},
        g | {"id": "last"},
    ]
    t = sim.procedure = build_procedure({"procedure": {"steps": steps}}, sim)
    for i in range(3):
        sim.step()
        assert (t.history.numpy()[0, :, 1] >= 0).sum() == i + 1
        assert (t.history.numpy()[1, :, 1] >= 0).sum() == 0
    assert t.result.numpy().tolist() == [1, 0]


def test_legacy_preflight_and_first_sequence_invalidation():
    sim, cfg = rod_task()
    flags = sim.scene.field("flags", sim.scene.body("wire"))
    values = flags.numpy().copy()
    values[0, 6] = 0
    flags.assign(values)
    cfg["procedure"]["target"] = {"body": "wire", "index": 6}
    with pytest.raises(ValueError, match="inactive or padded"):
        build_procedure(cfg, sim)
    cfg["procedure"] = {"steps": [cfg["procedure"] | {"id": "first"}]}
    t = build_procedure(cfg, sim)
    assert t.result.numpy().tolist() == [3, 0]
    assert not t.elapsed.numpy().any()


@pytest.mark.parametrize("timeout,horizon,source", [(.2, .5, "procedure"), (.5, .2, "host_horizon")])
@pytest.mark.parametrize("complete", [False, True])
def test_effective_deadline_and_completion_precedence(timeout, horizon, source, complete):
    sim, _, cfg = sequence(count=1, hold=.2, timeout=timeout)
    if not complete:
        cfg["procedure"]["steps"][0]["target"] = {"position": [100, 0, 0]}
    t = build_procedure(cfg, sim, horizon_s=horizon)
    t.update(.1)
    t.update(.1)
    assert t.result.numpy().tolist() == ([1, 1] if complete else [2, 2])
    assert all(row["timeout_source"] == source and row["timeout_s"] == .2 for row in t.results())


def test_empty_worker_reset_after_previous_masked_reset():
    from types import SimpleNamespace

    from i4h_standalone.app.worker import PhysicsWorker

    sim, cfg = rod_task()
    sim = sim.simulation
    task = build_procedure(cfg, sim)
    driver_resets = []
    worker = PhysicsWorker(sim, [SimpleNamespace(reset=lambda ids: driver_resets.append(ids))],
                           SimpleNamespace(), procedure=task)
    worker._do_reset([0])
    task.update(sim.dt)
    worker.t = .25
    before = [a.numpy().copy() for a in task._state()]
    q = sim.scene.q.numpy().copy()
    assert worker._do_reset([]) is False
    assert worker.t == .25 and driver_resets == [[0]]
    np.testing.assert_array_equal(sim.scene.q.numpy(), q)
    for a, old in zip(task._state(), before):
        np.testing.assert_array_equal(a.numpy(), old)
