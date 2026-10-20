import os

import numpy as np
import pytest
import warp as wp
import yaml
from assessment_host import AssessmentHost
from i4h_isaaclab.composition import build_procedure, load_case, read_case

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0




def rod_task(device="cpu", graph=False):
    cfg = {
        "sim": {"device": device, "envs": 2, "gravity": [0, 0, 0], "cuda_graph": graph},
        "assets": {
            "wire": {
                "type": "rod",
                "count": 8,
                "segment_length": 0.05,
                "start": [0, 0, 0],
                "direction": [1, 0, 0],
                "radius": 0.01,
                "mass": 1,
            }
        },
        "systems": [
            {"type": "particles"},
            {"type": "rod_direct", "id": "rod", "body": "wire"},
            {"type": "catheter_control", "rod": "rod", "track_length": 1},
        ],
        "procedure": {"type": "reach_target", "tool": "wire", "target": {"position": [0.35, 0, 0]}, "radius": 0.01},
    }
    sim = AssessmentHost(load_case(cfg))
    sim.procedure = build_procedure(cfg, sim)
    return sim, cfg


@pytest.mark.parametrize(
    "task,base", [("aorta_static_reach", "aorta_static"), ("aorta_shell_reach", "aorta_shell"), ("chole_grasp", "chole_tet_med")]
)
def test_reference_preserves_physics(task, base):
    cfg = read_case(task)
    procedure = cfg.pop("procedure")
    assert cfg == read_case(base)
    assert procedure["tool"] in (cfg.get("assets", {}) | cfg.get("instruments", {}))


def test_case_references(tmp_path, monkeypatch):
    folder = tmp_path / "tasks"
    folder.mkdir()
    base = {"sim": {"device": "cpu"}, "assets": {}, "systems": []}
    (folder / "physics.yaml").write_text(yaml.safe_dump(base))
    procedure = {"type": "reach_target", "tool": "wire", "target": {"position": [0, 0, 0]}, "radius": 0.1}
    wrapper = {"case": "physics.yaml", "procedure": procedure}
    path = folder / "task.yaml"
    path.write_text(yaml.safe_dump(wrapper))
    # Resolve beside the wrapper even when the current directory contains a same-named case.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "physics.yaml").write_text("sim: {envs: 99}\n")
    assert read_case(path) == base | {"procedure": procedure}
    assert read_case({"case": str(folder / "physics.yaml"), "procedure": procedure}) == read_case(path)
    assert read_case(base) == base
    assert "procedure" not in base
    assert not hasattr(load_case(path), "procedure")  # loading remains physics-only
    for extra in ("sim", "assets", "systems", "scene", "render", "typo"):
        with pytest.raises(ValueError, match="permits only"):
            read_case(wrapper | {extra: {}})
    for ref in (None, 42, {}, ""):
        with pytest.raises(ValueError, match="nonempty"):
            read_case({"case": ref})
    with pytest.raises(FileNotFoundError):
        read_case({"case": "missing-case"})
    for nested in ({"case": "physics.yaml"}, {"case": "task.yaml"}, base | {"procedure": procedure}):
        (folder / "physics.yaml").write_text(yaml.safe_dump(nested))
        with pytest.raises(ValueError, match="directly to a physics case"):
            read_case(path)
    with pytest.raises(ValueError, match="unknown top-level"):
        read_case(base | {"scene": {"template": "endoluminal"}})


def test_procedure_validation():
    sim, cfg = rod_task()
    cfg["procedure"]["typo"] = 1
    with pytest.raises(ValueError, match="procedure: unsupported fields"):
        build_procedure(cfg, sim)
    del cfg["procedure"]["typo"]
    cfg["procedure"]["target"]["typo"] = 1
    with pytest.raises(ValueError, match="procedure.target: unsupported fields"):
        build_procedure(cfg, sim)


@pytest.mark.parametrize("device,graph", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_hold_reward_reset_capture(device, graph):
    sim, _ = rod_task(device, graph)
    task = sim.procedure
    before = task.rows.numpy().copy()
    sim.prepare()
    np.testing.assert_array_equal(task.rows.numpy(), before)
    for _ in range(29):
        sim.step()
    assert np.all(task.rows.numpy()[:, 11] == 0)
    sim.step()
    assert np.all(task.rows.numpy()[:, 11] == 1)
    assert np.all(task.reward.numpy() == 1)
    frozen = task.rows.numpy().copy()
    sim.step()
    np.testing.assert_array_equal(task.rows.numpy(), frozen)
    assert np.all(task.reward.numpy() == 0) and np.all(task.action_reward.numpy() == 1)
    sim.reset([0])
    np.testing.assert_array_equal(task.rows.numpy()[1], frozen[1])
    assert task.rows.numpy()[0, 9] == 0
    task.begin_action()
    assert not task.action_reward.numpy().any()


def test_exit_attached_invalid_timeout():
    sim, cfg = rod_task()
    task = sim.procedure
    for _ in range(10):
        sim.step()
    actions = sim.systems[-1].actions.numpy()
    actions[0, 0] = 1
    sim.systems[-1].actions.assign(actions)
    for _ in range(10):
        sim.step()
    assert task.rows.numpy()[0, 10] == 0 and task.rows.numpy()[1, 10] > 0
    cfg["procedure"]["target"] = {"body": "wire", "index": 7}
    task = build_procedure(cfg, sim)
    sim.procedure = task
    sim.step()
    np.testing.assert_allclose(task.rows.numpy()[:, :3], sim.scene.field("q", sim.scene.body("wire")).numpy()[:, -1])
    flags = sim.scene.field("flags", sim.scene.body("wire")).numpy()
    flags[0, 7] = 0
    sim.scene.field("flags", sim.scene.body("wire")).assign(flags)
    sim.step()
    assert task.rows.numpy()[0, 11] == 3
    cfg["procedure"]["target"]["index"] = 8
    with pytest.raises(ValueError, match="index"):
        build_procedure(cfg, sim)
    sim, cfg = rod_task()
    cfg["procedure"].update(target={"position": [10, 0, 0]}, timeout=0.1)
    sim.procedure = build_procedure(cfg, sim)
    for _ in range(6):
        sim.step()
    assert np.all(sim.procedure.rows.numpy()[:, 11] == 2)


def test_threaded_snapshot_matches_assessment():
    from i4h_standalone.app.worker import PhysicsWorker
    from i4h_standalone.render.snapshots import Snapshots

    sim, _ = rod_task("cuda:0", True)
    sim.prepare()
    ring = Snapshots(sim, procedure=sim.procedure)
    worker = PhysicsWorker(sim.simulation, [], ring, steps=35, procedure=sim.procedure)
    worker.start()
    worker.join(10)
    assert not worker.is_alive()
    worker.check()
    wp.synchronize_device(sim.scene.device)
    snap = ring.acquire()
    np.testing.assert_array_equal(snap.task_host, sim.procedure.rows.numpy())
    np.testing.assert_array_equal(snap.q.numpy(), sim.scene.q.numpy())
    assert snap.t == pytest.approx(35 * sim.dt)
    assert np.all(snap.task_host[:, 11] == 1)


@pytest.mark.parametrize("case", ["aorta_static", "aorta_shell"])
@pytest.mark.skipif(not os.environ.get("I4H_DATA"), reason="external case requires I4H_DATA")
def test_reference_physics_matches_original(case):
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl

    states = []
    for name in (case, case + "_reach"):
        sim = load_case(name, envs=2)
        assert not hasattr(sim, "procedure")
        control = next(s for s in sim.systems if isinstance(s, CatheterControl))
        control.actions.assign(np.float32([[0.5, 0.1, 0.2], [0.1, 0.2, -0.1]]))
        for _ in range(8):
            sim.step()
        states.append(sim.scene.q.numpy())
    np.testing.assert_allclose(*states, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not os.environ.get("I4H_DATA"), reason="external case requires I4H_DATA")
def test_real_grasp_proximity_release_and_stale_slots():
    from test_volumetric_grasp import block_and_grasper, servo

    sim, scene, body, control, grasp = block_and_grasper()
    sim = AssessmentHost(sim)
    spec = {
        "procedure": {
            "type": "grasp_target",
            "tool": "grasper",
            "body": "block",
            "target": {"position": [0, 0, 0]},
            "radius": 0.2,
        }
    }
    task = sim.procedure = build_procedure(spec, sim)
    actions = control.actions.numpy()
    servo(sim, control, actions, int(grasp.jaw_start.numpy()[0]) + 2, np.zeros(3))
    assert not task.rows.numpy()[:, 6].any()  # proximity with open jaws
    actions[0, 0, 7] = 1
    control.actions.assign(actions)
    sim.step()
    assert task.rows.numpy()[:, 6].tolist() == [1, 0]
    # A live but wrong-region attachment must not qualify.
    task.start, task.end = body.count, body.count
    task.update(sim.dt)
    assert task.rows.numpy()[0, 6] == 0
    task.start, task.end = 0, body.count
    task.update(sim.dt)
    assert task.rows.numpy()[0, 6] == 1
    # Disabling the mechanism leaves slots allocated but must not keep qualifying.
    grasp.enabled = False
    task.update(sim.dt)
    assert task.rows.numpy()[0, 6] == 0
    grasp.enabled = True
    # Stale slots cannot qualify dead particles or a released jaw.
    flags = scene.field("flags", body).numpy()
    scene.field("flags", body).zero_()
    task.update(sim.dt)
    assert task.rows.numpy()[0, 6] == 0
    scene.field("flags", body).assign(flags)
    control.held.zero_()
    task.update(sim.dt)
    assert task.rows.numpy()[0, 6] == 0
    actions[0, 0, 7] = 0
    control.actions.assign(actions)
    sim.step()
    assert not task.rows.numpy()[:, 6].any()


def test_padded_target_rejected():
    sim, cfg = rod_task()
    sim.scene.body("wire").rest.flags[1, 7] = 0
    cfg["procedure"]["target"] = {"body": "wire", "index": 7}
    with pytest.raises(ValueError, match="inactive or padded"):
        build_procedure(cfg, sim)


def test_surface_barycentric_wrong_tool_part_and_dead_tet():
    """Slot-level fixture: centroid qualifies though all vertices are outside the radius."""
    from i4h_surgical.assessment import surface

    dev = "cuda:0"
    q = wp.array(np.float32([[[0, 0, 0], [3, 0, 0], [0, 3, 0]]]), dtype=wp.vec3, device=dev)
    flags = wp.ones((1, 3), dtype=int, device=dev)
    active = wp.array(np.int32([[[0], [1]]]), dtype=int, device=dev)
    verts = wp.array(np.int32([[[[0, 1, 2]], [[0, 1, 2]]]]), dtype=wp.vec3i, device=dev)
    bary = wp.full((1, 2, 1), wp.vec3(1 / 3), dtype=wp.vec3, device=dev)
    tet = wp.zeros((1, 2, 1), dtype=int, device=dev)
    live = wp.ones((1, 1), dtype=int, device=dev)
    held = wp.ones((1, 2), dtype=int, device=dev)
    rows = wp.zeros((1, 12), dtype=float, device=dev)

    def evaluate(k=1, start=0, end=3):
        r = np.zeros((1, 12), np.float32)
        r[0, :3] = [1, 1, 0]
        rows.assign(r)
        wp.launch(
            surface,
            1,
            inputs=[q, flags, active, verts, bary, tet, live, held, k, start, end, 0.01],
            outputs=[rows],
            device=dev,
        )
        return rows.numpy()[0, 6]

    assert evaluate() == 1
    assert evaluate(k=0) == 0  # attachment belongs to the other tool
    assert evaluate(start=1) == 0  # triangle spans outside the selected part
    live.zero_()
    assert evaluate() == 0
    live.fill_(1)
    held.zero_()
    assert evaluate() == 0


def test_cli_reference_and_disable_procedure(tmp_path, capsys):
    from i4h_standalone.app import cli

    _, cfg = rod_task()
    procedure = cfg.pop("procedure")
    (tmp_path / "physics.yaml").write_text(yaml.safe_dump(cfg))
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump({"case": "physics.yaml", "procedure": procedure}))
    args = [str(path), "--device", "cpu", "--steps", "1", "--input", "noop", "--render", "noop"]
    assert cli.main(args) == 0
    assert '"result": "running"' in capsys.readouterr().out
    assert cli.main(args + ["--no-procedure"]) == 0
    assert '"result"' not in capsys.readouterr().out
