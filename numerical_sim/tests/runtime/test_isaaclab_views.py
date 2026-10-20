# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M7: the Isaac-Lab-free half of the env (torch views, action bounds and mapping, masked reset from a bool
tensor) and a chole stub loop in DirectRLEnv order. Real Isaac Lab coverage lives in test_isaaclab_env.py."""

import numpy as np
import pytest
import warp as wp
from helpers import find, needs_chole
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import load_case, read_case
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

torch = pytest.importorskip("torch")

from i4h_isaaclab.views import (  # noqa: E402
    action_bounds,
    action_views,
    map_actions,
    observation,
    observation_views,
    replay_actions,
    state_views,
    write_actions,
)


def scale(low, high, dev):
    low, high = torch.as_tensor(low, device=dev), torch.as_tensor(high, device=dev)
    return low, high, (low + high) / 2, (high - low) / 2


def test_action_bounds_columns(case):
    cfg = read_case(case)
    cfg["systems"].append({"type": "catheter_control", "rod": "rod_pbd", "track_length": 0.16})
    sim = load_case(cfg)
    scene, control = sim.scene, find(sim, InstrumentControl)
    dev = wp.device_to_torch(scene.device)
    views = action_views(sim)
    assert [tuple(v.shape) for v in views] == [(2, 10 * control.count), (2, 3)]
    assert views[0].data_ptr() == control.actions.ptr and views[1].data_ptr() == find(sim, CatheterControl).actions.ptr
    low, high = action_bounds(sim, 0.25)
    assert low.shape == high.shape == (10 * control.count + 3,) and low.dtype == np.float32
    rotations = [r.initial_rotation for r in scene.instruments.values()]
    for k in range(control.count):
        assert np.allclose(low[10 * k : 10 * k + 3], control.rest[k] - 0.25) and np.allclose(high[10 * k : 10 * k + 3], control.rest[k] + 0.25)
        assert np.allclose(low[10 * k + 3 : 10 * k + 7], rotations[k] - 1.0) and np.allclose(high[10 * k + 3 : 10 * k + 7], rotations[k] + 1.0)
        assert (low[10 * k + 7 : 10 * k + 10] == -1.0).all() and (high[10 * k + 7 : 10 * k + 10] == 1.0).all()
    assert (low[-3:] == -1.0).all() and (high[-3:] == 1.0).all()
    physical = torch.cat(views, dim=1)  # the reset pose the control systems wrote
    low_t, high_t, mid, half = scale(low, high, dev)
    assert (physical >= low_t).all() and (physical <= high_t).all()
    out = torch.empty_like(physical)
    for value, expected in ((-1.0, low_t), (0.0, mid), (1.0, high_t), (-3.0, low_t), (3.0, high_t)):
        map_actions(torch.full_like(physical, value), mid, half, out)
        assert torch.allclose(out, expected.expand_as(out))
    block, rod = scene.body("block").count, scene.body("rod").count  # 5^3 vertices of the 4^3-cell capacity, 9 nodes
    assert [tuple(v.shape) for v in observation_views(sim)] == [(2, 3 * block), (2, 3 * rod)]  # water is a count
    assert [int(v.shape[1]) for v in state_views(sim)] == [1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3]  # surface x2, tool force + count + jaw load, cutting cut_on, stretch max_stretch, fluid, suction, smoke, catheter targets
    assert observation(observation_views(sim), state_views(sim)).shape == (2, 3 * block + 3 * rod + 15)  # + tool_force force (3), count and jaw_load, cutting cut_on, stretch max_stretch


def test_zero_action_is_the_reset_pose(case):
    """A tool with a non-identity initial rotation keeps it under the zero action (the quaternion columns are centred
    on the initial rotation, not on zero, which the kernel would turn into identity)."""
    cfg = read_case(case)
    cfg["instruments"]["probe"]["initial"]["rotation"] = [0.7071068, 0.0, 0.0, 0.7071068]
    sim = load_case(cfg)
    control = find(sim, InstrumentControl)
    dev = wp.device_to_torch(sim.scene.device)
    views = action_views(sim)
    low, high, mid, half = scale(*action_bounds(sim, 0.5), dev)
    physical = torch.empty_like(views[0])
    map_actions(torch.zeros_like(physical), mid, half, physical)
    assert torch.allclose(physical[:, 3:7], torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=dev).expand(2, 4))
    assert torch.allclose(physical, torch.cat(views, 1), atol=1e-6), "the zero action is what reset wrote"
    sim.prepare()
    torch.cuda.synchronize()
    before = (control.root.numpy().copy(), control.shaft_q.numpy().copy(), control.jaw_q.numpy().copy())
    for _ in range(5):
        write_actions(views, physical)
        sim.step()
    torch.cuda.synchronize()
    after = (control.root.numpy(), control.shaft_q.numpy(), control.jaw_q.numpy())
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(before, after)), "the tool held its initial pose and rotation"


def test_replay_actions(case, tmp_path):
    """A recorded session (one row per physics step per device) becomes normalised actions, every ``decimation``-th
    row, padded with the last row; a device without a file holds the reset pose."""
    sim = load_case(case)
    control = find(sim, InstrumentControl)
    low, high = action_bounds(sim, 0.5)
    rows = np.zeros((7, 10), np.float32)
    rows[:, :3] = control.rest[0] + np.linspace(0.0, 0.25, 7)[:, None] * np.float32([1.0, 0.0, 0.0])  # x sweeps half the workspace
    rows[:, 6] = 1.0  # identity quaternion
    rows[:, 7] = np.linspace(0.0, 1.0, 7)  # grip closes
    rows[-1, 8] = 1.0  # energy on the last frame
    np.save(tmp_path / "session_probe.npy", rows)  # the probe has no input binding, so its device id is its name
    actions, excess = replay_actions(sim, tmp_path / "session", low, high, decimation=2)
    assert actions.shape == (4, 10) and actions.dtype == np.float32 and excess == 0.0  # rows 0, 2, 4, 6
    mid, half = (low + high) / 2, (high - low) / 2
    physical = actions * half + mid
    assert np.allclose(physical[:, 0], control.rest[0][0] + np.array([0.0, 0.25 / 3, 0.5 / 3, 0.25]), atol=1e-5)
    assert np.allclose(physical[:, 7], [0.0, 1.0 / 3, 2.0 / 3, 1.0], atol=1e-5) and physical[-1, 8] == 1.0
    assert np.allclose(actions[:, 3:7], [0.0, 0.0, 0.0, 0.0])  # the identity quaternion is the reset rotation: zero action
    far = rows.copy()
    far[:, 0] += 2.0  # outside the workspace: clamped and reported
    np.save(tmp_path / "far_probe.npy", far)
    clipped, excess = replay_actions(sim, tmp_path / "far", low, high, decimation=1)
    assert clipped.shape == (7, 10) and (clipped[:, 0] == 1.0).all() and excess > 1.0
    missing, _ = replay_actions(sim, tmp_path / "nothing", low, high)
    assert missing.shape == (1, 10) and (missing == 0.0).all(), "no recording: one step of the reset pose"


def test_bool_env_ids_rejected(case):
    sim = load_case(case)
    dev = wp.device_to_torch(sim.scene.device)
    with pytest.raises(TypeError):
        sim.reset(torch.tensor([False, True], device=dev))
    with pytest.raises(ValueError):
        sim.reset([0], mask=torch.tensor([False, True], device=dev))


@needs_chole
def test_isaaclab_stub_loop_chole():
    sim = load_case("chole_tet_med", envs=2)
    scene, control = sim.scene, find(sim, InstrumentControl)
    target = scene.body('tissue')
    E, dev = scene.envs, wp.device_to_torch(scene.device)
    acts, obs, state = action_views(sim), observation_views(sim), state_views(sim)
    assert len(acts) == 1 and acts[0].shape == (E, 10 * control.count) and acts[0].data_ptr() == control.actions.ptr  # 10 columns per tool
    width = sum(int(v.shape[1]) for v in state)
    assert width == 71 and [tuple(v.shape) for v in obs] == [(E, 3 * target.count)]  # M9b: + cut_on (5 tools), volumetric count (5), max_stretch; M11: + tool force (15), count (5), jaw load (5)
    full = observation(obs, state)
    assert full.shape == (E, 3 * target.count + width) and full.dtype == torch.float32
    part = observation_views(sim, "tissue.gallbladder")
    pc = target.parts["gallbladder"].particle_count
    assert part[0].shape == (E, 3 * pc) and part[0].data_ptr() == scene.part_view(scene.q, target, "gallbladder").ptr
    assert observation(part, state).shape == (E, 3 * pc + width)

    low, high, mid, half = scale(*action_bounds(sim, 0.5), dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    physical = torch.empty((E, 10 * control.count), device=dev)
    sim.prepare()
    for _ in range(10):  # DirectRLEnv.step order: map and apply the action, physics step, per decimation substep
        normalised = torch.rand((E, 10 * control.count), generator=gen, device=dev) * 2.0 - 1.0
        for _ in range(2):
            map_actions(normalised, mid, half, physical)
            write_actions(acts, physical)
            sim.step()
        assert torch.isfinite(observation(obs, state)).all()
    torch.cuda.synchronize()
    root = control.root.numpy()
    for k in range(control.count):
        assert (np.abs(root[:, k] - control.rest[k]) <= 0.5 + 1e-5).all(), "the root stays inside the workspace"

    def check_reset(before_env0):
        torch.cuda.synchronize()
        assert torch.equal(obs[0][0], before_env0), "env 0 untouched bitwise"
        assert torch.equal(obs[0][1], wp.to_torch(scene.view(scene.q0, target)).view(E, -1)[1]), "env 1 back at rest"
        assert all((v[1] == 0).all() for v in state), "env 1 state cleared"
        row = acts[0][1].view(control.count, 10)
        assert torch.allclose(row[:, :3], torch.as_tensor(control.rest, device=dev)) and (row[:, 7:] == 0).all()
        assert torch.equal(acts[0][0], physical[0]), "env 0 keeps the last action"

    before = obs[0][0].clone()
    sim.reset(mask=torch.tensor([False, True], device=dev))
    check_reset(before)
    for _ in range(6):
        for _ in range(2):
            write_actions(acts, physical)
            sim.step()
    before = obs[0][0].clone()
    sim.reset(torch.tensor([1], dtype=torch.int64, device=dev))  # Isaac Lab's env_ids dtype
    check_reset(before)
