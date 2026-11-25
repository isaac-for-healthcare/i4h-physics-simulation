# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLV-010-29: registered production tasks, without case or layout substitutions.

Run with the isolated installed-wheel environment via scripts/qualify_direct_tasks.py.
The normal CPU suite skips this module when Isaac Lab is unavailable.
"""

import json
import os
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
import warp as wp

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")
gym = pytest.importorskip("gymnasium")

import i4h_isaaclab.tasks  # noqa: E402,F401
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

TASKS = {
    "I4h-Chole-Direct": "chole_tet_med",
    "I4h-Aorta-Direct": "aorta_static",
    "I4h-ReachTarget-Direct": "aorta_static_reach",
}


def make(task, count, captured):
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    assert cfg.case == TASKS[task]
    cfg.scene.num_envs = count
    cfg.sim.physics.use_cuda_graph = captured
    cfg.stagger_episodes = False
    return gym.make(task, cfg=cfg)


def close(env):
    writer = getattr(env.unwrapped, "_acceptance_video", None)
    if writer is not None:
        writer.stdin.close()
        assert writer.wait() == 0, "video encoder failed"
    env.close()
    SimulationContext.clear_instance()


def evidence(task, count, captured, kind, result):
    folder = os.environ.get("I4H_DIRECT_RESULTS")
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{task}-{count}-{captured}-{kind}.json").write_text(
            json.dumps(result, indent=2) + "\n")


def finite(u, obs, reward=None):
    assert set(obs.keys()) == set(u.single_observation_space.spaces)
    for key, value in obs.items():
        assert tuple(value.shape) == tuple(u.observation_space[key].shape)
        assert torch.isfinite(value).all(), key
    if reward is not None:
        assert reward.shape == (u.num_envs,) and torch.isfinite(reward).all()
    scene = u.simulator.scene
    for body in (*scene.bodies, *scene.pools):
        for name in ("q", "q_prev", "qd"):
            assert np.isfinite(scene.field(name, body).numpy()).all(), (body.name, name)
    for system in u.simulator.systems:
        for name in ("frame", "frame_pred", "root", "targets", "actions"):
            array = getattr(system, name, None)
            if isinstance(array, wp.array):
                assert np.isfinite(array.numpy()).all(), (type(system).__name__, name)


def step(env, action):
    u = env.unwrapped
    assert torch.isfinite(action).all()
    assert (action.abs() <= 1).all(), "script must respect the registered normalized action space"
    result = env.step(action)
    finite(u, result[0], result[1])
    camera = getattr(u, "_acceptance_camera", None)
    if camera is not None:
        camera.render()
        frame = np.ascontiguousarray(camera.rgb.numpy()[0, :, :, :3])
        assert frame.max() > frame.min(), "visual demonstration must contain visible geometry"
        u._acceptance_video.stdin.write(frame.tobytes())
    return result


def start_video(u):
    folder = os.environ.get("I4H_DIRECT_VIDEO")
    if folder:
        from i4h_isaaclab.camera.tiled import TiledCamera
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        render = u.resolved_case.get("render", {})
        camera = TiledCamera(u.simulator, 640, 480, camera=render.get("camera"))
        camera.prepare()
        u._acceptance_camera = camera
        u._acceptance_video = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "640x480", "-framerate", "30", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path / "demonstration.mp4")], stdin=subprocess.PIPE)


def physical_action(u, rows):
    values = torch.as_tensor(rows.reshape(u.num_envs, -1), device=u.device)
    return ((values - u.action_mid) / u.action_half).clamp(-1, 1)


def find(u, name):
    return next(s for s in u.simulator.systems if type(s).__name__ == name)


def chole_contact(env):
    """Use the existing sphere servo (scissors/procedure tests) with the grasper, jaws open."""
    u = env.unwrapped
    control, contact = find(u, "InstrumentControl"), find(u, "InstrumentContact")
    k = control.names.index("grasper")
    sphere = int(control.host_sphere_start[k]) + 2
    rows = control.actions.numpy().copy()
    rows[:, :, :3] = control.rest[None]
    rows[:, :, 3:] = 0
    rows[:, :, 6] = 1
    for _ in range(15):
        step(env, physical_action(u, rows))
    initial = u.obs_views[0].clone()
    points = initial.reshape(u.num_envs, -1, 3).cpu().numpy()
    jaw = control.sphere_q.numpy()[:, sphere]
    targets = np.stack([p[np.argmin(np.linalg.norm(p - j, axis=1))] for p, j in zip(points, jaw)])
    impulse = np.zeros(u.num_envs)
    pairs = np.zeros(u.num_envs, dtype=np.int64)
    for _ in range(45):
        rows[:, k, :3] = control.root.numpy()[:, k] + targets - control.sphere_q.numpy()[:, sphere]
        step(env, physical_action(u, rows))
        impulse += np.linalg.norm(contact.impulse.numpy()[:, k], axis=1)
        pairs += contact.impulse_count.numpy()[:, k]
    displacement = (u.obs_views[0] - initial).abs().amax(dim=1).cpu().numpy()
    assert (pairs > 0).all() and (impulse > 1e-8).all(), (pairs, impulse)
    assert (displacement > 1e-5).all(), displacement
    before = control.root.numpy()[:, k].copy()
    rows[:, :, :3] = control.rest[None]
    for _ in range(30):
        step(env, physical_action(u, rows))
    after = control.root.numpy()[:, k]
    assert (np.linalg.norm(after - before, axis=1) > 1e-4).all(), "withdrawal must move the tool"
    assert (np.linalg.norm(after - control.rest[k], axis=1) <
            np.linalg.norm(before - control.rest[k], axis=1)).all()
    return {"contact_pairs": pairs.tolist(), "contact_impulse": impulse.tolist(),
            "tissue_displacement": displacement.tolist()}


def aorta_controls(env):
    u = env.unwrapped
    control = find(u, "CatheterControl")
    measurements = []
    for axis in range(3):
        targets = control.targets.numpy().copy()
        scene = u.simulator.scene
        q = scene.view(scene.q, control.body).numpy().copy()
        frames = control.rod.frame.numpy().copy().reshape(u.num_envs, -1, 4)
        bend = control.rod.bend_offset.numpy().copy().reshape(u.num_envs, -1, 3)
        action = torch.zeros((u.num_envs, 3), device=u.device)
        action[:, axis] = .5
        for _ in range(20):
            step(env, action)
        delta = control.targets.numpy() - targets
        assert (delta[:, axis] > 1e-5).all(), (axis, delta)
        moved = np.max(np.abs(scene.view(scene.q, control.body).numpy() - q), axis=(1, 2))
        rotated = np.max(np.abs(control.rod.frame.numpy().reshape(u.num_envs, -1, 4) - frames), axis=(1, 2))
        bent = np.max(np.abs(control.rod.bend_offset.numpy().reshape(u.num_envs, -1, 3) - bend), axis=(1, 2))
        assert (moved > 1e-6).all(), (axis, moved)
        if axis == 1:
            assert (rotated > 1e-6).all(), rotated
        if axis == 2:
            assert (bent > 1e-6).all(), bent
        measurements.append({"axis": axis, "target_delta": delta.tolist(),
                             "position_delta": moved.tolist(), "frame_delta": rotated.tolist(),
                             "bend_delta": bent.tolist()})
    return {"controls": measurements}


def selected_reset(env):
    u = env.unwrapped
    action = torch.zeros((u.num_envs, u.single_action_space.shape[0]), device=u.device)
    action[:, 0] = torch.linspace(.1, .5, u.num_envs, device=u.device)
    for _ in range(5):
        step(env, action)
    scene = u.simulator.scene
    bodies = (*scene.bodies, *scene.pools)
    before = {(name, body.name): scene.field(name, body).numpy().copy()
              for name in ("q", "q_prev", "qd", "flags") for body in bodies}
    products = [(v, v.clone()) for v in u.state_views]
    procedure_rows = u.task_rows.clone() if u.procedure is not None else None
    commands = [v.clone() for v in u.action_views]
    control_arrays = [(s, name, value.numpy().copy()) for s in u.simulator.systems if s.is_control
                      for name, value in vars(s).items() if isinstance(value, wp.array)
                      and value.ndim >= 1 and value.shape[0] == u.num_envs]
    rod_arrays = [(s, name, getattr(s, name).numpy().reshape(u.num_envs, -1, 4).copy())
                  for s in u.simulator.systems for name in ("frame", "frame_pred")
                  if isinstance(getattr(s, name, None), wp.array)]
    u._reset_idx(torch.tensor([0], device=u.device))
    u.sync_views()
    for (name, body_name), saved in before.items():
        now = scene.field(name, scene.body(body_name)).numpy()
        np.testing.assert_array_equal(now[1:], saved[1:])
    for body in bodies:
        if body.integrate or body.pool:
            np.testing.assert_array_equal(scene.field("q", body).numpy()[0],
                                          scene.field("q0", body).numpy()[0])
    for view, saved in products:
        assert torch.equal(view[1:], saved[1:])
    if procedure_rows is not None:
        assert torch.equal(u.task_rows[1:], procedure_rows[1:])
    for view, saved in zip(u.action_views, commands):
        assert torch.equal(view[1:], saved[1:])
    for system, name, saved in control_arrays:
        np.testing.assert_array_equal(getattr(system, name).numpy()[1:], saved[1:])
    for system, name, saved in rod_arrays:
        np.testing.assert_array_equal(getattr(system, name).numpy().reshape(u.num_envs, -1, 4)[1:], saved[1:])
    assert u.episode_length_buf[0] == 0


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "graph"])
def test_direct_lifecycle(task, count, captured):
    env = make(task, count, captured)
    report = {"task": task, "case": TASKS[task], "num_envs": count, "captured": captured}
    try:
        u = env.unwrapped
        obs, _ = env.reset()
        finite(u, obs)
        report["action_shape"] = list(u.single_action_space.shape)
        report["observation_shapes"] = {k: list(v.shape) for k, v in obs.items()}
        start_video(u)
        if task == "I4h-Chole-Direct":
            report.update(chole_contact(env))
        elif task == "I4h-Aorta-Direct":
            report.update(aorta_controls(env))
        selected_reset(env)
        env.reset()
        navigation = (runpy.run_path(str(Path(__file__).resolve().parents[2] /
                                        "scripts/qualify_procedures.py"))["navigation_action"]
                      if u.procedure is not None else None)
        rewards = []
        for cycle in range(3):
            total = torch.zeros(count, device=u.device)
            completed = torch.zeros(count, dtype=torch.bool, device=u.device)
            for tick in range(u.max_episode_length):
                action = torch.zeros((count, u.single_action_space.shape[0]), device=u.device)
                if u.procedure is not None:
                    action[:] = torch.as_tensor(navigation(u.cfg.case, tick * u.cfg.decimation), device=u.device)
                _, reward, term, trunc, extras = step(env, action)
                total += reward
                done = term | trunc
                if u.procedure is not None:
                    assert not trunc.any(), "authored ReachTarget must succeed before timeout"
                    if term.any():
                        assert (extras["terminal_task"][term, 11] == 1).all()
                        assert (reward[term] == 1).all(), reward
                else:
                    assert not term.any() and (reward == 0).all()
                    if trunc.any():
                        assert tick == u.max_episode_length - 1
                completed |= done
                if completed.all():
                    break
            assert completed.all(), (cycle, completed)
            if u.procedure is not None:
                assert (total == 1).all(), total
            rewards.append(total.tolist())
        report["episode_rewards"] = rewards
        from helpers import lab_runtime
        assert (lab_runtime().graph is not None) == captured
    finally:
        evidence(task, count, captured, "lifecycle", report)
        close(env)
    reopened = make(task, count, captured)
    try:
        reopened.reset()
        step(reopened, torch.zeros((count, reopened.unwrapped.single_action_space.shape[0]), device=reopened.unwrapped.device))
    finally:
        close(reopened)
    report["passed"] = True
    evidence(task, count, captured, "lifecycle", report)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "graph"])
def test_direct_ppo(task, count, captured):
    from i4h_isaaclab.agents import checked_iteration, runner_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env = make(task, count, captured)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent.device = str(env.unwrapped.device)
        runner = OnPolicyRunner(wrapped, runner_dict(agent), log_dir=None, device=agent.device)
        before = [p.detach().clone() for p in runner.alg.actor.parameters()]
        losses = checked_iteration(runner, wrapped)
        assert losses and all(np.isfinite(v) for v in losses.values()), losses
        finite(env.unwrapped, wrapped.get_observations())
        assert any(not torch.equal(p, old) for p, old in zip(runner.alg.actor.parameters(), before))
        evidence(task, count, captured, "ppo", {"passed": True, "losses": losses})
    finally:
        close(env)
