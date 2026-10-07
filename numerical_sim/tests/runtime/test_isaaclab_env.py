# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M7: I4hEnv inside a kitless Isaac Lab 3.0 process. Runs only with the Isaac Lab venv's python
(``$PY -m pytest tests/runtime/test_isaaclab_env.py``, see README); the worktree venv skips it. One env per
process at a time: the fixture closes the previous env and clears Isaac Lab's simulation context. Every test runs on
both backends (``backend`` fixture: Isaac Lab's NewtonManager and the fallback I4hPhysicsManager) unless it pins one."""

import numpy as np
import pytest
import warp as wp
import yaml
from helpers import needs_external_data

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")
gym = pytest.importorskip("gymnasium")

import i4h_isaaclab.tasks  # noqa: E402, F401  registers the ids
from helpers import find, needs_chole, needs_vessels  # noqa: E402
from i4h_isaaclab.composition import read_case  # noqa: E402
from i4h_isaaclab.env import I4hEnv, I4hEnvCfg  # noqa: E402
from i4h_isaaclab.physics import I4hPhysicsManager  # noqa: E402
from i4h_physics.systems.mechanics.particles import Particles  # noqa: E402
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402


def teardown():
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


@pytest.fixture(params=["newton", "i4h"])
def backend(request):
    return request.param


@pytest.fixture
def make_env(backend):
    envs = []

    def _make(case=None, num_envs=2, task=None, backend=backend, env_class=I4hEnv, **kw):
        for env in envs:
            env.close()
        envs.clear()
        teardown()
        if task:
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

            cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
            if case is not None:
                cfg.case = case
            for name, value in kw.items():
                setattr(cfg, name, value)
        else:
            cfg = I4hEnvCfg(case="two_worlds" if case is None else case, **kw)
        cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=1.0)
        cfg.backend = backend
        env = gym.make(task, cfg=cfg) if task else env_class(cfg)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
    teardown()


def tissue_positions(env):
    """The block's positions as the observation's leading columns, plus the q0 view for comparison."""
    u = env.unwrapped
    scene, body = u.simulator.scene, u.simulator.scene.bodies[0]
    return u.obs_views[0], wp.to_torch(scene.view(scene.q0, body)).view(u.num_envs, -1)


def test_spaces_steps_and_time(make_env):
    env = make_env(stagger_episodes=False)
    u = env.unwrapped
    obs, extras = env.reset()
    A = u.single_action_space.shape[0]
    assert isinstance(u.single_action_space, gym.spaces.Box) and A == 10
    assert (u.single_action_space.low == -1.0).all() and (u.single_action_space.high == 1.0).all()
    assert obs["policy"].shape == (2, u.cfg.observation_space) and obs["policy"].dtype == torch.float32
    assert u.observation_space["policy"].shape == (2, u.cfg.observation_space)
    gen = torch.Generator(device=u.device).manual_seed(0)
    torch.cuda.set_sync_debug_mode("error")  # any host synchronisation inside the loop raises
    try:
        for i in range(60):
            a = torch.zeros((2, A), device=u.device) if i < 30 else torch.rand((2, A), generator=gen, device=u.device) * 2 - 1
            obs, rew, term, trunc, extras = env.step(a)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(obs["policy"]).all() and rew.shape == (2,) and not term.any() and not trunc.any()
    assert abs(u.sim.physics_manager.get_simulation_time() - 60 * u.cfg.decimation * u.physics_dt) < 1e-6
    assert int(u.episode_length_buf[0]) == 60


def test_timeout_resets_one_env(make_env):
    env = make_env(stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    a = torch.zeros((2, 10), device=u.device)
    a[0, 2] = 1.0  # env 0's probe rises into the block, env 1 holds
    for _ in range(20):
        env.step(a)
    q, q0 = tissue_positions(env)
    torch.cuda.synchronize()
    assert not torch.equal(q[0], q0[0]) and not torch.equal(q[1], q0[1]), "both envs moved under gravity / the probe"
    u.episode_length_buf[0] = u.max_episode_length - 1  # env 0 times out on the next step
    obs, rew, term, trunc, extras = env.step(torch.zeros((2, 10), device=u.device))
    torch.cuda.synchronize()
    assert trunc.tolist() == [True, False] and u.episode_length_buf.tolist() == [0, 21]
    assert torch.equal(q[0], q0[0]), "env 0 back at rest after its time-out"
    assert not torch.equal(q[1], q0[1]), "env 1 untouched by the reset"
    assert torch.equal(obs["policy"][0, : q.shape[1]], q0[0]), "the observation after the reset is the reset state"
    control = find(u.simulator, InstrumentControl)
    assert torch.allclose(u.action_views[0][0, :3], torch.as_tensor(control.rest[0], device=u.device))


def test_gym_make_and_registry(make_env):
    env = make_env(task="I4h-TwoWorlds-Direct")
    assert isinstance(env.unwrapped, I4hEnv) and env.unwrapped.cfg.case == "two_worlds"
    spec = gym.spec("I4h-Chole-Direct")
    assert spec.kwargs["env_cfg_entry_point"] == "i4h_isaaclab.tasks:I4hCholeEnvCfg"
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == "i4h_isaaclab.agents:I4hPPORunnerCfg"
    obs, _ = env.reset()
    obs, *_ = env.step(torch.zeros((2, 10), device=env.unwrapped.device))
    assert torch.isfinite(obs["policy"]).all()


def rollout(env, sequence, stream=None):
    """Step through a fixed action sequence, optionally with the policy and the step on a torch side stream behind a
    few ms of side-stream work; returns the final observation and the probe roots (after a synchronize)."""
    u = env.unwrapped
    env.reset()
    pad = torch.randn((2048, 2048), device=u.device)
    for a in sequence:
        if stream is None:
            obs, *_ = env.step(a)
        else:
            with torch.cuda.stream(stream):
                slow = (pad @ pad).sum() * 0.0 + 1.0  # the action depends on work still running on the side stream
                obs, *_ = env.step(a * slow)
    torch.cuda.synchronize()
    return obs["policy"].clone(), wp.to_torch(find(u.simulator, InstrumentControl).root).clone()


class CountingEnv(I4hEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.reset_calls.append(torch.as_tensor(env_ids).tolist())


def test_subclass_reset_idx_runs_on_automatic_resets(make_env):
    """An overridden _reset_idx (the DirectRLEnv extension point) also runs on time-out resets, not only env.reset()."""
    CountingEnv.reset_calls = []
    env = make_env(stagger_episodes=False, env_class=CountingEnv)
    env.reset()
    a = torch.zeros((2, 10), device=env.device)
    env.step(a)
    calls = len(env.reset_calls)
    env.episode_length_buf[1] = env.max_episode_length - 1  # env 1 times out on the next step
    *_, trunc, _ = env.step(a)
    assert trunc.tolist() == [False, True]
    assert env.reset_calls[calls:] == [[1]]
    q, q0 = tissue_positions(env)
    torch.cuda.synchronize()
    assert torch.equal(q[1], q0[1]), "env 1 was reset through the id path"


def assert_obs_in_spaces(u, obs):
    """Every returned key is declared, and each env's row and the batch lie in the single and batched spaces."""
    host = {k: v.cpu().numpy() for k, v in obs.items()}
    assert set(host) == set(u.single_observation_space.spaces) == set(u.observation_space.spaces)
    for k, v in host.items():
        assert v.dtype == u.single_observation_space[k].dtype == u.observation_space[k].dtype, k
    assert u.observation_space.contains(host)
    for e in range(u.num_envs):
        assert u.single_observation_space.contains({k: v[e] for k, v in host.items()})


@pytest.mark.parametrize("kw", [{}, {"camera": 16}], ids=["state", "camera"])
def test_observation_spaces_match_observations(make_env, kw):
    env = make_env(stagger_episodes=False, **kw)
    u = env.unwrapped
    obs, _ = env.reset()
    assert_obs_in_spaces(u, obs)
    obs, *_ = env.step(torch.zeros((2, 10), device=u.device))
    assert_obs_in_spaces(u, obs)


@needs_vessels
def test_procedure_observation_spaces_match_observations(make_env):
    env = make_env(case="aorta_static_reach", stagger_episodes=False)
    u = env.unwrapped
    obs, _ = env.reset()
    assert "task" in obs
    assert_obs_in_spaces(u, obs)
    obs, *_ = env.step(torch.zeros((2, u.low.numel()), device=u.device))
    assert_obs_in_spaces(u, obs)


def gravity_case(tmp_path, gravity):
    """two_worlds with ``sim.gravity`` replaced (None: removed)."""
    cfg = read_case("two_worlds")
    cfg["sim"].pop("gravity", None)
    if gravity is not None:
        cfg["sim"]["gravity"] = list(gravity)
    path = tmp_path / "gravity.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def lab_sim(gravity):
    from i4h_isaaclab.physics import I4hPhysicsCfg
    from isaaclab.sim import SimulationCfg

    return SimulationCfg(dt=1.0 / 60.0, render_interval=2, device="cuda:0", physics=I4hPhysicsCfg(), gravity=gravity)


def test_case_gravity_is_the_default(make_env, tmp_path):
    """gravity_from_case (default): a declared case gravity sets Lab's, even over an explicit Lab value (both backends)."""
    env = make_env(case=gravity_case(tmp_path, (0.0, 0.0, 0.0)), stagger_episodes=False, sim=lab_sim((0.0, 0.0, -9.81)))
    u = env.unwrapped
    assert tuple(u.cfg.sim.gravity) == (0.0, 0.0, 0.0)
    assert np.allclose(u.simulator.scene.gravity.numpy(), 0.0)


def test_lab_gravity_rules_without_gravity_from_case(make_env, tmp_path):
    """gravity_from_case=False: a case without gravity takes Lab's on both backends; a different declared one raises."""
    env = make_env(case=gravity_case(tmp_path, None), stagger_episodes=False, sim=lab_sim((0.0, -9.81, 0.0)),
                   gravity_from_case=False)
    scene = env.unwrapped.simulator.scene
    assert np.allclose(scene.gravity.numpy(), [[0.0, -9.81, 0.0]] * 2)
    assert np.allclose(scene.down, [0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="differs from the case's sim.gravity"):
        make_env(case=gravity_case(tmp_path, (0.0, 0.0, -9.81)), stagger_episodes=False, sim=lab_sim((0.0, -9.81, 0.0)),
                 gravity_from_case=False)


class PositionRewardEnv(I4hEnv):
    """Custom hooks that read the zero-copy views without calling super() or any sync."""

    def _get_rewards(self):
        return self.obs_views[0][:, 2].clone()

    def _get_dones(self):
        return self._never_terminated, self.obs_views[0][:, 2] > 1.0e9


def test_overridden_hooks_read_synced_views(make_env):
    """The physics manager orders torch after each physics call: overridden hooks read the stepped state (decimation 2,
    a non-default torch stream), and the hook runs once per manager call."""
    env = make_env(stagger_episodes=False, decimation=2, env_class=PositionRewardEnv)
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    holder = u.simulator if u.cfg.backend == "newton" else I4hPhysicsManager
    calls = []
    hook = holder.after_step
    holder.after_step = lambda: (calls.append(1), hook())
    a = torch.zeros((2, 10), device=dev)
    a[:, 2] = 1.0
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        for _ in range(5):
            _, reward, *_ = env.step(a)
            torch.cuda.current_stream(dev).synchronize()
            assert torch.equal(reward, u.obs_views[0][:, 2]), "the hook read the state before the step"
    per_step = 1 if u._physics_handles_decimation else u.cfg.decimation
    assert len(calls) == 5 * per_step


def count_calls(monkeypatch, module, name):
    calls = []
    real = getattr(module, name)

    def counted(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_one_case_read_and_one_scene_load_per_construction(make_env, backend, monkeypatch):
    """The case is read once per construction; on newton the scene loaded for the replication is the one the solver
    allocates (a hard reset loads afresh, outside the construction)."""
    import i4h_isaaclab.env as env_module
    import i4h_isaaclab.newton_manager as nm

    reads = count_calls(monkeypatch, env_module, "read_case")
    loads = count_calls(monkeypatch, nm, "load_scene")
    for _ in range(2):  # two consecutive constructions, each on its own
        reads.clear(), loads.clear()
        env = make_env(stagger_episodes=False)
        assert len(reads) == 1 and len(loads) == (1 if backend == "newton" else 0), (len(reads), len(loads))
        assert nm.I4hNewtonManager.pending_scene is None
    if backend == "newton":
        old = env.unwrapped.simulator
        env.unwrapped.sim.reset(soft=False)
        assert len(loads) == 2 and env.unwrapped.simulator is not old


@pytest.mark.parametrize("backend", ["newton"])
def test_failed_construction_leaves_no_pending_scene(make_env, monkeypatch):
    import i4h_isaaclab.newton_manager as nm

    def fail(self, sim):
        raise RuntimeError("bind failed")

    with monkeypatch.context() as patch:
        patch.setattr(I4hEnv, "_bind", fail)
        with pytest.raises(RuntimeError, match="bind failed"):
            make_env(stagger_episodes=False)
    assert nm.I4hNewtonManager.pending_scene is None
    loads = count_calls(monkeypatch, nm, "load_scene")
    env = make_env(stagger_episodes=False)
    env.step(torch.zeros((2, 10), device=env.unwrapped.device))
    assert len(loads) == 1


def test_user_stream_ordering(make_env):
    """action -> step and step -> observation ordering hold when torch runs on a non-default stream."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1)
    sequence = [torch.rand((2, 10), generator=gen, device=dev) * 2 - 1 for _ in range(40)]
    obs_default, roots_default = rollout(make_env(stagger_episodes=False), sequence)
    env = make_env(stagger_episodes=False)
    obs_side, roots_side = rollout(env, sequence, stream=torch.cuda.Stream(device=dev))
    assert torch.allclose(roots_side, roots_default, atol=1e-5), "a stale action moves the probe by millimetres"
    assert torch.allclose(obs_side, obs_default, atol=1e-4)
    # step -> observation: the returned observation is the state after the step, read on the side stream
    q, _ = tissue_positions(env)
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        obs, *_ = env.step(sequence[0])
        copy = obs["policy"][:, : q.shape[1]].clone()
    torch.cuda.synchronize()
    assert torch.equal(copy, q)


def test_reset_ids_from_a_side_stream(make_env):
    """Reset ids computed by slow work on a torch side stream select the right env (the int32 cast is ordered before
    Warp reads the ids)."""
    env = make_env(stagger_episodes=False)
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    a = torch.zeros((2, 10), device=dev)
    a[:, 2] = 1.0  # both probes rise into their blocks
    for _ in range(20):
        env.step(a)
    q, q0 = tissue_positions(env)
    torch.cuda.synchronize()
    assert not torch.equal(q[0], q0[0]) and not torch.equal(q[1], q0[1])
    pad = torch.randn((4096, 4096), device=dev)
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        slow = ((pad @ pad).sum() * 0.0 + 1.0).to(torch.int64)  # a few ms of work the ids depend on
        ids = torch.ones(1, dtype=torch.int64, device=dev) * slow  # env 1, only known once the matmul is done
        u._reset_idx(ids)
    torch.cuda.synchronize()
    assert torch.equal(q[1], q0[1]), "env 1 was reset"
    assert not torch.equal(q[0], q0[0]), "env 0 was not"


def shift_observed(env, env_ids, offset):
    """A reset event that reads the reset positions and moves them: the observed body (env.obs_views[0])."""
    q = env.obs_views[0]
    q[env_ids] = q[env_ids] + offset


@configclass
class ShiftEvents:
    shift = EventTermCfg(func=shift_observed, mode="reset", params={"offset": 0.01})


def test_reset_event_survives_the_physical_reset(make_env):
    """An Isaac Lab reset event runs after the physical reset (on a non-default torch stream): its change stays."""
    env = make_env(stagger_episodes=False, events=ShiftEvents())
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    a = torch.zeros((2, 10), device=dev)
    a[:, 2] = 1.0
    for _ in range(20):
        env.step(a)
    q, q0 = tissue_positions(env)
    moved = q[0].clone()
    pad = torch.randn((2048, 2048), device=dev)
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        (pad @ pad).sum()  # queued work on the side stream ahead of the reset
        u._reset_idx(torch.tensor([1], device=dev))
    torch.cuda.synchronize()
    assert torch.allclose(q[1], q0[1] + 0.01, atol=1e-6), "the event shifted the reset state, and it stayed"
    assert torch.equal(q[0], moved), "env 0 untouched"


def shift_observed_late(env, env_ids, offset):
    """shift_observed after slow torch work: an unordered Warp consumer would read the state before the shift."""
    pad = torch.randn((2048, 2048), device=env.device)
    for _ in range(4):
        pad = pad @ pad * 1e-3
    q = env.obs_views[0]
    q[env_ids] = q[env_ids] + offset + pad.sum() * 0.0


@configclass
class LateShiftEvents:
    shift = EventTermCfg(func=shift_observed_late, mode="reset", params={"offset": 0.05})


def test_reset_event_reaches_the_camera(make_env):
    """Without a procedure, the camera still renders after the events' writes (non-default torch stream)."""
    env = make_env(camera=64, stagger_episodes=False, events=LateShiftEvents())
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    for _ in range(3):
        env.step(torch.zeros((2, 10), device=dev))
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        u._reset_idx(torch.tensor([0, 1], device=dev))
        obs = u._get_observations()
        rgb, depth = obs["rgb"].clone(), obs["depth"].clone()
    torch.cuda.synchronize()
    q, q0 = tissue_positions(env)
    assert torch.allclose(q, q0 + 0.05, atol=1e-6)
    u.camera.render()  # a synchronized rerender of the same state
    torch.cuda.synchronize()
    assert torch.equal(rgb, u.camera.rgb_t) and torch.equal(depth, u.camera.depth_t), "the camera saw the pre-event state"


@needs_vessels
def test_reset_event_reaches_the_procedure(make_env):
    """The procedure resets after the events: its initial tip sample is the shifted tip."""
    env = make_env(case="aorta_static_reach", observe="catheter", stagger_episodes=False, events=ShiftEvents())
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    for _ in range(5):
        env.step(torch.zeros((2, u.low.numel()), device=dev))
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        u._reset_idx(torch.tensor([0, 1], device=dev))
    torch.cuda.synchronize()
    tips = wp.to_torch(u.procedure.tip_q).view(2, -1, 3)[:, u.procedure.tip_index]
    scene = u.simulator.scene
    rest = wp.to_torch(scene.view(scene.q0, scene.body("catheter"))).view(2, -1, 3)[:, -1]
    assert torch.allclose(tips, rest + 0.01, atol=1e-5), "the event moved the catheter"
    assert torch.allclose(wp.to_torch(u.procedure.previous_tip), tips, atol=1e-6), "the procedure sampled the moved tip"


def test_terminal_camera_observation_survives_the_reset(make_env):
    """compute_final_obs: the terminal images must not be overwritten by the post-reset render."""
    env = make_env(camera=32, stagger_episodes=False, compute_final_obs=True)
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    a = torch.zeros((2, 10), device=dev)
    a[0, :3] = torch.tensor([0.16, -0.2, 0.3], device=dev)  # env 0's probe moves in front of the camera
    for _ in range(20):
        obs, *_ = env.step(a)
    u.episode_length_buf[0] = u.max_episode_length - 1
    obs, rew, term, trunc, extras = env.step(a)
    torch.cuda.synchronize()
    final = extras["final_obs"]
    assert trunc.tolist() == [True, False] and "rgb" in final and "depth" in final
    assert final["depth"].data_ptr() != obs["depth"].data_ptr()
    assert not torch.equal(final["depth"][0], obs["depth"][0]), "env 0's terminal frame is not its reset frame"
    assert torch.equal(final["depth"][1], obs["depth"][1]), "env 1 was not reset: same state, same frame"


def test_kept_camera_observations_do_not_change(make_env):
    """Images kept from one step (a replay buffer) are copies: the next render leaves them as they were."""
    env = make_env(camera=16, stagger_episodes=False)
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    a = torch.zeros((2, 10), device=dev)
    a[0, :3] = torch.tensor([0.16, -0.2, 0.3], device=dev)
    kept, *_ = env.step(a)
    snapshot = {k: kept[k].clone() for k in ("rgb", "depth")}
    for _ in range(3):
        env.step(a)
    torch.cuda.synchronize()
    assert kept["rgb"].data_ptr() != u.camera.rgb_t.data_ptr() and kept["depth"].data_ptr() != u.camera.depth_t.data_ptr()
    assert torch.equal(kept["rgb"], snapshot["rgb"]) and torch.equal(kept["depth"], snapshot["depth"])
    assert not torch.equal(kept["depth"][0], u.camera.depth_t[0]), "the probe moved: the live frame differs"


def test_camera_observation_ordering(make_env):
    """step -> camera -> observation: the images returned by step() are this step's frame, read on a side stream."""
    env = make_env(camera=32, stagger_episodes=False)
    u = env.unwrapped
    dev = torch.device(u.device)
    obs, _ = env.reset()
    assert obs["rgb"].shape == (2, 32, 32, 4) and obs["rgb"].dtype == torch.uint8
    assert obs["depth"].shape == (2, 32, 32) and obs["depth"].dtype == torch.float32
    a = torch.zeros((2, 10), device=dev)
    a[0, :3] = torch.tensor([0.16, -0.2, 0.3], device=dev)  # env 0's probe moves toward a spot in front of the camera
    previous = obs["depth"].clone()
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        for _ in range(3):
            obs, *_ = env.step(a)
            rgb, depth = obs["rgb"].clone(), obs["depth"].clone()
    torch.cuda.synchronize()
    assert torch.equal(rgb, wp.to_torch(u.camera.rgb)) and torch.equal(depth, wp.to_torch(u.camera.depth))
    assert not torch.equal(depth[0], previous[0]), "env 0's frame follows its moving probe"
    assert torch.isfinite(depth).all() and (depth >= 0).all()


@needs_chole
def test_chole_env_steps(make_env):
    env = make_env(task="I4h-Chole-Direct", num_envs=4)
    u = env.unwrapped
    obs, _ = env.reset()
    A = 10 * find(u.simulator, InstrumentControl).count  # the case's instruments (five since M9b)
    assert u.single_action_space.shape == (A,)
    assert obs["policy"].shape[1] == sum(v.shape[1] for v in u.obs_views + u.state_views)
    gen = torch.Generator(device=u.device).manual_seed(0)
    for _ in range(10):
        obs, *_ = env.step(torch.rand((4, A), generator=gen, device=u.device) * 2 - 1)
    assert torch.isfinite(obs["policy"]).all()


@pytest.mark.parametrize("task", [
    "I4h-TwoWorlds-Direct",
    pytest.param("I4h-GraspTarget-Direct", marks=needs_chole),
    pytest.param("I4h-ReachTarget-Direct", marks=needs_vessels),
])
def test_ppo_smoke(make_env, task):
    pytest.importorskip("rsl_rl")
    from i4h_isaaclab.agents import checked_iteration, runner_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    env = make_env(task=task, num_envs=8)
    u = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env)
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent.device = str(u.device)
    runner = OnPolicyRunner(wrapped, runner_dict(agent), log_dir=None, device=str(u.device))
    obs = wrapped.get_observations()
    expected = torch.cat([obs["policy"], obs["task"]], dim=-1) if u.procedure is not None else obs["policy"]
    for model in (runner.alg.actor, runner.alg.critic):
        model.eval()
        assert model.obs_dim == expected.shape[-1]
        with torch.inference_mode():
            assert torch.equal(model.get_latent(obs), model.obs_normalizer(expected))
            if u.procedure is not None:
                changed = obs.clone()
                changed["task"] += 1.0
                assert not torch.equal(model(obs), model(changed)), "the network must respond to task observations"
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    assert len(set(u.episode_length_buf.tolist())) > 1, "episode lengths were randomised at the start"
    losses = checked_iteration(runner, wrapped)
    assert losses and all(np.isfinite(v) for v in losses.values()), losses
    u.simulator.step()
    torch.cuda.synchronize()
    assert torch.isfinite(u.obs_views[0]).all()


def test_hard_reset_rebinds(make_env, backend):
    """sim.reset(soft=False): the newton backend re-finalizes the model and rebuilds the solver; the env rebinds its
    views and camera to the new arrays and keeps stepping from the rest state (the fallback resets every env)."""
    env = make_env(camera=16, stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    a = torch.zeros((2, 10), device=u.device)
    a[:, 2] = 1.0
    for _ in range(10):
        env.step(a)
    old_sim, old_q = u.simulator, u.simulator.scene.q.ptr
    u.sim.reset(soft=False)
    torch.cuda.synchronize()
    q, q0 = tissue_positions(env)
    assert torch.equal(q, q0), "every env back at rest"
    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager

        assert u.simulator is not old_sim and u.simulator.scene.q.ptr != old_q
        assert u.simulator.scene.q.ptr == NewtonManager.get_state_0().particle_q.ptr
        assert u.obs_views[0].data_ptr() == wp.to_torch(u.simulator.scene.q).data_ptr()
        assert u.camera.sim is u.simulator
    for _ in range(10):
        obs, *_ = env.step(a)
    torch.cuda.synchronize()
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["depth"]).all()
    assert not torch.equal(q[0], q0[0]), "stepping continues on the rebound arrays"


def test_close_and_reopen(make_env, backend):
    """A second env after closing the first: nothing of the first (callbacks, bundle, simulator) is left behind."""
    first = make_env()
    first.reset()
    first.step(torch.zeros((2, 10), device=first.unwrapped.device))
    previous = first.unwrapped.simulator.physics
    env = make_env(num_envs=3)
    assert not previous.valid
    with pytest.raises(RuntimeError, match="invalidated"):
        previous.reset()
    u = env.unwrapped
    assert u.simulator.scene.envs == 3
    if backend == "newton":
        from i4h_isaaclab.newton_manager import I4hNewtonManager

        assert len(I4hNewtonManager.on_solver) == 1 and I4hNewtonManager.simulator is u.simulator
    else:
        from i4h_isaaclab.physics import I4hPhysicsManager

        assert I4hPhysicsManager.sim is u.simulator
    env.reset()
    for _ in range(5):
        obs, *_ = env.step(torch.rand((3, 10), device=u.device) * 2 - 1)
    torch.cuda.synchronize()
    assert torch.isfinite(obs["policy"]).all()


@pytest.mark.parametrize("backend", ["newton"])
@pytest.mark.parametrize("case", ["two_worlds", pytest.param("aorta_static", marks=needs_vessels)])
def test_backends_match(make_env, case):
    """The same kernels on the same actions: both backends' observations agree up to atomics noise."""
    dev = torch.device("cuda:0")
    results = {}
    for name in ("i4h", "newton"):
        env = make_env(case=case, backend=name, stagger_episodes=False)
        A = env.unwrapped.single_action_space.shape[0]
        gen = torch.Generator(device=dev).manual_seed(3)
        sequence = [torch.rand((2, A), generator=gen, device=dev) * 2 - 1 for _ in range(40)]
        env.reset()
        for a in sequence:
            obs, *_ = env.step(a)
        torch.cuda.synchronize()
        results[name] = obs["policy"].clone()
    diff = float((results["i4h"] - results["newton"]).abs().max())
    assert diff < 1e-4, diff


@pytest.mark.parametrize("use_mask", [False, True], ids=["ids", "mask"])
def test_direct_reset_from_a_side_stream(make_env, use_mask):
    """simulator.reset called directly (not through the env) with int64 ids or a bool mask that slow side-stream work
    produces: select_envs orders Warp after torch itself, so env 1 and only env 1 is reset."""
    env = make_env(stagger_episodes=False)
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    a = torch.zeros((2, 10), device=dev)
    a[:, 2] = 1.0
    for _ in range(20):
        env.step(a)
    q, q0 = tissue_positions(env)
    torch.cuda.synchronize()
    pad = torch.randn((4096, 4096), device=dev)
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        slow = ((pad @ pad).sum() * 0.0 + 1.0).to(torch.int64)
        if use_mask:
            u.simulator.reset(mask=torch.arange(2, device=dev) == slow)
        else:
            u.simulator.reset(torch.ones(1, dtype=torch.int64, device=dev) * slow)
    torch.cuda.synchronize()
    assert torch.equal(q[1], q0[1]), "env 1 was reset"
    assert not torch.equal(q[0], q0[0]), "env 0 was not"


def test_structural_edit_needs_prepare(make_env, backend):
    """Disabling a system is structural on both backends: stepping refuses until simulator.prepare(), which drops it
    from the solver's active list and recaptures the graphs."""
    env = make_env(stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    a = torch.zeros((2, 10), device=u.device)
    env.step(a)
    particles = find(u.simulator, Particles)
    particles.enabled = False
    with pytest.raises(RuntimeError, match="prepare"):
        env.step(a)
    with pytest.raises(RuntimeError, match="prepare"):
        u.simulator.reset([0])
    from isaaclab_newton.physics import NewtonManager

    def step_graph():
        return NewtonManager._graph if backend == "newton" else u.simulator.graph

    old_graph = step_graph()
    u.simulator.prepare()
    assert particles not in u.simulator.solver.active
    if backend == "newton":
        # The qualified Lab (i4h_isaaclab.lab.QUALIFIED) runs the next requested step eagerly, then captures.
        assert NewtonManager._graph is None and NewtonManager._graph_capture_pending
        env.step(a)
        assert not NewtonManager._graph_capture_pending
    assert step_graph() is not None and step_graph() is not old_graph, "the step graph was recaptured"
    for _ in range(5):
        obs, *_ = env.step(a)
    u.simulator.reset([0])
    torch.cuda.synchronize()
    assert torch.isfinite(obs["policy"]).all()


@pytest.mark.parametrize("dt", [None, 0.016667], ids=["default", "rounded"])
def test_case_dt_default_and_rounding(make_env, tmp_path, dt):
    """A case without sim.dt takes the default and a rounded 1/60 passes; both backends then step with Isaac Lab's dt."""
    case = read_case("two_worlds")
    case["sim"].pop("dt")
    if dt is not None:
        case["sim"]["dt"] = dt
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(case))
    env = make_env(case=str(path), stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    for _ in range(3):
        obs, *_ = env.step(torch.zeros((2, 10), device=u.device))
    torch.cuda.synchronize()
    assert torch.isfinite(obs["policy"]).all() and u.simulator.dt == pytest.approx(u.physics_dt, abs=0)


@needs_external_data
def test_empty_reset_preserves_assessment_after_partial_reset(make_env, tmp_path):
    cfg = read_case("aorta_static_reach")
    cfg["procedure"].update(radius=0.001, target={"position": [0, 0, 0]}, timeout=120.0)
    path = tmp_path / "empty-reset.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = make_env(case=str(path), decimation=2, stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    action = torch.zeros((2, u.low.numel()), device=u.device)
    env.step(action)
    u._reset_idx(torch.tensor([0], device=u.device))
    u.sim.step(render=False)  # advance one physics sample without the host automatic-reset path
    assert u.simulator._mask.numpy().tolist() == [True, False]
    u._warp_to_torch()
    before = u.task_rows.clone()
    positions = u.simulator.scene.q.numpy().copy()
    assert torch.all(before[:, 9] > 0)
    u._reset_idx([])
    u._warp_to_torch()
    assert torch.equal(before, u.task_rows)
    assert np.array_equal(positions, u.simulator.scene.q.numpy())


@needs_vessels
def test_procedure_episode_ends_at_the_length_limit(make_env, tmp_path):
    """A running procedure (far from its own deadline) still ends at the env's episode-length limit."""
    cfg = read_case("aorta_static_reach")
    cfg["procedure"].update(radius=0.001, target={"position": [0, 0, 0]}, timeout=1000.0)
    path = tmp_path / "long.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = make_env(case=str(path), stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    action = torch.zeros((2, u.low.numel()), device=u.device)
    env.step(action)
    u.episode_length_buf[0] = u.max_episode_length - 1
    _, _, term, trunc, _ = env.step(action)
    assert trunc.tolist() == [True, False] and term.tolist() == [False, False]
    assert int(u.episode_length_buf[0]) == 0, "env 0 was reset"


@pytest.mark.parametrize('outcome', ['success', 'timeout'])
@needs_external_data
def test_target_decimation_terminal_and_masked_reset(make_env, tmp_path, outcome):
    """Real Isaac Lab: the 6th physics step terminates inside a 4-step action; reward survives steps 7/8."""
    cfg = read_case('aorta_static_reach')
    cfg['procedure'].update(radius=100.0 if outcome == 'success' else 0.001,
                            target={'position':[0,0,0]}, hold=0.1, timeout=0.1)
    path = tmp_path / 'target.yaml'
    path.write_text(yaml.safe_dump(cfg))
    env = make_env(case=str(path), decimation=4, stagger_episodes=False)
    u = env.unwrapped
    obs, _ = env.reset()
    assert obs['task'].shape == (2,8)
    action = torch.zeros((2, u.low.numel()), device=u.device)
    env.step(action)
    u._warp_to_torch()
    assert torch.allclose(u.task_rows[:,9], torch.full((2,),4*u.physics_dt,device=u.device))
    u._reset_idx(torch.tensor([0],device=u.device))
    obs, reward, term, trunc, extras = env.step(action)
    assert reward.tolist() == ([0,1] if outcome == 'success' else [0,0])
    assert term.tolist() == ([False,True] if outcome == 'success' else [False,False])
    assert trunc.tolist() == ([False,False] if outcome == 'success' else [False,True])
    assert extras['terminal_task'][1,11].item() == (1 if outcome == 'success' else 2)
    u._warp_to_torch()
    before = u.task_rows.clone()
    u._get_rewards()
    u._get_dones()
    u._get_observations()
    assert torch.equal(before, u.task_rows)
    assert u.task_rows[1,9].item() == 0


@pytest.mark.parametrize('graph', [True, False])
@pytest.mark.parametrize('case', ['aorta_static_reach','chole_grasp'])
@needs_external_data
def test_authored_target_through_controls(make_env, case, graph, backend, tmp_path):
    """Actual procedure demos on each real backend, using physical actions and decimation two."""
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
    from i4h_isaaclab.env import apply_backend
    from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp
    cfg = I4hEnvCfg()
    cfg.backend = backend
    apply_backend(cfg)
    if backend == 'newton':
        cfg.sim.physics.use_cuda_graph = graph
    case_cfg = read_case(case)
    case_cfg['sim']['cuda_graph'] = graph
    path = tmp_path / 'procedure.yaml'
    path.write_text(yaml.safe_dump(case_cfg))
    env = make_env(case=str(path),sim=cfg.sim,num_envs=1,decimation=2,stagger_episodes=False,action_mode='physical')
    u = env.unwrapped
    env.reset()
    task = u.procedure
    if case == 'chole_grasp':
        control = find(u.simulator,InstrumentControl)
        grasp = find(u.simulator,VolumetricGrasp)
        k = control.names.index('grasper')
        sphere = int(grasp.jaw_start.numpy()[k])+2
    else:
        control = find(u.simulator,CatheterControl)
    actions = control.actions.numpy()
    success = False
    for i in range(180):
        step = i*2
        if case == 'chole_grasp':
            if step >= 30:
                target = task.target_q.numpy()[0,task.index]
                jaw = control.sphere_q.numpy()[0,sphere]
                actions[0,k,:3] = control.root.numpy()[0,k]+target-jaw
                actions[0,k,7] = float(step >= 110)
        else:
            actions[0] = [0.5 if step < 60 else 0,0,0]
        obs,reward,term,trunc,extras = env.step(torch.as_tensor(actions.reshape(1,-1),device=u.device))
        if term.item():
            assert reward.item() == 1 and not trunc.item()
            assert extras['terminal_task'][0,11].item() == 1
            success = True
            break
    assert success, task.results()


def test_procedure_hard_reset_rebinds(make_env, backend):
    env = make_env(case='aorta_static_reach',decimation=3,stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    action = torch.zeros((2,3),device=u.device)
    env.step(action)
    previous = u.procedure
    u.sim.reset(soft=False)
    u._warp_to_torch()
    assert not u.task_rows[:,9:].any()
    assert u.procedure.sim is u.simulator
    if backend == 'newton':
        assert u.procedure is not previous
    obs, *_ = env.step(action)
    assert obs['task'].shape == (2,8)
    u._warp_to_torch()
    assert torch.allclose(u.task_rows[:,9],torch.full((2,),3*u.physics_dt,device=u.device))


@needs_chole
def test_enabled_tissue_contact(make_env, tmp_path):
    """Real backend loop with contact enabled, ordinary policy observations and partial reset."""
    from i4h_physics.systems.contact.part_contact import PartContact

    cfg = read_case("chole_tet_med")
    next(s for s in cfg["systems"] if s["type"] == "tissue_contact")["enabled"] = True
    path = tmp_path / "contact.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = make_env(case=str(path), stagger_episodes=False)
    obs, _ = env.reset()
    c = find(env.simulator, PartContact)
    assert c.enabled
    width = obs["policy"].shape[1]
    actions = torch.zeros((2, env.single_action_space.shape[0]), device=env.device)
    contacts = 0
    for _ in range(30):
        obs, _, _, _, _ = env.step(actions)
        contacts += int(c.contact_count.numpy().sum())
    assert torch.isfinite(obs["policy"]).all() and obs["policy"].shape == (2, width)
    assert contacts > 0
    saved = c.q.numpy()[1].copy()
    env._reset_idx(torch.tensor([0], device=env.device))
    np.testing.assert_array_equal(c.q.numpy()[1], saved)
    assert c.contact_count.numpy()[0] == 0


@pytest.mark.parametrize("ids", [False, True])
def test_staggered_terminal_notification_clears_next_action(make_env, ids):
    env = make_env(case="aorta_static_reach", num_envs=2, decimation=1,
                   compute_final_obs=ids)
    u = env.unwrapped
    env.reset()
    action = torch.zeros((2, 3), device=u.device)
    for index in (0, 1):
        wp.to_torch(u.procedure.elapsed)[index] = u.procedure.timeout - u.physics_dt
        u._torch_to_warp()
        _, _, term, trunc, extras = env.step(action)
        expected = [index == 0, index == 1]
        assert not term.any() and trunc.tolist() == expected
        assert extras["new_terminal"].tolist() == expected
        pulse = extras["new_terminal"]
        retained = extras["terminal_task"].clone()
        _, _, term, trunc, extras = env.step(action)
        assert not term.any() and not trunc.any()
        assert not extras["new_terminal"].any()
        assert pulse.tolist() == expected
        u._reset_idx(torch.tensor([index], device=u.device))
        assert not u.extras["new_terminal"].any()
        assert torch.equal(u.extras["terminal_task"], retained)


def test_newton_missing_regional_provider_fails_before_capture():
    from i4h_endoluminal.loaders.rod import add_rod
    from i4h_isaaclab.newton_manager import NewtonSim
    from i4h_physics import Scene
    from i4h_physics.systems.rods.material import RodDirectMaterial
    from i4h_physics.systems.rods.rod_direct import RodDirect

    scene = Scene("cpu", 2, (0, 0, 0))
    add_rod(scene, "wire", [0, 0, 0], [1, 0, 0], 5, .02, .002,
            regions={"schema_version": 1, "default": "unassigned",
                     "regions": [{"name": "tip", "interval": [.8, 1]}]})
    scene.allocate()
    provider = RodDirectMaterial(scene, body="wire")
    rod = RodDirect(scene, body="wire")
    sim = NewtonSim.__new__(NewtonSim)
    sim.scene, sim.systems = scene, [rod]  # a removed provider leaves the existing channels behind
    with pytest.raises(ValueError, match="must precede"):
        sim.prepare()
    assert sim.reset_graph is None and sim._prepared is None
    assert provider.body.rod_material is rod.body.rod_material


@needs_external_data
@pytest.mark.parametrize("deadline,source", [(0.1, "procedure"), (0.4, "host_horizon")])
def test_effective_timeout_source_is_exported(make_env, deadline, source):
    cfg = read_case("aorta_static_reach")
    cfg["procedure"]["timeout"] = deadline
    cfg["procedure"]["target"] = {"position": [9, 9, 9]}
    env = make_env(cfg, episode_length_s=.2, stagger_episodes=False)
    u = env.unwrapped
    env.reset()
    expected = min(deadline, u.max_episode_length * u.step_dt)
    assert u.procedure.timeout == pytest.approx(expected)
    assert u.procedure.timeout_source == source
    action = torch.zeros((2, u.cfg.action_space.shape[0]), device=u.device)
    for _ in range(u.max_episode_length + 2):
        _, _, term, trunc, extras = env.step(action)
        assert not term.any()
        if trunc.any():
            assert trunc.all()
            assert extras["terminal_timeout_source"] == source
            assert extras["terminal_timeout_s"] == pytest.approx(expected)
            assert extras["new_terminal"].all()
            break
    else:
        pytest.fail("the effective deadline never terminated the episode")


def test_adapter_without_assessment_or_standalone():
    """The Isaac Lab adapter without the [apps] extra or i4h-standalone: its modules import, physics cases build and
    step on both backends, and a procedure case asks for the extra."""
    import os
    import subprocess
    import sys
    import textwrap

    from helpers import VESSELS

    vessels = bool(os.environ.get("I4H_DATA")) or (VESSELS / "AortaWithVesselsStatic.usdc").is_file()  # as needs_vessels

    script = textwrap.dedent(f"""
        import importlib, sys
        class Block:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in ("i4h_assessment", "i4h_standalone"):
                    raise ModuleNotFoundError(fullname, name=fullname)
        sys.meta_path.insert(0, Block())
        import torch
        for name in ("newton_manager", "physics", "views", "env", "tasks", "recording", "composition", "lab"):
            importlib.import_module("i4h_isaaclab." + name)
        from i4h_isaaclab.env import I4hEnv, I4hEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        cases = ["two_worlds"] + (["aorta_static"] if {vessels!r} else [])
        for case in cases:
            for backend in ("newton", "i4h"):
                cfg = I4hEnvCfg(case=case, backend=backend, stagger_episodes=False)
                cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
                env = I4hEnv(cfg)
                env.reset()
                for _ in range(3):
                    obs, *_ = env.step(torch.zeros((2, env.low.numel()), device=env.device))
                assert env.procedure is None and torch.isfinite(obs["policy"]).all()
                env.close()
                SimulationContext.clear_instance()
        try:
            I4hEnv(I4hEnvCfg(case="chole_grasp"))
        except ModuleNotFoundError as error:
            assert error.name == "i4h_assessment" and "[apps]" in str(error), error
        else:
            raise AssertionError("a procedure case built without i4h-assessment")
        print("adapter ok", cases)
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0 and "adapter ok" in out.stdout, out.stderr[-3000:]



@pytest.mark.parametrize("backend", ["newton"])
def test_host_particles_share_the_model(make_env):
    """F1: a host hook adds its own particles to every world before ours: our rows are found among them, the env's
    observations equal those of a host-free run, the host particles stay put, and a hard rebuild keeps the layout."""
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import NewtonManager

    def host_particles(builder, env, _position, _quaternion):
        builder.add_particles([(3.0, 3.0, 3.0), (3.1, 3.0, 3.0), (3.2, 3.0, 3.0)], [(0.0, 0.0, 0.0)] * 3, [0.0] * 3)

    def rollout(env):
        env.reset()
        a = torch.zeros((2, 10), device=env.device)
        a[:, 2] = 1.0
        for _ in range(15):
            obs, *_ = env.step(a)
        torch.cuda.synchronize()
        return obs["policy"].clone()

    # The host env first: closing an env clears Lab's world hooks, so the host-free env is built afterwards.
    with newton_builder_world_hook(host_particles):  # registered before ours: its particles come first
        env = make_env(stagger_episodes=False)
    u = env.unwrapped
    main = u.simulator.scene.main
    assert main.placement is not None and main.placement[2] == 3, main.placement

    def host_rows():
        return NewtonManager.get_state_0().particle_q.numpy()[[e * main.placement[1] + i for e in range(2) for i in range(3)]]

    before = host_rows()
    shared = rollout(env)
    np.testing.assert_array_equal(host_rows(), before)
    old = u.simulator
    u.sim.reset(soft=False)  # rebuilt outside the construction hook: the recorded layout is reused
    assert u.simulator is not old and u.simulator.scene.main.placement == main.placement
    rebuilt = rollout(env)
    alone = rollout(make_env(stagger_episodes=False))
    assert torch.allclose(shared, alone, atol=1e-5) and torch.allclose(rebuilt, alone, atol=1e-5)


@pytest.mark.parametrize("backend", ["newton"])
def test_rigid_bodies_step_beside_i4h(make_env):
    """F2: with rigid_solver_cfg, Isaac Lab steps MuJoCo-Warp and i4h on one state (under its CUDA graph): a host
    pendulum in every world swings, and i4h's observations equal a run without it."""
    pytest.importorskip("mujoco_warp")
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager

    def host_pendulum(builder, env, _position, _quaternion):
        link = builder.add_link(xform=wp.transform((0.0, 2.0, 1.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_capsule(link, radius=0.05, half_height=0.2)
        j = builder.add_joint_revolute(-1, link, parent_xform=wp.transform((0.0, 2.0, 1.5), wp.quat_identity()),
                                       child_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), axis=(1.0, 0.0, 0.0))
        builder.add_articulation([j])

    def rollout(env):
        env.reset()
        a = torch.zeros((2, 10), device=env.device)
        a[:, 2] = 1.0
        for _ in range(15):
            obs, *_ = env.step(a)
        torch.cuda.synchronize()
        return obs["policy"].clone()

    with newton_builder_world_hook(host_pendulum):  # the host env first: closing an env clears Lab's world hooks
        env = make_env(stagger_episodes=False, rigid_solver_cfg=MJWarpSolverCfg())
    u = env.unwrapped
    assert u.simulator.coupled is not None and NewtonManager._solver is u.simulator.coupled
    state = NewtonManager.get_state_0()
    state.joint_q.assign(np.full(state.joint_q.shape[0], 0.5, np.float32))  # tilt the pendulums
    shared = rollout(env)
    assert NewtonManager._graph is not None, "Lab captured the coupled step"
    body_q = state.body_q.numpy()
    assert np.isfinite(body_q).all() and not np.allclose(body_q[:, 2], 1.0), "the pendulums swing"
    alone = rollout(make_env(stagger_episodes=False))
    assert torch.allclose(shared, alone, atol=1e-5)


def probe_holder(builder, env, position, quaternion):
    """Per world a holder on a stiff vertical slide that lifts two_worlds' probe into the block (F2b tests)."""
    import newton

    frame = wp.transform(wp.vec3(*map(float, position)), wp.quat(*map(float, quaternion)))
    start = frame * wp.transform((0.0, 0.0, -0.05), wp.quat_identity())
    link = builder.add_link(xform=start, mass=1.0, label="holder")
    builder.add_shape_sphere(link, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False))
    j = builder.add_joint_prismatic(-1, link, parent_xform=start, axis=(0.0, 0.0, 1.0), target_ke=1000.0,
                                    target_kd=100.0, target_pos=0.12, armature=0.0)
    builder.add_articulation([j])


@pytest.mark.parametrize("backend", ["newton"])
def test_mounted_probe_two_way_under_lab(make_env):
    """F2b through I4hEnvCfg.mounts: a host holder slides the probe up into the block in every world (Lab's CUDA graph);
    the probe follows the holder in the component frame and the block pushes the holder back down (two_way)."""
    pytest.importorskip("mujoco_warp")
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager

    def rollout(mode):
        with newton_builder_world_hook(probe_holder):
            env = make_env(stagger_episodes=False, rigid_solver_cfg=MJWarpSolverCfg(), coupling_mode=mode,
                           mounts={"probe": {"body": "holder"}})
        env.reset()
        a = torch.zeros((2, 10), device=env.device)
        for _ in range(60):
            env.step(a)
        torch.cuda.synchronize()
        u = env.unwrapped
        assert NewtonManager._graph is not None
        ic = next(s for s in u.simulator.systems if type(s).__name__ == "InstrumentControl")
        mounts = u.simulator.coupled.mounts.groups[0]
        result = ic.root.numpy()[:, 0].copy(), NewtonManager.get_state_0().body_q.numpy()[mounts.body.numpy()[:, 0]].copy()
        env.close()  # here, not inside the next hook context: closing clears Lab's world hooks
        return result

    root_one, body_one = rollout("one_way")
    root_two, body_two = rollout("two_way")
    for root, body in ((root_one, body_one), (root_two, body_two)):
        np.testing.assert_allclose(root[:, :2], 0.0, atol=1e-4)  # the slide is the component's z: env origins taken out
        assert (root[:, 2] > 0.03).all(), "the holder lifted the probe into the block"
    assert (root_two[:, 2] < root_one[:, 2] - 1e-4).all(), "two_way: the block pushes the holder back down"
    assert np.isfinite(body_two).all()



@pytest.mark.parametrize("backend", ["newton"])
def test_physics_reset_clears_the_selected_worlds_pending_reactions(make_env):
    """A masked physics reset (the captured reset graph) clears the reset worlds' queued two_way reactions, so a new
    episode does not push its holder with the last episode's contact; other worlds keep theirs."""
    pytest.importorskip("mujoco_warp")
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import MJWarpSolverCfg

    with newton_builder_world_hook(probe_holder):
        env = make_env(stagger_episodes=False, rigid_solver_cfg=MJWarpSolverCfg(), coupling_mode="two_way",
                       mounts={"probe": {"body": "holder"}})
    env.reset()
    for _ in range(60):
        env.step(torch.zeros((2, 10), device=env.device))
    sim = env.unwrapped.simulator
    assert sim.reset_graph is not None, "the captured reset path"
    group = sim.coupled.mounts.groups[0]
    torch.cuda.synchronize()
    before = group.force.numpy()[:, 0].copy()
    assert (before[:, 2] < -1.0).all(), "both holders carry the block's push"
    sim.reset_physics(env_ids=[0])
    wp.synchronize()
    after, torque = group.force.numpy()[:, 0], group.torque.numpy()[:, 0]
    assert not after[0].any() and not torque[0].any(), "the reset world's reaction is cleared"
    np.testing.assert_array_equal(after[1], before[1])


@pytest.mark.parametrize("backend", ["newton"])
def test_host_contact_sensor_works_in_the_coupled_step(make_env):
    """A workflow robot's Newton contact sensor keeps working when i4h is added: the coupled solver forwards
    update_contacts to MuJoCo-Warp and the manager allocates the solver's contact buffer (MuJoCo's own contacts)."""
    pytest.importorskip("mujoco_warp")
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager

    def ball_on_floor(builder, env, position, quaternion):
        frame = wp.transform(wp.vec3(*map(float, position)), wp.quat(*map(float, quaternion)))
        builder.add_shape_box(-1, xform=frame * wp.transform((0.0, 2.0, 0.0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05,
                              label="floor")
        ball = builder.add_body(xform=frame * wp.transform((0.0, 2.0, 0.2), wp.quat_identity()), mass=1.0, label="ball")
        builder.add_shape_sphere(ball, radius=0.1, label="ball_shape")

    with newton_builder_world_hook(ball_on_floor):
        env = make_env(stagger_episodes=False, rigid_solver_cfg=MJWarpSolverCfg())
    from i4h_isaaclab.newton_manager import I4hNewtonManager

    # A workflow's ContactSensor registers while its scene is built, before the solver exists; the active manager's
    # _initialize_contacts then allocates the buffer after _build_solver. Here the env is built already: same order.
    NewtonManager.add_contact_sensor(body_names_expr=".*ball")
    I4hNewtonManager._initialize_contacts()
    NewtonManager._invalidate_graph()  # recaptured with the sensor
    assert NewtonManager._contacts is not None and NewtonManager._contacts.force is not None
    env.reset()
    for _ in range(30):
        env.step(torch.zeros((2, 10), device=env.device))
    torch.cuda.synchronize()
    sensor = next(iter(NewtonManager._newton_contact_sensors.values()))
    force = sensor.total_force.numpy() if hasattr(sensor, "total_force") else None
    assert int(NewtonManager._contacts.rigid_contact_count.numpy()[0]) > 0, "MuJoCo's contacts reach Lab's buffer"
    if force is not None:
        assert np.abs(force).max() > 1.0, "the ball's weight is reported"


@pytest.mark.parametrize("backend", ["newton"])
def test_physics_reset_reads_the_mount_pose_the_host_just_set(make_env):
    """A host resets or randomizes its holder (joint state) before the component's reset: the reset instrument
    starts at the holder's new pose (forward kinematics and the mount read in the reset itself, captured too); a
    world that is not reset keeps its instrument."""
    pytest.importorskip("mujoco_warp")
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager

    with newton_builder_world_hook(probe_holder):
        env = make_env(stagger_episodes=False, rigid_solver_cfg=MJWarpSolverCfg(), mounts={"probe": {"body": "holder"}})
    env.reset()
    for _ in range(10):
        env.step(torch.zeros((2, 10), device=env.device))
    sim = env.unwrapped.simulator
    assert sim.reset_graph is not None
    ic = next(s for s in sim.systems if type(s).__name__ == "InstrumentControl")
    torch.cuda.synchronize()
    before = ic.root.numpy()[:, 0].copy()
    state = NewtonManager.get_state_0()
    q = state.joint_q.numpy()
    q[0] = q[0] - 0.04  # the host lowers world 0's holder on its slide (joint coordinates)
    state.joint_q.assign(q)
    sim.reset_physics(env_ids=[0])
    wp.synchronize()
    after = ic.root.numpy()[:, 0]
    np.testing.assert_allclose(after[0, 2], before[0, 2] - 0.04, atol=1e-4)  # the reset world reads the new pose
    np.testing.assert_array_equal(after[1], before[1])  # the other world keeps its instrument
