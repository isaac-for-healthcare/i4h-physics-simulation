# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The manager-based env (``i4h_isaaclab.manager``) beside I4hEnv: the same case through Isaac Lab's managers, and the
same observations, rewards, dones and terminal extras as the direct env on the same actions (parity). Runs only with
the Isaac Lab venv's python, one env per process at a time (the fixture closes the previous one)."""
import numpy as np
import pytest
import warp as wp
import yaml
from helpers import needs_external_data, needs_vessels

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.envs")
gym = pytest.importorskip("gymnasium")

import i4h_isaaclab.tasks  # noqa: E402, F401  registers the ids
from i4h_isaaclab.composition import read_case  # noqa: E402
from i4h_isaaclab.env import I4hEnv  # noqa: E402
from i4h_isaaclab.manager import I4hManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402


def teardown():
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


@pytest.fixture
def make_env():
    envs = []

    def _make(task="I4h-TwoWorlds-Manager", num_envs=2, **kw):
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        for env in envs:
            env.close()
        envs.clear()
        teardown()
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        cfg.scene.num_envs = num_envs
        if task.endswith("-Direct"):
            cfg.stagger_episodes = False
        for name, value in kw.items():
            setattr(cfg, name, value)
        env = gym.make(task, cfg=cfg)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
    teardown()


def positions(env):
    """The integrated bodies' live positions and their rest positions, ``(E, 3n)``."""
    u = env.unwrapped
    u.i4h.sync_views()
    scene = u.i4h.scene
    body = next(b for b in scene.bodies if b.integrate)
    q = wp.to_torch(scene.view(scene.q, body)).view(scene.envs, -1)
    q0 = wp.to_torch(scene.view(scene.q0, body)).view(scene.envs, -1)
    return q, q0


def target_case(tmp_path, outcome, timeout=0.1):
    """aorta_static_reach with a target every tip reaches (success) or none does (timeout)."""
    cfg = read_case("aorta_static_reach")
    cfg["procedure"].update(radius=100.0 if outcome == "success" else 0.001, target={"position": [0, 0, 0]}, hold=0.1,
                            timeout=timeout)
    path = tmp_path / f"{outcome}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_spaces_and_steps(make_env):
    env = make_env()
    u = env.unwrapped
    assert isinstance(u, I4hManagerBasedRLEnv)
    obs, _ = env.reset()
    A = u.action_manager.total_action_dim
    assert A == 10 and set(obs) == {"policy"}
    assert obs["policy"].shape == u.observation_space["policy"].shape and obs["policy"].dtype == torch.float32
    gen = torch.Generator(device=u.device).manual_seed(0)
    for _ in range(30):
        obs, rew, term, trunc, _ = env.step(torch.rand((2, A), generator=gen, device=u.device) * 2 - 1)
    assert torch.isfinite(obs["policy"]).all() and rew.shape == (2,) and not term.any() and not trunc.any()
    assert abs(u.sim.physics_manager.get_simulation_time() - 30 * u.cfg.decimation * u.physics_dt) < 1e-6


def test_timeout_resets_one_env(make_env):
    env = make_env()
    u = env.unwrapped
    env.reset()
    a = torch.zeros((2, 10), device=u.device)
    a[0, 2] = 1.0  # env 0's probe rises into the block, env 1 holds
    for _ in range(20):
        env.step(a)
    q, q0 = positions(env)
    torch.cuda.synchronize()
    assert not torch.equal(q[0], q0[0]) and not torch.equal(q[1], q0[1])
    u.episode_length_buf[0] = u.max_episode_length - 1
    obs, _, term, trunc, _ = env.step(torch.zeros((2, 10), device=u.device))
    torch.cuda.synchronize()
    assert trunc.tolist() == [True, False] and not term.any() and u.episode_length_buf.tolist() == [0, 21]
    assert torch.equal(q[0], q0[0]) and not torch.equal(q[1], q0[1])
    assert torch.equal(obs["policy"][0, : q.shape[1]], q0[0]), "the observation after the reset is the reset state"


def test_gym_registry():
    for name, cfg in i4h_isaaclab.tasks.MANAGER_TASKS.items():
        spec = gym.spec(name)
        assert spec.entry_point == "i4h_isaaclab.manager:I4hManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"] == f"i4h_isaaclab.tasks:{cfg}"
        direct = gym.spec(name.replace("-Manager", "-Direct"))
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == direct.kwargs["rsl_rl_cfg_entry_point"]


def shift_positions(env, env_ids, offset):
    q, _ = positions(env)
    q[env_ids] = q[env_ids] + offset


@configclass
class ShiftEvents:
    shift = EventTermCfg(func=shift_positions, mode="reset", params={"offset": 0.01})


@needs_vessels
def test_reset_event_reaches_the_procedure(make_env):
    """Physical reset, then the event (on a side stream), then the procedure, which samples the shifted tip."""
    env = make_env(task="I4h-ReachTarget-Manager", observe="catheter", events=ShiftEvents())
    u = env.unwrapped
    dev = torch.device(u.device)
    env.reset()
    for _ in range(5):
        env.step(torch.zeros((2, 3), device=dev))
    with torch.cuda.stream(torch.cuda.Stream(device=dev)):
        u._reset_idx(torch.tensor([0, 1], device=dev))
    torch.cuda.synchronize()
    procedure = u.i4h.procedure
    tips = wp.to_torch(procedure.tip_q).view(2, -1, 3)[:, procedure.tip_index]
    q, q0 = positions(env)
    rest = q0.view(2, -1, 3)[:, -1]
    assert torch.allclose(tips, rest + 0.01, atol=1e-5), "the event moved the catheter"
    assert torch.allclose(wp.to_torch(procedure.previous_tip), tips, atol=1e-6), "the procedure sampled the moved tip"


@pytest.mark.parametrize("outcome", ["success", "timeout"])
@needs_external_data
def test_target_terminal_and_masked_reset(make_env, tmp_path, outcome):
    """A procedure ends inside a 4-step action: reward, dones and the terminal extras of the ended env."""
    env = make_env(task="I4h-ReachTarget-Manager", case=target_case(tmp_path, outcome), decimation=4)
    u = env.unwrapped
    obs, _ = env.reset()
    assert obs["task"].shape == (2, 8)
    action = torch.zeros((2, 3), device=u.device)
    env.step(action)
    u._reset_idx(torch.tensor([0], device=u.device))
    obs, reward, term, trunc, extras = env.step(action)
    assert reward.tolist() == pytest.approx([0, 1] if outcome == "success" else [0, 0])
    assert term.tolist() == ([False, True] if outcome == "success" else [False, False])
    assert trunc.tolist() == ([False, False] if outcome == "success" else [False, True])
    assert extras["terminal_task"][1, 11].item() == (1 if outcome == "success" else 2)
    assert extras["new_terminal"].tolist() == [False, True]
    assert u.i4h_task.rows[1, 9].item() == 0, "env 1 restarted"


def test_hard_reset_rebinds(make_env):
    env = make_env()
    u = env.unwrapped
    env.reset()
    a = torch.zeros((2, 10), device=u.device)
    a[:, 2] = 1.0
    for _ in range(10):
        env.step(a)
    old_sim = u.i4h.sim
    u.sim.reset(soft=False)
    torch.cuda.synchronize()
    q, q0 = positions(env)
    assert u.i4h.sim is not old_sim and torch.equal(q, q0)
    for _ in range(10):
        obs, *_ = env.step(a)
    torch.cuda.synchronize()
    q, q0 = positions(env)
    assert torch.isfinite(obs["policy"]).all() and not torch.equal(q[0], q0[0]), "stepping continues on the new arrays"
    assert torch.equal(obs["policy"][:, : q.shape[1]], q), "the observation reads the rebound arrays"


def test_close_and_reopen(make_env):
    from i4h_isaaclab.newton_manager import I4hNewtonManager

    first = make_env()
    first.reset()
    first.step(torch.zeros((2, 10), device=first.unwrapped.device))
    previous = first.unwrapped.i4h.sim.physics
    env = make_env(num_envs=3)
    assert not previous.valid
    u = env.unwrapped
    assert u.i4h.scene.envs == 3 and len(I4hNewtonManager.on_solver) == 1
    env.reset()
    for _ in range(5):
        obs, *_ = env.step(torch.rand((3, 10), device=u.device) * 2 - 1)
    torch.cuda.synchronize()
    assert torch.isfinite(obs["policy"]).all()


@pytest.mark.parametrize("task", ["I4h-TwoWorlds-Manager", pytest.param("I4h-ReachTarget-Manager", marks=needs_vessels)])
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
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    losses = checked_iteration(runner, wrapped)
    assert losses and all(np.isfinite(v) for v in losses.values()), losses


# --- parity with I4hEnv ------------------------------------------------------------------------------------------


def record(env, actions):
    """Observations, rewards, dones and terminal extras of a fixed action sequence, after a reset."""
    obs, _ = env.reset()
    out = {f"obs.{k}": [v.clone()] for k, v in obs.items()}
    for a in actions:
        obs, rew, term, trunc, extras = env.step(a.clone())
        for k, v in obs.items():
            out[f"obs.{k}"].append(v.clone())
        out.setdefault("reward", []).append(rew.clone())
        out.setdefault("terminated", []).append(term.clone())
        out.setdefault("truncated", []).append(trunc.clone())
        for k in ("terminal_task", "new_terminal"):
            if k in extras:
                out.setdefault(k, []).append(extras[k].clone())
    torch.cuda.synchronize()
    return {k: torch.stack(v).cpu() for k, v in out.items()}


# task, cfg overrides, steps, actions, tolerance of the continuous outputs. Measured run-to-run spread of -Direct alone
# (Lab 6864dd1, 2026-10-29): the rod-only reach rollouts reproduce bitwise; tet tissue does not (float atomics in the
# tet and contact sums): two_worlds held still ~1e-6, chole held still ~0.03, and with random tool contact the spread is
# chaotic (1.4 in positions, tool_force spikes of hundreds), so only the discrete outputs are compared there.
PARITY = {
    "reach_success": ("I4h-ReachTarget", {"case": "success"}, 24, "random", 0.0),
    "reach_timeout": ("I4h-ReachTarget", {"case": "timeout"}, 24, "random", 0.0),
    "two_worlds_hold": ("I4h-TwoWorlds", {"episode_length_s": 0.3}, 24, "hold", 1.0e-5),
    "grasp_hold": ("I4h-GraspTarget", {"episode_length_s": 0.3}, 12, "hold", 0.1),
    "two_worlds_random": ("I4h-TwoWorlds", {"episode_length_s": 0.3}, 24, "random", None),
    "grasp_random": ("I4h-GraspTarget", {"episode_length_s": 0.3}, 12, "random", None),
}


@pytest.mark.parametrize("name", [
    pytest.param("reach_success", marks=needs_vessels),
    pytest.param("reach_timeout", marks=needs_vessels),
    "two_worlds_hold",
    pytest.param("grasp_hold", marks=needs_external_data),
    "two_worlds_random",
    pytest.param("grasp_random", marks=needs_external_data),
])
def test_direct_manager_parity(make_env, tmp_path, name):
    """The same task as -Direct and -Manager on the same actions, through automatic resets (episode-length limits,
    procedure successes and timeouts): the same dones, rewards and terminal flags, and the same observations and
    terminal rows within the -Direct env's own run-to-run spread (PARITY)."""
    prefix, kw, steps, kind, tolerance = PARITY[name]
    kw = dict(kw)
    if kw.get("case") in ("success", "timeout"):
        kw["case"] = target_case(tmp_path, kw["case"], timeout=0.2)
    results = []
    for style in ("Direct", "Manager"):
        env = make_env(task=f"{prefix}-{style}", **kw)
        u = env.unwrapped
        assert isinstance(u, I4hEnv if style == "Direct" else I4hManagerBasedRLEnv)
        A = env.action_space.shape[-1]
        gen = torch.Generator(device=u.device).manual_seed(0)
        actions = [torch.rand((2, A), generator=gen, device=u.device) * 2 - 1 if kind == "random"
                   else torch.zeros((2, A), device=u.device) for _ in range(steps)]
        results.append(record(env, actions))
    direct, manager = results
    assert set(direct) == set(manager), (sorted(direct), sorted(manager))
    assert int((direct["terminated"] | direct["truncated"]).sum()) > 0, "the rollout must cross a reset"
    for key in direct:
        a, m = direct[key], manager[key]
        assert a.shape == m.shape, key
        if a.dtype == torch.bool or key == "reward":
            assert torch.equal(a, m), key
        elif tolerance is None:
            assert torch.isfinite(m).all(), key
        else:
            torch.testing.assert_close(m, a, rtol=0.0, atol=tolerance, msg=lambda text, key=key: f"{key}: {text}")
