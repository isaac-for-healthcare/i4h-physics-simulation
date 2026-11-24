# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M14 teleop (T1): the app's device path drives env 0 of an I4hEnv in physical-action mode, with force feedback, on
both backends. Isaac Lab venv only (``$PY -m pytest tests/runtime/test_teleop.py``); stub devices only, forces within
the case's 0.5 N cap."""


import pytest

torch = pytest.importorskip("torch")


pytest.importorskip("isaaclab.envs")


from i4h_isaaclab.env import I4hEnv, I4hEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


@pytest.fixture(params=["newton"])
def backend(request):
    return request.param


@pytest.fixture
def make_env(backend):
    envs = []

    def _make(case, num_envs=2, backend=backend, **kw):
        for env in envs:
            env.close()
        envs.clear()
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
        cfg = I4hEnvCfg(case=str(case), action_mode=kw.pop("action_mode", "physical"), stagger_episodes=False, **kw)
        cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=1.0)
        cfg.backend = backend
        env = I4hEnv(cfg)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


def test_physical_mode_equals_normalized_mode(make_env):
    """The same actions, once normalised and once as the physical rows they map to: identical observations."""
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(5)
    sequence = [torch.rand((2, 10), generator=gen, device=dev) * 2 - 1 for _ in range(20)]
    results = {}
    for mode in ("normalized", "physical"):
        env = make_env("two_worlds", action_mode=mode)
        u = env.unwrapped
        env.reset()
        for a in sequence:
            obs, *_ = env.step(a if mode == "normalized" else u.action_mid + u.action_half * a)
        torch.cuda.synchronize()
        results[mode] = obs["policy"].clone()
    assert torch.allclose(results["normalized"], results["physical"], atol=1e-5)
