# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Isaac Lab navigation/reset and PPO checks (both backends), no stubs."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaaclab.envs")
import torch  # noqa: E402 - optional dependency guard must run first
from helpers import needs_external_data  # noqa: E402 - optional dependency guard must run first

pytestmark = needs_external_data

pytest_plugins = ("test_isaaclab_env",)

_qualification = runpy.run_path(str(Path(__file__).parents[2] / "scripts/qualify_procedures.py"))
navigation_action, navigation_steps = _qualification["navigation_action"], _qualification["navigation_steps"]

# Endoluminal reach cases through the generic reach task (the case and observation are configuration).
REACH_CASES = ("aorta_shell_reach",)


@pytest.mark.parametrize("case", REACH_CASES)
def test_navigation_reset_and_live_target(make_env, case):
    env = make_env(task="I4h-ReachTarget-Direct", case=case, observe="catheter", num_envs=2, decimation=1, action_mode="physical")
    u = env.unwrapped
    assert u.cfg.case == case
    obs, _ = env.reset()
    assert u.cfg.observe == "catheter" and obs["task"].shape == (2, 8)
    assert u.obs_views[0].shape[1] == 3*u.simulator.scene.body("catheter").count
    success = False
    for i in range(navigation_steps(case)):
        action = torch.zeros((2, 3), device=u.device)
        action[0] = torch.as_tensor(navigation_action(case, i), device=u.device)
        obs, reward, term, trunc, extras = env.step(action)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["task"]).all()
        assert not term[1] and not trunc.any() and reward[1] == 0
        if u.procedure.index >= 0 and not term[0]:
            u._warp_to_torch()
            target = u.procedure.target_q.numpy()[:, u.procedure.index]
            np.testing.assert_array_equal(u.procedure.rows.numpy()[:, :3], target)
        if term[0]:
            assert reward[0] == 1 and extras["terminal_task"][0, 11] == 1
            success = True
            break
    assert success, u.procedure.results()
    u._warp_to_torch()
    before = u.task_rows[1].clone()
    u._reset_idx(torch.tensor([0], device=u.device))
    assert torch.equal(u.task_rows[1], before) and u.task_rows[0, 9] == 0
    env.reset()
    assert not u.task_rows[:, 9:].any()


@pytest.mark.parametrize("case", REACH_CASES)
def test_ppo_update(make_env, case):
    from i4h_isaaclab.agents import I4hProcedurePPORunnerCfg, checked_iteration, runner_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    env = make_env(task="I4h-ReachTarget-Direct", case=case, observe="catheter", num_envs=8)
    assert env.unwrapped.cfg.case == case
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_dict(I4hProcedurePPORunnerCfg()), log_dir=None, device=env.unwrapped.device)
    losses = checked_iteration(runner, wrapped)
    assert losses and all(np.isfinite(v) for v in losses.values())
