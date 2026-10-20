# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small rsl_rl PPO configurations for physics smoke runs and target procedures."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class I4hPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 16
    max_iterations = 5
    save_interval = 50
    experiment_name = "i4h"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = None  # a value would replace the env's Box; the env clamps to [-1, 1] itself
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class I4hProcedurePPORunnerCfg(I4hPPORunnerCfg):
    obs_groups = {"actor": ["policy", "task"], "critic": ["policy", "task"]}


def runner_dict(agent: I4hPPORunnerCfg) -> dict:
    """The runner's ``train_cfg`` dict for the installed rsl_rl: Isaac Lab's migration drops the deprecated model
    fields (``stochastic``, ``init_noise_std``, ...) that rsl_rl 5.x's ``MLPModel`` no longer accepts."""
    from importlib import metadata

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    return handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-lib")).to_dict()


def checked_iteration(runner, wrapped):
    """One rollout and update through the algorithm API with finiteness asserted at every stage
    (rsl_rl 5.4 ``OnPolicyRunner.learn`` body); returns the loss dict."""
    import torch

    alg, device = runner.alg, runner.device
    obs = wrapped.get_observations().to(device)
    alg.train_mode()
    with torch.inference_mode():
        for _ in range(runner.cfg["num_steps_per_env"]):
            actions = alg.act(obs)
            assert torch.isfinite(actions).all(), "non-finite action from the policy"
            obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.device))
            for group in set(runner.cfg["obs_groups"]["actor"] + runner.cfg["obs_groups"]["critic"]):
                assert torch.isfinite(obs[group]).all(), f"non-finite {group} observation"
            assert torch.isfinite(rewards).all(), "non-finite rollout reward"
            obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    loss_dict = alg.update()
    for name, value in loss_dict.items():
        assert value == value and abs(float(value)) != float("inf"), f"non-finite loss {name}: {value}"
    with torch.inference_mode():
        alg.eval_mode()
        final = alg.get_policy().act_inference(obs) if hasattr(alg.get_policy(), "act_inference") else alg.act(obs)
    assert torch.isfinite(final).all(), "non-finite inference action"
    return {k: float(v) for k, v in loss_dict.items()}
