# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local RSL-RL 5.5.1 configuration and terminal-value correction."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PokeRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "i4h_solver_versius"
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[64, 64], activation="elu", obs_normalization=True
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="i4h_isaaclab.versius.manager_ppo:TerminalObservationPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=0.0003,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
