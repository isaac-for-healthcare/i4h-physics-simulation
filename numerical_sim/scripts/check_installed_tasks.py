# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify every gym ID with Lab, its owning domain and declared dependencies only."""
import sys
from importlib.util import find_spec

import gymnasium as gym
import i4h_isaaclab.tasks as tasks
import torch
from i4h_isaaclab.newton_manager import I4hNewtonManager
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics import NewtonManager

owning = sys.argv[1]
other = "i4h_surgical" if owning == "endoluminal" else "i4h_endoluminal"
assert find_spec(other) is None and find_spec("i4h_standalone") is None
surgical_ids = {"I4h-Chole-Direct", "I4h-TwoWorlds-Direct", "I4h-GraspTarget-Direct"}
selected = [task for task in tasks.TASKS if (task in surgical_ids) == (owning == "surgical")]
for task in selected:
    cfg = getattr(tasks, tasks.TASKS[task])()
    cfg.scene.num_envs = 2
    cfg.sim.device = "cuda:0"
    cfg.backend = "newton"
    env = gym.make(task, cfg=cfg)
    try:
        obs, info = env.reset()
        result = env.step(torch.zeros((2, env.unwrapped.cfg.action_space.shape[0]), device="cuda:0"))
        for group in result[0].values():
            if isinstance(group, torch.Tensor):
                assert torch.isfinite(group).all(), task
        assert torch.isfinite(result[1]).all(), task
        print(f"{task}: registration/configuration/reset/step passed", flush=True)
    finally:
        env.close()
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
        I4hNewtonManager.on_solver.clear()
        I4hNewtonManager.simulator = None
        NewtonManager.clear()
print(f"{len(selected)} installed {owning} gym IDs passed")
