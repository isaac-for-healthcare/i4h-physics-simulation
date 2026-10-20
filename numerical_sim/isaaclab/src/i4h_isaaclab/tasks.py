# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym task ids for the cases: ``I4h-Chole-Direct``, ``I4h-Aorta-Direct``, ``I4h-TwoWorlds-Direct``.

Import this module before ``gym.make``; Isaac Lab's own scripts import only ``isaaclab_tasks``, so they see
these ids only when something has imported ``i4h_isaaclab.tasks`` first (``scripts/isaaclab_run.py`` does).
"""

from __future__ import annotations

import gymnasium as gym
from i4h_isaaclab.env import I4hEnvCfg
from isaaclab.utils.configclass import configclass


@configclass
class I4hCholeEnvCfg(I4hEnvCfg):
    case: str = "chole_tet_med"
    observe: str = "tissue.gallbladder"  # 938 x 3 positions instead of the whole tissue's 5612 x 3


@configclass
class I4hAortaEnvCfg(I4hEnvCfg):
    case: str = "aorta_static"


@configclass
class I4hTwoWorldsEnvCfg(I4hEnvCfg):
    case: str = "two_worlds"


@configclass
class I4hGraspTargetEnvCfg(I4hCholeEnvCfg):
    case: str = "chole_grasp"
    stagger_episodes: bool = False
    episode_length_s: float = 120.0


@configclass
class I4hReachTargetEnvCfg(I4hAortaEnvCfg):
    case: str = "aorta_static_reach"
    stagger_episodes: bool = False
    episode_length_s: float = 120.0


TASKS = {"I4h-Chole-Direct": "I4hCholeEnvCfg", "I4h-Aorta-Direct": "I4hAortaEnvCfg", "I4h-TwoWorlds-Direct": "I4hTwoWorldsEnvCfg"}

PROCEDURE_TASKS = {"I4h-GraspTarget-Direct": "I4hGraspTargetEnvCfg",
                   "I4h-ReachTarget-Direct": "I4hReachTargetEnvCfg"}
TASKS.update(PROCEDURE_TASKS)

for _id, _cfg in TASKS.items():
    if _id not in gym.registry:
        _runner = "I4hProcedurePPORunnerCfg" if _id in PROCEDURE_TASKS else "I4hPPORunnerCfg"
        gym.register(
            id=_id,
            entry_point="i4h_isaaclab.env:I4hEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"i4h_isaaclab.tasks:{_cfg}",
                    "rsl_rl_cfg_entry_point": f"i4h_isaaclab.agents:{_runner}"},
        )
