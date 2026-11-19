# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym task ids for the cases: ``I4h-Chole-Direct``, ``I4h-Aorta-Direct``, ``I4h-TwoWorlds-Direct`` and the procedure
tasks, each also as ``-Manager`` (``i4h_isaaclab.manager``: the same case, observations, actions, reward and dones
through Isaac Lab's managers, for hosts that compose manager terms).

Import this module before ``gym.make``; Isaac Lab's own scripts import only ``isaaclab_tasks``, so they see
these ids only when something has imported ``i4h_isaaclab.tasks`` first (``scripts/isaaclab_run.py`` does).
"""

from __future__ import annotations

import gymnasium as gym
from i4h_isaaclab.env import I4hEnvCfg
from i4h_isaaclab.manager import I4hManagerEnvCfg
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


@configclass
class I4hCholeManagerEnvCfg(I4hManagerEnvCfg):
    case: str = "chole_tet_med"
    observe: str = "tissue.gallbladder"


@configclass
class I4hAortaManagerEnvCfg(I4hManagerEnvCfg):
    case: str = "aorta_static"


@configclass
class I4hTwoWorldsManagerEnvCfg(I4hManagerEnvCfg):
    case: str = "two_worlds"


@configclass
class I4hGraspTargetManagerEnvCfg(I4hCholeManagerEnvCfg):
    case: str = "chole_grasp"
    episode_length_s: float = 120.0


@configclass
class I4hReachTargetManagerEnvCfg(I4hAortaManagerEnvCfg):
    case: str = "aorta_static_reach"
    episode_length_s: float = 120.0


TASKS = {"I4h-Chole-Direct": "I4hCholeEnvCfg", "I4h-Aorta-Direct": "I4hAortaEnvCfg", "I4h-TwoWorlds-Direct": "I4hTwoWorldsEnvCfg"}

PROCEDURE_TASKS = {"I4h-GraspTarget-Direct": "I4hGraspTargetEnvCfg",
                   "I4h-ReachTarget-Direct": "I4hReachTargetEnvCfg"}
TASKS.update(PROCEDURE_TASKS)

MANAGER_TASKS = {name.replace("-Direct", "-Manager"): cfg.replace("EnvCfg", "ManagerEnvCfg") for name, cfg in TASKS.items()}

for _id, _cfg in {**TASKS, **MANAGER_TASKS}.items():
    if _id not in gym.registry:
        _runner = "I4hProcedurePPORunnerCfg" if _id.replace("-Manager", "-Direct") in PROCEDURE_TASKS else "I4hPPORunnerCfg"
        _entry = "i4h_isaaclab.manager:I4hManagerBasedRLEnv" if _id in MANAGER_TASKS else "i4h_isaaclab.env:I4hEnv"
        gym.register(
            id=_id,
            entry_point=_entry,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"i4h_isaaclab.tasks:{_cfg}",
                    "rsl_rl_cfg_entry_point": f"i4h_isaaclab.agents:{_runner}"},
        )

VERSIUS_TASKS = {
    f"I4h-Versius-Poke-{name}-v0": name.lower() for name in ("Block", "Chole", "Chole-1x")
}
for _id, _profile in VERSIUS_TASKS.items():
    if _id not in gym.registry:
        gym.register(
            id=_id,
            entry_point="i4h_isaaclab.versius.task:PokeEnv",
            disable_env_checker=True,
            kwargs={"profile_name": _profile,
                    "env_cfg_entry_point": f"i4h_isaaclab.versius.task:{_profile.replace('-', '_')}_cfg",
                    "rsl_rl_cfg_entry_point": "i4h_isaaclab.versius.agents:PokeRunnerCfg"},
        )
