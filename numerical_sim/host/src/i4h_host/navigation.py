# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import numpy as np

def scripted_rates(envs: int, t: float, actions: np.ndarray) -> np.ndarray:
    """``(E, 3)`` rates into ``actions``: insertion ``(env + 1) / (2 E)``, rotation and bend as slow sinusoids."""
    actions[:, 0] = 0.5 * (np.arange(envs, dtype=np.float32) + 1.0) / float(max(envs, 1))  # up to 0.25 m/s: a straight catheter driven faster jams in the branches
    actions[:, 1] = 0.3 * math.sin(2.0 * math.pi * t / 4.0)
    actions[:, 2] = 0.5 * math.sin(2.0 * math.pi * t / 6.0)
    return actions
