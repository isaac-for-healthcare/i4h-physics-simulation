# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catheter driver: scripted rates, the keyboard in the Slang window, or nothing -> catheter_control.actions.

``actions (E, 3)`` are normalised rates in ``[-1, 1]`` (insertion, rotation, tip bend) that the system
integrates into targets each step. Modes: ``scripted`` (every env inserts at its own speed,
``(env + 1) / (2 E)`` of the full rate, with a slow rotation and tip-bend swing shared by all envs),
``keyboard`` (the drawn env: I/K insert/retract, J/L rotate, U/O bend the tip; the other envs hold),
``noop`` (zero rates). There is no device backend for a catheter. Isaac Lab bypasses this and writes
``catheter_control.actions`` itself. The keys are the instrument keyboard hub's move keys: a case with
both an instrument and a catheter would drive both from one press (no case has both).
"""

from __future__ import annotations

import math

import numpy as np
from i4h_standalone.input.keyboard import MOVE_KEYS as KEYS
from i4h_standalone.input.keyboard import HeldKeys


def scripted_rates(envs: int, t: float, actions: np.ndarray) -> np.ndarray:
    """``(E, 3)`` rates into ``actions``: insertion ``(env + 1) / (2 E)``, rotation and bend as slow sinusoids."""
    actions[:, 0] = 0.5 * (np.arange(envs, dtype=np.float32) + 1.0) / float(max(envs, 1))  # up to 0.25 m/s: a straight catheter driven faster jams in the branches
    actions[:, 1] = 0.3 * math.sin(2.0 * math.pi * t / 4.0)
    actions[:, 2] = 0.5 * math.sin(2.0 * math.pi * t / 6.0)
    return actions


class CatheterDriver:
    def __init__(self, sim, control, mode: str, renderer=None):
        self.control, self.mode = control, mode
        self.envs = sim.scene.envs
        self.actions = np.zeros((self.envs, 3), np.float32)
        self.renderer = renderer
        self.env = int(getattr(renderer, "env", 0))
        self.held = HeldKeys(KEYS)
        if mode == "keyboard":
            if renderer is None or not hasattr(renderer, "key_handlers"):
                raise ValueError("--input keyboard needs the slang renderer window")
            renderer.key_handlers.append(self.held.handle_key)
        elif mode == "devices":
            raise ValueError("catheter_control has no device backend; use --input scripted, keyboard or noop")
        elif mode not in ("scripted", "noop"):
            raise ValueError(f"unknown input mode {mode!r}")

    def update(self, t: float) -> None:
        if self.mode == "scripted":
            scripted_rates(self.envs, t, self.actions)
        elif self.mode == "keyboard":
            self.env = int(getattr(self.renderer, "env", self.env))  # follow PgUp/PgDn
            self.actions[:] = 0.0
            for name in self.held:  # a copy: the window's key callback mutates the set on another thread
                axis, sign = KEYS[name]
                self.actions[self.env, axis] += sign
        else:
            self.actions[:] = 0.0
        self.control.actions.assign(self.actions)

    def reset(self, env_ids=None) -> None:
        pass  # rates are recomputed from the script or the held keys every update

    def close(self) -> None:
        pass
