# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch side of a procedure for an RL env, shared by ``I4hEnv`` and the manager-based env (no Isaac Lab import).

``ProcedureTorch`` holds zero-copy views of the procedure's rows, completions, progress and history, and the terminal
copies an automatic reset must not erase. The ``task`` observation, the completion reward and the result codes are
computed here once, so both envs give the same values. Stream ordering stays with the caller: read after torch waited
for Warp, and let Warp wait for torch after ``save_terminal``.

Rows: target xyz, approach xyz, qualifying, hold/required, distance, elapsed seconds, consecutive hold seconds, result
(``rows[:, 11]``: 0 running, 1 success, 2 timeout, 3 invalid target)."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

RESULT = 11


class ProcedureTorch:
    def __init__(self, procedure, step_weights, device: torch.device):
        self.procedure = procedure
        self.rows = wp.to_torch(procedure.rows)
        self.completions = wp.to_torch(procedure.action_completions)
        weights = np.asarray(step_weights or (1.0,) * procedure.step_count, dtype=np.float32)
        if weights.shape != (procedure.step_count,) or not np.isfinite(weights).all():
            raise ValueError("step_weights must contain one finite value per procedure step")
        self.step_weights = torch.as_tensor(weights, device=device)
        self.progress = wp.to_torch(procedure.progress)
        self.history = wp.to_torch(procedure.history)
        self.terminal_task = torch.zeros_like(self.rows)
        self.terminal_history = torch.full_like(self.history, -1)
        self.metrics = self.progress[:, 3:4] if procedure.has_path else self.progress[:, 3:3]
        self.terminal_metrics = torch.zeros_like(self.metrics)
        self.new_terminal = torch.zeros(self.rows.shape[0], dtype=torch.bool, device=device)
        self.recorded = torch.zeros_like(self.new_terminal)

    @property
    def observation_size(self) -> int:
        return int(self.procedure.observation_size)

    def begin_action(self, extras: dict) -> None:
        """Start a policy step: the procedure's per-action totals and the terminal notification are cleared."""
        self.procedure.begin_action()
        self.new_terminal.zero_()
        extras["new_terminal"] = self.new_terminal.clone()
        extras["terminal_timeout_source"] = self.procedure.timeout_source
        extras["terminal_timeout_s"] = self.procedure.timeout

    def observation(self) -> torch.Tensor:
        """The ``task`` observation: the first 8 row columns, plus the normalised step and elapsed time of a sequence."""
        obs = self.rows[:, :8].clone()
        if self.procedure.sequence:
            obs = torch.cat((obs, self.progress[:, :1] / max(self.procedure.step_count - 1, 1),
                             self.rows[:, 9:10] / self.procedure.timeout), dim=1)
        return obs

    def reward(self) -> torch.Tensor:
        """The weighted step-completion pulses of the last policy step."""
        return (self.completions * self.step_weights).sum(dim=1)

    def terminated(self) -> torch.Tensor:
        result = self.rows[:, RESULT]
        return (result == 1) | (result == 3)

    def timed_out(self) -> torch.Tensor:
        return self.rows[:, RESULT] == 2

    def save_terminal(self, extras: dict, env_ids=None, mask=None) -> None:
        """Keep the selected envs' terminal rows (first terminal state since their last reset) across the reset."""
        valid = self.rows[:, RESULT] != 0
        if mask is not None:
            selection = mask
        else:
            selection = torch.zeros_like(valid)
            selection[env_ids] = True
        fresh = valid & selection & ~self.recorded
        self.terminal_task.copy_(torch.where(fresh[:, None], self.rows, self.terminal_task))
        self.terminal_history.copy_(torch.where(fresh[:, None, None], self.history, self.terminal_history))
        self.terminal_metrics.copy_(torch.where(fresh[:, None], self.metrics, self.terminal_metrics))
        self.new_terminal.logical_or_(fresh)
        self.recorded.logical_or_(fresh)
        extras["terminal_task"] = self.terminal_task.clone()
        extras["terminal_history"] = self.terminal_history.clone()
        extras["terminal_metrics"] = self.terminal_metrics.clone()
        extras["new_terminal"] = self.new_terminal.clone()
        extras["terminal_timeout_source"] = self.procedure.timeout_source
        extras["terminal_timeout_s"] = self.procedure.timeout

    def clear_recorded(self, env_ids=None, mask=None) -> None:
        """After the selected envs' reset: their next terminal state is recorded again."""
        if mask is not None:
            self.recorded.masked_fill_(mask, False)
        else:
            self.recorded[env_ids] = False
