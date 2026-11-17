# SPDX-License-Identifier: Apache-2.0
# Adapted from i4h-v2 training/src/i4h_poking/manager_ppo.py; see NOTICE.
"""Use terminal observations for timeout bootstrapping with the pinned RSL-RL."""

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


class TerminalObservationPPO(PPO):
    """Keep native PPO, correcting its pre-step value bootstrap on timeouts."""

    def process_env_step(self, obs, rewards, dones, extras):
        if (
            "final_obs" in extras
            and "time_outs" in extras
            and extras["time_outs"].any()
        ):
            with torch.no_grad():
                terminal = TensorDict(
                    extras["final_obs"], batch_size=obs.batch_size, device=self.device
                )
                values = self.critic(terminal).detach()
                # Native PPO subsequently adds gamma * V(s_t). Replace that
                # contribution with gamma * V(terminal), while retaining V(s_t)
                # in the stored transition for the normal advantage calculation.
                correction = (values - self.transition.values).squeeze(-1)
                rewards = rewards + self.gamma * correction * extras["time_outs"].to(
                    self.device
                )
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        result = super().update()
        import math

        if any(not math.isfinite(float(value)) for value in result.values()):
            raise RuntimeError("Nonfinite PPO loss")
        for network in (self.actor, self.critic):
            if any(not torch.isfinite(p).all() for p in network.parameters()):
                raise RuntimeError("Nonfinite PPO parameter")
        return result
