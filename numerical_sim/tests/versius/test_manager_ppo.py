"""Verify timeout values come from the terminal state, preserving V(s_t)."""

from types import SimpleNamespace

import torch
from i4h_isaaclab.versius.manager_ppo import TerminalObservationPPO
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


def test_terminal_value_replaces_pre_step_bootstrap(monkeypatch):
    recorded = {}

    def native_step(self, obs, rewards, dones, extras):
        recorded["reward"] = (
            rewards
            + self.gamma * self.transition.values.squeeze(-1) * extras["time_outs"]
        )
        recorded["value"] = self.transition.values.clone()

    monkeypatch.setattr(PPO, "process_env_step", native_step)
    algorithm = TerminalObservationPPO.__new__(TerminalObservationPPO)
    algorithm.device = "cpu"
    algorithm.gamma = 0.98
    algorithm.transition = SimpleNamespace(values=torch.tensor([[2.0], [3.0]]))
    algorithm.critic = lambda obs: obs["policy"][:, :1]
    obs = TensorDict({"policy": torch.zeros(2, 24)}, batch_size=[2])
    final = torch.zeros(2, 24)
    final[:, 0] = torch.tensor([7.0, 11.0])
    extras = {"final_obs": {"policy": final}, "time_outs": torch.tensor([True, False])}
    algorithm.process_env_step(obs, torch.ones(2), torch.ones(2), extras)
    torch.testing.assert_close(recorded["reward"], torch.tensor([1 + 0.98 * 7, 1.0]))
    torch.testing.assert_close(recorded["value"], torch.tensor([[2.0], [3.0]]))


def test_manager_configuration_constructs_terminal_aware_algorithm():
    from i4h_isaaclab.versius.agents import PokeRunnerCfg
    from isaaclab.utils import to_dict

    cfg = to_dict(PokeRunnerCfg())
    cfg["multi_gpu"] = None
    obs = TensorDict({"policy": torch.zeros(2, 24)}, batch_size=[2])
    env = SimpleNamespace(num_actions=3, num_envs=2)
    algorithm = TerminalObservationPPO.construct_algorithm(obs, env, cfg, "cpu")
    assert isinstance(algorithm, TerminalObservationPPO)
    assert "TerminalObservationPPO" in PokeRunnerCfg().algorithm.class_name
