# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import yaml

pytest.importorskip("isaaclab.envs")
import torch  # noqa: E402 - optional dependency guard must run first
from helpers import needs_external_data  # noqa: E402 - optional dependency guard must run first
from i4h_isaaclab.composition import read_case  # noqa: E402 - optional dependency guard must run first

pytestmark = needs_external_data
pytest_plugins = ("test_isaaclab_env",)



def stationary_sequence():
    """aorta_centreline with a three-step reach sequence whose targets enclose the resting catheter tip."""
    cfg = read_case("aorta_centreline")
    rod = cfg["assets"]["catheter"]
    tip = np.add(rod["start"], np.multiply(rod["direction"], (rod["count"] - 1) * rod["segment_length"])).tolist()
    cfg["procedure"] = {
        "timeout": 120,
        "metrics": {"tip_path_length": {"tool": "catheter"}},
        "steps": [{"id": name, "type": "reach_target", "tool": "catheter", "target": {"position": tip}, "radius": 1,
                   "hold": 1 / 60} for name in ("near_landmark", "far_landmark", "return_to_entry")],
    }
    return cfg


@pytest.mark.parametrize("ids", [False, True])
@pytest.mark.parametrize("outcome", ["success", "timeout", "invalid_target"])
def test_weighted_action_terminal_exports(make_env, tmp_path, ids, outcome):
    cfg = stationary_sequence()  # small stationary fixture over the real backend
    if outcome == "timeout":
        cfg["procedure"]["timeout"] = 2 / 60
    if outcome == "invalid_target":
        cfg["procedure"]["steps"][1]["target"] = {"body": "aorta", "index": 3}
    path = tmp_path / "sequence.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = make_env(
        case=str(path),
        num_envs=2,
        observe="catheter",
        decimation=4,
        step_weights=(2.0, 3.0, 7.0),
        compute_final_obs=ids,
        action_mode="physical",
    )
    u = env.unwrapped
    obs, _ = env.reset()
    space = u.single_observation_space
    assert obs["task"].shape == (2, 10)
    assert not u.new_terminal.any()
    if outcome == "invalid_target":
        flags = u.simulator.scene.field("flags", u.simulator.scene.body("aorta"))
        values = flags.numpy()
        values[:, 3] = 0
        flags.assign(values)
    action = torch.zeros((2, 3), device=u.device)
    obs, reward, term, trunc, extras = env.step(action)
    expected = {
        "success": (12, 1, 3),
        "timeout": (5, 2, 2),
        "invalid_target": (2, 3, 1),
    }[outcome]
    assert reward.tolist() == [expected[0]] * 2
    assert (
        term.tolist() == [outcome != "timeout"] * 2
        and trunc.tolist() == [outcome == "timeout"] * 2
    )
    assert extras["new_terminal"].tolist() == [True, True]
    assert extras["terminal_task"][:, 11].tolist() == [expected[1]] * 2
    assert (extras["terminal_history"][:, :, 1] >= 0).sum(1).tolist() == [
        expected[2]
    ] * 2
    assert torch.isfinite(extras["terminal_metrics"]).all()
    assert u.single_observation_space == space
    previous = {
        k: extras[k].clone()
        for k in ("terminal_task", "terminal_history", "terminal_metrics")
    }
    u._pre_physics_step(action)
    assert not u.new_terminal.any() and not u.task_completions.any()
    # A running manual reset cannot synthesize a result or replace retained terminal data.
    u._reset_idx(torch.tensor([0], device=u.device))
    u._save_terminal_task(env_ids=torch.tensor([0], device=u.device))
    assert not u.new_terminal.any()
    for k, v in previous.items():
        assert torch.equal(u.extras[k], v)


def test_repeated_reads_and_one_time_export(make_env, tmp_path):
    cfg = stationary_sequence()
    path = tmp_path / "sequence.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = make_env(case=str(path), num_envs=2, decimation=1, step_weights=(2, 3, 7))
    u = env.unwrapped
    env.reset()
    action = torch.zeros((2, 3), device=u.device)
    for expected in (2, 3):
        _, reward, term, _, _ = env.step(action)
        assert reward.tolist() == [expected] * 2 and not term.any()
        before = u.task_history.clone()
        for _ in range(2):
            assert u._get_rewards().tolist() == [expected] * 2
            u._get_observations()
            u._get_dones()
        assert torch.equal(before, u.task_history)
    u._pre_physics_step(action)
    u.procedure.update(u.physics_dt)
    ids = torch.tensor([0], device=u.device)
    u._save_terminal_task(env_ids=ids)
    assert u.new_terminal.tolist() == [True, False]
    u.new_terminal.zero_()
    u._save_terminal_task(env_ids=ids)
    assert not u.new_terminal.any()
