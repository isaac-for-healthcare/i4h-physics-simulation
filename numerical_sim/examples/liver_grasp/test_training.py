# SPDX-License-Identifier: Apache-2.0
"""Contract tests for experimental scoring and adapter, including live CUDA state."""

import copy

import numpy as np
import pytest
import torch
from training_env import Evaluator, LiverEnv, schedule


def m(y=0.0, latch=1, counts=(1, 1, 1, 1, 0, 0), mechanism=(0, 1, 0, 1)):
    return dict(
        latched=latch,
        geometry=np.array([[0.0, y, 0.0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        counts=np.array(counts),
        stats=np.zeros(2),
        mechanism_counts=np.array(mechanism),
    )


def observe(task, sample, n=1, grip=0.6):
    for _ in range(n):
        task.observe(sample, grip, True, True)


def test_physical_full_cycle_and_interrupted_hold():
    t = Evaluator()
    observe(t, m())
    assert t.destination[1] == pytest.approx(0.02)
    observe(t, m(), 29)
    assert t.phase == "qualify"
    observe(t, m())
    assert t.phase == "retract"
    observe(t, m(0.02), 59)
    assert t.phase == "hold"
    observe(t, m(0.015))
    assert t.hold_streak == 0
    observe(t, m(0.02), 60)
    assert t.phase == "release"
    observe(t, m(0.02), 2)  # Still physically held until the command opens.
    assert t.release_step is None
    observe(t, m(0.02, mechanism=(0, 0, 0, 0)), grip=0.0)
    assert t.phase == "relax"
    observe(t, m(0.02, mechanism=(0, 0, 0, 0)), 59, grip=0.0)
    assert not t.done
    observe(t, m(0.02, mechanism=(0, 0, 0, 0)), grip=0.0)
    assert t.phase == "passed"


@pytest.mark.parametrize(
    "sample,reason",
    [
        (m(counts=(1, 0, 0, 0, 0, 1)), "wrong_patch_capture"),
        (m(counts=(3, 3, 3, 1, 0, 0)), "retention_loss"),
    ],
)
def test_acquisition_contract(sample, reason):
    t = Evaluator()
    observe(t, sample)
    assert t.failure == reason


def test_timeout_release_and_relaxation():
    t = Evaluator()
    observe(t, m(latch=0), 1200, grip=0)
    assert t.failure == "timeout"
    t = Evaluator()
    t.phase = "release"
    observe(t, m(latch=0, mechanism=(0, 0, 1, 0)), 30, grip=0)
    assert t.failure == "release_timeout"
    t = Evaluator()
    t.phase = "relax"
    t.phase_step = 0
    t.acquisition_step = 0
    t.acquired_position = np.zeros(3)
    t.destination = np.array([0, 0.02, 0])
    observe(t, m(mechanism=(0, 1, 0, 1)), grip=0)
    assert t.failure == "reacquisition_during_relaxation"


def test_balanced_seed_schedule():
    specs = schedule(1000, 51)
    assert specs == schedule(1000, 51)
    assert len({s["seed"] for s in specs}) == len(specs)
    for i in range(0, 1000, 5):
        assert {s["target_index"] for s in specs[i : i + 5]} == set(range(5))
    assert all(a["target_index"] != b["target_index"] for a, b in zip(specs, specs[1:]))
    assert np.abs([s["jitter_m"] for s in specs]).max() <= 0.005
    assert not {s["seed"] for s in specs} & {s["seed"] for s in schedule(100, 981304)}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from training_env import ROOT

    return LiverEnv(
        str(ROOT.parent.parent / "i4h-isaaclab/numerical_sim/data"),
        2,
        tmp_path_factory.mktemp("liver"),
    )


def test_live_action_observation_timing_and_masked_reset(env):
    from i4h_surgical.grasp_snapshot import continuation_arrays

    env.reset([0, 1])
    initial = env.probe.ic.actions.numpy().copy()
    requested = torch.tensor([[3.0, 4.0, 0.0, 5.0], [0.0, 0.0, 0.0, -1.0]])
    obs, _, _, extra = env.step(requested, auto_reset=False)
    applied = extra["applied"]
    assert np.linalg.norm(applied[0, :3]) == pytest.approx(1.0)
    after = env.probe.ic.actions.numpy()
    np.testing.assert_allclose(
        after[0, 0, :3] - initial[0, 0, :3], applied[0, :3] * 0.001, atol=3e-8
    )
    np.testing.assert_array_equal(after[:, :, 8:], 0.0)
    np.testing.assert_array_equal(after[:, :, 3:7], initial[:, :, 3:7])
    np.testing.assert_allclose(
        obs["policy"][:, :3].numpy(),
        (env.probe.geometry.numpy()[:, 0] - env.probe.ic.root.numpy()[:, 0]) * 100,
    )
    assert all(t.step == 2 for t in env.tasks)
    before = {
        k: v.numpy().copy()
        for k, v in continuation_arrays(
            env.sim.physics, env.sim.state, env.sim.control
        ).items()
    }
    peer = {k: getattr(env.probe, k).numpy()[1].copy() for k in env.probe.fields}
    task = copy.deepcopy(vars(env.tasks[1]))
    teacher = copy.deepcopy(vars(env.teachers[1]))
    env.reset([0])
    after = continuation_arrays(env.sim.physics, env.sim.state, env.sim.control)
    for (_, _, world, *_), (key, value) in zip(
        env.sim.solver.continuation.entries, before.items()
    ):
        mask = world.numpy() == 1
        np.testing.assert_array_equal(after[key].numpy()[mask], value[mask])
    for k, v in peer.items():
        np.testing.assert_array_equal(getattr(env.probe, k).numpy()[1], v)
    for k, v in task.items():
        np.testing.assert_equal(vars(env.tasks[1])[k], v)
    for k, v in teacher.items():
        np.testing.assert_equal(vars(env.teachers[1])[k], v)


def test_ppo_checkpoint_reload(env, tmp_path):
    from rsl_rl.algorithms import PPO
    from train import algorithm_cfg, load_policy, save_checkpoint

    cfg = algorithm_cfg()
    alg = PPO.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    obs = env.get_observations()
    for _ in range(64):
        with torch.inference_mode():
            action = alg.act(obs)
            obs, rewards, dones, extra = env.step(action)
            alg.process_env_step(obs, rewards, dones, extra)
    with torch.inference_mode():
        alg.compute_returns(obs)
    alg.update()
    path = tmp_path / "ppo.pt"
    save_checkpoint(path, alg.actor, alg.optimizer, ppo=alg.save())
    policy = load_policy(env, path)
    torch.testing.assert_close(policy(obs), alg.actor(obs))
    other = PPO.construct_algorithm(obs, env, algorithm_cfg(), "cpu")
    other.load(torch.load(path, weights_only=False)["ppo"], None, True)
    torch.testing.assert_close(other.actor(obs), alg.actor(obs))
    assert other.optimizer.state_dict()["state"]


@pytest.mark.parametrize("capture_step", [1, 2])
def test_adapter_first_acquisition_on_both_physics_steps(
    env, monkeypatch, capture_step
):
    env.reset([0, 1])
    p = env.probe
    original_step = env.sim.step
    calls = 0

    def step():
        nonlocal calls
        original_step()
        calls += 1
        if calls >= capture_step:
            target = int(p.target.numpy()[0])
            sphere = int(p.nearest.numpy()[0])
            for name, value in [
                ("particle", target),
                ("sphere", sphere),
                ("active", 1),
            ]:
                array = getattr(p.volume, name)
                host = array.numpy()
                host[0, 0, 0] = value
                array.assign(host)
            rest = p.volume.rest.numpy()
            rest[0, 0, 0] = (
                env.sim.scene.field("q", p.body).numpy()[0, target]
                - p.ic.sphere_q.numpy()[0, sphere]
            )
            p.volume.rest.assign(rest)

    monkeypatch.setattr(env.sim, "step", step)
    env.step(
        torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, -1.0]]), auto_reset=False
    )
    assert env.tasks[0].acquisition_step == capture_step
    assert env.probe.original_particle.numpy()[0, 0] == int(p.target.numpy()[0])


def test_lift_reward_stops_at_requested_height():
    from training_env import lift_progress_reward

    assert lift_progress_reward(0.019, 0.020) == pytest.approx(0.2)
    assert lift_progress_reward(0.020, 0.030) == 0
    assert lift_progress_reward(0.030, 0.040) == 0
    assert lift_progress_reward(0.0195, 0.040) == pytest.approx(0.1)


def test_wrong_patch_acquisition_has_no_positive_capture_bonus(env, monkeypatch):
    env.reset([0, 1])
    p = env.probe
    original_step = env.sim.step

    def step():
        original_step()
        target = int(p.target.numpy()[0])
        wrong = next(
            v for v in np.flatnonzero(p.eligible) if v not in p.patch_ids[target]
        )
        sphere = int(p.nearest.numpy()[0])
        for name, value in [("particle", wrong), ("sphere", sphere), ("active", 1)]:
            array = getattr(p.volume, name)
            host = array.numpy()
            host[0, 0, 0] = value
            array.assign(host)
        rest = p.volume.rest.numpy()
        rest[0, 0, 0] = (
            env.sim.scene.field("q", p.body).numpy()[0, wrong]
            - p.ic.sphere_q.numpy()[0, sphere]
        )
        p.volume.rest.assign(rest)

    monkeypatch.setattr(env.sim, "step", step)
    _, reward, _, _ = env.step(
        torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, -1.0]]), auto_reset=False
    )
    assert env.tasks[0].failure == "wrong_patch_capture"
    assert (
        reward[0] <= -2.9
    )  # Only the bounded approach term can offset the -3 failure penalty.
