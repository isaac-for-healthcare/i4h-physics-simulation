from types import SimpleNamespace as NS

import torch
from i4h_isaaclab.versius import terms


def fixture():
    zeros = torch.zeros(2, 3)
    env = NS(
        num_envs=2,
        device="cpu",
        step_dt=1 / 30,
        common_step_counter=0,
        cfg=NS(
            teleop=False,
            max_depth=0.012,
            max_tool_depth=0.018,
            target_depth=0.003,
            depth_tolerance=0.0015,
        ),
        i4h=NS(sync_views=lambda: None),
        patch_ids=[0],
        reference_patch=zeros.clone(),
        previous_patch=zeros.clone(),
        target_tcp=zeros.clone(),
        home_tcp=zeros.clone(),
        actions=zeros.clone(),
        previous_actions=zeros.clone(),
        hold_time=torch.zeros(2),
        peak_depth=torch.zeros(2),
        success=torch.zeros(2, dtype=torch.bool),
        failure=torch.zeros(2, dtype=torch.bool),
        progress_step=torch.full((2,), -1),
        ee_id=0,
        episode_length_buf=torch.zeros(2),
        max_episode_length=150,
    )
    env.points = zeros[:, None].clone()
    env.tcp = zeros.clone()
    env.world_points = lambda: env.points
    env.world_velocities = lambda: torch.zeros_like(env.points)
    env.tcp_position = lambda: env.tcp
    env.robot = NS(
        data=NS(
            **{
                name: NS(
                    torch=torch.zeros(2, 1, 3) if "body" in name else torch.zeros(2, 8)
                )
                for name in (
                    "joint_pos",
                    "joint_vel",
                    "body_com_pos_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                )
            }
        )
    )
    return env


def test_command_depth_cannot_succeed():
    env = fixture()
    env.target_tcp[:, 2] = -0.003
    for step in range(20):
        env.common_step_counter = step
        assert not terms.success(env).any()
    assert not env.hold_time.any()


def test_measured_hold_idempotence_and_failure_precedence():
    env = fixture()
    env.points[:, :, 2] = -0.003
    env.tcp[:, 2] = 0.003
    for step in range(15):
        env.common_step_counter = step
        terms.success(env)
        before = env.hold_time.clone()
        for _ in range(3):
            assert terms.observations(env).shape == (2, 24)
            terms.reward(env)
            terms.success(env)
        assert torch.equal(before, env.hold_time)
    assert env.success.all()
    env.tcp[0, 2] = -0.02
    assert not terms.success(env)[0]
    assert terms.failure(env)[0]
    assert env.success[1]


def test_nonfinite_and_distance_fail_without_sanitizing_observation():
    env = fixture()
    env.points[:, :, 2] = -0.003
    env.tcp[:, 2] = 0.03
    assert not terms.measurements(env)[-1].any()
    env.points[0, 0, 0] = float("nan")
    assert terms.failure(env)[0]
    assert not torch.isfinite(terms.observations(env)[0]).all()


def test_timeout_failure_precedence_and_teleop():
    env = fixture()
    env.episode_length_buf[:] = 150
    assert terms.timeout(env).all()
    env.failure[0] = True
    assert not terms.timeout(env)[0]
    env.cfg.teleop = True
    assert not terms.timeout(env).any()
