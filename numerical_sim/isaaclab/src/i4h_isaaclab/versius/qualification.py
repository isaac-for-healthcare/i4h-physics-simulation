# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded acceptance workflows. Scripted feasibility is a hard pre-training gate."""

import json

import numpy as np
import torch

from . import terms


def check_task_metadata(task, saved, label):
    """Keep checkpoint/gate compatibility explicit, including the reason for a mismatch."""
    different = sorted(key for key in task.keys() | saved.keys() if task.get(key) != saved.get(key))
    if different:
        details = "; ".join(f"{key}: task={task.get(key)!r}, saved={saved.get(key)!r}" for key in different)
        raise ValueError(f"{label} metadata does not match this task/physics: {details}")


def execute(env, args):
    if args.mode == "mechanics":
        return mechanics(env, args)
    if args.mode == "lifecycle":
        return lifecycle(env, args)
    if args.mode == "construct":
        return {
            "constructed": True,
            "finite": bool(torch.isfinite(terms.observations(env)).all()),
            "tcp": env.tcp_position().tolist(),
            "baseline": env.reference_patch.tolist(),
            "diagnostics": {k: v.tolist() for k, v in env.i4h.state().items()},
        }
    if args.mode == "scripted":
        return scripted(env, args)
    return policy(env, args)


def scripted(env, args):
    successes, failures, completed = 0, 0, 0
    history = []
    viewer = args.viewer_factory(env, args) if args.viewer else None
    try:
        while completed < args.episodes:
            patch, tcp, depth, _, _, bad, _ = terms.measurements(env)
            desired = patch.clone()
            # Feedback on measured indentation. Advance gently into contact.
            desired[:, 2] = (
                env.target_tcp[:, 2] - (0.003 - depth).clamp(-0.001, 0.001) * 0.35
            )
            desired[:, :2] = patch[:, :2]
            action = (desired - env.target_tcp) / (0.04 * env.step_dt)
            obs, reward, terminated, truncated, extras = env.step(action.clamp(-1, 1))
            if not torch.isfinite(obs["policy"]).all():
                raise RuntimeError("Nonfinite scripted observations")
            history.append(
                {
                    "tcp": tcp.tolist(),
                    "depth": depth.tolist(),
                    "contact": {k: v.tolist() for k, v in env.i4h.state().items()},
                    "bad": bad.tolist(),
                }
            )
            done = terminated | truncated
            if done.any():
                episode = env.last_episode
                successes += int(episode["success"].sum())
                failures += int(episode["failure"].sum())
                completed += int(done.sum())
            if viewer:
                viewer.draw()
        result = {
            "episodes": completed,
            "successes": successes,
            "failures": failures,
            "passed": successes / completed >= 0.9,
            "profile": env.physical.name,
            "coupling": args.coupling,
            "graph": args.graph,
            "task_metadata": env.task_metadata,
        }
        (args.output / "trajectory.json").write_text(json.dumps(history) + "\n")
        (args.output / "scripted-gate.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result
    finally:
        if viewer:
            viewer.close()


def policy(env, args):
    from isaaclab.utils import to_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    config = to_dict(load_cfg_from_registry(env.spec.id, "rsl_rl_cfg_entry_point"))
    if args.mode == "train":
        gate = args.output / "scripted-gate.json"
        gate_record = json.loads(gate.read_text()) if gate.exists() else {}
        if not gate_record.get("passed"):
            raise RuntimeError(
                "Run and pass scripted feasibility in this output directory before PPO"
            )
        check_task_metadata(env.task_metadata, gate_record.get("task_metadata", {}), "Scripted gate")
    runner = OnPolicyRunner(
        wrapped, config, log_dir=str(args.output), device=env.device
    )
    if args.checkpoint:
        metadata = args.checkpoint.with_suffix(".metadata.json")
        if not metadata.exists():
            raise ValueError(f"Checkpoint metadata sidecar is missing: {metadata}")
        check_task_metadata(env.task_metadata, json.loads(metadata.read_text()), "Checkpoint")
        runner.load(str(args.checkpoint))
    if args.mode == "train":
        before = [p.detach().clone() for p in runner.alg.actor.parameters()]
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
        changed = any(
            not torch.equal(a, b) for a, b in zip(before, runner.alg.actor.parameters())
        )
        if not changed:
            raise RuntimeError("PPO did not update parameters")
        checkpoint = args.output / f"i4h_solver_versius_{env.physical.name}_24obs.pt"
        runner.save(str(checkpoint))
        checkpoint.with_suffix(".metadata.json").write_text(
            json.dumps(env.task_metadata, indent=2) + "\n"
        )
        runner.load(str(checkpoint))
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for playback")
        checkpoint = args.checkpoint
        changed = None
    actor = runner.get_inference_policy(device=env.device)
    viewer = args.viewer_factory(env, args) if args.viewer else None
    obs = wrapped.get_observations()
    try:
        for _ in range(args.steps):
            with torch.no_grad():
                action = actor(obs)
            if not torch.isfinite(action).all():
                raise RuntimeError("Nonfinite policy action")
            obs, *_ = wrapped.step(action)
            if viewer:
                viewer.draw()
            if not torch.isfinite(obs["policy"]).all():
                raise RuntimeError("Nonfinite playback observation")
    finally:
        if viewer:
            viewer.close()
    return {
        "checkpoint": str(checkpoint),
        "parameter_update": changed,
        "playback_steps": args.steps,
        "finite": True,
    }


def lifecycle(env, args):
    """Selected reset preserves every peer's visible state, commands and pending reaction."""
    import warp as wp

    for _ in range(80):
        patch, tcp, depth, *_ = terms.measurements(env)
        actions = torch.zeros_like(env.actions)
        actions[:, 2] = -0.3
        env.step(actions)
    env.i4h.sync_views()
    binding = env.i4h.sim
    q = env.i4h.positions(env.physical.body).clone()
    body = binding.scene.body(env.physical.body)
    history = {
        name: binding.scene.field(name, body).numpy().copy()
        for name in ("qd", "q_prev")
    }
    products = {name: value.clone() for name, value in env.i4h.state().items()}
    controller = {
        name: getattr(env, name).clone()
        for name in (
            "actions",
            "previous_actions",
            "hold_time",
            "peak_depth",
            "progress_step",
        )
    }
    joint_velocity = env.robot.data.joint_vel.torch.clone()
    commands = [x.clone() for x in env.i4h.action_views()]
    force_array = binding.state.i4h.instrument_control_mount_force
    force = force_array.numpy().copy()
    if not (np.linalg.norm(force.reshape(env.num_envs, -1), axis=1) > 0).all():
        raise AssertionError("lifecycle fixture needs pending contact in every world")
    joint = env.robot.data.joint_pos.torch.clone()
    target = env.target_tcp.clone()
    env._reset_idx(torch.tensor([0], device=env.device))
    if env.num_envs > 1:
        assert torch.equal(env.i4h.positions(env.physical.body)[1:], q[1:])
        assert torch.equal(env.robot.data.joint_pos.torch[1:], joint[1:])
        assert torch.equal(env.robot.data.joint_vel.torch[1:], joint_velocity[1:])
        for name, before in history.items():
            np.testing.assert_array_equal(
                binding.scene.field(name, body).numpy()[1:], before[1:]
            )
        for name, before in products.items():
            assert torch.equal(env.i4h.state()[name][1:], before[1:])
        for name, before in controller.items():
            assert torch.equal(getattr(env, name)[1:], before[1:])
        assert torch.equal(env.target_tcp[1:], target[1:])
        for current, before in zip(env.i4h.action_views(), commands):
            assert torch.equal(current[1:], before[1:])
        np.testing.assert_array_equal(force_array.numpy()[1:], force[1:])
    saved_q = (
        binding.scene.field("q", binding.scene.body(env.physical.body))
        .numpy()[0]
        .copy()
    )
    env.i4h.restore(env.saved_state, env.saved_control, [0])
    np.testing.assert_array_equal(
        binding.scene.field("q", binding.scene.body(env.physical.body)).numpy()[0],
        saved_q,
    )
    handle = binding
    env.hard_rebuild()
    try:
        binding.physics.check_valid()
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale binding accepted after rebuild")
    try:
        handle.command_views("instrument_control")
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale command handle accepted after rebuild")
    env.step(torch.zeros_like(env.actions))
    assert torch.isfinite(terms.observations(env)).all()
    wp.synchronize()
    return {
        "selected_reset": True,
        "peer_state_commands_reactions_preserved": True,
        "hard_rebuild": True,
        "stale_handles_rejected": True,
    }


def mechanics(env, args):
    """Identical open-loop Cartesian commands for mode/reaction comparisons."""
    traces = []
    viewer = args.viewer_factory(env, args) if args.viewer else None
    try:
        for step in range(160):
            action = torch.zeros_like(env.actions)
            action[:, 2] = -0.5 if step < 42 else (0.5 if step >= 100 else 0.0)
            if env.num_envs > 1:
                action[1:] = 0
            if args.no_contact:
                action[:] = 0
            env.step(action)
            patch, tcp, depth, lateral, distance, bad, _ = terms.measurements(env)
            if bad.any():
                raise RuntimeError("Invalid physics in approach/hold/withdrawal")
            diagnostics = env.i4h.state()
            traces.append(
                {
                    "tcp": tcp.tolist(),
                    "patch": patch.tolist(),
                    "depth": depth.tolist(),
                    "force": diagnostics["tool_force.force"].tolist(),
                    "count": diagnostics["tool_force.count"].tolist(),
                }
            )
            if viewer:
                viewer.draw()
        np.savez_compressed(
            args.output / "state.npz",
            q=env.i4h.positions(env.physical.body).cpu().numpy(),
            joints=env.robot.data.joint_pos.torch.cpu().numpy(),
        )
        (args.output / "mechanics.json").write_text(json.dumps(traces) + "\n")
        return {
            "finite": True,
            "steps": 160,
            "peak_depth": max(row["depth"][0] for row in traces),
            "peak_reaction": max(
                float(np.linalg.norm(row["force"][0])) for row in traces
            ),
            "contact_steps": sum(row["count"][0][0] > 0 for row in traces),
        }
    finally:
        if viewer:
            viewer.close()
