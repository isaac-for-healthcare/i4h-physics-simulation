"""Bounded acceptance workflows. Scripted feasibility is a hard pre-training gate."""

import json

import numpy as np
import torch

from . import terms


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
    if args.mode == "teleop":
        return teleop(env, args)
    return policy(env, args)


def scripted(env, args):
    successes, failures, completed = 0, 0, 0
    history = []
    viewer = Viewer(env, args) if args.viewer else None
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
    from rsl_rl.runners import OnPolicyRunner

    from .agents import PokeRunnerCfg

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    config = to_dict(PokeRunnerCfg())
    if args.mode == "train":
        gate = args.output / "scripted-gate.json"
        gate_record = json.loads(gate.read_text()) if gate.exists() else {}
        if (
            not gate_record.get("passed")
            or gate_record.get("task_metadata") != env.task_metadata
        ):
            raise RuntimeError(
                "Run and pass scripted feasibility in this output directory before PPO"
            )
    runner = OnPolicyRunner(
        wrapped, config, log_dir=str(args.output), device=env.device
    )
    if args.checkpoint:
        metadata = args.checkpoint.with_suffix(".metadata.json")
        if (
            not metadata.exists()
            or json.loads(metadata.read_text()) != env.task_metadata
        ):
            raise ValueError("Checkpoint metadata does not match this task/physics")
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
    obs = wrapped.get_observations()
    for _ in range(args.steps):
        with torch.no_grad():
            action = actor(obs)
        if not torch.isfinite(action).all():
            raise RuntimeError("Nonfinite policy action")
        obs, *_ = wrapped.step(action)
        if not torch.isfinite(obs["policy"]).all():
            raise RuntimeError("Nonfinite playback observation")
    return {
        "checkpoint": str(checkpoint),
        "parameter_update": changed,
        "playback_steps": args.steps,
        "finite": True,
    }


class Viewer:
    """One Newton GL viewer for the native robot and explicitly published i4h mesh."""

    def __init__(self, env, args):
        import warp as wp
        from newton.viewer import ViewerGL

        self.env, self.wp, self.args, self.frame = env, wp, args, 0
        self.viewer = ViewerGL(width=1280, height=720, headless=args.headless)
        self.viewer.show_particles = False
        if not args.headless and not self.viewer.gui.is_available:
            raise RuntimeError(
                "Task UI requires imgui-bundle==1.93.0 in the active environment"
            )
        self.viewer.set_model(env.i4h.sim.model)
        self.viewer.set_camera(wp.vec3(1.6, -1.8, 1.4), -20.0, 125.0)
        body = env.i4h.scene.body(env.physical.body)
        self.faces = wp.array(
            body.tet.tris[0].reshape(-1).astype(np.int32),
            dtype=wp.int32,
            device=env.device,
        )
        self.points = wp.zeros(body.count, dtype=wp.vec3, device=env.device)
        from .status import StatusPanel

        self.status_panel = StatusPanel()
        self.status_panel.focus_tip = self.focus_tip
        self.status_panel.overview = lambda: self.viewer.set_camera(
            wp.vec3(1.6, -1.8, 1.4), -20.0, 125.0
        )
        self.viewer.register_ui_callback(self.status_panel.draw, position="free")
        self.recorder = None
        if args.record:
            import subprocess

            args.record.parent.mkdir(parents=True, exist_ok=True)
            self.recorder = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "1280x720",
                    "-framerate",
                    "30",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "24",
                    "-pix_fmt",
                    "yuv420p",
                    str(args.record),
                ],
                stdin=subprocess.PIPE,
            )

    def focus_tip(self):
        import math

        target = (
            self.env.world_points()[0, self.env.patch_ids]
            .mean(0)
            .detach()
            .cpu()
            .numpy()
        )
        eye = target + np.array((0.22, -0.30, 0.13))
        delta = target - eye
        self.viewer.set_camera(
            self.wp.vec3(*map(float, eye)),
            math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))),
            math.degrees(math.atan2(delta[1], delta[0])),
        )

    def draw(self):
        wp, env = self.wp, self.env
        self.points.assign(env.world_points()[0].detach().cpu().numpy())
        from .status import snapshot

        self.status_panel.state = snapshot(env)
        self.viewer.begin_frame(self.frame * env.step_dt)
        self.viewer.log_state(env.i4h.sim.state)
        self.viewer.log_mesh(
            "i4h/tissue", self.points, self.faces, color=(0.5, 0.12, 0.08)
        )
        from .profiles import PROXY_RADIUS

        for name, position, radius, color in (
            ("proxy", env.tcp_position()[0], PROXY_RADIUS, (0.2, 0.8, 1.0)),
            ("patch", env.reference_patch[0], 0.002, (0.1, 1.0, 0.1)),
            (
                "live_patch",
                env.world_points()[0, env.patch_ids].mean(0),
                0.0015,
                (1.0, 0.7, 0.1),
            ),
        ):
            self.viewer.log_points(
                name,
                wp.array(
                    position[None].detach().cpu().numpy(),
                    dtype=wp.vec3,
                    device=env.device,
                ),
                radii=radius,
                colors=wp.array([color], dtype=wp.vec3, device=env.device),
            )
        for name, value in env.i4h.state().items():
            self.viewer.log_scalar(name, float(value[0].float().norm()))
        self.viewer.end_frame()
        if self.recorder:
            self.recorder.stdin.write(
                self.viewer.get_frame(render_ui=True).numpy().tobytes()
            )
        self.frame += 1

    def close(self):
        if self.recorder:
            self.recorder.stdin.close()
            if self.recorder.wait() != 0:
                raise RuntimeError("OpenGL recording encoder failed")
        self.viewer.close()


def teleop(env, args):
    from .teleop_input import (
        FollouPositionInput,
        KeyboardInput,
        OpenHapticsInput,
        RelativePositionControl,
        ReplayInput,
        TeleopKeys,
    )

    keys = TeleopKeys()
    viewer = Viewer(env, args) if args.viewer else None
    if viewer:
        viewer.status_panel.keys = keys
        viewer.status_panel.input_status = "waiting for first device sample"
    if args.input == "keyboard" and viewer is None:
        raise ValueError("Keyboard requires --viewer")
    sources = {
        "keyboard": lambda: KeyboardInput(keys.held, 0.04),
        "replay": lambda: ReplayInput(args.replay),
        "follou": lambda: FollouPositionInput(args.follou_device_index),
        "openhaptics": lambda: OpenHapticsInput(args.hd_device, args.hd_library),
    }
    import time

    axes = args.axis_map or {"follou": "-x,-z,y", "openhaptics": "x,-z,y"}.get(
        args.input, "x,y,z"
    )
    source, control = (
        sources[args.input](),
        RelativePositionControl(gain=args.gain, axes=axes),
    )
    initial_tcp = env.tcp_position().clone()
    if viewer:
        window = viewer.viewer.renderer.window
        import pyglet

        def code(symbol):
            return {
                pyglet.window.key.SPACE: "Space",
                pyglet.window.key.ESCAPE: "Escape",
                pyglet.window.key.LCTRL: "ControlLeft",
                pyglet.window.key.RCTRL: "ControlRight",
            }.get(symbol, "Key" + pyglet.window.key.symbol_string(symbol).upper())

        window.push_handlers(
            on_key_press=lambda symbol, modifiers: keys.on_key(code(symbol), True),
            on_key_release=lambda symbol, modifiers: keys.on_key(code(symbol), False),
            on_deactivate=keys.focus_lost,
        )
    paused = False
    try:
        count = 0
        while not args.steps or count < args.steps:
            start = time.perf_counter()
            count += 1
            if keys.quit or (viewer and not viewer.viewer.is_running()):
                break
            if keys.reset:
                env.reset()
                control.recenter()
                keys.reset = False
                paused = False
            if keys.recenter:
                control.recenter()
                keys.recenter = False
            from .status import snapshot

            state = snapshot(env)
            if state["failed"]:
                if not paused:
                    print(
                        "Physics paused; reset required: "
                        + "; ".join(state["reasons"]),
                        flush=True,
                    )
                paused = True
            if viewer:
                viewer.status_panel.physics_paused = paused
            if not paused:
                current = env.target_tcp[0].detach().cpu().numpy()
                home = env.home_tcp[0].detach().cpu().numpy()
                sample = source.sample(env.step_dt)
                if viewer:
                    viewer.status_panel.input_status = sample.status
                target = control.target(
                    sample,
                    current,
                    env.step_dt,
                    home - np.array([0.06, 0.06, 0.08]),
                    home + np.array([0.06, 0.06, 0.04]),
                    clutch=keys.clutched,
                )
                action = (
                    torch.tensor(
                        (target - current) / (0.04 * env.step_dt), device=env.device
                    )
                    .float()
                    .repeat(env.num_envs, 1)
                )
                env.step(action)
            if viewer:
                viewer.draw()
            if not args.no_real_time:
                time.sleep(max(0.0, env.step_dt - (time.perf_counter() - start)))
        return {
            "steps": count,
            "episode_steps": env.episode_length_buf.tolist(),
            "tcp_displacement": (env.tcp_position() - initial_tcp).norm(dim=1).tolist(),
            "axis_map": axes,
            "paused_invalid_physics": paused,
            "physical_device_verified": False,
        }
    finally:
        source.close()
        if viewer:
            viewer.close()


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
    viewer = Viewer(env, args) if args.viewer else None
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
