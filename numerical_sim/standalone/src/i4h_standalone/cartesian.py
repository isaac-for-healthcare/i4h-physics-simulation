"""Cartesian-target teleoperation for the native arm."""
import numpy as np
import torch
from i4h_standalone.render.versius import Viewer


def teleop(env, args):
    from i4h_standalone.input.cartesian import (
        FollouPositionInput,
        KeyboardInput,
        OpenHapticsInput,
        RelativePositionControl,
        ReplayInput,
        TeleopKeys,
    )

    keys = TeleopKeys()
    viewer = source = None
    try:
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
        control = RelativePositionControl(gain=args.gain, axes=axes)
        source = sources[args.input]()
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
        from i4h_standalone.render.timing import LoopTiming
        timing = LoopTiming()
        paused = False
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
            from i4h_isaaclab.versius.status import failure_status

            state = failure_status(env)
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
                current, home = torch.stack((env.target_tcp[0], env.home_tcp[0])).detach().cpu().numpy()
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
                torch.cuda.synchronize(env.device)
            physics_end = time.perf_counter()
            if viewer:
                viewer.draw()
                torch.cuda.synchronize(env.device)
            timing.add(start, physics_end, time.perf_counter(), warmup=count <= 20)
            if not args.no_real_time:
                time.sleep(max(0.0, env.step_dt - (time.perf_counter() - start)))
        print(timing.report())
        return {
            "steps": count,
            "episode_steps": env.episode_length_buf.tolist(),
            "tcp_displacement": (env.tcp_position() - initial_tcp).norm(dim=1).tolist(),
            "axis_map": axes,
            "paused_invalid_physics": paused,
            "physical_device_verified": False,
        }
    finally:
        try:
            if source is not None:
                source.close()
        finally:
            if viewer:
                viewer.close()
