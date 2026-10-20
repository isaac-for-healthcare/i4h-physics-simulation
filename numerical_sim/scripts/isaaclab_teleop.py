# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate an i4h case inside Isaac Lab: the case's input devices drive env 0 of an ``I4hEnv`` (physical actions),
with force feedback from ``tool_force`` when the case's ``haptics:`` block (or ``--haptics``) enables it.

    $PY scripts/isaaclab_teleop.py --case chole_tet_med                       # MiniMous (+ box handles), GL view of env 0
    $PY scripts/isaaclab_teleop.py --case chole_tet_med --fake --render none --steps 600   # no hardware, headless
    $PY scripts/isaaclab_teleop.py --case chole_tet_med --replay /tmp/session    # the app's --record session (.npy per device)
    $PY scripts/isaaclab_teleop.py --case chole_tet_med --replay /tmp/demos.hdf5:demo_0   # a demo recorded here (--record)
    $PY scripts/isaaclab_teleop.py --case chole_tet_med --render xr --haptics   # in the headset: the Quest controllers, vibration
    $PY scripts/isaaclab_teleop.py --case two_worlds --teleop-device isaaclab:my_pkg.devices:MyHaplyCfg:probe --device-mode absolute

The GL window takes the case's pickup keys (``inputs.pickup.keys``); keyboard pose driving stays in the app
(``i4h-sim --input keyboard``). Force limits are the case's YAML caps (``haptics.max_force_n``, ``haptics.handles``).

``--record FILE`` writes env 0's episodes in Isaac Lab's HDF5 format (``isaaclab/recording.py``). Recording starts at
once; in the GL window K saves the take as successful and starts the next, X discards it and starts over, R stops
(the take is saved, unsuccessful) and resumes. A take still open at exit, or cut by env 0's time-out, is saved
unsuccessful (``--success-on-timeout``: successful). ``--replay FILE.hdf5[:demo_N]`` shows a demo in the GL view with
its drift; ``isaaclab_run.py --policy replay_hdf5:FILE[:demo_N] --check-replay`` checks one headless.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import warp as wp


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="production asset directory (overrides I4H_DATA)")
    parser.add_argument("--case", default="chole_tet_med", help="a case name or path")
    parser.add_argument("--backend", choices=("newton", "i4h"), default="newton")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--others", choices=("hold", "mirror"), default="hold", help="what the envs other than 0 do")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fake", action="store_true", help="fake devices at the calibration offsets instead of the hardware")
    source.add_argument("--replay", default=None, metavar="BASE|FILE.hdf5[:demo_N]",
                        help="replay the app's --record session (<BASE>_<device>.npy) or a demo recorded here (--record)")
    source.add_argument("--teleop-device", default=None, metavar="isaaclab:MODULE:CFG[:INSTRUMENT]",
                        help="another Isaac Lab DeviceBase (its cfg class; class_type builds it) drives one instrument "
                             "(default: the first); Se3Keyboard/Se3SpaceMouse need Kit")
    parser.add_argument("--device-mode", choices=("absolute", "delta"), default="absolute",
                        help="--teleop-device output: an absolute [x y z qx qy qz qw buttons..] row or a delta [dx dy dz rx ry rz grip]")
    parser.add_argument("--device-scale", type=float, default=1.0, help="--teleop-device position scale")
    parser.add_argument("--haptics", action=argparse.BooleanOptionalAction, default=None,
                        help="force feedback (default: the case's haptics.enabled; never with --replay)")
    parser.add_argument("--render", choices=("gl", "xr", "none"), default="gl",
                        help="gl: the GL view of env 0; xr: the headset (the case's render.xr, its controllers as inputs; docs/xr.md)")
    parser.add_argument("--steps", type=int, default=0, help="policy steps (0: until the window closes)")
    parser.add_argument("--episode-s", type=float, default=3600.0, help="episode length (env 0 resets at its end)")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None,
                        help="one policy step per step_dt of wall time (default: on with a window)")
    parser.add_argument("--record", default=None, metavar="FILE", help="write env 0's episodes to FILE(.hdf5)")
    parser.add_argument("--record-camera", action="store_true", help="also record the rgb/depth observation (needs --camera)")
    parser.add_argument("--camera", type=int, default=0, help="rgb/depth observation resolution (0: none)")
    parser.add_argument("--success-on-timeout", action="store_true", help="headless: episodes that time out are successful")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.data:
        os.environ["I4H_DATA"] = str(Path(args.data).expanduser().resolve())
    if args.record_camera and args.camera <= 0:
        parser.error("--record-camera needs --camera N")
    if args.render == "none" and args.steps <= 0:
        parser.error("--render none needs --steps > 0")
    if args.teleop_device is not None and (not args.teleop_device.startswith("isaaclab:") or len(args.teleop_device.split(":")) not in (3, 4)):
        parser.error("--teleop-device takes isaaclab:MODULE:CFG[:INSTRUMENT]")
    return args


def isaaclab_device(spec: str, records: list, mode: str, scale: float):
    """``isaaclab:MODULE:CFG[:INSTRUMENT]`` -> (the instrument's input id, an ``IsaacLabDeviceSource`` over the device)."""
    import importlib

    from i4h_standalone.devices import IsaacLabDeviceSource

    parts = spec.split(":")
    cfg = getattr(importlib.import_module(parts[1]), parts[2])()
    device = cfg.class_type(cfg)
    by_name = {r.name: r for r in records}
    record = by_name[parts[3]] if len(parts) == 4 else records[0]
    return record.input or record.name, IsaacLabDeviceSource(device, mode=mode, scale=scale,
                                                             start=(record.initial_position, record.initial_rotation))


def hdf5_replay(spec: str):
    """(file, demo) when ``--replay`` names a demo file (``FILE.hdf5``, or ``BASE`` with ``BASE.hdf5`` and no ``.npy``
    sessions), else None."""
    from pathlib import Path

    from i4h_isaaclab.recording import replay_path

    path, demo = replay_path(spec)
    if path.suffix == ".hdf5":
        return path, demo
    candidate = Path(f"{path}.hdf5")
    if candidate.is_file() and not list(path.parent.glob(f"{path.name}_*.npy")):
        return candidate, demo
    return None


def show_demo(env, renderer, path, demo, realtime: bool) -> None:
    """Step a recorded demo's physical actions from a reset, drawing env 0; prints env 0's drift from the recording."""
    import numpy as np
    import torch
    from i4h_isaaclab.recording import check_replay_compatible, load_demo, restore_episode_length

    u = env.unwrapped
    episode, env_args = load_demo(path, demo, device=str(u.device))
    if env_args.get("case") != u.cfg.case:
        print(f"note: the demo was recorded on case {env_args.get('case')!r}, replaying on {u.cfg.case!r}")
    check_replay_compatible(u, env_args)  # the action layout must match; differing dynamics settings warn
    actions, recorded = episode.data["actions"], episode.data["obs"]["policy"]
    env.reset()
    restore_episode_length(u, episode)
    period, deadline, drift = float(u.step_dt), time.perf_counter(), []
    for t in range(actions.shape[0]):
        if renderer is not None and not renderer.is_running():
            break
        obs, *_ = env.step(actions[t].expand(u.num_envs, -1))
        ref = recorded[t]
        drift.append(((obs["policy"][0] - ref).abs() / (1.0 + ref.abs())).max())
        if renderer is not None:
            renderer.publish()
            renderer.draw()
        if realtime:
            deadline += period
            time.sleep(max(0.0, deadline - time.perf_counter()))
    torch.cuda.synchronize()
    d = torch.stack(drift).cpu().numpy() if drift else np.zeros(1)
    print(f"replayed {path.name}:{demo or 'first demo'}: {len(drift)}/{actions.shape[0]} steps, "
          f"success={bool(episode.success)}, max drift {d.max():.2e} (relative; chaotic cases drift)")


def main(argv=None) -> int:
    wp.config.enable_backward = False
    args = parse(sys.argv[1:] if argv is None else argv)
    from i4h_isaaclab.composition import read_case
    from i4h_isaaclab.env import I4hEnv, I4hEnvCfg
    from i4h_standalone.app.haptics import HapticsConfig
    from i4h_standalone.teleop import Teleop, fake_inputs
    from isaaclab.app.sim_launcher import launch_simulation

    case = read_case(args.case)
    cfg = I4hEnvCfg(case=args.case, backend=args.backend, action_mode="physical", stagger_episodes=False,
                    episode_length_s=float(args.episode_s), seed=int(args.seed), camera=int(args.camera))
    cfg.scene.num_envs = int(args.num_envs)
    cfg.sim.device = args.device
    haptics_cfg = HapticsConfig(**dict(case.get("haptics") or {}))
    if args.haptics is not None:
        haptics_cfg.enabled = bool(args.haptics)
    if args.replay or args.teleop_device:
        haptics_cfg.enabled = False  # a recording, or a device the servo does not own
    with launch_simulation(cfg, args):
        env = I4hEnv(cfg)
        renderer = keys = teleop = recorder = view = None
        try:
            u = env.unwrapped
            demo_file = hdf5_replay(args.replay) if args.replay else None
            if demo_file is not None:  # a demo recorded here: its actions, no devices
                if args.render == "gl":
                    from i4h_standalone.render.gl import GlRenderer

                    render_cfg = case.get("render") or {}
                    renderer = GlRenderer(u.simulator, size=tuple(render_cfg.get("size", (1280, 720))), env=0,
                                          camera=render_cfg.get("camera"))
                show_demo(env, renderer, *demo_file, realtime=args.realtime if args.realtime is not None else renderer is not None)
                return 0
            if args.replay:
                from i4h_standalone.app.device_rows import replay_inputs

                inputs = replay_inputs(u.simulator.scene.instruments.values(), args.replay)
                if not inputs:
                    raise SystemExit(f"--replay {args.replay}: neither a demo file (.hdf5) nor an app session "
                                     f"(<base>_<device>.npy for this case's instruments)")
            elif args.teleop_device:
                records = list(u.simulator.scene.instruments.values())
                dev, external = isaaclab_device(args.teleop_device, records, args.device_mode, args.device_scale)
                inputs = {dev: {"backend": "noop"}}
            elif args.render == "xr" and not args.fake and ((case.get("render") or {}).get("xr") or {}).get("inputs"):
                inputs = case["render"]["xr"]["inputs"]  # the controllers replace the case's desk devices
            else:
                inputs = fake_inputs(case.get("inputs")) if args.fake else case.get("inputs")
            if args.render == "xr":
                from i4h_standalone.render.xr import XrRenderer
                from i4h_standalone.teleop import XrView

                render_cfg = {k: v for k, v in (case.get("render") or {}).items() if k not in ("backend", "size", "xr", "env")}  # teleop is env 0
                renderer = XrRenderer(u.simulator, xr=(case.get("render") or {}).get("xr"), env=0, **render_cfg)
                view = XrView(env, renderer)
                keys = renderer  # the mirror window's keys: the pickup keys and the recording keys
            elif args.render == "gl":
                from i4h_standalone.render.gl import GlKeys, GlRenderer

                render_cfg = case.get("render") or {}
                renderer = GlRenderer(u.simulator, size=tuple(render_cfg.get("size", (1280, 720))), env=0,
                                      camera=render_cfg.get("camera"))
                keys = GlKeys(renderer)
            teleop = Teleop(env, inputs, haptics_cfg, others=args.others, renderer=keys)
            if args.teleop_device:
                teleop.devices.replace_source(dev, external)
            obs = teleop.reset()
            if view is not None:
                view.start()
            commands = []  # GL keys -> recorder commands, applied between steps
            if args.record:
                from i4h_isaaclab.recording import Recorder

                recorder = Recorder(args.record, teleop, record_camera=args.record_camera)
                if keys is not None:
                    from i4h_isaaclab.camera.camera import key_event

                    def on_key(event):
                        name, press, repeat, _ = key_event(event)
                        if press and not repeat and name in ("r", "k", "x"):
                            commands.append(name)

                    keys.key_handlers.append(on_key)
                    print("recording (take 1 started): K save as successful + next take, X discard + restart, R stop/resume"
                          " (each new take starts from a reset)")
                recorder.start(reset=False)  # teleop.reset() just ran
            realtime = args.realtime if args.realtime is not None else renderer is not None
            period, deadline = float(u.step_dt), time.perf_counter()
            resets = 0
            while (args.steps <= 0 or teleop.steps < args.steps) and (
                    renderer is None or (view.running if view is not None else renderer.is_running())):
                obs, done = teleop.step()
                if view is not None:
                    view.check()
                    view.publish(teleop.steps * float(u.step_dt))
                resets += int(done)
                if recorder is not None:
                    recorder.record(obs)
                    if done and recorder.recording:  # env 0's episode ended: the take ends with it, the next begins
                        print(f"recorded {recorder.end(args.success_on_timeout)} (env 0's episode ended)")
                        recorder.start(reset=False)  # Isaac Lab reset env 0 already
                    while commands:
                        c = commands.pop(0)
                        if c == "k" and recorder.recording:
                            print(f"recorded {recorder.end(True)} (successful); next take started")
                            recorder.start()
                        elif c == "x" and recorder.recording:
                            recorder.discard()
                            recorder.start()
                            print("take discarded; recording again")
                        elif c == "r" and recorder.recording:
                            print(f"recorded {recorder.end(False)} (stopped, unsuccessful); R resumes")
                        elif c == "r":
                            recorder.start()
                            print("recording resumed")
                if renderer is not None and view is None:
                    renderer.publish()
                    renderer.draw()
                if realtime:
                    deadline += period
                    remaining = deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    elif remaining < -period:
                        deadline = time.perf_counter()
            if view is not None:
                view.check()  # the loop also ends when the render thread failed: report it, not a clean run
            import torch

            torch.cuda.synchronize()
            finite = bool(torch.isfinite(obs["policy"]).all())
            print(f"{args.case}: {teleop.steps} teleop steps, {u.num_envs} envs, backend {args.backend}, env 0 resets {resets}, "
                  f"finite={finite}")
        finally:
            if recorder is not None:
                recorder.close()
                print(f"demos: {', '.join(recorder.written) or 'none'} in {recorder.path}"
                      + ("" if recorder.written else " (the takes were discarded or empty)"))
            if teleop is not None:
                teleop.close()
                if teleop.haptics is not None and teleop.haptics.final_status:
                    print("haptics: " + teleop.haptics.final_status.replace("\n", "; "))
            if view is not None:
                view.close()  # the render thread closes the XR renderer; raises (the env is not freed) if it hangs
            elif renderer is not None:
                renderer.close()
            env.close()
        if view is not None:
            view.check()  # an error the render thread hit while closing
    return 0


if __name__ == "__main__":
    sys.exit(main())
