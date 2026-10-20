# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an i4h task inside a kitless Isaac Lab 3.0 process: a random, random-walk or hold policy with
CUDA-event timing, an optional nsys range, the rsl_rl PPO smoke run, and ``--check`` for the environment.

    $PY scripts/isaaclab_run.py --check
    $PY scripts/isaaclab_run.py --task I4h-Chole-Direct --num_envs 64 --steps 300 --policy random
    $PY scripts/isaaclab_run.py --task I4h-Chole-Direct --num_envs 64 --train 5
    $PY scripts/isaaclab_run.py --task I4h-Chole-Direct --num_envs 64 --steps 300 --camera 128 --save-frame /tmp/chole.png
    $PY scripts/isaaclab_run.py --task I4h-Chole-Direct --num_envs 4 --steps 0 --render gl --policy walk
    $PY scripts/isaaclab_run.py --task I4h-Chole-Direct --num_envs 4 --policy replay:/tmp/session --render gl

``$PY`` is the Isaac Lab venv's python with installed numerical package members (see README). Timing: ``warmup``
untimed steps, a synchronize, then ``steps`` steps each bracketed by CUDA events on the current stream and
one synchronize at the end; median and p95 ms per policy step. ``--sync-debug`` is torch's own detector of
host synchronisations (a correctness aid, not the evidence); ``--nsys-range`` brackets the timed loop with
``cudaProfilerStart/Stop`` for ``nsys --capture-range=cudaProfilerApi``. ``--train N`` runs N PPO learning
iterations of ``num_steps_per_env`` policy steps each, then one explicitly checked rollout and update.
Looking: ``--render gl`` opens the GL debug view (Newton's ViewerGL, mouse camera) on env ``--env`` and refreshes it
after every step (host copies: not for timing; ``--steps 0`` runs until the window closes); ``--save-frame`` writes a
contact sheet of every env's last camera frame; ``--policy replay:<base>`` replays a session recorded by the app
(``i4h-sim <case> --record <base>``) in every env, two recorded physics steps per policy step.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import warp as wp


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="import and version check plus one two_worlds step")
    parser.add_argument("--task", default="I4h-TwoWorlds-Direct", help="a registered task id (i4h_isaaclab.tasks)")
    parser.add_argument("--data", help="production asset directory (overrides I4H_DATA)")
    parser.add_argument("--case", default=None, help="a case name or path instead of --task (default env cfg)")
    parser.add_argument("--backend", choices=("newton", "i4h"), default=None,
                        help="newton (default): Isaac Lab's NewtonManager runs I4hSolver; i4h: the fallback I4hPhysicsManager")
    parser.add_argument("--num_envs", type=int, default=None, help="default 2 (replay: the recorded count)")
    parser.add_argument("--steps", type=int, default=200, help="timed policy steps")
    parser.add_argument("--warmup", type=int, default=20, help="untimed policy steps before the timed loop")
    parser.add_argument("--policy", default="random", help="random | walk | hold | replay:<base> (the app's --record base path) | "
                                                           "replay_hdf5:<file>[:demo_N] (a teleop demo, isaaclab_teleop.py --record)")
    parser.add_argument("--check-replay", action="store_true", help="replay_hdf5: fail when env 0 leaves the recording by more "
                                                                     "than 1e-3 (relative); chaotic cases drift, see the printed curve")
    parser.add_argument("--render", choices=("none", "gl"), default="none", help="gl: live GL debug view of one env")
    parser.add_argument("--env", type=int, default=0, help="env drawn by --render gl")
    parser.add_argument("--seed", type=int, default=None, help="default 0 (replay: the episode's seed)")
    parser.add_argument("--workspace", type=float, default=None, help="override I4hEnvCfg.workspace")
    parser.add_argument("--observe", default=None, help='override I4hEnvCfg.observe ("body.part")')
    parser.add_argument("--camera", type=int, default=None, help="square rgb/depth observation resolution (default 0 = none)")
    parser.add_argument("--no-stagger", action="store_true", help="episodes end together (isolates reset-containing steps)")
    parser.add_argument("--user-stream", action="store_true", help="policy and action tensors on a torch side stream")
    parser.add_argument("--nsys-range", action="store_true", help="cudaProfilerStart/Stop around the timed loop")
    parser.add_argument("--sync-debug", action="store_true", help='torch.cuda.set_sync_debug_mode("warn") in the timed loop')
    parser.add_argument("--train", type=int, default=0, help="PPO learning iterations (rsl_rl) instead of the policy loop")
    parser.add_argument("--log_dir", default=None, help="rsl_rl log directory (tensorboard, checkpoints); none by default")
    parser.add_argument("--save-frame", default=None, help="write a contact sheet of every env's last camera frame (needs --camera)")
    from isaaclab.app.sim_launcher import add_launcher_args

    add_launcher_args(parser)  # --device, --visualizer, ...
    args = parser.parse_args(argv)
    if args.data:
        os.environ["I4H_DATA"] = str(Path(args.data).expanduser().resolve())
    return args


def main(argv=None):
    wp.config.enable_backward = False  # before any kernel module is built (Isaac Lab's scripts do the same)
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.check:
        return check(args)
    from isaaclab.app.sim_launcher import launch_simulation

    cfg = make_cfg(args)
    with launch_simulation(cfg, args):
        import torch

        free0 = torch.cuda.mem_get_info()[0]
        t0 = time.perf_counter()
        env = make_env(args, cfg)
        torch.cuda.synchronize()
        print(f"built ({cfg.backend} backend) in {time.perf_counter() - t0:.2f} s, "
              f"device memory +{(free0 - torch.cuda.mem_get_info()[0]) / 2**20:.0f} MB")
        try:
            if args.train:
                train(env, args)
            elif args.render == "gl":
                view_loop(env, args)
            else:
                if args.policy.startswith("replay_hdf5:"):
                    replay_loop(env, args)
                else:
                    policy_loop(env, args)
        finally:
            env.close()
            clear_context()
    return 0


def make_cfg(args):
    import i4h_isaaclab.tasks as tasks  # registers the ids
    from i4h_isaaclab.env import I4hEnvCfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if args.policy.startswith("replay_hdf5:"):  # the recorded env; only explicitly given options override it
        from i4h_isaaclab.recording import env_cfg_from_recording, load_demo, replay_path

        path, demo = replay_path(args.policy[len("replay_hdf5:"):])
        episode, env_args = load_demo(path, demo, device="cpu")
        overrides = {"case": args.case, "backend": args.backend, "num_envs": args.num_envs, "camera": args.camera,
                     "workspace": args.workspace, "observe": args.observe, "stagger_episodes": False if args.no_stagger else None}
        cfg = env_cfg_from_recording(env_args, overrides, data=args.data,
                                     seed=args.seed if args.seed is not None else getattr(episode, "seed", None))
        cfg.sim.device = args.device or "cuda:0"
        return cfg
    elif args.case:
        cfg = I4hEnvCfg(case=args.case)
    else:
        if args.task not in tasks.TASKS:
            raise SystemExit(f"unknown task {args.task!r}; known: {sorted(tasks.TASKS)}")
        cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = int(args.num_envs if args.num_envs is not None else 2)
    cfg.backend = args.backend or "newton"
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = int(args.seed or 0)
    cfg.camera = int(args.camera or 0)
    cfg.stagger_episodes = not args.no_stagger
    if args.workspace is not None:
        cfg.workspace = float(args.workspace)
    if args.observe is not None:
        cfg.observe = args.observe
    return cfg


def make_env(args, cfg):
    import gymnasium as gym

    if args.case or args.policy.startswith("replay_hdf5:"):  # a replay builds the recorded env, not --task's
        from i4h_isaaclab.env import I4hEnv

        return I4hEnv(cfg)
    return gym.make(args.task, cfg=cfg)


def clear_context():
    from isaaclab.sim import SimulationContext

    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()


def simulator_pins() -> dict[str, str]:
    """The exact versions the installed i4h-physics pins (``name==version`` requirements), so the check follows the
    package instead of repeating its numbers."""
    from importlib.metadata import requires

    pins = {}
    for requirement in requires("i4h-physics") or ():
        spec = requirement.split(";")[0].strip()
        if "==" in spec and "extra" not in requirement:
            name, version = spec.split("==", 1)
            pins[name.split("[")[0].strip().lower()] = version.strip()
    return pins


def check(args):
    import newton
    import torch

    versions = {}
    for name in ("isaaclab", "gymnasium", "rsl_rl", "tensorboard", "pxr", "isaaclab_rl"):
        module = __import__(name)
        versions[name] = getattr(module, "__version__", "ok")
    versions["torch"], versions["warp"], versions["newton"] = torch.__version__, wp.__version__, newton.__version__
    pins = simulator_pins()
    assert wp.__version__ == pins["warp-lang"], f"warp {wp.__version__} != {pins['warp-lang']} (i4h-physics' pin)"
    assert newton.__version__ == pins["newton"], f"newton {newton.__version__} != {pins['newton']} (i4h-physics' pin)"
    assert torch.cuda.is_available(), "torch has no CUDA device"
    args.task, args.case, args.num_envs = "I4h-TwoWorlds-Direct", None, 2
    from isaaclab.app.sim_launcher import launch_simulation

    cfg = make_cfg(args)
    with launch_simulation(cfg, args):
        env = make_env(args, cfg)
        try:
            u = env.unwrapped
            obs, _ = env.reset()
            a = torch.zeros((u.num_envs, u.single_action_space.shape[0]), device=u.device)
            obs, rew, term, trunc, extras = env.step(a)
            assert torch.isfinite(obs["policy"]).all()
            u.simulator.reset([0])
            torch.cuda.synchronize()
        finally:
            env.close()
            clear_context()
    print("check ok:", ", ".join(f"{k} {v}" for k, v in versions.items()), f"device {cfg.sim.device}, backend {cfg.backend}")
    return 0


class Policy:
    """Actions in the normalised [-1, 1] space, sampled on the device; ``replay:<base>`` plays a recorded session."""

    def __init__(self, kind, env, seed):
        import torch
        from i4h_isaaclab.views import replay_actions

        u = env.unwrapped
        self.kind, self.device = kind, torch.device(u.device)
        self.shape = (u.num_envs, u.single_action_space.shape[0])
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.last = torch.zeros(self.shape, device=self.device)  # hold: the reset pose is the zero action
        self.sequence, self.t = None, 0
        if kind.startswith("replay:"):
            rows, excess = replay_actions(u.simulator, kind[len("replay:"):], u.low.cpu().numpy(), u.high.cpu().numpy(),
                                          decimation=int(u.cfg.decimation))
            self.sequence = torch.as_tensor(rows, device=self.device)
            print(f"replay: {rows.shape[0]} policy steps" + (f", clamped by up to {excess:.3f} of the range" if excess > 1e-3 else ""))
        elif kind not in ("random", "walk", "hold"):
            raise SystemExit(f"unknown policy {kind!r}")

    def __call__(self):
        import torch

        if self.sequence is not None:
            self.last = self.sequence[min(self.t, self.sequence.shape[0] - 1)].expand(self.shape).contiguous()
            self.t += 1
        elif self.kind == "random":
            self.last = torch.rand(self.shape, generator=self.gen, device=self.device) * 2.0 - 1.0
        elif self.kind == "walk":
            noise = torch.randn(self.shape, generator=self.gen, device=self.device)
            self.last = (self.last + 0.04 * noise).clamp_(-1.0, 1.0)
        return self.last


def view_loop(env, args):
    """``--render gl``: one env in Newton's ViewerGL, refreshed after every policy step (host copies: not a timing
    run); ``--steps 0`` runs until the window closes."""
    import torch
    from i4h_isaaclab.composition import read_case
    from i4h_standalone.render.gl import GlRenderer

    u = env.unwrapped
    obs, _ = env.reset()
    policy = Policy(args.policy, env, args.seed or 0)
    render_cfg = read_case(u.cfg.case).get("render") or {}
    renderer = GlRenderer(u.simulator, size=tuple(render_cfg.get("size", (1280, 720))), env=int(args.env),
                          camera=render_cfg.get("camera"))
    n = 0
    while renderer.is_running() and (args.steps == 0 or n < args.steps):
        obs, rew, term, trunc, extras = env.step(policy())
        renderer.publish()
        renderer.draw()
        n += 1
    torch.cuda.synchronize()
    renderer.close()
    print(f"{args.case or args.task}: GL view closed after {n} steps, finite={bool(torch.isfinite(obs['policy']).all())}")
    if u.camera is not None and args.save_frame:
        save_frame(u.camera, args.save_frame)


def replay_loop(env, args):
    """``--policy replay_hdf5:FILE[:demo_N]``: the demo's physical actions from a reset; prints how far env 0 drifts from
    the recorded observations (relative to 1 + |recorded|)."""
    import numpy as np
    from i4h_isaaclab.recording import load_demo, replay_episode, replay_path

    path, demo = replay_path(args.policy[len("replay_hdf5:"):])
    episode, env_args = load_demo(path, demo, device=str(env.unwrapped.device))
    diffs = replay_episode(env, episode, env_args)
    first = int(np.argmax(diffs > 1e-3)) if (diffs > 1e-3).any() else None
    marks = ", ".join(f"step {t}: {diffs[t]:.2e}" for t in sorted({0, len(diffs) // 4, len(diffs) // 2, 3 * len(diffs) // 4, len(diffs) - 1}))
    print(f"replay {path.name}:{demo or 'first demo'}: {len(diffs)} steps, max drift {diffs.max():.2e}"
          + (f" (above 1e-3 from step {first})" if first is not None else "") + f"; {marks}")
    if args.check_replay and first is not None:
        raise SystemExit(f"replay drifted above 1e-3 at step {first}")


def policy_loop(env, args):
    import contextlib

    import torch

    if args.steps < 1:
        raise SystemExit("--steps 0 is the GL view's 'until the window closes'; a timing run needs --steps N")
    u = env.unwrapped
    E, dev = u.num_envs, torch.device(u.device)
    obs, _ = env.reset()
    policy = Policy(args.policy, env, args.seed or 0)
    # --user-stream: the policy and env.step run on a torch side stream (as an RL library might), with a few ms of
    # side-stream work in front of every action, so a missing event bridge would feed Warp stale actions.
    side = torch.cuda.Stream(device=dev) if args.user_stream else None
    pad = torch.randn((2048, 2048), device=dev) if side is not None else None

    def act():
        a = policy()
        if side is not None:
            a = a * ((pad @ pad).sum() * 0.0 + 1.0)  # the action now depends on slow side-stream work
        return a

    def stream():
        return torch.cuda.stream(side) if side is not None else contextlib.nullcontext()

    with stream():
        for _ in range(args.warmup):
            env.step(act())
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    if args.nsys_range:
        torch.cuda.profiler.start()
    if args.sync_debug:
        torch.cuda.set_sync_debug_mode("warn")
    with stream():
        for i in range(args.steps):
            a = act()
            torch.cuda.nvtx.range_push("env.step")
            starts[i].record()
            obs, rew, term, trunc, extras = env.step(a)
            ends[i].record()
            torch.cuda.nvtx.range_pop()
    if args.sync_debug:
        torch.cuda.set_sync_debug_mode("default")
    if args.nsys_range:
        torch.cuda.profiler.stop()  # the end-of-loop synchronize below stays outside the profiled range
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    period = int(u.max_episode_length)
    if args.no_stagger:  # every env timed out together at fixed steps; report those steps apart
        first = args.warmup + 1  # steps elapsed before the timed loop (reset() left the buffer at 0)
        reset_steps = [i for i in range(args.steps) if (first + i) % period == 0]
        plain = [t for i, t in enumerate(times) if i not in set(reset_steps)]
        with_reset = [times[i] for i in reset_steps]
        extra = f", reset steps {len(with_reset)} mean {statistics.fmean(with_reset):.3f}" if with_reset else ""
        extra += f", other steps mean {statistics.fmean(plain):.3f}"
    else:
        extra = ""
    med, p95 = statistics.median(times), sorted(times)[min(len(times) - 1, int(0.95 * len(times)))]
    finite = bool(torch.isfinite(obs["policy"]).all())
    line = (f"{args.task if not args.case else args.case}: {E} envs, {args.steps} steps, policy={args.policy}, "
            f"median {med:.3f} ms/step, p95 {p95:.3f}, {E * 1000.0 / med:.0f} env-steps/s, finite={finite}, "
            f"episode ends ~{int(u.common_step_counter * E // period)}{extra}")
    if u.camera is not None:
        cam = u.camera
        shape = wp.to_torch(cam.shape_index).view(torch.int32)  # torch has no CUDA compare for uint32: TRIANGLES -3, PARTICLES -2, MISS -1
        n = shape.numel()
        line += (f", camera hits: tissue {(shape == -3).sum().item() / n:.3f}, shapes {(shape >= 0).sum().item() / n:.3f}, "
                 f"particles {(shape == -2).sum().item() / n:.3f}")
        # the camera alone: its graph replayed on the current state, timed like the steps
        cam_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(50)]
        for s, e in cam_events:
            s.record(torch.cuda.current_stream(dev))
            cam.render()
            e.record(torch.cuda.current_stream(dev))
        torch.cuda.synchronize()
        cam_times = sorted(s.elapsed_time(e) for s, e in cam_events)
        line += f", camera alone median {statistics.median(cam_times):.3f} ms (p95 {cam_times[int(0.95 * len(cam_times))]:.3f})"
        if args.save_frame:
            save_frame(cam, args.save_frame)
    print(line)
    assert finite, "non-finite observation"


def save_frame(camera, path):
    """Every env's last camera frame as one contact sheet, a PNG (Pillow) or, without Pillow, ``<path>.npy``."""
    import math

    import numpy as np

    E = camera.scene.envs
    image = np.ascontiguousarray(camera.sheet(worlds_per_row=max(1, int(math.ceil(math.sqrt(E)))))[..., :3])
    try:
        from PIL import Image

        Image.fromarray(image).save(path)
    except ImportError:
        path = str(path) + ".npy"
        np.save(path, image)
    print(f"wrote {path}")


def train(env, args):
    import torch
    from i4h_isaaclab.agents import I4hPPORunnerCfg, I4hProcedurePPORunnerCfg, checked_iteration, runner_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    u = env.unwrapped
    device = str(u.device)
    wrapped = RslRlVecEnvWrapper(env)  # calls env.reset()
    agent = I4hProcedurePPORunnerCfg() if u.procedure is not None else I4hPPORunnerCfg()
    agent.seed, agent.device = int(args.seed or 0), device
    runner = OnPolicyRunner(wrapped, runner_dict(agent), log_dir=args.log_dir, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=int(args.train), init_at_random_ep_len=True)
    torch.cuda.synchronize()
    per_iteration = (time.perf_counter() - t0) / int(args.train)
    result = checked_iteration(runner, wrapped)
    steps = agent.num_steps_per_env
    print(f"{args.task}: {u.num_envs} envs, PPO {args.train} iterations of {steps} steps: {per_iteration:.2f} s/iteration, "
          f"{steps * u.num_envs / per_iteration:.0f} env-steps/s; checked iteration losses "
          + ", ".join(f"{k} {v:.4g}" for k, v in result.items()))


if __name__ == "__main__":
    sys.exit(main())
