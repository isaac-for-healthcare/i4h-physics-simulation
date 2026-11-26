# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput of an i4h case over a range of environment counts, one fresh process per count.

    uv run python scripts/benchmark/i4h_scaling.py aorta_static --controls navigation --envs 1 4 16 64 --json i4h-aorta.json
    uv run python scripts/benchmark/i4h_scaling.py scripts/benchmark/chole_tissue.yaml --controls random --frames 60

All environments run in one batched GPU simulation. A frame advances every environment by 1/60 s (8 substeps).
After graph preparation and ``--warmup`` frames, each of ``--frames`` frames is timed with ``time.perf_counter()``
around ``sim.step()`` and ``wp.synchronize()``; writing the next actions is outside the interval. Controls:
``zero``; ``navigation`` (``i4h_host.navigation.scripted_rates``, as ``measure.py --navigation``); ``random``
(uniform normalised actions, re-drawn every frame from ``--seed`` and mapped onto each system's action bounds with
Isaac Lab's default 0.5 workspace, as ``export_scene.py`` records them for SOFA). Reported per environment count:
environment frames per second (``envs * 1000 / median frame ms``), median and p95 frame time, Warp memory and the
whole-GPU memory in use after the timed frames, and whether every final position is finite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def measure(case: str, envs: int, frames: int, warmup: int, controls: str, seed: int, device: str) -> dict:
    import numpy as np
    import warp as wp
    from i4h_host.composition import load_case, read_case

    wp.init()
    dev = wp.get_device(device)
    base = wp.get_mempool_used_mem_current(dev)
    sim = load_case(read_case(case), device=device, envs=envs)
    sim.prepare()
    wp.synchronize()
    rng = np.random.default_rng(seed)
    driven = [s for s in sim.systems if hasattr(s, "actions") and hasattr(s, "action_bounds")]
    bounds = [tuple(np.asarray(b, np.float32) for b in s.action_bounds(0.5)) for s in driven]

    def write(i: int) -> None:
        if controls == "navigation":
            from i4h_host.navigation import scripted_rates

            for s in driven:
                s.actions.assign(scripted_rates(envs, i * sim.dt, np.zeros((envs, 3), np.float32)))
        elif controls == "random":
            for s, (low, high) in zip(driven, bounds):
                u = rng.uniform(-1.0, 1.0, size=(envs, low.shape[0])).astype(np.float32)
                s.actions.assign(((low + high) / 2.0 + u * (high - low) / 2.0).reshape(s.actions.shape))

    for i in range(warmup):
        write(i)
        sim.step()
    wp.synchronize()
    times = []
    for i in range(warmup, warmup + frames):
        write(i)
        t0 = time.perf_counter()
        sim.step()
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1e3
    free, total = wp.get_device(device).free_memory, wp.get_device(device).total_memory
    finite = bool(np.isfinite(sim.scene.q.numpy()).all() and np.isfinite(sim.scene.pool.q.numpy()).all())
    p50 = float(np.median(times))
    return {
        "simulator": "i4h", "case": case, "envs": envs, "controls": controls, "frames": frames, "warmup": warmup,
        "particles": sim.scene.particle_count, "env_frames_per_s": round(envs * 1e3 / p50, 2),
        "frame_ms_p50": round(p50, 3), "frame_ms_p95": round(float(np.percentile(times, 95)), 3),
        "warp_MB": round((wp.get_mempool_used_mem_high(dev) - base) / 1e6, 1),
        "gpu_used_MiB": round((total - free) / 2**20, 0), "device": dev.name, "finite": finite,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case")
    p.add_argument("--envs", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--controls", choices=["zero", "navigation", "random"], default="zero")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", help="also write the rows to this file")
    p.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # child: measure at envs[0]
    args = p.parse_args(argv)
    if args.one:
        row = measure(args.case, args.envs[0], args.frames, args.warmup, args.controls, args.seed, args.device)
        print(json.dumps(row))
        return 0
    rows = []
    for envs in args.envs:
        cmd = [sys.executable, __file__, args.case, "--one", "--envs", str(envs), "--frames", str(args.frames),
               "--warmup", str(args.warmup), "--controls", args.controls, "--seed", str(args.seed),
               "--device", args.device]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
