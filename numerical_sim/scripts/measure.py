# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction time, step time and device memory of a case, one case and env count per process.

    uv run python scripts/measure.py two_worlds chole_tet_med aorta --envs 1 64 [--json out.json]

Per case and env count, in a fresh process: ``load_case`` wall time, ``prepare`` (graph capture) wall time,
the median and p95 of the step replay over ``--steps`` steps after ``--warmup`` (``wp.synchronize`` per
step), the Warp mempool's current and high-water bytes after loading and after stepping, and the
``Scene.estimate_bytes`` breakdown of the particle arrays. Numbers are reported, not judged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def measure(case: str, envs: int, steps: int, warmup: int, device: str, navigation=False) -> dict:
    import numpy as np
    import warp as wp
    from i4h_isaaclab.composition import load_case, read_case

    wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
    wp.init()
    dev = wp.get_device(device)
    wp.synchronize()
    base = wp.get_mempool_used_mem_current(dev)
    t0 = time.perf_counter()
    sim = load_case(read_case(case), device=device, envs=envs)
    wp.synchronize()
    t_load = time.perf_counter() - t0
    loaded, loaded_high = wp.get_mempool_used_mem_current(dev) - base, wp.get_mempool_used_mem_high(dev) - base
    t0 = time.perf_counter()
    sim.prepare()
    wp.synchronize()
    t_prepare = time.perf_counter() - t0
    control = actions = None
    if navigation:
        from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
        from i4h_standalone.app.catheter import scripted_rates
        control = next(s for s in sim.systems if isinstance(s, CatheterControl))
        actions = np.zeros((envs, 3), np.float32)
    for i in range(warmup):
        if control is not None:
            control.actions.assign(scripted_rates(envs, i*sim.dt, actions))
        sim.step()
    wp.synchronize()
    times = []
    for i in range(steps):
        if control is not None:
            control.actions.assign(scripted_rates(envs, (warmup+i)*sim.dt, actions))
        t0 = time.perf_counter()
        sim.step()
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1e3
    # Untimed: validate final positions and velocities, including fluid pools.
    finite = {name: bool(np.isfinite(array.numpy()).all()) for name, array in
              (("q", sim.scene.q), ("qd", sim.scene.qd), ("pool_q", sim.scene.pool.q), ("pool_qd", sim.scene.pool.qd))}
    if not all(finite.values()):
        raise RuntimeError(f"nonfinite final state: {finite}")
    return {
        "finite": finite,
        "case": case, "envs": envs, "particles": sim.scene.particle_count,
        "controls": "scripted" if navigation else "zero",
        "containment": {s.id: {"depth": s.contact_depth.numpy().tolist(), "count": s.contact_count.numpy().tolist()}
                        for s in sim.systems if hasattr(s, "contact_depth")},
        "load_s": round(t_load, 3), "prepare_s": round(t_prepare, 3),
        "step_ms_p50": round(float(np.median(times)), 3), "step_ms_p95": round(float(np.percentile(times, 95)), 3),
        "mem_loaded_MB": round(loaded / 1e6, 1), "mem_load_high_MB": round(loaded_high / 1e6, 1),
        "mem_after_steps_MB": round((wp.get_mempool_used_mem_current(dev) - base) / 1e6, 1),
        "mem_high_MB": round((wp.get_mempool_used_mem_high(dev) - base) / 1e6, 1),
        "particle_arrays_MB": {k: round(v / 1e6, 2) for k, v in sim.scene.estimate_bytes().items()},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", nargs="+")
    p.add_argument("--envs", type=int, nargs="+", default=[1, 64])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--navigation", action="store_true", help="exercise scripted catheter controls during timing")
    p.add_argument("--json", help="also write the rows to this file")
    p.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # child: measure cases[0] at envs[0]
    args = p.parse_args(argv)
    if args.one:
        print(json.dumps(measure(args.cases[0], args.envs[0], args.steps, args.warmup, args.device, args.navigation)))
        return 0
    rows = []
    for case in args.cases:
        for envs in args.envs:
            cmd = [sys.executable, __file__, case, "--one", "--envs", str(envs), "--steps", str(args.steps),
                   "--warmup", str(args.warmup), "--device", args.device]
            if args.navigation:
                cmd.append("--navigation")
            out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
            row = json.loads(out.strip().splitlines()[-1])
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
