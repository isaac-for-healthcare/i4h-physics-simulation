# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput of the SOFA replicas over a range of environment counts, on every CPU core.

    source sofa-env.sh   # written by setup_sofa.sh
    $SOFA_PYTHON scripts/benchmark/sofa_scaling.py aorta --scene aorta.npz --envs 1 4 16 64 --json sofa-aorta.json

SOFA steps one scene per thread, so parallel environments are independent SOFA root nodes spread over a pool of
single-threaded worker processes (default: one per physical core). Each worker builds its share of environments,
then all workers run ``--warmup`` frames, meet at a barrier and run ``--frames`` timed frames; a frame advances
every environment of the worker by 1/60 s (8 substeps). Reported per environment count: wall time of the timed
frames, environment frames per second (``envs * frames / wall``), the median and p95 over frames of the slowest
worker's frame time (the time for every environment to advance one frame), resident memory summed over the workers
(current and peak), and whether every final position is finite.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def physical_cores() -> int:
    cores = set()
    for cpu in os.sched_getaffinity(0):
        try:
            sibling = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list").read_text().strip()
        except OSError:
            sibling = str(cpu)
        cores.add(sibling)
    return len(cores)


def rss_mb() -> tuple[float, float]:
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return int(status["VmRSS"].split()[0]) / 1024.0, int(status["VmHWM"].split()[0]) / 1024.0


def worker(scene_name, scene_path, env_ids, envs, warmup, frames, start, ready, queue):
    import numpy as np

    sys.path.insert(0, str(HERE))
    module = __import__(f"sofa_{scene_name}")
    scene = dict(np.load(scene_path))
    batch = [module.Env(scene, e, envs) for e in env_ids]
    ready.wait()
    for _ in range(warmup):
        for env in batch:
            env.frame()
    start.wait()
    times = []
    for _ in range(frames):
        t0 = time.perf_counter()
        for env in batch:
            env.frame()
        times.append(time.perf_counter() - t0)
    queue.put({"times": times, "finite": all(env.finite() for env in batch), "rss": rss_mb()})


def run(scene_name, scene_path, envs, warmup, frames, workers) -> dict:
    import numpy as np

    workers = max(1, min(workers, envs))
    ctx = mp.get_context("spawn")
    ready, start, queue = ctx.Barrier(workers), ctx.Barrier(workers + 1), ctx.Queue()
    shares = [list(range(envs))[w::workers] for w in range(workers)]
    procs = [ctx.Process(target=worker, args=(scene_name, scene_path, share, envs, warmup, frames, start, ready,
                                              queue)) for share in shares]
    for p in procs:
        p.start()
    start.wait()
    t0 = time.perf_counter()
    results = [queue.get() for _ in procs]
    wall = time.perf_counter() - t0
    for p in procs:
        p.join()
    slowest = np.max(np.asarray([r["times"] for r in results]), axis=0) * 1e3
    return {
        "simulator": "sofa", "scene": scene_name, "envs": envs, "workers": workers, "frames": frames,
        "warmup": warmup, "wall_s": round(wall, 3), "env_frames_per_s": round(envs * frames / wall, 2),
        "frame_ms_p50": round(float(np.median(slowest)), 3), "frame_ms_p95": round(float(np.percentile(slowest, 95)), 3),
        "rss_MB": round(sum(r["rss"][0] for r in results), 1), "rss_peak_MB": round(sum(r["rss"][1] for r in results), 1),
        "finite": all(r["finite"] for r in results),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_name", choices=["aorta", "tissue"])
    p.add_argument("--scene", required=True, help="export_scene.py output (.npz)")
    p.add_argument("--envs", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--workers", type=int, default=physical_cores())
    p.add_argument("--json", help="also write the rows to this file")
    args = p.parse_args(argv)
    rows = []
    for envs in args.envs:
        row = run(args.scene_name, str(Path(args.scene).resolve()), envs, args.warmup, args.frames, args.workers)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
