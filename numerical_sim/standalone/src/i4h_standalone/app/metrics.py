# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timings of the standalone app: named scopes, percentiles, a console table and a JSON file.

The physics worker records per step ``pace`` (waiting for the real-time deadline), ``input`` (the drivers),
``submit`` (the host time of ``sim.step()``: launches or one graph launch), ``publish`` (the snapshot copy, mostly
the wait for the previous step's GPU work), ``interval`` (wall time between step starts) and ``gpu`` (the step's
device time between two CUDA events, read one step late without a stall). The render loop records ``frame.age``
(a new snapshot's age when acquired), ``frame.publish``, ``frame.draw`` and ``frame.pace``. The first ``warmup``
samples of every scope are dropped (module load, first captures, texture uploads). omnisurg's ``perf.py`` kept
count, mean and max per scope and printed once a second; ours keeps the samples and reports percentiles at exit.

``profile(sim, steps)`` is the eager per-kernel profile: Warp's activity timing over ``steps`` uncaptured steps,
aggregated per kernel and attributed to the module that defines it (a system's kernels live in its module;
``tet.apply_average`` used by the grasps shows under ``tet``; a kernel name defined in several modules lists them
all, ``a|b``). Graph replays cannot be attributed this way.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

PERCENTILES = (50, 95, 99)


class Timings:
    def __init__(self, maxlen: int = 20000, warmup: int = 60):
        self.samples: dict[str, collections.deque] = {}
        self.seen: dict[str, int] = {}  # samples offered per scope, warm-up included
        self.maxlen, self.warmup = int(maxlen), int(warmup)

    def add(self, name: str, seconds: float) -> None:
        n = self.seen.get(name, 0) + 1
        self.seen[name] = n
        if n <= self.warmup:
            return
        if name not in self.samples:
            self.samples[name] = collections.deque(maxlen=self.maxlen)
        self.samples[name].append(float(seconds))

    def summary(self, names=None) -> dict[str, dict]:
        """Per scope (``names``: a subset; default all): count, p50, p95, p99, max and mean in ms (warm-up excluded)."""
        out = {}
        for name, samples in list(self.samples.items()):  # the worker appends and adds scopes meanwhile: snapshot both
            if names is not None and name not in names:
                continue
            values = list(samples)  # one C-level copy of the deque, not an iteration it can invalidate
            if not values:
                continue
            a = np.asarray(values, dtype=np.float64) * 1e3
            p = np.percentile(a, PERCENTILES)
            out[name] = {"count": int(a.size), "p50": float(p[0]), "p95": float(p[1]), "p99": float(p[2]),
                         "max": float(a.max()), "mean": float(a.mean())}
        return out

    def brief(self, names) -> str:
        """One line per scope, ``name p50/p95 ms``, for the on-screen block; scopes without samples are skipped."""
        summary = self.summary(names)
        return "\n".join(f"{name:<14}{s['p50']:7.2f} /{s['p95']:7.2f} ms" for name in names if (s := summary.get(name)))

    def table(self, header: dict) -> str:
        lines = ["timings (ms; warm-up of %d samples per scope excluded)" % self.warmup]
        lines += [f"  {k}: {v}" for k, v in header.items()]
        lines.append(f"{'scope':<16}{'count':>7}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}{'mean':>10}")
        for name, s in sorted(self.summary().items()):
            lines.append(f"{name:<16}{s['count']:>7d}{s['p50']:>10.3f}{s['p95']:>10.3f}{s['p99']:>10.3f}{s['max']:>10.3f}{s['mean']:>10.3f}")
        return "\n".join(lines)

    def write(self, path, header: dict) -> None:
        Path(path).write_text(json.dumps({"run": header, "scopes": self.summary(), "unit": "ms"}, indent=1))


def kernel_modules() -> dict[str, str]:
    """Kernel function name -> defining module, for every ``wp.Kernel`` in the loaded numerical-package modules."""
    found: dict[str, set] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(("i4h_physics.", "i4h_surgical.", "i4h_endoluminal.", "i4h_assessment.", "i4h_isaaclab.", "i4h_standalone.")) or module is None:
            continue
        for obj in vars(module).values():
            if isinstance(obj, wp.Kernel) and obj.module is not None:
                found.setdefault(obj.key, set()).add(obj.module.name)
    return {key: "|".join(sorted(mods)) for key, mods in found.items()}  # a name defined in several modules lists them all


def profile(sim, steps: int, top: int = 40) -> str:
    """Eager per-kernel GPU time over ``steps`` steps, attributed to the defining module. Leaves the simulator eager."""
    sim.cuda_graph = False
    sim.prepare()
    wp.synchronize_device(sim.scene.device)
    wp.timing_begin(wp.TIMING_KERNEL)
    for _ in range(int(steps)):
        sim.step()
    results = wp.timing_end()
    modules = kernel_modules()
    keys = sorted(modules, key=len, reverse=True)  # longest match first: `deltas` must not claim `fluid_deltas`
    rows: dict[tuple[str, str], list] = {}
    for r in results:
        name = r.name.split("kernel ", 1)[-1]  # "forward kernel solve_volume_36778518"
        key = next((k for k in keys if name.startswith(k + "_") or name == k), name.rsplit("_", 1)[0])
        row = rows.setdefault((modules.get(key, "?"), key), [0, 0.0])
        row[0] += 1
        row[1] += float(r.elapsed)
    total = sum(v[1] for v in rows.values()) or 1.0
    lines = [f"eager kernel profile over {steps} steps ({len(results)} launches, {total / steps:.3f} ms/step on the device)",
             f"{'module':<36}{'kernel':<34}{'launches':>9}{'ms/step':>10}{'share':>8}"]
    for (module, key), (count, ms) in sorted(rows.items(), key=lambda kv: -kv[1][1])[:top]:
        lines.append(f"{module.removeprefix('i4h_physics.'):<36}{key:<34}{count:>9d}{ms / steps:>10.3f}{100.0 * ms / total:>7.1f}%")
    by_module: dict[str, float] = {}
    for (module, _), (_, ms) in rows.items():
        by_module[module] = by_module.get(module, 0.0) + ms
    lines.append("per module: " + ", ".join(f"{m.removeprefix('i4h_physics.')} {100.0 * v / total:.1f}%"
                                            for m, v in sorted(by_module.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)
