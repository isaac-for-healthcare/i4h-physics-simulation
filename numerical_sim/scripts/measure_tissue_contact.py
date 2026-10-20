# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chole contact off/on in fresh processes, synchronized wall step time and Warp pool memory.

    uv run python scripts/measure_tissue_contact.py --envs 1 16 --steps 300

Both modes allocate contact at construction; enabling changes captured work, not scratch capacity.
No rendering or instrument driver. Each child runs 20 warmup steps, then the measured steps.
Contact changes trajectories; this measures whole-step cost, not an isolated-kernel speedup.
"""
import argparse
import json
import subprocess
import sys
import time


def measure(envs, enabled, steps):
    import numpy as np
    import warp as wp
    from i4h_isaaclab.composition import load_case
    from i4h_physics.systems.contact.part_contact import PartContact

    wp.init()
    base = wp.get_mempool_used_mem_current('cuda:0')
    sim = load_case('chole_tet_med', envs=envs)
    contact = next(s for s in sim.systems if isinstance(s, PartContact))
    contact.enabled = enabled
    sim.prepare()
    for _ in range(20):
        sim.step()
    wp.synchronize()
    samples = []
    overflows = contacts = 0
    for _ in range(steps):
        start = time.perf_counter()
        sim.step()
        wp.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        # Readback outside the timed region; diagnostics count events in this step.
        overflows += int(contact.overflow_count.numpy().sum())
        contacts += int(contact.contact_count.numpy().sum())
    assert np.isfinite(sim.scene.q.numpy()).all()
    scratch = sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in
                  (contact.targets, contact.lowers, contact.uppers, contact.groups, contact.candidates,
                   contact.candidate_count, contact.delta, contact.delta_count, contact.contact_count,
                   contact.overflow_count, contact.margin, contact.stiffness, contact.max_correction))
    return {'envs': envs, 'enabled': enabled, 'steps': steps,
            'step_ms_p50': float(np.median(samples)), 'step_ms_p95': float(np.percentile(samples, 95)),
            'pool_MiB': (wp.get_mempool_used_mem_current('cuda:0') - base) / 2**20,
            'pool_high_MiB': (wp.get_mempool_used_mem_high('cuda:0') - base) / 2**20,
            'contact_arrays_MiB_excluding_BVH': scratch / 2**20,
            'overflow_events': overflows, 'contact_events': contacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', nargs='+', type=int, default=[1, 16])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--one', choices=['off', 'on'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(measure(args.envs[0], args.one == 'on', args.steps)))
        return
    for envs in args.envs:
        for mode in ('off', 'on'):
            out = subprocess.run([sys.executable, __file__, '--one', mode, '--envs', str(envs),
                                  '--steps', str(args.steps)], capture_output=True, text=True)
            if out.returncode:
                sys.stderr.write(out.stdout + out.stderr)
                raise SystemExit(out.returncode)
            print(out.stdout.strip().splitlines()[-1], flush=True)


if __name__ == '__main__':
    main()
