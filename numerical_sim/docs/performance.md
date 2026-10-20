# Performance

These measurements give the order of magnitude of physics throughput and memory in v0.9.0.
They are not tuned baselines; measure on your own hardware with the commands below.

## Isaac Lab throughput (64 environments)

Median environment physics steps per second over three fresh-process trials per
configuration, each with 64 warm-up and 1,024 measured physics samples per environment.
`arena` and `newton` are the Arena and direct Newton schedules; decimation is the number
of physics samples per policy step. Rendering and recording were off and final
observations were on. No other GPU compute workload ran during the trials. These runs
predate the last v0.9.0 setup and rendering changes, so treat them as an order of magnitude.

| Schedule | Execution | Decimation | Median environment physics steps/s |
| --- | --- | --- | --- |
| arena | eager | 1 | 5739 |
| arena | eager | 8 | 6077 |
| arena | captured | 1 | 5704 |
| arena | captured | 8 | 5986 |
| newton | eager | 1 | 6030 |
| newton | eager | 8 | 5926 |
| newton | captured | 1 | 5950 |
| newton | captured | 8 | 6068 |

## Standalone capacity (64 to 1,024 environments)

`aorta_static` with zero controls ran three fresh-process trials at each size below.
All nine processes exited 0, and final particle and pool positions/velocities were
finite. The GPU was an NVIDIA GB300 (aarch64) that reported **hardware power-brake
slowdown throughout every telemetry sample**. These are capacity figures under that
condition, not a healthy-hardware performance baseline.

| Environments | Median trial p50 step (ms) | Median trial p95 step (ms) | Warp loaded / after steps / high-water (MB) | Peak sampled device memory (MiB) |
| --- | --- | --- | --- | --- |
| 64 | 125.822 | 131.742 | 17.7 / 17.7 / 20.0 | 832 |
| 256 | 266.112 | 272.253 | 53.9 / 53.9 / 56.2 | 854 |
| 1,024 | 301.343 | 313.629 | 198.7 / 198.7 / 201.0 | 986 |

Each step advances all environments. Timing is `time.perf_counter()` around
`sim.step()` plus `wp.synchronize()`, after graph preparation and 20 warm-up steps,
for 200 measured steps. The table reports medians of the three per-trial p50/p95
values, not pooled percentiles. Loading, preparation and final finite-state checks
are outside that interval. Controls are zero, rendering is off, and no training
runs concurrently. This measurement does not cover scripted navigation quality.

Warp memory is the mempool allocation above its post-initialization baseline,
in decimal MB; all three trials had the same reported allocations. Device memory
is whole-GPU usage sampled once per second, in MiB, including runtime and display
allocations; its peak is across the three trials and can miss subsecond peaks.
No other GPU compute process ran during the trials.

Stack: Linux aarch64, driver 595.58.03 (reports CUDA 13.2), Python 3.12.12,
Warp 1.17.0 (CUDA toolkit 12.9), Newton 1.6.1, NumPy 2.4.6 and Torch 2.14.0+cu130
(built with CUDA 13.0), from the locked numerical environment.

From `numerical_sim/`, with `I4H_DATA` pointing to the external assets:

```bash
for trial in 1 2 3; do
    for envs in 64 256 1024; do
        .venv/bin/python scripts/measure.py aorta_static --envs "$envs" --warmup 20 --steps 200 --json "measure-aorta-static-$envs-$trial-rows.json"
    done
done
```
