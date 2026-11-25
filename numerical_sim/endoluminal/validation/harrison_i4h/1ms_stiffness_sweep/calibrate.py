# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit numerical stiffness at a 1 ms substep. Run with the project interpreter."""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
VALIDATION = OUT.parent
sys.path.insert(0, str(VALIDATION.parent))
from harrison_inputs import SLOPES  # noqa: E402


def simulate(job):
    wire, scale, label = job
    import guidewire_solver_bench as bench

    path = OUT / (label + ".csv")
    if (
        "--reuse-sweep" in ORIGINAL_ARGS
        and label.startswith("input_")
        and path.exists()
    ):
        return path
    sys.argv = [
        "bench",
        "--wire",
        wire,
        "--dt",
        ".016",
        "--substeps",
        "16",
        "--stiffness-scale",
        str(scale),
        "--output",
        str(path),
    ]
    bench.main()
    return path


def response(path):
    data = np.loadtxt(path, skiprows=1, delimiter=",")
    assert data.shape == (939, 5) and np.isfinite(data).all() and data[-1, 0] == 5
    initial = data[data[:, 0] <= 0.3]
    return np.polyfit(initial[:, 0], initial[:, 1], 1)[0], data[:, :2]


def write(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    baseline = VALIDATION / "16ms_16substeps"
    samples = {k: response(baseline / (w + ".csv")) for w, k in SLOPES.items()}
    # All wires share geometry, mass, boundary conditions and solver parameters.
    # Equal EI inputs therefore give identical simulated force histories.
    extra = [
        0.245,
        0.315,
        0.37,
        0.45,
        0.50,
        0.62,
        0.82,
        1.0,
        1.25,
        1.7,
        2.0,
        2.5,
        4.5,
        6.0,
        8.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        18.0,
        20.0,
        50.0,
        150.0,
        500.0,
    ]
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [("amplatz", k / SLOPES["amplatz"], f"input_{k:g}") for k in extra]
        failures = []
        for k, path in zip(extra, pool.map(simulate, jobs)):
            slope, a = response(path)
            gap = np.loadtxt(path, skiprows=1, delimiter=",", usecols=4)
            if gap.max() > 0.001:
                failures.append(
                    dict(
                        input_k_N_mm=k,
                        initial_slope_N_mm=slope,
                        peak_gap_mm=gap.max(),
                        first_gap_over_1um_at_mm=a[np.flatnonzero(gap > 0.001)[0], 0],
                    )
                )
            else:
                samples[k] = slope, a
        if failures:
            write("excluded.csv", failures)
        keys = np.array(sorted(samples))
        slopes = np.array([samples[k][0] for k in keys])
        paper = list(csv.DictReader((VALIDATION / "summary.csv").open()))
        fig = list(csv.DictReader((VALIDATION / "figure2_digitized.csv").open()))
        write(
            "sweep.csv",
            [
                dict(
                    input_k_N_mm=k,
                    fitted_k_N_mm=samples[k][0],
                    force_at_4p407mm_N=np.interp(
                        4.407081545, samples[k][1][:, 0], samples[k][1][:, 1]
                    ),
                )
                for k in keys
            ],
        )
        jobs = []
        for (wire, k), row in zip(SLOPES.items(), paper):
            idx = np.flatnonzero(slopes >= k)
            if len(idx):
                j = idx[0]
                value = np.interp(k, slopes[j - 1 : j + 1], keys[j - 1 : j + 1])
                jobs.append((wire, value / k, wire + "_slope"))
            targets = [r for r in fig if r["wire"] == row["wire"]]
            if targets:
                x = np.array([float(r["published_travel_mm"]) - 0.55 for r in targets])
                y = np.array([float(r["load_N"]) for r in targets])
                matrix = np.array(
                    [
                        np.interp(x, samples[key][1][:, 0], samples[key][1][:, 1])
                        for key in keys
                    ]
                )
                # Least-squares fit to all seven points on each interpolation interval.
                delta = np.diff(matrix, axis=0)
                fraction = np.clip(
                    np.sum((y - matrix[:-1]) * delta, axis=1)
                    / np.maximum(1e-20, np.sum(delta**2, axis=1)),
                    0,
                    1,
                )
                j = np.argmin(
                    np.sum((matrix[:-1] + fraction[:, None] * delta - y) ** 2, axis=1)
                )
                value = keys[j] + fraction[j] * (keys[j + 1] - keys[j])
                jobs.append((wire, value / k, wire + "_curve"))
        result = []
        for (wire, scale, label), path in zip(jobs, pool.map(simulate, jobs)):
            slope, a = response(path)
            assert np.loadtxt(path, delimiter=",", skiprows=1, usecols=4).max() <= 0.001
            k = SLOPES[wire]
            row = paper[list(SLOPES).index(wire)]
            targets = [r for r in fig if r["wire"] == row["wire"]]
            rmse = base_rmse = last_error = None
            if targets:
                x = [float(r["published_travel_mm"]) - 0.55 for r in targets]
                y = np.array([float(r["load_N"]) for r in targets])
                pred = np.interp(x, a[:, 0], a[:, 1])
                base = samples[k][1]
                rmse = np.sqrt(np.mean((pred - y) ** 2))
                base_rmse = np.sqrt(
                    np.mean((np.interp(x, base[:, 0], base[:, 1]) - y) ** 2)
                )
                last_error = 100 * (pred[-1] / y[-1] - 1)
            result.append(
                dict(
                    wire=row["wire"],
                    objective=label.split("_")[-1],
                    scale=scale,
                    target_slope=k,
                    achieved_slope=slope,
                    slope_error_pct=100 * (slope / k - 1),
                    baseline_curve_rmse_N=base_rmse,
                    curve_rmse_N=rmse,
                    last_error_pct=last_error,
                )
            )
        write("verified_fits.csv", result)


ORIGINAL_ARGS = tuple(sys.argv)
if __name__ == "__main__":
    main()
