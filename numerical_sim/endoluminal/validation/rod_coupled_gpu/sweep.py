# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run all 15 PR #22 fixtures and compare with the archived displacement traces."""

import argparse
import csv
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np


def cases(validation):
    harrison = runpy.run_path(str(validation / "harrison_inputs.py"))["SLOPES"]
    for name in harrison:
        yield (
            name, ["--wire", name], 4.407,
            validation / "harrison_sofa/traces", validation / "harrison_vbd/traces",
        )
    catheter = validation / "imperial/catheter_bending"
    with (catheter / "inputs.csv").open() as stream:
        for row in csv.DictReader(stream):
            yield (
                row["slug"], ["--ei-n-mm2", row["EI_N_mm2"], "--radius-mm", str(float(row["OD_mm"]) / 2)],
                4.0, catheter / "traces/sofa", catheter / "traces/vbd",
            )


def read(path, external=False):
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2) if external else None)
    if not np.isfinite(data).all() or data[-1, 0] < 4.99:
        raise ValueError(f"Incomplete or nonfinite force trace in {path}")
    return data


def slope(data):
    initial = data[data[:, 0] <= .3]
    return np.polyfit(initial[:, 0], initial[:, 1], 1)[0]


def rms_relative(reference, measured):
    return 100 * np.linalg.norm(measured - reference) / np.linalg.norm(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--reference-dir", type=Path, help="Optional same-fixture CPU results for device comparison.")
    parser.add_argument(
        "--archive-references-from", type=Path, metavar="VALIDATION_DIR",
        help="Archive generated SOFA and VBD traces and exit without running the sweep.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = Path(__file__).resolve().parent.parent
    if args.archive_references_from is not None:
        curves = {
            f"{backend}_{name}": read(folder / f"{name}.csv", external=True)
            for name, _, _, sofa, vbd in cases(args.archive_references_from)
            for backend, folder in (("sofa", sofa), ("vbd", vbd))
        }
        np.savez_compressed(args.output_dir / "reference_curves.npz", **curves)
        return
    with np.load(Path(__file__).with_name("reference_curves.npz"), allow_pickle=False) as archive:
        curves = {name: archive[name] for name in archive.files}
    pythonpath = os.pathsep.join(filter(None, (str(validation), os.environ.get("PYTHONPATH"))))
    env = dict(os.environ, PYTHONPATH=pythonpath)
    bench = Path(__file__).with_name("run.py")
    sample = np.linspace(.3, 4.8, 451)
    results = []
    for name, material, reporting_point, _, _ in cases(validation):
        output = args.output_dir / f"{name}.csv"
        command = [sys.executable, str(bench), *material, "--device", args.device, "--iterations",
                   str(args.iterations), "--output", str(output)]
        if args.no_cuda_graph:
            command.append("--no-cuda-graph")
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = read(output)
        force = np.interp(reporting_point, data[:, 0], data[:, 1])
        sampled_force = np.interp(sample, data[:, 0], data[:, 1])
        result = dict(
            shaft=name,
            reporting_displacement_mm=reporting_point,
            slope_N_per_mm=slope(data),
            force_N=force,
            max_guide_gap_mm=np.max(data[:, 4]),
            max_force_balance_error_N=np.max(np.abs(data[:, 1] - data[:, 2] - data[:, 3])),
        )
        comparisons = {backend: curves[f"{backend}_{name}"] for backend in ("sofa", "vbd")}
        if args.reference_dir is not None:
            comparisons["reference"] = read(args.reference_dir / output.name)
        for backend, reference in comparisons.items():
            reference_force = np.interp(reporting_point, reference[:, 0], reference[:, 1])
            sampled_reference = np.interp(sample, reference[:, 0], reference[:, 1])
            result[f"slope_diff_vs_{backend}_pct"] = 100 * (result["slope_N_per_mm"] / slope(reference) - 1)
            result[f"force_diff_vs_{backend}_pct"] = 100 * (force / reference_force - 1)
            result[f"curve_rms_diff_vs_{backend}_pct"] = rms_relative(sampled_reference, sampled_force)
        results.append(result)
        print(name, flush=True)
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
