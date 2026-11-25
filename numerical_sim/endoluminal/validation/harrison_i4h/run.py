# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the shaft validation campaign from the project virtual environment."""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from harrison_inputs import SLOPES  # noqa: E402


def cases():
    for i, wire in enumerate(SLOPES):
        for label, dt, edges, stroke, flags in [
            ("250us_24e", 0.00025, 24, 5, []),
            ("100us_24e", 0.0001, 24, 5, []),
            ("125us_24e", 0.000125, 24, 0.3, []),
            ("125us_48e", 0.000125, 48, 0.3, []),
            ("62p5us_24e", 0.0000625, 24, 5 if i >= 6 else 0.3, []),
            (
                "62p5us_24e_rate",
                0.0000625,
                24,
                0.3,
                ["--damping-reference-dt", ".00025"],
            ),
            ("load_250us_24e", 0.00025, 24, 5, ["--load-test"]),
        ]:
            yield wire, label, dt, edges, stroke, flags


def run(case):
    wire, label, dt, edges, stroke, flags = case
    path = ROOT / "traces" / f"{wire}_{label}.csv"
    command = [
        sys.executable,
        str(ROOT.parent / "guidewire_solver_bench.py"),
        "--wire",
        wire,
        "--dt",
        str(dt),
        "--edges",
        str(edges),
        "--stroke-mm",
        str(stroke),
        "--sample-stride",
        str(round(0.03 / dt)),
        "--output",
        str(path),
        *flags,
    ]
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    return read_fit(wire, label, output.stdout)


def read_fit(wire, label, output):
    if label.startswith("load_"):
        return None
    match = re.search(
        r"Initial slope \(N/mm\), final force \(N\): ([\d.eE+-]+) ([\d.eE+-]+)", output
    )
    if not match:
        raise ValueError(f"Missing fitted slope for {wire} {label}")
    return (wire, label, float(match[1]), float(match[2]))


def write_fits(rows):
    with (ROOT / "fits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wire", "configuration", "slope_N_mm", "last_force_N"])
        writer.writerows(row for row in rows if row is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    (ROOT / "traces").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        write_fits(list(pool.map(run, cases())))
