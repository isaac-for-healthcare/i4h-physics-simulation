# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the four Imperial catheter shafts in the three Harrison fixtures."""

import csv
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Outer diameters in mm, lumen diameters in inches, from catheters.txt.
DEVICES = [
    ("Cordis Angiographic 4 Fr", "cordis4", 1.35, 0.035, ""),
    ("Terumo Straight 4 Fr", "terumo4", 1.40, 0.038, "terumoST4F"),
    ("Terumo Straight 5 Fr", "terumo5", 1.70, 0.038, "terumoST5F"),
    ("Cook Royal Flush 5 Fr", "beacon5", 5 / 3, 0.038, "beacon"),
]


def materials():
    with (HERE.parent / "measured/flexural_modulus_profiles_SI.csv").open() as f:
        station = {
            r["device"]: float(r["flexural_modulus_Pa"]) / 1e6
            for r in csv.DictReader(f)
            if r["kind"] == "catheter" and float(r["distance_from_tip_m"]) == 0.31
        }
    for name, slug, outer, lumen, ct in DEVICES:
        inner = lumen * 25.4
        ei = station[name] * math.pi * (outer**4 - inner**4) / 64
        yield dict(
            device=name,
            slug=slug,
            station_mm=310,
            E_MPa=station[name],
            OD_mm=outer,
            ID_mm=inner,
            EI_N_mm2=ei,
            beam_slope_N_mm=48 * ei / 40**3,
            ct_prefix=ct,
        )


def simulate(job):
    row, solver = job
    out = HERE / "traces" / solver
    out.mkdir(parents=True, exist_ok=True)
    slug = row["slug"]
    args = ["--ei-n-mm2", str(row["EI_N_mm2"]), "--radius-mm", str(row["OD_mm"] / 2)]
    if solver == "i4h":
        cmd = [
            sys.executable,
            ROOT / "guidewire_solver_bench.py",
            "--dt",
            ".016",
            "--substeps",
            "16",
            "--output",
            out / f"{slug}.csv",
            *args,
        ]
    else:
        launcher = {"sofa": "harrison_sofa", "vbd": "harrison_vbd"}[solver]
        cmd = [
            ROOT / launcher / "run.sh",
            slug,
            "--inner-radius-mm",
            str(row["ID_mm"] / 2),
            "--output",
            out,
            *args,
        ]
    result = subprocess.run(
        cmd,
        env={**os.environ, "BENCH_PYTHON": sys.executable},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode or "[ERROR]" in result.stdout:
        raise RuntimeError(f"{solver}/{slug}\n{result.stdout}")
    print(f"{solver}/{slug} completed", flush=True)


if __name__ == "__main__":
    rows = list(materials())
    with (HERE / "inputs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    # Keep each backend sequential to avoid concurrent GPU module compilation.
    def run_backend(solver):
        for row in rows:
            simulate((row, solver))

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(run_backend, ("sofa", "vbd", "i4h")))
