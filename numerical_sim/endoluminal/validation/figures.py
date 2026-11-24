# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate all eleven report figures from saved traces and registered centrelines."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pxr import Usd, UsdGeom

here = Path(__file__).resolve().parent
out = here / "figures"
src = here / "harrison_i4h"
teal = "#006d77"
orange = "#bd4916"
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "savefig.facecolor": "white",
    }
)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
for ax, name, slug in zip(
    axs,
    ["Amplatz Extra Stiff", "Lunderquist Extra Stiff"],
    ["amplatz-extra-stiff", "lunderquist-extra-stiff"],
):
    for path, label, c, ls in [
        (src / "traces" / f"{slug}_100us_24e.csv", "i4h · 0.1 ms", teal, "-"),
        (src / "16ms_16substeps" / f"{slug}.csv", "i4h · 1 ms", orange, "--"),
    ]:
        a = np.genfromtxt(path, delimiter=",", names=True)
        ax.plot(a["displacement_mm"], a["force_N"], label=label, color=c, ls=ls, lw=2.3)
    pts = [
        (float(r["published_travel_mm"]) - 0.55, float(r["load_N"]))
        for r in csv.DictReader((src / "figure2_digitized.csv").open())
        if r["wire"] == name
    ]
    ax.scatter(*zip(*pts), color="#242d35", label="Harrison · Fig. 2", s=22, zorder=5)
    ax.set(title=name, xlabel="Midpoint displacement (mm)", xlim=(0, 5), ylim=(0, None))
    ax.grid(axis="y", alpha=0.2)
axs[0].set_ylabel("Force (N)")
axs[0].legend(frameon=False, fontsize=9)
fig.savefig(out / "guidewire-bending.png", dpi=190)
base = here / "elastrat"
stage = Usd.Stage.Open(str(base / "phantom/phantom.usdc"))
mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Phantom"))
assert set(mesh.GetFaceVertexCountsAttr().Get()) == {3}
pts = np.asarray(mesh.GetPointsAttr().Get()) * 1000
f = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
fig, axs = plt.subplots(1, 3, figsize=(10, 5.4), layout="constrained", sharey=True)
for ax, key, name, d in zip(
    axs,
    ["cookStraight3", "bentson3", "beacon3"],
    ["Cook Straight", "Bentson", "Cook Beacon 5 Fr"],
    ["guidewires_profiled", "guidewires_profiled", "catheters"],
):
    ref = np.load(base / "phantom/references" / f"{key}.npy") * 1000
    q = np.load(base / d / f"{key}.npy") * 1000
    q = q[((q - ref[-1]) @ (ref[-1] - ref[-9:-1]).sum(axis=0)) < 0]
    ax.add_collection(
        PolyCollection(
            pts[:, [0, 2]][f],
            facecolors="#dce3e8",
            edgecolors="none",
            antialiased=False,
        )
    )
    for curve, color, ls, label, tip, m in [
        (ref, teal, "-", "Measured CT", 0, "o"),
        (q, orange, "--", "i4h", -1, "x"),
    ]:
        ax.plot(curve[:, 0], curve[:, 2], ls, color=color, lw=2, label=label)
        ax.scatter(*curve[tip, [0, 2]], c=color, marker=m, s=47, zorder=5)
    ax.set(
        title=name,
        xlim=(-105, 90),
        ylim=(-175, 175),
        aspect="equal",
        xlabel="CT x (mm)",
    )
    ax.tick_params(labelsize=9)
axs[0].set_ylabel("CT z (mm)")
axs[0].legend(frameon=False, loc="lower right", fontsize=9)
fig.savefig(out / "phantom-insertion.png", dpi=190)

groups = {
    "guidewires_profiled": [
        ("cookStraight", "Cook Straight"),
        ("cookJ", "Cook Safe-T-J"),
        ("rosen", "Cook Rosen"),
        ("amplatz", "Amplatz Super Stiff"),
        ("bentson", "Bentson"),
        ("terumoStiff", "Terumo Stiff Angled"),
    ],
    "catheters": [
        ("terumoST4F", "Terumo Straight 4 Fr"),
        ("terumoST5F", "Terumo Straight 5 Fr"),
        ("beacon", "Cook Beacon 5 Fr"),
    ],
}
seen = set()
for group, instruments in groups.items():
    results = {
        r["reference"]: r for r in csv.DictReader((base / group / "results.csv").open())
    }
    for stem, name in instruments:
        fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.9))
        fig.subplots_adjust(left=0.04, right=0.96, top=0.77, bottom=0.09, wspace=0.03)
        fig.suptitle(name, fontsize=16, fontweight="bold", y=0.984)
        for i, (ax, title) in enumerate(
            zip(
                axs,
                ["Iliac bifurcation", "Aortic bifurcation", "Left renal artery origin"],
            ),
            1,
        ):
            key = f"{stem}{i}"
            seen.add(key)
            ref = np.load(base / "phantom/references" / f"{key}.npy") * 1000
            rod = np.load(base / group / f"{key}.npy") * 1000
            signed = (rod - ref[-1]) @ (ref[-1] - ref[-9:-1]).sum(axis=0)
            assert signed[-1] < 0
            last = np.flatnonzero(signed >= 0)[-1]
            q = np.vstack(
                (
                    rod[last]
                    + (rod[last + 1] - rod[last])
                    * signed[last]
                    / (signed[last] - signed[last + 1]),
                    rod[last + 1 :],
                )
            )
            tip = np.linalg.norm(ref[0] - q[-1])
            assert abs(tip - float(results[key]["tip_error_mm"])) < 1e-7
            ax.add_collection(
                PolyCollection(
                    pts[:, [0, 2]][f],
                    facecolor="#dce3e8",
                    edgecolor="none",
                    antialiased=False,
                )
            )
            ax.plot(*ref[:, [0, 2]].T, color="#006d77", lw=2)
            ax.plot(*q[:, [0, 2]].T, "--", color="#bd4916", lw=1.7)
            ax.plot(*ref[0, [0, 2]], "o", color="#006d77", ms=6)
            ax.plot(*q[-1, [0, 2]], "x", color="#bd4916", ms=7, mew=1.8)
            ax.set(xlim=(-100, 75), ylim=(-176, 166), aspect="equal")
            ax.axis("off")
            ax.set_title(title, fontsize=10, pad=3)
            ax.text(
                0.5,
                -0.05,
                f"Tip error  {tip:.1f} mm",
                ha="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        fig.legend(
            handles=[
                Line2D([], [], color="#006d77", marker="o", label="Measured CT"),
                Line2D([], [], color="#bd4916", ls="--", marker="x", label="i4h"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            frameon=False,
            ncol=2,
            fontsize=9,
        )
        file = out / f"elastrat-{stem}.png"
        fig.savefig(file, dpi=180, facecolor="white")
        plt.close(fig)
assert len(seen) == 27
