# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Charts of the i4h and SOFA scaling runs for ``physics/performance.md``.

    uv run --with matplotlib python scripts/benchmark/plot_scaling.py RESULTS_DIR --out physics/images/performance

``RESULTS_DIR`` holds ``i4h-aorta.json``, ``sofa-aorta.json``, ``i4h-tissue.json`` and ``sofa-tissue.json`` (from
``i4h_scaling.py`` and ``sofa_scaling.py``). Writes, per scene, ``<scene>-throughput.svg`` (environment frames per
second against the number of environments) and ``<scene>-frame-rate.svg`` (frames per second each environment
gets, against 60 and 30 frames per second of real time), and ``speedup.svg`` (i4h throughput over SOFA's).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

I4H, SOFA = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated pair): the simulators
AORTA, TISSUE = "#1baf7a", "#6250d6"  # slots 3 and 7 (validated pair): the scenes, in the speedup chart only
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCENES = {"aorta": "Catheter in the aorta", "tissue": "Gallbladder tissue and instruments"}


def load(path: Path) -> dict[int, dict]:
    return {row["envs"]: row for row in json.loads(path.read_text())}


def style(ax, title, ylabel):
    import matplotlib.ticker as ticker

    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Parallel environments", color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def line(ax, xs, ys, color, label):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK,
                fontsize=9)


def figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, ax, path: Path, legend: str = "upper left"):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc=legend)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=SURFACE)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")  # repo whitespace hook
    print(path)


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    speedups = {}
    for scene, name in SCENES.items():
        i4h, sofa = load(args.results / f"i4h-{scene}.json"), load(args.results / f"sofa-{scene}.json")
        xi, xs = sorted(i4h), sorted(sofa)

        fig, ax = figure()
        line(ax, xi, [i4h[n]["env_frames_per_s"] for n in xi], I4H, "i4h (Newton, GPU)")
        line(ax, xs, [sofa[n]["env_frames_per_s"] for n in xs], SOFA, "SOFA (CPU, all cores)")
        style(ax, f"{name}: total simulation throughput", "Environment frames per second")
        ax.set_xlim(right=max(xi + xs) * 4)
        save(fig, ax, args.out / f"{scene}-throughput.svg", legend="lower right")

        fig, ax = figure()
        line(ax, xi, [1e3 / i4h[n]["frame_ms_p50"] for n in xi], I4H, "i4h (Newton, GPU)")
        line(ax, xs, [1e3 / sofa[n]["frame_ms_p50"] for n in xs], SOFA, "SOFA (CPU, all cores)")
        for fps, label in ((60, "60 fps: real time"), (30, "30 fps")):
            ax.axhline(fps, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(label, (xi[0], fps), xytext=(2, 4), textcoords="offset points", color=MUTED, fontsize=8)
        style(ax, f"{name}: frame rate of every environment", "Frames per second, each environment")
        ax.set_xlim(right=max(xi + xs) * 4)
        save(fig, ax, args.out / f"{scene}-frame-rate.svg", legend="lower left")
        common = sorted(set(xi) & set(xs))
        speedups[scene] = (common, [i4h[n]["env_frames_per_s"] / sofa[n]["env_frames_per_s"] for n in common])

    fig, ax = figure()
    for (scene, (xs, ys)), color in zip(speedups.items(), (AORTA, TISSUE)):
        line(ax, xs, ys, color, SCENES[scene])
    style(ax, "How many times faster i4h is than SOFA", "Speedup (i4h throughput / SOFA throughput)")
    ax.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlim(right=max(max(x) for x, _ in speedups.values()) * 8)
    save(fig, ax, args.out / "speedup.svg", legend="center right")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
