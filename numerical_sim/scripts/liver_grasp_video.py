# SPDX-License-Identifier: Apache-2.0
"""Render retained numerical traces as a labelled mesh projection, without resimulation."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.campaign / "report.json").read_text())
    results = report.get("results", [])
    if not results:
        raise RuntimeError(
            "no scored attempt available; use settling.npz for a settling-only clip"
        )
    result = max(results, key=lambda r: (r["acquisition_size"], r["steps"]))
    name = f"{result['mode']}_v{result['target']}_x{result['offset_x_m']:+.3f}"
    directory = args.campaign / name
    trace = np.load(directory / "physical_trace.npz")
    rows = [
        json.loads(line)
        for line in (directory / "trace.jsonl").read_text().splitlines()
    ]
    q = trace["positions"][:, 0]
    geometry = trace["geometry"]
    triangles = np.loadtxt(args.data / "choly-mid/liver/model.tris", dtype=np.int32)
    destination = (
        np.asarray(result["destination"])
        if result["destination"] is not None
        else geometry[0, 0] + [0, 0.01, 0]
    )
    lo = np.minimum(q.min(axis=(0, 1)), geometry[:, [0, 2, 3]].min(axis=(0, 1)))
    hi = np.maximum(q.max(axis=(0, 1)), geometry[:, [0, 2, 3]].max(axis=(0, 1)))
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 0.6
    centre = (lo + hi) / 2
    scale = 280 / max(span, 0.01)

    def project(points):
        points = np.asarray(points)
        return np.column_stack(
            [
                500 + (points[..., 0] - centre[0]) * scale,
                435 - (points[..., 1] - centre[1]) * scale,
            ]
        )

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    output = args.campaign / "review.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1000x800",
        "-r",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frame_ids = list(range(0, len(q), 2))
    if frame_ids[-1] != len(q) - 1:
        frame_ids.append(len(q) - 1)
    for frame, i in enumerate(frame_ids):
        image = Image.new("RGB", (1000, 800), (20, 25, 33))
        draw = ImageDraw.Draw(image)
        draw.text(
            (24, 15),
            f"Pinned liver | {result['mode']} | target {result['target']} | start X {result['offset_x_m']:+.3f} m",
            font=font,
            fill="white",
        )
        draw.text(
            (24, 48),
            f"{i / 60:.2f} s   {rows[i]['phase']}   playback 1x   physics 60 Hz / commands 30 Hz",
            font=font,
            fill="white",
        )
        draw.text(
            (24, 82),
            "Numerical X/Y projection; nearest jaw sphere shown in cyan",
            font=small,
            fill="#bec8d7",
        )
        draw.text(
            (24, 108),
            "Yellow: material target | Cyan: nearest jaw sphere | Green: "
            + (
                "frozen destination"
                if result["destination"] is not None
                else "planned +Y (no acquisition)"
            ),
            font=small,
            fill="#bec8d7",
        )
        projected = project(q[i])
        points = q[i][triangles]
        normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        length = np.linalg.norm(normal, axis=1)
        light = 0.3 + 0.7 * np.abs(normal[:, 2]) / np.maximum(length, 1.0e-12)
        for f in np.argsort(points[:, :, 2].mean(axis=1)):
            shade = float(light[f])
            colour = (int(140 * shade + 25), int(55 * shade + 12), int(50 * shade + 12))
            draw.polygon([tuple(v) for v in projected[triangles[f]]], fill=colour)
        for point, colour, label in (
            (geometry[i, 0], "#ffdf30", "target"),
            (geometry[i, 3], "#48dfff", "nearest jaw sphere"),
            (
                destination,
                "#65ff8b",
                "destination"
                if result["destination"] is not None
                else "planned +Y (no acquisition)",
            ),
        ):
            x, y = project([point])[0]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=colour, width=3)

        row = rows[i]
        draw.rectangle((0, 705, 1000, 800), fill=(20, 25, 33))
        draw.text(
            (24, 715),
            f"volume {row['volume_count']} | surface {row['surface_count']} | retained {row['retained']}/{row['patch_size']} | max radial residual {1000 * row['max_radial_residual_m']:.3f} mm",
            font=small,
            fill="white",
        )
        draw.text(
            (24, 742),
            f"Inverted tets: {row['inverted']} | attachment drift max: {1000 * row['attachment_max_drift_m']:.1f} mm",
            font=small,
            fill="#ffb89e",
        )
        draw.text(
            (24, 768),
            f"Attempt result: {result['status']} / {result['failure']}",
            font=small,
            fill="#ffb89e",
        )
        if frame in (0, len(frame_ids) // 2, len(frame_ids) - 1):
            image.save(args.campaign / f"review_frame_{frame:04d}.png")
        pipe.stdin.write(image.tobytes())
    pipe.stdin.close()
    if pipe.wait():
        raise RuntimeError("ffmpeg failed")
    release = []
    for r in results:
        command = r["release_command_step"]
        release.append(
            dict(
                target=r["target"],
                mode=r["mode"],
                offset_x_m=r["offset_x_m"],
                surface_latency_s=None
                if command is None or r["surface_clear_step"] is None
                else (r["surface_clear_step"] - command) / 60,
                volume_latency_s=None
                if command is None or r["volume_clear_step"] is None
                else (r["volume_clear_step"] - command) / 60,
            )
        )
    (args.campaign / "release_latencies.json").write_text(
        json.dumps(release, indent=2) + "\n"
    )
    pairs = []
    for eager in [r for r in results if r["mode"] == "eager"]:
        captured = next(
            (
                r
                for r in results
                if r["mode"] == "captured"
                and r["target"] == eager["target"]
                and r["offset_x_m"] == eager["offset_x_m"]
            ),
            None,
        )
        if captured is None:
            continue
        a = np.load(
            args.campaign
            / f"eager_v{eager['target']}_x{eager['offset_x_m']:+.3f}"
            / "physical_trace.npz"
        )["positions"]
        b = np.load(
            args.campaign
            / f"captured_v{eager['target']}_x{eager['offset_x_m']:+.3f}"
            / "physical_trace.npz"
        )["positions"]
        n = min(len(a), len(b))
        pairs.append(
            dict(
                target=eager["target"],
                offset_x_m=eager["offset_x_m"],
                compared_steps=n,
                max_pointwise_distance_m=float(
                    np.linalg.norm(a[:n] - b[:n], axis=-1).max()
                ),
                same_result=eager["status"] == captured["status"]
                and eager["failure"] == captured["failure"],
            )
        )
    (args.campaign / "eager_captured_comparison.json").write_text(
        json.dumps(pairs, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
