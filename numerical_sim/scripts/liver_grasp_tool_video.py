# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Local package paths must be set before importing simulation modules.
"""Render original recorded tissue with the authored grasper, without stepping tissue physics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for package in (
    "physics",
    "surgical",
    "standalone",
    "isaaclab",
    "assessment",
    "endoluminal",
):
    sys.path.insert(0, str(root / package / "src"))

import numpy as np
import warp as wp
from i4h_standalone.render.instruments import piece_skins, skin_pieces
from i4h_surgical.config import load_case
from i4h_surgical.systems.instruments.instrument_control import (
    place_instruments,
    shaft_rotation,
    wrist_rotation,
)
from PIL import Image, ImageDraw, ImageFont


@wp.kernel
def recorded_pose(
    position: wp.vec3,
    rotation: wp.quat,
    angle: float,
    trocar: wp.array(dtype=wp.vec3),
    has_trocar: wp.array(dtype=wp.int32),
    roots: wp.array2d(dtype=wp.vec3),
    shaft: wp.array2d(dtype=wp.quat),
    wrist: wp.array2d(dtype=wp.quat),
    jaw: wp.array2d(dtype=wp.quat),
    angles: wp.array2d(dtype=wp.float32),
):
    roots[0, 0] = position
    q = wp.normalize(rotation)
    s = shaft_rotation(position, q, trocar[0], has_trocar[0])
    shaft[0, 0] = s
    wrist[0, 0] = wrist_rotation(s, q)
    jaw[0, 0] = q
    angles[0, 0] = angle


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--data", required=True)
    args = p.parse_args()
    folder = args.campaign
    report = json.loads((folder / "report.json").read_text())
    result = max(report["results"], key=lambda r: (r["acquisition_size"], r["steps"]))
    name = f"{result['mode']}_v{result['target']}_x{result['offset_x_m']:+.3f}"
    trace = np.load(folder / name / "physical_trace.npz")
    rows = [
        json.loads(s) for s in (folder / name / "trace.jsonl").read_text().splitlines()
    ]
    q = trace["positions"][:, 0]
    geometry = trace["geometry"]
    actions = trace["actions"][:, 0]
    sim = load_case("liver_pinned_grasp_1x", data=args.data, device="cuda:0")
    ic = next(s for s in sim.systems if s.id == "instrument_control")
    body = sim.scene.body("tissue")
    triangles = body.tet.tris[0]
    skins = piece_skins(ic, sim.scene.device)
    buffers = [wp.zeros(s.count, dtype=wp.vec3, device=sim.scene.device) for s in skins]
    normals = [wp.zeros_like(v) for v in buffers]
    mask = wp.ones(1, dtype=wp.bool, device=sim.scene.device)
    print(
        "authored tool meshes:",
        [(s.name, s.count, len(s.indices) // 3) for s in skins],
        flush=True,
    )
    frames = list(range(0, len(q), 2))
    frames += [] if frames[-1] == len(q) - 1 else [len(q) - 1]
    az, el = np.deg2rad([20, 15])
    x = np.array([np.cos(az), 0, -np.sin(az)])
    z = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    y = np.cross(z, x)
    basis = np.array([x, y, z])
    all_points = (
        np.concatenate([q.reshape(-1, 3), geometry[:, [0, 2, 3]].reshape(-1, 3)])
        @ basis.T
    )
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    centre = (low + high) / 2
    scale = min(720 / max(high[0] - low[0], 0.01), 470 / max(high[1] - low[1], 0.01))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()

    def draw_scene(
        image, positions, tool, target, viewport, focus, magnification, baseline=False
    ):
        ox, oy, width, height = viewport
        layer = Image.new("RGB", (width, height), (20, 25, 33))
        draw = ImageDraw.Draw(layer)

        def project(points):
            a = np.asarray(points) @ basis.T - focus
            return np.column_stack(
                [
                    width / 2 + a[:, 0] * magnification,
                    height / 2 - a[:, 1] * magnification,
                ]
            ), a[:, 2]

        # Use one painter ordering for tool and tissue so the authored mesh occludes correctly.
        meshes = [(positions, triangles, np.array([166, 63, 53]))]
        meshes.extend(
            (v, s.indices.reshape(-1, 3), np.array([185, 201, 216]))
            for v, s in zip(tool, skins)
        )
        faces = []
        for points, tris, colour in meshes:
            pixels, depth = project(points)
            projected = pixels[tris]
            keep = (
                (projected[:, :, 0].max(axis=1) >= 0)
                & (projected[:, :, 0].min(axis=1) <= width)
                & (projected[:, :, 1].max(axis=1) >= 0)
                & (projected[:, :, 1].min(axis=1) <= height)
            )
            tris = tris[keep]
            projected = projected[keep]
            vertices = points[tris]
            cross = np.cross(
                vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
            )
            length = np.linalg.norm(cross, axis=1)
            light = 0.35 + 0.65 * np.abs(cross @ z) / np.maximum(length, 1.0e-20)
            colours = np.clip(colour[None] * light[:, None] + 10, 0, 255).astype(
                np.uint8
            )
            depths = depth[tris].mean(axis=1)
            faces.extend(
                (float(d), pts, tuple(map(int, c)))
                for d, pts, c in zip(depths, projected, colours)
            )
        for _, pts, colour in sorted(faces, key=lambda f: f[0]):
            draw.polygon([tuple(v) for v in pts], fill=colour)
        if baseline:
            original, _ = project(q[0])
            # Sparse material reference points; never magnify physical displacement.
            for a in original[::16]:
                draw.ellipse(
                    (a[0] - 1, a[1] - 1, a[0] + 1, a[1] + 1), fill=(85, 110, 130)
                )
        pt, _ = project([target])
        a, b = pt[0]
        draw.ellipse((a - 5, b - 5, a + 5, b + 5), outline="#ffe14b", width=2)
        image.paste(layer, (ox, oy))

    output = folder / "review_with_grasper.mp4"
    pipe = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1400x900",
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
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    maximum = 0.0
    sphere_error = 0.0
    for frame, i in enumerate(frames):
        wp.launch(
            recorded_pose,
            dim=1,
            inputs=[
                wp.vec3(*rows[i]["root"]),
                wp.quat(*actions[i, 3:7]),
                rows[i]["jaw_angle"],
                ic.trocar,
                ic.has_trocar,
            ],
            outputs=[ic.root, ic.shaft_q, ic.wrist_q, ic.jaw_q, ic.jaw_angle],
            device=sim.scene.device,
        )
        place_instruments(ic, sim.scene, mask, 0)
        skin_pieces(ic, 0, skins, buffers, normals, sim.scene.device)
        tool = [v.numpy() for v in buffers]
        spheres = ic.sphere_q.numpy()[0]
        active = ic.sphere_active.numpy()[0] != 0
        sphere_error = max(
            sphere_error,
            float(np.linalg.norm(spheres[active] - geometry[i, 3], axis=1).min()),
        )
        displacement = np.linalg.norm(q[i] - q[0], axis=1)
        maximum = max(maximum, float(displacement.max()))
        image = Image.new("RGB", (1400, 900), (20, 25, 33))
        draw = ImageDraw.Draw(image)
        draw_scene(image, q[i], tool, geometry[i, 0], (0, 150, 900, 600), centre, scale)
        focus = geometry[0, 0] @ basis.T
        draw_scene(
            image,
            q[i],
            tool,
            geometry[i, 0],
            (920, 185, 460, 460),
            focus,
            460 / 0.055,
            baseline=True,
        )
        draw.text(
            (24, 18),
            f"Actual grasper mesh | target {result['target']} | {result['mode']} | X {result['offset_x_m']:+.3f} m",
            font=font,
            fill="white",
        )
        draw.text(
            (24, 53),
            f"{i / 60:.2f} s | {rows[i]['phase']} | 1x playback | unscaled recorded tissue motion",
            font=font,
            fill="white",
        )
        draw.text(
            (24, 90),
            "Oblique orthographic view; tissue physics is read from the saved trace, never resimulated.",
            font=small,
            fill="#c6ced9",
        )
        draw.text((930, 150), "55 mm jaw/target close-up", font=small, fill="white")
        draw.text((930, 660), "Yellow: material target", font=small, fill="#ffe14b")
        draw.text(
            (930, 690),
            "Blue dots: initial material positions",
            font=small,
            fill="#91a9bf",
        )
        draw.text(
            (24, 770),
            f"Max tissue motion since clip start: {displacement.max() * 1000:.3f} mm | target: {displacement[result['target']] * 1000:.3f} mm",
            font=font,
            fill="white",
        )
        draw.text(
            (24, 807),
            f"Jaw angle {rows[i]['jaw_angle']:.3f} rad | captured particles: {rows[i]['volume_count']} | result: empty_capture",
            font=font,
            fill="#ffb89e",
        )
        draw.text(
            (24, 846),
            "Starts after settling. No acquisition occurred, so there is no grasp-driven pull or hold.",
            font=font,
            fill="#ffb89e",
        )
        if frame in (0, len(frames) // 2, len(frames) - 1):
            image.save(folder / f"tool_review_frame_{frame:04d}.png")
        pipe.stdin.write(image.tobytes())
    pipe.stdin.close()
    if pipe.wait():
        raise RuntimeError("ffmpeg failed")
    assert sphere_error < 1.0e-6, (
        f"reconstructed tool geometry disagrees with trace: {sphere_error}"
    )
    metrics = dict(
        attempt=name,
        max_recorded_vertex_motion_m=maximum,
        max_target_motion_m=float(
            np.linalg.norm(
                q[:, result["target"]] - q[0, result["target"]], axis=1
            ).max()
        ),
        reconstructed_nearest_sphere_max_error_m=sphere_error,
        frames=len(frames),
        source="unchanged recorded physical trace; authored tool skinning from recorded root, input quaternion and actual jaw angle",
    )
    (folder / "tool_video_validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(output, metrics, flush=True)


if __name__ == "__main__":
    main()
