# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offscreen Slang video of a learned policy, with actual 30 Hz control and no teacher corrections."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from i4h_standalone.render import make_renderer
from i4h_standalone.render.slang import Markers
from i4h_surgical.config import read_case
from PIL import Image, ImageDraw, ImageFont
from train import load_policy
from training_env import ROOT, LiverEnv, schedule, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--data", default=str(ROOT.parent.parent / "i4h-isaaclab/numerical_sim/data")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/liver-policy-20261119"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    env = LiverEnv(args.data, 1, args.output / "snapshots", seed=82828)
    policy = load_policy(env, args.checkpoint)
    cfg = read_case(ROOT / "examples/liver_grasp/minimou.yaml", data=args.data)[
        "render"
    ]
    cfg.update(window=False, size=[1280, 720])
    cfg["camera"]["position"] = [0.04, 0.075, -0.22]
    renderer = make_renderer(cfg, env.sim)
    renderer.markers = Markers(renderer, 2, idle_color=(1.0, 0.8, 0.05))
    path = args.output / (args.checkpoint.stem + "_slang.mp4")
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
            "1280x720",
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
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    results, frames = [], 0
    try:
        for episode, spec in enumerate(schedule(args.episodes, 67423)):
            env.reset([0], [spec])
            for tick in range(600):
                with torch.no_grad():
                    env.step(policy(env.get_observations()), auto_reset=False)
                t = env.tasks[0]
                target = env.measurement["geometry"][0, 0]
                renderer.markers.set(0, target, 0.0015, False)
                renderer.markers.set(
                    1,
                    target if t.destination is None else t.destination,
                    0.0 if t.destination is None else 0.0015,
                    True,
                )
                renderer.publish()
                renderer.draw()
                image = Image.fromarray(
                    np.clip(renderer.read_frame()[..., :3] * 255, 0, 255).astype(
                        np.uint8
                    )
                )
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 84), fill=(16, 21, 29))
                draw.text(
                    (16, 8),
                    f"{args.checkpoint.stem} | target {env.probe.target.numpy()[0]} | {t.phase} | {t.step / 60:.2f}s | 1x",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (16, 43),
                    "State policy | fixed demonstrated orientation | 20 mm lift | no teacher corrections",
                    font=font,
                    fill="white",
                )
                if frames in (0, 90, 180):
                    image.save(
                        args.output / f"{args.checkpoint.stem}_frame_{frames:04d}.png"
                    )
                repeats = 16 if tick == 0 else 1
                for _ in range(repeats):
                    pipe.stdin.write(image.tobytes())
                    frames += 1
                if t.done:
                    result = t.result(spec, int(env.probe.target.numpy()[0]))
                    results.append(result)
                    print(result, flush=True)
                    for _ in range(8):
                        pipe.stdin.write(image.tobytes())
                        frames += 1
                    break
    finally:
        renderer.close()
        pipe.stdin.close()
        code = pipe.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    write_json(
        args.output / f"{args.checkpoint.stem}_video.json",
        dict(renderer="Slang, offscreen", frames=frames, fps=30, results=results),
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
