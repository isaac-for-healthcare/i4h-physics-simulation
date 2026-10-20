# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation of completed task rows; threaded UI never reads live device arrays."""

import numpy as np
import warp as wp
from i4h_assessment.executor import RESULTS


def marker_points(row, radius):
    # A small centre and three rings show the acceptance sphere without hiding anatomy.
    points = [row[:3]]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            p = row[:3].copy()
            p[a] += radius * np.cos(angle)
            p[b] += radius * np.sin(angle)
            points.append(p)
    return points


def publish_task(renderer, snap, place=True):
    if renderer.task_markers is None:
        return
    rows = snap.task_host
    if rows is None:  # serial renderer only
        rows = snap.task_rows.numpy()
    row = rows[renderer.env].copy()
    renderer.task_row = row
    if not place:
        return
    progress = snap.task_progress_host
    if progress is None:
        progress = snap.task_progress.numpy()
    radius = progress[renderer.env, 1]
    for i, p in enumerate(marker_points(row, radius)):
        renderer.task_markers.set(i, p, radius * (0.06 if i else 0.12), bool(row[6]))


def task_text(task, row, progress=None, history=None):
    index = int(progress[0]) if progress is not None else 0
    step = task.steps[index]
    text = (
        f"{index + 1}/{task.step_count} {task.ids[index]}: {step.cfg['type']} / {step.cfg['tool']}\n"
        f"Target {row[:3]}  radius {step.radius:g}\n"
        f"Distance {row[8]:.4f}  hold {row[10]:.3f}/{step.hold:g} s\n"
        f"Elapsed {row[9]:.2f} s  {RESULTS[int(row[11])]}"
    )
    if progress is not None and task.has_path:
        text += f"\nTip path length {progress[3]:.4f}"
    if history is not None:
        for sid, (activation, completion) in zip(task.ids, history):
            status = f"completed {completion:.2f} s ({completion-activation:.2f} s)" if completion >= 0 else (
                f"active {row[9]-activation:.2f} s" if activation >= 0 else "pending")
            text += f"\n{sid}: {status}"
    return text


@wp.kernel
def place_task_markers(
    rows: wp.array2d(dtype=float), env: int, progress: wp.array2d(dtype=float), q: wp.array2d(dtype=wp.vec3), radii: wp.array2d(dtype=float)
):
    i = wp.tid()
    radius = progress[env, 1]
    p = wp.vec3(rows[env, 0], rows[env, 1], rows[env, 2])
    size = radius * 0.12
    if i > 0:
        ring = (i - 1) // 24
        angle = float((i - 1) % 24) * 6.283185307179586 / 24.0
        a, b = int(0), int(1)
        if ring == 1:
            b = 2
        elif ring == 2:
            a, b = 1, 2
        p[a] += radius * wp.cos(angle)
        p[b] += radius * wp.sin(angle)
        size = radius * 0.06
    q[0, i] = p
    radii[0, i] = size if rows[env, 6] == 0.0 else 0.0
    radii[1, i] = size if rows[env, 6] != 0.0 else 0.0
