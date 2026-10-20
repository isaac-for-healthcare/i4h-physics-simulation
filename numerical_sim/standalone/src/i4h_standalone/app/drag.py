# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Right-mouse dragging of a centreline-tree node in the Slang window -> the ``drag`` system.

Ported from omniendo xcath/xcath.py:1850-1920 (``_on_vessel_drag_press/motion/release``: pick the nearest movable
node within 18 px, drag it in the plane through the node facing the camera, keep the cursor offset) and
xcath/vessel_drag.py:48-81 (the pixel projection, here on the host: a few hundred to a thousand nodes) and
:576-593 (``ray_plane_intersection``). The picked positions come from the renderer's held snapshot, copied into
pinned host memory on the render stream (no ``.numpy()`` on this thread while the worker runs); the edits reach
the system between steps through ``apply`` (the worker's ``edit`` queue, or a direct call without a worker).
A consumed right-button event is not passed to the camera (which would pan).
"""

from __future__ import annotations

import numpy as np
import warp as wp

PICK_RADIUS_PIXELS = 18.0  # vessel_drag.py:15


def ray_plane(origin, direction, point, normal):
    """vessel_drag.py:576-593: the forward ray/plane hit, or None when parallel or behind."""
    origin, direction = np.asarray(origin, np.float64), np.asarray(direction, np.float64)
    point, normal = np.asarray(point, np.float64), np.asarray(normal, np.float64)
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-10:
        return None
    distance = float(np.dot(point - origin, normal) / denominator)
    if distance <= 0.0:
        return None
    return (origin + direction * distance).astype(np.float32)


def pick_node(points, inv_mass, camera, px: float, py: float, width: float, height: float, radius=PICK_RADIUS_PIXELS):
    """vessel_drag.py:18-81 on the host: the movable node whose projection is within ``radius`` pixels of the cursor,
    the nearest to the camera first, then the nearest to the cursor, then the lowest index. ``None`` when nothing."""
    screen = camera.project(points, width, height)  # (n, 3): px, py (y down), depth
    d2 = (screen[:, 0] - px) ** 2 + (screen[:, 1] - py) ** 2
    ok = (np.asarray(inv_mass) > 0.0) & (screen[:, 2] > 1.0e-8) & (d2 <= radius * radius)
    if not ok.any():
        return None
    candidates = np.flatnonzero(ok)
    order = np.lexsort((candidates, d2[candidates], screen[candidates, 2]))  # depth, then pixel distance, then index
    return int(candidates[order[0]])


class DragDriver:
    """Mouse handler for one ``drag`` system: right-button press picks in the drawn env, motion moves, release lets go."""

    def __init__(self, sim, drag, renderer, apply):
        self.scene, self.drag, self.renderer, self.apply = sim.scene, drag, renderer, apply
        b = drag.body
        self.body = b
        self.inv_mass = b.rest.inv_mass  # (E, n): each env's own row (a mesh variant pins other nodes)
        self.host = wp.zeros(b.count, dtype=wp.vec3, device="cpu", pinned=True)
        self.env = self.node = None
        self.plane_point = self.plane_normal = self.offset = None
        self.pending = False
        renderer.mouse_handlers.append(self.handle_mouse)

    def _viewport(self):
        w = getattr(self.renderer, "window", None)
        return (float(w.width), float(w.height)) if w is not None else (float(self.renderer.width), float(self.renderer.height))

    def positions(self, env: int) -> np.ndarray:
        """The drawn env's node positions from the held snapshot (host copy on the renderer's stream)."""
        r = self.renderer
        snap = r.snap if r.snap is not None else r.live
        stream = r.stream or wp.get_stream(self.scene.device)
        with r.lock:
            wp.copy(self.host, self.scene.view(snap.q, self.body)[env], stream=stream)
        wp.synchronize_stream(stream)
        return self.host.numpy().copy()

    def handle_mouse(self, event) -> bool:
        button = str(getattr(getattr(event, "button", None), "name", "")).lower()
        pos = (float(event.pos.x), float(event.pos.y)) if getattr(event, "pos", None) is not None else None
        if event.is_button_down() and button == "right" and pos is not None:
            return self.press(*pos)
        if event.is_move() and self.env is not None and pos is not None:
            self.motion(*pos)
            return True
        if event.is_button_up() and button == "right":
            if self.env is None:  # the press missed the vessel and went to the camera: the release is the camera's too
                return False
            self.release()
            return True
        return False

    def press(self, px: float, py: float) -> bool:
        env = int(self.renderer.env)
        width, height = self._viewport()
        points = self.positions(env)
        node = pick_node(points, self.inv_mass[env], self.renderer.camera, px, py, width, height)
        if node is None:
            return False
        origin, direction = self.renderer.camera.ray(px, py, width, height)
        _, _, _, forward = self.renderer.camera.basis()
        hit = ray_plane(origin, direction, points[node], forward)
        if hit is None:
            return False
        self.env, self.node = env, node
        self.plane_point, self.plane_normal, self.offset = points[node].copy(), forward.copy(), points[node] - hit
        target = points[node].copy()
        self.apply(lambda sim, d=self.drag, e=env, n=node, t=target: d.begin(e, n, t))
        return True

    def motion(self, px: float, py: float) -> None:
        if self.pending:  # at most one queued move: the worker applies edits between steps
            return
        width, height = self._viewport()
        origin, direction = self.renderer.camera.ray(px, py, width, height)
        hit = ray_plane(origin, direction, self.plane_point, self.plane_normal)
        if hit is None:
            return
        target = hit + self.offset
        self.pending = True

        def edit(sim, d=self.drag, e=self.env, t=target):
            d.move(e, t)
            self.pending = False

        self.apply(edit)

    def release(self) -> None:
        if self.env is None:
            return
        env = self.env
        self.env = self.node = None
        self.apply(lambda sim, d=self.drag, e=env: d.release(e))
