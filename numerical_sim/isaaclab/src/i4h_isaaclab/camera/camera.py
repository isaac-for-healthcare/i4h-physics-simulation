# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free camera with orbit, pan, dolly and fly controls, for any up axis.

Ported from omnisurg/rendering/slang.py:3077-3262 (camera) and 7920-7944, 8055-8088
(camera key and mouse handling), without the UI capture and pyglet symbol mapping.
The shaders take the camera as a basis (position, right, up, forward) plus fov, aspect
and clip planes; see ``basis()``.
"""

from __future__ import annotations

import math
import time

import numpy as np


def _normalize(v, fallback):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.asarray(fallback, dtype=np.float32)


def key_event(event) -> tuple[str, bool, bool, bool]:
    """``(name, press, repeat, release)`` of a slangpy keyboard event; ``name`` is lower-case (``page_up``, ``left_shift``)."""
    name = str(getattr(getattr(event, "key", None), "name", event.key)).lower()
    repeat = bool(event.is_key_repeat()) if hasattr(event, "is_key_repeat") else False  # tests pass stub events
    return name, bool(event.is_key_press()), repeat, bool(event.is_key_release())


def stable_frame(forward, up_hint, alt_hint, world_up):
    """``(right, up, forward)`` of a roll-stable camera looking along ``forward``: ``up`` is the hint made perpendicular to
    ``forward``, falling back to ``alt_hint`` (the frame's other axis), the world up, then world y when degenerate.
    Ported from omniendo/isaaclab/mdp.py:105-161 (catheter_tip_camera_pose: left = up x f, up = f x left); edit: the
    world fallback is the case's up axis instead of z."""
    f = _normalize(forward, (0.0, 0.0, -1.0))
    up = None
    for hint in (up_hint, alt_hint, world_up, (0.0, 1.0, 0.0)):
        h = np.asarray(hint, np.float32)
        candidate = h - float(np.dot(h, f)) * f
        if float(np.linalg.norm(candidate)) >= 1e-5:
            up = candidate / float(np.linalg.norm(candidate))
            break
    if up is None:  # forward along y as well: any perpendicular
        up = _normalize(np.cross(f, (1.0, 0.0, 0.0)), (0.0, 0.0, 1.0))
    left = _normalize(np.cross(up, f), (1.0, 0.0, 0.0))
    up = _normalize(np.cross(f, left), up)
    return -left, up, f


def _rotate(v, axis, angle):
    axis = _normalize(axis, (0.0, 1.0, 0.0))
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


class Camera:
    def __init__(self, position=(0.0, 1.2, -4.0), pitch=0.0, yaw=0.0, fov=45.0, up=(0.0, 1.0, 0.0),
                 near=0.01, far=100.0):
        self.up = _normalize(up, (0.0, 1.0, 0.0))
        self.position = np.asarray(position, dtype=np.float32)
        self.fov = float(np.clip(fov, 1.0, 179.0))
        self.near, self.far = float(near), float(far)
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError("camera.near must be positive and smaller than camera.far")
        self.target = self.position + self.forward_from(pitch, yaw)
        self.home = (self.position.copy(), self.target.copy())
        self.closed = False
        self._keys: set[str] = set()
        self._fast = self._slow = False
        self._drag: str | None = None
        self._mouse: tuple[float, float] | None = None
        self._last_move = time.perf_counter()

    # geometry ---------------------------------------------------------------

    def forward_from(self, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """Pitch and yaw are measured from a reference forward perpendicular to ``up``.

        For Y-up this is omnisurg's ``(cos p sin y, sin p, -cos p cos y)`` exactly."""
        up = self.up
        fwd0 = np.array([0.0, 0.0, -1.0], np.float32) if abs(float(up[2])) < 0.9 else np.array([0.0, 1.0, 0.0], np.float32)
        right0 = _normalize(np.cross(fwd0, up), (1.0, 0.0, 0.0))
        fwd0 = _normalize(np.cross(up, right0), fwd0)
        p, y = math.radians(pitch_deg), math.radians(yaw_deg)
        return _normalize(math.cos(p) * (math.sin(y) * right0 + math.cos(y) * fwd0) + math.sin(p) * up, fwd0)

    def basis(self):
        """``(position, right, up, forward)`` as float32 arrays; right-handed, up derived from world up."""
        forward = _normalize(self.target - self.position, (0.0, 0.0, -1.0))
        right = _normalize(np.cross(forward, self.up), (1.0, 0.0, 0.0))
        up = _normalize(np.cross(right, forward), self.up)
        return self.position.astype(np.float32), right, up, forward

    def project(self, points, width: float, height: float) -> np.ndarray:
        """``(n, 3)`` window coordinates (x right, y down, in pixels) and camera depth of world ``points``; the same
        perspective as the shaders (tissue.slang ``project``: clip = (x / (tan_half aspect), y / tan_half) / depth)."""
        pos, right, up, forward = self.basis()
        rel = np.asarray(points, np.float32).reshape(-1, 3) - pos
        depth = rel @ forward
        tan_half = math.tan(math.radians(self.fov) * 0.5)
        aspect = float(width) / max(float(height), 1.0)
        safe = np.where(np.abs(depth) > 1e-8, depth, 1e-8)
        x = ((rel @ right) / (safe * tan_half * aspect) + 1.0) * 0.5 * float(width)
        y = (1.0 - (rel @ up) / (safe * tan_half)) * 0.5 * float(height)
        return np.stack([x, y, depth], axis=1).astype(np.float32)

    def view_projection(self, width: float, height: float) -> np.ndarray:
        """The ``(4, 4)`` float32 world -> clip matrix of ``project`` (row-major, ``clip = M @ [p, 1]``): x right and y up
        scaled by the fov and aspect, z mapped to [0, 1] between ``near`` and ``far`` after the divide, w the camera
        depth. The shaders project with it (``camera_view_projection``), so an XR eye can pass its own off-axis one."""
        pos, right, up, forward = (np.asarray(v, np.float64) for v in self.basis())
        inv_tan = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        aspect = float(width) / max(float(height), 1.0)
        near, far = float(self.near), float(self.far)
        a, b = far / (far - near), -near * far / (far - near)
        m = np.zeros((4, 4), np.float64)
        m[0, :3], m[0, 3] = right * inv_tan / aspect, -right.dot(pos) * inv_tan / aspect
        m[1, :3], m[1, 3] = up * inv_tan, -up.dot(pos) * inv_tan
        m[2, :3], m[2, 3] = forward * a, -forward.dot(pos) * a + b
        m[3, :3], m[3, 3] = forward, -forward.dot(pos)
        return m.astype(np.float32)

    def ray(self, px: float, py: float, width: float, height: float):
        """``(origin, unit direction)`` of the world ray through window pixel ``(px, py)``; the inverse of ``project``."""
        pos, right, up, forward = self.basis()
        tan_half = math.tan(math.radians(self.fov) * 0.5)
        aspect = float(width) / max(float(height), 1.0)
        ndc_x, ndc_y = 2.0 * float(px) / float(width) - 1.0, 1.0 - 2.0 * float(py) / float(height)
        direction = forward + right * (ndc_x * tan_half * aspect) + up * (ndc_y * tan_half)
        return pos.copy(), _normalize(direction, forward)

    def look_at(self, position, target) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def set_pose(self, position, forward, up, lookahead: float = 0.05) -> None:
        """Position, target and up so that ``basis()`` returns ``(position, forward x up, up, forward)``: ``up`` must be
        unit and perpendicular to ``forward`` (``stable_frame`` gives one). ``target`` sits ``lookahead`` ahead
        (omniendo xcath.py:1332, at least 0.01)."""
        self.position = np.asarray(position, dtype=np.float32)
        f = _normalize(forward, (0.0, 0.0, -1.0))
        self.target = self.position + f * max(float(lookahead), 0.01)
        self.up = np.asarray(up, dtype=np.float32)

    def pose(self) -> tuple:
        return self.position.copy(), self.target.copy(), self.up.copy(), self.fov, self.near

    def restore(self, pose) -> None:
        self.position, self.target, self.up = (np.asarray(v, np.float32).copy() for v in pose[:3])
        self.fov, self.near = float(pose[3]), float(pose[4])

    # controls (slang.py:3174-3235) -----------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        rel = self.position - self.target
        distance = float(np.linalg.norm(rel))
        if distance <= 1e-6:
            return
        _, right, _, forward = self.basis()
        rel = _rotate(rel, self.up, -dx * 0.004)
        pitch_axis = _normalize(np.cross(self.up, rel), right)
        pitched = _rotate(rel, pitch_axis, -dy * 0.004)
        if abs(float(np.dot(_normalize(-pitched, forward), self.up))) < 0.98:
            rel = pitched
        self.position = self.target + _normalize(rel, self.position - self.target) * distance

    def pan(self, dx: float, dy: float) -> None:
        _, right, up, _ = self.basis()
        scale = max(float(np.linalg.norm(self.target - self.position)), 0.01) * 0.0015
        offset = (right * -dx + up * dy) * scale
        self.look_at(self.position + offset, self.target + offset)

    def dolly(self, scroll: float) -> None:
        _, _, _, forward = self.basis()
        distance = float(np.linalg.norm(self.target - self.position))
        if distance <= 1e-6:
            return
        distance = float(np.clip(distance * math.exp(-scroll * 0.12), 0.01, 1000.0))
        self.position = self.target - forward * distance

    def move(self) -> None:
        """Apply held WASDQE keys; call once per frame."""
        now = time.perf_counter()
        dt = min(max(now - self._last_move, 0.0), 0.1)
        self._last_move = now
        if not self._keys:
            return
        _, right, _, forward = self.basis()
        direction = np.zeros(3, np.float32)
        for key, vec in (("w", forward), ("s", -forward), ("d", right), ("a", -right), ("e", self.up), ("q", -self.up)):
            if key in self._keys:
                direction += vec
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return
        speed = max(float(np.linalg.norm(self.target - self.position)), 0.25) * 0.75
        speed *= 4.0 if self._fast else 1.0
        speed *= 0.25 if self._slow else 1.0
        offset = direction / length * speed * dt
        self.look_at(self.position + offset, self.target + offset)

    # slangpy events (slang.py:7920-7944, 8055-8088) --------------------------------

    def handle_key(self, event) -> None:
        key, press, repeat, release = key_event(event)
        if key in {"w", "a", "s", "d", "q", "e"}:
            if press or repeat:
                self._keys.add(key)
            elif release:
                self._keys.discard(key)
        elif key in {"shift", "left_shift", "right_shift"}:
            self._fast = not release
        elif key in {"ctrl", "control", "left_ctrl", "left_control", "right_ctrl", "right_control"}:
            self._slow = not release
        elif press or repeat:
            if key == "left":
                self.orbit(-32.0, 0.0)
            elif key == "right":
                self.orbit(32.0, 0.0)
            elif key == "up":
                self.orbit(0.0, -24.0)
            elif key == "down":
                self.orbit(0.0, 24.0)
            elif key == "home":
                self.look_at(*self.home)
            elif key == "escape":
                self.closed = True

    def handle_mouse(self, event) -> None:
        button = str(getattr(getattr(event, "button", None), "name", "")).lower()
        pos = (float(event.pos.x), float(event.pos.y)) if getattr(event, "pos", None) is not None else (0.0, 0.0)
        if event.is_button_up():
            if button in {"left", "right", "middle"}:
                self._drag, self._mouse = None, None
        elif event.is_button_down():
            if button == "left":
                self._drag, self._mouse = "orbit", pos
            elif button in {"right", "middle"}:
                self._drag, self._mouse = "pan", pos
        elif event.is_scroll():
            self.dolly(float(event.scroll.y))
        elif event.is_move() and self._drag is not None:
            previous, self._mouse = self._mouse, pos
            if previous is not None:
                dx, dy = pos[0] - previous[0], pos[1] - previous[1]
                (self.orbit if self._drag == "orbit" else self.pan)(dx, dy)
