# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proximity pickup and toggle: an input device takes the nearest instrument or puts the held one down.

Ported from omnisurg/instruments/controller.py: the per-device toggle key (:57-106, a rising edge; holding a tool
puts it down, holding nothing takes the nearest eligible tool within ``distance``), the grab that keeps the tool
where it is by remembering the device-to-tool offsets (:157-205: ``offset_p = rotate(conj(dev_q), tool_p - dev_p)``,
``offset_q = conj(dev_q) * tool_q``; thereafter ``tool_p = dev_p + rotate(dev_q, offset_p)``), the re-grab guard until
the device has moved farther than ``distance`` from the tool it dropped (:259-289), the soft jaw takeover (:207-226,
tolerance 0.05: the jaw keeps its parked value until the handle comes within the tolerance or crosses it), the
eligible kinds (:243-249) and the idle-device proxies (:286-307, grey; green when a tool is in reach). Config
``inputs.pickup: {distance 0.15, proxy_radius 0.05, keys {device: key}}`` replaces omnisurg's
``inputs.instrument_control: {enabled, pickup_distance, proxy_radius, toggle_keys}`` (the block's absence is off).

Edits: the toggle is a queue of press events (the window's key callback runs on the render thread, ``update`` on the
physics thread; a plain deque bridges them, like the keyboard hub's set); a put-down releases grip, energy and fire
(the source froze the jaw value: a tool dropped mid-grasp kept gripping); holdings survive the drawn-env switch with
the offsets recomputed against that env's rows; ``reset`` restores the ``input:`` bindings with a snap (the source's
``grab``); kinds ``grasper, hook, suction, scissors, clip_applier`` (``irrigation_suction`` -> ``suction``; a
``sphere`` probe is never picked); the console trace is ``[pickup] take|release``. Tool poses are the commanded
action rows (the source's ``control_pose``), never the velocity-clamped root.
"""

from __future__ import annotations

import collections

import numpy as np
from i4h_isaaclab.camera.camera import key_event
from i4h_physics.quat import normalize_quat, quat_conjugate, quat_mul, quat_rotate

ELIGIBLE = ("grasper", "hook", "suction", "scissors", "clip_applier")  # controller.py:243-249
JAW_TOLERANCE = 0.05  # controller.py:17
RESERVED_KEYS = {  # the window, the hub, the commands and the paint keys
    "w", "a", "s", "d", "q", "e", "left", "right", "up", "down", "home", "escape", "left_shift", "right_shift", "shift",
    "left_ctrl", "right_ctrl", "left_control", "right_control", "ctrl", "control", "i", "j", "k", "l", "u", "o", "comma", "period", "space", "f", "r", "tab",
    "p", "n", "backspace", "g", "x", "page_up", "page_down", "left_bracket", "right_bracket", "m", "v", "b", "f1", "c",
}


def key_name(key) -> str:
    """The slangpy key name of a YAML key: digits are ``key1`` .. ``key9``, ``key0``; letters lower-case."""
    text = str(key).strip().lower()
    if len(text) == 1 and text.isdigit():
        return f"key{text}"
    return text


class Pickup:
    """Per input device: which instrument it holds, how (offsets, parked jaw), and the toggle presses to apply."""

    def __init__(self, cfg: dict, records, initial: dict):
        """``records``: the scene's instruments in order; ``initial``: device id -> instrument index bound by ``input:``."""
        cfg = dict(cfg)
        self.distance = float(cfg.pop("distance", 0.15))
        self.proxy_radius = float(cfg.pop("proxy_radius", 0.05))
        keys = dict(cfg.pop("keys", None) or {})
        if cfg:
            raise ValueError(f"inputs.pickup: unknown keys {sorted(cfg)}; known distance, proxy_radius, keys")
        if self.distance <= 0.0 or self.proxy_radius <= 0.0:
            raise ValueError("inputs.pickup: distance and proxy_radius must be positive")
        if not keys:
            raise ValueError("inputs.pickup.keys must map at least one device to a key")
        self.keys = {str(dev): key_name(key) for dev, key in keys.items()}
        seen: dict[str, str] = {}
        for dev, key in self.keys.items():
            if key in RESERVED_KEYS:
                raise ValueError(f"inputs.pickup.keys[{dev!r}] = {key!r} is taken by the window")
            if key in seen:
                raise ValueError(f"inputs.pickup.keys: {dev!r} and {seen[key]!r} share the key {key!r}")
            seen[key] = dev
        self.names = [r.name for r in records]
        self.kinds = [r.kind for r in records]
        self.initial = {dev: (None if k is None else int(k)) for dev, k in initial.items()}
        for dev in self.keys:
            self.initial.setdefault(dev, None)
        self.holding: dict[str, int | None] = dict(self.initial)
        self.offset_p: dict[str, np.ndarray] = {}
        self.offset_q: dict[str, np.ndarray] = {}
        self.guard: dict[str, int] = {}  # device -> the tool it just put down
        self.jaw: dict[str, tuple[float, float, bool]] = {}  # device -> (parked grip, previous handle, takeover done)
        self.toggles: collections.deque = collections.deque()  # device ids whose key was pressed (render thread -> physics thread)
        self._button = {dev: False for dev in self.keys}  # a device's own pickup button (XR controllers): toggles on the press
        for dev, k in self.holding.items():
            if k is not None:
                self._snap(dev)

    # render thread ------------------------------------------------------------------------------------------------

    def handle_key(self, event) -> None:
        name, press, repeat, _ = key_event(event)
        if press and not repeat:
            for dev, key in self.keys.items():
                if key == name:
                    self.toggles.append(dev)

    # physics thread -----------------------------------------------------------------------------------------------

    def update(self, states: dict, rows: np.ndarray) -> None:
        """Once per frame: refresh the re-grab guards and apply the queued presses. ``rows`` is the drawn env's ``(n, 10)``
        action table (the commanded tool poses)."""
        for dev in self.keys:
            self._refresh_guard(dev, states.get(dev), rows)
            state = states.get(dev)
            pressed = bool(state is not None and state.active and state.buttons.get("pickup", False))
            if pressed and not self._button[dev]:
                self.toggles.append(dev)
            self._button[dev] = pressed
        while self.toggles:
            dev = self.toggles.popleft()
            state = states.get(dev)
            if state is None or not state.active:
                continue  # controller.py:79-80: a silent device cannot take or drop anything
            k = self.holding.get(dev)
            if k is not None:
                self.release(dev)
            else:
                k = self.nearest(dev, state.position, rows)
                if k is not None:
                    self.take(dev, k, state, rows[k])

    def take(self, dev: str, k: int, state, row) -> None:
        """The tool stays where it is: remember the device-to-tool offsets and park its jaw (controller.py:157-186)."""
        dev_q = normalize_quat(state.rotation)
        inv = quat_conjugate(dev_q)
        self.offset_p[dev] = quat_rotate(inv, np.asarray(row[:3], np.float32) - np.asarray(state.position, np.float32))
        self.offset_q[dev] = normalize_quat(quat_mul(inv, normalize_quat(row[3:7])))
        parked = float(row[7])
        self.jaw[dev] = (parked, float(state.grip), abs(float(state.grip) - parked) <= JAW_TOLERANCE)
        self.holding[dev] = int(k)
        self.guard.pop(dev, None)
        gap = float(np.linalg.norm(np.asarray(row[:3], np.float32) - np.asarray(state.position, np.float32)))
        print(f"[pickup] take {dev} -> {self.names[k]} ({gap:.3f} away, jaw parked {parked:.2f})")

    def release(self, dev: str) -> int | None:
        k = self.holding.get(dev)
        if k is None:
            return None
        self.holding[dev] = None
        self.guard[dev] = k
        self.jaw.pop(dev, None)
        print(f"[pickup] release {dev} -> {self.names[k]}")
        return k

    def nearest(self, dev: str, position, rows) -> int | None:
        """The nearest eligible, unheld, unguarded tool within ``distance`` of the device (controller.py:228-257)."""
        held = {k for k in self.holding.values() if k is not None}
        best, best_d = None, self.distance
        for k, kind in enumerate(self.kinds):
            if kind not in ELIGIBLE or k in held or self.guard.get(dev) == k:
                continue
            d = float(np.linalg.norm(np.asarray(rows[k, :3], np.float32) - np.asarray(position, np.float32)))
            if d <= best_d:
                best, best_d = k, d
        return best

    def _refresh_guard(self, dev: str, state, rows) -> None:
        """The dropped tool becomes takeable again once the device has moved farther than ``distance`` from it (:259-289)."""
        k = self.guard.get(dev)
        if k is None or state is None or not state.active:
            return
        if float(np.linalg.norm(np.asarray(rows[k, :3], np.float32) - np.asarray(state.position, np.float32))) > self.distance:
            del self.guard[dev]

    def pose(self, dev: str, position, rotation):
        """The held tool's pose from the device pose and the offsets (controller.py:188-205)."""
        dev_q = normalize_quat(rotation)
        p = np.asarray(position, np.float32) + quat_rotate(dev_q, self.offset_p[dev])
        return p, normalize_quat(quat_mul(dev_q, self.offset_q[dev]))

    def grip(self, dev: str, handle: float) -> float:
        """Soft jaw takeover (controller.py:207-226): the parked grip until the handle is within the tolerance or crosses it."""
        state = self.jaw.get(dev)
        if state is None:
            return float(handle)
        parked, previous, done = state
        if not done:
            crossed = (previous <= parked <= handle) or (handle <= parked <= previous)
            done = abs(float(handle) - parked) <= JAW_TOLERANCE or crossed
        self.jaw[dev] = (parked, float(handle), done)
        return float(handle) if done else parked

    def rebind(self, states: dict, rows: np.ndarray) -> None:
        """The drawn env changed: keep the holdings, recompute the offsets against this env's rows (no jump), drop the guards."""
        self.guard.clear()
        for dev, k in list(self.holding.items()):
            state = states.get(dev)
            if k is None or state is None or not state.active:
                continue
            self.take(dev, k, state, rows[k])

    def reset(self) -> None:
        """Back to the ``input:`` bindings, tools snapped to their devices."""
        self.holding = dict(self.initial)
        self.offset_p.clear()
        self.offset_q.clear()
        self.guard.clear()
        self.jaw.clear()
        self.toggles.clear()
        for dev, k in self.holding.items():
            if k is not None:
                self._snap(dev)

    def _snap(self, dev: str) -> None:  # controller.py:108-133 (grab): zero offsets, the tool follows the device exactly
        self.offset_p[dev] = np.zeros(3, np.float32)
        self.offset_q[dev] = np.float32([0.0, 0.0, 0.0, 1.0])

    def proxies(self, states: dict, rows: np.ndarray) -> list:
        """``(device, position, radius, candidate)`` per pickup device: a sphere of ``proxy_radius`` while the active device holds
        nothing (radius 0 otherwise), ``candidate`` when a tool is in reach (controller.py:286-307)."""
        out = []
        for dev in self.keys:
            state = states.get(dev)
            idle = state is not None and state.active and self.holding.get(dev) is None
            position = np.asarray(state.position, np.float32) if state is not None else np.zeros(3, np.float32)
            candidate = idle and self.nearest(dev, position, rows) is not None
            out.append((dev, position, self.proxy_radius if idle else 0.0, bool(candidate)))
        return out
