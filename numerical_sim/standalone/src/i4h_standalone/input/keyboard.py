# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyboard instrument driving for the Slang window (new; omnisurg had no keyboard source).

One ``KeyboardHub`` receives the window's key events and holds a pose per device id; Tab
cycles the active device. Held keys move the active device between polls at ``speed``
metres per second in camera axes: I/K forward/back, J/L left/right, U/O down/up (world
up), comma/period roll about the viewing direction (the shaft axis is set by the trocar), Space
closes the jaw (grip 1; on a hook the driver turns it into energy), Shift energises (cautery,
omnisurg's ``activation_button: left_shift``), F fires (a hook cuts, a grasper applies a clip).
``R`` returns the active device to its initial pose (``reset()`` every device, with its buttons, after a
simulation reset). ``KeyboardInputSource`` is the
``InputSource`` view of one device for the input manager; energy and fire travel as the
``secondary`` and ``fire`` buttons. The catheter driver reads the same I/K J/L U/O keys, so a
case with both an instrument and a catheter would drive both from one press; no case does.
"""

from __future__ import annotations

import time

import numpy as np
from i4h_isaaclab.camera.camera import key_event
from i4h_physics.quat import axis_angle_quat, quat_mul

MOVE_KEYS = {"i": (0, 1.0), "k": (0, -1.0), "l": (1, 1.0), "j": (1, -1.0), "o": (2, 1.0), "u": (2, -1.0)}  # axis, sign
ROLL_KEYS = {"comma": -1.0, "period": 1.0}
HELD_KEYS = {"space", "left_shift", "right_shift", "f"}  # grip, energy, energy, fire
PAINT_KEYS = {"m": 0, "v": 1, "b": 2}  # contact paint: damage, coagulation, blood (omnisurg used M/N/B; N is the window's step-once key)


class HeldKeys:
    """The keys of ``keys`` currently down: ``handle_key`` on the render thread, ``in`` on the physics thread (a plain
    set bridges them; iterate a ``tuple(held)`` since the callback mutates it). ``handle_key`` returns ``(name, pressed)``
    (pressed = press or repeat) for the caller's own edge-triggered keys."""

    def __init__(self, keys):
        self.keys, self.down = frozenset(keys), set()

    def handle_key(self, event) -> tuple[str, bool]:
        name, press, repeat, release = key_event(event)
        if name in self.keys:
            if press or repeat:
                self.down.add(name)
            elif release:
                self.down.discard(name)
        return name, press or repeat

    def __contains__(self, name) -> bool:
        return name in self.down

    def __iter__(self):
        return iter(tuple(self.down))


class PaintKeys:
    """Held paint keys -> a ``(3,)`` mask for ``contact_paint``."""

    def __init__(self):
        self.held = HeldKeys(PAINT_KEYS)
        self.handle_key = self.held.handle_key

    def mask(self) -> np.ndarray:
        return np.float32([1.0 if key in self.held else 0.0 for key in PAINT_KEYS])


class KeyboardHub:
    def __init__(self, devices: dict, camera, speed: float = 0.25, roll_speed: float = 1.5):
        """``devices``: id -> (initial position (3,), initial rotation xyzw (4,)). ``camera`` gives the move axes."""
        self.initial = {k: (np.asarray(p, np.float32).copy(), np.asarray(r, np.float32).copy()) for k, (p, r) in devices.items()}
        self.position = {k: p.copy() for k, (p, _) in self.initial.items()}
        self.rotation = {k: r.copy() for k, (_, r) in self.initial.items()}
        self.grip = {k: 0.0 for k in devices}
        self.energy = {k: 0.0 for k in devices}
        self.fire = {k: 0.0 for k in devices}
        self.order = list(devices)
        self.active = self.order[0] if self.order else None
        self.camera = camera
        self.speed, self.roll_speed = float(speed), float(roll_speed)
        self.held = HeldKeys(set(MOVE_KEYS) | set(ROLL_KEYS) | HELD_KEYS)
        self._last = time.perf_counter()

    def handle_key(self, event) -> None:
        name, press = self.held.handle_key(event)
        if press and name == "tab" and self.order:
            self.active = self.order[(self.order.index(self.active) + 1) % len(self.order)]
        elif press and name == "r" and self.active is not None:
            self.reset(self.active)

    def reset(self, device=None) -> None:
        """Initial pose and released buttons for one device (or all)."""
        for k in self.order if device is None else (device,):
            self.position[k], self.rotation[k] = (a.copy() for a in self.initial[k])
            self.grip[k] = self.energy[k] = self.fire[k] = 0.0

    def advance(self) -> None:
        """Apply held keys to the active device; call once per frame before polling."""
        now = time.perf_counter()
        dt = min(max(now - self._last, 0.0), 0.1)
        self._last = now
        k = self.active
        if k is None:
            return
        for other in self.order:  # momentary controls belong to the active tool only; grip stays latched on Tab
            if other != k:
                self.energy[other] = self.fire[other] = 0.0
        self.grip[k] = 1.0 if "space" in self.held else 0.0
        self.energy[k] = 1.0 if ("left_shift" in self.held or "right_shift" in self.held) else 0.0
        self.fire[k] = 1.0 if "f" in self.held else 0.0
        _, right, up, forward = self.camera.basis()
        axes = (forward, right, np.asarray(self.camera.up, np.float32))
        move = np.zeros(3, np.float32)
        for key, (axis, sign) in MOVE_KEYS.items():
            if key in self.held:
                move += sign * axes[axis]
        if np.linalg.norm(move) > 1e-6:
            self.position[k] = self.position[k] + move / np.linalg.norm(move) * self.speed * dt
        roll = sum(sign for key, sign in ROLL_KEYS.items() if key in self.held)
        if roll:
            tool_axis = forward  # roll about the viewing direction: the shaft axis is set by the trocar anyway
            self.rotation[k] = quat_mul(axis_angle_quat(tool_axis, roll * self.roll_speed * dt), self.rotation[k])


class KeyboardInputSource:
    def __init__(self, hub: KeyboardHub, device_id: str):
        self.hub, self.id = hub, device_id
        if device_id not in hub.position:
            raise ValueError(f"keyboard input {device_id!r} is not one of the hub's devices {list(hub.position)}")

    def poll(self) -> dict:
        return {"active": True, "position": self.hub.position[self.id].copy(),
                "rotation": self.hub.rotation[self.id].copy(), "grip": self.hub.grip[self.id],
                "buttons": {"secondary": self.hub.energy[self.id] > 0.5, "fire": self.hub.fire[self.id] > 0.5}}

    def close(self) -> None:
        pass
