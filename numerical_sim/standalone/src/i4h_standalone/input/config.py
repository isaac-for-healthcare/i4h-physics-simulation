# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input device configuration: the subset of omnisurg/config/schema.py:2467-2530 the input package reads.

A device has an ``id`` (instruments bind to it with ``input: <id>``), a ``backend``, a
position ``scale`` applied inside the source, and ``settings`` with the calibration offsets
(``position_offset``, ``rotation_offset`` or ``rotation_offset_euler_degrees``), the MiniMou
``device_index`` / ``handle_readout`` and the replay file. ``keyboard`` is new: the app feeds it
window key events; ``fake`` is a fixed-pose stand-in that records the forces it is sent (tests, the haptic
servo's timing measurement); ``avatar`` is the Follou 4-DOF pivot device (``device_index``, ``yaw_axis``, ``pitch_axis``;
``follou_haptic`` is its old name) and ``ibcontroller`` the instrument box (``device_index``, ``channels``), both in
``input/avatar.py``. A pose device (``minimou``, ``avatar``, ``fake``) may mount one of the box's force-feedback handles
with ``handle: {box: <ibcontroller device id>, channel: k}``: its grip and pedals then come from that channel (a
MiniMou's own handle port is bypassed) and the servo's handle force goes there. The box is optional: when it is not
configured or does not open, ``InputManager.init`` says so and the device keeps its own grip (a MiniMou's
``handle_readout``). The per-device haptics settings
live in the app's ``haptics:`` block (``app/haptics.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from i4h_physics.quat import euler_xyz_degrees_quat, normalize_quat

INPUT_BACKENDS = ("noop", "replay", "minimou", "keyboard", "fake", "avatar", "ibcontroller", "openxr")
XR_BUTTONS = ("a", "b", "x", "y", "thumbstick")
XR_DEFAULTS = {"right": {"energy": "a", "fire": "b"}, "left": {"energy": "x", "fire": "y"}}  # the face buttons of that hand
BACKEND_ALIASES = {"off": "noop", "none": "noop", "follou_haptic": "avatar"}
MINIMOU_HANDLE_READOUTS = ("raw", "normalized")
COMMON_SETTINGS = {"position_offset", "rotation_offset", "rotation_offset_euler_degrees"}
BACKEND_SETTINGS = {  # every settings key a backend reads; anything else is a typo and fails (AGENTS.md: unknown keys fail)
    "noop": set(), "keyboard": set(), "replay": {"replay_path"},
    "minimou": {"device_index", "handle_readout", "handle"},
    "avatar": {"device_index", "yaw_axis", "pitch_axis", "handle"},
    "fake": {"position", "rotation", "grip", "buttons", "io_ms", "handle"},
    "ibcontroller": {"device_index", "channels"},
    "openxr": {"hand", "grip", "energy", "fire", "pickup", "full_scale_n", "jaw_full_scale_n", "vibrate_on"},
}


def normalize_input_backend(value: str | None) -> str:
    backend = "noop" if value is None else str(value).strip().lower()
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in INPUT_BACKENDS:
        raise ValueError(f"unknown input backend {value!r}; expected one of: {', '.join(INPUT_BACKENDS)}")
    return backend


@dataclass
class DeviceConfig:
    id: str
    backend: str = "noop"
    scale: tuple = (1.0, 1.0, 1.0)
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.id = str(self.id)
        self.backend = normalize_input_backend(self.backend)
        self.scale = tuple(float(v) for v in (self.scale if not isinstance(self.scale, (int, float)) else (self.scale,) * 3))
        if len(self.scale) != 3:
            raise ValueError(f"input {self.id!r}: scale must have three components")
        s = dict(self.settings or {})
        if self.backend == "avatar" and ("handle_gain" in s or "handle_max_n" in s):
            raise ValueError(f"input {self.id!r}: handle_gain and handle_max_n moved to the haptics: block (haptics.handles)")
        unknown = sorted(set(s) - COMMON_SETTINGS - BACKEND_SETTINGS[self.backend])
        if unknown:
            raise ValueError(f"input {self.id!r}: unknown settings {unknown} for backend {self.backend!r}; expected among "
                             f"{sorted(COMMON_SETTINGS | BACKEND_SETTINGS[self.backend])}")
        s["position_offset"] = tuple(float(v) for v in s.get("position_offset", (0.0, 0.0, 0.0)))
        if "rotation_offset" in s and "rotation_offset_euler_degrees" in s:
            raise ValueError(f"input {self.id!r}: use only one of rotation_offset or rotation_offset_euler_degrees")
        if "rotation_offset_euler_degrees" in s:
            s["rotation_offset"] = tuple(float(v) for v in euler_xyz_degrees_quat([float(v) for v in s["rotation_offset_euler_degrees"]]))
        else:
            s["rotation_offset"] = tuple(float(v) for v in normalize_quat(s.get("rotation_offset", (0.0, 0.0, 0.0, 1.0))))
        if self.backend == "replay" and not s.get("replay_path"):
            raise ValueError(f"input {self.id!r}: replay backend requires settings.replay_path")
        if self.backend in ("minimou", "avatar", "fake"):  # a force-feedback handle on the instrument box
            handle = s.get("handle")
            if handle is not None:
                handle = dict(handle)
                if set(handle) - {"box", "channel"} or "box" not in handle or int(handle.get("channel", 0)) < 0:
                    raise ValueError(f"input {self.id!r}: handle needs box (an ibcontroller device id) and an optional channel >= 0")
                handle = {"box": str(handle["box"]), "channel": int(handle.get("channel", 0))}
            s["handle"] = handle
        if self.backend == "minimou":
            s["handle_readout"] = str(s.get("handle_readout", "normalized")).strip().lower()
            if s["handle_readout"] not in MINIMOU_HANDLE_READOUTS:
                raise ValueError(f"input {self.id!r}: handle_readout must be one of {', '.join(MINIMOU_HANDLE_READOUTS)}")
        if self.backend in ("minimou", "avatar", "ibcontroller"):
            s["device_index"] = int(s.get("device_index", 0))
            if s["device_index"] < 0:
                raise ValueError(f"input {self.id!r}: device_index must be a non-negative integer")
        if self.backend == "ibcontroller":
            s["channels"] = int(s.get("channels", 6))
            if s["channels"] < 1:
                raise ValueError(f"input {self.id!r}: channels must be at least 1")
        if self.backend == "avatar":
            for name, default in (("yaw_axis", (0.0, 1.0, 0.0)), ("pitch_axis", (1.0, 0.0, 0.0))):
                s[name] = tuple(float(v) for v in s.get(name, default))
                if len(s[name]) != 3 or not any(s[name]):
                    raise ValueError(f"input {self.id!r}: {name} must be a non-zero 3-vector")
        if self.backend == "openxr":  # a Quest controller (render/xr.py publishes it; xr/hands.py)
            hand = str(s.get("hand", "")).lower()
            if hand not in ("left", "right"):
                raise ValueError(f"input {self.id!r}: openxr needs settings.hand left or right")
            s["hand"] = hand
            s["grip"] = str(s.get("grip", "trigger")).lower()
            if s["grip"] not in ("trigger", "squeeze"):
                raise ValueError(f"input {self.id!r}: openxr grip must be trigger or squeeze")
            for name in ("energy", "fire", "pickup"):
                s[name] = str(s.get(name, XR_DEFAULTS[hand].get(name, "thumbstick"))).lower()
                if s[name] not in XR_BUTTONS:
                    raise ValueError(f"input {self.id!r}: openxr {name} must be one of {XR_BUTTONS}")
            s["full_scale_n"] = float(s.get("full_scale_n", 0.5))  # N of tool force at full vibration (the pose cap)
            s["jaw_full_scale_n"] = float(s.get("jaw_full_scale_n", 2.0))  # N of conditioned jaw load at full vibration
            s["vibrate_on"] = str(s.get("vibrate_on", "force+jaw")).lower()
            if not (s["full_scale_n"] > 0.0 and s["jaw_full_scale_n"] > 0.0) or s["vibrate_on"] not in ("force", "force+jaw"):
                raise ValueError(f"input {self.id!r}: openxr full_scale_n and jaw_full_scale_n must be positive, "
                                 "vibrate_on force or force+jaw")
        if self.backend == "fake":
            s["position"] = tuple(float(v) for v in s.get("position", (0.0, 0.0, 0.0)))
            s["rotation"] = tuple(float(v) for v in normalize_quat(s.get("rotation", (0.0, 0.0, 0.0, 1.0))))
            s["grip"] = float(s.get("grip", 0.0))
            s["buttons"] = {k: bool(v) for k, v in dict(s.get("buttons", {})).items()}
            s["io_ms"] = float(s.get("io_ms", 0.3))
            if len(s["position"]) != 3 or s["io_ms"] < 0.0 or set(s["buttons"]) - {"secondary", "fire"}:
                raise ValueError(f"input {self.id!r}: fake needs position (3), io_ms >= 0 and buttons among secondary, fire")
        self.settings = s
