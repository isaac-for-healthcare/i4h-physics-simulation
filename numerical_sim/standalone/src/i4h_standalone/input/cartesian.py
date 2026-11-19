# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from i4h-v2 training/src/i4h_poking/teleop_input.py; see NOTICE.
"""Input-only position sources and relative teleoperation; no simulator imports."""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PositionSample:
    position: np.ndarray | None
    clutch: bool = False
    status: str = "tracking"


def axis_mapping(spec):
    """Signed permutation of raw device axes, e.g. 'x,-z,y'."""
    tokens = spec.lower().split(",")
    if len(tokens) != 3 or {s.lstrip("+-") for s in tokens} != {"x", "y", "z"}:
        raise ValueError("axis map must use each of x,y,z once, e.g. x,-z,y")
    matrix = np.zeros((3, 3))
    for row, token in enumerate(tokens):
        if token not in ("x", "y", "z", "+x", "+y", "+z", "-x", "-y", "-z"):
            raise ValueError("invalid signed axis")
        matrix[row, "xyz".index(token[-1])] = -1 if token.startswith("-") else 1
    return matrix


class RelativePositionControl:
    """Re-anchor after clutch, focus loss, reset or invalid tracking, without jumps."""

    def __init__(self, *, gain=1.0, axes="x,y,z", speed=0.04):
        if not all(math.isfinite(x) and x > 0 for x in (gain, speed)):
            raise ValueError("gain and speed must be positive and finite")
        self.gain, self.speed = gain, speed
        self.axes = axis_mapping(axes)
        self.recenter()

    def recenter(self):
        self.anchor = self.target_anchor = None

    def target(self, sample, current_target, dt, lower, upper, *, clutch=False):
        current = np.asarray(current_target, dtype=np.float64).copy()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        position = (
            None
            if sample.position is None
            else np.asarray(sample.position, dtype=np.float64)
        )
        if (
            clutch
            or sample.clutch
            or position is None
            or position.shape != (3,)
            or not np.isfinite(position).all()
        ):
            self.recenter()
            return current
        position = self.axes @ position
        if self.anchor is None:
            self.anchor, self.target_anchor = position.copy(), current.copy()
        desired = np.clip(
            self.target_anchor + self.gain * (position - self.anchor), lower, upper
        )
        delta = desired - current
        length = np.linalg.norm(delta)
        return current + delta * min(1.0, self.speed * dt / max(length, 1e-12))


class KeyboardInput:
    def __init__(self, keys, speed):
        self.keys, self.speed = keys, speed
        self.position = np.zeros(3)

    def sample(self, dt):
        keys = self.keys
        direction = np.array(
            [
                int(a in keys) - int(b in keys)
                for a, b in (("KeyW", "KeyS"), ("KeyA", "KeyD"), ("KeyQ", "KeyE"))
            ]
        )
        self.position += (
            direction * self.speed * dt / max(1.0, np.linalg.norm(direction))
        )
        return PositionSample(self.position.copy(), status="keyboard")

    def close(self):
        pass


class ReplayInput:
    """Finite list of raw positions in metres; null positions simulate tracking loss."""

    def __init__(self, path):
        self.frames = json.loads(Path(path).read_text())
        if not isinstance(self.frames, list) or not self.frames:
            raise ValueError("replay must be a nonempty JSON list")
        self.index = 0

    def sample(self, dt):
        if self.index == len(self.frames):
            return PositionSample(None, status="replay finished; holding")
        frame = self.frames[self.index]
        self.index += 1
        return PositionSample(
            frame.get("position"), bool(frame.get("clutch", False)), "replay"
        )

    def close(self):
        pass


class FollouPositionInput:
    def __init__(self, device_index=0):
        from i4h_standalone.input.follou import MiniMouController

        self.device = MiniMouController(device_index=device_index)

    def sample(self, dt):
        try:
            return PositionSample(self.device.read_raw_position(), status="MiniMou tracking")
        except Exception as exc:  # SDK failures freeze motion and discard anchors.
            return PositionSample(None, status=f"MiniMou tracking lost: {exc}")

    def close(self):
        self.device.close()


class HDErrorInfo(ct.Structure):
    _fields_ = [
        ("errorCode", ct.c_uint),
        ("internalErrorCode", ct.c_int),
        ("hHD", ct.c_uint),
    ]


class OpenHapticsInput:
    """Linux HDAPI input-only adapter, sampled synchronously on the HD scheduler.

    Force output is disabled. No Python callback runs continuously at servo rate.
    Only one HDAPI owner is supported in this process.
    """

    FORCE_OUTPUT = 0x4000
    CURRENT_POSITION = 0x2050
    CURRENT_BUTTONS = 0x2000
    CALIBRATION_OK = 0x5000
    CALLBACK = ct.CFUNCTYPE(ct.c_uint, ct.c_void_p)

    def __init__(self, device_name="Default Device", library=None):
        name = library or ctypes_library()
        try:
            self.hd = ct.CDLL(name)
        except OSError as exc:
            raise RuntimeError(
                "OpenHaptics libHD.so is unavailable. Install the OpenHaptics SDK/runtime and device driver, "
                "configure/calibrate the device with its vendor utility, then use --hd-library /path/to/libHD.so "
                "if it is outside the library search path."
            ) from exc
        signatures = {
            "hdInitDevice": ([ct.c_char_p], ct.c_uint),
            "hdDisableDevice": ([ct.c_uint], None),
            "hdMakeCurrentDevice": ([ct.c_uint], None),
            "hdDisable": ([ct.c_uint], None),
            "hdStartScheduler": ([], None),
            "hdStopScheduler": ([], None),
            "hdBeginFrame": ([ct.c_uint], None),
            "hdEndFrame": ([ct.c_uint], None),
            "hdGetDoublev": ([ct.c_uint, ct.POINTER(ct.c_double)], None),
            "hdGetIntegerv": ([ct.c_uint, ct.POINTER(ct.c_int)], None),
            "hdGetError": ([], HDErrorInfo),
            "hdCheckCalibration": ([], ct.c_uint),
            "hdScheduleSynchronous": ([self.CALLBACK, ct.c_void_p, ct.c_ushort], None),
        }
        for symbol, (args, result) in signatures.items():
            fn = getattr(self.hd, symbol)
            fn.argtypes, fn.restype = args, result
        self.handle = None
        self.scheduler_started = False
        try:
            handle = self.hd.hdInitDevice(device_name.encode())
            self._check_error("initialize device")
            self.handle = handle
            self.hd.hdMakeCurrentDevice(handle)
            self.hd.hdDisable(self.FORCE_OUTPUT)
            self._check_error("disable force output")
            self.hd.hdStartScheduler()
            self.scheduler_started = True
            self._check_error("start scheduler")
        except BaseException:
            self.close()
            raise

    def _check_error(self, operation):
        error = self.hd.hdGetError()
        if error.errorCode:
            raise RuntimeError(
                f"OpenHaptics could not {operation}: HD error 0x{error.errorCode:04x}"
            )

    def sample(self, dt):
        if self.handle is None:
            return PositionSample(None, status="OpenHaptics closed")
        result = PositionSample(None, status="OpenHaptics waiting")

        @self.CALLBACK
        def read(_):
            nonlocal result
            try:
                self.hd.hdMakeCurrentDevice(self.handle)
                self.hd.hdBeginFrame(self.handle)
                try:
                    position, buttons = (ct.c_double * 3)(), ct.c_int()
                    self.hd.hdGetDoublev(self.CURRENT_POSITION, position)
                    self.hd.hdGetIntegerv(self.CURRENT_BUTTONS, ct.byref(buttons))
                    calibration = self.hd.hdCheckCalibration()
                finally:
                    self.hd.hdEndFrame(self.handle)
                self._check_error("read pose")
                if calibration != self.CALIBRATION_OK:
                    raise RuntimeError(
                        "Device requires calibration in the OpenHaptics vendor utility"
                    )
                result = PositionSample(
                    np.array(position) * 0.001,
                    bool(buttons.value & 1),
                    "OpenHaptics tracking",
                )
            except Exception as exc:  # noqa: BLE001 - exceptions must never escape a ctypes callback.
                result = PositionSample(
                    None, status=f"OpenHaptics tracking lost: {exc}"
                )
            return 0  # HD_CALLBACK_DONE

        self.hd.hdScheduleSynchronous(read, None, 32768)
        try:
            self._check_error("schedule pose read")
        except RuntimeError as exc:
            return PositionSample(None, status=str(exc))
        return result

    def close(self):
        if self.scheduler_started:
            self.hd.hdStopScheduler()
            self.scheduler_started = False
        if self.handle is not None:
            self.hd.hdDisableDevice(self.handle)
            self.handle = None


def ctypes_library():
    return ct.util.find_library("HD") or "libHD.so"


class TeleopKeys:
    """W3C key listener for both Kit and Newton viewers."""

    def __init__(self):
        self.held = set()
        self.reset = self.recenter = self.quit = self.paused = False

    def on_key(self, key, pressed):
        if not pressed:
            self.held.discard(key)
        elif key not in self.held:
            self.held.add(key)
            if key == "Space":
                self.paused = not self.paused
                self.recenter = True
            elif key == "KeyR":
                self.reset = True
            elif key == "KeyC":
                self.recenter = True
            elif key == "Escape":
                self.quit = True

    def focus_lost(self):
        self.held.clear()
        self.paused = self.recenter = True

    @property
    def clutched(self):
        return self.paused or bool({"ControlLeft", "ControlRight"} & self.held)
