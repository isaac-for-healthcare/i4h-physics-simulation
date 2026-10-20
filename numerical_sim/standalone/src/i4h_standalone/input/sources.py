# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input source backends (omnisurg/input/sources.py plus the keyboard backend).

A source's ``poll()`` returns a sample dict: ``active`` (False: the device has nothing this frame and the
state keeps its pose), ``position`` (3,), ``rotation`` xyzw (4,), ``grip`` in [0, 1], optionally ``button``
(the MiniMou handle switch, replay column 7) and ``buttons`` ({secondary, fire}; keyboard and replay).

Live devices (MiniMou, the fake) can be owned by the app's haptic servo (``app/haptics.py``): the servo calls
``io(force_n)`` once per tick (queue the force, one device round trip, read the pose) and publishes the sample
into ``mailbox``; while a mailbox is attached ``poll()`` returns it instead of touching the device, so one
thread does all of a device's I/O. Replay, keyboard and noop sources are never owned. A live source may carry a
``handle`` (``input/avatar.py``: one instrument-box channel): its samples then take grip and pedals from the box and
``io(force_n, handle_n)`` queues the handle force on the channel before the round trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from i4h_standalone.input.state import INACTIVE


class NoOpInputSource:
    def poll(self) -> dict:
        return INACTIVE

    def close(self) -> None:
        pass


class ReplayInputSource:
    """Replay rows of `[px, py, pz, qx, qy, qz, qw, grip?, energy?, fire?]` from `.npy` (the app's
    ``--record`` output); the last row holds once the file is exhausted."""

    def __init__(self, replay_path: str | Path, *, scale=(1.0, 1.0, 1.0)) -> None:
        self._path = Path(replay_path)
        self._data = np.asarray(np.load(self._path), dtype=np.float32)
        if self._data.ndim == 1:
            self._data = self._data.reshape(1, -1)
        if self._data.ndim != 2 or self._data.shape[1] < 7 or self._data.shape[0] == 0:
            raise ValueError(f"Replay input '{self._path}' requires a 2D .npy array with at least one row and 7 columns")
        self._scale = np.asarray(scale, dtype=np.float32)
        self._frame = 0

    def poll(self) -> dict:
        row = self._data[min(self._frame, len(self._data) - 1)]
        self._frame += 1
        result = {"active": True, "position": row[:3] * self._scale, "rotation": row[3:7]}
        if row.shape[0] >= 8:
            result["button"] = bool(row[7] > 0.5)
            result["grip"] = float(row[7])
        if row.shape[0] >= 10:
            result["buttons"] = {"secondary": bool(row[8] > 0.5), "fire": bool(row[9] > 0.5)}
        return result

    def close(self) -> None:
        pass


class XrHandSource:
    """A Quest controller (backend ``openxr``): the pose the XR renderer published this headset frame, already in world
    coordinates (``xr/hands.py``). The grip is the trigger (or squeeze), energy and fire and the pickup toggle are
    buttons of that hand.

    Force feedback is the controller's vibration: the haptic servo owns the source like any live device (``io``,
    ``mailbox``); the conditioned, capped tool force sets the amplitude ``|F| / full_scale_n`` and, with
    ``vibrate_on: force+jaw`` (``wants_jaw``: the servo conditions the jaw load for it as for a box handle), the squeeze
    adds ``jaw / jaw_full_scale_n``. The XR render thread applies it (``xr/hands.haptic``, stale after 0.1 s)."""

    def __init__(self, settings: dict) -> None:
        self.hand, self.grip = settings["hand"], settings["grip"]
        self.buttons = {"secondary": settings["energy"], "fire": settings["fire"], "pickup": settings["pickup"]}
        self.full_scale_n, self.jaw_full_scale_n = float(settings["full_scale_n"]), float(settings["jaw_full_scale_n"])
        self.wants_jaw = settings["vibrate_on"] == "force+jaw"
        self.mailbox = None  # the servo's latest sample once it owns the source
        self.handle = None  # no instrument-box handle: the jaw term comes through wants_jaw

    def io(self, force_n, handle_n: float = 0.0) -> dict:
        """Servo tick: the vibration amplitude from the force (and the jaw), then the fresh sample."""
        from i4h_standalone.xr import hands

        amplitude = float(np.linalg.norm(np.asarray(force_n, np.float64))) / self.full_scale_n
        if self.wants_jaw:
            amplitude = max(amplitude, max(float(handle_n), 0.0) / self.jaw_full_scale_n)
        hands.set_haptic(self.hand, amplitude if np.isfinite(amplitude) else 0.0)
        return self._sample()

    def poll(self) -> dict:
        if self.mailbox is not None:
            return self.mailbox
        return self._sample()

    def _sample(self) -> dict:
        from i4h_standalone.xr import hands

        s = hands.read(self.hand)
        if not s.get("active", False):
            return INACTIVE
        grip = float(s.get(self.grip, 0.0))
        pressed = s.get("buttons", {})
        return {"active": True, "position": s["position"], "rotation": s["rotation"], "grip": grip, "button": grip >= 0.5,
                "buttons": {name: bool(pressed.get(button, False)) for name, button in self.buttons.items()}}

    def close(self) -> None:
        pass


class FakeSource:
    """A stand-in device for tests and the servo's timing measurement: a fixed pose (``settings`` position, rotation,
    grip, buttons), ``io()`` records every force it is sent with its wall time and sleeps ``io_ms`` (the FFI round trip)."""

    def __init__(self, settings: dict, *, scale=(1.0, 1.0, 1.0)) -> None:
        import time

        self._time = time.perf_counter
        self._sleep = time.sleep
        self.io_s = float(settings["io_ms"]) * 1e-3
        self.sample = {"active": True, "position": np.asarray(settings["position"], np.float32) * np.asarray(scale, np.float32),
                       "rotation": np.asarray(settings["rotation"], np.float32), "grip": float(settings["grip"]),
                       "buttons": dict(settings["buttons"])}
        self.forces: list = []  # (t_wall, (fx, fy, fz)) every force received, newest last
        self.handle_forces: list = []  # (t_wall, N) every handle force received
        self.mailbox = None
        self.handle = None

    def mount_handle(self, handle) -> None:
        self.handle = handle

    def _with_handle(self, sample: dict) -> dict:
        return self.handle.apply(sample) if self.handle is not None else sample

    def poll(self) -> dict:
        return self.mailbox if self.mailbox is not None else self._with_handle(dict(self.sample))

    def io(self, force_n, handle_n: float = 0.0) -> dict:
        if self.handle is not None:
            self.handle_forces.append((self._time(), float(handle_n)))
            self.handle.queue_force(handle_n)
        self.forces.append((self._time(), tuple(float(v) for v in force_n)))
        if self.io_s > 0.0:
            self._sleep(self.io_s)
        return self._with_handle(dict(self.sample))

    def close(self) -> None:
        pass


class DeviceSource:
    """A live device behind an adapter with ``poll()``, ``queue_force(f)`` (optional) and ``close()`` (the MiniMou, the
    Avatar, the instrument box): the servo's ``io`` queues the force and polls; ``poll()`` returns the mailbox once the servo owns it."""

    def __init__(self, adapter, description: str) -> None:
        self.adapter, self._description = adapter, description
        self._reported_failure = False
        self.mailbox = None
        self.handle = None

    def mount_handle(self, handle) -> None:
        """A box handle supplies the grip and buttons; the adapter may bypass its own handle (``on_handle_mounted``)."""
        self.handle = handle
        mounted = getattr(self.adapter, "on_handle_mounted", None)
        if mounted is not None:
            mounted()

    def _with_handle(self, sample: dict) -> dict:
        return self.handle.apply(sample) if self.handle is not None else sample

    def poll(self) -> dict:
        if self.mailbox is not None:
            return self.mailbox
        if self.adapter is None:
            return INACTIVE
        try:
            return self._with_handle(self.adapter.poll())
        except Exception as exc:
            if not self._reported_failure:
                print(f'Live input source "{self._description}" stopped updating: {exc}')
                self._reported_failure = True
            self.close()
            return INACTIVE

    @property
    def error(self) -> str:
        """Why the adapter's last poll failed ("" when it is fine)."""
        return getattr(self.adapter, "last_error", "") if self.adapter is not None else "closed"

    def io(self, force_n, handle_n: float = 0.0) -> dict:
        if self.adapter is None:
            return INACTIVE
        if self.handle is not None:  # first: a failed pose write must not leave the handle's previous command queued
            self.handle.queue_force(handle_n)
        queue = getattr(self.adapter, "queue_force", None)
        if queue is not None:
            queue(force_n)
        return self._with_handle(self.adapter.poll())

    def close(self) -> None:
        if self.adapter is None:
            return
        self.adapter.close()
        self.adapter = None


def create_input_source(config, keyboard=None):
    """``config`` is a normalised ``DeviceConfig``."""
    if config.backend == "fake":
        return FakeSource(config.settings, scale=config.scale)
    if config.backend == "ibcontroller":
        from i4h_standalone.input.avatar import InstrumentBox

        s = config.settings
        return DeviceSource(InstrumentBox.open(s["device_index"], s["channels"]), f"IBController[{s['device_index']}]")
    if config.backend == "avatar":
        from i4h_standalone.input.avatar import Avatar

        s = config.settings  # a handle is attached by InputManager.init() once every source exists
        return DeviceSource(Avatar.open(s["device_index"], scale=config.scale, yaw_axis=s["yaw_axis"], pitch_axis=s["pitch_axis"]),
                            f"Avatar[{s['device_index']}]")
    if config.backend == "keyboard":
        if keyboard is None:
            raise ValueError(f"input {config.id!r}: the keyboard backend needs a window (run with the slang renderer)")
        from i4h_standalone.input.keyboard import KeyboardInputSource

        return KeyboardInputSource(keyboard, config.id)
    if config.backend == "noop":
        return NoOpInputSource()
    if config.backend == "openxr":
        return XrHandSource(config.settings)
    if config.backend == "replay":
        return ReplayInputSource(config.settings["replay_path"], scale=config.scale)
    from i4h_standalone.input.follou import MiniMouController

    s = config.settings
    return DeviceSource(MiniMouController(device_index=s["device_index"], scale=config.scale, handle_readout=s["handle_readout"]),
                        f"MiniMou[{s['device_index']}]")
