# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input manager: one ``InputDevice`` per configured device polls its source and maps the sample onto an
``InputState`` (omnisurg/input/manager.py with the config types replaced by ``DeviceConfig``). omnisurg's
per-poll haptic output (conditioning, stale timeouts, delivery on the sim thread) is not carried; the app's
servo (``app/haptics.py``) owns the live sources instead and this thread reads their latest sample."""

from __future__ import annotations

import numpy as np
from i4h_physics.quat import quat_mul
from i4h_standalone.input.sources import create_input_source
from i4h_standalone.input.state import InputState


class InputDevice:
    """Configured input device."""

    def __init__(self, config, keyboard=None) -> None:
        self.config = config
        self._keyboard = keyboard
        self.state = InputState(input_id=config.id)
        self._source = None
        self._position, self._rotation = np.zeros(3, np.float32), np.float32([0.0, 0.0, 0.0, 1.0])  # written in place per poll
        self._position_offset = np.asarray(config.settings["position_offset"], np.float32)
        self._rotation_offset = np.asarray(config.settings["rotation_offset"], np.float32)

    def init(self) -> None:
        if self._source is None:
            self._source = create_input_source(self.config, keyboard=self._keyboard)

    def poll(self) -> InputState:
        state = self.state
        sample = self._source.poll() if self._source is not None else {}
        state.active = bool(sample.get("active", False))
        if state.active:
            np.add(np.asarray(sample["position"], np.float32), self._position_offset, out=self._position)
            self._rotation[:] = quat_mul(self._rotation_offset, sample["rotation"])
            state.position, state.rotation = self._position, self._rotation
        # An inactive device keeps its pose and releases its buttons, so release edges fire.
        grip = float(sample.get("grip", 0.0)) if state.active else 0.0
        state.grip = min(max(grip, 0.0), 1.0)
        buttons = sample.get("buttons", {}) if state.active else {}
        state.buttons = {"grab": bool(sample.get("button", grip >= 0.5)) if state.active else False,
                         "secondary": bool(buttons.get("secondary", False)), "fire": bool(buttons.get("fire", False)),
                         "pickup": bool(buttons.get("pickup", False))}  # a device's own take / put-down button (XR controllers)
        return state

    @property
    def source(self):
        return self._source

    def replace_source(self, source) -> None:
        """Poll ``source`` (``poll()`` -> sample dict) instead of the configured backend's, which is closed: an input the
        backends do not cover (``isaaclab.devices.IsaacLabDeviceSource``). The offsets and grip handling stay."""
        if self._source is not None:
            self._source.close()
        self._source = source

    def close(self) -> None:
        if self._source is not None:
            source, self._source = self._source, None
            source.close()


class InputManager:
    """Owns the configured input devices and their states (keyed by device id)."""

    def __init__(self, devices: list[InputDevice]) -> None:
        self.devices = devices
        self.states = {device.config.id: device.state for device in devices}

    @classmethod
    def from_configs(cls, configs, keyboard=None) -> "InputManager":
        return cls([InputDevice(config, keyboard=keyboard) for config in configs])

    def init(self) -> None:
        opened, absent = [], []
        try:
            for device in self.devices:
                try:
                    device.init()
                except Exception as exc:
                    if device.config.backend != "ibcontroller" or getattr(exc, "__notes__", ()):
                        raise
                    print(f"input {device.config.id!r}: instrument box not available ({exc}); its handles fall back to their devices' own grip")
                    absent.append(device)
                    continue
                opened.append(device)
            for device in absent:  # the box is optional: dropped, the handles that name it are not mounted
                self.devices.remove(device)
                del self.states[device.config.id]
            self._pair_handles()
        except BaseException as original:
            for device in reversed(opened):
                try:
                    device.close()
                except BaseException as cleanup:
                    original.add_note(f"input {device.config.id}: cleanup failed: {cleanup}")
            raise

    def _pair_handles(self) -> None:
        """A pose device with ``settings.handle`` mounts one force-feedback handle of an ``ibcontroller`` device: its grip and
        pedals come from that channel and the servo's handle force goes there (input/avatar.py ``Handle``). A box that is
        not configured or did not open leaves the device its own grip (a MiniMou's ``handle_readout``)."""
        from i4h_standalone.input.avatar import Handle

        by_id = {d.config.id: d for d in self.devices}
        for device in self.devices:
            handle = device.config.settings.get("handle")
            if handle is None:
                continue
            box = by_id.get(handle["box"])
            if box is None:
                print(f"input {device.config.id!r}: no instrument box {handle['box']!r}; the grip comes from the device's own handle")
                continue
            if box.config.backend != "ibcontroller":
                raise ValueError(f"input {device.config.id!r}: handle.box {handle['box']!r} must name an ibcontroller device")
            try:
                device.source.mount_handle(Handle(box.source.adapter, handle["channel"]))
            except ValueError as exc:
                raise ValueError(f"input {device.config.id!r}: {exc}") from exc

    def poll(self) -> None:
        for device in self.devices:
            device.poll()

    def close(self) -> None:
        errors = []
        for device in reversed(self.devices):
            try:
                device.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("input device cleanup failed", errors)
