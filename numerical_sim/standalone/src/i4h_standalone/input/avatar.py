# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Follou Avatar (a 4-DOF laparoscopic port: yaw, pitch, insertion, roll about a fixed pivot), the instrument box
(IBController: up to six 1-DOF force-feedback handles with an opening sensor, two pedals) and the ``Handle`` that
mounts one of the box's channels on a pose device (Avatar or MiniMou).

The SDK classes (``follou_sdk/devices/haptic.py``, ``ibcontroller.py``) were never driven by omnisurg; the mappings
here are new and the axis conventions are the SDK's documented ones, confirmed on the hardware step by step. The
adapters take an SDK controller, or any object with the same getters, so the mappings are tested with stubs.

Avatar -> tool pose: the SDK reports the tip position (cm) and the tool direction in the device frame with the pivot
at the origin, and the roll angle. The sample position is the tip in metres times ``scale``; the generic
``position_offset`` (the case's trocar, the pivot in the sim frame) and ``rotation_offset`` complete the mapping, so
``instrument_control``'s RCM sees the shaft through its trocar. The rotation is the shaft direction with the roll
about it (``quat_from_direction_roll``). The Avatar has no grip of its own: a ``Handle`` supplies it.

Avatar force: the servo sends the force on the tool (N, sim frame; the device's translational axes are the sim's).
``set_force_and_torques(z, yaw, pitch, rot)``: the insertion force is the component along the tool direction, the
yaw and pitch torques are the moment of the force about the pivot, ``r x F`` with ``r`` the tip position (m),
projected on the device's yaw and pitch axes (``settings.yaw_axis``, ``pitch_axis``; defaults y and x, signs
measured on hardware), roll torque 0.

Handle (``settings.handle: {box, channel}`` on any pose device, attached by ``InputManager.init``): the box channel's
opening replaces the device's own grip readout (a MiniMou's handle port then goes unused), the box's two pedals are
the ``secondary`` (energy) and ``fire`` buttons, and the servo's handle force (the tool's jaw load, conditioned) is
queued on the channel and leaves with the box's own round trip. A box that stops answering leaves the pose active
with grip 0 and no pedals. The box binds to no instrument and has no pose; its sample is always inactive.
"""

from __future__ import annotations

import math

import numpy as np
from i4h_physics.quat import normalize_quat, quat_mul
from i4h_standalone.input.state import INACTIVE

CM = 0.01  # the SDK reports centimetres


class ROT:  # HapticDevice.get_value takes a Dof member; the SDK reads ``.value`` (Dof.ROT = 0)
    value = 0


def quat_from_direction_roll(direction, roll: float) -> np.ndarray:
    """A unit quaternion (xyzw) whose local -z points along ``direction`` (the shaft convention of the instrument
    assets), rolled by ``roll`` radians about it."""
    d = np.asarray(direction, np.float64)
    n = float(np.linalg.norm(d))
    if n <= 1e-9:
        return np.float32([0.0, 0.0, 0.0, 1.0])
    d = d / n
    z = np.array([0.0, 0.0, -1.0])
    c = float(np.dot(z, d))
    if c < -1.0 + 1e-9:  # antiparallel: half turn about x
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = np.cross(z, d)
        s = math.sqrt((1.0 + c) * 2.0)
        q = np.array([axis[0] / s, axis[1] / s, axis[2] / s, s / 2.0])
    twist = np.array([d[0] * math.sin(roll / 2.0), d[1] * math.sin(roll / 2.0), d[2] * math.sin(roll / 2.0), math.cos(roll / 2.0)])
    return normalize_quat(quat_mul(twist, q)).astype(np.float32)


def close_controller(adapter) -> None:
    """An adapter from ``open()`` releases its share of the SDK manager, which closes the controllers with the last share
    (as a MiniMou does), so a reopen rescans; an injected controller (tests) is closed directly. Once only."""
    if getattr(adapter, "closed", False):
        return
    adapter.closed = True
    if getattr(adapter, "owns_manager", False):
        from i4h_standalone.input.follou import release_manager

        release_manager()
        return
    close = getattr(adapter.controller, "close", None)
    if callable(close):
        close()


class InstrumentBox:
    """The IBController adapter: one ``poll()`` per servo tick reads every channel's opening, measured force and handle id
    and the pedals; the handle forces queued by the pose devices go out with that same round trip."""

    def __init__(self, controller, channels: int = 6):
        self.controller, self.channels = controller, int(channels)
        self.openings = np.zeros(self.channels, np.float32)  # 0 open .. 1 closed
        self.measured = np.zeros(self.channels, np.float32)  # the handle's force sensor (N, the SDK's opto force)
        self.ids = np.zeros(self.channels, np.int32)  # the handle id the box reports per channel (0: none), status only
        self.queued = np.zeros(self.channels, np.float32)  # the last force queued per channel (status)
        self.pwm = np.zeros(self.channels, np.int32)  # the motor drive the box reports per channel (status)
        self.pedals = (0, 0)
        self.active = False
        self.last_error = ""  # why the last poll failed (the Haptics status shows it); "" after a good poll
        self.status_error = ""  # a failed status-only read (ids, pwm): shown, but the grips and pedals stay live
        self.status_every = 100  # polls between the status-only reads (3 SDK calls per channel less on the other ticks)
        self._polls = 0

    @classmethod
    def open(cls, device_index: int = 0, channels: int = 6) -> "InstrumentBox":
        from i4h_standalone.input.follou import acquire_manager
        manager, _ = acquire_manager()
        from follou_sdk.devices.ibcontroller import IBController
        controller = manager.get_device_controller(IBController, count=int(device_index))
        if controller is None:
            from i4h_standalone.input.follou import fail_and_release

            fail_and_release(RuntimeError(f"IBController device index {device_index} not available"))
        box = cls(controller, channels)
        box.owns_manager = True
        return box

    def queue_handle_force(self, channel: int, force_n: float) -> None:
        value = float(force_n) if math.isfinite(force_n) else 0.0
        self.queued[channel] = value
        self.controller.set_handle_force_to_channel(int(channel), value)

    def poll(self) -> dict:
        try:
            self.controller.perform_update()
            for c in range(self.channels):
                opening = float(self.controller.get_opening_prc_from_channel(c)) / 100.0  # percent open
                self.openings[c] = float(np.clip(1.0 - opening, 0.0, 1.0))
            self.pedals = (int(self.controller.get_pedal_state(0)), int(self.controller.get_pedal_state(1)))
        except Exception as exc:
            self.active = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            return INACTIVE
        self.active, self.last_error = True, ""
        if self._polls % self.status_every == 0:
            self._read_status()
        self._polls += 1
        return {"active": False, "box": True, "grips": self.openings.copy(), "pedals": self.pedals}  # binds to no instrument

    def _read_status(self) -> None:
        """Measured force, handle ids and PWM (every ``status_every`` polls): status only, so a bad reading never takes
        the grip and pedals down; it reads as 0 and is reported in ``status_error``."""
        errors = []
        for name, out, getter in (("measured", self.measured, "get_force_from_channel"), ("ids", self.ids, "get_handle_id_from_channel"),
                                  ("pwm", self.pwm, "get_last_pwm_from_channel")):
            read = getattr(self.controller, getter, None)
            if read is None:
                continue
            for c in range(self.channels):
                try:
                    out[c] = read(c)
                except Exception as exc:  # an unreadable or out-of-range value (ids have come back as garbage)
                    out[c] = 0
                    errors.append(f"{name}[{c}]: {type(exc).__name__}: {exc}")
        self.status_error = "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")

    def close(self) -> None:
        try:
            for c in range(self.channels):
                self.queue_handle_force(c, 0.0)
            self.controller.perform_update()
        except Exception as exc:
            error = exc
        else:
            error = None
        try:
            close_controller(self)
        except Exception as exc:
            if error is None:
                raise
            error.add_note(f"device close failed: {exc}")
        if error is not None:
            raise error


class Handle:
    """One box channel mounted on a pose device (module docstring): grip and pedals into the device's samples, the
    servo's handle force onto the channel."""

    def __init__(self, box: InstrumentBox, channel: int):
        if not 0 <= int(channel) < box.channels:
            raise ValueError(f"handle channel {channel} exceeds the box's {box.channels} channels")
        self.box, self.channel = box, int(channel)

    def grip(self) -> float:
        return float(self.box.openings[self.channel]) if self.box.active else 0.0

    def buttons(self) -> dict:
        if not self.box.active:
            return {}
        return {"secondary": bool(self.box.pedals[0]), "fire": bool(self.box.pedals[1])}

    def apply(self, sample: dict) -> dict:
        """The device's grip and buttons replaced by the channel's; an inactive sample is returned untouched (``INACTIVE``
        is shared, never written)."""
        if not sample.get("active", False):
            return sample
        out = dict(sample)
        grip = self.grip()
        out["grip"], out["button"], out["buttons"] = grip, grip >= 0.5, self.buttons()
        return out

    def queue_force(self, force_n: float) -> None:
        self.box.queue_handle_force(self.channel, force_n)

    def status(self) -> dict:
        return {"channel": self.channel, "id": int(self.box.ids[self.channel]), "opening": self.grip(), "queued": float(self.box.queued[self.channel]),
                "measured": float(self.box.measured[self.channel]), "pwm": int(self.box.pwm[self.channel]), "connected": bool(self.box.active)}


class Avatar:
    """The HapticDevice adapter (module docstring): ``poll()`` reads the pose, ``queue_force(F)`` maps a force on the
    tool to the four motors, ``perform_update`` sends and refreshes."""

    def __init__(self, controller, *, scale=(1.0, 1.0, 1.0), yaw_axis=(0.0, 1.0, 0.0), pitch_axis=(1.0, 0.0, 0.0)):
        self.controller = controller
        self.scale = np.asarray(scale, np.float32).reshape(3)
        self.yaw_axis = np.asarray(yaw_axis, np.float64) / max(np.linalg.norm(yaw_axis), 1e-9)
        self.pitch_axis = np.asarray(pitch_axis, np.float64) / max(np.linalg.norm(pitch_axis), 1e-9)
        self.tip = np.zeros(3)  # metres, device frame, pivot at the origin (the last poll)
        self.direction = np.array([0.0, 0.0, 1.0])
        self.command = np.zeros(4)  # z force, yaw, pitch, rot torque last queued

    @classmethod
    def open(cls, device_index: int = 0, **kwargs) -> "Avatar":
        from i4h_standalone.input.follou import acquire_manager
        manager, _ = acquire_manager()
        from follou_sdk.devices.haptic import HapticDevice
        controller = manager.get_device_controller(HapticDevice, count=int(device_index))
        if controller is None:
            from i4h_standalone.input.follou import fail_and_release

            fail_and_release(RuntimeError(f"HapticDevice index {device_index} not available"))
        device = cls(controller, **kwargs)
        device.owns_manager = True
        return device

    def queue_force(self, force_n) -> None:
        f = np.asarray(force_n, np.float64).reshape(3)
        if not np.all(np.isfinite(f)):
            f = np.zeros(3)
        insertion = float(np.dot(f, self.direction))
        moment = np.cross(self.tip, f)
        self.command[:] = (insertion, float(np.dot(moment, self.yaw_axis)), float(np.dot(moment, self.pitch_axis)), 0.0)
        self.controller.set_force_and_torques(*self.command)

    def poll(self) -> dict:
        try:
            self.controller.perform_update()
            tip = np.asarray(self.controller.get_tip_position(), np.float64) * CM
            direction = np.asarray(self.controller.get_instrument_direction(), np.float64)
            roll = float(self.controller.get_value(ROT))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return INACTIVE
        if not (np.all(np.isfinite(tip)) and np.all(np.isfinite(direction)) and math.isfinite(roll)) or np.linalg.norm(direction) < 1e-6:
            self.last_error = "non-finite or zero pose"
            return INACTIVE
        self.last_error = ""
        self.tip, self.direction = tip, direction / np.linalg.norm(direction)
        position = (self.tip * self.scale).astype(np.float32)
        return {"active": True, "position": position, "rotation": quat_from_direction_roll(self.direction, roll), "grip": 0.0,
                "button": False, "buttons": {}}

    def close(self) -> None:
        try:
            self.controller.set_force_and_torques(0.0, 0.0, 0.0, 0.0)
            self.controller.perform_update()
        except Exception as exc:
            error = exc
        else:
            error = None
        try:
            close_controller(self)
        except Exception as exc:
            if error is None:
                raise
            error.add_note(f"device close failed: {exc}")
        if error is not None:
            raise error
