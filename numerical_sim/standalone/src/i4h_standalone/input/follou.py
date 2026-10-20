# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMou adapter for the separately installed Follou SDK (omnisurg/input/follou.py without the sample diagnostics)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from i4h_physics.quat import normalize_quat, quat_mul
from i4h_standalone.input.state import INACTIVE


@dataclass
class _MiniMouSampleState:
    handle_min: float | None = None
    handle_max: float | None = None
    tool_min: float | None = None
    tool_max: float | None = None
    handle_connection_confirmed: bool = False
    last_error: str = ""  # why the last poll returned INACTIVE ("" after a good one; the Haptics status shows it)


def ensure_follou_importable():
    try:
        from follou_sdk.devices.minimou import MiniMou
        from follou_sdk.manager import DeviceManager
    except Exception as exc:
        raise RuntimeError(f"Follou device unavailable: install follou-sdk with uv run --with /path/to/follou-sdk and set FOLLOU_LIB_PATH to the native library file: {exc}") from exc
    return DeviceManager, MiniMou


def acquire_manager():
    _, MiniMou = ensure_follou_importable()
    from follou_sdk.manager import acquire_manager as acquire

    return acquire(), MiniMou


def release_manager() -> None:
    from follou_sdk.manager import release_manager as release

    release()


def fail_and_release(original):
    """Release discovery ownership while retaining the device-selection failure."""
    try:
        release_manager()
    except Exception as cleanup:
        original.add_note(f"Follou initialization cleanup failed: {cleanup}")
    raise original


class MiniMouController:
    def __init__(self, *, device_index: int = 0, scale=1.0, handle_readout: str = "normalized") -> None:
        """``device_index`` and ``handle_readout`` (raw | normalized) are validated by ``DeviceConfig``."""
        self._manager, self._mini_mou_cls = acquire_manager()
        self.device_index = int(device_index)
        self.scale = scale
        self.handle_readout = handle_readout
        self._closed = False
        self._sample_state = _MiniMouSampleState()

        self._controller = self._manager.get_device_controller(self._mini_mou_cls, count=self.device_index)

        if self._controller is None:
            available = [type(device).__name__ for device in getattr(self._manager, "devices", [])]
            fail_and_release(RuntimeError(
                f"MiniMou device index {self.device_index} not available; "
                f"discovered devices: {available or ['none']}"
            ))

    def poll(self) -> dict:
        if self._closed:
            return INACTIVE
        return poll_minimou_controller(self._controller, self._sample_state, scale=self.scale, handle_readout=self.handle_readout)

    @property
    def last_error(self) -> str:
        return self._sample_state.last_error

    def on_handle_mounted(self) -> None:
        """A box handle supplies the grip and buttons (``DeviceSource.mount_handle``): the MiniMou's own handle port is
        bypassed."""
        self.handle_readout = "box"

    def queue_force(self, world_force_n) -> None:
        """Stage a world-frame force (N) for the next ``perform_update`` (the servo's one round trip per tick sends it
        and refreshes the readings; ``poll()`` performs that update)."""
        if self._closed:
            raise RuntimeError("MiniMou controller is closed")
        force = np.asarray(world_force_n, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(force)):
            force.fill(0.0)
        # Measured on the hardware (2026-09-23, 0.5 N pushes on every axis, felt by the owner): the SDK's set_force
        # drives the motors against the position frame on all three axes, so the world force is negated as a whole;
        # the input X mirror (poll) and this negation leave X positive. set_force keeps its internal Y/Z motor swap.
        # (omnisurg sent -fx, fy, fz, which pushed the wrong way; its force path never ran on a device.)
        self._controller.set_force(float(force[0]), -float(force[1]), -float(force[2]))

    def set_feedback_force(self, world_force_n) -> None:
        """Queue and send a force now (one device round trip); the non-servo path and the motor release on close."""
        self.queue_force(world_force_n)
        update = getattr(self._controller, "perform_update", None)
        if callable(update):
            update()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.set_feedback_force((0.0, 0.0, 0.0))
        except Exception as exc:
            error = exc
        else:
            error = None
        self._closed = True
        try:
            release_manager()
        except Exception as exc:
            if error is None:
                raise
            error.add_note(f"Follou close failed: {exc}")
        if error is not None:
            raise error



def poll_minimou_controller(controller, state: _MiniMouSampleState, *, scale=1.0, handle_readout: str = "normalized") -> dict:
    try:
        update = getattr(controller, "perform_update", None)
        if callable(update):
            update()
        connection_reader = getattr(controller, "get_handle_connection_state", None)
        if handle_readout != "box" and callable(connection_reader):  # a box handle: the MiniMou's own handle port is unused
            connection_state = int(connection_reader())
            if connection_state > 0:
                state.handle_connection_confirmed = True
            elif connection_state < 0 or state.handle_connection_confirmed:
                state.last_error = f"handle disconnected (state {connection_state})"
                return INACTIVE

        position = np.asarray(controller.get_position()[:3], dtype=np.float32)
        raw_rotation = np.asarray(tuple(controller.get_orientation())[:4], dtype=np.float32)
        if (
            position.shape != (3,)
            or raw_rotation.shape != (4,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(raw_rotation))
            or float(np.linalg.norm(raw_rotation)) < 1.0e-8
        ):
            state.last_error = "non-finite or zero pose"
            return INACTIVE
        position *= np.asarray(scale, dtype=np.float32)
        if not np.all(np.isfinite(position)):
            state.last_error = "non-finite pose"
            return INACTIVE
        # MiniMou X motion is mirrored relative to the OmniSurg scene axes.
        position[0] *= -1.0
        rotation = _map_minimou_quaternion_to_position_frame(normalize_quat(raw_rotation))

        tool_pos = float(getattr(controller, "get_tool_pos", lambda: 0.0)())
        handle_pos = float(
            getattr(controller, "get_handle_opening_value", lambda: tool_pos)()
        )
        handle_activity = float(
            getattr(controller, "get_handle_activity", lambda: 0.0)()
        )
        if not all(math.isfinite(item) for item in (tool_pos, handle_pos, handle_activity)):
            state.last_error = "non-finite handle reading"
            return INACTIVE
        handle_active = bool(handle_activity)
    except Exception as exc:
        state.last_error = f"{type(exc).__name__}: {exc}"
        return INACTIVE
    state.last_error = ""
    handle_grip = _closing_grip(
        handle_pos, state, "handle_min", "handle_max"
    )
    tool_grip = _closing_grip(tool_pos, state, "tool_min", "tool_max")
    normalized_grip = max(handle_grip, tool_grip)
    handle_calibrated = (
        state.handle_min is not None
        and state.handle_max is not None
        and state.handle_max - state.handle_min >= 1.0e-4
    )
    if handle_active and not handle_calibrated:
        normalized_grip = 1.0
    raw_opening = (
        handle_pos if state.handle_connection_confirmed else tool_pos
    )
    raw_grip = float(np.clip(1.0 - raw_opening, 0.0, 1.0))
    if handle_readout == "box":  # the grip and buttons come from the instrument-box handle (Handle.apply)
        return {"active": True, "position": position, "rotation": rotation, "button": False, "grip": 0.0}
    grip = raw_grip if handle_readout == "raw" else normalized_grip
    return {"active": True, "position": position, "rotation": rotation, "button": handle_active or grip >= 0.5, "grip": grip}


def _closing_grip(value: float, state: _MiniMouSampleState, min_attr: str, max_attr: str) -> float:
    if not math.isfinite(value):
        return 0.0

    current_min = getattr(state, min_attr)
    current_max = getattr(state, max_attr)
    if current_min is None or current_max is None:
        # TODO: replace lifetime min/max with explicit handle calibration; first poll reports open grip.
        setattr(state, min_attr, float(value))
        setattr(state, max_attr, float(value))
        return 0.0

    current_min = min(float(current_min), float(value))
    current_max = max(float(current_max), float(value))
    setattr(state, min_attr, current_min)
    setattr(state, max_attr, current_max)
    span = current_max - current_min
    if span < 1.0e-4:
        return 0.0

    opening = (float(value) - current_min) / span
    return float(np.clip(1.0 - opening, 0.0, 1.0))


def _map_minimou_quaternion_to_position_frame(rotation: np.ndarray) -> np.ndarray:
    # The Follou SDK returns a quaternion, not axis/angle. Rotate its basis into
    # the same frame as get_position() without changing the position mapping.
    half_sqrt2 = np.sqrt(0.5)
    tet_to_hex = (half_sqrt2, 0.0, 0.0, half_sqrt2)
    hex_to_tet = (-half_sqrt2, 0.0, 0.0, half_sqrt2)
    return quat_mul(quat_mul(hex_to_tet, rotation), tet_to_hex)
