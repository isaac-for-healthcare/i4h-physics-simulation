# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMou input, adapted from i4h_standalone/input/follou.py.

The SDK is imported only by open(). Motion is relative to the first good pose;
lost tracking freezes the target and the next good pose anchors it again.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

WORKSPACE_LOWER = np.array([-0.25, -0.75, 0.205])
WORKSPACE_UPPER = np.array([0.25, -0.25, 0.60])


def quat_mul(a, b):
    av, bv = np.asarray(a)[:3], np.asarray(b)[:3]
    return np.append(
        a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - np.dot(av, bv)
    )


def conjugate(q):
    return np.asarray(q) * [-1, -1, -1, 1]


class FollouInput:
    def __init__(
        self,
        controller,
        release: Callable[[], None],
        *,
        scale: float = 0.001,
        handle_readout: str = "normalized",
    ):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Follou scale must be positive and finite")
        if handle_readout not in ("raw", "normalized"):
            raise ValueError("Follou handle readout must be raw or normalized")
        self.controller, self._release = controller, release
        self.scale, self.handle_readout = scale, handle_readout
        self.closed = False
        self.status = "Waiting for first valid MiniMou sample"
        self._handle_confirmed = False
        self._ranges: dict[str, tuple[float, float]] = {}
        self._anchor_position = None
        self._anchor_rotation = None
        self._anchor_target = None
        self.raw_position = (
            None  # SDK coordinates, millimetres; never input-gain scaled
        )
        self.raw_rotation = None
        self.scene_pose = (
            None  # anchored scene position + quaternion before motion limits
        )
        self.position_clutched = False
        self._position_reanchor_pending = False

    @classmethod
    def open(cls, *, device_index=0, scale=0.001, handle_readout="normalized"):
        if device_index < 0:
            raise ValueError("Follou device index must be nonnegative")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Follou scale must be positive and finite")
        try:
            from follou_sdk.devices.minimou import MiniMou
            from follou_sdk.manager import acquire_manager, release_manager
        except Exception as exc:
            raise RuntimeError(
                "MiniMou input requires Python 3.12 and the local SDK: "
                "install follou_sdk in examples/versius/.venv. "
                f"SDK import failed: {exc}"
            ) from exc
        manager = acquire_manager()
        try:
            controller = manager.get_device_controller(MiniMou, count=device_index)
            if controller is None:
                found = [type(device).__name__ for device in manager.devices]
                raise RuntimeError(
                    f"MiniMou index {device_index} unavailable; discovered devices: {found or ['none']}"
                )
            result = cls(
                controller, release_manager, scale=scale, handle_readout=handle_readout
            )
            # This is an input adapter: no simulated force or jaw feedback.
            controller.set_force(0.0, 0.0, 0.0)
            controller.perform_update()
            return result
        except BaseException as original:
            try:
                release_manager()
            except Exception as cleanup:
                original.add_note(f"MiniMou cleanup failed: {cleanup}")
            raise

    def _closing_grip(self, name: str, value: float) -> float:
        lower, upper = self._ranges.get(name, (value, value))
        lower, upper = min(lower, value), max(upper, value)
        self._ranges[name] = lower, upper
        return (
            float(np.clip(1.0 - (value - lower) / (upper - lower), 0.0, 1.0))
            if upper - lower >= 1e-4
            else 0.0
        )

    def _poll(self):
        c = self.controller
        c.set_force(0.0, 0.0, 0.0)
        c.perform_update()
        connected = int(c.get_handle_connection_state())
        if connected > 0:
            self._handle_confirmed = True
        elif connected < 0 or self._handle_confirmed:
            raise RuntimeError(f"MiniMou handle disconnected (state {connected})")
        position = np.asarray(c.get_position()[:3], dtype=np.float64)
        raw_position = position.copy()
        rotation = np.asarray(c.get_orientation()[:4], dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (4,):
            raise ValueError("MiniMou returned an invalid pose shape")
        rotation_norm = float(np.linalg.norm(rotation))
        if (
            not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
            or not math.isfinite(rotation_norm)
            or rotation_norm < 1e-8
        ):
            raise ValueError("MiniMou returned a non-finite or zero pose")
        # Reference input mirrors SDK X in a Y-up frame. Rotate that frame
        # +90 degrees around X to this Z-up scene: (-x, -z, y).
        position = position[[0, 2, 1]] * [-self.scale, -self.scale, self.scale]
        if not np.isfinite(position).all():
            raise ValueError("MiniMou position scale overflow")
        # The reference quaternion correction is Rx(-90) * q * Rx(+90).
        # Converting that frame to Z-up conjugates by Rx(+90), leaving SDK q.
        rotation /= rotation_norm
        tool, handle = float(c.get_tool_pos()), float(c.get_handle_opening_value())
        activity = float(c.get_handle_activity())
        if not all(math.isfinite(value) for value in (tool, handle, activity)):
            raise ValueError("MiniMou returned a non-finite grip")
        grip = max(
            self._closing_grip("handle", handle), self._closing_grip("tool", tool)
        )
        lower, upper = self._ranges["handle"]
        if activity and upper - lower < 1e-4:
            grip = 1.0
        if self.handle_readout == "raw":
            grip = float(
                np.clip(1.0 - (handle if self._handle_confirmed else tool), 0.0, 1.0)
            )
        self.raw_position = raw_position
        self.raw_rotation = rotation.copy()
        return position, rotation, grip

    def set_position_clutch(self, held: bool) -> None:
        if self.closed:
            return
        held = bool(held)
        if self.position_clutched and not held:
            self._position_reanchor_pending = True
        self.position_clutched = held

    def read_position(self) -> np.ndarray:
        """Read a finite, Z-up position in metres without standalone workspace limits.

        Consumers own their relative anchoring, clutching and workspace. Tracking
        errors propagate so they can freeze and re-anchor on the next good sample.
        """
        if self.closed:
            raise RuntimeError("MiniMou input is closed")
        return self._poll()[0]

    def update(
        self,
        target: np.ndarray,
        dt: float,
        *,
        move_speed=0.10,
        gripper_speed=0.03,
        position_clutch: bool | None = None,
    ) -> None:
        if self.closed:
            return
        if position_clutch is not None:
            self.set_position_clutch(position_clutch)
        try:
            position, rotation, grip = self._poll()
        except Exception as exc:
            self.status = f"Tracking lost: {exc}"
            self.raw_position = self.raw_rotation = None
            self.recenter()
            return
        if self._anchor_position is None:
            self._anchor_position = position.copy()
            self._anchor_rotation = rotation.copy()
            self._anchor_target = target.copy()
        if self._position_reanchor_pending and not self.position_clutched:
            # Consume the release on a valid sample, including after a pause.
            # Keep the original orientation anchors: only translation is rebased.
            self._anchor_position = position.copy()
            self._anchor_target[:3] = target[:3]
            self._position_reanchor_pending = False
        device_position = self._anchor_target[:3] + position - self._anchor_position
        desired_rotation = quat_mul(
            quat_mul(rotation, conjugate(self._anchor_rotation)),
            self._anchor_target[3:7],
        )
        desired_rotation /= np.linalg.norm(desired_rotation)
        self.scene_pose = np.concatenate((device_position, desired_rotation))
        if not self.position_clutched:
            desired = np.clip(device_position, WORKSPACE_LOWER, WORKSPACE_UPPER)
            delta = desired - target[:3]
            target[:3] += delta * min(
                1.0, move_speed * dt / max(float(np.linalg.norm(delta)), 1e-12)
            )
            target[:3] = np.clip(target[:3], WORKSPACE_LOWER, WORKSPACE_UPPER)
        current = target[3:7].astype(np.float64)
        current /= np.linalg.norm(current)
        dot = float(np.clip(np.dot(current, desired_rotation), -1.0, 1.0))
        if dot < 0:
            desired_rotation = -desired_rotation
            dot = -dot
        # At most 1.5 rad/s, following the shortest quaternion arc.
        half_angle = math.acos(dot)
        fraction = min(1.0, 1.5 * dt / max(2.0 * half_angle, 1e-12))
        if half_angle > 1e-6:
            q = (
                math.sin((1 - fraction) * half_angle) * current
                + math.sin(fraction * half_angle) * desired_rotation
            ) / math.sin(half_angle)
        else:
            q = desired_rotation
        target[3:7] = q / np.linalg.norm(q)
        amount = gripper_speed * dt / 0.04
        target[-1] = np.clip(
            target[-1] + np.clip(1.0 - grip - target[-1], -amount, amount), 0.0, 1.0
        )
        self.status = (
            "MiniMou tracking (position clutched)"
            if self.position_clutched
            else "MiniMou tracking"
        )

    def recenter(self):
        self._anchor_position = self._anchor_rotation = self._anchor_target = None
        self.scene_pose = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.position_clutched = self._position_reanchor_pending = False
        self.raw_position = self.raw_rotation = self.scene_pose = None
        try:
            self.controller.set_force(0.0, 0.0, 0.0)
            self.controller.perform_update()
        finally:
            self._release()
