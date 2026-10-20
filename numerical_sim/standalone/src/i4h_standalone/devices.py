# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab ``DeviceBase`` adapters (import only inside an Isaac Lab process; the follou SDK is imported only when a
case device needs it).

``I4hTeleopDevice``: a case's ``inputs:`` devices (MiniMou, Avatar, instrument box handles, fake, replay) as one Isaac
Lab teleop device. ``advance()`` runs the app's device path (``app.device_rows.DeviceRows``: bindings, pickup,
squeeze-to-fire, ``kind_columns``) and returns every instrument's ``[px, py, pz, qx, qy, qz, qw, grip, energy, fire]``
row concatenated, ``(10 n,)`` on ``cfg.sim_device`` (``output = "normalized"``: mapped into [-1, 1] with the attached
env's bounds), or None while no device reports. Force feedback, one of:

- ``attach(env)`` with ``cfg.haptics``: the env's ``tool_force`` drives the servo (``app.haptics.HapticsDriver``, env
  0's force staged by each ``advance()``, i.e. before the ``env.step`` it feeds);
- ``push_force(forces, position)`` for loops that compute forces themselves (the Haply convention: rows selected by
  ``position`` and summed, here per instrument), through ``app.haptics.HostForceMailbox``.

Either way the forces go through the servo's ``Conditioner`` and the case's YAML caps (``haptics.max_force_n``,
``haptics.handles``); the servo thread owns the devices' I/O (no second service thread).

``IsaacLabDeviceSource``: the reverse direction, any other Isaac Lab ``DeviceBase`` as an i4h input source, so it can
drive an instrument through ``DeviceRows`` (``scripts/isaaclab_teleop.py --device``): ``advance()`` output that is an
absolute ``[x, y, z, qx, qy, qz, qw, (buttons...)]`` row (Haply) or a delta ``[dx, dy, dz, rx, ry, rz, (gripper)]``
(``Se3Keyboard`` / ``Se3SpaceMouse``, which need Kit, so not in the kitless env) integrated into a pose.
"""

from __future__ import annotations

import numpy as np
import torch
from i4h_physics.quat import axis_angle_quat, normalize_quat, quat_mul
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils.configclass import configclass


@configclass
class I4hTeleopDeviceCfg(DeviceCfg):
    """Isaac Lab device configuration for ``I4hTeleopDevice``; ``configure_teleop`` installs it in an env cfg, whose
    case it then follows (``i4h_isaaclab.env.sync_teleop_case``)."""

    case: str = "chole_tet_med"
    inputs: dict | None = None
    fake: bool = False
    output: str = "physical"
    haptics: bool = False
    sim_device: str = "cuda:0"
    class_type: type | str = "i4h_standalone.devices:I4hTeleopDevice"


BUTTONS = ("grab", "secondary", "fire")  # device button edges a callback can bind (any device of the case)


class I4hTeleopDevice(DeviceBase):
    def __init__(self, cfg, retargeters=None, key_source=None):
        """``cfg``: an ``I4hTeleopDeviceCfg``; ``key_source``: an object with ``key_handlers`` (``render.gl.GlKeys``) for the pickup keys and the key
        callbacks."""
        super().__init__(retargeters)
        from i4h_isaaclab.composition import read_case
        from i4h_physics.scene import Scene
        from i4h_standalone.app.device_rows import DeviceRows
        from i4h_standalone.app.haptics import HapticsConfig
        from i4h_standalone.teleop import fake_inputs
        from i4h_surgical.loaders.instrument_usd import add_instrument

        if cfg.output not in ("physical", "normalized"):
            raise ValueError(f"I4hTeleopDeviceCfg.output must be physical or normalized, got {cfg.output!r}")
        self.cfg = cfg
        case = read_case(cfg.case)
        inputs = cfg.inputs if cfg.inputs is not None else case.get("inputs")
        if cfg.fake:
            inputs = fake_inputs(inputs)
        scene = Scene("cpu", 1)  # the instruments' records only (bindings, kinds, initial poses): no tissue is loaded
        for name, entry in (case.get("instruments") or {}).items():
            add_instrument(scene, name, **dict(entry))
        if not scene.instruments:
            raise ValueError(f"case {cfg.case!r} has no instruments to teleoperate")
        records = list(scene.instruments.values())
        self.tool_names = [r.name for r in records]
        self.haptics_cfg = HapticsConfig(**dict(case.get("haptics") or {}))
        self.haptics_cfg.enabled = bool(cfg.haptics)
        self.key_source = key_source
        self.devices = DeviceRows(records, inputs, "devices", renderer=key_source)
        self.initial = np.zeros((len(records), 10), np.float32)
        for k, r in enumerate(records):
            self.initial[k, :3], self.initial[k, 3:7] = r.initial_position, r.initial_rotation
        self.rows = self.initial.copy()
        self._callbacks: dict[str, list] = {}
        self._pressed = {b: False for b in BUTTONS}
        self._haptics = None  # attach(): the env's tool_force drives the servo
        self._servo = self._mailbox = None  # push_force(): the caller's forces drive it
        self._mid = self._half = None

    def __str__(self) -> str:
        return f"I4hTeleopDevice({self.cfg.case}: {', '.join(d.config.id for d in self.devices.manager.devices)})"

    # Isaac Lab DeviceBase ---------------------------------------------------------------------------------------

    def reset(self) -> None:
        """The driven env was reset: rows to the initial poses, holdings to the bindings, the servo zeroes until fresh."""
        self.rows[:] = self.initial
        self.devices.reset()
        if self._haptics is not None:
            self._haptics.reset([0])
        if self._mailbox is not None:
            self._mailbox.generation += 1

    def add_callback(self, key, func) -> None:
        """``key``: a device button (``grab``, ``secondary``, ``fire``: its rising edge on any device) or a window key
        (through ``key_source``; ``key_name`` spelling: ``"r"``, ``"1"``)."""
        name = str(key).lower()
        if name in BUTTONS:
            self._callbacks.setdefault(name, []).append(func)
            return
        if self.key_source is None:
            raise ValueError(f"callback key {key!r}: not a device button {BUTTONS} and no key_source to bind a window key")
        from i4h_isaaclab.camera.camera import key_event
        from i4h_standalone.app.pickup import key_name

        wanted = key_name(key)

        def on_key(event, func=func, wanted=wanted):
            pressed, press, repeat, _ = key_event(event)
            if press and not repeat and pressed == wanted:
                func()

        self.key_source.key_handlers.append(on_key)

    def advance(self) -> torch.Tensor | None:
        if self._haptics is not None:
            self._haptics.update(0.0)  # env 0's force after the previous step, before the step this advance() feeds
        self.devices.update(self.rows)
        states = self.devices.states
        for button in BUTTONS:
            down = any(st.active and st.buttons.get(button, False) for st in states.values())
            if down and not self._pressed[button]:
                for func in self._callbacks.get(button, ()):
                    func()
            self._pressed[button] = down
        if not any(st.active for st in states.values()):
            return None
        out = torch.from_numpy(self.rows.reshape(-1).copy()).to(self.cfg.sim_device)
        if self.cfg.output == "normalized":
            if self._mid is None:
                raise RuntimeError("output = 'normalized' needs attach(env) for the action bounds")
            out = torch.clamp((out - self._mid) / self._half, -1.0, 1.0)
        if self._retargeters:
            return torch.cat([r.retarget(out) for r in self._retargeters], dim=-1)
        return out

    # i4h ---------------------------------------------------------------------------------------------------------

    def attach(self, env) -> None:
        """Bind to an ``I4hEnv``: its action bounds (normalised output) and, with ``cfg.haptics``, its ``tool_force``."""
        from i4h_isaaclab.views import controls
        from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

        u = env.unwrapped
        sim = u.simulator
        control = next((s for s in sim.systems if isinstance(s, InstrumentControl) and s.count), None)
        if control is None or control.count != len(self.tool_names):
            raise ValueError("the env's case has other instruments than the device's case")
        offset = 0
        for s in controls(sim):
            if s is control:
                break
            offset += int(s.actions.size // sim.scene.envs)
        columns = slice(offset, offset + 10 * control.count)
        self._mid, self._half = u._mid[columns].to(self.cfg.sim_device), u._half[columns].to(self.cfg.sim_device)
        if self.haptics_cfg.enabled:
            if self._servo is not None:
                raise ValueError("push_force already drives the servo: attach haptics or push forces, not both")
            from i4h_standalone.app.haptics import HapticsDriver

            force = next((s for s in sim.systems if s.id == self.haptics_cfg.system), None)
            if force is None or not hasattr(force, "force"):
                raise ValueError(f"haptics.system {self.haptics_cfg.system!r}: the case has no tool_force system of that id")
            self._haptics = HapticsDriver(sim, force, self.devices, self.haptics_cfg)

    def push_force(self, forces, position=None) -> None:
        """Per-instrument forces (sim units, as ``tool_force``; the case's ``newtons_per_unit`` and caps apply), sent to
        the device holding each instrument. ``forces``: ``(N, 3)``; ``position``: ``(N,)`` instrument indices, the rows
        of one index summed (Haply's select-and-sum), or None for ``N`` = the instrument count, row k = instrument k.
        Non-finite rows count as zero. Without ``cfg.haptics`` nothing is sent."""
        if self._haptics is not None:
            raise ValueError("attach() bound the env's tool_force to the servo: push forces or attach haptics, not both")
        if not self.haptics_cfg.enabled:
            return
        f = forces.detach().cpu().numpy() if isinstance(forces, torch.Tensor) else np.asarray(forces)
        f = np.asarray(f, np.float32).reshape(-1, 3)
        f = np.where(np.isfinite(f).all(axis=1, keepdims=True), f, 0.0)
        n = len(self.tool_names)
        if position is None:
            if len(f) != n:
                raise ValueError(f"push_force: {len(f)} rows for {n} instruments (pass position)")
            table = f
        else:
            idx = position.detach().cpu().numpy() if isinstance(position, torch.Tensor) else np.asarray(position)
            idx = np.asarray(idx, np.int64).reshape(-1)
            if idx.shape[0] != len(f) or (idx.size and (idx.min() < 0 or idx.max() >= n)):
                raise ValueError(f"push_force: position must give one instrument index in [0, {n}) per force row")
            table = np.zeros((n, 3), np.float32)
            np.add.at(table, idx, f)
        if self._servo is None:
            from i4h_standalone.app.haptics import HapticServo, HostForceMailbox, live_sources

            self._mailbox = HostForceMailbox(n)
            self._servo = HapticServo(self.haptics_cfg, live_sources(self.devices.manager), self._mailbox, self.tool_names)
            self._servo.start()
        self._servo.check()
        self._mailbox.push(table, self.devices.holders_by_device)

    def close(self) -> None:
        """The servo stops (final zeros) before the devices close."""
        try:
            if self._haptics is not None:
                self._haptics.close()
            if self._servo is not None:
                self._servo.stop()
        finally:
            self.devices.close()


def configure_teleop(cfg, **settings):
    """Install the standalone adapter in an Isaac Lab environment's device lookup."""
    from isaaclab.devices.device_base import DevicesCfg

    cfg.teleop_devices = DevicesCfg(devices={"i4h": I4hTeleopDeviceCfg(case=cfg.case, **settings)})
    return cfg


class IsaacLabDeviceSource:
    """An Isaac Lab ``DeviceBase`` as an i4h input source (``poll()`` -> sample dict; ``DeviceRows.replace_source``).
    ``mode``: ``absolute`` (``advance()`` gives ``[x, y, z, qx, qy, qz, qw, a, b, ...]``: pose, then buttons: ``a`` the
    grip, ``b`` energy, ``c`` fire) or ``delta`` (``[dx, dy, dz, rx, ry, rz, (gripper)]`` per call, integrated from
    ``start``: ``(position, rotation xyzw)``; a positive gripper closes the jaw). ``scale`` multiplies positions."""

    def __init__(self, device, mode: str = "absolute", scale: float = 1.0, start=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))):
        if mode not in ("absolute", "delta"):
            raise ValueError(f"IsaacLabDeviceSource mode must be absolute or delta, got {mode!r}")
        self.device, self.mode, self.scale = device, mode, float(scale)
        self.position = np.asarray(start[0], np.float32).copy()
        self.rotation = normalize_quat(np.asarray(start[1], np.float32))

    def poll(self) -> dict:
        data = self.device.advance()
        if data is None:
            return {"active": False}
        v = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        v = np.asarray(v, np.float32).reshape(-1)
        if not np.all(np.isfinite(v)):
            return {"active": False}
        if self.mode == "absolute":
            if v.shape[0] < 7:
                raise ValueError(f"absolute device output needs at least 7 values, got {v.shape[0]}")
            self.position = v[:3] * self.scale
            self.rotation = normalize_quat(v[3:7])
            extra = v[7:]
            grip = float(np.clip(extra[0], 0.0, 1.0)) if extra.size > 0 else 0.0
            buttons = {"secondary": bool(extra.size > 1 and extra[1] > 0.5), "fire": bool(extra.size > 2 and extra[2] > 0.5)}
        else:
            if v.shape[0] < 6:
                raise ValueError(f"delta device output needs at least 6 values, got {v.shape[0]}")
            self.position = self.position + v[:3] * self.scale
            angle = float(np.linalg.norm(v[3:6]))
            if angle > 1e-9:
                self.rotation = normalize_quat(quat_mul(axis_angle_quat(v[3:6] / angle, angle), self.rotation))
            grip = 1.0 if v.shape[0] > 6 and v[6] > 0.0 else 0.0
            buttons = {}
        return {"active": True, "position": self.position.astype(np.float32), "rotation": self.rotation.astype(np.float32),
                "grip": grip, "buttons": buttons}

    def close(self) -> None:
        reset = getattr(self.device, "reset", None)
        if callable(reset):
            reset()
