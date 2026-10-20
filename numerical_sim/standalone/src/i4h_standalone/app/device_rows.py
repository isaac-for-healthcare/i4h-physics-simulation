# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input devices -> one env's ``(n, 10)`` instrument rows, host only (no simulator, no device arrays).

The app's ``InstrumentDriver`` (``devices`` and ``keyboard`` modes) and the Isaac Lab teleop loop
(``scripts/isaaclab_teleop.py``, ``i4h_standalone.devices``) share it. It owns the ``InputManager`` of the case's
``inputs:`` block (or the keyboard hub's devices), the instrument bindings, the proximity ``Pickup``, the idle-device
markers and the squeeze-to-fire rule, and writes ``[px, py, pz, qx, qy, qz, qw, grip, energy, fire]`` rows:

- a held tool follows its device (through the pickup's take offsets; its jaw is parked until the handle takes over);
- energy and fire come from the device's ``secondary`` / ``fire`` buttons (keyboard hub: Shift, F; the box's pedals);
  a clip applier on a MiniMou without a box handle fires when its handle is squeezed (``SQUEEZE``);
- ``kind_columns``: a hook's handle is its cautery pedal;
- an unheld tool, or a device that stopped reporting, keeps its pose and releases jaw, pedal and trigger; without
  pickup, a tool bound to no device is left untouched.

``holders_by_device`` (device id -> tool index this step, a fresh dict per update) is what the haptic servo reads.
"""

from __future__ import annotations

import numpy as np

SQUEEZE = 0.9  # a clip applier on a MiniMou fires when its handle is squeezed this far (the device has no fire button)
SQUEEZE_BACKENDS = ("minimou",)  # keyboard and replay inputs carry an explicit fire value


def kind_columns(kind: str, grip: float, energy: float, fire: float):
    """The grip, energy and fire columns of a tool of ``kind`` from its handle, pedal and trigger: a hook has no jaw,
    so its handle is the cautery pedal (the scripted and the device paths share this rule)."""
    if kind == "hook":
        return 0.0, np.maximum(energy, grip), fire
    return grip, energy, fire


def replay_inputs(records, base) -> dict:
    """An ``inputs:`` block replaying a session recorded with ``--record <base>``: one ``replay`` device per instrument
    whose ``<base>_<input or name>.npy`` exists (empty when none does)."""
    from pathlib import Path

    base = Path(base)
    files = {dev: base.with_name(f"{base.stem}_{dev}.npy") for dev in (r.input or r.name for r in records)}
    return {dev: {"backend": "replay", "settings": {"replay_path": str(p)}} for dev, p in files.items() if p.is_file()}


class DeviceRows:
    def __init__(self, records, inputs_cfg, mode: str = "devices", renderer=None, record: bool = False):
        """``records``: the scene's instrument records (``scene.instruments.values()``, their order is the rows' order);
        ``inputs_cfg``: the case's ``inputs:`` block (``devices`` mode); ``mode``: ``devices`` or ``keyboard`` (the hub
        drives every instrument from the window's keys: needs ``renderer.key_handlers`` and ``renderer.camera``);
        ``renderer``: optional, its ``key_handlers`` get the pickup keys and its ``add_markers`` the idle-device
        proxies; ``record``: keep every row per frame (``recorded``, the app's ``--record``)."""
        from i4h_standalone.input import DeviceConfig, InputManager

        if mode not in ("devices", "keyboard"):
            raise ValueError(f"DeviceRows mode must be devices or keyboard, got {mode!r}")
        self.records, self.n, self.mode = list(records), len(records), mode
        self.env = 0  # the env whose rows these are (the haptic servo stages that env's force; the app's driver follows PgUp/PgDn)
        self.ids = [rec.input or rec.name for rec in self.records]  # an explicit input: must exist; a name may match one
        self.pickup = self.markers = self.hub = None
        self.holders_by_device: dict = {}
        self.record = bool(record)
        self.recorded: dict[str, list] = {}
        if mode == "keyboard":
            from i4h_standalone.input.keyboard import KeyboardHub

            if renderer is None or not hasattr(renderer, "key_handlers"):
                raise ValueError("--input keyboard needs the slang renderer window")
            self.hub = KeyboardHub({dev: (rec.initial_position, rec.initial_rotation) for dev, rec in zip(self.ids, self.records)},
                                   renderer.camera)
            renderer.key_handlers.append(self.hub.handle_key)
            configs = [DeviceConfig(dev, "keyboard") for dev in self.ids]
            self.binding = dict(zip(range(self.n), self.ids))
        else:
            devices = dict(inputs_cfg or {})
            pickup_cfg = devices.pop("pickup", None)  # not a device: the proximity pickup block
            missing = {rec.input for rec in self.records if rec.input} - set(devices)
            if missing:
                raise ValueError(f"instruments bind to inputs {sorted(missing)} that the case's inputs: block does not define")
            configs = [DeviceConfig(dev, **dict(spec)) for dev, spec in devices.items()]
            self.binding = {k: dev for k, dev in enumerate(self.ids) if dev in devices}
            self.holders_by_device = {dev: k for k, dev in self.binding.items()}
            if pickup_cfg is not None:
                from i4h_standalone.app.pickup import Pickup

                pickup_cfg = dict(pickup_cfg or {})
                start = dict(pickup_cfg.pop("start", None) or {})  # device -> the instrument it holds at the start (by name)
                unknown = set(pickup_cfg.get("keys", {})) - set(devices)
                if unknown:
                    raise ValueError(f"inputs.pickup.keys name devices {sorted(unknown)} that the inputs: block does not define")
                initial = {}
                for k, dev in self.binding.items():
                    if initial.get(dev) is not None:
                        raise ValueError(f"inputs.pickup: device {dev!r} is bound by two instruments; a device holds one tool at a time")
                    initial[dev] = k
                if start:
                    names = [r.name for r in self.records]
                    bad = {d: n for d, n in start.items() if d not in devices or n not in names}
                    if bad or len(set(start.values())) != len(start):
                        raise ValueError(f"inputs.pickup.start: {bad or start} must map input devices to distinct instruments "
                                         f"({names})")
                    initial = {dev: None for dev in initial} | {d: names.index(n) for d, n in start.items()}
                self.pickup = Pickup(pickup_cfg, self.records, initial)
                if renderer is not None and hasattr(renderer, "key_handlers"):
                    renderer.key_handlers.append(self.pickup.handle_key)
                if renderer is not None and hasattr(renderer, "add_markers"):
                    self.markers = renderer.add_markers(len(self.pickup.keys))  # before the worker starts: allocation is allowed
        self.manager = InputManager.from_configs(configs, keyboard=self.hub)
        self.manager.init()
        self.squeeze_fire = {d.config.id for d in self.manager.devices  # a mounted handle has a fire pedal; a missing box leaves none
                             if d.config.backend in SQUEEZE_BACKENDS and getattr(d.source, "handle", None) is None}

    @property
    def states(self) -> dict:
        return self.manager.states

    def update(self, rows: np.ndarray) -> np.ndarray:
        """Poll the devices and write this frame's rows into ``rows`` (``(n, 10)`` float32, in place; returned)."""
        if self.hub is not None:
            self.hub.advance()
        self.manager.poll()
        if self.pickup is not None:
            self.pickup.update(self.manager.states, rows)
            holders = {k: dev for dev, k in self.pickup.holding.items() if k is not None}
        else:
            holders = self.binding
        self.holders_by_device = {dev: k for k, dev in holders.items()}  # a fresh dict: the servo reads the attribute
        for k in range(self.n):
            dev = holders.get(k)
            state = self.manager.states[dev] if dev is not None else None
            if state is not None and state.active:
                if self.pickup is not None:  # the tool follows the device through the offsets of its take; parked jaw until taken over
                    rows[k, :3], rows[k, 3:7] = self.pickup.pose(dev, state.position, state.rotation)
                    grip = self.pickup.grip(dev, float(state.grip))
                else:
                    rows[k, :3], rows[k, 3:7] = state.position, state.rotation
                    grip = float(state.grip)
                energy = 1.0 if state.buttons.get("secondary") else 0.0
                fire = 1.0 if state.buttons.get("fire") else 0.0
                kind = self.records[k].kind
                if kind == "clip_applier" and dev in self.squeeze_fire and grip >= SQUEEZE:  # a MiniMou's squeezed handle fires
                    fire = 1.0
                rows[k, 7:10] = kind_columns(kind, grip, energy, fire)
            elif dev is None and self.pickup is None:
                continue  # an unbound tool without pickup: its row is untouched (today's static binding)
            else:  # unheld, or a device that stopped reporting: the pose stays, the jaw, the pedal and the trigger release
                rows[k, 7:10] = 0.0
            if self.record:  # one row per frame per instrument, held or not, so a replay keeps the session's time base
                self.recorded.setdefault(self.ids[k], []).append(rows[k].tolist())
        if self.markers is not None:
            for i, (_, position, radius, candidate) in enumerate(self.pickup.proxies(self.manager.states, rows)):
                self.markers.set(i, position, radius, candidate)
        return rows

    def replace_source(self, dev: str, source) -> None:
        """Device ``dev`` (configured, e.g. as ``noop``) polls ``source`` instead (``InputDevice.replace_source``)."""
        device = next((d for d in self.manager.devices if d.config.id == dev), None)
        if device is None:
            raise ValueError(f"no input device {dev!r}; configured: {[d.config.id for d in self.manager.devices]}")
        device.replace_source(source)

    def rebind(self, rows: np.ndarray) -> None:
        """The rows now belong to another env (the app's PgUp/PgDn): the pickup keeps its holdings with offsets against
        these rows (no jump), the keyboard hub continues from their poses and buttons."""
        if self.pickup is not None:
            self.pickup.rebind(self.manager.states, rows)
        if self.hub is not None:
            for k, dev in self.binding.items():
                self.hub.position[dev] = rows[k, :3].copy()
                self.hub.rotation[dev] = rows[k, 3:7].copy()
                self.hub.grip[dev] = float(rows[k, 7])
                self.hub.energy[dev] = float(rows[k, 8])
                self.hub.fire[dev] = float(rows[k, 9])

    def reset(self) -> None:
        """The driven env was reset: the holdings return to the ``input:`` bindings, the hub to the initial poses."""
        if self.pickup is not None:
            self.pickup.reset()
        if self.hub is not None:
            self.hub.reset()

    def close(self) -> None:
        self.manager.close()
