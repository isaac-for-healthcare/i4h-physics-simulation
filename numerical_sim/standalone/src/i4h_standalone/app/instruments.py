# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument driver: input devices or scripts -> the (E, n, 10) actions array, once per frame.

Modes: ``scripted`` (every env, phase-shifted: approach a target part, close, retract),
``keyboard`` (the Slang window drives the drawn env; other envs hold their pose),
``devices`` (the case's ``inputs:`` devices, e.g. MiniMou or replay, drive the drawn env),
``noop`` (actions stay at the initial pose). Devices and the keyboard go through ``device_rows.DeviceRows``
(``i4h_standalone.input``, omnisurg's manager: calibration offsets, grip, buttons; the pickup; squeeze-to-fire), which the
Isaac Lab teleop loop shares; Isaac Lab's policies write ``instrument_control.actions`` themselves. Columns 8 and 9 (energy, fire) come
from the keyboard hub (Shift, F) or a device's ``secondary`` / ``fire`` buttons; a hook has no
jaw, so its handle (grip: the MiniMou's only analog input, Space on the keyboard) energises
it as well (omnisurg: ``grip >= diathermy_activation_threshold``) and its grip column stays 0;
the script energises the hook near its target.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp
from i4h_standalone.app.device_rows import SQUEEZE as SQUEEZE
from i4h_standalone.app.device_rows import SQUEEZE_BACKENDS as SQUEEZE_BACKENDS
from i4h_standalone.app.device_rows import DeviceRows, kind_columns

SCRIPT_TARGETS = {"grasper": "gallbladder", "hook": "fat", "suction": "gallbladder", "scissors": "fat"}  # part whose centroid the scripted tool approaches
SCRIPT_LINES = {"clip_applier": "cystic_duct"}  # centreline whose middle rest node the scripted tool approaches
APPLIER_REACH = 0.8  # scripted clip applier: fraction of the way to the live duct node at the closest approach (0.9 ploughs the fat away from the hook)
IDENTITY = np.float32([0.0, 0.0, 0.0, 1.0])


def scripted_actions(rest: np.ndarray, envs: int, t: float, out: np.ndarray | None = None) -> np.ndarray:
    """two_worlds probe: sinusoid in x below the hanging block (the sphere pushes its bottom face up),
    phase-shifted per env; identity rotation, jaw open, no energy, no fire.
    Shape ``(E, n, 10)`` = ``[px, py, pz, qx, qy, qz, qw, grip, energy, fire]``."""
    actions = np.zeros((envs, len(rest), 10), np.float32) if out is None else out
    actions[:, :, :3] = rest[None]
    phase = 2.0 * math.pi * t / 2.0 + np.arange(envs, dtype=np.float32) * (math.pi / 2.0)
    actions[:, :, 0] += (0.01 * np.sin(phase))[:, None]
    actions[:, :, 2] = 0.045
    actions[:, :, 3:7] = IDENTITY
    actions[:, :, 7:] = 0.0
    return actions


PERIOD, REACH = 6.0, 0.85  # scripted approach: seconds per in-and-out cycle, fraction of the way to the target


@wp.kernel
def follow_node(
    positions: wp.array2d(dtype=wp.vec3),
    node: int,
    initial: wp.vec3,
    frac: wp.array(dtype=wp.float32),
    k: int,
    actions: wp.array3d(dtype=wp.float32),
):
    """A scripted tool's position at ``frac[env]`` of the way from ``initial`` to a live centreline node (the clip
    applier's duct target hangs with the tissue)."""
    env = wp.tid()
    p = initial + (positions[env, node] - initial) * frac[env]
    actions[env, k, 0] = p[0]
    actions[env, k, 1] = p[1]
    actions[env, k, 2] = p[2]


def resets_env(env: int, env_ids) -> bool:
    """Whether a reset of ``env_ids`` (host ids; None: every env) includes ``env`` (the drawn env's host state)."""
    return env_ids is None or int(env) in np.asarray(env_ids, dtype=np.int64).reshape(-1).tolist()


class InstrumentDriver:
    def __init__(self, sim, control, inputs_cfg, mode: str, renderer=None, record: str | None = None, paint=None):
        """``paint``: a ``contact_paint`` system's live ``(E, 3)`` mask; with a window, held M/V/B keys paint the drawn env."""
        self.sim, self.control, self.mode = sim, control, mode
        scene = sim.scene
        self.envs, self.n = scene.envs, control.count
        self.paint, self.paint_keys = paint, None
        if paint is not None and mode in ("keyboard", "devices") and renderer is not None and hasattr(renderer, "key_handlers"):
            from i4h_standalone.input.keyboard import PaintKeys

            self.paint_keys = PaintKeys()
            renderer.key_handlers.append(self.paint_keys.handle_key)
            self.paint_host = np.zeros((self.envs, 3), np.float32)
        self.actions = control.actions.numpy()  # (E, n, 10) initial poses from the control system's reset
        self.initial = self.actions.copy()  # what reset() restores
        self.records = list(scene.instruments.values())
        self.env = int(getattr(renderer, "env", 0))
        self.renderer = renderer
        self.devices: DeviceRows | None = None  # keyboard / devices modes: the drawn env's rows from the inputs
        self.record_path = record
        if mode == "scripted":
            self._script_targets()
        elif mode in ("keyboard", "devices"):
            self.devices = DeviceRows(self.records, inputs_cfg, mode, renderer=renderer, record=bool(record))
        elif mode != "noop":
            raise ValueError(f"unknown input mode {mode!r}")

    # scripted --------------------------------------------------------------------------------------------

    def _script_targets(self):
        """Per instrument: initial position and a target near the surface of its part (or the two_worlds probe)."""
        scene = self.sim.scene
        self.script = []  # (k, initial (3,), target (3,), kind) for the instruments with a target part
        for k, rec in enumerate(self.records):
            part, line = SCRIPT_TARGETS.get(rec.kind), SCRIPT_LINES.get(rec.kind)
            for body in scene.bodies:
                if part in body.parts:
                    p = body.parts[part]
                    target = body.rest.q0[0, p.particle_start : p.particle_start + p.particle_count].mean(axis=0)
                    self.script.append((k, rec.initial_position.copy(), np.asarray(target, np.float32), rec.kind))
                c = getattr(body, "centrelines", None)
                if line is not None and c is not None and line in c.names:  # the line's middle node, followed live (the tissue hangs)
                    nodes = np.flatnonzero(c.line == c.names.index(line))
                    self.script.append((k, rec.initial_position.copy(), int(nodes[len(nodes) // 2]), rec.kind))
        from i4h_surgical.systems.mechanics.centreline import Centreline

        self.centreline = next((s for s in self.sim.systems if isinstance(s, Centreline)), None)
        # A live node target is followed on the device after the upload (no per-step readback): the host sends the
        # approach fraction per env, the kernel writes the row's position from this step's node.
        self.node_targets = [(k, target, initial) for k, initial, target, _ in self.script if isinstance(target, int)]
        self.node_frac = wp.zeros(self.envs, dtype=wp.float32, device=scene.device) if self.node_targets else None
        self.node_frac_host = np.zeros(self.envs, np.float32)

    def _scripted(self, t: float) -> np.ndarray:
        if not self.script:  # the two_worlds probe
            return scripted_actions(self.control.rest, self.envs, t, self.actions)
        phase = t / PERIOD + np.arange(self.envs, dtype=np.float64) / max(self.envs, 1) * 0.5
        s = 0.5 - 0.5 * np.cos(2.0 * math.pi * phase)  # (E,): 0 -> 1 -> 0 over one period, phase-shifted per env
        near = (s > 0.7).astype(np.float32)  # near the target on the way in, off on the way back
        deep = (s > 0.9).astype(np.float32)
        closest = (s > 0.98).astype(np.float32)
        self.node_frac_host[:] = s * APPLIER_REACH  # the applier brings its root within clipping.reach of its live node (follow_node)
        for k, initial, target, kind in self.script:
            if not isinstance(target, int):
                self.actions[:, k, :3] = initial + (target - initial) * (s * REACH)[:, None]
            self.actions[:, k, 3:7] = IDENTITY
            if kind == "suction":  # irrigate on the approach, suck at the closest
                grip, energy, fire = near * (1.0 - deep), deep, 0.0
            elif kind == "clip_applier":  # the trigger's rising edge at the closest places one clip per cycle
                grip, energy, fire = near, 0.0, closest
            else:  # a grasper (or the scissors) closes its jaw near the target; a hook's handle is its pedal (kind_columns)
                grip, energy, fire = near, 0.0, 0.0
            grip, energy, fire = kind_columns(kind, grip, energy, fire)
            self.actions[:, k, 7] = grip
            self.actions[:, k, 8] = energy
            self.actions[:, k, 9] = fire
        return self.actions

    # devices (DeviceRows) --------------------------------------------------------------------------------

    @property
    def manager(self):
        return self.devices.manager if self.devices is not None else None

    @property
    def hub(self):
        return self.devices.hub if self.devices is not None else None

    @property
    def pickup(self):
        return self.devices.pickup if self.devices is not None else None

    @property
    def markers(self):
        return self.devices.markers if self.devices is not None else None

    @property
    def holders_by_device(self) -> dict:
        """Device id -> tool index it drives this step (the haptic servo reads it)."""
        return self.devices.holders_by_device if self.devices is not None else {}

    @property
    def recorded(self) -> dict:
        return self.devices.recorded if self.devices is not None else {}

    @property
    def binding(self) -> dict:
        return self.devices.binding

    @binding.setter
    def binding(self, value: dict) -> None:
        self.devices.binding = value

    @property
    def squeeze_fire(self) -> set:
        """Devices whose squeezed handle fires a clip applier (no fire button on the hardware)."""
        return self.devices.squeeze_fire if self.devices is not None else set()

    @squeeze_fire.setter
    def squeeze_fire(self, value: set) -> None:
        self.devices.squeeze_fire = value

    # per frame -------------------------------------------------------------------------------------------

    def update(self, t: float) -> None:
        if self.renderer is not None:
            env = int(getattr(self.renderer, "env", self.env))  # follow PgUp/PgDn
            if env != self.env:
                self.actions[self.env, :, 8:10] = 0.0  # the env we leave: no cautery or trigger held unwatched (grip stays: a grasp holds)
                if self.mode in ("keyboard", "devices"):
                    self.control.actions[self.env].assign(self.actions[self.env])  # only the drawn env's row is uploaded per step
                if self.paint_keys is not None:
                    self.paint_host[self.env] = 0.0
                self.env = env
                if self.devices is not None:  # the holdings and the hub continue from the viewed env's rows (no jump)
                    self.devices.rebind(self.actions[env])
        if self.mode == "scripted":  # every row changes: the whole table
            self.control.actions.assign(self._scripted(t))
            if self.node_targets:  # the live-node rows, from this step's centreline on the device
                self.node_frac.assign(self.node_frac_host)
                for k, node, initial in self.node_targets:
                    wp.launch(follow_node, dim=self.envs, inputs=[self.centreline.positions, int(node), wp.vec3(*initial.tolist()),
                                                                  self.node_frac, int(k)], outputs=[self.control.actions], device=self.sim.scene.device)
        elif self.mode in ("keyboard", "devices"):  # only the drawn env's rows change
            self.devices.update(self.actions[self.env])
            self.control.actions[self.env].assign(self.actions[self.env])
        if self.paint_keys is not None:  # the drawn env paints with the held keys; the others do not
            self.paint_host[self.env] = self.paint_keys.mask()
            self.paint.assign(self.paint_host)

    def reset(self, env_ids=None) -> None:
        """Host state after ``sim.reset(env_ids)`` (all envs when None; host ids): the selected rows return to the
        initial poses, as the control system's reset did on the device, and with them the keyboard hub's poses and
        buttons when the drawn env is among them. A device pose is physical and stays; a replay continues."""
        rows = slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.int64).reshape(-1)
        self.actions[rows] = self.initial[rows]
        if self.devices is not None and resets_env(self.env, env_ids):  # the holdings and the hub belong to the drawn env
            self.devices.reset()

    def close(self) -> None:
        if self.devices is not None:
            self.devices.close()
        if self.record_path and self.recorded:
            base = Path(self.record_path)
            for dev, rows in self.recorded.items():
                path = base.with_name(f"{base.stem}_{dev}.npy")
                np.save(path, np.asarray(rows, np.float32))
                print(f"recorded {len(rows)} frames of {dev!r} to {path}")
