# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tip camera: an endoscope view from a rod's tip or a laparoscope view along an instrument's shaft, posed every frame
from the held snapshot (``render.camera_mode: tip``, ``render.tip: {body | instrument, distance, lookahead, fov, near}``).

Ported from omniendo: the roll-stable pose of ``isaaclab/mdp.py:105-161`` (``camera.stable_frame``: the up hint is the
rod frame's own y axis, so roll follows the catheter instead of world up; ``position = tip - distance * forward``) and
the interactive viewer of ``xcath/xcath.py:1292-1341`` (the last two rod particles copied into a pinned host buffer,
``target = tip + tangent * lookahead``, fov clamped to 30..150, the free camera saved and restored on toggle). Edits:
the pose comes from the renderer's held ``Snapshot`` through one pinned copy on the render stream (the render thread
never reads a device array); instruments are targets too (forward = the shaft's local -z, ``instrument_control.py``
``shaft_rotation``: the shaft runs from the trocar toward the root), so a chole tool can carry the camera; a
degenerate tangent keeps the previous pose. ``resolve_tip`` is shared with the tiled camera.
"""

from __future__ import annotations

import warp as wp
from i4h_physics.quat import quat_rotate

try:
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
except ModuleNotFoundError as error:
    if error.name != "i4h_surgical":
        raise
    InstrumentControl = ()
from i4h_isaaclab.camera.camera import Camera, stable_frame
from i4h_physics.array_copy import copy
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd

TIP_DEFAULTS = dict(body=None, instrument=None, distance=0.04, lookahead=0.05, fov=100.0, near=0.005)


def resolve_tip(sim, cfg: dict | None):
    """``(kind, target, settings, names)``: kind ``"rod"`` with ``(body, rod_system)`` or ``"instrument"`` with the tool index;
    ``names`` lists every possible target (``rod:<body>``, ``instrument:<name>``) for the panel."""
    cfg = {**TIP_DEFAULTS, **dict(cfg or {})}
    unknown = set(cfg) - set(TIP_DEFAULTS)
    if unknown:
        raise ValueError(f"render.tip: unknown keys {sorted(unknown)}; known {sorted(TIP_DEFAULTS)}")
    rods = {s.body.name: (s.body, s) for s in sim.systems if isinstance(s, (RodDirect, RodPbd)) and s.body.rod is not None}
    control = next((s for s in sim.systems if isinstance(s, InstrumentControl)), None)
    names = [f"rod:{n}" for n in rods] + [f"instrument:{n}" for n in (control.names if control is not None else [])]
    if not names:
        raise ValueError("render.tip: the case has no chain rod and no instrument to attach the camera to")
    if cfg["body"] is not None and cfg["instrument"] is not None:
        raise ValueError("render.tip: give body or instrument, not both")
    if cfg["body"] is not None:
        if cfg["body"] not in rods:
            raise ValueError(f"render.tip.body {cfg['body']!r} is not a chain rod body; choose from {names}")
        kind, target = "rod", rods[cfg["body"]]
    elif cfg["instrument"] is not None:
        if control is None or cfg["instrument"] not in control.names:
            raise ValueError(f"render.tip.instrument {cfg['instrument']!r} is not an instrument; choose from {names}")
        kind, target = "instrument", control.names.index(cfg["instrument"])
    elif rods:
        kind, target = "rod", next(iter(rods.values()))
    else:
        kind, target = "instrument", 0
    settings = {k: float(cfg[k]) for k in ("distance", "lookahead", "fov", "near")}
    settings["fov"] = min(max(settings["fov"], 30.0), 150.0)  # xcath.py:1334
    return kind, target, settings, names


class TipCamera:
    def __init__(self, sim, cfg: dict | None):
        self.sim, self.scene = sim, sim.scene
        self.kind, self.target, self.settings, self.names = resolve_tip(sim, cfg)
        self.rods = {s.body.name: (s.body, s) for s in sim.systems if isinstance(s, (RodDirect, RodPbd)) and s.body.rod is not None}
        self.control = next((s for s in sim.systems if isinstance(s, InstrumentControl)), None)
        self.host_q = wp.zeros(2, dtype=wp.vec3, device="cpu", pinned=True)  # rod: nodes P-2, P-1; instrument: [1] = root
        self.host_f = wp.zeros(1, dtype=wp.quat, device="cpu", pinned=True)  # the tip frame or the shaft quaternion
        self.world_up = (-self.scene.down).astype("float32")
        self.event = wp.Event(self.scene.device)  # recorded after the two copies: apply() waits for them alone
        self.pending = False

    @property
    def index(self) -> int:
        return self.names.index(f"rod:{self.target[0].name}" if self.kind == "rod" else f"instrument:{self.control.names[self.target]}")

    def select(self, index: int) -> None:
        name = self.names[int(index) % len(self.names)]
        kind, value = name.split(":", 1)
        self.kind = kind
        self.target = self.rods[value] if kind == "rod" else self.control.names.index(value)

    def publish(self, snap, env: int, stream=None) -> None:
        """Enqueue the tip copies from ``snap`` (between its ``wait`` and ``done``) on ``stream``."""
        if self.kind == "rod":
            body, rod = self.target
            P, F = body.count, rod.frame_count
            copy(self.host_q, self.scene.view(snap.q, body)[env][P - 2 : P], stream)
            i = env * F + min(P - 1, F - 1)
            copy(self.host_f, snap.frame[rod][i : i + 1], stream)
        else:
            k = self.target
            copy(self.host_q[1:2], snap.control.root[env, k : k + 1], stream)
            copy(self.host_f, snap.control.shaft_q[env, k : k + 1], stream)
        (stream if stream is not None else wp.get_stream(self.scene.device)).record_event(self.event)
        self.pending = True

    def apply(self, camera: Camera, stream=None) -> bool:
        """Pose ``camera`` from the copied tip; False (pose kept) on a degenerate tangent. Waits for the two copies only
        (the rest of the publish keeps running on the stream)."""
        if not self.pending:
            return False
        wp.synchronize_event(self.event)
        self.pending = False
        q, f = self.host_q.numpy(), self.host_f.numpy()[0]  # pinned host arrays: no device read
        tip = q[1]
        if self.kind == "rod":
            forward = q[1] - q[0]
        else:
            forward = quat_rotate(f, (0.0, 0.0, -1.0))  # into the body: the shaft runs from the trocar to the root
        if not (float((forward * forward).sum()) > 1e-16) or not bool((forward == forward).all()):  # xcath.py:1327
            return False
        right, up, fwd = stable_frame(forward, quat_rotate(f, (0.0, 1.0, 0.0)), quat_rotate(f, (1.0, 0.0, 0.0)), self.world_up)
        s = self.settings
        camera.set_pose(tip - fwd * s["distance"], fwd, up, s["lookahead"])
        return True
