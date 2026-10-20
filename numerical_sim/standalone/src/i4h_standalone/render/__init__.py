# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Renderers. Not systems: they read Scene arrays between steps and never run inside the graph.

A renderer reads a ``Snapshot`` (``render/snapshots.py``): ``publish(snapshot)`` does the CUDA work
(copy positions, compute normals) on the renderer's ``stream`` and brackets it with the snapshot's
ready and release events; ``draw()`` is graphics. With no argument ``publish()`` reads the live
scene on Warp's current stream, which is what tests and the Isaac Lab GL view do after each
``sim.step()``. The threaded app publishes ring slots. ``noop`` is the default and what headless runs
and Isaac Lab use; ``slang`` is the textured renderer; ``gl`` is a colors-only debug view on Newton's
ViewerGL (single-threaded: the viewer copies through the host).
"""

from __future__ import annotations

from i4h_physics.systems.mechanics.surface import Surface
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

MARKER_IDLE_COLOR = (0.5, 0.5, 0.5)  # an idle input device (omnisurg controller.py:15-16)
MARKER_CANDIDATE_COLOR = (0.1, 0.9, 0.2)  # an idle device with a tool in reach
CUT_SPHERE_COLOR = (1.0, 0.85, 0.10)  # a scissors' cutting spheres in the overlay (yellow against the magenta collision spheres)
SPHERE_COLOR = (0.95, 0.28, 0.85)  # omnisurg/instruments/runtime_state.py:23
FLUID_COLOR = (0.32, 0.015, 0.02)  # omnisurg examples/ chole fluid.color (blood)
CAMERA = ("camera_pos", "camera_right", "camera_up", "camera_forward", "camera_inv_tan_half_fovy", "camera_aspect",
          "camera_near", "camera_far")  # the camera subset of SlangRenderer._frame every screen-space pass reads
PART_COLORS = {"liver": (0.45, 0.08, 0.07), "fat": (0.95, 0.72, 0.36), "gallbladder": (0.18, 0.42, 0.16)}  # flat tissue tints


def draw_sources(sim, env):
    """What both renderers draw from ``sim``: the drawn env (range-checked), the surface system per tet body,
    the rod system per rod body and the instrument control (or None)."""
    scene = sim.scene
    if not 0 <= int(env) < scene.envs:
        raise ValueError(f"render.env {env} outside [0, {scene.envs})")
    surfaces = {id(s.body): s for s in sim.systems if isinstance(s, Surface)}
    rods = {id(s.body): s for s in sim.systems if isinstance(s, (RodDirect, RodPbd))}
    control = next((s for s in sim.systems if isinstance(s, InstrumentControl)), None)
    for b in scene.bodies:
        if getattr(b, "skin", None) is not None and id(b) not in rods:
            raise ValueError(f"body {b.name!r} has a skinned surface but no rod_pbd system drives its tree")
    return int(env), surfaces, rods, control


class NoopRenderer:
    def publish(self, snap=None) -> None:
        pass

    def draw(self) -> bool:
        return True

    def set_status(self, text: str) -> None:
        pass

    def is_running(self) -> bool:
        return True

    def read_frame(self):
        """Every renderer: the last drawn frame as float32 ``(H, W, 3 or 4)`` in [0, 1], or None when it draws nothing."""
        return None

    def close(self) -> None:
        pass


def make_renderer(render_cfg, sim, procedure=None):
    cfg = dict(render_cfg or {})
    backend = cfg.pop("backend", "noop")
    if backend == "noop":
        return NoopRenderer()
    if backend == "xr":  # the headset (render.xr: world_units_per_meter, runtime, mirror, ...); needs the xr extra
        from i4h_standalone.render.xr import XrRenderer

        try:
            return XrRenderer(sim, procedure=procedure, **cfg)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            raise ValueError(f"render: {str(exc).split('argument ', 1)[-1]} is not a render key") from None
    cfg.pop("xr", None)  # the XR block only matters to the xr backend
    if backend == "slang":
        from i4h_standalone.render.slang import SlangRenderer

        try:
            return SlangRenderer(sim, procedure=procedure, **cfg)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            raise ValueError(f"render: {str(exc).split('argument ', 1)[-1]} is not a render key") from None
    if backend == "gl":  # the debug view reads size, env and camera and ignores the rest of the render block
        from i4h_standalone.render.gl import GlRenderer

        return GlRenderer(sim, procedure=procedure, **{k: v for k, v in cfg.items() if k in ("size", "env", "camera")})
    raise ValueError(f"render.backend must be noop, slang, gl or xr, got {backend!r}")
