# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from i4h_assessment.executor import TargetMeasurement
from i4h_physics.scene import ACTIVE
from i4h_surgical.systems.instruments.grasp import Grasp
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp


@wp.kernel
def volume(
    q: wp.array2d(dtype=wp.vec3),
    flags: wp.array2d(dtype=int),
    active: wp.array3d(dtype=int),
    particle: wp.array3d(dtype=int),
    sphere: wp.array3d(dtype=int),
    sphere_active: wp.array2d(dtype=int),
    held: wp.array2d(dtype=int),
    k: int,
    start: int,
    end: int,
    radius: float,
    rows: wp.array2d(dtype=float),
):
    e = wp.tid()
    if rows[e, 11] == 3.0 or held[e, k] == 0:
        return
    target = wp.vec3(rows[e, 0], rows[e, 1], rows[e, 2])
    for g in range(active.shape[2]):
        p = particle[e, k, g]
        if active[e, k, g] != 0 and p >= start and p < end:
            s = sphere[e, k, g]
            if (flags[e, p] & ACTIVE) != 0 and sphere_active[e, s] != 0:
                if wp.length(q[e, p] - target) <= radius:
                    rows[e, 6] = 1.0


@wp.kernel
def surface(
    q: wp.array2d(dtype=wp.vec3),
    flags: wp.array2d(dtype=int),
    active: wp.array3d(dtype=int),
    verts: wp.array3d(dtype=wp.vec3i),
    bary: wp.array3d(dtype=wp.vec3),
    tet: wp.array3d(dtype=int),
    live_tet: wp.array2d(dtype=int),
    held: wp.array2d(dtype=int),
    k: int,
    start: int,
    end: int,
    radius: float,
    rows: wp.array2d(dtype=float),
):
    e = wp.tid()
    if rows[e, 11] == 3.0 or held[e, k] == 0:
        return
    target = wp.vec3(rows[e, 0], rows[e, 1], rows[e, 2])
    for g in range(active.shape[2]):
        t = tet[e, k, g]
        if active[e, k, g] != 0 and t >= 0 and t < live_tet.shape[1]:
            v = verts[e, k, g]
            if (
                live_tet[e, t] != 0
                and v[0] >= start
                and v[0] < end
                and v[1] >= start
                and v[1] < end
                and v[2] >= start
                and v[2] < end
            ):
                if (
                    (flags[e, v[0]] & ACTIVE) != 0
                    and (flags[e, v[1]] & ACTIVE) != 0
                    and (flags[e, v[2]] & ACTIVE) != 0
                ):
                    w = bary[e, k, g]
                    p = w[0] * q[e, v[0]] + w[1] * q[e, v[1]] + w[2] * q[e, v[2]]
                    if wp.length(p - target) <= radius:
                        rows[e, 6] = 1.0


class GraspTarget(TargetMeasurement):
    extra_keys = ("body",)
    def _bind_tool(self):
        self.tool_body, self.start, self.end = self._selection(self.cfg["body"])
        self.control = next(
            (
                s
                for s in self.sim.systems
                if isinstance(s, InstrumentControl) and self.cfg["tool"] in s.names
            ),
            None,
        )
        if self.control is None:
            raise ValueError(f"unknown procedure instrument {self.cfg['tool']!r}")
        self.tool_index = self.control.names.index(self.cfg["tool"])
        if self.scene.instruments[self.cfg["tool"]].kind != "grasper":
            raise ValueError("grasp_target tool must be a grasper")
        self.tool_q = self.control.root
        self.tool_flags = self.scene.field(
            "flags", self.tool_body
        )  # unused by grasp measurement
        self.grasps = [
            s
            for s in self.sim.systems
            if isinstance(s, (Grasp, VolumetricGrasp))
            and s.body is self.tool_body
            and s.instruments is self.control
        ]
        if not self.grasps:
            raise ValueError(
                "grasp_target requires a grasp mechanism for its body/tool"
            )

    def _qualify(self):
        for s in self.grasps:
            if not s.enabled or not self.control.enabled:
                continue
            args = [self.scene.field("q", self.tool_body), self.tool_flags, s.active]
            if isinstance(s, VolumetricGrasp):
                kernel = volume
                args += [s.particle, s.sphere, self.control.sphere_active]
            else:
                kernel = surface
                args += [s.verts, s.bary, s.tet, s.surface.active]
            wp.launch(
                kernel,
                self.scene.envs,
                inputs=args
                + [
                    self.control.held,
                    self.tool_index,
                    self.start,
                    self.end,
                    self.radius,
                ],
                outputs=[self.rows],
                device=self.scene.device,
            )
