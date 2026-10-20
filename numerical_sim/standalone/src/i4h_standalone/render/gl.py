# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Colors-only debug view on Newton's public ``ViewerGL``.

Static meshes are logged once with their diffuse texture or flat colour; tet parts, instrument
pieces and rod tubes are re-logged every frame with a flat colour (a part's triangles come from
its surface system's live buffer, so cuts show; a rod's tube is placed from its live frames),
and every collision sphere of the drawn env is logged as a point with its radius (so the
sphere-kind probe and the tool chains are visible); alive fluid particles are points too.
Newton re-uploads a texture on every ``log_mesh`` and copies vertices through the host, so
this is a geometry check, not the renderer. ViewerGL is Z-up: Y-up cases such as chole
appear rotated here. Ordered-procedure markers are omitted; legacy single-target markers remain supported.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from i4h_isaaclab.appearance import appearance
from i4h_physics.config import require_asset
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.mechanics.surface import Surface
from i4h_standalone.render import (
    CUT_SPHERE_COLOR,
    FLUID_COLOR,
    MARKER_CANDIDATE_COLOR,
    MARKER_IDLE_COLOR,
    PART_COLORS,
    SPHERE_COLOR,
    draw_sources,
)
from i4h_standalone.render.clips import CLIP_COLOR, ClipDraw
from i4h_standalone.render.instruments import piece_skins, skin_pieces
from i4h_standalone.render.kernels import Scratch, update_normals_tangents
from i4h_standalone.render.rods import RodDraw
from i4h_standalone.render.skin import Skinner
from i4h_standalone.render.snapshots import Snapshot
from i4h_surgical.loaders.instrument_usd import CHAIN_CUT
from i4h_surgical.systems.fluids.smoke import Smoke
from i4h_surgical.systems.instruments.clipping import Clipping


class GlRenderer:
    def __init__(self, sim, *, size=(1280, 720), env=0, camera=None, procedure=None):
        from newton.viewer import ViewerGL

        scene = sim.scene
        self.env, surfaces, rod_systems, self.control = draw_sources(sim, env)
        self.sim, self.scene = sim, scene
        self.live = Snapshot(sim, procedure=procedure)
        self.task_markers = _GlMarkers(73, scene.device) if procedure is not None and not procedure.sequence else None
        self.task_procedure = procedure
        self.task_row = None
        self.viewer = ViewerGL(width=int(size[0]), height=int(size[1]))
        camera = dict(camera or {})
        if "position" in camera:
            self.viewer.set_camera(wp.vec3(*[float(v) for v in camera["position"]]),
                                   float(camera.get("pitch", 0.0)), float(camera.get("yaw", 0.0)))
        self.time = 0.0
        dev = scene.device
        for m in scene.static_meshes:
            look = {"texture": str(require_asset(appearance(m)["diffuse"]))} if "diffuse" in appearance(m) else {"color": tuple(appearance(m).get("color", (0.6, 0.6, 0.6)))}
            self.viewer.log_mesh(
                f"static/{m.name}",
                wp.array(m.points, dtype=wp.vec3, device=dev),
                wp.array(m.indices.reshape(-1).astype(np.int32), dtype=wp.int32, device=dev),
                normals=wp.array(m.normals, dtype=wp.vec3, device=dev),
                uvs=wp.array(m.uvs, dtype=wp.vec2, device=dev) if m.uvs is not None else None,
                **look,
            )
        self.parts = []  # (name, points view, part, triangles source, color, index array)
        for body in scene.bodies:
            m = body.tet if body.tet is not None else body.shell
            if m is None:
                continue
            source = surfaces.get(id(body))
            if source is None:
                source = m.tris[body.variant_of_env[self.env]]  # static: the variant's authored triangles
            for pname, p in body.parts.items():
                points = scene.part_view(scene.q, body, pname)[self.env]
                color = tuple((appearance(p) or {}).get("color", PART_COLORS.get(pname, (0.6, 0.6, 0.6))))
                self.parts.append((f"{body.name}/{pname}", points, p, source, color,
                                   wp.zeros(3 * p.tri_count, dtype=wp.int32, device=dev)))
        self.skinned = []  # (body, Skinner, positions, normals, tangents, zero uvs, Scratch, indices, colour)
        for body in scene.bodies:
            if getattr(body, "skin", None) is not None:
                sk = Skinner(body, rod_systems[id(body)], dev)
                self.skinned.append((body, sk, wp.zeros(sk.count, dtype=wp.vec3, device=dev), wp.zeros(sk.count, dtype=wp.vec3, device=dev),
                                     wp.zeros(sk.count, dtype=wp.vec4, device=dev), wp.zeros(sk.count, dtype=wp.vec2, device=dev),
                                     Scratch(sk.count, dev), wp.array(body.skin.indices.reshape(-1), dtype=wp.int32, device=dev),
                                     tuple(body.skin.visual.get("color", (0.6, 0.6, 0.6)))))
        self.clips = []  # (ClipDraw, positions, normals, indices)
        for s in sim.systems:
            if isinstance(s, Clipping):
                cd = ClipDraw(s, dev)
                self.clips.append((cd, wp.zeros(cd.count, dtype=wp.vec3, device=dev), wp.zeros(cd.count, dtype=wp.vec3, device=dev),
                                   wp.array(cd.indices, dtype=wp.int32, device=dev)))
        self.rods = []  # (RodDraw, positions, normals, indices)
        for body in scene.bodies:
            if body.rod is not None and id(body) in rod_systems:
                rd = RodDraw(body, rod_systems[id(body)])
                self.rods.append((rd, wp.zeros(rd.count, dtype=wp.vec3, device=dev), wp.zeros(rd.count, dtype=wp.vec3, device=dev),
                                  wp.array(rd.indices, dtype=wp.int32, device=dev)))
        self.skins = piece_skins(self.control, dev) if self.control is not None else []
        self.piece_points = [wp.zeros(s.count, dtype=wp.vec3, device=dev) for s in self.skins]
        self.piece_normals = [wp.zeros(s.count, dtype=wp.vec3, device=dev) for s in self.skins]
        self.piece_indices = [wp.array(s.indices, dtype=wp.int32, device=dev) for s in self.skins]
        # Particles as points: persistent radius (0 when dead) and colour buffers, filled on device in publish().
        self.fluids = []  # (body, radius, radii buffer, colours)
        for b in scene.pools:
            if b.fluid is not None:
                visual = appearance(b.fluid)
                color = tuple(visual.get("color", FLUID_COLOR))
                self.fluids.append((b, float(visual.get("render_radius", b.fluid.rest_distance)),
                                    wp.zeros(b.count, dtype=wp.float32, device=dev),
                                    wp.array(np.tile(np.float32(color), (b.count, 1)), dtype=wp.vec3, device=dev)))
        self.smokes = []  # (system, positions, radii, colours)
        for s in sim.systems:
            if isinstance(s, Smoke):
                n = int(s.count)
                self.smokes.append((s, wp.zeros(n, dtype=wp.vec3, device=dev), wp.zeros(n, dtype=wp.float32, device=dev),
                                    wp.array(np.tile(np.float32((0.8, 0.78, 0.72)), (n, 1)), dtype=wp.vec3, device=dev)))
        S = self.control.sphere_count if self.control is not None else 0
        self.sphere_radii = wp.zeros(S, dtype=wp.float32, device=dev) if S else None  # 0 for inactive spheres
        if S:  # collision spheres magenta, a scissors' cutting spheres yellow
            colors = np.where((self.control.host_sphere_chain == CHAIN_CUT)[:, None], np.float32(CUT_SPHERE_COLOR), np.float32(SPHERE_COLOR))
            self.sphere_colors = wp.array(np.ascontiguousarray(colors, dtype=np.float32), dtype=wp.vec3, device=dev)
        else:
            self.sphere_colors = None

    def add_markers(self, count: int):
        """The app's idle-device proxies as points: host arrays assigned each publish (the GL view is single-threaded)."""
        self.markers = _GlMarkers(count, self.scene.device)
        return self.markers

    def publish(self, snap: Snapshot | None = None) -> None:
        snap = self.live if snap is None else snap
        if self.task_markers is not None:
            from i4h_standalone.render.procedure import publish_task
            publish_task(self, snap)
            self.task_markers.publish()
        if getattr(self, "markers", None) is not None:
            self.markers.publish()
        for rd, pos, normals, _ in self.rods:
            rd.place(snap, self.env, pos, normals, self.scene.device)
        for _, sk, pos, normals, tangents, uvs, scratch, indices, _ in self.skinned:
            sk.place(snap, self.env, pos, self.scene.device)
            update_normals_tangents(pos, indices, uvs, scratch, normals, tangents, self.scene.device)
        for cd, pos, normals, _ in self.clips:
            cd.place(snap, self.env, pos, normals, self.scene.device)
        if self.skins:
            skin_pieces(snap.control, self.env, self.skins, self.piece_points, self.piece_normals, self.scene.device)
        dev, env = self.scene.device, int(self.env)
        if self.sphere_radii is not None:
            c = snap.control
            wp.launch(active_radii, dim=c.sphere_count, inputs=[c.sphere_radius, c.sphere_active, env],
                      outputs=[self.sphere_radii], device=dev)
        for body, radius, radii, _ in self.fluids:
            wp.launch(alive_radii, dim=body.count, inputs=[self.scene.view(snap.pool_flags, body)[env], radius],
                      outputs=[radii], device=dev)
        for smoke, pos, radii, _ in self.smokes:
            n = int(smoke.count)
            pos_age, vel_life = snap.smoke[smoke]
            wp.launch(smoke_points, dim=n, inputs=[pos_age, vel_life, env, n, 0.5 * smoke.start_size],
                      outputs=[pos, radii], device=dev)

    def draw(self) -> bool:
        self.viewer.begin_frame(self.time)
        host = {}  # one device -> host copy per surface system per frame (debug view: the host path is allowed)
        for name, points, p, source, color, indices in self.parts:
            if isinstance(source, Surface):
                if id(source) not in host:
                    host[id(source)] = source.tris[self.env].numpy()  # one env's row, not the whole table
                tris = host[id(source)]
            else:
                tris = source
            local = tris[p.tri_start : p.tri_start + p.tri_count] - p.particle_start
            indices.assign(np.ascontiguousarray(local.reshape(-1), dtype=np.int32))
            self.viewer.log_mesh(name, points, indices, color=color)
        for skin, points, normals, indices in zip(self.skins, self.piece_points, self.piece_normals, self.piece_indices):
            self.viewer.log_mesh(f"instruments/{self.control.names[skin.instrument]}/{skin.name}", points, indices, normals=normals,
                                 color=skin.color)
        for rd, pos, normals, indices in self.rods:
            self.viewer.log_mesh(f"rods/{rd.name}", pos, indices, normals=normals, color=rd.color)
        for body, _, pos, normals, _, _, _, indices, color in self.skinned:
            self.viewer.log_mesh(f"skin/{body.name}", pos, indices, normals=normals, color=color)
        for cd, pos, normals, indices in self.clips:
            self.viewer.log_mesh(f"clips/{cd.name}", pos, indices, normals=normals, color=CLIP_COLOR)
        if self.task_markers is not None:
            m = self.task_markers
            self.viewer.log_points("procedure/target", m.q, m.radii, m.colors)
        if getattr(self, "markers", None) is not None:
            m = self.markers
            self.viewer.log_points("markers", m.q, m.radii, m.colors)
        if self.sphere_radii is not None:
            self.viewer.log_points("instruments/spheres", self.control.sphere_q[self.env], self.sphere_radii, self.sphere_colors)
        for body, _, radii, colors in self.fluids:  # dead particles have radius 0 (the viewer wants arrays for both)
            self.viewer.log_points(f"fluid/{body.name}", self.scene.field("q", body)[self.env], radii, colors)
        for i, (_, pos, radii, colors) in enumerate(self.smokes):
            self.viewer.log_points(f"smoke/{i}", pos, radii, colors)
        self.viewer.end_frame()
        self.time += self.sim.dt
        return self.viewer.is_running()

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def read_frame(self):
        return self.viewer.get_frame().numpy().astype(np.float32) / 255.0  # (H, W, 3) uint8 from Newton -> [0, 1]

    def close(self) -> None:
        self.viewer.close()


@wp.kernel
def active_radii(radius: wp.array(dtype=wp.float32), active: wp.array2d(dtype=wp.int32), env: wp.int32, out: wp.array(dtype=wp.float32)):
    s = wp.tid()
    out[s] = radius[s] * wp.float32(active[env, s])


@wp.kernel
def alive_radii(flags: wp.array(dtype=wp.int32), radius: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = radius * wp.float32(flags[i] & ACTIVE)


@wp.kernel
def smoke_points(pos_age: wp.array(dtype=wp.vec4f), vel_life: wp.array(dtype=wp.vec4f), env: wp.int32, count: wp.int32,
                 radius: wp.float32, pos: wp.array(dtype=wp.vec3), radii: wp.array(dtype=wp.float32)):
    i = wp.tid()
    pa = pos_age[env * count + i]
    pos[i] = wp.vec3(pa[0], pa[1], pa[2])
    radii[i] = radius * wp.float32(pa[3] < vel_life[env * count + i][3])


class _GlMarkers:
    """Idle-device proxies for the GL view: positions, radii and colours assigned from the host each publish."""

    def __init__(self, count: int, device):
        self.count = count
        self.host_q = np.zeros((count, 3), np.float32)
        self.host_r = np.zeros(count, np.float32)
        self.host_c = np.tile(np.float32(MARKER_IDLE_COLOR), (count, 1))
        self.q = wp.zeros(count, dtype=wp.vec3, device=device)
        self.radii = wp.zeros(count, dtype=wp.float32, device=device)
        self.colors = wp.array(self.host_c, dtype=wp.vec3, device=device)

    def set(self, i: int, position, radius: float, candidate: bool) -> None:
        self.host_q[i] = position
        self.host_r[i] = radius
        self.host_c[i] = MARKER_CANDIDATE_COLOR if candidate else MARKER_IDLE_COLOR

    def publish(self) -> None:
        self.q.assign(self.host_q)
        self.radii.assign(self.host_r)
        self.colors.assign(self.host_c)


class _KeyPress:
    """A pyglet key press in the shape ``render.camera.key_event`` reads (slangpy names: ``key1``, ``a``, ``page_up``)."""

    def __init__(self, symbol: int):
        from types import SimpleNamespace

        from pyglet.window import key

        name = key.symbol_string(symbol).lower()
        if name.startswith("_") and name[1:].isdigit():  # pyglet's digits are _0.._9, slangpy's key0..key9
            name = "key" + name[1:]
        self.key = SimpleNamespace(name={"pageup": "page_up", "pagedown": "page_down"}.get(name, name))

    def is_key_press(self) -> bool:
        return True

    def is_key_release(self) -> bool:
        return False

    def is_key_repeat(self) -> bool:
        return False


class GlKeys:
    """The GL view's key presses for handlers written against the Slang window (``Pickup.handle_key``): pass this as
    the ``renderer`` of a ``DeviceRows``. Opt-in, so the app's ``--render gl`` keeps its own key bindings."""

    def __init__(self, renderer: GlRenderer):
        self.key_handlers: list = []
        register = getattr(getattr(renderer.viewer, "renderer", None), "register_key_press", None)
        if register is not None:
            register(lambda symbol, modifiers: [handler(_KeyPress(symbol)) for handler in self.key_handlers])
