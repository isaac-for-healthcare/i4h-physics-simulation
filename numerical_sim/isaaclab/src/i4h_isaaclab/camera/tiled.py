# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera observation for every env: Newton's ``SensorTiledCamera`` (a Warp ray tracer) over a render model
derived from the scene, plus our own shading of the tissue pixels. A renderer in the sense of AGENTS.md: it
reads ``Scene`` arrays and system products between steps, never inside the step graph, and owns its own capture.

Render model (a Newton ``Model``/``State`` private to this class, never seen by Isaac Lab; built once):
- tissue: ``model.tri_indices`` is a fixed ``(E * F_total, 3)`` table of global particle ids over
  ``state.particle_q``: ``scene.q`` itself when the case has no fluid pools, else a render buffer holding the main block
  and then the pool block, copied in every render (the sensor draws triangles and spheres from one particle array);
  every render rewrites the table from
  ``surface.tris`` (dead slots stay degenerate, so cuts show) and refits the sensor's ``wp.Mesh``;
- static meshes (the cavity, a vessel): one static shape per env sharing one ``newton.Mesh`` with the
  asset's smooth normals, UVs and diffuse texture;
- instruments: one link per piece per env with the piece mesh in bind space (``bind`` baked into the
  vertices, area-weighted smooth normals), posed every render from ``instrument_control``'s root, shaft,
  wrist and jaw rotations; a piece-less tool (the ``sphere`` kind) shows its collision spheres;
- rods: one capsule per segment per env (radius, rest length, the part's colour), posed every render from
  the node positions, so a catheter is a continuous tube; centreline trees (a ``rod_pbd`` without a ``RodMesh``)
  are not drawn;
- fluid pools: spheres from ``model.particle_radius`` (ours): the pool's ``render_radius`` while a particle
  is ACTIVE, 0 for everything else (tissue nodes, rod nodes and parked slots vanish);
- smoke and procedure target markers are not rendered (including ordered sequences).
Newton shades shapes with their colours, normals and textures but draws the deformable triangle path and
the particles white with face normals, so a second pass (``shade_pixels``) re-traces those pixels: tissue
against our own grouped ``wp.Mesh`` over the same table with per-vertex normals accumulated every render
and a colour per part (the asset's ``visual.color``, else the mean of its diffuse texture, else
``PART_COLORS``) tinted by the surface system's damage, coagulation and blood channels; particles against
the model's particle BVH with the pool's colour; both lit by a headlight. A static mesh with ``cull: back``
(a vessel seen from outside, inward-facing) is see-through as in the Slang renderer: Newton culls its
near wall and stops, so ``shade_wall`` re-traces such meshes skipping back-facing hits and writes the far
wall wherever it is nearer than what Newton found, which keeps a catheter inside visible.
Per world the ray tracer traverses only that world's BVH group, so envs sharing one local frame do not see
each other. Outputs per env: ``rgb`` ``(E, H, W, 4)`` uint8 RGBA (alpha 255 on a hit), ``depth`` ``(E, H, W)``
float32 ray distance in sim units (0 = miss) and ``shape_index`` ``(E, H, W)`` uint32 (``TRIANGLES`` for
tissue, ``PARTICLES`` for fluid and rods, a shape id for instruments and statics, ``MISS``). The camera is
the case's render camera for every env. ``prepare()`` captures the simulator's graphs first if needed,
renders once eagerly (the sensor builds its mesh and compiles its kernel) and captures a second render into
a graph that ``render()`` replays. Known limits: no textures or normal maps on the tissue (a flat colour per
part), Newton's ambient term assumes Z-up, capsules keep a segment's rest length, one camera pose per env.
"""

from __future__ import annotations

import math

import newton
import numpy as np
import warp as wp
from i4h_isaaclab.appearance import appearance
from i4h_physics.scene import ACTIVE
from newton.sensors import SensorTiledCamera

try:
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
except ModuleNotFoundError as error:
    if error.name != "i4h_surgical":
        raise
    InstrumentControl = ()
from i4h_isaaclab.camera import FLUID_COLOR, PART_COLORS, ROD_COLOR
from i4h_isaaclab.camera.camera import Camera
from i4h_isaaclab.camera.rig import resolve_tip
from i4h_physics.loaders.static_mesh import smooth_normals
from i4h_physics.quat import matrix_to_quat
from i4h_physics.systems.mechanics.surface import Surface
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd

STATIC_COLOR = (0.80, 0.55, 0.50)  # untextured static meshes
DAMAGE_COLOR = wp.constant(wp.vec3(0.22, 0.10, 0.06))  # tint targets of the surface's visual channels
COAG_COLOR = wp.constant(wp.vec3(0.85, 0.80, 0.70))
BLOOD_COLOR = wp.constant(wp.vec3(*FLUID_COLOR))
AMBIENT = wp.constant(0.35)


def flat_part_color(name: str, visual: dict | None) -> tuple:
    """A flat colour for a tissue part: ``visual.color``, else the mean of its diffuse texture, else the palette."""
    visual = visual or {}
    if "color" in visual:
        return tuple(float(c) for c in visual["color"])[:3]
    if "diffuse" in visual:
        from PIL import Image
        try:
            with Image.open(visual["diffuse"]) as image:
                mean = np.asarray(image.convert("RGB").resize((64, 64)), np.float32).reshape(-1, 3).mean(axis=0) / 255.0
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Expected texture {visual['diffuse']}. Set I4H_DATA or pass --data.\n"
                                    "cd numerical_sim\nuv run i4h-sim chole_tet_med --data ./data") from error
        return tuple(float(c) for c in mean)
    return PART_COLORS.get(name, (0.6, 0.6, 0.6))


class TiledCamera:
    TRIANGLES = 0xFFFFFFFD  # shape_index of a tissue (triangle-mesh) hit
    PARTICLES = 0xFFFFFFFE  # shape_index of a fluid or rod (particle) hit
    MISS = 0xFFFFFFFF

    def __init__(self, sim, width: int, height: int, camera=None, camera_mode="free", tip=None):
        """``camera_mode: tip`` poses every env's camera at its rod tip or along its instrument's shaft each render
        (``render.tip``, ``render/rig.py``); ``free`` tiles the case camera."""
        scene = sim.scene
        E, dev = scene.envs, scene.device
        self.sim, self.scene, self.width, self.height = sim, scene, int(width), int(height)
        self.graph = None
        self.control = next((s for s in sim.systems if isinstance(s, InstrumentControl)), None)
        self.surfaces = [s for s in sim.systems if isinstance(s, Surface)]
        rods = {id(s.body): s for s in sim.systems if isinstance(s, (RodDirect, RodPbd))}
        cam = Camera(**dict(camera or {}))
        self.fov = float(cam.fov)
        if camera_mode not in ("free", "tip"):
            raise ValueError(f"camera_mode must be free or tip, got {camera_mode!r}")
        self.tip = None
        if camera_mode == "tip":
            kind, target, settings, _ = resolve_tip(sim, tip)
            self.tip = (kind, target, settings)
            self.fov = settings["fov"]
        self.world_up = wp.vec3f(*(-scene.down).astype(np.float32).tolist())
        self.camera_transforms = None  # created with the sensor below; the tip kernels write it once it exists

        # --- shapes: static meshes, instrument pieces (bind baked in), the spheres of piece-less tools ---------------
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.flags = newton.ShapeFlags.VISIBLE  # render only: no collision work in the model
        statics = []
        for m in scene.static_meshes:
            textured = m.uvs is not None and "diffuse" in appearance(m)
            color = (1.0, 1.0, 1.0) if textured else tuple(appearance(m).get("color", STATIC_COLOR))[:3]
            statics.append(newton.Mesh(m.points, m.indices.reshape(-1), normals=m.normals, uvs=m.uvs if textured else None,
                                       texture=appearance(m)["diffuse"] if textured else None, compute_inertia=False, color=color))
        pieces = []  # (instrument index, role, jaw_sign, bind rotation, bind translation, newton.Mesh)
        spheres = []  # (sphere index, radius) of tools without pieces
        self.piece_points = []  # bind-space vertices per piece (host), what body_q transforms; for tests
        if self.control is not None:
            radius = self.control.sphere_radius.numpy() if self.control.sphere_count else np.zeros(0, np.float32)
            owner = self.control.sphere_instrument.numpy() if self.control.sphere_count else np.zeros(0, np.int32)
            for k, record in enumerate(scene.instruments.values()):
                for piece in record.pieces:
                    bind = np.asarray(piece.bind, np.float64)
                    R, t = bind[:3, :3], bind[:3, 3]
                    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0.0:
                        raise ValueError(f"instrument {record.name!r} piece {piece.name!r}: bind is not a rigid transform")
                    points = (np.asarray(piece.points, np.float64) @ R.T + t).astype(np.float32)
                    faces = np.asarray(piece.indices, np.int32).reshape(-1, 3)
                    mesh = newton.Mesh(points, faces.reshape(-1), normals=smooth_normals(points, faces, np.arange(len(points))), compute_inertia=False,
                                       color=tuple(float(c) for c in piece.color)[:3])
                    pieces.append((k, int(piece.role), float(piece.jaw_sign), matrix_to_quat(R), t, mesh))
                    self.piece_points.append(points)
                if not record.pieces:
                    spheres += [(int(s), float(radius[s])) for s in np.flatnonzero(owner == k)]
        capsules = []  # (body offset, env stride, segment, radius, half length, colour): one per rod segment
        for b in scene.bodies:
            if id(b) in rods and b.rod is not None:  # a centreline tree has no RodMesh: not drawn
                part = next(iter(b.parts.values()))
                color = tuple(float(c) for c in (appearance(part) or {}).get("color", ROD_COLOR))[:3]
                r = float(b.rest.radius[0, 0])
                for j in range(b.count - 1):
                    capsules.append((*scene.layout(b), j, r, 0.5 * float(b.rod.rest_len[0, j]), color))
        self.P, self.S, self.C = len(pieces), len(spheres), len(capsules)
        self.static_count = len(statics)
        for e in range(E):
            builder.begin_world()
            for mesh in statics:
                builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
            for _, _, _, _, _, mesh in pieces:
                builder.add_shape_mesh(builder.add_link(), mesh=mesh, cfg=cfg)
            for _, r in spheres:
                builder.add_shape_sphere(builder.add_link(), radius=r, cfg=cfg, color=(0.95, 0.28, 0.85))
            for _, _, _, r, half, color in capsules:
                builder.add_shape_capsule(builder.add_link(), radius=r, half_height=half, cfg=cfg, color=color)
            builder.end_world()
        self.model = model = builder.finalize(device=dev)
        self.B = self.P + self.S + self.C  # bodies per env: pieces, spheres, capsules
        assert model.body_count == E * self.B
        self.cap_start = wp.array(np.asarray([c[0] for c in capsules], np.int32).reshape(-1), dtype=wp.int32, device=dev)
        self.cap_stride = wp.array(np.asarray([c[1] for c in capsules], np.int32).reshape(-1), dtype=wp.int32, device=dev)
        self.cap_node = wp.array(np.asarray([c[2] for c in capsules], np.int32).reshape(-1), dtype=wp.int32, device=dev)
        # See-through statics (``cull: back``): our own mesh per such asset for the far-wall pass.
        self.walls = [(wp.Mesh(points=wp.array(m.points, dtype=wp.vec3, device=dev),
                               indices=wp.array(m.indices.reshape(-1).astype(np.int32), dtype=wp.int32, device=dev)),
                       wp.vec3(*tuple(appearance(m).get("color", STATIC_COLOR))[:3]), index)
                      for index, m in enumerate(scene.static_meshes) if appearance(m).get("cull") == "back"]
        # Deformable shells (vessel walls): a grouped mesh per shell body over its live positions (one group per env,
        # static topology), refit each render and drawn by the shell pass: see-through from outside with ``cull: back``
        # like the static walls above, opaque otherwise. Newton's sensor does not see them (main-block radius 0, no
        # triangles in its table).
        self.shells = []  # (mesh, colour, cull back, body)
        for b in scene.bodies:
            if b.shell is None:
                continue
            if b.shell.tris.shape[0] != 1:
                raise ValueError(f"TiledCamera: shell body {b.name!r} has {b.shell.tris.shape[0]} mesh variants; one is supported")
            start, stride = scene.layout(b)
            tris = np.asarray(b.shell.tris[0], np.int32)
            indices = np.concatenate([start + e * stride + tris for e in range(E)]).reshape(-1)
            groups = np.repeat(np.arange(E, dtype=np.int32), tris.shape[0])
            name, part = next(iter(b.parts.items()))
            look = appearance(part)
            self.shells.append((wp.Mesh(points=scene.q, indices=wp.array(indices, dtype=wp.int32, device=dev),
                                        groups=wp.array(groups, dtype=wp.int32, device=dev), bvh_constructor="sah"),
                                wp.vec3(*flat_part_color(name, look)), int(look.get("cull") == "back"), b))
        self.body_q = wp.zeros(max(1, model.body_count), dtype=wp.transformf, device=dev)
        def i32(values):
            return wp.array(np.asarray(values, np.int32).reshape(-1), dtype=wp.int32, device=dev)
        self.piece_instrument = i32([p[0] for p in pieces])
        self.piece_role = i32([p[1] for p in pieces])
        self.piece_jaw_sign = wp.array(np.asarray([p[2] for p in pieces], np.float32), dtype=wp.float32, device=dev)
        self.piece_bind_q = wp.array(np.asarray([p[3] for p in pieces], np.float32).reshape(-1, 4), dtype=wp.quat, device=dev)
        self.piece_bind_t = wp.array(np.asarray([p[4] for p in pieces], np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
        self.sphere_index = i32([s[0] for s in spheres])

        # --- particles: our own point buffer (the main block, then the pool block), radii, triangles, colours ----------
        # The sensor draws triangles and spheres from one particle array, so the render keeps a buffer holding the
        # main block (triangle corners, radius 0) followed by the pool block (fluid spheres) and copies both in per
        # render; the triangle table indexes the first part unchanged.
        N, M = scene.q.shape[0], scene.pool.q.shape[0]
        self.N = N
        env_of = np.zeros(N + M, np.int32)  # host particles (not drawn) stay in world 0
        base_radius = np.zeros(M, np.float32)
        particle_color = np.ones((N + M, 3), np.float32)
        for b in scene.bodies + scene.pools:  # flat slots of (env, local) from Scene.layout; pools follow the main block
            start, stride = scene.layout(b)
            shift = N if b.pool else 0
            for e in range(E):
                rows = slice(start + e * stride, start + e * stride + b.count)
                env_of[shift + rows.start : shift + rows.stop] = e
                if b.pool:
                    base_radius[rows] = float(appearance(b.fluid).get("render_radius", b.fluid.rest_distance))
                    particle_color[N + rows.start : N + rows.stop] = np.float32(tuple(appearance(b.fluid).get("color", FLUID_COLOR))[:3])
        self.base_radius = wp.array(base_radius, dtype=wp.float32, device=dev)  # (M,): pool slots only
        self.particle_color = wp.array(particle_color, dtype=wp.vec3, device=dev)
        self.points = scene.q if M == 0 else wp.zeros(N + M, dtype=wp.vec3, device=dev)  # no pools: alias, no copy
        self.radii = wp.zeros(N + M, dtype=wp.float32, device=dev)  # main-block slots stay 0 (not drawn as spheres)
        self.has_particles = M > 0  # pools draw as spheres; rods are capsules
        self.offsets = np.cumsum([0] + [s.tris.shape[1] for s in self.surfaces])  # slot offset of each surface's table
        self.F_total = int(self.offsets[-1])
        rows = E * max(1, self.F_total)
        self.tris = wp.zeros((rows, 3), dtype=wp.int32, device=dev)
        row_color = np.zeros((E, max(1, self.F_total), 3), np.float32)
        for s, offset in zip(self.surfaces, self.offsets):
            colors = np.asarray([flat_part_color(name, appearance(p)) for name, p in s.body.parts.items()], np.float32)
            row_color[:, offset : offset + s.tris.shape[1]] = colors[s.tri_part.numpy()][None]
        self.row_color = wp.array(row_color.reshape(rows, 3), dtype=wp.vec3, device=dev)
        self.tint = wp.zeros(N, dtype=wp.vec4f, device=dev)  # the surface systems' (damage, coag, blood, -) per particle
        self.normal_accum = wp.zeros(N, dtype=wp.vec3, device=dev)
        self.normals = wp.zeros(N, dtype=wp.vec3, device=dev)
        self._copy_points()
        model.particle_q = self.points
        model.particle_world = wp.array(env_of, dtype=wp.int32, device=dev)
        model.particle_radius = self.radii
        model.particle_count = N + M
        model.tri_indices = self.tris if self.F_total else None
        model.tet_indices = None
        self.state = newton.State()
        self.state.particle_q = self.points
        self.state.body_q = self.body_q if model.body_count else None
        self._update_geometry()  # groups are derived from tri[0] when the meshes are built
        # The pool slots of an env are parked at one point, so Warp's default LBVH (Morton codes) degenerates on
        # them: a 1.1 M-leaf refit took 13 ms and the render 15 ms at 64 envs; a median split keeps the tree
        # balanced (0.2 ms refit, 1.6 ms render).
        model.bvh_build_particles(self.state, bvh_constructor="median")
        # Our own grouped mesh over the same table, for the shading pass (Newton's is private to the sensor).
        self.tris_flat = self.tris.reshape((rows * 3,))
        assert self.tris_flat.ptr == self.tris.ptr
        self.mesh = None
        if self.F_total:
            row_env = wp.array(np.repeat(np.arange(E, dtype=np.int32), self.F_total), dtype=wp.int32, device=dev)
            self.mesh = wp.Mesh(points=scene.q, indices=self.tris_flat, groups=row_env, bvh_constructor="sah")

        # --- the sensor, its camera, light and outputs -------------------------------------------------------------------
        config = SensorTiledCamera.RenderConfig(enable_global_world=False, enable_shadows=False, enable_particles=self.has_particles,
                                                enable_textures=True, render_order=SensorTiledCamera.RenderOrder.TILED)
        self.sensor = SensorTiledCamera(model, default_render_config=config, load_textures=True)
        position, right, up, forward = cam.basis()
        rotation = np.stack([right, up, -forward], axis=1).astype(np.float64)  # Newton's camera looks down -Z, +Y up
        q = matrix_to_quat(rotation)
        xf = np.tile(np.concatenate([position, q]).astype(np.float32), (1, E, 1))
        self.camera_transforms = wp.array(xf, dtype=wp.transformf, device=dev)  # (1 camera, E worlds)
        self.rays = self.sensor.utils.compute_camera_rays_pinhole(self.width, self.height, camera_fovs=math.radians(self.fov))
        self.sensor.utils.create_default_light(enable_shadows=False, direction=wp.vec3f(*forward))  # a headlight
        self.color = self.sensor.utils.create_color_image_output(self.width, self.height, 1)  # (E, 1, H, W) uint32
        self.depth4 = self.sensor.utils.create_depth_image_output(self.width, self.height, 1)
        self.shape4 = self.sensor.utils.create_shape_index_image_output(self.width, self.height, 1)
        self.rgb = self.sensor.utils.to_rgba_from_color(self.color)  # zero-copy uint8 view (E, H, W, 4)
        self.depth = self.depth4.reshape((E, self.height, self.width))
        self.shape_index = self.shape4.reshape((E, self.height, self.width))
        self.rgb_t = wp.to_torch(self.rgb)
        self.depth_t = wp.to_torch(self.depth)

    # per render ---------------------------------------------------------------------------------------------------

    def _copy_points(self):
        scene, N = self.scene, self.N
        if self.points.ptr == scene.q.ptr:  # no pools: the render buffer is scene.q itself
            return
        if N:
            wp.copy(self.points[:N], scene.q)
        if scene.pool.q.shape[0]:
            wp.copy(self.points[N:], scene.pool.q)

    def _update_geometry(self):
        scene, dev, E = self.scene, self.scene.device, self.scene.envs
        self._copy_points()
        if self.P or self.S:
            c = self.control
            wp.launch(place_bodies, dim=(E, self.P + self.S),
                      inputs=[c.root, c.shaft_q, c.wrist_q, c.jaw_q, c.jaw_angle, c.wrist_pivot, c.jaw_anchor, c.sphere_q,
                              self.piece_instrument, self.piece_role, self.piece_jaw_sign, self.piece_bind_q, self.piece_bind_t,
                              self.sphere_index, self.P, self.B],
                      outputs=[self.body_q], device=dev)
        if self.C:
            wp.launch(place_capsules, dim=(E, self.C),
                      inputs=[scene.q, self.cap_start, self.cap_stride, self.cap_node, self.P + self.S, self.B],
                      outputs=[self.body_q], device=dev)
        for s, offset in zip(self.surfaces, self.offsets):
            b = s.body
            start, stride = scene.layout(b)
            wp.launch(global_tris, dim=(E, s.tris.shape[1]),
                      inputs=[s.tris, start, stride, int(offset), self.F_total], outputs=[self.tris], device=dev)
            if hasattr(s, "appearance"):
                wp.launch(scatter_tint, dim=(E, b.count), inputs=[s.appearance.visual, start, stride, b.count], outputs=[self.tint], device=dev)
        if self.has_particles:
            wp.launch(render_radii, dim=self.base_radius.shape[0], inputs=[scene.pool.flags, self.base_radius],
                      outputs=[self.radii[self.N :]], device=dev)
        if self.F_total:
            wp.launch(accumulate_normals, dim=self.tris.shape[0], inputs=[self.tris, scene.q], outputs=[self.normal_accum], device=dev)
            for s in self.surfaces:  # the tissue rows only: nothing reads a pool slot's normal
                b = s.body
                wp.launch(finalize_normals, dim=(E, b.count), inputs=[*scene.layout(b)], outputs=[self.normal_accum, self.normals],
                          device=dev)
        if self.tip is not None and self.camera_transforms is not None:  # per-env poses from the live tip (graph-safe: fixed arrays)
            kind, target, s = self.tip
            if kind == "rod":
                body, rod = target
                wp.launch(tip_pose_rod, dim=E, inputs=[scene.field("q", body), rod.frame, body.count, rod.frame_count, s["distance"], self.world_up],
                          outputs=[self.camera_transforms], device=dev)
            else:
                wp.launch(tip_pose_instrument, dim=E, inputs=[self.control.root, self.control.shaft_q, int(target), s["distance"], self.world_up],
                          outputs=[self.camera_transforms], device=dev)

    def _render(self):
        self._update_geometry()
        if self.model.shape_count:
            self.model.bvh_refit_shapes(self.state)
        if self.has_particles:
            self.model.bvh_refit_particles(self.state)
        self.sensor.update(self.state, self.camera_transforms, self.rays, color_image=self.color, depth_image=self.depth4,
                           shape_index_image=self.shape4)
        dims, dev = (self.scene.envs, self.height, self.width), self.scene.device
        if self.mesh is not None:
            self.mesh.refit()
        if self.F_total or self.has_particles:  # tissue and sphere pixels; a capsules-and-statics scene skips the pass
            mesh_id = self.mesh.id if self.mesh is not None else wp.uint64(0)
            wp.launch(shade_pixels, dim=dims,
                      inputs=[self.shape4, self.camera_transforms, self.rays, mesh_id, self.tris, self.normals, self.tint, self.row_color,
                              self.model.bvh_particles.id, self.model.bvh_particles_group_roots, self.points, self.radii, self.particle_color],
                      outputs=[self.color], device=dev)
        for mesh, color, index in self.walls:
            wp.launch(shade_wall, dim=dims,
                      inputs=[self.camera_transforms, self.rays, mesh.id, color, self.static_count + self.B, index],
                      outputs=[self.color, self.depth4, self.shape4], device=dev)
        for mesh, color, cull, _ in self.shells:
            mesh.refit()
            wp.launch(shade_shell, dim=dims, inputs=[self.camera_transforms, self.rays, mesh.id, color, cull],
                      outputs=[self.color, self.depth4, self.shape4], device=dev)

    def prepare(self):
        """One eager render (the sensor builds its triangle mesh and compiles its kernel), then a captured one.
        The simulator's own graphs are captured first, so the step graph never captures after this one."""
        self.graph = None
        dev = self.scene.device
        if self.sim._prepared is None:
            self.sim.prepare()
        self._render()
        with wp.ScopedDevice(dev):
            with wp.ScopedCapture(device=dev) as capture:
                self._render()
        self.graph = capture.graph

    def render(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._render()

    def sheet(self, worlds_per_row: int | None = None) -> np.ndarray:
        """All envs' last frames tiled into one ``(rows * H, cols * W, 4)`` uint8 image (host)."""
        tiled = self.sensor.utils.flatten_color_image_to_rgba(self.color, worlds_per_row=worlds_per_row)
        wp.synchronize_device(self.scene.device)
        return tiled.numpy()


# --- kernels ------------------------------------------------------------------------------------------------------


@wp.kernel
def place_bodies(
    root: wp.array2d(dtype=wp.vec3),
    shaft_q: wp.array2d(dtype=wp.quat),
    wrist_q: wp.array2d(dtype=wp.quat),
    jaw_q: wp.array2d(dtype=wp.quat),
    jaw_angle: wp.array2d(dtype=wp.float32),
    wrist_pivot: wp.array(dtype=wp.vec3),
    jaw_anchor: wp.array(dtype=wp.vec3),
    sphere_q: wp.array2d(dtype=wp.vec3),
    piece_instrument: wp.array(dtype=wp.int32),
    piece_role: wp.array(dtype=wp.int32),
    piece_jaw_sign: wp.array(dtype=wp.float32),
    piece_bind_q: wp.array(dtype=wp.quat),
    piece_bind_t: wp.array(dtype=wp.vec3),
    sphere_index: wp.array(dtype=wp.int32),
    P: wp.int32,
    B: wp.int32,
    body_q: wp.array(dtype=wp.transformf),
):
    """Body ``e * B + i``: piece ``i < P`` gets ``instrument_control.skin_point``'s rigid chain as one transform on
    bind-space vertices (the jaw hinge becomes a rotation about the bind-transformed X axis); sphere ``i - P`` sits
    at its collision-sphere centre. ``B`` bodies per env (pieces, spheres, then the rod capsules)."""
    env, i = wp.tid()
    if i >= P:
        body_q[env * B + i] = wp.transformf(sphere_q[env, sphere_index[i - P]], wp.quat_identity())
        return
    k = piece_instrument[i]
    hinge = wp.transform_identity()
    if piece_jaw_sign[i] != 0.0:
        axis = wp.quat_rotate(piece_bind_q[i], wp.vec3(1.0, 0.0, 0.0))
        rot = wp.quat_from_axis_angle(axis, piece_jaw_sign[i] * jaw_angle[env, k])
        t = piece_bind_t[i]
        hinge = wp.transformf(t - wp.quat_rotate(rot, t), rot)
    role = piece_role[i]
    pivot_world = root[env, k] + wp.quat_rotate(shaft_q[env, k], wrist_pivot[k])
    if role == 0:
        link = wp.transformf(root[env, k], shaft_q[env, k])
    elif role == 1:
        link = wp.transformf(pivot_world - wp.quat_rotate(wrist_q[env, k], wrist_pivot[k]), wrist_q[env, k])
    else:
        anchor_world = pivot_world + wp.quat_rotate(wrist_q[env, k], jaw_anchor[k] - wrist_pivot[k])
        link = wp.transformf(anchor_world - wp.quat_rotate(jaw_q[env, k], jaw_anchor[k]), jaw_q[env, k])
    body_q[env * B + i] = wp.transform_multiply(link, hinge)


@wp.func
def quat_from_z(direction: wp.vec3) -> wp.quat:
    """Rotation taking +Z (a capsule's axis) onto ``direction``."""
    z = wp.vec3(0.0, 0.0, 1.0)
    axis = wp.cross(z, direction)
    s = wp.length(axis)
    c = wp.dot(z, direction)
    if s < 1.0e-6:
        if c > 0.0:
            return wp.quat_identity()
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 3.14159265)
    return wp.quat_from_axis_angle(axis / s, wp.atan2(s, c))


@wp.kernel
def place_capsules(
    q: wp.array(dtype=wp.vec3),
    cap_start: wp.array(dtype=wp.int32),
    cap_stride: wp.array(dtype=wp.int32),
    cap_node: wp.array(dtype=wp.int32),
    first: wp.int32,
    B: wp.int32,
    body_q: wp.array(dtype=wp.transformf),
):
    """Capsule ``c`` of env ``e`` (body ``e * B + first + c``) spans rod nodes ``j`` and ``j + 1``."""
    env, c = wp.tid()
    i = cap_start[c] + env * cap_stride[c] + cap_node[c]  # flat index (Scene.layout)
    p0, p1 = q[i], q[i + 1]
    d = p1 - p0
    length = wp.length(d)
    rot = wp.quat_identity()
    if length > 1.0e-8:
        rot = quat_from_z(d / length)
    body_q[env * B + first + c] = wp.transformf(0.5 * (p0 + p1), rot)


@wp.kernel
def global_tris(
    tris: wp.array2d(dtype=wp.vec3i),
    start: wp.int32,
    stride: wp.int32,
    offset: wp.int32,
    F_total: wp.int32,
    out: wp.array2d(dtype=wp.int32),
):
    """Row ``env * F_total + offset + f``: the surface's body-local triangle as global particle ids (a degenerate
    slot stays degenerate and keeps a particle of the same env, so its BVH group never changes)."""
    env, f = wp.tid()
    tri = tris[env, f]
    base = start + env * stride  # Scene.layout
    row = env * F_total + offset + f
    out[row, 0] = base + tri[0]
    out[row, 1] = base + tri[1]
    out[row, 2] = base + tri[2]


@wp.kernel
def scatter_tint(visual: wp.array(dtype=wp.vec4f), start: wp.int32, stride: wp.int32, count: wp.int32, tint: wp.array(dtype=wp.vec4f)):
    """A surface system's body-local ``(E * count)`` visual state into the flat per-particle tint (Scene.layout)."""
    env, local = wp.tid()
    tint[start + env * stride + local] = visual[env * count + local]


@wp.kernel
def render_radii(flags: wp.array(dtype=wp.int32), base: wp.array(dtype=wp.float32), radii: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if (flags[i] & ACTIVE) != 0:
        radii[i] = base[i]
    else:
        radii[i] = 0.0


# render/kernels.py accumulate_frames / finalize_frames without the tangents: area-weighted normals over the global
# table; the accumulator is cleared by the finalize pass, so it is zero between renders.
@wp.kernel
def accumulate_normals(tris: wp.array2d(dtype=wp.int32), positions: wp.array(dtype=wp.vec3), accum: wp.array(dtype=wp.vec3)):
    row = wp.tid()
    i, j, k = tris[row, 0], tris[row, 1], tris[row, 2]
    n = wp.cross(positions[j] - positions[i], positions[k] - positions[i])
    if wp.length(n) > 1.0e-12:  # degenerate (dead) slots add nothing
        wp.atomic_add(accum, i, n)
        wp.atomic_add(accum, j, n)
        wp.atomic_add(accum, k, n)


@wp.kernel
def finalize_normals(offset: wp.int32, stride: wp.int32, accum: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3)):
    env, local = wp.tid()
    i = offset + env * stride + local  # Scene.layout
    n = accum[i]
    if wp.length(n) > 1.0e-12:
        normals[i] = wp.normalize(n)
    else:
        normals[i] = wp.vec3(0.0, 1.0, 0.0)
    accum[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def pack_rgba(c: wp.vec3) -> wp.uint32:
    r = wp.uint32(wp.clamp(c[0], 0.0, 1.0) * 255.0 + 0.5)
    g = wp.uint32(wp.clamp(c[1], 0.0, 1.0) * 255.0 + 0.5)
    b = wp.uint32(wp.clamp(c[2], 0.0, 1.0) * 255.0 + 0.5)
    return r + g * wp.uint32(256) + b * wp.uint32(65536) + wp.uint32(255) * wp.uint32(16777216)


@wp.func
def headlight(n: wp.vec3, direction: wp.vec3) -> float:
    """Ambient plus Lambert from a light along the view ray; the normal is flipped toward the camera."""
    facing = n
    if wp.dot(n, direction) > 0.0:
        facing = -n
    return AMBIENT + (1.0 - AMBIENT) * wp.max(wp.dot(facing, -direction), 0.0)


@wp.kernel
def shade_pixels(
    shape_index: wp.array4d(dtype=wp.uint32),
    camera_transforms: wp.array2d(dtype=wp.transformf),
    rays: wp.array4d(dtype=wp.vec3),
    mesh: wp.uint64,
    tris: wp.array2d(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
    tint: wp.array(dtype=wp.vec4f),
    row_color: wp.array(dtype=wp.vec3),
    particles: wp.uint64,
    particle_roots: wp.array(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    particle_color: wp.array(dtype=wp.vec3),
    color: wp.array4d(dtype=wp.uint32),
):
    """Newton's white pixels: a tissue hit is re-traced against our grouped mesh and shaded with the interpolated
    vertex normal and the part colour tinted by damage, coagulation and blood; a particle hit against the model's
    particle BVH (nearest sphere) and shaded with the pool's colour; both with the headlight."""
    env, y, x = wp.tid()
    kind = shape_index[env, 0, y, x]
    if kind != wp.uint32(0xFFFFFFFD) and kind != wp.uint32(0xFFFFFFFE):
        return
    xf = camera_transforms[0, env]
    origin = wp.transform_point(xf, rays[0, y, x, 0])
    direction = wp.normalize(wp.transform_vector(xf, rays[0, y, x, 1]))
    if kind == wp.uint32(0xFFFFFFFD):
        if mesh == wp.uint64(0):
            return
        query = wp.mesh_query_ray(mesh, origin, direction, 1.0e6, wp.mesh_get_group_root(mesh, env))
        if not query.result:
            return
        f = query.face
        i, j, k = tris[f, 0], tris[f, 1], tris[f, 2]
        w = 1.0 - query.u - query.v
        n = normals[i] * query.u + normals[j] * query.v + normals[k] * w
        if wp.length(n) > 1.0e-12:
            n = wp.normalize(n)
        else:
            n = query.normal
        t = tint[i] * query.u + tint[j] * query.v + tint[k] * w
        c = row_color[f]
        c = wp.lerp(c, DAMAGE_COLOR, wp.clamp(t[0], 0.0, 1.0))
        c = wp.lerp(c, COAG_COLOR, wp.clamp(t[1], 0.0, 1.0))
        c = wp.lerp(c, BLOOD_COLOR, wp.clamp(t[2], 0.0, 1.0))
        color[env, 0, y, x] = pack_rgba(c * headlight(n, direction))
        return
    root = particle_roots[env]
    if root < 0:
        return
    best = float(1.0e30)
    hit = int(-1)
    sphere_query = wp.bvh_query_ray(particles, origin, direction, root)
    index = int(0)
    while wp.bvh_query_next(sphere_query, index):
        r = radii[index]
        if r <= 0.0:
            continue
        oc = origin - q[index]
        b = wp.dot(oc, direction)
        disc = b * b - (wp.dot(oc, oc) - r * r)
        if disc >= 0.0:
            t_hit = -b - wp.sqrt(disc)
            if t_hit > 0.0 and t_hit < best:
                best = t_hit
                hit = index
    if hit < 0:
        return
    n = wp.normalize(origin + direction * best - q[hit])
    color[env, 0, y, x] = pack_rgba(particle_color[hit] * headlight(n, direction))


@wp.kernel
def shade_wall(
    camera_transforms: wp.array2d(dtype=wp.transformf),
    rays: wp.array4d(dtype=wp.vec3),
    mesh: wp.uint64,
    wall_color: wp.vec3,
    shapes_per_env: wp.int32,
    static_index: wp.int32,
    color: wp.array4d(dtype=wp.uint32),
    depth: wp.array4d(dtype=wp.float32),
    shape_index: wp.array4d(dtype=wp.uint32),
):
    """A ``cull: back`` static seen from outside: skip its back-facing (near) walls and draw the first front-facing
    one where it is nearer than what Newton found (Newton culls the near wall and stops, so the mesh vanished)."""
    env, y, x = wp.tid()
    xf = camera_transforms[0, env]
    origin = wp.transform_point(xf, rays[0, y, x, 0])
    direction = wp.normalize(wp.transform_vector(xf, rays[0, y, x, 1]))
    start = float(0.0)
    found = float(-1.0)
    n = wp.vec3(0.0, 0.0, 1.0)
    for _ in range(6):
        query = wp.mesh_query_ray(mesh, origin + direction * start, direction, 1.0e6)
        if not query.result:
            break
        t = start + query.t
        if wp.dot(query.normal, direction) < 0.0:
            found = t
            n = query.normal
            break
        start = t + 1.0e-4
    if found < 0.0:
        return
    current = depth[env, 0, y, x]
    if current > 0.0 and current <= found:
        return
    color[env, 0, y, x] = pack_rgba(wall_color * headlight(n, direction))
    depth[env, 0, y, x] = found
    shape_index[env, 0, y, x] = wp.uint32(env * shapes_per_env + static_index)


@wp.kernel
def shade_shell(
    camera_transforms: wp.array2d(dtype=wp.transformf),
    rays: wp.array4d(dtype=wp.vec3),
    mesh: wp.uint64,
    wall_color: wp.vec3,
    cull_back: wp.int32,
    color: wp.array4d(dtype=wp.uint32),
    depth: wp.array4d(dtype=wp.float32),
    shape_index: wp.array4d(dtype=wp.uint32),
):
    """A deformable shell (its own env's group of the grouped mesh): the first hit, or with ``cull_back`` the first
    front-facing one (the far wall of an inward-facing vessel seen from outside), drawn where it is nearer than what
    is already there; ``shape_index`` is ``TiledCamera.TRIANGLES``."""
    env, y, x = wp.tid()
    xf = camera_transforms[0, env]
    origin = wp.transform_point(xf, rays[0, y, x, 0])
    direction = wp.normalize(wp.transform_vector(xf, rays[0, y, x, 1]))
    root = wp.mesh_get_group_root(mesh, env)
    start = float(0.0)
    found = float(-1.0)
    n = wp.vec3(0.0, 0.0, 1.0)
    for _ in range(6):
        query = wp.mesh_query_ray(mesh, origin + direction * start, direction, 1.0e6, root)
        if not query.result:
            break
        t = start + query.t
        if cull_back == 0 or wp.dot(query.normal, direction) < 0.0:
            found = t
            n = query.normal
            break
        start = t + 1.0e-4
    if found < 0.0:
        return
    current = depth[env, 0, y, x]
    if current > 0.0 and current <= found:
        return
    color[env, 0, y, x] = pack_rgba(wall_color * headlight(n, direction))
    depth[env, 0, y, x] = found
    shape_index[env, 0, y, x] = wp.uint32(0xFFFFFFFD)


# omniendo/isaaclab/mdp.py:113-159 (catheter_tip_camera_pose) as a device function: the up hint made perpendicular to
# forward, falling back to the frame's other axis, the world up, then world y; returns Newton's camera quaternion
# (columns right, up, -forward: the sensor looks down -Z with +Y up, as the free pose above).
@wp.func
def stable_frame_q(forward: wp.vec3f, up_hint: wp.vec3f, alt_hint: wp.vec3f, world_up: wp.vec3f) -> wp.quatf:
    f = wp.normalize(forward)
    up = up_hint - f * wp.dot(up_hint, f)
    if wp.length(up) < 1.0e-5:
        up = alt_hint - f * wp.dot(alt_hint, f)
    if wp.length(up) < 1.0e-5:
        up = world_up - f * wp.dot(world_up, f)
    if wp.length(up) < 1.0e-5:
        y = wp.vec3f(0.0, 1.0, 0.0)
        up = y - f * wp.dot(y, f)
    if wp.length(up) < 1.0e-5:
        up = wp.cross(f, wp.vec3f(1.0, 0.0, 0.0))
    up = wp.normalize(up)
    left = wp.normalize(wp.cross(up, f))
    up = wp.normalize(wp.cross(f, left))
    return wp.quat_from_matrix(wp.matrix_from_cols(-left, up, -f))


@wp.kernel
def tip_pose_rod(
    q: wp.array2d(dtype=wp.vec3),
    frame: wp.array(dtype=wp.quat),
    P: int,
    F: int,
    distance: float,
    world_up: wp.vec3f,
    camera_transforms: wp.array2d(dtype=wp.transformf),
):
    env = wp.tid()
    d = q[env, P - 1] - q[env, P - 2]
    if wp.length(d) <= 1.0e-8:  # a degenerate tangent keeps the previous pose (omniendo xcath.py:1327)
        return
    f = wp.normalize(d)
    fq = frame[env * F + wp.min(P - 1, F - 1)]
    rot = stable_frame_q(f, wp.quat_rotate(fq, wp.vec3f(0.0, 1.0, 0.0)), wp.quat_rotate(fq, wp.vec3f(1.0, 0.0, 0.0)), world_up)
    camera_transforms[0, env] = wp.transformf(q[env, P - 1] - f * distance, rot)


@wp.kernel
def tip_pose_instrument(
    root: wp.array2d(dtype=wp.vec3),
    shaft_q: wp.array2d(dtype=wp.quat),
    k: int,
    distance: float,
    world_up: wp.vec3f,
    camera_transforms: wp.array2d(dtype=wp.transformf),
):
    env = wp.tid()
    sq = shaft_q[env, k]
    f = wp.quat_rotate(sq, wp.vec3f(0.0, 0.0, -1.0))  # into the body: the shaft runs from the trocar to the root
    rot = stable_frame_q(f, wp.quat_rotate(sq, wp.vec3f(0.0, 1.0, 0.0)), wp.quat_rotate(sq, wp.vec3f(1.0, 0.0, 0.0)), world_up)
    camera_transforms[0, env] = wp.transformf(root[env, k] - f * distance, rot)
