# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lean Slang renderer: textured tissue and static meshes, zero-copy from Warp.

Ported from omnisurg/rendering/slang.py: device, window and surface setup (1923-2167),
pipelines (2502-3052), textures and environment (4562-4731), common uniforms and lights
(4982-5070), tissue material binding (5995-6074), draw (5883-5951), frame loop
(6726-6936). Fluids: ``render/fluid.py`` (four screen-space passes into a second HDR target that the
present pass and ``read_frame`` then take); smoke: ``render/smoke.py`` (billboards in a final pass over it).
omnisurg's snapshot ring is ``render/snapshots.py``; the post chain is ``render/post.py`` (every pass every frame into
the display texture ``ldr``, which ``read_frame`` returns); dropped: RT.

Data flow per frame: ``publish(snapshot)`` copies the drawn env's positions, live triangle list
and visual state (damage, coagulation, blood) from a ``Snapshot`` (the live scene by default, a
ring slot in the threaded app) into shared buffers, skins the instrument pieces and runs the
normal and tangent kernels on the renderer's stream (``stream``; None: Warp's current stream);
``draw()`` encodes one render pass into an HDR target (tissue pipeline for tissue and cavity, pbr
pipeline with flat colours for instruments, rod tubes and the optional collision-sphere overlay),
hands it to the post chain (``render/post.py``), and submits on that stream so graphics runs after
the copies and the next copies wait for graphics. ``lock`` (the snapshot ring's) guards
``set_env``, which reallocates draws. A static mesh's ``visual`` may give a
flat ``color`` instead of textures and ``cull: back`` to hide its outside (an inward-facing vessel
seen from outside; omniendo ``hide_external_faces``).
"""

from __future__ import annotations

import atexit
import contextlib
import math
import time
from pathlib import Path

import numpy as np
import warp as wp
from i4h_isaaclab.appearance import appearance
from i4h_isaaclab.camera.camera import Camera, key_event
from i4h_isaaclab.camera.rig import TipCamera
from i4h_physics.config import require_asset
from i4h_standalone.render import (
    CUT_SPHERE_COLOR,
    MARKER_CANDIDATE_COLOR,
    MARKER_IDLE_COLOR,
    SPHERE_COLOR,
    draw_sources,
)
from i4h_standalone.render.capture import FrameCapture
from i4h_standalone.render.clips import CLIP_COLOR, ClipDraw
from i4h_standalone.render.fluid import DEFAULTS as FLUID_DEFAULTS
from i4h_standalone.render.fluid import FluidDraw, FluidPasses
from i4h_standalone.render.gbuffer import GBuffer
from i4h_standalone.render.instruments import (
    piece_skins,
    place_sphere_mesh,
    skin_pieces,
    sphere_mesh_indices,
    uv_sphere,
)
from i4h_standalone.render.interop import GraphicsInterop, SharedBuffer, cuda_driver
from i4h_standalone.render.kernels import Scratch, update_normals_tangents
from i4h_standalone.render.pn import DEFAULTS as PN_DEFAULTS
from i4h_standalone.render.pn import PnDraw
from i4h_standalone.render.post import TONEMAPS, Post, PostPasses
from i4h_standalone.render.rods import RodDraw
from i4h_standalone.render.skin import Skinner
from i4h_standalone.render.smoke import SmokeDraw
from i4h_standalone.render.snapshots import Snapshot, copy
from i4h_surgical.loaders.instrument_usd import CHAIN_CUT
from i4h_surgical.systems.fluids.smoke import Smoke
from i4h_surgical.systems.instruments.clipping import Clipping

SHADER_DIR = Path(__file__).resolve().parent / "shaders"
VIEWS = 3  # render views per frame: the window (or offscreen target) and the two XR eyes

# omnisurg/rendering/slang.py:191-215 TissueMaterialParams defaults.
TISSUE_PARAMS = dict(
    normal_strength=0.65, specular_scale=0.35, roughness_bias=0.45, ambient=0.22, rim_strength=0.12,
    coat_weight=0.28, coat_ior=1.33, color_gain=1.18, blend_damage=1.0, blend_coag=1.0, blend_blood=1.0,
    wetness=0.0, wet_spec_scale=1.0, wet_roughness=0.18, subsurface_strength=0.0, blood_wetness=1.0,
    specular_aa_strength=0.35, specular_aa_min_roughness=0.04,
)
SUBSURFACE_COLOR = (0.8, 0.22, 0.16)
# omnisurg/rendering/slang.py:216-223 LightingParams and the chole environment YAML.
HEADLIGHT = dict(color=(1.0, 0.92, 0.78), intensity=0.59, range=3.0, spot_degrees=78.0, softness=0.35,
                 directional_fill_scale=0.25)
# Directional lights (camera space) are disabled in the chole case; directions kept for the uniforms.
DIRECTIONAL = {"key": (-0.20, 0.42, -1.0), "fill": (0.55, -0.15, -1.0), "back": (0.0, 0.48, 1.0)}  # slang.py:4982-5070 (disabled)
AMBIENT = (0.22, 0.20, 0.18)
LAYERS = ("base", "damage", "coag", "blood")
LAYER_KEYS = frozenset(k if layer == "base" else f"{k}_{layer}" for k in ("diffuse", "normal", "specular") for layer in LAYERS)
VISUAL_KEYS = LAYER_KEYS | {"color", "cull", "set", "render_radius"}  # render_radius: fluid pools (read by the fluid draw)
MASKS = ("layer_mask_damage_tex", "layer_mask_coag_tex", "layer_mask_blood_tex", "blood_mask_tex", "heat_mask_tex")
ENV_KEYS = {"page_up": 1, "right_bracket": 1, "page_down": -1, "left_bracket": -1}
INSTRUMENT_MATERIAL = dict(material_roughness=0.35, material_metallic=0.8, specular_scale=0.35, ambient=1.0,
                           rim_strength=0.12)  # slang.py _set_common_uniforms values for flat meshes


def read_rgba(path) -> np.ndarray:
    """``(H, W, 4)`` uint8 with the source channel values untouched (no gamma conversion: normal and
    specular maps are data, diffuse is decoded by its ``rgba8_unorm_srgb`` GPU format), rows flipped
    so UV v=0 is the bottom row (slang.py:4641-4649 used ``ImageOps.flip``)."""
    import slangpy as spy

    bitmap = spy.Bitmap(str(require_asset(path))).convert(pixel_format=spy.Bitmap.PixelFormat.rgba,
                                           component_type=spy.Bitmap.ComponentType.uint8)
    return np.ascontiguousarray(np.asarray(bitmap)[::-1])


PRESENT_KEYS = ("source_tex", "post_sampler", "output_size", "frame_index", "sensor_noise_enabled", "sensor_noise_strength",
                "sensor_noise_shadow_boost", "manual_srgb_encode")  # present.slang; checked by test_render


@wp.kernel
def default_surface_colors(colors: wp.array(dtype=wp.vec4f)):
    colors[wp.tid()] = wp.vec4f(0.0)


class BodyDraw:
    """Render resources of one tet or shell body: shared (Warp-writable) vertex streams and a static uv buffer.
    With a ``surface`` system the index and colour buffers are shared too and refilled every frame
    from its live triangles and visual state (cut faces appear, removed ones go degenerate, burns
    and damage tint); without one (a shell vessel, a tet body without cutting) they are the variant's
    authored triangles and zero colours. A part's ``visual`` may give a flat ``color`` and ``cull: back``
    like a static mesh."""

    def __init__(self, renderer: "SlangRenderer", body, surface=None, pn=None):
        r, spy, dev = renderer, renderer.spy, renderer.scene.device
        m = body.tet if body.tet is not None else body.shell
        count = body.count
        self.pn = None
        v = int(body.variant_of_env[r.env])
        self.body = body
        self.scene = r.scene
        self.surface = surface
        self.variant = v
        self.pos = r.shared(count, wp.vec3f, 12, f"{body.name}-positions")
        self.normals = r.shared(count, wp.vec3f, 12, f"{body.name}-normals")
        self.tangents = r.shared(count, wp.vec4f, 16, f"{body.name}-tangents")
        uvs = np.ascontiguousarray(m.uvs[v], dtype=np.float32)
        self.uv_buffer = r.device.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=uvs, label=f"{body.name}-uvs")
        self.uv_array = wp.array(uvs, dtype=wp.vec2f, device=dev)
        self.indices = self.colors = None
        if surface is not None:
            F = surface.tris.shape[1]
            self.indices = r.shared(F, wp.vec3i, 12, f"{body.name}-indices", usage=spy.BufferUsage.index_buffer)
            self.index_buffer = self.indices.buffer
            self.index_array = wp.array(ptr=self.indices.array.ptr, dtype=wp.int32, shape=(3 * F,), device=dev, copy=False)
            self.colors = r.shared(count, wp.vec4f, 16, f"{body.name}-colors")
            color_buffer = self.colors.buffer
        else:
            indices = np.ascontiguousarray(m.tris[v].reshape(-1), dtype=np.int32)
            self.index_buffer = r.device.create_buffer(usage=spy.BufferUsage.index_buffer,
                                                       data=indices.astype(np.uint32), label=f"{body.name}-indices")
            self.index_array = wp.array(indices, dtype=wp.int32, device=dev)
            color_buffer = r.zero_color
        self.scratch = Scratch(count, dev)
        self.parts = [(3 * p.tri_start, 3 * p.tri_count, r.material(appearance(p)), part_color(appearance(p)), part_cull(appearance(p), body.name),
                       r.label(f"{body.name}/{name}")) for name, p in body.parts.items()]  # ..., G-buffer label
        self.streams = [self.pos.buffer, self.normals.buffer, self.tangents.buffer, self.uv_buffer, color_buffer]
        self.color_array = self.colors.array if self.colors is not None else wp.zeros(count, dtype=wp.vec4f, device=dev)
        if pn is not None:  # the drawn geometry is the curved lattice; the base streams feed it
            F = self.index_array.shape[0] // 3
            self.pn = PnDraw(r, body.name, F, pn["segments"], pn["curvature"])
            S = self.pn.S
            self.parts = [(S * start, S * count_, textures, color, cull, label) for start, count_, textures, color, cull, label in self.parts]
            self.streams, self.index_buffer = self.pn.streams, self.pn.index_buffer

    def publish(self, snap, env: int, device, stream=None):
        """The drawn env's positions (and with a surface its triangles and visual state) from ``snap`` into the
        shared buffers, then the normal and tangent kernels."""
        scene = self.scene
        copy(self.pos.array, scene.view(snap.q, self.body)[env], stream)  # (count,)
        if self.surface is not None:
            copy(self.indices.array, snap.tris[self.surface][env], stream)  # (F,) vec3i
            if self.surface in snap.visual:
                copy(self.colors.array, snap.visual[self.surface].reshape((scene.envs, self.body.count))[env], stream)
            else:
                wp.launch(default_surface_colors, dim=self.body.count, outputs=[self.colors.array], device=device, stream=stream)
        update_normals_tangents(self.pos.array, self.index_array, self.uv_array, self.scratch, self.normals.array,
                                self.tangents.array, device, stream)
        if self.pn is not None:
            self.pn.expand(self.pos.array, self.normals.array, self.tangents.array, self.uv_array, self.color_array, self.index_array,
                           device, stream, static=self.surface is None)

    def close(self):
        for shared in (self.pos, self.normals, self.tangents, self.indices, self.colors):
            if shared is not None:
                shared.close()
        if self.pn is not None:
            self.pn.close()


def part_color(visual) -> tuple:
    return tuple(float(c) for c in (visual or {}).get("color", (1.0, 1.0, 1.0)))


def part_cull(visual, name: str) -> str:
    cull = str((visual or {}).get("cull", "none"))
    if cull not in ("none", "back"):
        raise ValueError(f"{name!r}: visual.cull must be none or back")
    return cull


class FlatDraw:
    """A flat-coloured mesh whose positions and normals a kernel writes every frame (skinned instrument pieces,
    rod tubes, the collision-sphere overlay): shared position and normal streams, static indices."""

    def __init__(self, renderer: "SlangRenderer", name: str, count: int, indices: np.ndarray, color):
        r, spy = renderer, renderer.spy
        self.color = tuple(color)
        self.pos = r.shared(count, wp.vec3f, 12, f"{name}-positions")
        self.normals = r.shared(count, wp.vec3f, 12, f"{name}-normals")
        indices = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1)
        self.index_buffer = r.device.create_buffer(usage=spy.BufferUsage.index_buffer, data=indices.astype(np.uint32), label=f"{name}-indices")
        self.index_count = int(indices.size)
        self.streams = [self.pos.buffer, self.normals.buffer]

    def close(self):
        self.pos.close()
        self.normals.close()


class SkinDraw:
    """Render resources of a skinned vessel: the tree body's surface placed by dual-quaternion skinning every frame
    (``render/skin.py``), drawn like a shell body without a surface: shared position, normal and tangent streams,
    static zero uvs (fallback tangents) and indices, zero colours, the ``visual``'s flat colour and culling."""

    def __init__(self, renderer: "SlangRenderer", body, rod):
        r, spy, dev = renderer, renderer.spy, renderer.scene.device
        skin = body.skin
        n = len(skin.points)
        self.body = body
        self.skinner = Skinner(body, rod, dev)
        self.pos = r.shared(n, wp.vec3f, 12, f"{body.name}-skin-positions")
        self.normals = r.shared(n, wp.vec3f, 12, f"{body.name}-skin-normals")
        self.tangents = r.shared(n, wp.vec4f, 16, f"{body.name}-skin-tangents")
        uvs = np.zeros((n, 2), np.float32)
        self.uv_buffer = r.device.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=uvs, label=f"{body.name}-skin-uvs")
        self.uv_array = wp.array(uvs, dtype=wp.vec2f, device=dev)
        indices = np.ascontiguousarray(skin.indices.reshape(-1), dtype=np.int32)
        self.index_buffer = r.device.create_buffer(usage=spy.BufferUsage.index_buffer, data=indices.astype(np.uint32),
                                                   label=f"{body.name}-skin-indices")
        self.index_array = wp.array(indices, dtype=wp.int32, device=dev)
        self.index_count = int(indices.size)
        self.scratch = Scratch(n, dev)
        self.textures = r.material(skin.visual)
        self.color, self.cull = part_color(skin.visual), part_cull(skin.visual, body.name)
        self.streams = [self.pos.buffer, self.normals.buffer, self.tangents.buffer, self.uv_buffer, r.zero_color]

    def publish(self, snap, env: int, device, stream=None):
        self.skinner.place(snap, env, self.pos.array, device, stream)
        update_normals_tangents(self.pos.array, self.index_array, self.uv_array, self.scratch, self.normals.array,
                                self.tangents.array, device, stream)

    def close(self):
        for shared in (self.pos, self.normals, self.tangents):
            shared.close()


class Markers:
    """Small spheres the app places per frame (the idle input devices of the pickup): pinned host arrays the physics thread
    writes, copied on the render stream in ``publish``, one ``FlatDraw`` per colour (idle, candidate) through the sphere
    overlay kernel. Written and read without a lock: a torn frame moves a marker by one frame's motion."""

    def __init__(self, renderer: "SlangRenderer", count: int, idle_color=MARKER_IDLE_COLOR):
        dev = renderer.scene.device
        self.count = count
        self.q = wp.zeros((1, count), dtype=wp.vec3, device="cpu", pinned=True)
        self.radius = wp.zeros((2, count), dtype=wp.float32, device="cpu", pinned=True)  # row 0 idle, row 1 candidate
        self.q_host, self.radius_host = self.q.numpy(), self.radius.numpy()  # views of the pinned memory, made once
        self.q_dev = wp.zeros((1, count), dtype=wp.vec3, device=dev)
        self.radius_dev = wp.zeros((2, count), dtype=wp.float32, device=dev)
        self.ones = wp.ones((1, count), dtype=wp.int32, device=dev)
        self.show = wp.ones(count, dtype=wp.int32, device=dev)
        points, faces = uv_sphere()
        self.template = wp.array(points, dtype=wp.vec3, device=dev)
        indices = sphere_mesh_indices(count, faces, len(points))
        self.draws = [FlatDraw(renderer, "markers-idle", count * len(points), indices, idle_color),
                      FlatDraw(renderer, "markers-candidate", count * len(points), indices, MARKER_CANDIDATE_COLOR)]

    def set(self, i: int, position, radius: float, candidate: bool) -> None:
        """Physics thread: marker ``i`` at ``position`` with ``radius`` (0 hides it) in the idle or the candidate colour."""
        q, r = self.q_host, self.radius_host
        q[0, i] = position
        r[0, i] = 0.0 if candidate else radius
        r[1, i] = radius if candidate else 0.0

    def publish(self, device, stream, task_rows=None, env=0, progress=None) -> None:
        if task_rows is None:
            wp.copy(self.q_dev, self.q, stream=stream)
            wp.copy(self.radius_dev, self.radius, stream=stream)
        else:
            from i4h_standalone.render.procedure import place_task_markers
            wp.launch(place_task_markers, self.count, inputs=[task_rows, env, progress],
                      outputs=[self.q_dev, self.radius_dev], device=device, stream=stream)
        for j, d in enumerate(self.draws):
            wp.launch(place_sphere_mesh, dim=d.pos.array.shape[0],
                      inputs=[self.template, self.q_dev, self.radius_dev[j], self.ones, self.show, 0],
                      outputs=[d.pos.array, d.normals.array], device=device, stream=stream)

    def close(self):
        for d in self.draws:
            d.close()


class StaticDraw:
    """Render resources of a static mesh: everything uploaded once."""

    def __init__(self, renderer: "SlangRenderer", mesh):
        spy, dev = renderer.spy, renderer.device
        n = len(mesh.points)
        def make(data, label):
            return dev.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=np.ascontiguousarray(data, dtype=np.float32), label=f"{mesh.name}-{label}")
        tangents = mesh.tangents if mesh.tangents is not None else np.tile(np.float32([1.0, 0.0, 0.0, 1.0]), (n, 1))
        uvs = mesh.uvs if mesh.uvs is not None else np.zeros((n, 2), np.float32)
        self.streams = [make(mesh.points, "positions"), make(mesh.normals, "normals"), make(tangents, "tangents"),
                        make(uvs, "uvs"), make(np.zeros((n, 4), np.float32), "colors")]
        indices = np.ascontiguousarray(mesh.indices.reshape(-1), dtype=np.uint32)
        self.index_buffer = dev.create_buffer(usage=spy.BufferUsage.index_buffer, data=indices, label=f"{mesh.name}-indices")
        self.index_count = int(indices.size)
        self.textures = renderer.material(appearance(mesh))
        self.color = tuple(float(c) for c in appearance(mesh).get("color", (1.0, 1.0, 1.0)))
        self.cull = str(appearance(mesh).get("cull", "none"))
        if self.cull not in ("none", "back"):
            raise ValueError(f"static mesh {mesh.name!r}: visual.cull must be none or back")

    def close(self):
        self.streams, self.index_buffer, self.textures = [], None, {}


class SlangRenderer:
    def __init__(self, sim, *, size=(1280, 720), env=0, vsync=False, window=True, exposure=1.0, tonemap="none",
                 camera=None, environment=None, headlight=None, clear=(0.025, 0.030, 0.034), debug_spheres=False, fluid=None,
                 ui=True, post=None, pn=None, gbuffer=False, camera_mode="free", tip=None, device=None, xr=None, procedure=None):
        import slangpy as spy

        del xr  # the case's render.xr block belongs to the xr backend (render/xr.py); the window ignores it
        self.spy = spy
        self.scene = scene = sim.scene
        if not scene.device.is_cuda:
            raise RuntimeError("the Slang renderer needs a CUDA scene device (shared-buffer interop)")
        self.env, surfaces, rod_systems, self.control = draw_sources(sim, env)
        self.live = Snapshot(sim, procedure=procedure)  # what publish() reads without a ring
        self.snap = None  # the last snapshot published (set_env republishes it)
        self.stream = None  # the renderer's CUDA stream; None: Warp's current stream (single-threaded callers)
        self.lock = contextlib.nullcontext()  # the snapshot ring's lock in the threaded app
        self.status = ""
        self.width, self.height = int(size[0]), int(size[1])
        self.vsync = bool(vsync)
        self.exposure = float(exposure)
        if str(tonemap) not in TONEMAPS:
            raise ValueError(f"render.tonemap must be one of {TONEMAPS}, got {tonemap!r}")
        self.tonemap = TONEMAPS.index(str(tonemap))
        try:
            self.post = Post(**dict(post or {}))
        except TypeError as exc:
            raise ValueError(f"render.post: {exc}") from None
        self.labels: dict[str, int] = {}  # G-buffer object labels in draw order; 0 is the background (label())
        self.pn = None if pn is None else {**PN_DEFAULTS, **dict(pn)}  # PN subdivision of the tet and shell surfaces
        if self.pn is not None and set(self.pn) - set(PN_DEFAULTS):
            raise ValueError(f"render.pn: unknown keys {sorted(set(self.pn) - set(PN_DEFAULTS))}; known {sorted(PN_DEFAULTS)}")
        self.frame_index = 0  # draws so far: sensor noise seed, auto exposure warm-up
        self.frame_dt = None  # tests: a fixed auto-exposure dt instead of the wall clock
        self._last_draw = None
        self.clear = [float(c) for c in clear] + [1.0]
        self.camera = Camera(**dict(camera or {}))
        if camera_mode not in ("free", "tip"):
            raise ValueError(f"render.camera_mode must be free or tip, got {camera_mode!r}")
        self.rig = TipCamera(sim, tip) if (tip is not None or camera_mode == "tip") else None  # render/rig.py
        self.camera_mode = "free"
        self.free_pose = self.camera.pose()
        env_cfg = dict(environment or {})
        env_path = env_cfg.pop("path", None)
        self.env_intensity = float(env_cfg.pop("intensity", 1.0))
        self.env_lighting = bool(env_cfg.pop("lighting", True))
        self.env_rotation = float(env_cfg.pop("rotation_degrees", 0.0))
        self.env_pitch = float(env_cfg.pop("pitch_degrees", -90.0))
        if env_cfg:
            raise ValueError(f"render.environment: unknown keys {sorted(env_cfg)}")
        self.headlight = dict(HEADLIGHT)
        unknown = set(headlight or {}) - set(HEADLIGHT)
        if unknown:
            raise ValueError(f"render.headlight: unknown keys {sorted(unknown)}")
        self.headlight.update(headlight or {})
        fluid = dict(fluid or {})
        unknown = set(fluid) - set(FLUID_DEFAULTS)
        if unknown:
            raise ValueError(f"render.fluid: unknown keys {sorted(unknown)}; known {sorted(FLUID_DEFAULTS)}")

        # Device sharing Warp's CUDA context (slang.py:1953-1962), or the caller's (an XR session's, made with the Vulkan
        # extensions OpenXR needs on the GPU it chose; offscreen, and not closed here: the caller owns it).
        self.cuda = cuda_driver()
        self.cuda.set_current_context(int(scene.device.context))
        self.owns_device = device is None
        if device is not None and window:
            raise ValueError("an injected device renders offscreen (window=False); its owner presents")
        self.device = device if device is not None else spy.create_device(
            spy.DeviceType.vulkan,
            include_paths=[str(SHADER_DIR)],
            enable_cuda_interop=True,
            existing_device_handles=spy.get_cuda_current_context_native_handles(),
        )
        if not self.device.supports_cuda_interop:
            raise RuntimeError("the Vulkan device does not support CUDA interop")
        self.interop = GraphicsInterop(self.device, scene.device)
        self.ui = None  # slangpy's retained-widget ImGui context; the app builds its panels on ``ui.screen`` (app/ui.py)
        if ui:
            import slangpy.ui  # noqa: F401  (registers spy.ui)

            self.ui = spy.ui.Context(self.device)
        self._mouse_owner = None  # "ui" or "scene" while a button is held: a drag stays where it started

        # Window and surface, or offscreen (fixed for the renderer's life).
        self.window = self.surface = None
        if window:
            self.window = spy.Window(width=self.width, height=self.height, title="i4h_physics", resizable=True)
            self.window.on_resize = self._on_resize
            self.window.on_keyboard_event = self._on_key
            self.window.on_mouse_event = self._on_mouse
            self.surface = self.device.create_surface(self.window)
            self.surface.configure(width=self.window.width, height=self.window.height, vsync=self.vsync)
            self.color_format = self.surface.info.preferred_format
        else:
            self.color_format = spy.Format.rgba8_unorm
        self.manual_srgb = "srgb" not in str(self.color_format).lower()
        self.scene_format = spy.Format.rgba16_float
        self.depth_format = spy.Format.d32_float

        # Programs, layouts, pipelines (slang.py:2502-2860, trimmed to tissue and present).
        tissue = self.device.load_program("tissue.slang", ["vertex_main", "fragment_main"])
        present = self.device.load_program("present.slang", ["vertex_main", "fragment_main"])
        def element(name, fmt, slot):
            return {"semantic_name": name, "semantic_index": 0, "format": fmt, "buffer_slot_index": slot}
        F = spy.Format
        tissue_layout = self.device.create_input_layout(
            input_elements=[element("POSITION", F.rgb32_float, 0), element("NORMAL", F.rgb32_float, 1),
                            element("TANGENT", F.rgba32_float, 2), element("TEXCOORD", F.rg32_float, 3),
                            element("COLOR", F.rgba32_float, 4)],
            vertex_streams=[{"stride": 12}, {"stride": 12}, {"stride": 16}, {"stride": 8}, {"stride": 16}],
        )
        opaque = dict(primitive_topology=spy.PrimitiveTopology.triangle_list, targets=[{"format": self.scene_format}],
                      depth_stencil={"format": self.depth_format, "depth_test_enable": True, "depth_write_enable": True,
                                     "depth_func": spy.ComparisonFunc.less})
        self.tissue_pipeline = self.device.create_render_pipeline(
            program=tissue, input_layout=tissue_layout, rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-tissue", **opaque)
        self.tissue_pipeline_cull = self.device.create_render_pipeline(
            program=tissue, input_layout=tissue_layout, rasterizer={"cull_mode": spy.CullMode.back}, label="i4h-tissue-cull", **opaque)
        pbr = self.device.load_program("mesh.slang", ["textured_vertex_main", "pbr_fragment_main"])
        pbr_layout = self.device.create_input_layout(
            input_elements=[element("POSITION", F.rgb32_float, 0), element("NORMAL", F.rgb32_float, 1)],
            vertex_streams=[{"stride": 12}, {"stride": 12}],
        )
        self.pbr_pipeline = self.device.create_render_pipeline(
            program=pbr, input_layout=pbr_layout, rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-pbr", **opaque)
        self.task_pipeline = None
        if procedure is not None or any(b.regions is not None for b in scene.bodies):
            overlay = {**opaque, "depth_stencil": {"format": self.depth_format,
                       "depth_test_enable": False, "depth_write_enable": False}}
            self.task_pipeline = self.device.create_render_pipeline(
                program=pbr, input_layout=pbr_layout, rasterizer={"cull_mode": spy.CullMode.none},
                label="i4h-task-marker", **overlay)
        # root shader objects: the frame block is written once per frame and view, a draw writes its colour and textures;
        # per view (0 the window or offscreen target, 1 and 2 the XR eyes) two, alternated by frame, so a frame's writes
        # never touch the object the in-flight frame still reads and two views of one frame keep their own cameras
        self.roots = {id(p): [self.device.create_root_shader_object(prog) for _ in range(2 * VIEWS)]
                      for p, prog in ((self.tissue_pipeline, tissue), (self.tissue_pipeline_cull, tissue),
                                     (self.pbr_pipeline, pbr), (self.task_pipeline, pbr)) if p is not None}
        self._slot = 0  # the root slot encode_scene binds (view * 2 + frame parity)
        self.present_pipeline = self.device.create_render_pipeline(
            program=present, input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[{"format": self.color_format}], rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-present",
        )
        self.ldr_format = spy.Format.rgba8_unorm  # the display texture: what read_frame() and the recorder take
        self.ldr_pipeline = self.device.create_render_pipeline(
            program=present, input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[{"format": self.ldr_format}], rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-present-ldr",
        )
        self.sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear, mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear, address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge, address_w=spy.TextureAddressingMode.clamp_to_edge,
            label="i4h-linear-clamp",
        )
        self._textures: dict = {}
        self.environment = self._load_environment(env_path)
        max_count = max([b.count for b in scene.bodies if b.tet is not None or b.shell is not None]
                        + [len(b.skin.points) for b in scene.bodies if getattr(b, "skin", None) is not None] + [1])
        self.zero_color = self.device.create_buffer(usage=spy.BufferUsage.vertex_buffer,
                                                    data=np.zeros((max_count, 4), np.float32), label="i4h-zero-colors")
        self.bodies = [BodyDraw(self, b, surfaces.get(id(b)), self.pn) for b in scene.bodies if b.tet is not None or b.shell is not None]
        self.statics = [StaticDraw(self, m) for m in scene.static_meshes]
        for m, s in zip(scene.static_meshes, self.statics):
            s.label = self.label(m.name)
        self.skinned = [SkinDraw(self, b, rod_systems[id(b)]) for b in scene.bodies if getattr(b, "skin", None) is not None]
        for d in self.skinned:
            d.label = self.label(f"{d.body.name}/skin")
        self.skins = piece_skins(self.control, scene.device) if self.control else []
        self.pieces = [FlatDraw(self, skin.name, skin.count, skin.indices, skin.color) for skin in self.skins]
        for skin, d in zip(self.skins, self.pieces):
            d.label = self.label(skin.name)
        self.rods = []  # (RodDraw, FlatDraw) per rod body with a rod system; the tube kernel writes the normals
        for b in scene.bodies:
            if b.rod is not None and id(b) in rod_systems:
                rd = RodDraw(b, rod_systems[id(b)])
                self.rods.append((rd, FlatDraw(self, f"rod-{b.name}", rd.count, rd.indices, rd.color)))
                self.rods[-1][1].label = self.label(f"rod/{b.name}")
        self.fluids = [FluidDraw(self, b) for b in scene.pools if b.fluid is not None]
        for d in self.fluids:
            d.label = self.label(f"{d.body.name}/fluid")
        self.fluid_passes = FluidPasses(self, **fluid) if self.fluids else None  # fixed for the surface's life
        self.post_passes = PostPasses(self, self.post)  # every pass every frame; toggles are uniforms
        self.ldr = self.ldr_view = self.display = None  # the display texture (post + present, sRGB) and the post chain's output
        self._surface_views: dict = {}  # id(swapchain texture) -> (texture, view)
        self._ldr_frame = -1  # the frame index encoded into ldr (read_frame presents on demand when a windowed frame skipped it)
        self.smokes = [SmokeDraw(self, s) for s in sim.systems if isinstance(s, Smoke)]
        self.clips = []  # (ClipDraw, FlatDraw) per clipping system: a torus per clip slot, placed from the snapshot
        for s in sim.systems:
            if isinstance(s, Clipping):
                cd = ClipDraw(s, scene.device)
                self.clips.append((cd, FlatDraw(self, cd.name, cd.count, cd.indices, CLIP_COLOR)))
                self.clips[-1][1].label = self.label(cd.name)
        self.gbuffer = GBuffer(self) if gbuffer else None  # normals + labels beside the RGB frame (dataset capture)
        self.capture = None  # enable_capture(): the dataset capture's GPU side (render/capture.py)
        self.overlay = self.overlay_cut = None  # collision spheres, and a scissors' cutting spheres in their own colour
        if self.control is not None and self.control.sphere_count:
            points, faces = uv_sphere()
            S = self.control.sphere_count
            self.sphere_template = wp.array(points, dtype=wp.vec3, device=scene.device)
            cut = self.control.host_sphere_chain == CHAIN_CUT
            self.show_regular = wp.array((~cut).astype(np.int32), dtype=wp.int32, device=scene.device)
            self.overlay = FlatDraw(self, "spheres", S * len(points), sphere_mesh_indices(S, faces, len(points)), SPHERE_COLOR)
            if cut.any():
                self.show_cut = wp.array(cut.astype(np.int32), dtype=wp.int32, device=scene.device)
                self.overlay_cut = FlatDraw(self, "cut-spheres", S * len(points), sphere_mesh_indices(S, faces, len(points)), CUT_SPHERE_COLOR)
        self.debug_spheres = bool(debug_spheres)
        from i4h_standalone.render.regions import RegionHighlight
        self.region_highlights = [RegionHighlight(self, b) for b in scene.bodies if b.regions is not None]
        self.task_markers = Markers(self, 73, idle_color=(1.0, 0.6, 0.05)) if procedure is not None else None
        self.task_procedure = procedure
        self.task_row = None
        self.markers = None  # the app's idle-device proxies (add_markers), drawn every frame
        self.tissue_params = {**{k: float(v) for k, v in TISSUE_PARAMS.items()}, "material_sampler": self.sampler, "debug_mode": 0,
                              "specular_aa_enabled": 1, "subsurface_color": spy.float3(*SUBSURFACE_COLOR)}  # slang.py:5995-6031
        def f3(v):
            return spy.float3(float(v[0]), float(v[1]), float(v[2]))
        black = spy.float3(0.0, 0.0, 0.0)
        self._lights = {**{f"{n}_light_enabled": 0 for n in DIRECTIONAL},  # disabled: the fluid still takes the key direction
                        "light_key_color": black, "light_fill_color": black, "light_rim_color": black, "light_ambient_color": f3(AMBIENT)}
        self._set_title()

        self.scene_color = self.depth = self.scene_view = self.depth_view = self.final = None
        self.pending_resize = None
        self.surface_failures = 0  # consecutive frames whose acquire or present failed (an out-of-date swapchain)
        self.closed = False
        self.key_handlers: list = []  # extra keyboard listeners (the app's instrument driver)
        if camera_mode == "tip":
            self.set_camera_mode("tip")
        self.mouse_handlers: list = []  # extra mouse listeners (the app's drag driver); one returning True consumes the event
        atexit.register(self.close)  # slangpy closes devices at exit; our objects must go first

    def enable_capture(self, slots: int, depth_max: float) -> FrameCapture:
        """Dataset capture at the current frame size (before the physics thread starts: this allocates)."""
        self.capture = FrameCapture(self, slots, self.width, self.height, depth_max)
        return self.capture

    def label(self, name: str) -> int:
        """The G-buffer label of a drawn object (allocated at construction, 1-based; 0 is the background)."""
        if name not in self.labels:
            self.labels[name] = len(self.labels) + 1
        return self.labels[name]

    def shared(self, count: int, dtype, stride: int, label: str, usage=None) -> SharedBuffer:
        """A vertex (default) or index buffer Warp writes directly (slangpy shared memory imported into the
        scene's CUDA context)."""
        usage = self.spy.BufferUsage.vertex_buffer if usage is None else usage
        return SharedBuffer(self.spy, self.device, self.cuda, self.scene.device, count, dtype, stride, usage, label)

    def set_env(self, env: int) -> None:
        """Switch the drawn env. Bodies whose mesh variant is the same keep their buffers and only
        change the source slice; the others are rebuilt."""
        env = int(env)
        if not 0 <= env < self.scene.envs:
            raise ValueError(f"render.env {env} outside [0, {self.scene.envs})")
        if env == self.env:
            return
        with self.lock:  # rebuilding allocates: never while the physics thread captures a graph
            rebuild = [d for d in self.bodies if int(d.body.variant_of_env[env]) != d.variant]
            if rebuild:  # the buffers being closed: after the last draw that reads them and the last publish that writes them
                self.device.wait()
                wp.synchronize_stream(self.stream or self.scene.device)
            self.env = env
            self.bodies = [self._rebuild(d) if d in rebuild else d for d in self.bodies]
            if rebuild and self.stream is not None:  # the rebuild's uploads and memsets ran on Warp's current stream:
                self.stream.wait_stream(wp.get_stream(self.scene.device))  # order the render stream's reads after them
            self._set_title()
            if self.snap is not None:
                self.publish(self.snap)  # fill the buffers from the new env before the next draw

    def _rebuild(self, d: BodyDraw) -> BodyDraw:
        d.close()
        return BodyDraw(self, d.body, d.surface, self.pn)

    def select_region(self, body, region):
        for h in self.region_highlights:
            h.selected = int(region) if h.body.name == body else -1

    def add_markers(self, count: int) -> Markers:
        """Spheres the app places per frame (before the physics worker starts: this allocates)."""
        self.markers = Markers(self, int(count))
        return self.markers

    def set_status(self, text: str) -> None:
        """Extra text in the window title (the app shows physics and render rates)."""
        self.status = str(text)
        self._set_title()

    def _set_title(self):
        if self.window is not None:
            status = f"   {self.status}" if self.status else ""
            keys = "PgUp/PgDn env, X spheres" + (", C tip camera" if self.rig is not None else "") + ", Esc quit"
            self.window.title = f"i4h_physics   env {self.env}/{self.scene.envs}{status}   {keys}"

    def set_camera_mode(self, mode: str) -> None:
        """``free`` (the fly camera) or ``tip`` (posed from the rig every frame; the free pose is saved and restored,
        omniendo xcath.py:1297-1305)."""
        if mode not in ("free", "tip"):
            raise ValueError(f"camera mode must be free or tip, got {mode!r}")
        if mode == "tip" and self.rig is None:
            raise ValueError("no tip camera: set render.tip (a chain rod body or an instrument)")
        if mode == self.camera_mode:
            return
        if mode == "tip":
            self.free_pose = self.camera.pose()
            self.camera.fov, self.camera.near = self.rig.settings["fov"], self.rig.settings["near"]
        else:
            self.camera.restore(self.free_pose)
        self.camera_mode = mode

    def _on_key(self, event):
        name, press, _, release = key_event(event)
        if self.ui is not None and self.ui.handle_keyboard_event(event):  # a focused widget owns the keyboard
            if release:  # held tool and camera keys must still clear
                self.camera.handle_key(event)
                for handler in self.key_handlers:
                    handler(event)
            return
        if press and name in ENV_KEYS:
            self.set_env((self.env + ENV_KEYS[name]) % self.scene.envs)
        elif press and name == "x":
            self.debug_spheres = not self.debug_spheres
        elif press and name == "c" and self.rig is not None:
            self.set_camera_mode("free" if self.camera_mode == "tip" else "tip")
        elif self.camera_mode == "free" or name == "escape":
            self.camera.handle_key(event)
        for handler in self.key_handlers:
            handler(event)

    def _on_mouse(self, event):
        wants = self.ui is not None and self.ui.handle_mouse_event(event)  # always fed: ImGui tracks hover state
        if event.is_button_down() and self._mouse_owner is None:
            self._mouse_owner = "ui" if wants else "scene"
        owner = self._mouse_owner or ("ui" if wants else "scene")
        if event.is_button_up():
            self._mouse_owner = None
        if owner == "ui":
            return
        for handler in self.mouse_handlers:
            if handler(event):
                return
        if self.camera_mode == "free":
            self.camera.handle_mouse(event)

    # textures ---------------------------------------------------------------

    def _texture(self, path, srgb: bool):
        """rgba8 texture from PNG/TGA; ``srgb`` selects the decoding GPU format (diffuse only)."""
        key = (str(path), srgb)
        if key not in self._textures:
            spy = self.spy
            data = read_rgba(require_asset(path))
            self._textures[key] = self.device.create_texture(
                format=spy.Format.rgba8_unorm_srgb if srgb else spy.Format.rgba8_unorm,
                width=int(data.shape[1]), height=int(data.shape[0]),
                usage=spy.TextureUsage.shader_resource, data=data, label=Path(path).name,
            )
        return self._textures[key]

    def _default(self, kind: str):
        pixel = {"normal": (128, 128, 255, 255), "spec": (128, 128, 128, 255), "black": (0, 0, 0, 255)}.get(kind, (255, 255, 255, 255))
        key = ("default", kind)
        if key not in self._textures:
            self._textures[key] = self.device.create_texture(
                format=self.spy.Format.rgba8_unorm, width=1, height=1, usage=self.spy.TextureUsage.shader_resource,
                data=np.asarray([[pixel]], dtype=np.uint8), label=f"i4h-default-{kind}",
            )
        return self._textures[key]

    def _load_environment(self, path):
        if not path:
            return None
        spy = self.spy
        bitmap = spy.Bitmap(str(require_asset(path))).convert(pixel_format=spy.Bitmap.PixelFormat.rgba,
                                               component_type=spy.Bitmap.ComponentType.float32)
        return self.device.create_texture(format=spy.Format.rgba32_float, width=bitmap.width, height=bitmap.height,
                                          usage=spy.TextureUsage.shader_resource, data=np.asarray(bitmap),
                                          label=Path(path).name)

    def material(self, visual) -> dict:
        """Tissue shader textures (slang.py:6035-6056, renderers.py:40-68 TISSUE_LAYER_TEXTURE_KEYS): the base maps
        ``diffuse/normal/specular`` and the damage, coag and blood sets (``diffuse_damage`` ... ``specular_blood``); a
        missing layer map falls back to the base map. ``set: DIR`` names the authored files
        ``{diffuse,normal,spec}-{base,damage,coag,blood}.png`` in one directory (explicit keys override). Masks were
        never authored: layer masks white, blood and heat masks black."""
        visual = dict(visual or {})
        unknown = set(visual) - VISUAL_KEYS
        if unknown:
            raise ValueError(f"visual: unknown keys {sorted(unknown)}; known {sorted(VISUAL_KEYS)}")
        paths = {}
        if "set" in visual:
            root = Path(visual["set"])
            for kind, prefix in (("diffuse", "diffuse"), ("normal", "normal"), ("specular", "spec")):
                for layer in LAYERS:
                    paths[kind if layer == "base" else f"{kind}_{layer}"] = root / f"{prefix}-{layer}.png"
        paths.update({k: v for k, v in visual.items() if k in LAYER_KEYS})
        defaults = {"diffuse": "white", "normal": "normal", "specular": "spec"}
        textures = {}
        for kind, slot in (("diffuse", "diffuse"), ("normal", "normal"), ("specular", "spec")):
            base = self._texture(paths[kind], kind == "diffuse") if kind in paths else self._default(defaults[kind])
            for layer in LAYERS:
                key = kind if layer == "base" else f"{kind}_{layer}"
                textures[f"{slot}_{layer}_tex"] = self._texture(paths[key], kind == "diffuse") if key in paths else base
        for mask in MASKS:
            textures[mask] = self._default("white" if mask.startswith("layer_mask") else "black")
        return textures

    # per frame ----------------------------------------------------------------

    def publish(self, snap: Snapshot | None = None) -> None:
        """CUDA side: refresh the shared vertex streams from ``snap`` (the live scene when None) on the renderer's
        stream. The previous draw's graphics-done fence orders these writes after it on the GPU; no host wait.
        Reads are bracketed by the snapshot's ready and release events."""
        dev, env, stream = self.scene.device, self.env, self.stream
        events_on = stream or wp.get_stream(dev)
        snap = self.live if snap is None else snap
        self.snap = snap
        snap.wait(events_on)
        if self.rig is not None and self.camera_mode == "tip":
            self.rig.publish(snap, env, stream)
        for d in self.bodies:
            d.publish(snap, env, dev, stream)
        for d in self.skinned:
            d.publish(snap, env, dev, stream)
        if self.pieces:
            skin_pieces(snap.control, env, self.skins, [d.pos.array for d in self.pieces], [d.normals.array for d in self.pieces],
                        dev, stream)
        for rd, d in self.rods:  # the tube kernel writes positions and radial normals directly
            rd.place(snap, env, d.pos.array, d.normals.array, dev, stream)
        for highlight in self.region_highlights:
            highlight.publish(self, snap, env, stream)
        for cd, d in self.clips:
            cd.place(snap, env, d.pos.array, d.normals.array, dev, stream)
        for d in self.fluids:
            d.publish(snap, env, stream)
        for d in self.smokes:
            d.publish(snap, env, stream)
        if self.task_markers is not None:
            from i4h_standalone.render.procedure import publish_task
            publish_task(self, snap, place=False)
            self.task_markers.publish(dev, stream, snap.task_rows, env, snap.task_progress)
        if self.markers is not None:
            self.markers.publish(dev, stream)
        if self.overlay is not None and self.debug_spheres:
            c = snap.control
            for o, show in ((self.overlay, self.show_regular), (self.overlay_cut, getattr(self, "show_cut", None))):
                if o is not None:
                    wp.launch(place_sphere_mesh, dim=o.pos.array.shape[0],
                              inputs=[self.sphere_template, c.sphere_q, c.sphere_radius, c.sphere_active, show, int(env)],
                              outputs=[o.pos.array, o.normals.array], device=dev, stream=stream)
        snap.done(events_on)

    def draw(self) -> bool:
        """Graphics side. Returns False once the window closed."""
        spy = self.spy
        if self.closed:
            return False
        surface_texture = None
        if self.window is not None:
            self.window.process_events()
            if self.window.should_close() or self.camera.closed:
                self.closed = True
                return False
            self._apply_resize()
            if self.camera_mode == "free":
                self.camera.move()
            try:
                surface_texture = self.surface.acquire_next_image()
            except RuntimeError:  # out of date: the window changed size again after the resize event (a drag)
                return self._surface_lost()
            if not surface_texture:
                return True
            self.surface_failures = 0
            width, height = int(surface_texture.width), int(surface_texture.height)
        else:
            width, height = self.width, self.height
        self._ensure_targets(width, height)
        if self.rig is not None and self.camera_mode == "tip":
            self.rig.apply(self.camera, self.stream)  # the pose from this frame's snapshot (publish enqueued the copies)
        self._frame = self._frame_uniforms(width, height)
        encoder = self.device.create_command_encoder()
        self.encode_scene(encoder, self.scene_view, self.depth_view, self._frame, view=0)
        encoder.set_texture_state(self.scene_color, spy.ResourceState.shader_resource)
        encoder.set_texture_state(self.depth, spy.ResourceState.shader_resource)  # the fluid composite and AO read it
        self.final, final_view, fluid_depth = self.scene_color, self.scene_view, None
        if self.fluid_passes is not None:  # always, so the present source never changes between frames
            self.final = self.fluid_passes.encode(encoder, self.fluids, self._frame, self._render_state, self.sampler, self.scene_color,
                                                  self.depth, width, height)
            final_view, fluid_depth = self.fluid_passes.views["color"], self.fluid_passes.smooth_depth
        if self.gbuffer is not None:  # normals and labels of the same geometry (smoke has no label)
            self.gbuffer.encode(encoder, width, height)
        if self.smokes:  # last: over the liquid, behind it hidden by the fluid depth, behind opaque geometry by the scene depth
            encoder.set_texture_state(self.final, spy.ResourceState.render_target)
            with encoder.begin_render_pass({
                "color_attachments": [{"view": final_view, "load_op": spy.LoadOp.load, "store_op": spy.StoreOp.store}],
                "depth_stencil_attachment": {"view": self.depth_view, "depth_load_op": spy.LoadOp.load,
                                             "depth_store_op": spy.StoreOp.store},
            }) as pass_:
                for d in self.smokes:
                    d.draw(pass_, self, fluid_depth, width, height)
            encoder.set_texture_state(self.final, spy.ResourceState.shader_resource)
        now = time.perf_counter()
        dt = self.frame_dt if self.frame_dt is not None else (min(max(now - self._last_draw, 0.0), 0.25) if self._last_draw else 1.0 / 60.0)
        self._last_draw = now
        self.display = self.post_passes.encode(encoder, self.final, self.depth, self._frame, self.sampler, width, height, dt,
                                               self.frame_index, self.exposure, self.tonemap)
        if self.capture is not None or surface_texture is None:  # the display frame: read_frame(), the dataset capture
            self._present(encoder, self.ldr, width, height, manual_srgb=True)
            self._ldr_frame = self.frame_index
        if self.capture is not None:
            self.capture.encode(encoder, width, height)
        if surface_texture is not None:
            self._present(encoder, surface_texture, width, height, manual_srgb=self.manual_srgb)
        if self.ui is not None:  # widget callbacks fire inside begin_frame: the frame is encoded but not submitted
            target = surface_texture if surface_texture is not None else self.ldr
            if surface_texture is None:
                encoder.set_texture_state(self.ldr, spy.ResourceState.render_target)
            self.ui.begin_frame(width, height)
            self.ui.end_frame(target, encoder)
            if surface_texture is None:
                encoder.set_texture_state(self.ldr, spy.ResourceState.shader_resource)
        self.frame_index += 1
        command_buffer = encoder.finish()
        stream = self.stream or wp.get_stream(self.scene.device)
        self.interop.submit(command_buffer, stream)
        if self.capture is not None:
            self.capture.after_submit(stream)  # the packed frame to pinned memory, on the same stream
        if self.surface is not None:
            try:
                self.surface.present()
            except RuntimeError:
                return self._surface_lost()
        return True

    def encode_scene(self, encoder, color_view, depth_view, frame: dict, view: int = 0) -> None:
        """The opaque scene (tissue, statics, skinned vessels, instruments, rods, clips, markers) into ``color_view`` and
        ``depth_view``, cleared, with ``frame`` (``_frame_uniforms``) as the camera and lights. ``view`` picks the root
        objects (0 the window or offscreen target, 1 and 2 the XR eyes), so several views of one frame keep their own
        cameras. The window path adds fluids, smoke, post and present around it (``draw``)."""
        spy = self.spy
        self._slot = int(view) * 2 + (self.frame_index & 1)
        for pipeline, block in ((self.tissue_pipeline, self.tissue_params), (self.tissue_pipeline_cull, self.tissue_params),
                                (self.pbr_pipeline, INSTRUMENT_MATERIAL), (self.task_pipeline, INSTRUMENT_MATERIAL)):
            if pipeline is None:
                continue
            cursor = spy.ShaderCursor(self.roots[id(pipeline)][self._slot])
            cursor.write(frame)
            cursor.write(block)
        with encoder.begin_render_pass({
            "color_attachments": [{"view": color_view, "clear_value": self.clear,
                                   "load_op": spy.LoadOp.clear, "store_op": spy.StoreOp.store}],
            "depth_stencil_attachment": {"view": depth_view, "depth_load_op": spy.LoadOp.clear,
                                         "depth_store_op": spy.StoreOp.store, "depth_clear_value": 1.0},
        }) as pass_:
            for d in self.bodies:
                for start, count, textures, color, cull, _ in d.parts:
                    if count:
                        self._draw(pass_, d.streams, d.index_buffer, start, count, textures, color=color, cull=cull)
            for s in self.statics:
                self._draw(pass_, s.streams, s.index_buffer, 0, s.index_count, s.textures, color=s.color, cull=s.cull)
            for d in self.skinned:
                self._draw(pass_, d.streams, d.index_buffer, 0, d.index_count, d.textures, color=d.color, cull=d.cull)
            debug = list(self.markers.draws) if self.markers is not None else []
            if self.debug_spheres:
                debug = debug + [o for o in (self.overlay, self.overlay_cut) if o is not None]
            for d in self.flats + debug:
                self._draw_pbr(pass_, d)
            if self.task_markers is not None or self.region_highlights:
                overlays = [h.draw for h in self.region_highlights]
                if self.task_markers is not None:
                    overlays += self.task_markers.draws
                for d in overlays:
                    self._bind(pass_, self.task_pipeline, d.streams, d.index_buffer, d.color)
                    pass_.draw_indexed({"vertex_count": int(d.index_count), "start_index_location": 0})

    def _surface_lost(self) -> bool:
        """The swapchain no longer matches the window: reconfigure it to the window's current size on the next
        frame and skip this one. A window of zero size (minimised) just skips; anything else that fails thirty
        frames in a row is a real error."""
        width, height = int(self.window.width), int(self.window.height)
        if width > 0 and height > 0:
            self.surface_failures += 1
            if self.surface_failures > 30:
                raise RuntimeError(f"the swapchain could not be re-created at {width}x{height} after 30 frames")
        self.pending_resize = (width, height)
        return True

    def _bind(self, pass_, pipeline, streams, index_buffer, color):
        """Bind a pipeline with its vertex streams, the frame uniforms and the flat colour; returns the cursor."""
        spy = self.spy
        pass_.set_render_state({"vertex_buffers": streams, "index_buffer": index_buffer, "index_format": spy.IndexFormat.uint32,
                                **self._render_state})
        root = self.roots[id(pipeline)][self._slot]  # the frame block and the material were written by encode_scene
        pass_.bind_pipeline(pipeline, root)
        cursor = spy.ShaderCursor(root)
        cursor.mesh_color = spy.float4(float(color[0]), float(color[1]), float(color[2]), 1.0)
        return cursor

    def _draw(self, pass_, streams, index_buffer, start, count, textures, color=(1.0, 1.0, 1.0), cull="none"):
        """Tissue draw: ``mesh_color`` multiplies the diffuse (tissue.slang)."""
        cursor = self._bind(pass_, self.tissue_pipeline_cull if cull == "back" else self.tissue_pipeline, streams, index_buffer, color)
        cursor.write(textures)
        pass_.draw_indexed({"vertex_count": int(count), "start_index_location": int(start)})

    def _draw_pbr(self, pass_, d: FlatDraw):
        """Flat-coloured pbr draw (instruments, rods, overlay; omnisurg/rendering/renderers.py:1278-1357)."""
        self._bind(pass_, self.pbr_pipeline, d.streams, d.index_buffer, d.color)
        pass_.draw_indexed({"vertex_count": int(d.index_count), "start_index_location": 0})

    @property
    def flats(self) -> list:
        """The flat-coloured draws that carry a label (instrument pieces, rod tubes, clips); the markers and the sphere
        overlay are debug views without one."""
        return self.pieces + [d for _, d in self.rods] + [d for _, d in self.clips]

    def _frame_uniforms(self, width, height, basis=None, view_projection=None) -> dict:
        """Uniforms every draw call of one view shares: camera basis, projection, lights, headlight and environment
        (slang.py:4982-5070, 4718-4731). The window camera by default; an XR eye passes its ``basis``
        ``(position, right, up, forward)`` and its off-axis ``view_projection`` (the fov and aspect entries then only
        serve the window-only passes, which an eye does not run)."""
        spy = self.spy
        pos, right, up, forward = self.camera.basis() if basis is None else basis
        matrix = self.camera.view_projection(width, height) if view_projection is None else view_projection
        def f3(v):
            return spy.float3(float(v[0]), float(v[1]), float(v[2]))
        u = {"camera_pos": f3(pos), "camera_right": f3(right), "camera_up": f3(up), "camera_forward": f3(forward),
             "camera_inv_tan_half_fovy": float(1.0 / math.tan(math.radians(self.camera.fov) * 0.5)),
             "camera_aspect": float(width) / max(float(height), 1.0), "camera_near": self.camera.near, "camera_far": self.camera.far,
             "camera_view_projection": spy.float4x4(np.asarray(matrix, np.float32))}
        u.update(self._lights)  # the directional lights are disabled (black), the fluid's specular reads light_key_dir
        for name, direction in DIRECTIONAL.items():  # camera-space directions
            d = right * direction[0] + up * direction[1] + forward * direction[2]
            u[f"light_{name}_dir"] = f3(d / max(float(np.linalg.norm(d)), 1e-8))
        h = self.headlight
        outer = math.radians(float(np.clip(h["spot_degrees"], 5.0, 160.0)) * 0.5)
        inner = outer * (1.0 - float(np.clip(h["softness"], 0.0, 1.0)))
        u.update(endoscope_light_enabled=int(float(h["intensity"]) > 0.0),
                 endoscope_light_color=f3(np.asarray(h["color"], np.float32) * float(h["intensity"])),
                 endoscopic_enabled=1,
                 headlight_intensity=1.4,  # LightingParams.headlight_intensity; the light colour carries the YAML intensity
                 headlight_range=float(np.clip(h["range"], 0.05, 20.0)),
                 headlight_inner_cos=math.cos(inner), headlight_outer_cos=math.cos(outer),
                 directional_fill_scale=float(np.clip(h["directional_fill_scale"], 0.0, 1.0)),
                 environment_map=self.environment if self.environment is not None else self._default("black"),
                 environment_sampler=self.sampler, has_environment_map=int(self.environment is not None),
                 environment_lighting_enabled=int(self.env_lighting), environment_intensity=self.env_intensity,
                 environment_rotation=self.env_rotation / 360.0, environment_pitch=math.radians(self.env_pitch))
        return u

    def _present(self, encoder, target, width, height, manual_srgb: bool, frame_index: int | None = None):
        """The post chain's output into ``target`` (the swapchain image or ``ldr``) with sensor noise and the sRGB encode."""
        spy = self.spy
        offscreen = target is self.ldr
        if offscreen:
            view = self.ldr_view
            encoder.set_texture_state(target, spy.ResourceState.render_target)
        else:  # the swapchain rotates a few images: one view each, dropped with the surface on resize
            cached = self._surface_views.get(id(target))
            if cached is None or cached[0] is not target:
                cached = self._surface_views[id(target)] = (target, target.create_view({}))
            view = cached[1]
        with encoder.begin_render_pass({"color_attachments": [{"view": view, "load_op": spy.LoadOp.dont_care, "store_op": spy.StoreOp.store}]}) as pass_:
            pass_.set_render_state({"viewports": [spy.Viewport.from_size(width, height)],
                                    "scissor_rects": [spy.ScissorRect.from_size(width, height)]})
            cursor = spy.ShaderCursor(pass_.bind_pipeline(self.present_pipeline if not offscreen else self.ldr_pipeline))
            cursor.write({"source_tex": self.display, "post_sampler": self.sampler,
                          "output_size": spy.float2(float(max(1, width)), float(max(1, height))),
                          "frame_index": int(self.frame_index if frame_index is None else frame_index),
                          "sensor_noise_enabled": int(self.post.sensor_noise_enabled), "sensor_noise_strength": float(self.post.sensor_noise_strength),
                          "sensor_noise_shadow_boost": float(self.post.sensor_noise_shadow_boost), "manual_srgb_encode": int(manual_srgb)})
            pass_.draw({"vertex_count": 3})
        if offscreen:
            encoder.set_texture_state(target, spy.ResourceState.shader_resource)

    def _ensure_targets(self, width, height):
        spy = self.spy
        if self.scene_color is None or self.scene_color.width != width or self.scene_color.height != height:
            self.scene_color = self.device.create_texture(
                format=self.scene_format, width=width, height=height,
                usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource, label="i4h-scene-color")
            self.depth = self.device.create_texture(
                format=self.depth_format, width=width, height=height,
                usage=spy.TextureUsage.depth_stencil | spy.TextureUsage.shader_resource, label="i4h-depth")
            self.scene_view, self.depth_view = self.scene_color.create_view({}), self.depth.create_view({})
            self._render_state = {"viewports": [spy.Viewport.from_size(width, height)],
                                  "scissor_rects": [spy.ScissorRect.from_size(width, height)]}
            self.ldr = self.device.create_texture(format=self.ldr_format, width=width, height=height,
                                                  usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource, label="i4h-ldr")
            self.ldr_view = self.ldr.create_view({})
        if self.fluid_passes is not None:
            self.fluid_passes.ensure_targets(width, height)
        self.post_passes.ensure_targets(width, height)
        if self.gbuffer is not None:
            self.gbuffer.ensure_targets(width, height)

    def _on_resize(self, width, height):
        self.pending_resize = (int(width), int(height))

    def _apply_resize(self):
        if self.pending_resize is None:
            return
        self.pending_resize = None
        width, height = int(self.window.width), int(self.window.height)  # the size now, not the event's
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=self.vsync)
            self.width, self.height = width, height
        self.scene_color = self.depth = self.scene_view = self.depth_view = self.final = self.ldr = self.ldr_view = self.display = None
        self._surface_views = {}
        if self.fluid_passes is not None:
            self.fluid_passes.drop_targets()
        self.post_passes.drop_targets()
        if self.gbuffer is not None:
            self.gbuffer.drop_targets()

    # misc ---------------------------------------------------------------------

    def read_frame(self) -> np.ndarray:
        """The display frame (post chain, sensor noise, sRGB-encoded: what the window shows) as float32 ``(H, W, 4)`` in
        [0, 1]; syncs the device. Encoded offscreen and while a dataset capture runs; a windowed frame without a capture
        presents to the swapchain only, so the last frame is presented into ``ldr`` here on demand."""
        if self.ldr is None or self.display is None:
            raise RuntimeError("no frame drawn yet")
        if self._ldr_frame != self.frame_index - 1:
            encoder = self.device.create_command_encoder()
            self._present(encoder, self.ldr, self.ldr.width, self.ldr.height, manual_srgb=True, frame_index=self.frame_index - 1)  # its noise seed
            self.device.submit_command_buffer(encoder.finish())
            self._ldr_frame = self.frame_index - 1
        self.device.wait()
        return self.ldr.to_numpy().astype(np.float32) / 255.0

    def read_gbuffer(self) -> tuple[np.ndarray, np.ndarray]:
        """``(normal_depth (H, W, 4), labels (H, W))`` of the last frame (``render.gbuffer: true``); syncs the device."""
        if self.gbuffer is None:
            raise RuntimeError("the G-buffer is off (render.gbuffer: true)")
        return self.gbuffer.read()

    def read_hdr(self) -> np.ndarray:
        """Linear HDR colour before the post chain (after the fluid composite when the scene has pools); syncs the device."""
        self.device.wait()
        if self.final is None:
            raise RuntimeError("no frame drawn yet")
        return self.final.to_numpy().astype(np.float32)

    def is_running(self) -> bool:
        return not self.closed

    def close(self) -> None:
        """Release everything in dependency order: CUDA mappings and GPU objects, then the surface, then
        its window, then the device (slangpy requires an explicit ``Device.close``; a surface destroyed
        after its window, or any object destroyed after the device, segfaults)."""
        if self.device is None:
            return
        atexit.unregister(self.close)  # otherwise the registration keeps the renderer (and its scene) alive
        self.closed = True
        self.device.wait()
        wp.synchronize_stream(self.stream or self.scene.device)  # pending publish copies into the shared buffers
        self.interop.close()
        for h in self.region_highlights:
            h.draw.close()
        self.region_highlights = []
        if self.task_markers is not None:
            self.task_markers.close()
            self.task_markers = None
        if self.markers is not None:
            self.markers.close()
            self.markers = None
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        for d in (self.bodies + self.statics + self.skinned + self.flats + self.fluids + self.smokes
                  + [o for o in (self.overlay, self.overlay_cut) if o is not None]):
            d.close()
        if self.fluid_passes is not None:
            self.fluid_passes.close()
        self.post_passes.close()
        self.post_passes = None
        if self.gbuffer is not None:
            self.gbuffer.close()
            self.gbuffer = None
        self.bodies = self.statics = self.skinned = self.pieces = self.rods = self.clips = self.fluids = self.smokes = []
        self.overlay = self.overlay_cut = self.fluid_passes = None
        self.scene_color = self.depth = self.scene_view = self.depth_view = self.final = self.environment = None
        self.tissue_pipeline = self.tissue_pipeline_cull = self.pbr_pipeline = self.task_pipeline = self.present_pipeline = self.ldr_pipeline = None
        self.roots = {}
        self.sampler = self.zero_color = self.ldr = self.ldr_view = self.display = None
        self._frame = self._render_state = None
        self._textures.clear()
        self._surface_views.clear()
        self.surface = None
        self.ui = None  # widgets and the ImGui pipelines go before the device
        if self.window is not None:
            self.window.close()
            self.window = None
        d = h = None  # loop locals must release their GPU buffers before Device.close
        if self.owns_device:
            self.device.close()
        self.device = None
