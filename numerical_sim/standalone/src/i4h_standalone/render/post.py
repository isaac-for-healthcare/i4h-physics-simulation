# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post chain of the Slang renderer: AO, bloom, auto exposure, the composite (exposure, tonemap, lens dirt, colour
grade, lens distortion, chromatic aberration, vignette, scope mask) and FXAA, into the display texture.

Port of omnisurg/rendering/slang.py: ``PostProcessParams`` 239-288 and its ranges 104-141 (``Post``, ``RANGES``),
the AO passes 5182-5236 and textures 4871-4891, the bloom pyramid 5624-5665 with ``_BLOOM_MAX_LEVELS`` 60 and
``_bloom_level_sizes`` 4892-4903, auto exposure 5680-5726 with its 1x1 ping-pong 4957-4974, the composite 5727-5815
and FXAA 5817-5860. Edits: **every pass runs every frame and a toggle only changes a uniform** (omnisurg ran AO, bloom,
exposure and FXAA conditionally with fallback textures and kept only the composite unconditional so the swapchain
never saw its present source change, slang.py:6813-6816; ours has one pass list for the surface's life, no fallback
textures but the lens-dirt default, and offscreen frames equal windowed ones); the exposure state is ``rgba32_float``
(half floats lose the adapted value); ``scope_enabled`` is the canonical flag, default off; ``exposure`` and the
tonemap stay the renderer's top-level settings. ``frame_dt`` on the renderer replaces the wall clock in tests.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

BLOOM_LEVELS = 5  # omnisurg _BLOOM_MAX_LEVELS
TONEMAPS = ("none", "aces", "reinhard")


@dataclass
class Post:
    white_balance: tuple = (1.0, 1.0, 1.0)
    auto_exposure_enabled: bool = False
    auto_exposure_target_luma: float = 0.35
    auto_exposure_min: float = 0.35
    auto_exposure_max: float = 1.8
    auto_exposure_speed: float = 4.0
    auto_exposure_highlight_weight: float = 0.8
    ao_enabled: bool = False
    ao_intensity: float = 2.0
    ao_radius: float = 0.18
    ao_bias: float = 0.0025
    ao_power: float = 1.7
    fxaa_enabled: bool = False
    fxaa_subpix: float = 0.75
    fxaa_edge_threshold: float = 0.125
    fxaa_edge_threshold_min: float = 0.0312
    bloom_enabled: bool = False
    bloom_threshold: float = 1.0
    bloom_intensity: float = 0.25
    bloom_radius: float = 12.0
    lens_dirt_enabled: bool = False
    lens_dirt_texture: str | None = None  # a path; omnisurg picked one of four packaged files by index
    lens_dirt_intensity: float = 0.45
    lens_dirt_threshold: float = 0.20
    lens_dirt_base_opacity: float = 0.05
    lens_dirt_global_drive: float = 1.0
    lens_dirt_mask_gamma: float = 0.60
    lens_distortion_enabled: bool = False
    lens_distortion_strength: float = 0.08
    lens_distortion_zoom: float = 1.04
    chromatic_aberration_enabled: bool = False
    chromatic_aberration_strength: float = 0.6
    sensor_noise_enabled: bool = False
    sensor_noise_strength: float = 0.008
    sensor_noise_shadow_boost: float = 1.5
    color_grade_enabled: bool = False
    color_saturation: float = 1.0
    color_contrast: float = 1.0
    color_gamma: float = 1.0
    color_warmth: float = 0.0
    vignette_strength: float = 0.0
    vignette_radius: float = 0.78
    scope_enabled: bool = False
    scope_radius: float = 1.5
    scope_softness: float = 0.035


RANGES = {  # omnisurg/rendering/slang.py:106-141 (exposure lives on the renderer)
    "auto_exposure_target_luma": (0.05, 1.0), "auto_exposure_min": (0.05, 2.0), "auto_exposure_max": (0.1, 4.0),
    "auto_exposure_speed": (0.1, 12.0), "auto_exposure_highlight_weight": (0.0, 4.0),
    "ao_intensity": (0.0, 4.0), "ao_radius": (0.0, 0.30), "ao_bias": (0.0, 0.03), "ao_power": (0.25, 4.0),
    "fxaa_subpix": (0.0, 1.0), "fxaa_edge_threshold": (0.0312, 0.333), "fxaa_edge_threshold_min": (0.0, 0.0833),
    "bloom_threshold": (0.0, 1.0), "bloom_intensity": (0.0, 10.0), "bloom_radius": (0.0, 64.0),
    "lens_dirt_intensity": (0.0, 10.0), "lens_dirt_threshold": (0.0, 1.0), "lens_dirt_base_opacity": (0.0, 1.0),
    "lens_dirt_global_drive": (0.0, 4.0), "lens_dirt_mask_gamma": (0.25, 2.0),
    "lens_distortion_strength": (-0.35, 0.35), "lens_distortion_zoom": (0.8, 1.2), "chromatic_aberration_strength": (0.0, 4.0),
    "sensor_noise_strength": (0.0, 0.10), "sensor_noise_shadow_boost": (0.0, 4.0),
    "color_saturation": (0.0, 2.0), "color_contrast": (0.5, 1.5), "color_gamma": (0.5, 2.0), "color_warmth": (-1.0, 1.0),
    "vignette_strength": (0.0, 1.0), "vignette_radius": (0.0, 1.5), "scope_radius": (0.0, 1.5), "scope_softness": (0.001, 0.5),
}
SKIP = ("white_balance", "lens_dirt_texture")  # not plain scalars: bound by hand, not exposed as knobs


def bloom_sizes(width: int, height: int) -> list[tuple[int, int]]:
    """omnisurg _bloom_level_sizes: half resolution first, halving down to at most BLOOM_LEVELS levels."""
    sizes = []
    w, h = max(1, (int(width) + 1) // 2), max(1, (int(height) + 1) // 2)
    for _ in range(BLOOM_LEVELS):
        sizes.append((w, h))
        if w == 1 and h == 1:
            break
        w, h = max(1, (w + 1) // 2), max(1, (h + 1) // 2)
    return sizes


def uniforms(post: Post) -> dict:
    """Every scalar field as a shader uniform value (bools as ints)."""
    return {f.name: (int(v) if isinstance(v, bool) else float(v)) for f in dataclasses.fields(post)
            if f.name not in SKIP for v in (getattr(post, f.name),)}


SHADER_KEYS = {  # every name a pass may write, per shader; test_render checks each against the .slang declarations
    "ao": ("depth_tex", "source_ao_tex", "post_sampler", "output_size", "source_size", "camera_inv_tan_half_fovy", "camera_aspect",
           "camera_near", "camera_far", "frame_index", "ao_enabled", "ao_intensity", "ao_radius", "ao_bias", "ao_power"),
    "bloom": ("source_tex", "base_tex", "post_sampler", "source_size", "output_size", "exposure", "white_balance", "bloom_threshold",
              "bloom_radius", "bloom_enabled"),
    "exposure": ("scene_color_tex", "bloom_global_tex", "previous_exposure_tex", "post_sampler", "exposure", "white_balance",
                 "bloom_intensity", "auto_exposure_target_luma", "auto_exposure_min", "auto_exposure_max", "auto_exposure_speed",
                 "auto_exposure_highlight_weight", "auto_exposure_dt", "auto_exposure_initialized"),
    "post": ("scene_color_tex", "bloom_tex", "bloom_global_tex", "ao_tex", "lens_dirt_tex", "auto_exposure_tex", "post_sampler",
             "output_size", "tonemap_operator", "exposure", "white_balance", "auto_exposure_enabled", "ao_enabled", "bloom_enabled",
             "bloom_intensity", "bloom_radius", "lens_dirt_enabled", "lens_dirt_intensity", "lens_dirt_threshold",
             "lens_dirt_base_opacity", "lens_dirt_global_drive", "lens_dirt_mask_gamma", "lens_distortion_enabled",
             "lens_distortion_strength", "lens_distortion_zoom", "chromatic_aberration_enabled", "chromatic_aberration_strength",
             "color_grade_enabled", "color_saturation", "color_contrast", "color_gamma", "color_warmth", "vignette_strength",
             "vignette_radius", "scope_enabled", "scope_radius", "scope_softness"),
    "fxaa": ("source_tex", "post_sampler", "output_size", "fxaa_enabled", "fxaa_subpix", "fxaa_edge_threshold", "fxaa_edge_threshold_min"),
}
PASS_SHADER = {"ao": "ao", "ao_blur": "ao", "prefilter": "bloom", "down": "bloom", "up": "bloom", "exposure": "exposure",
               "composite": "post", "fxaa": "fxaa"}


class Target(NamedTuple):
    """A render target with its view and size, made once per surface size."""

    tex: object
    view: object
    size: tuple


class PostPasses:
    def __init__(self, renderer, post: Post):
        r, spy, dev = renderer, renderer.spy, renderer.device
        self.spy, self.device, self.renderer, self.post = spy, dev, r, post
        programs = {name: dev.load_program(f"{shader}.slang", ["vertex_main", entry]) for name, shader, entry in (
            ("ao", "ao", "ao_fragment"), ("ao_blur", "ao", "blur_fragment"),
            ("prefilter", "bloom", "prefilter_downsample_fragment"), ("down", "bloom", "downsample_fragment"),
            ("up", "bloom", "upsample_fragment"), ("exposure", "exposure", "adapt_exposure_fragment"),
            ("composite", "post", "fragment_main"), ("fxaa", "fxaa", "fragment_main"))}
        common = dict(input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list, rasterizer={"cull_mode": spy.CullMode.none})
        fmt = {"exposure": spy.Format.rgba32_float}
        self.pipelines = {name: dev.create_render_pipeline(program=p, targets=[{"format": fmt.get(name, r.scene_format)}],
                                                           label=f"i4h-post-{name}", **common) for name, p in programs.items()}
        self.exposure_tex = [self._target(spy.Format.rgba32_float, 1, 1, f"i4h-exposure-{i}")
                             for i in range(2)]  # (adapted, target, measured luma): ping-pong, size independent
        self.exposure_index = 0
        self.lens_dirt = r._texture(post.lens_dirt_texture, True) if post.lens_dirt_texture else r._default("black")
        self.size = None
        self.ao = self.ao_blur = self.color = self.fxaa = None
        self.sizes: list = []
        self.down: list = []
        self.up: list = []

    def _target(self, fmt, w: int, h: int, label: str) -> Target:
        spy = self.spy
        tex = self.device.create_texture(format=fmt, width=w, height=h, usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
                                         label=label)
        return Target(tex, tex.create_view({}), (w, h))

    def ensure_targets(self, width: int, height: int):
        if self.size == (width, height):
            return
        fmt = self.renderer.scene_format
        hw, hh = max(1, width // 2), max(1, height // 2)  # omnisurg _ao_texture_sizes: half resolution
        self.ao, self.ao_blur = self._target(fmt, hw, hh, "i4h-ao"), self._target(fmt, hw, hh, "i4h-ao-blur")
        self.sizes = bloom_sizes(width, height)
        self.down = [self._target(fmt, w, h, f"i4h-bloom-down-{i}") for i, (w, h) in enumerate(self.sizes)]
        self.up = [self._target(fmt, w, h, f"i4h-bloom-up-{i}") for i, (w, h) in enumerate(self.sizes[:-1])]  # the last level is never upsampled into
        self.color, self.fxaa = self._target(fmt, width, height, "i4h-post-color"), self._target(fmt, width, height, "i4h-fxaa")
        self.size = (width, height)

    def drop_targets(self):
        self.size = None
        self.ao = self.ao_blur = self.color = self.fxaa = None
        self.sizes, self.down, self.up = [], [], []

    def _pass(self, encoder, name: str, target: Target, values: dict):
        """One fullscreen pass into ``target`` (every pixel is written: no clear); ``values`` must be names the shader declares."""
        spy = self.spy
        unknown = set(values) - set(SHADER_KEYS[PASS_SHADER[name]])
        if unknown:
            raise KeyError(f"post pass {name}: {sorted(unknown)} are not declared by {PASS_SHADER[name]}.slang")
        encoder.set_texture_state(target.tex, spy.ResourceState.render_target)
        with encoder.begin_render_pass({"color_attachments": [{"view": target.view, "load_op": spy.LoadOp.dont_care,
                                                               "store_op": spy.StoreOp.store}]}) as pass_:
            pass_.set_render_state({"viewports": [spy.Viewport.from_size(*target.size)], "scissor_rects": [spy.ScissorRect.from_size(*target.size)]})
            cursor = spy.ShaderCursor(pass_.bind_pipeline(self.pipelines[name]))
            cursor.write(values)
            pass_.draw({"vertex_count": 3})
        encoder.set_texture_state(target.tex, spy.ResourceState.shader_resource)

    def encode(self, encoder, source, depth, frame: dict, sampler, width: int, height: int, dt: float, frame_index: int,
               exposure: float, tonemap: int):
        """AO, bloom, exposure, composite, FXAA; returns the display texture (in ``shader_resource`` state)."""
        spy, p = self.spy, self.post
        def f2(w, h):
            return spy.float2(float(max(1, w)), float(max(1, h)))
        full, u = f2(width, height), uniforms(p)
        wb = spy.float3(*(float(v) for v in p.white_balance))
        ao = {**{k: frame[k] for k in ("camera_inv_tan_half_fovy", "camera_aspect", "camera_near", "camera_far")},
              "depth_tex": depth, "post_sampler": sampler, "output_size": f2(*self.ao.size), "source_size": full,
              "frame_index": int(frame_index), **{k: u[k] for k in ("ao_enabled", "ao_intensity", "ao_radius", "ao_bias", "ao_power")}}
        self._pass(encoder, "ao", self.ao, ao)
        self._pass(encoder, "ao_blur", self.ao_blur, {**ao, "source_ao_tex": self.ao.tex})
        bloom_on = p.bloom_enabled and p.bloom_intensity > 0.0 and p.bloom_radius > 0.0
        bloom = {"post_sampler": sampler, "exposure": float(exposure), "white_balance": wb, "bloom_threshold": u["bloom_threshold"],
                 "bloom_radius": u["bloom_radius"],
                 "bloom_enabled": int(bloom_on or p.lens_dirt_enabled)}  # the lens dirt's global drive reads the pyramid too
        down, up = self.down, self.up
        self._pass(encoder, "prefilter", down[0], {**bloom, "source_tex": source, "source_size": full, "output_size": f2(*down[0].size)})
        for i in range(1, len(down)):
            self._pass(encoder, "down", down[i], {**bloom, "source_tex": down[i - 1].tex, "source_size": f2(*down[i - 1].size),
                                                  "output_size": f2(*down[i].size)})
        current = down[-1]
        for i in range(len(down) - 2, -1, -1):
            self._pass(encoder, "up", up[i], {**bloom, "source_tex": current.tex, "base_tex": down[i].tex,
                                              "source_size": f2(*current.size), "output_size": f2(*up[i].size)})
            current = up[i]
        previous = self.exposure_tex[self.exposure_index]
        target = self.exposure_tex[1 - self.exposure_index]
        self._pass(encoder, "exposure", target, {
            "scene_color_tex": source, "bloom_global_tex": down[-1].tex, "previous_exposure_tex": previous.tex, "post_sampler": sampler,
            "exposure": float(exposure), "white_balance": wb, "bloom_intensity": u["bloom_intensity"] if bloom_on else 0.0,
            **{k: u[k] for k in ("auto_exposure_target_luma", "auto_exposure_min", "auto_exposure_max", "auto_exposure_speed",
                                 "auto_exposure_highlight_weight")},
            "auto_exposure_dt": float(dt), "auto_exposure_initialized": int(frame_index > 0)})
        self.exposure_index = 1 - self.exposure_index
        table = {**u, "scene_color_tex": source, "bloom_tex": current.tex, "bloom_global_tex": down[-1].tex, "ao_tex": self.ao_blur.tex,
                 "lens_dirt_tex": self.lens_dirt, "auto_exposure_tex": target.tex, "post_sampler": sampler, "output_size": full,
                 "tonemap_operator": int(tonemap), "exposure": float(exposure), "white_balance": wb}
        self._pass(encoder, "composite", self.color, {k: table[k] for k in SHADER_KEYS["post"]})
        self._pass(encoder, "fxaa", self.fxaa, {"source_tex": self.color.tex, "post_sampler": sampler, "output_size": full,
                                                **{k: u[k] for k in ("fxaa_enabled", "fxaa_subpix", "fxaa_edge_threshold", "fxaa_edge_threshold_min")}})
        return self.fxaa.tex

    def exposure_value(self) -> np.ndarray:
        """(adapted exposure, target, measured luma) of the last frame; syncs the device (tests)."""
        self.device.wait()
        return np.asarray(self.exposure_tex[self.exposure_index].tex.to_numpy(), dtype=np.float32).reshape(-1)[:3]

    def close(self):
        self.drop_targets()
        self.pipelines = {}
        self.exposure_tex = []
        self.lens_dirt = None
