# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Screen-space fluid for the Slang renderer: one draw per pool, four passes into a second HDR target.

Port of omnisurg/rendering/slang.py fluid passes (pipelines 2894-2993, textures 4811-4866, passes
5285-5624) and ``omnisurg_fluid.slang``. Per frame: the drawn env's positions and flags of every pool are
copied into shared buffers in ``publish()``; ``encode()`` then renders instanced sprites twice (additive
thickness with the pool's colour, then depth-tested eye depth), blurs the depth bilaterally and composites
refraction, absorption and a Fresnel reflection over the scene colour into ``color`` (the texture the
present pass and ``read_frame`` take). The passes always run when the scene has pools (an empty frame is
an early-out in the composite fragment), so the pass structure is fixed for the surface's life; dead
particles are degenerate quads, so nothing is compacted or read back. Dropped: anisotropy and the smoothed
positions, the thickness blur, the debug views, the legacy blur mode, the blurred environment reflection.
"""

from __future__ import annotations

import warp as wp
from i4h_isaaclab.appearance import appearance
from i4h_standalone.render import CAMERA, FLUID_COLOR
from i4h_standalone.render.snapshots import copy

FRAME_UNIFORMS = CAMERA + (  # the subset of SlangRenderer._frame the composite reads
    "light_key_dir", "light_fill_dir", "light_back_dir", "key_light_enabled",
    "fill_light_enabled", "back_light_enabled", "endoscope_light_enabled", "light_key_color", "light_fill_color",
    "light_ambient_color", "light_rim_color", "endoscope_light_color", "environment_map", "environment_sampler",
    "has_environment_map", "environment_lighting_enabled", "environment_intensity", "environment_rotation",
    "environment_pitch",
)
LIGHTING = dict(specular_scale=0.35, ambient=0.22, rim_strength=0.12)  # TISSUE_PARAMS values the fluid shader shares
DEFAULTS = dict(blur=0.02, thickness_scale=0.1, ior=1.33, reflection_strength=0.35, refraction_strength=1.0,
                absorption_strength=4.0)  # omnisurg examples/ chole fluid block and its's getattr defaults


class FluidDraw:
    """Shared position and flag buffers of one pool's drawn env, plus its colour and sprite radius."""

    def __init__(self, renderer, body):
        r, spy = renderer, renderer.spy
        self.body = body
        self.scene = r.scene
        self.count = body.count
        self.pos = r.shared(body.count, wp.vec3f, 12, f"{body.name}-fluid-positions", usage=spy.BufferUsage.shader_resource)
        self.flags = r.shared(body.count, wp.int32, 4, f"{body.name}-fluid-flags", usage=spy.BufferUsage.shader_resource)
        visual = appearance(body.fluid)
        self.color = tuple(float(c) for c in visual.get("color", FLUID_COLOR))
        self.radius = float(visual.get("render_radius", body.fluid.rest_distance))  # old chole: render_radius == rest_distance

    def publish(self, snap, env: int, stream=None):
        """The drawn env's positions and flags from ``snap`` into the shared buffers."""
        scene = self.scene
        copy(self.pos.array, scene.view(snap.pool_q, self.body)[env], stream)  # pools live in the pool block
        copy(self.flags.array, scene.view(snap.pool_flags, self.body)[env], stream)

    def close(self):
        self.pos.close()
        self.flags.close()


class FluidPasses:
    def __init__(self, renderer, **settings):
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"render.fluid: unknown keys {sorted(unknown)}; known {sorted(DEFAULTS)}")
        self.settings = {**DEFAULTS, **{k: float(v) for k, v in settings.items()}}
        r, spy, dev = renderer, renderer.spy, renderer.device
        self.spy, self.device, self.renderer = spy, dev, r
        programs = {name: dev.load_program("fluid.slang", entries) for name, entries in (
            ("depth", ["depth_vertex_main", "depth_fragment_main"]),
            ("thickness", ["depth_vertex_main", "thickness_fragment_main"]),
            ("smooth", ["fullscreen_vertex_main", "smooth_fragment_main"]),
            ("composite", ["fullscreen_vertex_main", "composite_fragment_main"]),
        )}
        common = dict(input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list,
                      rasterizer={"cull_mode": spy.CullMode.none})
        additive = spy.ColorTargetDesc()  # omnisurg _additive_color_target: thickness sums every splat on the ray
        additive.format = spy.Format.rgba16_float
        additive.enable_blend = True
        additive.color.src_factor = additive.alpha.src_factor = spy.BlendFactor.one
        additive.color.dst_factor = additive.alpha.dst_factor = spy.BlendFactor.one
        additive.color.op = additive.alpha.op = spy.BlendOp.add
        self.depth_pipeline = dev.create_render_pipeline(
            program=programs["depth"], targets=[{"format": spy.Format.r32_float}],
            depth_stencil={"format": r.depth_format, "depth_test_enable": True, "depth_write_enable": True,
                           "depth_func": spy.ComparisonFunc.less}, label="i4h-fluid-depth", **common)
        self.thickness_pipeline = dev.create_render_pipeline(
            program=programs["thickness"], targets=[additive],
            depth_stencil={"format": r.depth_format, "depth_test_enable": False, "depth_write_enable": False},
            label="i4h-fluid-thickness", **common)
        self.smooth_pipeline = dev.create_render_pipeline(program=programs["smooth"], targets=[{"format": spy.Format.r32_float}],
                                                          label="i4h-fluid-smooth", **common)
        self.composite_pipeline = dev.create_render_pipeline(program=programs["composite"], targets=[{"format": r.scene_format}],
                                                             label="i4h-fluid-composite", **common)
        self.size = None
        self.thickness = self.eye_depth = self.smooth_depth = self.depth = self.color = None
        self.views: dict = {}

    def ensure_targets(self, width: int, height: int):
        if self.size == (width, height):
            return
        spy, dev, r = self.spy, self.device, self.renderer
        usage = spy.TextureUsage.render_target | spy.TextureUsage.shader_resource
        def make(fmt, label, use=usage):
            return dev.create_texture(format=fmt, width=width, height=height, usage=use, label=label)
        self.thickness = make(spy.Format.rgba16_float, "i4h-fluid-thickness")
        self.eye_depth = make(spy.Format.r32_float, "i4h-fluid-eye-depth")
        self.smooth_depth = make(spy.Format.r32_float, "i4h-fluid-smooth-depth")
        self.depth = make(r.depth_format, "i4h-fluid-depth", spy.TextureUsage.depth_stencil)
        self.color = make(r.scene_format, "i4h-fluid-color")
        self.views = {name: getattr(self, name).create_view({}) for name in ("thickness", "eye_depth", "smooth_depth", "depth", "color")}
        self.size = (width, height)

    def drop_targets(self):
        self.size = None
        self.thickness = self.eye_depth = self.smooth_depth = self.depth = self.color = None
        self.views = {}

    def encode(self, encoder, draws, frame: dict, state: dict, sampler, scene_color, scene_depth, width: int, height: int):
        """Encode the four passes; returns the composited HDR texture (in ``shader_resource`` state)."""
        spy = self.spy
        def clear(view, value):
            return {"view": view, "clear_value": value, "load_op": spy.LoadOp.clear, "store_op": spy.StoreOp.store}
        depth_attachment = {"view": self.views["depth"], "depth_load_op": spy.LoadOp.clear, "depth_store_op": spy.StoreOp.store,
                            "depth_clear_value": 1.0}
        camera = {name: frame[name] for name in CAMERA}
        for target, view, pipeline in ((self.thickness, self.views["thickness"], self.thickness_pipeline),
                                       (self.eye_depth, self.views["eye_depth"], self.depth_pipeline)):
            encoder.set_texture_state(target, spy.ResourceState.render_target)
            with encoder.begin_render_pass({"color_attachments": [clear(view, [0.0, 0.0, 0.0, 0.0])],
                                            "depth_stencil_attachment": depth_attachment}) as pass_:
                pass_.set_render_state(state)
                for d in draws:
                    cursor = spy.ShaderCursor(pass_.bind_pipeline(pipeline))
                    cursor.write(camera)
                    cursor.mesh_color = spy.float4(d.color[0], d.color[1], d.color[2], 1.0)
                    cursor.fluid_positions = d.pos.buffer
                    cursor.fluid_flags = d.flags.buffer
                    cursor.fluid_count = int(d.count)
                    cursor.fluid_radius = d.radius
                    cursor.fluid_thickness_scale = self.settings["thickness_scale"]
                    pass_.draw({"vertex_count": 6, "instance_count": int(d.count)})
            encoder.set_texture_state(target, spy.ResourceState.shader_resource)

        screen = {"fluid_thickness_tex": self.thickness, "post_sampler": sampler, "fluid_blur": self.settings["blur"],
                  "output_size": spy.float2(float(max(1, width)), float(max(1, height)))}
        encoder.set_texture_state(self.smooth_depth, spy.ResourceState.render_target)
        with encoder.begin_render_pass({"color_attachments": [clear(self.views["smooth_depth"], [0.0, 0.0, 0.0, 0.0])]}) as pass_:
            pass_.set_render_state(state)
            cursor = spy.ShaderCursor(pass_.bind_pipeline(self.smooth_pipeline))
            cursor.write({**camera, **screen, "fluid_depth_tex": self.eye_depth})
            pass_.draw({"vertex_count": 3})
        encoder.set_texture_state(self.smooth_depth, spy.ResourceState.shader_resource)

        encoder.set_texture_state(self.color, spy.ResourceState.render_target)
        with encoder.begin_render_pass({"color_attachments": [clear(self.views["color"], [0.0, 0.0, 0.0, 1.0])]}) as pass_:
            pass_.set_render_state(state)
            cursor = spy.ShaderCursor(pass_.bind_pipeline(self.composite_pipeline))
            s = self.settings
            color = draws[0].color  # the composite's fallback colour where the thickness is empty; one colour for all pools
            cursor.write({**{name: frame[name] for name in FRAME_UNIFORMS}, **LIGHTING, **screen, "fluid_depth_tex": self.smooth_depth,
                          "scene_color_tex": scene_color, "scene_depth_tex": scene_depth,
                          "fluid_color": spy.float3(color[0], color[1], color[2]), "fluid_ior": s["ior"],
                          "fluid_reflection_strength": s["reflection_strength"], "fluid_refraction_strength": s["refraction_strength"],
                          "fluid_absorption_strength": s["absorption_strength"]})
            pass_.draw({"vertex_count": 3})
        encoder.set_texture_state(self.color, spy.ResourceState.shader_resource)
        return self.color

    def close(self):
        self.drop_targets()
        self.depth_pipeline = self.thickness_pipeline = self.smooth_pipeline = self.composite_pipeline = None
