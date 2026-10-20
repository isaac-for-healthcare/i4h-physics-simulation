# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke billboards for the Slang renderer: one draw per smoke system inside the scene pass.

Port of omnisurg/rendering/slang.py:6327-6399 (``draw_smoke_particles``) and ``omnisurg_smoke.slang``: the
two vec4 arrays of the drawn env are copied into shared buffers in ``publish()``; ``draw()`` renders
``count * 6`` vertices alpha-blended with depth test on and depth write off (no sorting, as omnisurg) in a
final pass over the fluid composite, where the fluid's smoothed eye depth hides smoke behind the liquid
(omnisurg drew it before the fluid, which then refracted smoke in front of it). The sprite texture is the
system's ``texture`` (omnisurg's ``smoke1.png``); without one a white pixel.
"""

from __future__ import annotations

import warp as wp
from i4h_standalone.render import CAMERA
from i4h_standalone.render.snapshots import copy


class SmokeDraw:
    def __init__(self, renderer, smoke):
        r, spy = renderer, renderer.spy
        self.smoke = smoke
        self.count = int(smoke.count)
        self.buffers = [r.shared(self.count, wp.vec4f, 16, f"smoke-{name}", usage=spy.BufferUsage.shader_resource)
                        for name in ("position-age", "velocity-lifetime")]
        self.texture = r._texture(smoke.texture, True) if smoke.texture else r._default("white")
        blended = spy.ColorTargetDesc()  # omnisurg _alpha_blended_color_target
        blended.format = r.scene_format
        blended.enable_blend = True
        blended.color.src_factor = spy.BlendFactor.src_alpha
        blended.color.dst_factor = spy.BlendFactor.inv_src_alpha
        blended.color.op = spy.BlendOp.add
        blended.alpha.src_factor = spy.BlendFactor.one
        blended.alpha.dst_factor = spy.BlendFactor.inv_src_alpha
        blended.alpha.op = spy.BlendOp.add
        program = r.device.load_program("smoke.slang", ["vertex_main", "fragment_main"])
        self.pipeline = r.device.create_render_pipeline(
            program=program, input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list, targets=[blended],
            depth_stencil={"format": r.depth_format, "depth_test_enable": True, "depth_write_enable": False,
                           "depth_func": spy.ComparisonFunc.less},
            rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-smoke")

    def publish(self, snap, env: int, stream=None):
        E = self.smoke.scene.envs
        for shared, src in zip(self.buffers, snap.smoke[self.smoke]):
            copy(shared.array, src.reshape((E, self.count))[env], stream)

    def draw(self, pass_, renderer, fluid_depth, width: int, height: int):
        """``fluid_depth``: the fluid's smoothed eye-depth texture, or None without pools."""
        spy = renderer.spy
        pass_.set_render_state(dict(renderer._render_state))
        cursor = spy.ShaderCursor(pass_.bind_pipeline(self.pipeline))
        for name in CAMERA:
            setattr(cursor, name, renderer._frame[name])
        cursor.has_fluid = int(fluid_depth is not None)
        cursor.fluid_depth_tex = fluid_depth if fluid_depth is not None else renderer._default("black")
        cursor.output_size = spy.float2(float(max(1, width)), float(max(1, height)))
        cursor.mesh_color = spy.float4(*self.smoke.start_color)
        cursor.end_color = spy.float4(*self.smoke.end_color)
        cursor.start_size = self.smoke.start_size
        cursor.end_size = self.smoke.end_size
        cursor.diffuse_tex = self.texture
        cursor.material_sampler = renderer.sampler
        cursor.smoke_position_age, cursor.smoke_velocity_lifetime = (b.buffer for b in self.buffers)
        cursor.smoke_particle_count = self.count
        pass_.draw({"vertex_count": 6 * self.count})

    def close(self):
        for shared in self.buffers:
            shared.close()
        self.pipeline = None
