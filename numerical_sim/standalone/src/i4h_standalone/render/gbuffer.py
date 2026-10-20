# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G-buffer pass of the Slang renderer (``render.gbuffer: true``): camera-space normals with eye depth and a per-object
label, drawn every frame from the same vertex streams as the RGB pass, for dataset capture.

Port of the canonical omnisurg ``rendering/recording.py`` MRT pass (266-490: the draws registered during the RGB
frame are replayed into ``normal_depth`` + ``label`` with their own ``object_id``, then the screen-space fluid adds
its label where its smoothed depth is nearest). Edits: no arming state (the pass runs while the G-buffer exists;
the present source never changes); the labels are a fixed table built at construction from the draw list
(``body/part``, static mesh name, ``body/skin``, instrument piece, ``rod/body``, clip system, ``body/fluid``; 0 is
the background) instead of ids assigned as draws are registered; its own depth target (identical geometry gives
identical depth; the RGB depth is being read by the post chain); the frame resolution instead of 1024x576.
"""

from __future__ import annotations

import numpy as np
from i4h_standalone.render import CAMERA


class GBuffer:
    def __init__(self, renderer):
        r, spy, dev = renderer, renderer.spy, renderer.device
        self.spy, self.device, self.renderer = spy, dev, r
        mesh = dev.load_program("gbuffer.slang", ["smooth_vertex", "dataset_fragment"])
        def element(name, slot):
            return {"semantic_name": name, "semantic_index": 0, "format": spy.Format.rgb32_float, "buffer_slot_index": slot}
        layout = dev.create_input_layout(input_elements=[element("POSITION", 0), element("NORMAL", 1)], vertex_streams=[{"stride": 12}, {"stride": 12}])
        self.formats = [spy.Format.rgba32_float, spy.Format.r32_uint]
        targets = [{"format": f} for f in self.formats]
        depth = {"format": r.depth_format, "depth_test_enable": True, "depth_write_enable": True, "depth_func": spy.ComparisonFunc.less}
        self.pipelines = {cull: dev.create_render_pipeline(program=mesh, input_layout=layout, primitive_topology=spy.PrimitiveTopology.triangle_list,
                                                            targets=targets, depth_stencil=depth, rasterizer={"cull_mode": mode},
                                                            label=f"i4h-gbuffer-{cull}")
                          for cull, mode in (("none", spy.CullMode.none), ("back", spy.CullMode.back))}
        self.roots = {cull: [dev.create_root_shader_object(mesh) for _ in range(2)] for cull in self.pipelines}  # per frame parity
        self.fluid_pipeline = self.pool_pipeline = self.pool_root = None
        if r.fluid_passes is not None:
            fluid = dev.load_program("gbuffer_fluid.slang", ["fullscreen_vertex_main", "dataset_fluid_fragment"])
            self.fluid_pipeline = dev.create_render_pipeline(program=fluid, input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list,
                                                             targets=targets, rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-gbuffer-fluid")
            pool = dev.load_program("gbuffer_fluid.slang", ["depth_vertex_main", "fluid_label_fragment"])
            self.pool_pipeline = dev.create_render_pipeline(  # the nearest pool's label per pixel against the fluid depth the RGB passes wrote
                program=pool, input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list, targets=[{"format": self.formats[1]}],
                depth_stencil={"format": r.depth_format, "depth_test_enable": True, "depth_write_enable": False,
                               "depth_func": spy.ComparisonFunc.less_equal},
                rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-gbuffer-pool")
            self.pool_root = [dev.create_root_shader_object(pool) for _ in range(2)]
        self.size = None
        self.normal_depth = self.label = self.depth = self.pool_label = None
        self.views: dict = {}

    def ensure_targets(self, width: int, height: int):
        if self.size == (width, height):
            return
        spy, dev, r = self.spy, self.device, self.renderer
        usage = spy.TextureUsage.render_target | spy.TextureUsage.shader_resource
        self.normal_depth = dev.create_texture(format=self.formats[0], width=width, height=height, usage=usage, label="i4h-gbuffer-normal-depth")
        self.label = dev.create_texture(format=self.formats[1], width=width, height=height, usage=usage, label="i4h-gbuffer-label")
        self.depth = dev.create_texture(format=r.depth_format, width=width, height=height, usage=spy.TextureUsage.depth_stencil, label="i4h-gbuffer-depth")
        self.pool_label = dev.create_texture(format=self.formats[1], width=width, height=height, usage=usage, label="i4h-gbuffer-pool-label")
        self.views = {name: getattr(self, name).create_view({}) for name in ("normal_depth", "label", "depth", "pool_label")}
        self.size = (width, height)

    def drop_targets(self):
        self.size = None
        self.normal_depth = self.label = self.depth = self.pool_label = None
        self.views = {}

    def encode(self, encoder, width: int, height: int):
        """The RGB draw list (bodies, statics, skins, pieces, rods, clips; not the debug spheres or markers) into the MRT,
        then the fluid label pass. Call after the fluid passes (their smoothed depth) with the scene depth readable."""
        r, spy = self.renderer, self.spy
        camera = {k: r._frame[k] for k in CAMERA}
        parity = r.frame_index & 1
        roots = {cull: pair[parity] for cull, pair in self.roots.items()}
        pool_root = self.pool_root[parity] if self.pool_root is not None else None
        for root in list(roots.values()) + ([pool_root] if pool_root is not None else []):
            spy.ShaderCursor(root).write(camera)
        for name in ("normal_depth", "label"):
            encoder.set_texture_state(getattr(self, name), spy.ResourceState.render_target)
        attachments = [{"view": self.views["normal_depth"], "clear_value": [0.0, 0.0, 0.0, 0.0], "load_op": spy.LoadOp.clear, "store_op": spy.StoreOp.store},
                       {"view": self.views["label"], "clear_value": [0, 0, 0, 0], "load_op": spy.LoadOp.clear, "store_op": spy.StoreOp.store}]
        with encoder.begin_render_pass({"color_attachments": attachments,
                                        "depth_stencil_attachment": {"view": self.views["depth"], "depth_load_op": spy.LoadOp.clear,
                                                                     "depth_store_op": spy.StoreOp.store, "depth_clear_value": 1.0}}) as pass_:
            def draw(streams, index_buffer, start, count, label, cull="none"):
                pass_.set_render_state({"vertex_buffers": streams[:2], "index_buffer": index_buffer, "index_format": spy.IndexFormat.uint32,
                                        **r._render_state})
                pass_.bind_pipeline(self.pipelines[cull], roots[cull])
                spy.ShaderCursor(roots[cull]).object_id = int(label)
                pass_.draw_indexed({"vertex_count": int(count), "start_index_location": int(start)})

            for d in r.bodies:
                for start, count, _, _, cull, label in d.parts:
                    if count:
                        draw(d.streams, d.index_buffer, start, count, label, cull)
            for s in r.statics:
                draw(s.streams, s.index_buffer, 0, s.index_count, s.label, s.cull)
            for d in r.skinned:
                draw(d.streams, d.index_buffer, 0, d.index_count, d.label, d.cull)
            for d in r.flats:
                draw(d.streams, d.index_buffer, 0, d.index_count, d.label)
        if self.fluid_pipeline is not None and r.fluids:
            fp = r.fluid_passes
            encoder.set_texture_state(self.pool_label, spy.ResourceState.render_target)
            with encoder.begin_render_pass({  # the nearest pool per pixel: sprites depth-tested against the fluid depth (write off)
                "color_attachments": [{"view": self.views["pool_label"], "clear_value": [0, 0, 0, 0], "load_op": spy.LoadOp.clear, "store_op": spy.StoreOp.store}],
                "depth_stencil_attachment": {"view": fp.views["depth"], "depth_load_op": spy.LoadOp.load, "depth_store_op": spy.StoreOp.store},
            }) as pass_:
                pass_.set_render_state(r._render_state)
                for d in r.fluids:
                    pass_.bind_pipeline(self.pool_pipeline, pool_root)
                    cursor = spy.ShaderCursor(pool_root)
                    cursor.mesh_color = spy.float4(d.color[0], d.color[1], d.color[2], 1.0)
                    cursor.fluid_positions = d.pos.buffer
                    cursor.fluid_flags = d.flags.buffer
                    cursor.fluid_count = int(d.count)
                    cursor.fluid_radius = d.radius
                    cursor.fluid_thickness_scale = fp.settings["thickness_scale"]
                    cursor.object_id = int(d.label)
                    pass_.draw({"vertex_count": 6, "instance_count": int(d.count)})
            encoder.set_texture_state(self.pool_label, spy.ResourceState.shader_resource)
            attachments = [{"view": self.views["normal_depth"], "load_op": spy.LoadOp.load, "store_op": spy.StoreOp.store},
                           {"view": self.views["label"], "load_op": spy.LoadOp.load, "store_op": spy.StoreOp.store}]
            with encoder.begin_render_pass({"color_attachments": attachments}) as pass_:
                pass_.set_render_state(r._render_state)
                cursor = spy.ShaderCursor(pass_.bind_pipeline(self.fluid_pipeline))
                cursor.write({**camera, "fluid_depth_tex": fp.smooth_depth, "scene_depth_tex": r.depth, "post_sampler": r.sampler,
                              "pool_label_tex": self.pool_label, "output_size": spy.float2(float(max(1, width)), float(max(1, height))),
                              "object_id": 0})
                pass_.draw({"vertex_count": 3})
        for name in ("normal_depth", "label"):
            encoder.set_texture_state(getattr(self, name), spy.ResourceState.shader_resource)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """``(normal_depth (H, W, 4) float32, labels (H, W) uint32)`` of the last frame; syncs the device (tests)."""
        self.device.wait()
        if self.normal_depth is None:
            raise RuntimeError("no frame drawn yet")
        nd = np.asarray(self.normal_depth.to_numpy(), dtype=np.float32)
        labels = np.asarray(self.label.to_numpy()).reshape(nd.shape[0], nd.shape[1])
        return nd, labels.astype(np.uint32)

    def close(self):
        self.drop_targets()
        self.pipelines = self.roots = {}
        self.fluid_pipeline = self.pool_pipeline = self.pool_root = None
