# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU side of dataset capture: one compute dispatch packs the display frame and the G-buffer into a Warp-visible
staging slot; the copy to pinned host memory rides the render stream and an event tells the host when it landed.

Port of the canonical omnisurg ``rendering/recording.py`` capture path (330-360, 462-490: a pack compute into
``uint3`` frames, 12 bytes per pixel). Edits: staging slots are shared buffers (``render/interop.py``) so Warp copies
them asynchronously into pinned host arrays (slangpy's ``Buffer.to_numpy`` would wait for the whole device); one
slot per frame instead of chunks of eight; the frame size is the renderer's; rgb comes from the display texture
``ldr`` (post chain, noise, sRGB) and depth, labels and normals from the G-buffer (``render/gbuffer.py``). The host
side (queue, drop policy, ffmpeg, manifest) is ``app/capture.py``.
"""

from __future__ import annotations

import warp as wp

BYTES_PER_PIXEL = 12


class FrameCapture:
    def __init__(self, renderer, slots: int, width: int, height: int, depth_max: float):
        """Allocate ``slots`` staging buffers of ``width x height`` pixels: before the physics thread starts."""
        r, spy, dev = renderer, renderer.spy, renderer.scene.device
        if r.gbuffer is None:
            raise ValueError("dataset capture needs the G-buffer (render.gbuffer: true)")
        self.renderer, self.spy = r, spy
        self.width, self.height, self.depth_max = int(width), int(height), float(depth_max)
        n = self.width * self.height * 3
        self.shared = [r.shared(n, wp.uint32, 4, f"i4h-capture-{i}", usage=spy.BufferUsage.unordered_access) for i in range(int(slots))]
        self.host = [wp.zeros(n, dtype=wp.uint32, device="cpu", pinned=True) for _ in range(int(slots))]
        self.events = [wp.Event(dev) for _ in range(int(slots))]
        self.pipeline = r.device.create_compute_pipeline(r.device.load_program("capture.slang", ["pack_frame"]), label="i4h-capture-pack")
        self.pending: int | None = None  # the slot this draw fills
        self.encoded: int | None = None  # the slot the last dispatch filled (the host tells a cancelled capture from a done one)

    def encode(self, encoder, width: int, height: int) -> None:
        """Pack the frame into the pending slot (after the post chain and the present into ``ldr``)."""
        if self.pending is None:
            return
        if (width, height) != (self.width, self.height):  # a resize landed after the frame was armed: cancel it (counted by the host)
            self.pending = None
            return
        self.encoded = self.pending
        r, spy, p = self.renderer, self.spy, self.renderer.post
        buffer = self.shared[self.pending].buffer
        encoder.set_buffer_state(buffer, spy.ResourceState.unordered_access)
        with encoder.begin_compute_pass() as pass_:
            cursor = spy.ShaderCursor(pass_.bind_pipeline(self.pipeline))
            cursor.write({"capture_rgb": r.ldr, "capture_normal_depth": r.gbuffer.normal_depth, "capture_labels": r.gbuffer.label,
                          "capture_frames": buffer, "width": self.width, "height": self.height, "depth_max": self.depth_max,
                          "camera_aspect": float(width) / max(float(height), 1.0),
                          "dataset_distortion_enabled": int(p.lens_distortion_enabled), "dataset_distortion_strength": float(p.lens_distortion_strength),
                          "dataset_distortion_zoom": float(p.lens_distortion_zoom), "dataset_scope_enabled": int(p.scope_enabled),
                          "dataset_scope_radius": float(p.scope_radius)})
            pass_.dispatch([self.width, self.height, 1])

    def after_submit(self, stream) -> None:
        """Enqueue the device-to-host copy of the pending slot on the render stream and record its event."""
        if self.pending is None:
            return
        slot, self.pending = self.pending, None
        wp.copy(self.host[slot], self.shared[slot].array, stream=stream)
        stream.record_event(self.events[slot])

    def completed(self, slot: int) -> bool:
        """Whether the copy into ``host[slot]`` has landed (query events only under the ring lock)."""
        return self.events[slot].is_complete

    def close(self) -> None:
        for s in self.shared:
            s.close()
        self.shared, self.host, self.events = [], [], []
        self.pipeline = None
