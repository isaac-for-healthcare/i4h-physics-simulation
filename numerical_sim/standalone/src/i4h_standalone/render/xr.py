# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The XR renderer (``render.backend: xr``): the scene in stereo in a headset, with the app's renderer interface
(``publish``, ``draw``, ``is_running``, ``close``, ``key_handlers``).

``xr.session.XrSession`` owns OpenXR and, once a headset connects, the Slang Vulkan device. On that device this renderer
builds an offscreen ``SlangRenderer`` (the same draws, materials and snapshot publishing as the window) and, per frame,
renders each eye with ``SlangRenderer.encode_scene``. The camera of an eye is its XR pose through the ``PoseAnchor``,
which maps the first head pose onto the case's camera, plus its off-axis frustum. The eye's HDR colour then goes into
the swapchain image through ``shaders/xr_resolve.slang`` (the renderer's exposure and tonemap). The desktop post chain,
fluids, smoke and UI are not drawn in the headset. A reconnect rebuilds the scene renderer on the new device.

``draw()`` is paced by ``xrWaitFrame`` (the headset's display rate), so the app skips its own frame pacer
(``self_paced``). An optional mirror window shows one eye and takes the app's command keys; ``headless`` has none.
Case keys, under ``render.xr``: ``world_units_per_meter`` (required), ``runtime`` (``xr.runtime``),
``mirror: left|right|off``, ``headless``, ``debug_layers``.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext

import warp as wp

XR_KEYS = {"world_units_per_meter", "runtime", "mirror", "headless", "debug_layers", "inputs", "max_fps"}  # inputs: the app's (cli)
SKIPPED = ("size", "window", "ui", "gbuffer", "device")  # render keys the XR path sets itself


class XrRenderer:
    self_paced = True  # xrWaitFrame paces draw(): the app's frame pacer stays out

    def __init__(self, sim, *, xr: dict | None = None, env: int = 0, **render_cfg):
        import slangpy as spy
        import xr as xr_module
        from i4h_standalone.render.interop import cuda_driver
        from i4h_standalone.render.slang import SHADER_DIR
        from i4h_standalone.xr.runtime import XrRuntime
        from i4h_standalone.xr.session import XrSession

        cfg = dict(xr or {})
        unknown = set(cfg) - XR_KEYS
        if unknown:
            raise ValueError(f"render.xr: unknown keys {sorted(unknown)}; known {sorted(XR_KEYS)}")
        if "world_units_per_meter" not in cfg:
            raise ValueError("render.xr.world_units_per_meter is required (scene units per metre of head motion)")
        self.world_units_per_meter = float(cfg["world_units_per_meter"])
        self.mirror = str(cfg.get("mirror", "left"))
        if self.mirror not in ("left", "right", "off"):
            raise ValueError(f"render.xr.mirror must be left, right or off, got {self.mirror!r}")
        self.headless = bool(cfg.get("headless", False))
        self.max_fps = float(cfg.get("max_fps", 0.0) or 0.0)  # 0: the headset's rate; else a cap (the runtime reprojects)
        self._next_frame = 0.0
        self.spy, self.sim, self.env = spy, sim, int(env)
        self.render_cfg = {k: v for k, v in render_cfg.items() if k not in SKIPPED}
        self.key_handlers: list = []
        self.stream = None  # the app's render stream (threaded_loop); the snapshot copies run on it
        self.lock = threading.RLock()
        self.scene_renderer = None
        self.anchor = None
        self.window = self.surface = None
        self._snap = None
        self._views: dict = {}
        self._closed = False
        self.frames = 0
        self.skipped: dict[str, int] = {}  # why frames were not rendered (the periodic XR status line)
        self._status_at = 0.0
        self.runtime = XrRuntime.from_cfg(cfg.get("runtime"))
        self.runtime.start()  # before the session: selects the OpenXR runtime (XR_RUNTIME_JSON)
        try:
            self.session = XrSession(xr_module, spy, cuda_driver(), sim.scene.device, runtime_mode=self.runtime.mode,
                                     shader_dir=SHADER_DIR, on_device=self._on_device, on_release=self._on_release,
                                     debug_layers=bool(cfg.get("debug_layers", False)))
            self.session.start()
        except BaseException:
            self.runtime.close()
            raise
        print("XR: waiting for the headset to connect")

    # session callbacks (render thread) -------------------------------------------------------------------------------

    def _on_device(self, session) -> None:
        with self.lock if self.lock is not None else nullcontext():
            from i4h_standalone.render.slang import SlangRenderer
            from i4h_standalone.xr.anchor import PoseAnchor

            spy = self.spy
            eye = session.swapchains[0]
            self.scene_renderer = SlangRenderer(self.sim, device=session.device, window=False, ui=False,
                                                size=(eye.width, eye.height), env=self.env, **self.render_cfg)
            self.scene_renderer.stream, self.scene_renderer.lock = self.stream, self.lock
            self.scene_renderer._ensure_targets(eye.width, eye.height)  # the eye-sized HDR colour, depth and render state
            self.anchor = PoseAnchor.from_camera(self.scene_renderer.camera, self.world_units_per_meter)
            program = session.device.load_program("xr_resolve.slang", ["vertex_main", "fragment_main"])
            pipeline = lambda fmt: session.device.create_render_pipeline(  # noqa: E731
                program=program, input_layout=None, primitive_topology=spy.PrimitiveTopology.triangle_list,
                targets=[{"format": fmt}], rasterizer={"cull_mode": spy.CullMode.none}, label="i4h-xr-resolve")
            self._pipelines = {session.color_format: pipeline(session.color_format)}
            if self.mirror != "off" and not self.headless:
                aspect = eye.width / max(eye.height, 1)
                height = 720
                self.window = spy.Window(width=int(round(height * aspect)), height=height, title="i4h-sim XR mirror", resizable=False)
                self.window.on_keyboard_event = lambda event: [h(event) for h in self.key_handlers]
                self.surface = session.device.create_surface(self.window)
                self.surface.configure(width=self.window.width, height=self.window.height, vsync=False)
                fmt = self.surface.info.preferred_format
                self._pipelines.setdefault(fmt, pipeline(fmt))
            if self._snap is not None:
                self.scene_renderer.publish(self._snap)

    def _publish_hands(self, frame) -> None:
        """The controllers in world coordinates for the ``openxr`` input sources (xr/hands.py), once anchored."""
        from i4h_standalone.xr import hands

        anchored = self.anchor is not None and self.anchor.anchored
        for hand, s in frame.samples.items():
            if not (anchored and s.get("active", False)):
                hands.publish(hand, {"active": False})
                continue
            position, rotation = self.anchor.transform(s["position"], s["rotation"])
            hands.publish(hand, {**s, "position": position, "rotation": rotation})

    def _on_release(self, session) -> None:
        with self.lock if self.lock is not None else nullcontext():
            from i4h_standalone.xr import hands

            del session
            hands.clear()
            self._views.clear()
            self._pipelines = {}
            if self.scene_renderer is not None:
                self.scene_renderer.close()
                self.scene_renderer = None
            self.surface = None
            if self.window is not None:
                self.window.close()
                self.window = None
            self.anchor = None

        # the app's renderer interface -------------------------------------------------------------------------------

    def publish(self, snap=None) -> None:
        self._snap = snap
        if self.scene_renderer is not None:
            self.scene_renderer.stream, self.scene_renderer.lock = self.stream, self.lock
            self.scene_renderer.publish(snap)

    def is_running(self) -> bool:
        if self._closed or not self.session.alive:
            return False
        return not (self.window is not None and self.window.should_close())

    def set_status(self, text: str) -> None:
        if self.window is not None:
            self.window.title = f"i4h-sim XR mirror  {text}"

    def read_frame(self):
        return None

    def _view(self, texture):
        cached = self._views.get(id(texture))
        if cached is None or cached[0] is not texture:
            cached = self._views[id(texture)] = (texture, texture.create_view({}))
        return cached[1]

    def _resolve(self, encoder, target, fmt, width: int, height: int) -> None:
        spy, sr = self.spy, self.scene_renderer
        with encoder.begin_render_pass({"color_attachments": [{"view": self._view(target), "load_op": spy.LoadOp.dont_care,
                                                               "store_op": spy.StoreOp.store}]}) as pass_:
            pass_.set_render_state({"viewports": [spy.Viewport.from_size(width, height)],
                                    "scissor_rects": [spy.ScissorRect.from_size(width, height)]})
            cursor = spy.ShaderCursor(pass_.bind_pipeline(self._pipelines[fmt]))
            cursor.write({"source_tex": sr.scene_color, "post_sampler": sr.sampler, "exposure": float(sr.exposure),
                          "tonemap_operator": int(sr.tonemap), "manual_srgb_encode": int("srgb" not in str(fmt).lower())})
            pass_.draw({"vertex_count": 3})

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _report(self) -> None:
        """Every 2 s: the session state, the frames rendered and why the others were skipped (the console)."""
        now = time.perf_counter()
        if now - self._status_at < 2.0:
            return
        self._status_at = now
        s = self.session
        state = str(getattr(s.state, "name", s.state)).lower() if s.state is not None else "none"
        skipped = ", ".join(f"{k} {v}" for k, v in sorted(self.skipped.items())) or "none"
        print(f"XR: session {'connected' if s.session is not None else 'waiting for headset'}, state {state}, "
              f"running {s.running}, rendered {self.frames} frames, skipped: {skipped}", flush=True)

    def draw(self) -> bool:
        """One XR frame: wait (paced by the headset), render both eyes, submit, end. Returns False once closed."""
        if not self.is_running():
            return False
        if self.window is not None:
            self.window.process_events()
        self._report()
        if self.max_fps > 0.0 and self.session.running:  # leave the physics thread the GIL and the GPU between frames
            wait = self._next_frame - time.perf_counter()
            if wait > 0.0:
                time.sleep(wait)
            self._next_frame = max(self._next_frame + 1.0 / self.max_fps, time.perf_counter())
        self.session.resource_lock = self.lock
        frame = self.session.begin_frame()
        if frame is None:  # no headset yet, or the session is not running: poll again shortly
            self._skip("no headset" if self.session.session is None else "session not running")
            time.sleep(0.02)
            return self.session.alive
        sr = self.scene_renderer
        if sr is not None and self.anchor is not None and frame.head is not None:
            self.anchor.observe_head(*frame.head, valid=True)
        self._publish_hands(frame)
        from i4h_standalone.xr import hands

        for hand in hands.HANDS:  # the servo's vibration commands, applied with this frame (end_frame)
            self.session.set_haptic(hand, hands.haptic(hand), duration=2.0 * frame.period)
        reason = ("runtime says do not render" if not frame.should_render and frame.views else
                  "eye views invalid" if not frame.views else
                  "no scene renderer" if sr is None else
                  "no valid head pose yet" if self.anchor is None or not self.anchor.anchored else None)
        if reason is not None:
            self._skip(reason)
            self.session.end_frame(None)
            return True
        eye = self.session.swapchains[0]
        width, height = eye.width, eye.height
        mirror_target = None

        def encode(encoder, textures) -> bool:
            nonlocal mirror_target
            spy = self.spy
            for index, (position, rotation, fov) in enumerate(frame.views):
                basis, view_projection = self.anchor.eye_camera(position, rotation, fov, sr.camera.near, sr.camera.far)
                uniforms = sr._frame_uniforms(width, height, basis, view_projection)
                sr.encode_scene(encoder, sr.scene_view, sr.depth_view, uniforms, view=1 + index)
                encoder.set_texture_state(sr.scene_color, spy.ResourceState.shader_resource)
                self._resolve(encoder, textures[index], self.session.color_format, width, height)
                if self.surface is not None and index == (0 if self.mirror == "left" else 1):
                    try:
                        mirror_target = self.surface.acquire_next_image()
                    except RuntimeError:
                        mirror_target = None
                    if mirror_target:
                        self._resolve(encoder, mirror_target, self.surface.info.preferred_format,
                                      int(mirror_target.width), int(mirror_target.height))
            sr.frame_index += 1  # the next frame's uniforms go into the other root objects
            return True

        stream = self.stream or wp.get_stream(self.sim.scene.device)
        self.session.end_frame(encode, cuda_stream=self.spy.NativeHandle.from_cuda_stream(int(stream.cuda_stream)))
        if mirror_target and self.surface is not None:
            try:
                self.surface.present()
            except RuntimeError:
                pass
        self.frames += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        from i4h_standalone.xr import hands

        try:
            self.session.resource_lock = self.lock
            self.session.close()  # on_release drops the scene renderer before the device goes; the controllers stop
        finally:
            hands.clear()
            self.runtime.close()
