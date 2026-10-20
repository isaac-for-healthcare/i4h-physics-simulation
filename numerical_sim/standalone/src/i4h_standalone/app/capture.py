# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset capture, host side: which frames to keep, the bounded hand-off from the render thread to a writer thread,
four lossless ffmpeg streams (rgb, depth, segmentation, normals) and the manifest.

Ported from the canonical omnisurg ``rendering/recording.py``: ``unpack_frame`` 30-46 (the uint3 layout into video
planes; edit: depth kept at 16 bits as gray16le instead of an 8-bit preview), ``label_color`` 49-51, the encoder
options 54-85 and ``check_encoder`` 88-129, the slot clock 345-353 (one frame per ``1/fps`` of simulation time), the
hold-fill of missed slots 536-554 and the manifest 561-597. Edits: the ffmpeg processes are opened at start and fed
from a writer thread while the run goes on (omnisurg read every frame back at exit and encoded then); the render
thread never blocks on the GPU or on a pipe: it takes a free staging slot or counts a drop; the frame size is the
renderer's; per-frame samples go to ``frames.jsonl`` so a crash mid-run leaves a usable file; the flag is
``--capture`` (``--record`` is the input recorder).
"""

from __future__ import annotations

import json
import logging
import math
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CAPTURE_FPS = 10.0

ENCODERS = ("ffv1", "h264_nvenc", "hevc_nvenc")  # omnisurg/config/recording.py:7
STREAMS = (("rgb", "rgba"), ("depth", "gray16le"), ("segmentation", "rgba"), ("normals", "rgba"))  # name, ffmpeg input pixel format


@dataclass
class CaptureSettings:
    directory: str
    fps: float = DEFAULT_CAPTURE_FPS  # one video frame per 1/fps of simulation time
    encoder: str = "ffv1"
    max_frames: int = 0  # 0: unbounded
    depth_max: float = 5.0  # scene units mapped onto the 16-bit depth
    slots: int = 4  # staging frames in flight between the GPU and the writer

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("capture fps must be finite and positive")


def encoder_options(encoder: str, output_format: str, gpu: int = 0) -> list[str]:
    """recording.py:54-85."""
    if encoder == "ffv1":
        return ["-c:v", "ffv1", "-level", "3", "-threads", "2", "-pix_fmt", output_format]
    if encoder not in ENCODERS:
        raise ValueError(f"unsupported capture encoder {encoder!r}; choose from {ENCODERS}")
    return ["-c:v", encoder, "-gpu", str(gpu), "-preset", "p4", "-tune", "lossless", "-pix_fmt", "gbrp", "-colorspace", "rgb",
            "-color_range", "pc"]


def check_ffmpeg(encoder: str, width: int, height: int, gpu: int = 0) -> None:
    """ffmpeg on PATH, and for NVENC a one-frame dry run (recording.py:88-129)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("--capture needs ffmpeg on PATH (apt install ffmpeg)")
    if encoder == "ffv1":
        return
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
               "-i", "pipe:0", *encoder_options(encoder, "gbrp", gpu), "-frames:v", "1", "-f", "null", "-"]
    try:
        result = subprocess.run(command, input=bytes(width * height * 3), capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"capture encoder {encoder} startup check timed out") from exc
    if result.returncode:
        raise RuntimeError(f"capture encoder {encoder} unavailable on GPU {gpu}:\n" + result.stderr.decode(errors="replace"))


LABEL_MULTIPLIER = 0x9E3779  # odd: distinct 24-bit colours for every 16-bit label id, 0 stays black


def label_color(label: int) -> list[int]:
    color = (int(label) * LABEL_MULTIPLIER) & 0xFFFFFF
    return [color & 255, (color >> 8) & 255, color >> 16]


def unpack_frame(packed, width: int, height: int) -> tuple[bytes, bytes, bytes, bytes]:
    """The GPU layout (uint3 per pixel: rgb8 | depth16 + label16 | normal rgb8) into the four planes, in STREAMS order."""
    packed = np.asarray(packed, dtype="<u4").reshape(height, width, 3)
    rgb = np.ascontiguousarray(packed[..., 0]).view(np.uint8).reshape(height, width, 4)
    depth = (packed[..., 1] & 0xFFFF).astype("<u2")
    labels = packed[..., 1] >> 16
    segmentation = np.ascontiguousarray(((labels * LABEL_MULTIPLIER) & 0xFFFFFF) | 0xFF000000).view(np.uint8).reshape(height, width, 4)
    normals = np.ascontiguousarray(packed[..., 2]).view(np.uint8).reshape(height, width, 4)
    return rgb.tobytes(), depth.tobytes(), segmentation.tobytes(), normals.tobytes()


class SlotClock:
    """One video frame per ``1/fps`` of simulation time (recording.py:345-353); a time going backwards (a reset)
    continues from the last slot."""

    def __init__(self, fps: float):
        self.fps = float(fps)
        self.start = self.previous = None
        self.last = -1
        self.offset = 0.0

    def slot(self, t: float) -> int | None:
        t = float(t)
        if self.start is None:
            self.start = self.previous = t
        if t < self.previous - 1e-9:  # the clock restarted (a reset): continue at the next slot
            self.offset = self.start - t + (self.last + 1) / self.fps
        self.previous = t
        s = math.floor((t + self.offset - self.start) * self.fps + 1e-6)
        if s <= self.last:
            return None
        self.last = s
        return s


class ListSink:
    """Keeps the planes in memory (tests)."""

    def __init__(self):
        self.frames: dict[str, list] = {name: [] for name, _ in STREAMS}
        self.opened = None

    def open(self, width, height, fps):
        self.opened = (width, height, fps)

    def write(self, name, data: bytes):
        self.frames[name].append(data)

    def close(self) -> dict:
        return {name: len(v) for name, v in self.frames.items()}


class FfmpegSink:
    """Four ffmpeg processes fed through stdin for the life of the capture; depth is always ffv1 gray16le
    (NVENC has no 16-bit grey), the others use the chosen encoder."""

    def __init__(self, directory: Path, encoder: str, gpu: int = 0):
        self.directory, self.encoder, self.gpu = Path(directory), encoder, gpu
        self.processes: dict[str, subprocess.Popen] = {}
        self.logs: dict[str, object] = {}

    def open(self, width, height, fps):
        for name, pixel_format in STREAMS:
            if name == "depth":
                options = ["-c:v", "ffv1", "-level", "3", "-threads", "2", "-pix_fmt", "gray16le"]
            else:
                options = encoder_options(self.encoder, "bgr0" if self.encoder == "ffv1" else "gbrp", self.gpu)
            log = open(self.directory / f"{name}.log", "wb")
            self.logs[name] = log
            self.processes[name] = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", pixel_format,
                 "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an", *options,
                 str(self.directory / f"{name}.mkv")], stdin=subprocess.PIPE, stderr=log)

    def write(self, name, data: bytes):
        self.processes[name].stdin.write(data)

    def close(self) -> dict:
        failed = []
        for name, process in self.processes.items():
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            if process.wait() != 0:
                failed.append(name)
        for log in self.logs.values():
            log.close()
        if failed:
            messages = [(self.directory / f"{name}.log").read_text(errors="replace") for name in failed]
            raise RuntimeError(f"ffmpeg failed for {', '.join(failed)}:\n" + "\n".join(messages))
        return {name: str(self.directory / f"{name}.mkv") for name, _ in STREAMS}


FREE, ARMED, INFLIGHT = 0, 1, 2  # a staging slot's state


class Capture:
    """Render-thread side: pick the frames, hand landed staging slots to the writer, own the manifest.

    Each staging slot is FREE, ARMED (taken for the next draw) or INFLIGHT (packed, its host copy landing). One slot is
    armed at a time; a draw that returned before rendering (a lost surface, a minimise) leaves it armed and the next
    frame re-arms it for the newer snapshot, so its metadata always describes the frame that was drawn."""

    def __init__(self, renderer, settings: CaptureSettings, *, case: str, envs: int, sink=None, gpu: int = 0):
        self.renderer, self.settings, self.case, self.envs = renderer, settings, str(case), int(envs)
        if settings.encoder not in ENCODERS:
            raise ValueError(f"unsupported capture encoder {settings.encoder!r}; choose from {ENCODERS}")
        self.directory = Path(settings.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(renderer.width), int(renderer.height)
        if sink is None:
            check_ffmpeg(settings.encoder, self.width, self.height, gpu)
            sink = FfmpegSink(self.directory, settings.encoder, gpu)
        self.sink = sink
        self.gpu = renderer.enable_capture(settings.slots, settings.depth_max)  # allocates: before the worker starts
        self.state = [FREE] * settings.slots
        self.meta: list[dict | None] = [None] * settings.slots
        self.inflight: list[int] = []  # INFLIGHT slots in capture order
        self.armed: int | None = None
        self.free_lock = threading.Lock()  # the writer frees slots; the render thread arms them
        self.queue: queue.Queue = queue.Queue(maxsize=settings.slots)
        self.clock = SlotClock(settings.fps)
        self.dropped = {"queue_full": 0, "resized": 0}
        self.captured = 0
        self.video_frames = 0
        self.error: BaseException | None = None
        self.samples_file = open(self.directory / "frames.jsonl", "w")
        self._write_manifest("recording")
        self.writer = threading.Thread(target=self._write, name="i4h-capture", daemon=True)
        self.writer.start()

    # render thread ------------------------------------------------------------------------------------------------

    def frame(self, snap) -> None:
        """After ``renderer.publish(snap)`` and before ``draw()``: decide whether this draw is captured. The pose-dependent
        metadata is taken in ``drawn()`` after the draw."""
        self.collect()
        if self.error is not None:
            return
        r = self.renderer
        index = self.clock.slot(snap.t)  # first: a dropped frame still advances the video clock
        if (int(r.width), int(r.height)) != (self.width, self.height) or getattr(r, "pending_resize", None) is not None:
            self.dropped["resized"] += 1
            if self.armed is not None:  # a slot armed by a skipped draw: the frame it would describe is gone
                self._release(self.armed)
                self.captured -= 1
                self.gpu.pending = self.armed = None
            return
        if index is None or (self.settings.max_frames and self.captured >= self.settings.max_frames):
            if self.armed is not None:  # the armed slot keeps its video index and describes the snapshot the draw will render
                self.meta[self.armed].update(step=int(snap.step), sim_time=float(snap.t), wall_time=time.time(), env=int(r.env))
                self.gpu.encoded, self.gpu.pending = None, self.armed
            return
        if self.armed is None:  # else: the previous draw did not render; the slot is re-armed for this newer snapshot
            slot = self._take_free()
            if slot is None:
                self.dropped["queue_full"] += 1
                if self.dropped["queue_full"] == 1:
                    logging.getLogger(__name__).warning(
                        "Capture queue full at %g fps, %dx%d; frames are being dropped. "
                        "Reduce --capture-fps or render resolution to reduce capture load.",
                        self.settings.fps, self.width, self.height)
                return
            self.state[slot] = ARMED
            self.armed = slot
            self.captured += 1
        self.meta[self.armed] = {"index": index, "step": int(snap.step), "sim_time": float(snap.t), "wall_time": time.time(), "env": int(r.env)}
        self.gpu.encoded = None  # drawn() tells a cancelled capture (encoded stays None) from a done one
        self.gpu.pending = self.armed

    def drawn(self) -> None:
        """After ``renderer.draw()``: record the pose the frame was drawn with; cancel a capture the draw rejected (a
        resize); leave a slot armed when the draw returned before rendering."""
        if self.armed is None or self.gpu.pending is not None:
            return
        slot, self.armed = self.armed, None
        if self.gpu.encoded != slot:  # cancelled: the frame changed size
            self._release(slot)
            self.captured -= 1
            self.dropped["resized"] += 1
            return
        r = self.renderer
        pos, right, up, forward = r.camera.basis()
        rig = getattr(r, "rig", None)
        mode = getattr(r, "camera_mode", "free")
        self.meta[slot].update({
            "camera": {"position": pos.tolist(), "right": right.tolist(), "up": up.tolist(), "forward": forward.tolist(),
                       "fov": float(r.camera.fov), "near": float(r.camera.near), "far": float(r.camera.far)},
            "camera_mode": mode, "tip": rig.names[rig.index] if rig is not None and mode == "tip" else None,
            "post": {"lens_distortion": bool(r.post.lens_distortion_enabled), "lens_distortion_strength": float(r.post.lens_distortion_strength),
                     "lens_distortion_zoom": float(r.post.lens_distortion_zoom), "scope": bool(r.post.scope_enabled),
                     "scope_radius": float(r.post.scope_radius)}})
        self.state[slot] = INFLIGHT
        self.inflight.append(slot)

    def collect(self) -> None:
        """Hand every landed slot to the writer (events queried under the ring lock); with the writer gone, free them."""
        alive = self.writer.is_alive()
        with self.renderer.lock:
            while self.inflight and self.gpu.completed(self.inflight[0]):
                slot = self.inflight.pop(0)
                if alive:
                    self.queue.put((slot, self.meta[slot]))
                else:
                    self._release(slot)

    def _take_free(self) -> int | None:
        with self.free_lock:
            for slot, state in enumerate(self.state):
                if state == FREE:
                    return slot
        return None

    def _release(self, slot: int) -> None:
        with self.free_lock:
            self.state[slot] = FREE
            self.meta[slot] = None

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("dataset capture failed") from self.error

    def finish(self, timeout: float = 60.0) -> dict:
        """After the physics thread stopped and before the renderer closes: drain, encode the tail, write the manifest.
        A writer that died (an encoder failure) is not waited on. A second call returns the first call's summary."""
        if getattr(self, "_summary", None) is not None:
            self.check()
            return self._summary
        self._sync()
        if self.armed is not None:  # armed but never drawn
            self._release(self.armed)
            self.captured -= 1
            self.armed = None
            self.gpu.pending = None
        while self.inflight:
            slot = self.inflight.pop(0)
            if self.writer.is_alive():
                try:
                    self.queue.put((slot, self.meta[slot]), timeout=timeout)
                except queue.Full:
                    self.error = self.error or RuntimeError("the capture writer stopped taking frames")
                    break
        if self.writer.is_alive():
            try:
                self.queue.put(None, timeout=timeout)
            except queue.Full:
                pass
            self.writer.join(timeout)
            if self.writer.is_alive():
                self.error = self.error or RuntimeError(f"the capture writer did not finish within {timeout:.0f} s")
        self.samples_file.close()
        status = "failed" if self.error is not None else ("complete" if self.captured else "empty")
        self._summary = self._write_manifest(status)
        self.check()
        return self._summary

    def _sync(self) -> None:
        """Wait for every in-flight device-to-host copy (a no-op on a fake renderer without a device)."""
        import warp as wp

        device = getattr(self.renderer.scene, "device", None)
        if getattr(device, "is_cuda", None) is None:  # a fake renderer in tests has no Warp device
            return
        wp.synchronize_stream(self.renderer.stream or wp.get_stream(device))

    # writer thread ------------------------------------------------------------------------------------------------

    def _write(self) -> None:
        self.outputs = {}
        opened = True  # a sink that fails half-way through open() still gets closed
        try:
            self.sink.open(self.width, self.height, self.settings.fps)
            n = self.width * self.height
            black = (bytes(n * 4), bytes(n * 2), bytes(n * 4), bytes(n * 4))  # the hold-fill before the first captured frame
            last, last_index = black, -1
            while True:
                item = self.queue.get()
                if item is None:
                    break
                slot, meta = item
                planes = unpack_frame(self.gpu.host[slot].numpy(), self.width, self.height)  # a pinned host array; the planes are bytes
                self._release(slot)
                for _ in range(meta["index"] - last_index - 1):  # hold-fill the slots no frame was captured for
                    for (name, _), data in zip(STREAMS, last):
                        self.sink.write(name, data)
                for (name, _), data in zip(STREAMS, planes):
                    self.sink.write(name, data)
                self.samples_file.write(json.dumps(meta) + "\n")
                last, last_index = planes, meta["index"]
                self.video_frames = last_index + 1
            self.outputs = self.sink.close()
            opened = False
        except BaseException as exc:  # noqa: BLE001 - reported through check() and the manifest
            self.error = exc
        finally:
            if opened:  # the sink is always closed; a second failure keeps the first
                try:
                    self.sink.close()
                except BaseException as exc:  # noqa: BLE001
                    self.error = self.error or exc

    def _write_manifest(self, status: str) -> dict:
        r = self.renderer
        data = {"status": status, "case": self.case, "envs": self.envs, "env": int(r.env), "width": self.width, "height": self.height,
                "fps": self.settings.fps, "encoder": self.settings.encoder,
                "streams": {"rgb": "rgba, sRGB display frame", "depth": "gray16le: value = round((1 - d / depth_max) * 65535), 0 = background",
                            "segmentation": "rgba: label colours (objects)", "normals": "rgba: camera space (right, up, forward), 2 * v / 255 - 1"},
                "depth_max": self.settings.depth_max,
                "segmentation": {"background_rgb": [0, 0, 0], "objects": {name: {"id": label, "rgb": label_color(label)} for name, label in r.labels.items()}},
                "captured_frames": self.captured, "video_frames": self.video_frames, "dropped": dict(self.dropped), "frames": "frames.jsonl",
                "timing": "simulation time; a missed slot repeats the previous captured frame in every stream (black before the first)",
                "error": None if self.error is None else str(self.error)}
        (self.directory / "manifest.json").write_text(json.dumps(data, indent=1) + "\n")
        return data
