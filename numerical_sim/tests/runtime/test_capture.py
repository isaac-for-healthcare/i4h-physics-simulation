# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9e: dataset capture — the slot clock, the frame layout, the writer's hold-fill and drops, the manifest, and with
ffmpeg present a lossless round trip through the video files."""

import hashlib
import json
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import render_tests
from i4h_standalone.app.capture import (
    STREAMS,
    Capture,
    CaptureSettings,
    ListSink,
    SlotClock,
    check_ffmpeg,
    encoder_options,
    label_color,
    unpack_frame,
)

has_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def test_slot_clock():
    clock = SlotClock(30.0)
    assert [clock.slot(t) for t in (0.0, 0.01, 0.04, 0.2)] == [0, None, 1, 6]
    assert clock.slot(0.0) == 7  # a reset: time restarts, the slots keep counting up
    assert clock.slot(1.0 / 30.0 + 0.001) == 8


def pack(rgb, depth16, labels, normals8):
    """The numpy mirror of capture.slang pack_frame for a frame of already-quantised planes."""
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = rgb, 255
    n = np.zeros((h, w, 4), np.uint8)
    n[..., :3], n[..., 3] = normals8, 255
    word0 = rgba.view("<u4").reshape(h, w)
    word1 = depth16.astype("<u4") | (labels.astype("<u4") << 16)
    word2 = n.view("<u4").reshape(h, w)
    return np.stack([word0, word1, word2], axis=-1).reshape(-1)


def test_unpack_frame_round_trip():
    rng = np.random.default_rng(0)
    h, w = 6, 8
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    labels = rng.integers(0, 4, (h, w), dtype=np.uint32)
    depth16 = np.where(labels == 0, 0, rng.integers(1, 65536, (h, w))).astype(np.uint16)
    normals8 = np.where(labels[..., None] == 0, 0, rng.integers(0, 256, (h, w, 3))).astype(np.uint8)
    planes = unpack_frame(pack(rgb, depth16, labels, normals8), w, h)
    assert [len(p) for p in planes] == [h * w * 4, h * w * 2, h * w * 4, h * w * 4]
    assert np.array_equal(np.frombuffer(planes[0], np.uint8).reshape(h, w, 4)[..., :3], rgb)
    assert np.array_equal(np.frombuffer(planes[1], "<u2").reshape(h, w), depth16)
    seg = np.frombuffer(planes[2], np.uint8).reshape(h, w, 4)
    for label in np.unique(labels):
        assert (seg[labels == label][:, :3] == label_color(int(label))).all()
    assert label_color(0) == [0, 0, 0] and len({tuple(label_color(i)) for i in range(1, 300)}) == 299
    assert np.array_equal(np.frombuffer(planes[3], np.uint8).reshape(h, w, 4)[..., :3], normals8)


class _FakeGpu:
    def __init__(self, slots, n):
        self.host = [SimpleNamespace(numpy=lambda a=np.full(n, i + 1, "<u4"): a) for i in range(slots)]
        self.pending = self.encoded = None
        self.done = set(range(slots))

    def completed(self, slot):
        return slot in self.done

    def draw(self, ok=True, skip=False):  # what renderer.draw() does: dispatch the pending slot, reject it (resize) or return early (skip)
        if skip:
            return
        if self.pending is not None and ok:
            self.encoded = self.pending
        self.pending = None


def fake_renderer(width=8, height=6, slots=4):
    r = SimpleNamespace(width=width, height=height, env=0, lock=threading.Lock(), stream=None, labels={"block/block": 1, "rod/rod": 2},
                        camera=SimpleNamespace(basis=lambda: tuple(np.zeros(3, np.float32) for _ in range(4)), fov=45.0, near=0.01, far=100.0),
                        camera_mode="free", rig=None,
                        post=SimpleNamespace(lens_distortion_enabled=False, lens_distortion_strength=0.0, lens_distortion_zoom=1.0, scope_enabled=False,
                                             scope_radius=1.5),
                        scene=SimpleNamespace(device="cpu"))
    gpu = _FakeGpu(slots, width * height * 3)
    r.enable_capture = lambda s, d: gpu
    return r, gpu


def _frame(capture, gpu, t, step=0, ok=True):
    """One render-loop iteration on the fake: arm, draw, record the pose."""
    capture.frame(SimpleNamespace(t=t, step=step))
    gpu.draw(ok)
    capture.drawn()


def test_writer_hold_fill_drops_and_manifest(tmp_path, caplog):
    r, gpu = fake_renderer()
    sink = ListSink()
    capture = Capture(r, CaptureSettings(str(tmp_path), fps=30.0, slots=4), case="two_worlds", envs=2, sink=sink)
    for t in (0.0, 1 / 30, 3 / 30):  # slots 0, 1, 3: slot 2 is missed and hold-filled with slot 1's frame
        _frame(capture, gpu, t, int(t * 60))
    data = capture.finish()
    assert sink.frames["rgb"][2] == sink.frames["rgb"][1] and len(sink.frames["depth"]) == 4
    assert data["status"] == "complete" and data["video_frames"] == 4 and data["captured_frames"] == 3 and data["dropped"] == {"queue_full": 0, "resized": 0}
    assert data["segmentation"]["objects"]["rod/rod"] == {"id": 2, "rgb": label_color(2)}
    lines = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert [m["index"] for m in lines] == [0, 1, 3] and lines[0]["camera"]["fov"] == 45.0
    # a slow sink: the render thread drops when every staging slot is in flight
    r, gpu = fake_renderer(slots=2)
    gate = threading.Event()

    class BlockedSink(ListSink):
        def write(self, name, data):
            gate.wait()
            super().write(name, data)

    capture = Capture(r, CaptureSettings(str(tmp_path / "b"), fps=60.0, slots=2), case="x", envs=1, sink=BlockedSink())
    gpu.done = set()  # nothing lands until the GPU says so
    for k in range(4):
        _frame(capture, gpu, k / 60.0, k)
    assert capture.dropped["queue_full"] == 2 and capture.captured == 2
    warnings = [record for record in caplog.records if "Capture queue full" in record.message]
    assert len(warnings) == 1
    assert "60 fps, 8x6" in warnings[0].message and "Reduce --capture-fps" in warnings[0].message
    gpu.done = {0, 1}
    gate.set()
    data = capture.finish()
    assert data["captured_frames"] == 2 and data["dropped"]["queue_full"] == 2
    # a resize drops and counts: a pending resize before arming, a changed size, and a draw that rejected the armed frame
    r, gpu = fake_renderer()
    capture = Capture(r, CaptureSettings(str(tmp_path / "c"), fps=60.0), case="x", envs=1, sink=ListSink())
    r.pending_resize = (400, 240)
    _frame(capture, gpu, 0.0)
    r.pending_resize, r.width = None, 16
    _frame(capture, gpu, 1 / 60.0)
    r.width = 8
    _frame(capture, gpu, 2 / 60.0, ok=False)  # armed, then draw() saw a new size and cancelled the pack
    assert capture.dropped["resized"] == 3 and capture.captured == 0 and not capture.inflight and capture.state.count(0) == 4
    _frame(capture, gpu, 3 / 60.0)
    data = capture.finish()
    assert data["captured_frames"] == 1 and data["dropped"]["resized"] == 3


def test_encoder_options_and_ffmpeg_check(monkeypatch):
    assert "-level" in encoder_options("ffv1", "bgr0") and "-tune" in encoder_options("h264_nvenc", "gbrp")
    with pytest.raises(ValueError, match="unsupported"):
        encoder_options("libx264", "bgr0")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg on PATH"):
        check_ffmpeg("ffv1", 8, 8)


@render_tests
@has_ffmpeg
def test_capture_two_worlds_lossless_round_trip(tmp_path, case):
    """Twenty offscreen frames through the real pack, pinned copies and ffmpeg; decoding the files gives the bytes back."""
    import warp as wp
    from i4h_isaaclab.composition import load_case, read_case
    from i4h_standalone.render.slang import SlangRenderer
    from i4h_standalone.render.snapshots import Snapshot

    class TeeSink(ListSink):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def open(self, *a):
            super().open(*a)
            self.inner.open(*a)

        def write(self, name, data):
            super().write(name, data)
            self.inner.write(name, data)

        def close(self):
            super().close()
            return self.inner.close()

    from i4h_standalone.app.capture import FfmpegSink

    sim = load_case(case)
    cfg = read_case(case)["render"]
    cfg.pop("backend")
    renderer = SlangRenderer(sim, **dict(cfg, size=(320, 180), window=False, gbuffer=True))
    renderer.stream = wp.Stream(sim.scene.device)
    tee = TeeSink(FfmpegSink(tmp_path, "ffv1"))
    capture = Capture(renderer, CaptureSettings(str(tmp_path), fps=60.0, slots=8), case=case, envs=1, sink=tee)  # unpaced loop: room for the writer
    sim.prepare()
    snap = Snapshot(sim)
    for k in range(20):
        sim.step()
        snap.t = (k + 1) * sim.dt
        snap.step = k
        renderer.publish()
        while 0 not in capture.state:  # the paced app has a frame's time for the writer; the test loop does not
            time.sleep(0.002)
        capture.frame(snap)
        assert renderer.draw()
        capture.drawn()
        wp.synchronize_stream(renderer.stream)  # the app has a frame's time here; the test would otherwise outrun the staging slots
        capture.check()
    data = capture.finish()
    capture.check()
    last_display = renderer.read_frame()
    renderer.close()
    assert data["status"] == "complete" and data["captured_frames"] == 20 and data["video_frames"] == 20 and data["dropped"]["queue_full"] == 0
    for name, pixel_format in STREAMS:
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / f"{name}.mkv"), "-f", "rawvideo", "-pix_fmt", pixel_format, "-"],
                                 capture_output=True, check=True).stdout
        raw = b"".join(tee.frames[name])
        assert len(decoded) == len(raw) == 20 * 320 * 180 * (2 if pixel_format == "gray16le" else 4)
        assert hashlib.sha1(decoded).hexdigest() == hashlib.sha1(raw).hexdigest()  # lossless
    rgb = np.frombuffer(tee.frames["rgb"][-1], np.uint8).reshape(180, 320, 4)
    assert np.abs(rgb[..., :3].astype(int) - np.round(last_display[..., :3] * 255.0).astype(int)).max() <= 1  # the display frame itself
    labels = np.frombuffer(tee.frames["segmentation"][-1], np.uint8).reshape(180, 320, 4)
    assert (labels[..., :3] != 0).any(axis=-1).mean() > 0.1  # something labelled
    assert json.loads((tmp_path / "manifest.json").read_text())["segmentation"]["objects"]


def test_cli_capture_flags_need_slang():
    from i4h_standalone.app.cli import main

    with pytest.raises(SystemExit):
        main(["two_worlds", "--steps", "1", "--render", "noop", "--capture", "/tmp/x"])


def test_camera_metadata_is_the_drawn_pose(tmp_path):
    """The pose is sampled after draw(): a tip camera or a fly move applied inside the draw is what the frame shows."""
    r, gpu = fake_renderer()
    capture = Capture(r, CaptureSettings(str(tmp_path), fps=60.0), case="x", envs=1, sink=ListSink())
    capture.frame(SimpleNamespace(t=0.0, step=0))
    moved = np.float32([1.0, 2.0, 3.0])
    r.camera.basis = lambda: (moved, np.float32([1, 0, 0]), np.float32([0, 1, 0]), np.float32([0, 0, -1]))  # what draw() does to the camera
    gpu.draw()
    capture.drawn()
    capture.finish()
    meta = json.loads((tmp_path / "frames.jsonl").read_text().splitlines()[0])
    assert meta["camera"]["position"] == [1.0, 2.0, 3.0] and meta["camera_mode"] == "free" and meta["tip"] is None


def test_dead_writer_does_not_hang_shutdown(tmp_path):
    class FailingSink(ListSink):
        closed = False

        def open(self, *a):
            raise RuntimeError("encoder refused")

        def close(self):
            FailingSink.closed = True
            return {}

    r, gpu = fake_renderer(slots=2)
    capture = Capture(r, CaptureSettings(str(tmp_path), fps=60.0, slots=2), case="x", envs=1, sink=FailingSink())
    capture.writer.join(5.0)
    assert not capture.writer.is_alive() and FailingSink.closed  # the sink is closed even when open() failed
    for k in range(4):
        _frame(capture, gpu, k / 60.0, k)  # the render loop keeps going; the slots are freed, not queued
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="capture failed"):
        capture.finish(timeout=5.0)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert time.perf_counter() - t0 < 2.0 and data["status"] == "failed" and "encoder refused" in data["error"]
    with pytest.raises(RuntimeError, match="capture failed"):
        capture.check()


def test_skipped_draw_rearms_for_the_newer_snapshot(tmp_path):
    """A draw that returns before rendering keeps the slot armed; the next frame's snapshot is what gets recorded."""
    r, gpu = fake_renderer()
    sink = ListSink()
    capture = Capture(r, CaptureSettings(str(tmp_path), fps=60.0), case="x", envs=1, sink=sink)
    capture.frame(SimpleNamespace(t=10 / 60.0, step=10))
    gpu.draw(skip=True)  # a lost surface: nothing rendered
    capture.drawn()
    assert capture.armed is not None and capture.captured == 1
    capture.frame(SimpleNamespace(t=11 / 60.0, step=11))
    gpu.draw()
    capture.drawn()
    data = capture.finish()
    lines = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["step"] == 11 and data["captured_frames"] == 1 and capture.state.count(0) == 4


def test_skipped_draw_then_same_interval_snapshot_records_the_drawn_step(tmp_path):
    """A slot armed by a skipped draw, then a newer snapshot inside the same video interval (or past --capture-max): the
    metadata describes the snapshot the draw renders, not the one it was armed for."""
    r, gpu = fake_renderer()
    sink = ListSink()
    capture = Capture(r, CaptureSettings(str(tmp_path), fps=30.0, max_frames=1), case="x", envs=1, sink=sink)
    capture.frame(SimpleNamespace(t=10 / 60.0, step=10))
    gpu.draw(skip=True)
    capture.drawn()
    capture.frame(SimpleNamespace(t=11 / 60.0, step=11))  # same 1/30 s slot as step 10, and max_frames already counted
    gpu.draw()
    capture.drawn()
    capture.finish()
    lines = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["step"] == 11 and lines[0]["index"] == 0  # the slot it was armed for
    r, gpu = fake_renderer()
    capture = Capture(r, CaptureSettings(str(tmp_path / "b"), fps=60.0), case="x", envs=1, sink=ListSink())
    capture.frame(SimpleNamespace(t=10 / 60.0, step=10))
    gpu.draw(skip=True)
    capture.drawn()
    r.pending_resize = (1, 1)  # the armed frame is gone with the surface: cancelled, counted
    capture.frame(SimpleNamespace(t=11 / 60.0, step=11))
    assert capture.armed is None and capture.captured == 0 and capture.dropped["resized"] == 1 and capture.state.count(0) == 4
    capture.finish()


def test_hold_fill_before_the_first_frame(tmp_path):
    """Frame 0 dropped (a resize), frame 1 captured: the videos still start at slot 0, filled with black."""
    r, gpu = fake_renderer()
    sink = ListSink()
    capture = Capture(r, CaptureSettings(str(tmp_path), fps=60.0), case="x", envs=1, sink=sink)
    r.pending_resize = (1, 1)
    _frame(capture, gpu, 0.0, 0)
    r.pending_resize = None
    _frame(capture, gpu, 1 / 60.0, 1)
    data = capture.finish()
    assert data["status"] == "complete" and data["video_frames"] == 2 and data["dropped"]["resized"] == 1
    assert len(sink.frames["rgb"]) == 2 and sink.frames["rgb"][0] == bytes(8 * 6 * 4) and sink.frames["depth"][0] == bytes(8 * 6 * 2)
