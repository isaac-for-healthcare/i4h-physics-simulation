# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner: `i4h-sim CASE --steps N [--render noop|slang|gl] [--input scripted|keyboard|devices|noop] [--record|--replay BASE]
[--overrides FILE] [--metrics] [--profile N] [--capture DIR]`.

With a Slang window physics runs on a worker thread (``app/worker.py``): instrument and catheter actions are written
between steps the same way Isaac Lab writes them (scripted poses and rates, the keyboard in the
Slang window, or the case's ``inputs:`` devices), then ``sim.step()`` and a snapshot into the ring
(``render/snapshots.py``). The main thread acquires the newest complete snapshot, publishes it on the
renderer's own CUDA stream and draws, paced at ``--fps``; physics is paced to real time when a window
is open (``--realtime`` overrides). With ``--haptics`` (or the case's ``haptics.enabled``) a third thread, the
haptic servo (``app/haptics.py``), owns the devices and renders the ``tool_force`` of the drawn env's tools.
Window keys: P pause, N step once, Backspace reset all envs, G gravity on/off (a live edit through the command queue). ``--steps 0``, the default with a window, runs until the window closes.
Headless runs (``--render noop``), ``--render gl`` and CPU devices step on the main thread with no ring
and no worker (``serial_loop``: nothing consumes snapshots; Newton's viewer copies through the host; CPU
Warp has no streams or events).
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
import yaml
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_endoluminal.systems.rods.drag import Drag
from i4h_isaaclab.composition import load_case, read_case, resolve_case
from i4h_standalone.app.capture import DEFAULT_CAPTURE_FPS, ENCODERS, Capture, CaptureSettings
from i4h_standalone.app.catheter import CatheterDriver
from i4h_standalone.app.device_rows import replay_inputs
from i4h_standalone.app.drag import DragDriver
from i4h_standalone.app.haptics import HapticsConfig, HapticsDriver
from i4h_standalone.app.instruments import InstrumentDriver
from i4h_standalone.app.metrics import Timings, profile
from i4h_standalone.app.ui import METRIC_SCOPES, Panels, SimControls, load_overrides
from i4h_standalone.app.worker import Pacer, PhysicsWorker, timed_step
from i4h_standalone.render import make_renderer
from i4h_standalone.render.snapshots import Snapshots, warn_overflows
from i4h_surgical.systems.instruments.contact_paint import ContactPaint
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="i4h-sim", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("case", help="case YAML path or a shipped domain case name")
    parser.add_argument("--steps", type=int, default=None,
                        help="physics steps; 0 runs until the window closes (default: 0 with a window or headset, 200 headless)")
    parser.add_argument("--render", choices=("noop", "slang", "gl", "xr"), default=None, help="override render.backend (xr: the headset, render.xr)")
    parser.add_argument("--input", choices=("scripted", "keyboard", "devices", "noop"), default=None,
                        help="instrument driver: keyboard with a Slang window, scripted otherwise; devices = the case's inputs")
    parser.add_argument("--record", default=None, help="write the driven poses per device to <path>_<device>.npy for replay")
    parser.add_argument("--replay", default=None, help="replay a --record session: <path>_<device>.npy per instrument (implies --input devices)")
    parser.add_argument("--fps", type=float, default=60.0, help="render loop pacing (frames per second)")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None,
                        help="pace physics to one dt per dt of wall time (default: with a window; headless free-runs)")
    parser.add_argument("--device", default=None, help="override sim.device")
    parser.add_argument("--envs", type=int, default=None, help="override sim.envs")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--debug", action="store_true", help="check launch inputs never alias outputs")
    parser.add_argument("--metrics", nargs="?", const="", default=None, metavar="JSON",
                        help="print the timing table at exit (p50/p95/p99 per scope); with a path also write it as JSON")
    parser.add_argument("--profile", type=int, default=0, metavar="N",
                        help="after the run, N eager steps under Warp's kernel timing: GPU time per kernel and module (headless only)")
    parser.add_argument("--capture", default=None, metavar="DIR",
                        help="dataset capture into DIR: rgb, depth, segmentation and normals as lossless videos plus manifest.json and "
                             "frames.jsonl (slang backend; needs ffmpeg)")
    parser.add_argument("--capture-fps", type=float, default=DEFAULT_CAPTURE_FPS, help="video frames per second of simulation time; independent of --fps")
    parser.add_argument("--capture-encoder", choices=ENCODERS, default="ffv1")
    parser.add_argument("--capture-max", type=int, default=0, help="stop capturing after N frames (0: unbounded)")
    parser.add_argument("--capture-depth-max", type=float, default=5.0, help="scene units mapped onto the 16-bit depth")
    parser.add_argument("--haptics", action=argparse.BooleanOptionalAction, default=None,
                        help="force feedback on the case's devices from the tool_force system (default: the case's haptics.enabled; "
                             "needs --input devices)")
    parser.add_argument("--overrides", default=None, metavar="YAML",
                        help="knob, enabled and sim overrides applied before the first step when the file exists; the panel's Save target "
                             "(default <case>.overrides.yaml in the working directory)")
    parser.add_argument("--no-procedure", action="store_true", help="disable optional target assessment")
    parser.add_argument("--data", type=Path, help="external production asset directory (overrides I4H_DATA)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.capture_fps) or args.capture_fps <= 0.0:
        parser.error("--capture-fps must be finite and positive")
    if args.data is not None:
        os.environ["I4H_DATA"] = str(args.data.expanduser().resolve())
    if args.fps <= 0.0:
        parser.error("--fps must be positive")

    startup = time.perf_counter()
    print(f"[startup] Loading case {args.case!r} and allocating physics...", flush=True)
    cfg = read_case(args.case)
    if args.no_procedure:
        cfg.pop("procedure", None)
    sim = load_case(cfg, device=args.device, envs=args.envs, debug=args.debug)
    from i4h_isaaclab.composition import build_procedure

    procedure = build_procedure(cfg, sim)
    from i4h_standalone.app.workspace import resolve_workspace
    sim.workspace = resolve_workspace(sim, cfg.get("ui"))
    print(f"[startup +{time.perf_counter() - startup:.1f}s] Physics loaded on {sim.scene.device}.", flush=True)
    if args.no_cuda_graph:
        sim.cuda_graph = False
    overrides_path = Path(args.overrides) if args.overrides else Path.cwd() / f"{resolve_case(args.case).stem}.overrides.yaml"
    loaded = {}
    if overrides_path.is_file():
        loaded = yaml.safe_load(overrides_path.read_text()) or {}
        load_overrides(sim, loaded)
        print(f"overrides applied from {overrides_path}")
    scene = sim.scene
    render_cfg = dict(cfg.get("render") or {})
    if args.render is not None:
        render_cfg["backend"] = args.render
    backend = render_cfg.get("backend", "noop")
    if args.steps is None:  # a window or headset runs until closed; a headless run needs an end
        interactive = backend in ("gl", "xr") or (backend == "slang" and render_cfg.get("window", True))
        args.steps = 0 if interactive else 200
    if args.steps == 0 and backend == "noop":
        parser.error("--steps 0 needs a window (render.backend or --render slang|gl)")
    if args.profile and backend != "noop":
        parser.error("--profile is headless: use --render noop")
    if args.capture and backend != "slang":
        parser.error("--capture needs the slang backend")
    if args.capture:
        render_cfg["gbuffer"] = True  # normals and labels for the capture
    timings = Timings() if args.metrics is not None or backend == "slang" else None  # the panel shows them; headless opt-in
    control = next((s for s in sim.systems if isinstance(s, InstrumentControl) and s.count), None)
    mode = args.input or ("keyboard" if backend == "slang" else "devices" if backend == "xr" else "scripted")
    inputs_cfg = cfg.get("inputs")
    if backend == "xr" and (render_cfg.get("xr") or {}).get("inputs") is not None:
        inputs_cfg = render_cfg["xr"]["inputs"]  # the controllers replace the case's desk devices
    if args.replay:  # the recordings replace the case's inputs: one replay device per instrument whose file exists
        if args.input not in (None, "devices"):
            parser.error("--replay drives the instruments itself; drop --input")
        inputs_cfg = replay_inputs(scene.instruments.values(), args.replay)
        if not inputs_cfg:
            parser.error(f"--replay {args.replay}: no <base>_<device>.npy recording of this case's instruments exists")
        mode = "devices"
    if args.record and (control is None or mode not in ("devices", "keyboard") or args.replay):
        parser.error("--record requires instruments and --input keyboard|devices, without --replay")
    haptics_cfg = HapticsConfig(**dict(cfg.get("haptics") or {}))
    if args.haptics is not None:
        haptics_cfg.enabled = bool(args.haptics)
    force_system = None
    if haptics_cfg.enabled:
        if mode != "devices" or control is None:
            parser.error("haptics needs --input devices (or --replay is not haptic) and instruments")
        force_system = next((s for s in sim.systems if s.id == haptics_cfg.system), None)
        if force_system is None or not hasattr(force_system, "force"):
            parser.error(f"haptics.system {haptics_cfg.system!r}: the case has no tool_force system of that id")
    realtime = args.realtime if args.realtime is not None else backend != "noop"

    # everything above only validates; everything below opens something (a window, ffmpeg, devices, threads) and is
    # built inside the try, so a failure part way closes what is already open
    renderer = capture = haptics = worker = None
    drivers = []
    haptics_status = None
    close_errors = []
    switch_interval = sys.getswitchinterval()
    try:
        print(f"[startup +{time.perf_counter() - startup:.1f}s] Creating {backend} renderer (shader compilation may take time)...", flush=True)
        renderer = make_renderer(render_cfg, sim, procedure=procedure)
        print(f"[startup +{time.perf_counter() - startup:.1f}s] Renderer created; setting up inputs...", flush=True)
        if args.capture:  # allocates staging buffers: before the physics thread starts
            capture = Capture(renderer, CaptureSettings(args.capture, fps=args.capture_fps, encoder=args.capture_encoder,
                                                        max_frames=args.capture_max, depth_max=args.capture_depth_max),
                              case=str(args.case), envs=scene.envs, gpu=int(scene.device.ordinal))
        if control is not None:
            paint = next((s.paint for s in sim.systems if isinstance(s, ContactPaint)), None)
            drivers.append(InstrumentDriver(sim, control, inputs_cfg, mode, renderer=renderer, record=args.record, paint=paint))
        for s in sim.systems:
            if isinstance(s, CatheterControl):
                drivers.append(CatheterDriver(sim, s, mode, renderer=renderer))
        if force_system is not None:
            haptics = HapticsDriver(sim, force_system, drivers[0], haptics_cfg, timings=timings)
            drivers.append(haptics)  # after the instrument driver: it reads that driver's env and holders
            # a 1 kHz thread cannot wait the default 5 ms for the GIL (restored at exit)
            sys.setswitchinterval(float(os.environ.get("I4H_SWITCH_INTERVAL", "0.0002")))
        print(f"[startup +{time.perf_counter() - startup:.1f}s] Preparing physics (kernel compilation / CUDA graph capture)...", flush=True)
        sim.prepare()  # inside the block: a failed capture still closes the window and releases the devices
        print(f"[startup +{time.perf_counter() - startup:.1f}s] Physics prepared; setting up the simulation loop...", flush=True)
        gc.collect()
        gc.freeze()  # setup left ~600k Python objects (Warp codegen, USD, numpy); a gen-2 collection over them holds the
        if backend in ("gl", "noop") or not scene.device.is_cuda:  # GIL for ~120 ms and would stall both threads: never scan them again
            stats = serial_loop(sim, renderer, drivers, args.steps, realtime, timings, procedure=procedure)
        else:
            ring = Snapshots(sim, procedure=procedure)
            ring.publish()  # a complete snapshot for the first frame, before the worker starts
            if haptics is not None:
                haptics.mailbox.capture_lock = ring.lock  # the servo's event query never overlaps a recapture
            worker = PhysicsWorker(sim, drivers, ring, steps=args.steps, realtime=realtime, timings=timings, procedure=procedure)
            stats = threaded_loop(worker, ring, renderer, args.fps, cfg["sim"].get("gravity", (0.0, 0.0, -9.81)), timings,
                                  overrides=(overrides_path, loaded), capture=capture, haptics=haptics)
        graph_on = sim.graph is not None  # the timed run's state, before a profile leaves the simulator eager
        if args.profile:
            print(profile(sim, args.profile))
    finally:
        sim.physics.invalidate()
        close_errors = shutdown(renderer, capture, drivers, haptics, worker)
        if haptics is not None:  # close() kept it; after a hung worker the stopped servo can still be read
            haptics_status = haptics.final_status or (None if haptics.servo.is_alive() else haptics.servo.status_text())
        gc.unfreeze()
        sys.setswitchinterval(switch_interval)
    if close_errors:
        raise close_errors[0]

    q = scene.q.numpy()
    finite = all(bool(np.isfinite(a.numpy()).all()) for a in (scene.q, scene.qd, scene.pool.q, scene.pool.qd))
    lo, hi = (q.min(axis=0), q.max(axis=0)) if len(q) else (np.zeros(3), np.zeros(3))  # the main block; pools park far away
    steps = stats["steps"]
    gap = f", max gap {1e3 * stats['gap']:.1f} ms" if stats["gap"] else ""
    rate = (f"physics {steps / max(stats['elapsed'], 1e-9):.1f} Hz ({1e3 * stats['elapsed'] / max(steps, 1):.3f} ms/step"
            f"{', real time' if stats['realtime'] else ''}{gap}), {stats['frames']} frames at "
            f"{stats['frames'] / max(stats['wall'], 1e-9):.1f} fps")
    print(
        f"{steps} steps, {scene.envs} envs, {scene.particle_count} particles (+{scene.pool_count} fluid pool slots), "
        f"graph={'on' if graph_on else 'off'}, render={backend}, input={mode}: {rate}, "
        f"finite={finite}, bounds [{lo[0]:.3f} {lo[1]:.3f} {lo[2]:.3f}] .. [{hi[0]:.3f} {hi[1]:.3f} {hi[2]:.3f}]"
    )
    if capture is not None and "capture" in stats:
        c = stats["capture"]
        print(f"captured {c['captured_frames']} frames ({c['video_frames']} in the videos, {sum(c['dropped'].values())} dropped) -> {args.capture}"
              + (f" [{c['status']}: {c['error']}]" if c["status"] != "complete" else ""))
    if haptics_status is not None:
        print("haptics: " + haptics_status.replace("\n", "; "))
    if timings is not None and args.metrics is not None:
        header = {"device": str(scene.device), "case": str(args.case), "envs": scene.envs, "particles": scene.particle_count,
                  "dt": sim.dt, "substeps": sim.substeps, "iterations": sim.iterations, "graph": graph_on,
                  "input": mode, "render": backend, "fps": args.fps, "realtime": stats["realtime"], "steps": steps,
                  "frames": stats["frames"], "warmup": timings.warmup}
        print(timings.table(header))
        if args.metrics:
            timings.write(args.metrics, header)
    if procedure is not None:
        import json
        print("procedure: " + json.dumps(procedure.results()))
    return 0 if finite else 1


def shutdown(renderer, capture, drivers, haptics, worker) -> list:
    """Stop and close everything ``main`` opened, each step on its own so one failure cannot skip the rest; the errors
    are printed and returned (``main`` re-raises the first when the run itself succeeded). The haptic servo stops
    first and always, so its final zeros go out even when the physics worker hangs. A worker that did not stop still
    issues device work, so nothing that frees device memory runs then: the error propagates and the process exits."""
    errors = []

    def attempt(what, fn):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - collected, printed, the first re-raised by main
            errors.append(exc)
            print(f"i4h-sim: {what} failed on shutdown: {type(exc).__name__}: {exc}", file=sys.stderr)

    if haptics is not None:
        attempt("stopping the haptics servo", haptics.servo.stop)
    if worker is None or not worker.is_alive():
        if capture is not None:
            attempt("finishing the capture", capture.finish)  # before the renderer frees the staging buffers (a no-op if done)
        for driver in reversed(drivers):  # reversed: the haptic servo stops before the instrument driver closes its devices
            attempt(f"closing {type(driver).__name__}", driver.close)
        if renderer is not None:
            attempt("closing the renderer", renderer.close)
    return errors


def serial_loop(sim, renderer, drivers, steps: int, realtime: bool, timings=None, procedure=None) -> dict:
    """One thread, one stream: pace, input, step, publish, draw (headless, the GL debug view, CPU devices). The scopes
    are the worker's (``pace``, ``input``, ``submit``, ``interval``, ``gpu`` from the previous step's event pair after a
    wait on its end event, which the current step's launches keep cheap) plus ``frame.draw`` (the GL viewer's host
    copies; a no-op for noop)."""
    dev = sim.scene.device
    wp.synchronize_device(dev)
    pacer = Pacer(sim.dt) if realtime else None
    stream = events = None
    if timings is not None and dev.is_cuda:
        stream = wp.get_stream(dev)
        events = [(wp.Event(dev, enable_timing=True), wp.Event(dev, enable_timing=True)) for _ in range(2)]
    t0 = time.perf_counter()
    n = 0
    last = None
    max_gap = 0.0
    warned: set = set()  # debug: overflow readouts already reported

    def frame():
        if n == 0:
            wp.synchronize_device(dev)
            print("[running] Simulation started: first physics step completed.", flush=True)
        renderer.publish()
        renderer.draw()
        if n == 0:
            print("[running] First renderer frame submitted.", flush=True)

    while (steps == 0 or n < steps) and renderer.is_running():
        start = timed_step(sim, drivers, n * sim.dt, n, stream, events, timings, pacer.wait if pacer is not None else None,
                           frame, "frame.draw", sync_previous=True, procedure=procedure)
        if last is not None:
            max_gap = max(max_gap, start - last)
            if timings is not None:
                timings.add("interval", start - last)
        last = start
        n += 1
        if sim.debug:  # the step is complete once its readouts reach the host (a sync, in debug only)
            warn_overflows({(s.id, name): a.numpy() for s in sim.systems for name, a in s.readouts().items()}, warned)
    wp.synchronize_device(dev)
    elapsed = time.perf_counter() - t0
    return {"steps": n, "elapsed": elapsed, "realtime": realtime, "frames": n, "wall": elapsed, "gap": max_gap}


def threaded_loop(worker: PhysicsWorker, ring: Snapshots, renderer, fps: float, gravity, timings=None, overrides=None, capture=None,
                  haptics=None) -> dict:
    """Physics on ``worker``, render here: acquire the newest complete snapshot, publish it on the renderer's
    stream, draw, pace. The ring's lock also guards the renderer's reallocations (``set_env``). ``overrides`` is
    ``(path, loaded mapping)`` for the panel's Save and Load; ``capture`` records frames (``app/capture.py``)."""
    panels = None
    if hasattr(renderer, "key_handlers"):  # the Slang window: own stream, ring lock, command keys, mouse dragging, panels
        renderer.stream = wp.Stream(ring.scene.device)
        renderer.lock = ring.lock
        controls = SimControls(worker, gravity)
        renderer.key_handlers.append(controls.handle_key)
        for s in worker.sim.systems:
            if isinstance(s, Drag) and hasattr(renderer, "mouse_handlers"):  # picking needs a window camera (not XR)
                DragDriver(worker.sim, s, renderer, worker.edit)
        if getattr(renderer, "ui", None) is not None:
            hub = next((d.hub for d in worker.drivers if getattr(d, "hub", None) is not None), None)
            path, loaded = overrides if overrides is not None else ("overrides.yaml", {})
            panels = Panels(worker, renderer, controls, hub, timings=timings, overrides_path=path, loaded=loaded, haptics=haptics, procedure=worker.procedure)
            renderer.key_handlers.append(panels.handle_key)
    pacer = Pacer(1.0 / fps)
    frames = 0
    t0 = last_time = last_haptics = time.perf_counter()
    last_steps = last_frames = 0
    seen = -1  # the newest snapshot step published so far (frame.age counts new snapshots only)
    warned: set = set()  # debug: overflow readouts already reported
    print("[startup] Starting physics worker; waiting for the first completed step and renderer frame...", flush=True)
    worker.start()
    try:
        while renderer.is_running() and not worker.done:
            worker.check()
            if capture is not None:
                capture.check()
            if panels is not None:
                panels.update()
            ta = time.perf_counter()
            snap = ring.acquire()
            if snap is not None:
                if timings is not None and snap.step != seen:
                    timings.add("frame.age", ta - snap.time)
                seen = snap.step
                renderer.publish(snap)
                if panels is not None:
                    panels.publish_readouts(snap.readout_host)
                if worker.sim.debug:
                    warn_overflows(snap.readout_host, warned)
                if panels is not None and snap.task_host is not None:
                    panels.publish_procedure(snap.task_host[renderer.env], snap.task_progress_host[renderer.env],
                                             snap.task_history_host[renderer.env])
                if capture is not None:
                    tcap = time.perf_counter()
                    capture.frame(snap)
                    if timings is not None:
                        timings.add("frame.capture", time.perf_counter() - tcap)
            tb = time.perf_counter()
            renderer.draw()
            if frames == 0:
                print("[running] First renderer frame submitted.", flush=True)
            if capture is not None:
                capture.drawn()  # the pose the frame was drawn with
            tc = time.perf_counter()
            if timings is not None:
                timings.add("frame.publish", tb - ta)
                timings.add("frame.draw", tc - tb)
            frames += 1
            now = time.perf_counter()
            if haptics is not None and haptics.console and now - last_haptics >= 5.0:  # the Haptics window's console printout switch
                last_haptics = now
                print(f"[haptics t={worker.t:6.1f}s] " + haptics.servo.status_text().replace("\n", " | "), flush=True)
            if now - last_time >= 1.0:
                hz, rate = (worker.steps - last_steps) / (now - last_time), (frames - last_frames) / (now - last_time)
                gaps = tuple(worker.intervals)  # a copy: the worker appends concurrently
                gap = f" (max gap {1e3 * max(gaps):.1f} ms)" if gaps else ""
                status = f"physics {hz:.0f} Hz{gap}   render {rate:.0f} fps" + ("   PAUSED" if worker.paused else "")
                renderer.set_status(status)
                if panels is not None and timings is not None:
                    panels.set_metrics(f"{status}\nsteps {worker.steps}  t {worker.t:.2f} s\n" + timings.brief(METRIC_SCOPES))
                last_time, last_steps, last_frames = now, worker.steps, frames
            if not getattr(renderer, "self_paced", False):  # XR: xrWaitFrame paces draw()
                pacer.wait()  # never busy-spin: the worker needs the GIL for its launches
            if timings is not None:
                timings.add("frame.pace", time.perf_counter() - now)
    finally:
        if panels is not None:
            panels.close()
        try:
            worker.stop()  # joined before anything frees device memory; raises if the thread hangs
        finally:
            summary = capture.finish() if capture is not None else None  # before the renderer frees the staging buffers
    worker.check()
    out = {"steps": worker.steps, "elapsed": worker.elapsed, "realtime": worker.realtime, "frames": frames,
           "wall": time.perf_counter() - t0, "gap": max(worker.intervals, default=0.0)}
    if summary is not None:
        out["capture"] = summary
    return out


if __name__ == "__main__":
    sys.exit(main())
