# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics worker thread and frame pacing for the standalone app.

``PhysicsWorker`` steps the simulator on its own thread: input drivers, ``sim.step()`` and
``ring.publish()`` once per ``dt``, paced to wall-clock time when ``realtime`` is set, as fast
as the GPU allows otherwise. It keeps Warp's current stream (process-global; the renderer
passes its own stream explicitly) and checks every step that no other thread swapped it (a
``.numpy()`` on the render thread would); Warp pushes the CUDA context itself on every call, so
the thread needs no device scope. Commands from the main thread (pause, step once, reset, a live
edit) are a queue drained between steps; each runs under the snapshot ring's lock so the renderer
never queries an event or allocates while a command recaptures a graph. A reset also resets the
drivers' host state; a pause is forwarded to drivers with a ``pause(on)`` hook. An exception ends the thread and is re-raised on the main thread by ``check()``.
With ``timings`` (``app/metrics.py``) every step records its host scopes and its GPU time between two CUDA events;
the events of step k are read after step k+1's ``publish()``, which has waited for step k's work, so the read
never synchronises.

``Pacer`` sleeps to absolute deadlines and, when more than one period behind, resyncs instead
of catching up (a hitch does not make the next seconds free-run).
"""

from __future__ import annotations

import collections
import queue
import threading
import time

import warp as wp


def timed_step(sim, drivers, t: float, n: int, stream, events, timings, pace, after, after_scope: str,
               sync_previous: bool = False, procedure=None) -> float:
    """One physics step with the scaffolding both loops share (the worker's ``_loop`` and ``cli.serial_loop``): ``pace``
    (the pacer's wait, or None), the drivers' updates (input poll and actions upload on the current stream), the event
    pair around ``sim.step()``, ``after()`` (publish, or publish and draw), and the scopes ``pace``, ``input``,
    ``submit``, ``after_scope`` and ``gpu`` (the previous step's event pair; ``sync_previous`` waits on its end event
    first). Returns the step's start time (the callers keep their own gap statistics)."""
    before = time.perf_counter()
    if pace is not None:
        pace()
    start = time.perf_counter()
    for d in drivers:
        d.update(t)
    t1 = time.perf_counter()
    if events is not None:
        stream.record_event(events[n % 2][0])
    sim.step()
    if procedure is not None:
        procedure.update(sim.dt)
    if events is not None:
        stream.record_event(events[n % 2][1])
    t2 = time.perf_counter()
    after()
    if timings is not None:
        add = timings.add
        add("pace", start - before)
        add("input", t1 - start)
        add("submit", t2 - t1)
        add(after_scope, time.perf_counter() - t2)
        if events is not None and n >= 1:  # the previous step: its end event is behind this step's launches
            e0, e1 = events[(n - 1) % 2]
            if sync_previous:
                wp.synchronize_event(e1)
            add("gpu", wp.get_event_elapsed_time(e0, e1, synchronize=False) * 1e-3)
    return start


class Pacer:
    def __init__(self, period: float):
        self.period = float(period)
        self.reset()

    def reset(self) -> None:
        self.next = time.perf_counter() + self.period

    def wait(self) -> None:
        remaining = self.next - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        elif remaining < -self.period:  # more than a period behind: no catch-up
            self.next = time.perf_counter()
        self.next += self.period


class PhysicsWorker(threading.Thread):
    stop_timeout = 10.0  # seconds ``stop()`` waits for the thread

    def __init__(self, sim, drivers, ring, steps: int = 0, realtime: bool = False, timings=None, procedure=None):
        """``steps``: stop after this many (0: until ``stop()``); ``realtime``: one step per ``sim.dt`` of wall time;
        ``timings``: a ``Timings`` collecting the per-step scopes."""
        super().__init__(name="i4h-physics", daemon=True)
        self.sim, self.drivers, self.ring = sim, list(drivers), ring
        self.procedure = procedure
        self.timings = timings
        self._events = None
        if timings is not None:
            dev = sim.scene.device
            self._events = [(wp.Event(dev, enable_timing=True), wp.Event(dev, enable_timing=True)) for _ in range(2)]
        self.limit, self.realtime = int(steps), bool(realtime)
        self.steps = 0  # steps taken
        self.t = 0.0  # simulation time fed to the drivers (reset restarts it)
        self.elapsed = 0.0  # wall time from the first step to the last, GPU work included
        self.paused = False
        self.done = False
        self.error: BaseException | None = None
        self.intervals = collections.deque(maxlen=600)  # wall time between consecutive step starts (s)
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._step_once = False
        self._stopping = False
        self._pacer = Pacer(sim.dt)

    # commands (main thread) -------------------------------------------------

    def pause(self, on: bool | None = None) -> None:
        """Pause (True), resume (False) or toggle (None)."""
        self._queue.put(lambda: self._do_pause(on))

    def step_once(self) -> None:
        """Take one step and stay paused."""
        self._queue.put(self._do_step_once)

    def reset(self, env_ids=None) -> None:
        """``sim.reset(env_ids)`` and every driver's ``reset(env_ids)`` between steps (all envs when None; host ids);
        time restarts, a snapshot is published."""
        self._queue.put(lambda: self._do_reset(env_ids))

    def set_realtime(self, on: bool) -> None:
        """Pace to real time (True) or free-run (False), applied between steps with the deadline re-anchored."""
        self._queue.put(lambda: self._do_realtime(bool(on)))

    def edit(self, fn) -> None:
        """Run ``fn(sim)`` between steps on the physics thread and stream: live edits write device arrays in
        place; a structural edit must call ``sim.prepare()`` itself or the next step raises."""
        self._queue.put(lambda: self._do_edit(fn))

    def stop(self) -> None:
        """Stop after the current step and join. Raises if the thread does not finish within ``stop_timeout``."""
        self._queue.put(self._do_stop)
        self.join(self.stop_timeout)
        if self.is_alive():
            raise RuntimeError(f"the physics thread did not stop within {self.stop_timeout:.0f} s")

    def check(self) -> None:
        """Re-raise an exception that ended the physics thread."""
        if self.error is not None:
            raise RuntimeError("the physics thread failed") from self.error

    # thread ---------------------------------------------------------------

    def run(self) -> None:
        try:
            self._loop()
        except BaseException as exc:  # noqa: BLE001 - reported through check()
            self.error = exc
        finally:
            self.done = True

    def _loop(self) -> None:
        dev = self.sim.scene.device
        stream = wp.get_stream(dev)
        first = last = None
        while not self._stopping and (self.limit == 0 or self.steps < self.limit):
            self._drain(block=self.paused and not self._step_once)
            if self._stopping or (self.paused and not self._step_once):
                continue
            if wp.get_stream(dev) is not stream:
                raise RuntimeError("another thread changed Warp's current stream (a .numpy() or ScopedStream on the render thread?)")
            start = timed_step(self.sim, self.drivers, self.t, self.steps, stream, self._events, self.timings,
                               self._pacer.wait if self.realtime else None, lambda: self.ring.publish(self.t + self.sim.dt),
                               "publish", procedure=self.procedure)  # publish() waited for the previous step's work: its events need no sync
            if first is None:
                wp.synchronize_stream(stream)
                print("[running] Simulation started: first physics step completed.", flush=True)
                first = start
            if last is not None:
                self.intervals.append(start - last)
                if self.timings is not None:
                    self.timings.add("interval", start - last)
            last = start
            self.steps += 1
            self.t += self.sim.dt
            self._step_once = False
        wp.synchronize_device(dev)
        if first is not None:
            self.elapsed = time.perf_counter() - first

    def _drain(self, block: bool) -> None:
        """Run the queued commands (callables returning True when the state changed without a step, so the renderer
        needs a new snapshot); with ``block`` wait for the first one (paused: no busy loop)."""
        if not block and self._queue.empty():
            return
        republish = False
        while True:
            try:
                command = self._queue.get(timeout=0.25) if block else self._queue.get_nowait()
            except queue.Empty:
                break
            block = False
            with self.ring.lock:
                republish |= bool(command())
        if republish:
            self.ring.publish(self.t)

    # the commands, on the physics thread under the ring lock -----------------------------------------------

    def _do_stop(self) -> bool:
        self._stopping = True
        return False

    def _do_pause(self, on) -> bool:
        self.paused = (not self.paused) if on is None else bool(on)
        self._pacer.reset()
        self._notify_pause()
        return False

    def _notify_pause(self) -> None:
        for d in self.drivers:  # a driver with a pause hook (the haptic servo zeroes its output while paused)
            pause = getattr(d, "pause", None)
            if pause is not None:
                pause(self.paused)

    def _do_step_once(self) -> bool:
        self.paused = True
        self._step_once = True
        self._notify_pause()
        return False

    def _do_reset(self, env_ids) -> bool:
        if not self.sim.reset_physics(env_ids):
            return False
        if self.procedure is not None:
            self.procedure.reset(mask=self.sim._mask)
        for d in self.drivers:
            d.reset(env_ids)
        self.t = 0.0
        self._pacer.reset()
        return True

    def _do_realtime(self, on: bool) -> bool:
        self.realtime = on
        self._pacer.reset()
        return False

    def _do_edit(self, fn) -> bool:
        fn(self.sim)
        if self._pacer.period != float(self.sim.dt):  # a structural edit changed dt: re-pace (a live edit keeps the deadline)
            self._pacer.period = float(self.sim.dt)
            self._pacer.reset()
        return True
