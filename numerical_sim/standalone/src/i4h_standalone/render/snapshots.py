# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshots of the render state, and a ring of them for the threaded app.

A ``Snapshot`` holds exactly what the renderers read: the scene positions and flags, every
surface system's live triangles and visual state, every rod system's frames, the instrument
control's poses and collision spheres, every smoke system's particles. ``Snapshot(sim, procedure=procedure)``
aliases the live arrays (no copy) and is what a single-threaded caller renders from.

``Snapshots`` is the ring (default three slots) between the physics thread and the render
thread. The physics thread calls ``publish()`` after every step: it waits for the previous
snapshot's ready event (bounding the host's run-ahead to one step, so the GPU stays busy and the
previous snapshot is always complete), picks a slot the renderer is not holding, makes the
current stream wait for the renderer's last reads of that slot, copies the live arrays into it
and records the slot's ready event. The render thread calls ``acquire()`` before every frame:
the newest slot whose ready event has completed (at most one step old), held until the next
``acquire()`` so it is never overwritten while the renderer reads it. A renderer
brackets its reads with ``snapshot.wait(stream)`` and ``snapshot.done(stream)`` on its own
stream; the release event is recorded right after the copies, before the draw is submitted, so the
physics stream never waits on graphics. Both ``publish`` and ``acquire`` run entirely under
``lock``, and so does every command the physics thread executes, so the render thread never
queries an event or allocates while a graph is being captured.

Warp's current stream is process-global: the physics thread keeps it and issues unqualified
launches, the renderer passes ``stream=`` on every copy and launch. A cross-stream copy of a
non-contiguous array would make Warp synchronise the two streams (and break a capture), so
``copy`` asserts contiguity.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import warp as wp
from i4h_physics.systems.mechanics.surface import Surface
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd
from i4h_surgical.systems.fluids.smoke import Smoke
from i4h_surgical.systems.instruments.clipping import Clipping
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.centreline import Centreline


def warn_overflows(readouts: dict, warned: set) -> None:
    """Debug: warn once per readout named ``*overflow`` that is nonzero in a completed snapshot's host copies
    (``readout_host``; never a device read). ``warned`` holds the keys already reported."""
    import warnings

    for (sid, name), values in readouts.items():
        if name.endswith("overflow") and (sid, name) not in warned and np.any(values):
            warned.add((sid, name))
            warnings.warn(f"{sid}.{name} is {int(np.max(values))} in some env: a fixed-capacity buffer dropped entries "
                          f"(raise the system's capacity setting)", RuntimeWarning, stacklevel=2)


def copy(dst: wp.array, src: wp.array, stream=None) -> None:
    """``wp.copy`` on ``stream`` (None: the destination's current stream); both sides must be contiguous."""
    assert src.is_contiguous and dst.is_contiguous, "render copies must be contiguous (a strided cross-stream copy syncs the streams)"
    wp.copy(dst, src, stream=stream)


class ControlState:
    """The instrument control's per-step products a renderer reads (copied), and its constants (aliased)."""

    MUTABLE = ("root", "shaft_q", "wrist_q", "jaw_q", "jaw_angle", "sphere_q", "sphere_active")

    def __init__(self, control: InstrumentControl):
        for name in self.MUTABLE:
            setattr(self, name, wp.empty_like(getattr(control, name)))
        self.wrist_pivot, self.jaw_anchor, self.sphere_radius = control.wrist_pivot, control.jaw_anchor, control.sphere_radius
        self.sphere_count = control.sphere_count


class Snapshot:
    def __init__(self, sim, copy: bool = False, procedure=None):
        """``copy=False`` aliases the live arrays (a single-threaded caller renders from that); ``copy=True`` allocates a
        ring slot: device copies of the same arrays in the same order, with a ready and a release event."""
        mk = wp.empty_like if copy else (lambda a: a)
        scene = sim.scene
        self.q = mk(scene.q)  # main-block positions (its flags are not drawn: tissue visibility comes from the triangles)
        self.pool_q, self.pool_flags = mk(scene.pool.q), mk(scene.pool.flags)  # fluid pools (the private block)
        self.tris = {s: mk(s.tris) for s in sim.systems if isinstance(s, Surface)}
        self.visual = {s: mk(s.appearance.visual) for s in self.tris if getattr(s, "appearance", None) is not None}
        self.frame = {s: mk(s.frame) for s in sim.systems if isinstance(s, (RodDirect, RodPbd))}
        self.smoke = {s: (mk(s.pos_age), mk(s.vel_life)) for s in sim.systems if isinstance(s, Smoke)}
        self.clip = {s: (mk(s.node), mk(s.active)) for s in sim.systems if isinstance(s, Clipping)}  # (E, G) slots for the drawn clips
        self.centreline = {s: mk(s.positions) for s in sim.systems if isinstance(s, Centreline)}  # (E, K) node positions
        control = next((s for s in sim.systems if isinstance(s, InstrumentControl)), None)
        self.control = ControlState(control) if copy and control is not None else control
        self.arrays = [self.q, self.pool_q, self.pool_flags, *self.tris.values(), *self.visual.values(), *self.frame.values(),
                       *(a for pair in self.smoke.values() for a in pair), *(a for pair in self.clip.values() for a in pair),
                       *self.centreline.values()]  # a fixed order shared by live and slots
        if control is not None:
            self.arrays += [getattr(self.control, name) for name in ControlState.MUTABLE]
        self.readout_arrays = {(s.id, name): mk(a) for s in sim.systems for name, a in s.readouts().items()}
        self.readout_host_arrays = {}
        self.readout_host = {}
        self.arrays += list(self.readout_arrays.values())
        if copy:
            for key, a in self.readout_arrays.items():
                host = wp.empty(a.shape, dtype=a.dtype, device='cpu', pinned=True)
                self.readout_host_arrays[key] = host
                self.readout_host[key] = host.numpy()
        task = procedure
        self.task_rows = mk(task.rows) if task is not None else None
        self.task_host = self.task_host_array = None
        if self.task_rows is not None:
            self.arrays.append(self.task_rows)
            if copy:
                self.task_host_array = wp.empty(self.task_rows.shape, dtype=float, device='cpu', pinned=True)
                self.task_host = self.task_host_array.numpy()  # view made before the worker starts
        self.task_progress = mk(task.progress) if task is not None else None
        self.task_history = mk(task.history) if task is not None else None
        self.task_progress_host = self.task_history_host = None
        self.task_progress_host_array = self.task_history_host_array = None
        if task is not None:
            self.arrays += [self.task_progress, self.task_history]
            if copy:
                self.task_progress_host_array = wp.empty(self.task_progress.shape, dtype=float, device='cpu', pinned=True)
                self.task_history_host_array = wp.empty(self.task_history.shape, dtype=wp.float64, device='cpu', pinned=True)
                self.task_progress_host = self.task_progress_host_array.numpy()
                self.task_history_host = self.task_history_host_array.numpy()
        self.event = self.release = None  # ready (recorded by physics) and released (recorded by the renderer)
        if copy:
            self.event, self.release = wp.Event(scene.device), wp.Event(scene.device)
        self.step = -1  # publish counter; -1: never published
        self.time = 0.0  # host time of the publish (the renderer's snapshot age)
        self.t = 0.0  # simulation time of the state (dataset capture slots)
        self.held = False  # the renderer is reading it

    def wait(self, stream) -> None:
        """The renderer's stream waits for this snapshot's copies; call before the first read."""
        if self.event is not None:
            stream.wait_event(self.event)

    def done(self, stream) -> None:
        """Record that the renderer's reads are enqueued; call right after the last copy, before the draw."""
        if self.release is not None:
            stream.record_event(self.release)


class Snapshots:
    def __init__(self, sim, slots: int = 3, procedure=None):
        if slots < 2:
            raise ValueError("the snapshot ring needs at least two slots (one held by the renderer, one written)")
        self.scene = scene = sim.scene
        if not scene.device.is_cuda:
            raise ValueError("the snapshot ring needs a CUDA device (streams and events); step and publish on one thread on CPU")
        self.live = Snapshot(sim, procedure=procedure)
        per_slot = sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in self.live.arrays)
        scene.check_memory({"snapshots": slots * per_slot})
        for a in self.live.arrays:
            assert a.is_contiguous, "snapshot sources must be contiguous"
        self.slots = [Snapshot(sim, copy=True, procedure=procedure) for _ in range(slots)]
        self.lock = threading.RLock()
        self.count = 0  # publishes so far
        self.last: Snapshot | None = None  # the slot published last

    def publish(self, t: float = 0.0) -> None:
        """Physics thread, after a step: copy the live state into a free slot on the current stream. ``t`` is the
        simulation time of the state.

        Blocks the calling thread (GIL released) until the previous snapshot's copies have completed: the host
        then runs at most one step ahead of the GPU, and the renderer always finds a complete slot."""
        if self.last is not None:
            wp.synchronize_event(self.last.event)
        stream = wp.get_stream(self.scene.device)
        with self.lock:
            free = [s for s in self.slots if not s.held]
            if len(free) > 1:  # keep the newest published slot acquirable while this one is written
                free.remove(max(free, key=lambda s: s.step))
            slot = min(free, key=lambda s: s.step)
            stream.wait_event(slot.release)  # after the renderer's last reads of this slot (a no-op until recorded)
            for dst, src in zip(slot.arrays, self.live.arrays):
                wp.copy(dst, src)
            for key, host in slot.readout_host_arrays.items():
                wp.copy(host, slot.readout_arrays[key])
            if slot.task_host_array is not None:
                wp.copy(slot.task_host_array, slot.task_rows)
                wp.copy(slot.task_progress_host_array, slot.task_progress)
                wp.copy(slot.task_history_host_array, slot.task_history)
            stream.record_event(slot.event)
            slot.step = self.count
            slot.time = time.perf_counter()
            slot.t = float(t)
            self.count += 1
            self.last = slot

    def acquire(self) -> Snapshot | None:
        """Render thread, before a frame: the newest complete slot, held until the next call. Returns the slot
        already held when nothing newer has completed, None before the first publish."""
        with self.lock:
            held = next((s for s in self.slots if s.held), None)
            floor = held.step if held is not None else -1
            ready = [s for s in self.slots if s.step > floor and s.event.is_complete]
            if not ready:
                return held
            newest = max(ready, key=lambda s: s.step)
            if held is not None:
                held.held = False
            newest.held = True
            return newest
