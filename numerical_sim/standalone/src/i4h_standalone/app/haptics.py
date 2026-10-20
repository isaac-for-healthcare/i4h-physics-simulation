# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Haptics: the servo thread that drives force-feedback devices from the ``tool_force`` product (M11).

The physics system publishes ``force (E, n)`` on the device once per step; the devices want a command every
millisecond, with the pose read at the same rate. The SDK is poll-only (one ``UpdateDevice`` round trip sends the
queued force and refreshes the readings), so the servo is a host thread that owns every live device: per tick it
conditions the newest force of the tool each device holds, calls ``source.io(force_n)`` and publishes the pose
sample into the source's ``mailbox``, which the physics thread's ``InputManager.poll()`` returns. One thread does
all of a device's I/O. omnisurg had no servo (it conditioned and sent once per frame on the sim thread from the
excluded unified backend, instruments/system.py:1084-1126); its conditioner (input/haptics.py:20-76) and the
parameter defaults (config/schema.py:150-158) are ported, the force product is new.

Data path (no host sync anywhere): ``HapticsDriver.update(t)`` runs on the physics thread before ``sim.step()``
and stages the drawn env's force row into a pinned host slot with an asynchronous copy on the physics stream
plus a CUDA event (``ForceMailbox``, three slots: free, in flight, published). The servo takes the newest slot
whose event has completed (an event query, the same the snapshot ring's ``acquire`` does on the render thread;
under the ring lock when one is given, so it never overlaps a recapture) and copies its few vectors out under
the mailbox lock. The servo makes no other Warp call. Zero output, with the conditioner reset so the switch ramp
restarts: no tool held, tool or env changed, a reset (a slot staged before it carries the old ``generation`` and is
refused until fresh data arrives), pause and step-once, a stale force sample (``stale_timeout_s``: physics hung or
paused), an inactive device, a backend error, ``enabled`` off, shutdown (every device gets a zero twice, so a zero one
device queues into another goes out too, then the input mailboxes are invalidated). The driver re-raises a failed
servo on the physics thread and marks the devices inactive when the servo stops refreshing their samples.

A pose device that mounts an instrument-box handle (``input/avatar.py``) also receives a **handle force**: the tool's
``jaw_load`` (the squeeze against its closing jaws) times ``haptics.handles.newtons_per_unit``, through its own
conditioner (the ``handles`` block inherits the pose values and states what differs; per-device ``devices.<id>.handle``
overrides), under the same gates as the pose force. Boxes are served last in the tick so a queued handle force leaves
with the box's own round trip.

``haptics:`` is an app-only top-level YAML key like ``render:`` (Isaac Lab ignores it); the servo settings are
device tuning and stay out of the overrides file. The Haptics panel edits them live (``set_param``, ``set_handle_param``).
"""

from __future__ import annotations

import collections
import dataclasses
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp
from i4h_standalone.app.instruments import resets_env

NUMERIC = ("newtons_per_unit", "max_force_n", "low_pass_hz", "max_slew_n_per_s", "switch_ramp_s", "stale_timeout_s")
FLAGS = {"low_pass_hz": "low_pass", "max_slew_n_per_s": "slew", "switch_ramp_s": "ramp", "stale_timeout_s": "stale"}  # stage -> its on/off switch
CONDITIONING = NUMERIC + tuple(FLAGS.values()) + ("axis_gains",)
RANGES = {"newtons_per_unit": (1e-3, 10.0, True), "max_force_n": (0.0, 10.0, False), "low_pass_hz": (1.0, 200.0, False),
          "max_slew_n_per_s": (1.0, 500.0, False), "switch_ramp_s": (0.0, 0.5, False), "stale_timeout_s": (0.02, 1.0, False)}  # panel: lo, hi, log


@dataclass(frozen=True)
class Conditioning:
    """Per-device output conditioning (omnisurg INPUT_HAPTICS_DEFAULTS, config/schema.py:150-158, plus the unit gain)."""

    newtons_per_unit: float = 1.0  # sim force units -> N at the handle
    max_force_n: float = 0.0  # norm cap after conditioning; 0 = no output (omnisurg required an explicit positive cap)
    low_pass_hz: float = 20.0
    max_slew_n_per_s: float = 20.0
    switch_ramp_s: float = 0.05
    stale_timeout_s: float = 0.1  # zero when the newest force sample is older than this
    low_pass: bool = True  # the stage switches (the panel's checkboxes): off bypasses the stage, the value is kept
    slew: bool = True
    ramp: bool = True
    stale: bool = True
    axis_gains: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "axis_gains", tuple(float(v) for v in self.axis_gains))
        if len(self.axis_gains) != 3 or not all(math.isfinite(v) for v in self.axis_gains):
            raise ValueError("haptics.axis_gains needs three values")
        for name in FLAGS.values():
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"haptics.{name} must be a bool")
        for name in NUMERIC:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0.0:
                raise ValueError(f"haptics.{name} must be a finite number >= 0, got {v!r}")


@dataclass
class HapticsConfig:
    """The ``haptics:`` block: the servo and the conditioning defaults, with per-device overrides under ``devices``."""

    enabled: bool = False
    system: str = "tool_force"  # id of the tool_force system whose force row is sent
    rate_hz: float = 1000.0
    newtons_per_unit: float = 1.0
    max_force_n: float = 0.0
    low_pass_hz: float = 20.0
    max_slew_n_per_s: float = 20.0
    switch_ramp_s: float = 0.05
    stale_timeout_s: float = 0.1
    low_pass: bool = True
    slew: bool = True
    ramp: bool = True
    stale: bool = True
    axis_gains: tuple = (1.0, 1.0, 1.0)
    handles: dict = field(default_factory=dict)  # the handle force (the tool's jaw load): conditioning keys that differ from the pose force's
    devices: dict = field(default_factory=dict)  # device id -> {conditioning key: value, handle: {conditioning key: value}}

    def __post_init__(self):
        if not (isinstance(self.rate_hz, (int, float)) and 1.0 <= float(self.rate_hz) <= 10000.0):
            raise ValueError(f"haptics.rate_hz must be within 1..10000, got {self.rate_hz!r}")
        self.base = Conditioning(**{k: getattr(self, k) for k in CONDITIONING})
        unknown = set(self.handles or {}) - set(CONDITIONING)
        if unknown:
            raise ValueError(f"haptics.handles: unknown keys {sorted(unknown)}; expected {list(CONDITIONING)}")
        self.handle_base = dataclasses.replace(self.base, **dict(self.handles or {}))  # inherits the pose values; states what differs
        self.devices = {str(dev): dict(over or {}) for dev, over in dict(self.devices or {}).items()}
        self.handle_devices = {dev: dict(over.pop("handle", None) or {}) for dev, over in self.devices.items()}
        for dev, over in self.devices.items():
            unknown = (set(over) | set(self.handle_devices[dev])) - set(CONDITIONING)
            if unknown:
                raise ValueError(f"haptics.devices.{dev}: unknown keys {sorted(unknown)}; expected {list(CONDITIONING)} (and handle: {{...}})")
            self.conditioning(dev)  # validates
            self.handle_conditioning(dev)

    def conditioning(self, device_id: str) -> Conditioning:
        return dataclasses.replace(self.base, **self.devices.get(device_id, {}))

    def handle_conditioning(self, device_id: str) -> Conditioning:
        return dataclasses.replace(self.handle_base, **self.handle_devices.get(device_id, {}))


class Conditioner:
    """omnisurg/input/haptics.py:20-76 (HapticForceConditioner): finite guard, source-switch reset, first-order low-pass,
    slew limit, switch ramp, axis gains, norm cap. ``condition`` returns ``(force, saturated)``; a new ``source`` (the
    tool the device holds, in the drawn env) outputs zero on that tick and ramps back up."""

    def __init__(self, params: Conditioning):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self._source = None
        self._filtered = np.zeros(3)
        self._slewed = np.zeros(3)
        self._ramp = 0.0

    def condition(self, force_n, dt: float, source) -> tuple[np.ndarray, bool]:
        p = self.params
        raw = np.asarray(force_n, np.float64).reshape(3)
        if not np.all(np.isfinite(raw)):
            raw = np.zeros(3)
        step = max(0.0, float(dt)) if math.isfinite(float(dt)) else 0.0
        if source != self._source:
            self.reset()
            self._source = source
            return np.zeros(3), False
        alpha = 1.0 if not p.low_pass or p.low_pass_hz <= 0.0 or step <= 0.0 else 1.0 - math.exp(-2.0 * math.pi * p.low_pass_hz * step)
        self._filtered += alpha * (raw - self._filtered)
        delta = self._filtered - self._slewed
        norm = float(np.linalg.norm(delta))
        if p.slew and p.max_slew_n_per_s > 0.0 and norm > p.max_slew_n_per_s * step and norm > 1e-12:
            delta *= p.max_slew_n_per_s * step / norm
        self._slewed += delta
        if p.ramp and p.switch_ramp_s > 0.0:
            self._ramp = min(p.switch_ramp_s, self._ramp + step)
            ramp = self._ramp / p.switch_ramp_s
        else:
            ramp = 1.0
        command = self._slewed * ramp * np.asarray(p.axis_gains)
        norm = float(np.linalg.norm(command))
        saturated = norm > p.max_force_n  # max_force_n 0: everything saturates to zero (no output without an explicit cap)
        if saturated:
            command = command * (p.max_force_n / norm) if norm > 1e-12 else np.zeros(3)
        return command, saturated


@dataclass
class Sample:
    """What the servo reads: the drawn env's force per tool (sim units), the roots and shaft rotations (for pivot devices),
    which device holds which tool, when the row was staged and the reset generation."""

    force: np.ndarray  # (n, 3)
    root: np.ndarray  # (n, 3)
    shaft: np.ndarray  # (n, 4) xyzw
    step: int
    env: int
    holders: dict  # device id -> tool index
    t_wall: float
    generation: int
    jaw: np.ndarray | None = None  # (n,) the tools' jaw loads (sim units), for the handles


class ForceMailbox:
    """Three pinned host slots the physics thread fills asynchronously and the servo reads (see the module docstring)."""

    FREE, INFLIGHT, PUBLISHED = 0, 1, 2

    def __init__(self, control, force, device, capture_lock=None, jaw_load=None):
        self.control, self.force, self.jaw_load = control, force, jaw_load
        self.cuda = bool(device.is_cuda)
        n = max(control.count, 1)
        host = "cpu"
        self.slots = [{"force": wp.zeros(n, dtype=wp.vec3, device=host, pinned=self.cuda), "root": wp.zeros(n, dtype=wp.vec3, device=host, pinned=self.cuda),
                       "shaft": wp.zeros(n, dtype=wp.quat, device=host, pinned=self.cuda), "jaw": wp.zeros(n, dtype=wp.float32, device=host, pinned=self.cuda),
                       "event": wp.Event(device) if self.cuda else None,
                       "state": self.FREE, "step": -1, "env": 0, "holders": {}, "t_wall": 0.0, "generation": 0} for _ in range(3)]
        self.lock = threading.Lock()  # slot bookkeeping, the copy submission and the copy-out: microseconds on both sides
        self.capture_lock = capture_lock  # the ring lock when given: no event query while a command recaptures
        self.generation = 0  # bumped by a reset; a slot carries the generation it was staged under
        self.staged = 0

    # physics thread ---------------------------------------------------------------------------------------------

    def stage(self, env: int, holders: dict, step: int) -> bool:
        """Copy the drawn env's rows into a free slot on the current stream and record its event. Returns False when
        the in-flight slot has not completed and no slot is free (the sample is then one step older)."""
        with self.lock:  # the copies are enqueued and the event recorded before the slot can be seen in flight
            free = next((s for s in self.slots if s["state"] == self.FREE), None)
            if free is None:
                return False
            free.update(step=int(step), env=int(env), holders=dict(holders), t_wall=time.perf_counter(), generation=self.generation)
            stream = wp.get_stream(self.force.device) if self.cuda else None
            wp.copy(free["force"], self.force[env], stream=stream)
            wp.copy(free["root"], self.control.root[env], stream=stream)
            wp.copy(free["shaft"], self.control.shaft_q[env], stream=stream)
            if self.jaw_load is not None:
                wp.copy(free["jaw"], self.jaw_load[env], stream=stream)
            if self.cuda:
                stream.record_event(free["event"])
                free["state"] = self.INFLIGHT
            else:
                self._publish(free)
            self.staged += 1
        return True

    def close(self) -> None:
        """Free the pinned slots and events now, outside any capture: left to the garbage collector they could be
        freed on the physics thread while a later graph capture is open, which poisons that capture."""
        with self.lock:
            if self.cuda and self.slots:
                wp.synchronize_device(self.force.device)
            self.slots = []

    def _publish(self, slot) -> None:  # under self.lock: the newest step is the published one, never an older slot
        current = next((s for s in self.slots if s["state"] == self.PUBLISHED), None)
        if current is not None and current["step"] > slot["step"]:
            slot["state"] = self.FREE
            return
        if current is not None:
            current["state"] = self.FREE
        slot["state"] = self.PUBLISHED

    # servo thread -----------------------------------------------------------------------------------------------

    def latest(self) -> Sample | None:
        """The newest complete sample, copied out; None before the first one."""
        with self.lock:
            inflight = next((s for s in self.slots if s["state"] == self.INFLIGHT), None)
            if inflight is not None and self.cuda:
                lock = self.capture_lock
                if lock is None or lock.acquire(blocking=False):
                    try:
                        done = inflight["event"].is_complete
                    finally:
                        if lock is not None:
                            lock.release()
                    if done:
                        self._publish(inflight)
            slot = next((s for s in self.slots if s["state"] == self.PUBLISHED), None)
            if slot is None:
                return None
            return Sample(slot["force"].numpy().copy(), slot["root"].numpy().copy(), slot["shaft"].numpy().copy(), slot["step"],
                          slot["env"], slot["holders"], slot["t_wall"], slot["generation"], slot["jaw"].numpy().copy())


class HostForceMailbox:
    """The servo's sample source for forces a caller computes itself (``isaaclab.devices.I4hTeleopDevice.push_force``):
    ``push`` publishes one ``(n, 3)`` per-instrument force table (sim units, as ``tool_force``) with the holders; the
    servo routes, conditions and caps it exactly as a ``ForceMailbox`` sample. No root, shaft or jaw load."""

    def __init__(self, n: int):
        self.n = int(n)
        self.lock = threading.Lock()
        self.generation = 0  # bumped by a reset: the servo zeroes until the next push
        self._sample: Sample | None = None
        self._step = 0

    def push(self, force, holders: dict, env: int = 0) -> None:
        force = np.asarray(force, np.float32).reshape(self.n, 3)
        with self.lock:
            self._sample = Sample(force.copy(), np.zeros((self.n, 3), np.float32), np.tile(np.float32([0.0, 0.0, 0.0, 1.0]), (self.n, 1)),
                                  self._step, int(env), dict(holders), time.perf_counter(), self.generation, None)
            self._step += 1

    def latest(self) -> Sample | None:
        with self.lock:
            return self._sample

    def close(self) -> None:
        pass


def live_sources(manager) -> dict:
    """The manager's live device sources (the ones with ``io``: replay, keyboard and noop keep the physics-thread poll),
    by device id, each switched to the servo's mailbox (inactive until the servo's first publish)."""
    sources = {}
    for device in manager.devices:
        src = device.source
        if hasattr(src, "io"):
            src.mailbox = {"active": False}
            sources[device.config.id] = src
    if not sources:
        raise ValueError("haptics: none of the case's inputs is a live device (minimou, fake, ...)")
    return sources


@dataclass
class DeviceStatus:
    tool: str = ""
    force_n: float = 0.0
    saturated: bool = False
    reason: str = "off"  # why the output is zero, or "on"
    age_ms: float = 0.0
    connected: bool = False
    error: str = ""
    handle: dict | None = None  # a mounted handle: channel, id, opening, force sent (N), measured (N), saturated, box connected
    box: dict | None = None  # an instrument box: channels, handle ids, openings


class HapticServo(threading.Thread):
    """Owns the live sources: per tick condition, send, read, publish (module docstring). ``sources`` maps a device id to
    its source. The force stays in the sim frame: ``InputDevice.poll`` adds ``position_offset`` and rotates only the
    orientation by ``rotation_offset`` (input/manager.py), so the device's translational axes are the sim's and the
    force needs no rotation (omnisurg inverse-rotated it, input/manager.py:111-120, wrong for the same reason); the
    device adapter undoes its own axis mapping."""

    stop_timeout = 5.0

    def __init__(self, cfg: HapticsConfig, sources: dict, mailbox: ForceMailbox, tool_names, timings=None):
        super().__init__(name="i4h-haptics", daemon=True)
        self.cfg, self.mailbox, self.timings = cfg, mailbox, timings
        self.started_at = time.perf_counter()
        self.tool_names = list(tool_names)
        self.period = 1.0 / float(cfg.rate_hz)
        self.enabled = bool(cfg.enabled)
        self.paused = False
        boxes = [dev for dev, src in sources.items() if is_box(src)]  # boxes last: a handle force queued this tick leaves with the
        order = [dev for dev in sources if dev not in boxes] + boxes  # box's round trip (its openings the handles read are a tick old)
        self.devices = {dev: {"source": sources[dev], "conditioner": Conditioner(cfg.conditioning(dev)),
                              "handle": Conditioner(cfg.handle_conditioning(dev))
                              if getattr(sources[dev], "handle", None) is not None or getattr(sources[dev], "wants_jaw", False) else None,
                              "box": sources[dev].adapter if dev in boxes else None, "handle_force_n": 0.0,
                              "handle_saturated": False, "status": DeviceStatus()} for dev in order}
        # Immutable YAML ceilings remain independent of live conditioner parameters.
        from types import MappingProxyType

        self.pose_ceilings = MappingProxyType({dev: d["conditioner"].params.max_force_n for dev, d in self.devices.items()})
        self.handle_ceilings = MappingProxyType({dev: d["handle"].params.max_force_n for dev, d in self.devices.items() if d["handle"] is not None})
        self.handles_enabled = True  # the handles group's switch; the pose force has ``enabled``
        self.published_at = 0.0  # wall time of the last sample the servo published (the driver's freshness check)
        self.ticks = 0
        self.missed = 0  # deadline overruns (a tick started more than one period late)
        self.intervals = collections.deque(maxlen=2000)
        self.io_times = collections.deque(maxlen=2000)
        self._lock = threading.Lock()  # the servo's writes of what status() / handle_forces() read from other threads
        self._generation = -1
        self._stopping = False
        self.error: BaseException | None = None

    # main thread --------------------------------------------------------------------------------------------------

    def set_param(self, name: str, value) -> None:
        """A conditioning value for every device (the panel's sliders); the YAML per-device overrides are initial values."""
        self._set_conditioning("conditioner", self.pose_ceilings, name, value)

    def force_ceiling(self, *, handle=False) -> float:
        return max((self.handle_ceilings if handle else self.pose_ceilings).values(), default=0.0)

    def _set_conditioning(self, key, ceilings, name, value):
        if name not in CONDITIONING:
            raise ValueError(f"unknown haptic parameter: {name}")
        # Validate even when no device has this output; never partially apply an edit.
        dataclasses.replace(self.cfg.base, **{name: value})
        if name == "max_force_n" and value > max(ceilings.values(), default=0.0):
            raise ValueError("requested force exceeds configured output ceiling")
        with self._lock:
            updates = [(d[key], dataclasses.replace(d[key].params, **{
                name: min(value, ceilings[dev]) if name == "max_force_n" else value}))
                for dev, d in self.devices.items() if d[key] is not None]
            for conditioner, params in updates:
                conditioner.params = params

    def params(self) -> Conditioning:
        d = next(iter(self.devices.values()), None)
        return d["conditioner"].params if d is not None else self.cfg.base

    def set_handle_param(self, name: str, value) -> None:
        """A handle conditioning value for every mounted handle (the handles group's sliders)."""
        self._set_conditioning("handle", self.handle_ceilings, name, value)

    def handle_params(self) -> Conditioning:
        d = next((d for d in self.devices.values() if d["handle"] is not None), None)
        return d["handle"].params if d is not None else self.cfg.handle_base

    def handle_forces(self) -> dict:
        """The force (N) sent to each mounted handle on the last tick, by device id (the panel's display sliders)."""
        with self._lock:
            return {dev: d["handle_force_n"] for dev, d in self.devices.items() if d["handle"] is not None}

    def has_handles(self) -> bool:
        return any(d["handle"] is not None for d in self.devices.values())

    def stop(self) -> None:
        self._stopping = True
        self.join(self.stop_timeout)
        if self.is_alive():
            raise RuntimeError(f"the haptics thread did not stop within {self.stop_timeout:.0f} s")

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("the haptics thread failed") from self.error

    def status(self) -> dict:
        """A consistent snapshot, copied under the servo's lock (safe while the servo runs); the box and handle entries
        are built here from the adapters' arrays, not on every tick."""
        with self._lock:
            intervals, io = list(self.intervals), list(self.io_times)
            missed, ticks = self.missed, self.ticks
            devices = {dev: (dataclasses.replace(d["status"]), d["box"], d["source"], d["handle"] is not None, d["handle_force_n"],
                             d["handle_saturated"]) for dev, d in self.devices.items()}
        out = {}
        for dev, (st, box, src, has_handle, handle_n, handle_sat) in devices.items():
            if box is not None:
                st.box = {"channels": box.channels, "ids": box.ids.tolist(), "openings": box.openings.tolist(),
                          "status_error": box.status_error}
            if has_handle and getattr(src, "handle", None) is not None:
                st.handle = {**src.handle.status(), "force_n": handle_n, "saturated": handle_sat}
            out[dev] = st
        n = len(intervals)
        io.sort()
        return {"tick_hz": (n / sum(intervals)) if n else 0.0, "missed": missed, "ticks": ticks,
                "io_p95_ms": 1e3 * io[int(0.95 * (len(io) - 1))] if io else 0.0, "devices": out}

    def status_text(self) -> str:
        s = self.status()
        lines = [f"servo {s['tick_hz']:.0f} Hz  missed {s['missed']}  io p95 {s['io_p95_ms']:.2f} ms" + ("" if self.enabled else "  OFF")]
        for dev, st in s["devices"].items():
            if st.box is not None:
                ids = " ".join(str(i) for i in st.box["ids"])
                opens = " ".join(f"{100 * (1 - o):.0f}" for o in st.box["openings"])
                state = "connected" if st.connected else "disconnected" + (f": {st.error}" if st.error else "")
                diag = f"  status read failed: {st.box['status_error']}" if st.box["status_error"] else ""
                lines.append(f"{dev}: box, {st.box['channels']} channels, open % [{opens}], handle ids [{ids}]  {state}{diag}")
                continue
            if not st.connected:
                state = "disconnected" + (f": {st.error}" if st.error else "")
            else:
                state = f"error: {st.error}" if st.error else st.reason
            line = f"{dev}: {st.tool or '-'}  {st.force_n:.2f} N{' sat' if st.saturated else ''}  age {st.age_ms:.0f} ms  {state}"
            if st.handle is not None:
                h = st.handle
                line += (f"\n    handle ch {h['channel']} id {h['id']}  grip {100 * h['opening']:.0f} %  sent {h['force_n']:.2f} N"
                         f"{' sat' if h['saturated'] else ''}  queued {h['queued']:.2f} N  pwm {h['pwm']}  measured {h['measured']:.2f} N"
                         + ("" if h["connected"] else "  box disconnected"))
            lines.append(line)
        return "\n".join(lines)

    # servo thread -------------------------------------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._loop()
        except BaseException as exc:  # noqa: BLE001 - re-raised by check()
            self.error = exc
        finally:
            for _ in range(2):  # motors off, whatever ended the loop: twice, so a zero one device queues into another (a
                for d in self.devices.values():  # pivot's handle force into its box) goes out with the second round trip
                    try:
                        d["source"].io((0.0, 0.0, 0.0))
                    except Exception as exc:  # noqa: BLE001
                        if self.error is None:
                            self.error = exc
                        else:
                            self.error.add_note(f"zeroing device failed: {exc}")
            for d in self.devices.values():  # no servo: the physics thread must see inactive devices, not the last samples
                d["source"].mailbox = {"active": False}

    def _loop(self) -> None:
        period = self.period
        deadline = time.perf_counter() + period
        last = None
        while not self._stopping:
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            now = time.perf_counter()
            late = now - deadline
            if late > period:  # behind by whole periods: those ticks are missed, resync instead of catching up
                skipped = int(late // period)
                self.missed += skipped
                if self.timings is not None:
                    for _ in range(skipped):
                        self.timings.add("haptics.missed", late)  # its count is the number of missed ticks
                deadline = now
            deadline += period
            if last is not None:
                with self._lock:
                    self.intervals.append(now - last)
                if self.timings is not None:
                    self.timings.add("haptics.tick", now - last)
            dt = (now - last) if last is not None else period
            last = now
            sample = self.mailbox.latest()
            for dev, d in self.devices.items():
                src = d["source"]
                with self._lock:
                    force_n, handle_n = self._command(dev, d, sample, now, dt)
                t0 = time.perf_counter()
                try:  # the device round trip runs outside the lock
                    pose = src.io(force_n, handle_n) if d["handle"] is not None else src.io(force_n)
                    error = getattr(src, "error", "")  # an adapter's poll catches its SDK errors and keeps the reason
                except Exception as exc:  # noqa: BLE001 - a device error zeroes and keeps retrying
                    pose = {"active": False}
                    error = f"{type(exc).__name__}: {exc}"
                    d["conditioner"].reset()
                    if d["handle"] is not None:
                        d["handle"].reset()
                        try:  # the handle is a box channel: zeroed whatever failed on the pose device
                            d["source"].handle.queue_force(0.0)
                        except Exception:  # noqa: BLE001
                            pass
                io = time.perf_counter() - t0
                box = d["box"]
                with self._lock:
                    self.io_times.append(io)
                    d["status"].error = error
                    d["status"].connected = bool(box.active) if box is not None else bool(pose.get("active", False))  # a box has no pose
                    d["handle_force_n"] = float(handle_n)
                if self.timings is not None:
                    self.timings.add("haptics.io", io)
                pose = dict(pose)
                pose["t_wall"] = time.perf_counter()  # the driver drops a sample the servo stopped refreshing
                d["source"].mailbox = pose
                self.published_at = pose["t_wall"]
            self.ticks += 1

    def _command(self, dev: str, d: dict, sample: Sample | None, now: float, dt: float):
        """The pose force (N, sim frame) and the handle force (N) for one device this tick; zero with the conditioners reset
        under every gate (module docstring)."""
        st, cond, hcond = d["status"], d["conditioner"], d["handle"]
        st.tool, st.force_n, st.saturated = "", 0.0, False
        d["handle_saturated"] = False
        reason = None
        if not self.enabled:
            reason = "off"
        elif self.paused:
            reason = "paused"
        elif sample is None:
            reason = "no force sample yet"
        elif sample.generation != self.mailbox.generation:
            reason = "reset"  # staged before the reset: zero until a sample of the new generation arrives
        elif not st.connected:
            reason = "device inactive"
        else:
            tool = sample.holders.get(dev)
            st.age_ms = 1e3 * (now - sample.t_wall)
            if tool is None:
                reason = "no tool held"
            elif cond.params.stale and now - sample.t_wall > cond.params.stale_timeout_s:
                reason = "stale"
        if reason is not None:
            cond.reset()
            if hcond is not None:
                hcond.reset()
            st.reason = reason
            return (0.0, 0.0, 0.0), 0.0
        st.tool = self.tool_names[tool] if tool < len(self.tool_names) else str(tool)
        force_n = sample.force[tool] * cond.params.newtons_per_unit  # sim frame; the adapter maps it onto the device axes
        command, saturated = cond.condition(force_n, dt, (sample.env, tool))
        st.force_n, st.saturated, st.reason = float(np.linalg.norm(command)), saturated, "on"
        handle_n = 0.0
        if hcond is not None:
            handle_stale = hcond.params.stale and now - sample.t_wall > hcond.params.stale_timeout_s  # its own gate
            if self.handles_enabled and sample.jaw is not None and not handle_stale:
                jaw_n = float(sample.jaw[tool]) * hcond.params.newtons_per_unit  # the tool's jaw load, a scalar on the x axis
                h, d["handle_saturated"] = hcond.condition((jaw_n, 0.0, 0.0), dt, (sample.env, tool))
                handle_n = max(float(h[0]), 0.0)
            else:
                hcond.reset()
        return (float(command[0]), float(command[1]), float(command[2])), handle_n


def is_box(source) -> bool:
    """An instrument-box source: no pose, binds no tool, carries the handles' channels."""
    from i4h_standalone.input.avatar import InstrumentBox

    return isinstance(getattr(source, "adapter", None), InstrumentBox)


class HapticsDriver:
    """The drivers-list entry (after the ``InstrumentDriver``): stages the drawn env's force row every step, forwards
    pause and reset to the servo, and stops it before the devices close. Owns the servo and the mailbox."""

    def __init__(self, sim, system, instrument_driver, cfg: HapticsConfig, timings=None, capture_lock=None):
        self.sim, self.system, self.instruments, self.cfg = sim, system, instrument_driver, cfg
        control = system.instruments
        manager = instrument_driver.manager
        if manager is None:
            raise ValueError("haptics needs the case's input devices (--input devices)")
        sources = live_sources(manager)
        self.mailbox = ForceMailbox(control, system.force, sim.scene.device, capture_lock=capture_lock, jaw_load=getattr(system, "jaw_load", None))
        self.servo = HapticServo(cfg, sources, self.mailbox, control.names, timings=timings)
        self.servo.start()
        self.step = 0
        self.console = False  # the render loop prints the servo's status to the terminal every 5 s (the panel's switch)
        self.final_status = None  # the servo's status text once stopped (close() keeps it; the devices are gone after)

    def update(self, t: float) -> None:
        self.servo.check()  # a failed servo ends the run rather than leaving the tools on their last sample
        if self.servo.published_at > 0.0 and time.perf_counter() - self.servo.published_at > self.cfg.stale_timeout_s:
            for d in self.servo.devices.values():  # the servo stopped refreshing (a hang): the devices go inactive
                d["source"].mailbox = {"active": False}
        self.mailbox.stage(self.instruments.env, self.instruments.holders_by_device, self.step)
        self.step += 1

    def pause(self, on: bool) -> None:
        self.servo.paused = bool(on)

    def reset(self, env_ids=None) -> None:
        if resets_env(self.instruments.env, env_ids):
            self.mailbox.generation += 1  # the servo zeroes and restarts its ramps

    def close(self) -> None:
        self.servo.stop()  # the final tick sends zero; the InstrumentDriver closes the devices afterwards (reverse order)
        self.final_status = self.servo.status_text()  # a stopped servo: nothing changes under the read
        self.mailbox.close()
        self.servo.devices = {}  # break the driver <-> servo <-> mailbox cycles so nothing waits for the collector
        self.servo.check()
