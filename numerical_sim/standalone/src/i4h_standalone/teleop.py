# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleoperation of an ``I4hEnv``: the case's input devices drive env 0, with force feedback (import only inside an
Isaac Lab process; ``scripts/isaaclab_teleop.py`` is the command line).

The env runs with ``action_mode = "physical"``: each step ``app.device_rows.DeviceRows`` (the app's device path: the
bindings, the pickup, squeeze-to-fire, ``kind_columns``) writes env 0's ``(n, 10)`` instrument rows on the host, they
go into env 0's columns of the ``(E, A)`` action, and the other envs hold their reset pose (``others = "hold"``) or
follow env 0 (``"mirror"``). With haptics, ``app.haptics.HapticsDriver`` stages env 0's ``tool_force`` row before
``env.step`` (outside the step graph, on Warp's current stream) and its 1 kHz servo owns the devices' I/O, as in the
app. When env 0's episode ends (one host read of ``terminated | truncated`` per step, at teleop rate) its rows return
to the reset pose, the pickup to the ``input:`` bindings, and the servo zeroes until a fresh sample arrives.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext

import torch
from i4h_isaaclab.views import controls
from i4h_standalone.app.device_rows import DeviceRows
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

SERVO_SWITCH_INTERVAL = 0.0002  # s: a 1 kHz servo thread cannot wait Python's default 5 ms for the GIL (the app's value)


def fake_inputs(inputs_cfg: dict) -> dict:
    """The case's ``inputs:`` with every pose device replaced by the ``fake`` backend (a fixed pose at the device's
    calibration offset, open jaw, recording the forces it is sent) and the instrument box dropped: no hardware."""
    out = {}
    for dev, spec in (inputs_cfg or {}).items():
        if dev == "pickup":
            out[dev] = spec
            continue
        spec = dict(spec)
        if spec.get("backend") == "ibcontroller":
            continue
        settings = {k: v for k, v in dict(spec.get("settings") or {}).items()
                    if k in ("position_offset", "rotation_offset", "rotation_offset_euler_degrees")}
        out[dev] = {"backend": "fake", "scale": spec.get("scale", 1.0), "settings": {**settings, "io_ms": 0.3}}
    return out


class Teleop:
    def __init__(self, env, inputs_cfg: dict, haptics_cfg=None, others: str = "hold", renderer=None, record: bool = False):
        """``env``: an ``I4hEnv`` (or its gym wrapper) with ``action_mode = "physical"``; ``inputs_cfg``: an ``inputs:``
        block (the case's, ``fake_inputs``, ``replay_inputs``); ``haptics_cfg``: an ``app.haptics.HapticsConfig``
        (None or disabled: no force feedback); ``renderer``: optional key source for the pickup keys
        (``render.gl.GlKeys``); ``record``: keep env 0's rows per step (``devices.recorded``)."""
        u = env.unwrapped
        if u.cfg.action_mode != "physical":
            raise ValueError("teleop needs I4hEnvCfg.action_mode = 'physical'")
        if others not in ("hold", "mirror"):
            raise ValueError(f"others must be hold or mirror, got {others!r}")
        self.env, self.u, self.others = env, u, others
        self.renderer = renderer
        sim = u.simulator
        control = next((s for s in sim.systems if isinstance(s, InstrumentControl) and s.count), None)
        if control is None:
            raise ValueError(f"case {u.cfg.case!r} has no instruments to teleoperate")
        offset = 0
        for s in controls(sim):  # the instrument columns within the concatenated action (views.action_views order)
            if s is control:
                break
            offset += int(s.actions.size // sim.scene.envs)
        self.columns = slice(offset, offset + 10 * control.count)
        self.devices = DeviceRows(sim.scene.instruments.values(), inputs_cfg, "devices", renderer=renderer, record=record)
        self.haptics = None
        self._switch_interval = None
        try:
            if haptics_cfg is not None and haptics_cfg.enabled:
                from i4h_standalone.app.haptics import HapticsDriver

                force = next((s for s in sim.systems if s.id == haptics_cfg.system), None)
                if force is None or not hasattr(force, "force"):
                    raise ValueError(f"haptics.system {haptics_cfg.system!r}: the case has no tool_force system of that id")
                self._switch_interval = sys.getswitchinterval()
                sys.setswitchinterval(SERVO_SWITCH_INTERVAL)
                self.haptics = HapticsDriver(sim, force, self.devices, haptics_cfg)
        except BaseException:
            self.close()
            raise
        self.physical = self.rows = self.initial_rows = None
        self.steps = 0

    def reset(self):
        """``env.reset()``; the actions start from every env's reset pose. Returns the observations."""
        with getattr(self.renderer, "lock", None) or nullcontext():
            obs, _ = self.env.reset()
        torch.cuda.synchronize()  # the reset poses, read once (not per step)
        self.physical = torch.cat([v.clone() for v in self.u.action_views], dim=1)
        self.initial_rows = self.physical[0, self.columns].cpu().numpy().reshape(-1, 10).copy()
        self.rows = self.initial_rows.copy()
        self._env0_reset()
        return obs

    def step(self):
        """Poll the devices, stage the force, one ``env.step``. Returns ``(obs, env0_reset)``."""
        if self.physical is None:
            self.reset()
        self.devices.update(self.rows)
        row = torch.from_numpy(self.rows.reshape(-1)).to(self.physical.device)
        self.physical[0, self.columns] = row
        if self.others == "mirror":
            self.physical[1:, self.columns] = row
        if self.haptics is not None:
            self.haptics.update(self.steps * self.u.step_dt)  # env 0's force after the previous step
        with getattr(self.renderer, "lock", None) or nullcontext():
            obs, _, terminated, truncated, _ = self.env.step(self.physical)
        done = bool((terminated[0] | truncated[0]).item())
        if done:
            self._env0_reset()
        self.steps += 1
        return obs, done

    def _env0_reset(self) -> None:
        self.rows[:] = self.initial_rows
        self.devices.reset()
        if self.haptics is not None:
            self.haptics.reset([0])

    def close(self) -> None:
        """The servo stops (final zeros) before the devices close."""
        try:
            if self.haptics is not None:
                self.haptics.close()
        finally:
            self.devices.close()
            if self._switch_interval is not None:
                sys.setswitchinterval(self._switch_interval)


class XrView:
    """The headset for a teleop loop (``isaaclab_teleop.py --render xr``): the main thread steps the env and publishes a
    snapshot after each step (``publish``); a render thread draws the newest one with the ``render.xr.XrRenderer`` at
    the headset's rate, as the app's render thread does beside its physics worker. All OpenXR and Slang work (frames,
    reconnects, teardown) stays on the render thread."""

    def __init__(self, env, renderer):
        import threading

        import warp as wp
        from i4h_standalone.render.snapshots import Snapshots

        u = env.unwrapped
        self.renderer, self.device = renderer, u.simulator.scene.device
        self.ring = Snapshots(u.simulator, procedure=getattr(u, "procedure", None))
        renderer.stream = wp.Stream(self.device)  # the renderer's copies out of the snapshots
        renderer.lock = self.ring.lock
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="i4h-xr-render", daemon=True)

    def start(self) -> None:
        self.ring.publish()  # a complete snapshot before the first frame
        self._thread.start()

    def publish(self, t: float = 0.0) -> None:
        """Main thread, after ``env.step``: the stepped state for the headset."""
        self.ring.publish(t)

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and self.error is None

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("the XR render thread failed") from self.error

    def _run(self) -> None:
        from i4h_standalone.render.interop import cuda_driver

        try:
            cuda_driver().set_current_context(int(self.device.context))  # the Slang/CUDA interop runs on this thread
            while not self._stop.is_set() and self.renderer.is_running():
                snap = self.ring.acquire()
                if snap is not None:
                    self.renderer.publish(snap)
                self.renderer.draw()
        except BaseException as exc:  # noqa: BLE001 - re-raised by check() on the main thread
            self.error = exc
        finally:
            try:
                self.renderer.close()  # OpenXR and the Slang device go on the thread that used them
            except BaseException as exc:  # noqa: BLE001
                self.error = self.error or exc

    join_timeout = 10.0

    def close(self) -> None:
        """Stop the render thread and wait for it (it closes the renderer). Raises if it does not stop: a thread blocked
        in OpenXR could resume against freed resources, so the caller must not free the env then (the app's hung-worker
        rule). The thread's own error is reported by ``check()``."""
        self._stop.set()
        if self._thread.ident is None:  # never started (a failure before the loop): close here, stopping a managed runtime
            self.renderer.close()
            return
        self._thread.join(self.join_timeout)
        if self._thread.is_alive():
            raise RuntimeError(f"the XR render thread did not stop within {self.join_timeout:.0f} s; not freeing the env")
