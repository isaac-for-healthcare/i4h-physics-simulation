# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stepper: drives an ``I4hSolver`` on the scene's Newton state, one captured graph per step and per masked reset.

``load_case`` returns one. It is the single owner of ``dt``, ``substeps``, ``iterations`` and ``cuda_graph``.
``prepare()``: drop the graphs and the prepared status, call every system's
``prepare(scene)`` (capacity checks, configuration refresh), push ``dt``/``substeps``/``iterations`` into the solver,
record the structural signature, capture the step graph (``solver.step(state, state, control, None, dt)``) and the
masked-reset graph (``solver.reset(state, world_mask)``, a physical reset) when ``cuda_graph`` is set on a CUDA
device, and only then mark itself prepared. ``step()`` and ``reset()`` refuse to run after a structural edit that has not
been prepared again, so a forgotten recapture fails loudly instead of replaying stale launches.

``reset(env_ids | mask)`` fills a persistent ``(E + 1,)`` Newton world mask (the last slot, world -1, stays False):
host ids are uploaded, device ids (Warp or torch) are marked by a kernel without a sync, a device bool mask (Isaac
Lab's ``reset_buf``) is copied device to device into its first ``E`` entries. Torch inputs are ordered by
``select_envs`` itself (Warp's stream waits for torch's current stream, temporaries outlive Warp's use).
"""

from __future__ import annotations

import gc

import numpy as np
import warp as wp
from i4h_physics.checks import check_bool, check_float, check_int
from i4h_physics.scene import Scene
from i4h_physics.sim import I4hSolver


class Stepper:
    structural = ("dt", "substeps", "iterations", "cuda_graph", "debug")

    def __init__(self, solver: I4hSolver, control=None, dt=1.0 / 60.0, substeps=1, iterations=4, cuda_graph=True,
                 debug=False):
        self.solver = solver
        self.control = control if control is not None else solver.model.control()
        self.dt = check_float("dt", dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.substeps = check_int("substeps", substeps, 1)
        self.iterations = check_int("iterations", iterations, 1)
        self.cuda_graph = check_bool("cuda_graph", cuda_graph)
        self.debug = check_bool("debug", debug)
        self.graph = self.reset_graph = None
        self._prepared = None
        E, dev = self.scene.envs, self.scene.device
        self.world_mask = wp.zeros(E + 1, dtype=wp.bool, device=dev)  # the reset graph's selection; [E] is world -1
        self._mask = self.world_mask[:E]
        from i4h_physics.bindings import PhysicsBinding

        self.physics = PhysicsBinding(self)

    @classmethod
    def create(cls, scene: Scene, systems, **kwargs) -> Stepper:
        """A stepper over a new ``I4hSolver`` on the scene's own Newton model and state."""
        return cls(I4hSolver(scene.model, scene, systems, scene.state), **kwargs)

    # the solver's objects ---------------------------------------------------------------------------------------

    @property
    def scene(self) -> Scene:
        return self.solver.scene

    @property
    def systems(self) -> list:
        return self.solver.systems

    @property
    def model(self):
        return self.solver.model

    @property
    def state(self):
        return self.solver.state

    # structural contract ----------------------------------------------------------------------------------------

    def _signature(self):
        return signature(self, self.systems)

    def prepare(self) -> None:
        """Validate the systems, configure the solver and (re)capture the step and masked-reset graphs."""
        self.graph = self.reset_graph = None
        self._prepared = None
        from i4h_physics.systems.rods.material import validate_material_systems
        validate_material_systems(self.scene, self.systems)
        for s in self.systems:
            s.prepare(self.scene)
        self.solver.configure(self.dt, self.substeps, self.iterations, self.debug)
        signature = self._signature()
        dev = self.scene.device
        if self.cuda_graph and dev.is_cuda:
            # Cyclic owners of earlier scenes may release CUDA arrays/graphs when
            # Python collects them. Defer that collection until both captures end.
            # Ordinary reference-counted lifetime and explicit system hooks remain unchanged.
            collecting = gc.isenabled()
            if collecting:
                gc.disable()
            try:
                with wp.ScopedDevice(dev):
                    with wp.ScopedCapture(device=dev) as capture:
                        self._step()
                    self.graph = capture.graph
                    with wp.ScopedCapture(device=dev) as capture:
                        self.solver.reset(self.state, self.world_mask)
                    self.reset_graph = capture.graph
            finally:
                if collecting:
                    gc.enable()
        self._prepared = signature  # only after a successful capture, so a failed capture cannot fall back silently

    def _check_prepared(self) -> None:
        if self._prepared is None:
            self.prepare()
        elif self._signature() != self._prepared:
            raise RuntimeError("a structural setting changed or a live array was rebound after prepare(); call prepare() to recapture")

    def _step(self) -> None:
        self.solver.step(self.state, self.state, self.control, None, self.dt)

    def step(self) -> None:
        self._check_prepared()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._step()

    # reset ------------------------------------------------------------------------------------------------------

    def reset(self, env_ids=None, *, mask=None) -> None:
        """Reset selected physical state; hosts reset their assessment separately."""
        self.reset_physics(env_ids, mask=mask)

    def reset_physics(self, env_ids=None, *, mask=None) -> bool:
        """Full episode reset of the given envs (all when neither ``env_ids`` nor ``mask`` is given).

        ``env_ids`` may be a sequence of ints, a numpy array, a ``wp.array`` of int32 or a torch int tensor (Isaac Lab
        hands over int64 on the device). Host ids are uploaded (a sync); device ids are cast without one. ``mask`` is an
        ``(E,)`` selection instead: a ``wp.bool`` array or a torch bool tensor on the device, copied device-to-device
        (Isaac Lab's per-step ``reset_buf`` without its ``nonzero()`` sync). Replays the reset graph when captured.
        Torch inputs need no ordering by the caller (see ``select_envs``); Warp arrays are read on Warp's stream."""
        self._check_prepared()
        if not select_envs(self._mask, env_ids, mask, debug=self.debug):
            return False
        if self.reset_graph is not None:
            wp.capture_launch(self.reset_graph)
        else:
            self.solver.reset(self.state, self.world_mask)
        return True


def signature(owner, systems) -> tuple:
    """The structural snapshot ``prepare()`` records and ``step()`` compares: ``owner``'s ``structural`` attributes and
    every system's (identity included, so a replaced system counts as a change), plus the buffer of each live array
    (a rebound one counts as a change; a scalar or reshaped one raises in ``System.live_pointers``)."""
    own = tuple(getattr(owner, name) for name in owner.structural)
    return own, tuple((type(s).__name__, id(s)) + tuple(getattr(s, name) for name in s.structural) + (s.live_pointers(),)
                      for s in systems)


def select_envs(dst: wp.array, env_ids=None, mask=None, *, debug: bool = False) -> bool:
    """Write the env selection into ``dst``, an ``(E,)`` wp.bool array (the first ``E`` entries of a Newton world mask):
    every env when both are None, the ids, or a copy of an ``(E,)`` bool ``mask`` (Warp or torch, on the device). Returns
    False when the ids select nothing. Device ids and masks need no host sync. A torch input is ordered here: its int32
    cast runs on torch's current stream, Warp's stream then waits for that stream (one reused event, no sync), and the
    tensors Warp reads are kept alive until Warp's stream has consumed them. Host ids are always range-checked;
    device ids only with ``debug`` (one host sync, before the int32 cast), else ``mark_envs`` ignores out-of-range ids."""
    E, dev = dst.shape[0], dst.device
    if mask is not None:
        if env_ids is not None:
            raise ValueError("pass env_ids or mask, not both")
        if type(mask).__module__.split(".")[0] == "torch":
            _warp_after_torch(mask, dev)
            mask = wp.from_torch(mask, dtype=wp.bool)
        if not isinstance(mask, wp.array) or mask.dtype is not wp.bool or mask.shape != (E,) or mask.device != dev:
            raise ValueError(f"mask must be a ({E},) bool array on {dev}")
        wp.copy(dst, mask)
    elif env_ids is None:
        dst.fill_(True)
    else:
        ids = _int32_ids(env_ids, dev, E, debug)
        if ids.shape[0] == 0:
            return False
        dst.zero_()
        wp.launch(mark_envs, dim=ids.shape[0], inputs=[ids], outputs=[dst], device=dev)
    return True


def _int32_ids(env_ids, device, envs: int, debug: bool = False) -> wp.array:
    """Host ids are range-checked here, before the int32 cast (an int64 4294967296 would wrap to 0); device ids (wp,
    torch) are checked by ``mark_envs`` without a sync, or here with ``debug``."""
    if isinstance(env_ids, wp.array):
        if env_ids.dtype is not wp.int32 or env_ids.device != device:
            raise ValueError("env_ids must be an int32 wp.array on the scene device")
        if debug:
            _check_range(env_ids.numpy().reshape(-1), envs)
        return env_ids
    if type(env_ids).__module__.split(".")[0] == "torch":
        import torch

        if env_ids.dtype == torch.bool:
            raise TypeError("a bool tensor selects envs as a mask: pass it as reset(mask=...)")
        if env_ids.numel() == 0:  # selects nothing, whatever its dtype (torch.as_tensor([]) is float32)
            return wp.empty(0, dtype=wp.int32, device=device)
        if env_ids.dtype.is_floating_point or env_ids.dtype.is_complex:  # metadata only: no sync
            raise ValueError(f"env_ids must be integers, got {env_ids.dtype}")
        if debug:  # the original values, before the int32 cast can wrap them
            _check_range(env_ids.reshape(-1).cpu().numpy(), envs)
        t = env_ids.to(device=wp.device_to_torch(device), dtype=torch.int32).reshape(-1)
        _warp_after_torch(t, device)  # after the cast: an event recorded before it would let Warp read unwritten ids
        return wp.from_torch(t, dtype=wp.int32)
    ids = np.arange(envs)[env_ids] if isinstance(env_ids, slice) else np.asarray(env_ids).reshape(-1)
    if ids.size and not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"env_ids must be integers, got {ids.dtype}")
    _check_range(ids, envs)
    return wp.array(ids.astype(np.int32), dtype=wp.int32, device=device)


def _check_range(ids: np.ndarray, envs: int) -> None:
    if ids.size and (ids.min() < 0 or ids.max() >= envs):
        raise ValueError(f"env_ids must be in [0, {envs}); got {ids[(ids < 0) | (ids >= envs)].tolist()}")


_ordering: dict = {}  # Warp stream handle -> (torch's view of it, a reusable event): nothing created per reset


def _warp_after_torch(t, device) -> None:
    """Warp's current stream on ``device`` waits for torch's current stream; ``t`` lives until Warp's stream used it."""
    if not t.is_cuda:
        return
    import torch

    stream = wp.get_stream(device)
    entry = _ordering.get(stream.cuda_stream)
    if entry is None:
        entry = _ordering[stream.cuda_stream] = (wp.stream_to_torch(stream), torch.cuda.Event())
    ext, event = entry
    current = torch.cuda.current_stream(t.device)
    if current.cuda_stream == ext.cuda_stream:
        return
    event.record(current)
    ext.wait_event(event)
    t.record_stream(ext)


@wp.kernel
def mark_envs(ids: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool)):
    e = ids[wp.tid()]
    if e >= 0 and e < mask.shape[0]:  # out-of-range device ids are ignored, never written
        mask[e] = True
