# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System base class.

A system owns its arrays and launches its kernels directly. ``I4hSolver.step`` calls the
five stage hooks in list order; ``dt`` is the frame dt in ``step_start``/``step_end``
and the substep dt in the other three. ``substep_start`` receives the substep index and
count, ``iteration`` the iteration index and count, so a system can schedule work on a
subset of iterations with ``due()`` (Python values, fixed at capture).

Settings come in two kinds:

- **structural**: Python values baked into launches (counts, flags, tuples, scalars
  passed to ``wp.launch``). List their names in ``structural``; ``Stepper.prepare()`` snapshots
  them and refuses to ``step()`` if any changed. ``enabled`` is one.
  ``prepare(scene)`` runs before every capture: a system that sized buffers from a
  structural value raises there if the value changed (rebuilding is not supported).
- **live**: device arrays read by kernels every step. Declare the dataclass field with
  its YAML default, then in ``__post_init__`` replace it with ``live(scene, value)`` (an
  ``(E,)`` or ``(E, n)`` array). Write it in place (``fill_``, ``assign``, torch ``copy_``);
  never rebind it, or the captured graph keeps pointing at the old buffer.

Arrays in ``inputs`` are never written by the kernel. An array the owning thread reads
and writes goes in ``outputs`` only. Debug mode checks ``inputs ∩ outputs = ∅``.

A system that needs another system's product declares the field name in ``refs``; the
loader replaces the YAML id string with the instance built earlier in the list.

``id`` is the YAML id the loader passes (the type name when absent); tests constructing systems directly leave it empty.

What a host (Isaac Lab, a workflow) reads of a system is declared here, not listed by the host:

- ``state_products``: names of per-env ``(E, ...)`` int32/float32 products exposed as observations and recorded as
  ``states/<id>.<product>``, in the given order.
- ``is_control``: the system reads ``actions (E, ...)`` written from outside the step (an action term). It also defines
  ``action_bounds(workspace)``: the physical ``(low, high)`` per column of ``actions`` flattened per env, and may
  define ``recorded_actions(base)`` for replaying a recorded session.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from i4h_physics.scene import fill_rows

if TYPE_CHECKING:
    from i4h_physics.scene import Scene


def _frozen_value(v):
    return tuple(v) if isinstance(v, list) else v


def due(k: int, events: int, iterations: int) -> tuple[int, int]:
    """Schedule ``events`` launches evenly over the ``iterations`` loop turns (omnisurg stages.py:81-90).

    Returns ``(event index or -1, events)``; ``events`` is capped at ``iterations`` so no interval is skipped."""
    events = max(1, min(int(events), int(iterations)))
    before, after = (k * events) // iterations, ((k + 1) * events) // iterations
    return (after - 1 if after > before else -1), events


def live(scene: Scene, values) -> wp.array:
    """A live parameter array: a scalar becomes ``(E,)``, an ``(n,)`` vector becomes ``(E, n)``."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        return wp.full(scene.envs, float(arr), dtype=wp.float32, device=scene.device)
    return wp.array(np.tile(arr[None], (scene.envs,) + (1,) * arr.ndim), dtype=wp.float32, device=scene.device)




@dataclass(eq=False)
class System:
    scene: Scene
    _: KW_ONLY
    enabled: bool = True
    id: str = ""  # the YAML id (the loader sets it); not structural

    structural: ClassVar[tuple[str, ...]] = ("enabled",)
    refs: ClassVar[tuple[str, ...]] = ()
    state_products: ClassVar[tuple[str, ...]] = ()  # per-env products a host observes and records
    is_control: ClassVar[bool] = False  # reads ``actions`` written from outside the step

    def action_bounds(self, workspace: float) -> tuple[list, list]:
        """A control's physical ``(low, high)`` per flattened column of ``actions``; ``workspace`` is the host's
        position range around a tool's initial pose (a control without positions ignores it)."""
        raise NotImplementedError(f"{type(self).__name__} is not a control system")

    def recorded_actions(self, base) -> list:
        """A recorded session's ``(width, rows or None)`` column blocks for this control (``None``: hold the reset
        pose); none recorded by default."""
        return [(int(np.prod(self.actions.shape[1:])), None)]


    def readouts(self) -> dict:
        """Per-environment products copied to completed snapshots for app readouts."""
        return {}

    # the three idioms every system shares ---------------------------------------------------------------------

    def _live(self, *names: str) -> None:
        """Replace the named dataclass fields (YAML scalars or vectors) with live ``(E,)`` / ``(E, n)`` arrays."""
        for name in names:
            setattr(self, name, live(self.scene, getattr(self, name)))
        self.register_live(*names)

    def register_live(self, *names: str) -> None:
        """Record live arrays assigned directly (``self.x = live(scene, values)``): their layout is fixed, and the
        step and reset graphs bake their buffers, so ``signature()`` checks them."""
        layouts = self.__dict__.setdefault("_live_layouts", {})
        for name in names:
            a = getattr(self, name)
            layouts[name] = (a.shape, a.strides, a.dtype, a.device)

    def live_pointers(self) -> tuple:
        """``(name, buffer)`` of every live array, after checking each still has its registered layout (shape, strides,
        dtype, device): a live parameter is written in place, never rebound to a scalar, another shape or a strided
        view such as a transpose (same buffer, other element order)."""
        out = []
        for name, layout in getattr(self, "_live_layouts", {}).items():
            a = getattr(self, name)
            if not isinstance(a, wp.array) or (a.shape, a.strides, a.dtype, a.device) != layout:
                got = (f"{a.dtype.__name__} array {a.shape} strides {a.strides}" if isinstance(a, wp.array)
                       else type(a).__name__)
                raise TypeError(f"{self.id or type(self).__name__}.{name} is a live parameter, a {layout[0]} array: "
                                f"write into it with .fill_(), .assign() or a torch view, do not rebind it (got {got})")
            out.append((name, a.ptr))
        return tuple(out)

    def freeze(self, *names: str) -> None:
        """Record the structural fields that size buffers or select indices at construction; ``prepare()`` raises
        when one changed (rebuilding is not supported). Lists compare as tuples."""
        self._frozen = names
        self._built = tuple(_frozen_value(getattr(self, n)) for n in names)

    def prepare(self, scene: Scene) -> None:
        """Called by ``Stepper.prepare()`` before the capture; check or rebuild structural state here (a subclass
        with more to do calls ``super().prepare(scene)`` first)."""
        names = getattr(self, "_frozen", ())
        if names and tuple(_frozen_value(getattr(self, n)) for n in names) != self._built:
            kind = type(self).__module__.rsplit(".", 1)[-1]  # the YAML kind (module name)
            fields = ", ".join(f"{kind}.{n}" for n in names)
            raise ValueError(f"{fields}: size buffers or select indices and are fixed at construction")

    def zero(self, mask: wp.array | None, *names: str) -> None:
        """Zero the named per-env products: every env when ``mask`` is None (a disabled system's ``prepare``), else the
        selected envs (``reset``). A system clears its own products; a product a disabled system stopped refreshing
        must not linger for its readers."""
        for name in names:
            arr = getattr(self, name)
            if mask is None:
                arr.zero_()
            else:
                self.clear(mask, (arr, arr.dtype()))

    def clear(self, mask: wp.array, *pairs) -> None:
        """Masked reset of per-env scratch: ``dst[env, :] = value`` for every ``(array, value)`` where ``mask[env]``.
        Arrays are ``(E, n)``, ``(E,)`` or body-local ``(E*n,)`` (reshaped here, a view)."""
        E = self.scene.envs
        for arr, value in pairs:
            if arr.ndim != 2 and arr.shape[0] % E:
                raise ValueError(f"{type(self).__name__}.clear: a {arr.shape} array is not (E, n), (E,) or (E*n,) for E = {E}")
            dst = arr if arr.ndim == 2 else arr.reshape((E, arr.shape[0] // E))
            wp.launch(fill_rows, dim=dst.shape, inputs=[mask, value], outputs=[dst], device=self.scene.device)

    def step_start(self, scene: Scene, dt: float) -> None:
        pass

    def substep_start(self, scene: Scene, dt: float, s: int, substeps: int) -> None:
        pass

    def iteration(self, scene: Scene, dt: float, k: int, iterations: int) -> None:
        pass

    def substep_end(self, scene: Scene, dt: float) -> None:
        pass

    def step_end(self, scene: Scene, dt: float) -> None:
        pass

    def reset(self, scene: Scene, mask: wp.array) -> None:
        """Restore this system's state for envs where ``mask[env]`` is true."""
        pass
