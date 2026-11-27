# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save and restore everything a rod step can change, so a substep can be re-solved.

Coupling iterations re-run the rod from the start of a substep with a new guess
for the holder's force. The Newton state carries only particle positions and
velocities; orientations, constraint multipliers, containment and friction
state, the pending root command and an attached vessel's deformation live on
the solver and the objects it owns. Rather than enumerate them, which would go
stale the first time a hook gains a buffer, this walks every object of this
package reachable from the rod and records what it holds.

Restoration is in place. Kernels, CUDA graphs and other objects hold references
to the solver's arrays, so contents are copied back into the same arrays and
attribute bindings are reset to the objects they named when saved, which also
undoes a step that swapped two buffers.
"""

from __future__ import annotations

import numpy as np
import warp as wp

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency of the solver
    torch = None

_PACKAGE = __name__.split(".", 1)[0]


def _is_tensor(value) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


class RodStateSnapshot:
    """One reusable save slot for a rod solver and everything of this package it owns.

    Call :meth:`save` at the start of a substep and :meth:`restore` before each
    re-solve. Shadow buffers are allocated on first sight of an array and reused,
    so after the first substep saving costs device copies and a Python walk.

    Args:
        root: The rod solver. Walked whatever its type; objects it reaches are
            walked only when they belong to this package.
    """

    def __init__(self, root) -> None:
        self.root = root
        self._shadows: dict[int, tuple] = {}
        self._bindings: list[tuple] = []
        self._containers: list[tuple] = []
        self._arrays: dict[int, object] = {}
        self._meshes: dict[int, object] = {}

    def _owned(self, value) -> bool:
        return (
            hasattr(value, "__dict__") and not isinstance(value, type) and not callable(value)
            and type(value).__module__.split(".", 1)[0] == _PACKAGE
        )

    def _visit_value(self, value, seen: set) -> None:
        if isinstance(value, wp.array):
            self._arrays[id(value)] = value
        elif _is_tensor(value):
            self._arrays[id(value)] = value
        elif isinstance(value, np.ndarray):
            if value.flags.writeable:
                self._arrays[id(value)] = value
        elif isinstance(value, wp.Mesh):
            self._meshes[id(value)] = value
        elif isinstance(value, (list, dict)):
            if id(value) in seen:
                return
            seen.add(id(value))
            self._containers.append((value, list(value) if isinstance(value, list) else dict(value)))
            for item in (value.values() if isinstance(value, dict) else value):
                self._visit_value(item, seen)
        elif isinstance(value, tuple):
            for item in value:
                self._visit_value(item, seen)
        elif self._owned(value):
            self._visit_object(value, seen)

    def _visit_object(self, obj, seen: set) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        attrs = vars(obj)
        self._bindings.append((obj, dict(attrs)))
        for value in attrs.values():
            self._visit_value(value, seen)

    def save(self) -> None:
        """Record the rod's current state, overwriting the previous save."""
        self._bindings, self._containers = [], []
        self._arrays, self._meshes = {}, {}
        self._visit_object(self.root, set())

        shadows = {}
        for key, array in self._arrays.items():
            held = self._shadows.get(key)
            if held is None or held[0] is not array:
                held = (array, self._allocate(array))
            self._store(array, held[1])
            shadows[key] = held
        self._shadows = shadows

    def restore(self) -> None:
        """Put the rod back exactly as it was at the last :meth:`save`."""
        if not self._bindings:
            raise RuntimeError("RodStateSnapshot.restore() called before save()")
        for obj, attrs in self._bindings:
            current = vars(obj)
            for name in [name for name in current if name not in attrs]:
                delattr(obj, name)
            current.update(attrs)
        for container, contents in self._containers:
            if isinstance(container, list):
                container[:] = contents
            else:
                container.clear()
                container.update(contents)
        for array, shadow in self._shadows.values():
            self._load(array, shadow)
        for mesh in self._meshes.values():
            mesh.refit()

    @staticmethod
    def _allocate(array):
        if isinstance(array, wp.array):
            return wp.empty_like(array)
        if isinstance(array, np.ndarray):
            return np.empty_like(array)
        return torch.empty_like(array)

    @staticmethod
    def _store(array, shadow) -> None:
        if isinstance(array, wp.array):
            wp.copy(shadow, array)
        elif isinstance(array, np.ndarray):
            np.copyto(shadow, array)
        else:
            shadow.copy_(array)

    @staticmethod
    def _load(array, shadow) -> None:
        if isinstance(array, wp.array):
            wp.copy(array, shadow)
        elif isinstance(array, np.ndarray):
            np.copyto(array, shadow)
        else:
            array.copy_(shadow)


__all__ = ["RodStateSnapshot"]
