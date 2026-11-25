# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint copy of i4h-owned rows between State and Control containers of one finalized Model.

The copied set is every declared State and Control field plus the scene's own rows of the native particle
position, velocity and force arrays. Rows are selected by world; foreign host rows and padding are never written.
"""
from typing import Any

import newton
import numpy as np
import warp as wp
from i4h_physics.storage import Owner

NATIVE = ("particle_q", "particle_qd", "particle_f")
FOREIGN = -2  # row map entry of a row this scene does not own


@wp.kernel
def copy_rows(world: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool), width: int,
              src: wp.array2d(dtype=Any), dst: wp.array2d(dtype=Any)):
    """Copy row ``r`` when its world is selected; shared rows (world -1) follow the last mask slot."""
    r = wp.tid()
    w = world[r]
    if w < -1:
        return
    if w < 0:
        w = mask.shape[0] - 1
    if mask[w]:
        for j in range(width):
            dst[r, j] = src[r, j]


def words(array):
    """``(rows, n)`` unsigned view of an array's rows without copying; keeps the backing array alive."""
    row_bytes = wp.types.type_size_in_bytes(array.dtype)
    dtype, size = (wp.uint32, 4) if row_bytes % 4 == 0 else (wp.uint8, 1)
    view = wp.array(ptr=array.ptr, dtype=dtype, shape=(array.shape[0], row_bytes // size), device=array.device,
                    capacity=array.shape[0] * row_bytes, copy=False)
    view._ref = array
    return view


def layout(array):
    return None if array is None else (array.ptr, array.shape, array.strides, array.dtype, array.device)


class Continuation:
    """Row maps and word views of one binding, built once at binding."""

    def __init__(self, scene):
        self.scene, self.storage = scene, scene.storage
        declarations, model, dev, E = self.storage.declarations, scene.model, scene.device, scene.envs
        self.state, self.control = self.storage.container(Owner.STATE), self.storage.container(Owner.CONTROL)
        particles = np.full(max(1, model.particle_count), FOREIGN, np.int32)
        if scene.main.width and model.particle_count:
            base, stride, col = scene.main.place(E)
            for world in range(E):
                start = base + col + world * stride
                particles[start:start + scene.main.width] = world
        maps = {}

        def particle_map():
            if "particles" not in maps:
                maps["particles"] = (wp.array(particles, dtype=wp.int32, device=dev), bool((particles != FOREIGN).all()))
            return maps["particles"]

        def row_map(domain, rows):
            if domain in maps:
                return maps[domain]
            if domain.custom:
                host = declarations.rows(domain)[0] if domain.count else np.full(1, FOREIGN, np.int32)
                device = getattr(model.i4h, f"{domain.frequency[4:]}_world") if domain.count else None
            elif domain.frequency == newton.Model.AttributeFrequency.PARTICLE:
                return particle_map()
            elif domain.frequency == newton.Model.AttributeFrequency.WORLD:
                # The optional trailing global row belongs to the host, not this scene.
                host, device = np.full(rows, FOREIGN, dtype=np.int32), None
                host[:E] = np.arange(E, dtype=np.int32)
            else:
                raise NotImplementedError(f"no world layout for native frequency {domain.frequency}")
            if host.shape[0] != rows:
                raise ValueError(f"row identities of {domain.instance} do not cover its backing rows")
            if device is None:
                device = wp.array(host, dtype=wp.int32, device=dev)
            maps[domain] = (device, bool((host != FOREIGN).all()))
            return maps[domain]

        # (owner, key or native name, row map, whole-array copy allowed without a mask, bound array, bound words)
        self.entries = []
        for key, field in declarations.fields.items():
            if field.owner is Owner.MODEL:
                continue
            array = self.storage.backing(key)
            world, whole = row_map(field.domain, array.shape[0])
            self.entries.append((field.owner, key, world, whole, array, words(array)))
        if model.particle_count:
            world, whole = particle_map()
            for name in NATIVE:
                array = getattr(self.state, name)
                if array is not None:
                    self.entries.append((Owner.STATE, name, world, whole, array, words(array)))
        self._all = wp.ones(E + 1, dtype=wp.bool, device=dev)
        #: the bound Control is the Model's template (``model.control(clone_variables=False)``)
        self.shared = any(owner is Owner.CONTROL and bound.size and bound.ptr == getattr(model.i4h, declarations.name(key)).ptr
                          for owner, key, _, _, bound, _ in self.entries)
        owned = [device for device, _ in maps.values() if not any(device is a for a in vars(model.i4h).values())]
        #: device bytes this object allocated: its row maps and the all-worlds mask
        self.allocated_bytes = sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in (*owned, self._all))

    def _other(self, state, control, save):
        """Validated arrays and word views of another State and Control of this Model. Nothing is retained: the
        views do not own memory, and a host may allocate checkpoints freely."""
        if state is self.state or control is self.control:
            raise ValueError("continuation needs a State and Control other than the bound ones")
        self.storage.check_containers(state, control)
        if self.shared and not save:
            raise ValueError("the bound Control shares its buffers with the Model's initialization templates; "
                             "restoring into it would rewrite them")
        model, arrays = self.scene.model, []
        for owner, key, _, _, bound, _ in self.entries:
            container = state if owner is Owner.STATE else control
            array = getattr(container, key, None) if isinstance(key, str) else self.storage.backing(key, container)
            if layout(array) is None or layout(array)[1:] != layout(bound)[1:]:
                raise ValueError(f"incompatible continuation buffer: {key}")
            if owner is Owner.CONTROL and array.ptr == getattr(model.i4h, self.storage.declarations.name(key)).ptr:
                raise ValueError("this Control shares its buffers with the Model's initialization templates; "
                                 "continuation needs a Control created with clone_variables=True")
            arrays.append(array)
        return arrays, [words(a) for a in arrays]

    def copy(self, state, control, mask, save):
        """``save``: bound containers to ``state`` and ``control``; otherwise the reverse. ``mask`` is ``(E+1,)``
        bool or None for every row this scene owns."""
        arrays, views = self._other(state, control, save)
        dev = self.scene.device
        for (_, _, world, whole, bound, bound_words), other, other_words in zip(self.entries, arrays, views):
            if not bound.size:
                continue
            if mask is None and whole:
                wp.copy(other if save else bound, bound if save else other)
                continue
            src, dst = (bound_words, other_words) if save else (other_words, bound_words)
            wp.launch(copy_rows, dim=world.shape[0], inputs=[world, self._all if mask is None else mask, src.shape[1], src], outputs=[dst], device=dev)
