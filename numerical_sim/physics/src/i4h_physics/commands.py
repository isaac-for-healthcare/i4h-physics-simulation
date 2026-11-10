# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Package-specified, checked final-axis aliases of Newton Control storage."""
from collections.abc import Mapping
from types import MappingProxyType

import warp as wp

from i4h_physics.storage import Owner


def specification(system):
    spec = system.command_view_spec
    if not isinstance(spec, Mapping) or not spec or not system.is_control:
        raise ValueError("command_view_spec must be a nonempty mapping on a control system")
    result = []
    for name, bounds in spec.items():
        if (not isinstance(name, str) or not name or not isinstance(bounds, tuple) or len(bounds) != 2
                or any(not isinstance(n, int) or isinstance(n, bool) for n in bounds) or not 0 <= bounds[0] < bounds[1]):
            raise ValueError("command views require names and integer half-open (start, stop) slices")
        result.append((name, bounds))
    return tuple(result)


def layout(array):
    return array.ptr, array.dtype, array.device, array.shape, array.strides, array.capacity


class CommandViews:
    def __init__(self, scene, system):
        self.system, self.scene = system, scene
        self.spec = specification(system)
        self.actions = actions = getattr(system, "actions", None)
        if (not isinstance(actions, wp.array) or actions.dtype != wp.float32 or actions.device != scene.device
                or actions.ndim < 2 or actions.shape[0] != scene.envs or not actions.is_contiguous):
            raise ValueError("command actions require contiguous world-major float32 storage on the Model device")
        if any(stop > actions.shape[-1] for _, (_, stop) in self.spec):
            raise ValueError("command view slice exceeds actions' final axis")
        owned = [f for f in scene.declarations.fields.values()
                 if f.owner is Owner.CONTROL and scene.attr(f.domain, f.key) is actions]
        if len(owned) != 1:
            raise ValueError("command actions must alias one declared Newton Control field")
        self.field = owned[0]
        self.backing = getattr(scene.control.i4h, self.field.name)
        self.actions_layout = layout(actions)
        views = {}
        for name, (start, stop) in self.spec:
            if start == 0 and stop == actions.shape[-1]:
                view = actions
            else:
                offset = start * actions.strides[-1]
                view = wp.array(ptr=actions.ptr + offset, dtype=actions.dtype,
                                shape=(*actions.shape[:-1], stop-start), strides=actions.strides,
                                capacity=max(0, self.backing.capacity-offset), device=actions.device, copy=False)
                view._ref = self.backing
            views[name] = view
        self.views = MappingProxyType(views)
        self.layouts = {key: layout(a) for key, a in views.items()}
        self.tensors = None
        self.tensor_layouts = None

    def check(self):
        if (self.system.command_view_spec is None or specification(self.system) != self.spec
                or self.system.actions is not self.actions or layout(self.actions) != self.actions_layout
                or getattr(self.scene.control.i4h, self.field.name) is not self.backing):
            raise RuntimeError("command specification or backing changed; bind a fresh runtime")
        if any(layout(a) != self.layouts[key] for key, a in self.views.items()):
            raise RuntimeError("command view layout changed; bind a fresh runtime")
        if self.tensors is not None:
            if any(tensor_layout(t) != self.tensor_layouts[key] for key, t in self.tensors.items()):
                raise RuntimeError("command tensor layout changed; bind a fresh runtime")

    def torch_views(self):
        self.check()
        if self.tensors is None:
            import torch
            tensors = {}
            for key, array in self.views.items():
                tensor = wp.to_torch(array)
                expected_device = torch.device(f"cuda:{array.device.ordinal}" if array.device.is_cuda else "cpu")
                if (tensor.dtype != torch.float32 or tensor.device != expected_device
                        or tuple(tensor.shape) != array.shape or tuple(s * 4 for s in tensor.stride()) != array.strides
                        or (tensor.numel() and tensor.data_ptr() != array.ptr)):
                    raise ValueError("command tensor must preserve dtype, device, shape, strides and backing")
                tensors[key] = tensor
            self.tensors = MappingProxyType(tensors)
            self.tensor_layouts = {key: tensor_layout(t) for key, t in tensors.items()}
        return self.tensors


def tensor_layout(tensor):
    return (tensor.data_ptr(), tensor.dtype, tensor.device, tuple(tensor.shape), tensor.stride(),
            tensor.storage_offset(), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
