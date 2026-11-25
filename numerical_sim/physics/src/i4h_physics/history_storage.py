# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only helper for a system's episode history as Newton State.

A package publishes one record per system before schema registration (``Layout``) and the system aliases the
finalized arrays at binding (``bind_history``). Templates carry each array's construction value; binding writes
nothing and the systems' own resets are unchanged.
"""
import newton
import numpy as np
import warp as wp
from i4h_physics.storage import Domain, Field, Owner, vector_view

RECORDS = "interaction_domains"


class Layout:
    """One system's fields: ``attribute -> (Field, vector dtype or None, exposed shape)``."""

    def __init__(self, scene, declarations, system_id, kind):
        self.scene, self.declarations, self.id, self.kind = scene, declarations, system_id, kind
        self.fields, self.domains = {}, {}

    def domain(self, label, shape, reference, owners):
        E = self.scene.envs
        d = Domain("system", f"{self.id}_{label}", f"i4h:{self.id}_{label}_rows", shape, reference)
        worlds = np.broadcast_to(np.arange(E, dtype=np.int32).reshape((E,) + (1,) * (len(shape) - 1)), shape)
        self.declarations.domain(d, worlds=worlds, owners=owners)
        self.domains[label] = d
        return d

    def world(self):
        d = Domain("system", self.id, newton.Model.AttributeFrequency.WORLD, (self.scene.envs,))
        self.declarations.domain(d)
        self.domains["world"] = d
        return d

    def add(self, key, d, dtype, *, default=None, values=None, flat=False, width=1):
        """A State field on ``d``. On another system's rows the name says whose history it is: the instrument
        rows already hold the control's own ``grasp_pending``. ``flat`` exposes one axis over all rows (kernels
        that index ``world * rows + row``); ``width`` stores that many scalar columns per row."""
        name = f"{self.id}_{key}" if d in self.domains.values() else f"{self.id}_history_{key}"
        shape = (d.count * width,) if flat else d.shape + ((width,) if width > 1 else ())
        f = Field(d, name, Owner.STATE, dtype, shape, "world, local row", "the system's existing selected-world reset",
                  row_width=width, attribute_name=name)
        self.declarations.add(f, values, default=default)
        self.fields[key] = (f, None, shape)

    def residue(self, d, rows, tools):
        """The per-tool vec3 shares of the body's particles, as scalar columns of existing particle rows."""
        for key in ("delta_k", "delta_jaw"):
            name = f"{self.id}_{key}"
            f = Field(d, name, Owner.STATE, wp.float32, d.shape + (3 * tools,), "world, body-local particle, tool",
                      "never cleared by reset; the apply kernel clears what it reduced", row_width=3 * tools, attribute_name=name)
            self.declarations.add(f, default=0.0)
            self.fields[key] = (f, wp.vec3, (rows, tools))

    def publish(self, **settings):
        record = dict(kind=self.kind, fields=self.fields, domains=self.domains, envs=self.scene.envs, **settings)
        self.declarations.register_package(records=((self.scene, RECORDS, {self.id: record}),))
        return record


def published(scene, system_id, kind, user):
    """An earlier system's record, for dependants that add history to its rows."""
    record = getattr(scene, RECORDS, {}).get(system_id)
    if record is None or record["kind"] != kind:
        raise ValueError(f"{user}: {system_id!r} is not a declared {kind}; declare it first")
    return record


def bind_history(system, kind, target=None, record_id=None, **expected):
    """Alias the declared history onto ``target`` (default: the system); binding writes nothing.

    ``expected`` are the capacities the system was built for.
    """
    scene = system.scene
    system.id = system.id or kind
    record_id = record_id or system.id
    record = getattr(scene, RECORDS, {}).get(record_id)
    if record is None or record["kind"] != kind:
        raise RuntimeError(f"{kind} {record_id!r} storage missing; call declare_{kind} before allocation and reconstruct the model")
    scene.storage.check()
    for key, value in expected.items():
        if record[key] != value and record[key] is not value:
            raise ValueError(f"{kind} {record_id!r}: {key} changed after declaration; reconstruct the model")
    target = system if target is None else target
    for key, (f, vector, shape) in record["fields"].items():
        array = scene.attr(f.domain, f.key)
        if vector is not None:
            array = vector_view(array, vector, shape)
        setattr(target, key, array)
        scene.storage.register_package(identities=((target, key, array),))
    return record
