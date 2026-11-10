# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Read-only completed-step exports. Domains contain output layout, never physics state.

Rows are world-major, then authored system order, then local tool/capsule/slot.
Surgical records retain recording-event geometry; reconstructed points use the
completed-step positions and may belong to tissue cut after recording.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
import json

import newton
import warp as wp
from newton.solvers import SolverBase


class ObservableKind(SolverBase.ObservableKind):
    I4H_WRENCH = "i4h_wrench"
    I4H_CONTACT_DEPTH = "i4h_contact_depth"
    I4H_SURGICAL_CONTACT = "i4h_surgical_contact"
    I4H_SURGICAL_COUNT = "i4h_surgical_count"


@wp.struct
class ContactDepth:
    depth: wp.float32
    count: wp.int32


@wp.struct
class SurgicalCount:
    found: wp.int32
    valid: wp.int32
    capacity: wp.int32
    overflow: wp.int32  # final recording event only, not the source's frame sum


@wp.struct
class SurgicalContact:
    valid: wp.int32
    triangle: wp.int32
    tet: wp.int32
    vertices: wp.vec3i
    barycentric: wp.vec3
    normal: wp.vec3
    penetration: wp.float32
    capsule_coordinate: wp.float32
    surface_point_completed: wp.vec3
    capsule_point_completed: wp.vec3


@dataclass(eq=False)
class Observables(SolverBase.Observables):
    i4h_wrench: wp.array = SolverBase.Observables.field(dtype=wp.spatial_vector, frequency="i4h:output_tool")
    i4h_contact_depth: wp.array = SolverBase.Observables.field(dtype=ContactDepth, frequency="i4h:output_depth")
    i4h_surgical_contact: wp.array = SolverBase.Observables.field(dtype=SurgicalContact, frequency="i4h:output_record")
    i4h_surgical_count: wp.array = SolverBase.Observables.field(dtype=SurgicalCount, frequency="i4h:output_capsule")

    def select(self, kinds):
        result = super().select(kinds)
        result._identities = identities(result)
        return result


KINDS = frozenset((ObservableKind.I4H_WRENCH, ObservableKind.I4H_CONTACT_DEPTH,
                   ObservableKind.I4H_SURGICAL_CONTACT, ObservableKind.I4H_SURGICAL_COUNT))
DOMAINS = ("tool", "depth", "capsule", "record")


def declare_layout(builder, scene, specs, constructors):
    """Only host metadata is registered, before finalization. No runtime allocation."""
    layout = {name: [] for name in DOMAINS}
    for spec in specs:
        cls = constructors[spec["type"]]
        declare = getattr(cls, "output_layout", None)
        if declare is None:
            continue
        sid = spec.get("id", spec["type"])
        for domain, rows in declare(scene, spec).items():
            layout[domain].extend([[sid, *row] for row in rows])
    # String metadata is host-owned and does not create device state/templates.
    builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
        name="output_layout", namespace="i4h", dtype=str,
        frequency=newton.Model.AttributeFrequency.ONCE,
        assignment=newton.Model.AttributeAssignment.MODEL, default=json.dumps(layout)))
    for domain in DOMAINS:
        name = f"output_{domain}"
        builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=name, namespace="i4h"))
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=name + "_label", namespace="i4h", dtype=str, frequency=f"i4h:{name}",
            assignment=newton.Model.AttributeAssignment.MODEL, default=""))
    return layout


def add_world_layout(builder, layout, world):
    for domain, rows in layout.items():
        for row in rows:
            builder.add_custom_values(**{f"i4h:output_{domain}_label": json.dumps([world, *row])})


def identities(container):
    """Snapshot nested observable containers and every array, before capture."""
    result = []
    for name, value in vars(container).items():
        if isinstance(value, (wp.array, SolverBase.Observables)):
            result.append((container, name, value))
            if isinstance(value, SolverBase.Observables):
                result.extend(identities(value))
    return result


def check_identities(entries):
    for owner, name, value in entries:
        if getattr(owner, name) is not value:
            raise RuntimeError(f"observable {name} buffer or container was replaced; bind a fresh runtime")


class OutputPublisher:
    """Cached source views, read only. All allocations precede graph capture."""

    def __init__(self, solver):
        scene = solver.scene
        self.scene = scene
        meta = getattr(getattr(solver.model, "i4h", None), "output_layout", None)
        self.layout = json.loads(meta[0]) if meta else {name: [] for name in DOMAINS}
        self.widths = {k: len(v) for k, v in self.layout.items()}
        by_id = {s.id: s for s in solver.systems}
        self.wrenches, self.depths, self.contacts = [], [], []
        for sid in dict.fromkeys(row[0] for row in self.layout["tool"]):
            control = by_id[sid]
            rows = [i for i, row in enumerate(self.layout["tool"]) if row[0] == sid]
            sources = [s for s in solver.systems if (s is control or getattr(s, "instruments", None) is control)
                       and hasattr(s, "impulse") and hasattr(s, "moment")]
            # The system list is authoritative; reject duplicate instances instead of double counting.
            if len({id(s) for s in sources}) != len(sources):
                raise ValueError("duplicate wrench source instance")
            self.wrenches.append((control, sources, rows[0], len(rows)))
        for i, (sid, *_) in enumerate(self.layout["depth"]):
            self.depths.append((by_id[sid], i))
        for sid in dict.fromkeys(row[0] for row in self.layout["capsule"]):
            source = by_id[sid]
            capsule_offset = next(i for i, row in enumerate(self.layout["capsule"]) if row[0] == sid)
            record_offset = next((i for i, row in enumerate(self.layout["record"]) if row[0] == sid), 0)
            self.contacts.append((source, capsule_offset, record_offset, scene.field("q", source.body)))
        self._identities = []
        for control, sources, _, _ in self.wrenches:
            for source in sources:
                self._identities.extend((source, name, getattr(source, name)) for name in ("impulse", "moment"))
        for source, _ in self.depths:
            self._identities.extend((source, name, getattr(source, name)) for name in ("contact_depth", "contact_count"))
        for source, _, _, _ in self.contacts:
            self._identities.extend((source, name, getattr(source, name)) for name in
                                    ("record_count", "rec_tri", "rec_tet", "rec_verts", "rec_bary", "rec_pen", "rec_t", "rec_normal"))
            ic = getattr(source, "instruments")
            self._identities.extend((ic, name, getattr(ic, name)) for name in ("sphere_q", "prim_a", "prim_b"))

    def initialize(self, obs):
        obs.row_keys = MappingProxyType({
            kind: tuple(tuple(json.loads(label)) for label in getattr(self.scene.model.i4h, f"output_{domain}_label", ()))
            for kind, domain in ((ObservableKind.I4H_WRENCH, "tool"), (ObservableKind.I4H_CONTACT_DEPTH, "depth"),
                                 (ObservableKind.I4H_SURGICAL_CONTACT, "record"), (ObservableKind.I4H_SURGICAL_COUNT, "capsule"))
            if kind in obs.kinds})
        if obs.i4h_surgical_count is not None:
            for source, offset, _, _ in self.contacts:
                wp.launch(copy_counts, dim=source.record_count.shape,
                          inputs=[source.record_count, source.records, 0, offset, self.widths["capsule"]],
                          outputs=[obs.i4h_surgical_count], device=self.scene.device)

    def publish(self, obs, dt, substeps):
        if obs is None:
            return
        E, dev = self.scene.envs, self.scene.device
        if obs.i4h_wrench is not None:
            obs.i4h_wrench.zero_()
            for control, sources, offset, count in self.wrenches:
                if not control.enabled:
                    continue
                for source in sources:
                    if source.enabled:
                        wp.launch(add_wrench, dim=(E, count),
                                  inputs=[source.impulse, source.moment, 1.0 / ((dt / substeps) * dt),
                                          offset, self.widths["tool"]], outputs=[obs.i4h_wrench], device=dev)
        if obs.i4h_contact_depth is not None:
            obs.i4h_contact_depth.zero_()
            for source, offset in self.depths:
                if source.enabled:
                    wp.launch(copy_depth, dim=E, inputs=[source.contact_depth, source.contact_count, offset, self.widths["depth"]],
                              outputs=[obs.i4h_contact_depth], device=dev)
        if obs.i4h_surgical_count is not None:
            for source, offset, _, _ in self.contacts:
                wp.launch(copy_counts, dim=source.record_count.shape,
                          inputs=[source.record_count, source.records, int(source.enabled), offset, self.widths["capsule"]],
                          outputs=[obs.i4h_surgical_count], device=dev)
        if obs.i4h_surgical_contact is not None:
            for source, _, offset, q in self.contacts:
                ic = getattr(source, "instruments")
                wp.launch(copy_records, dim=source.rec_tri.shape,
                          inputs=[source.record_count, int(source.enabled), source.rec_tri, source.rec_tet, source.rec_verts,
                                  source.rec_bary, source.rec_pen, source.rec_t, source.rec_normal, q, ic.sphere_q, ic.prim_a, ic.prim_b,
                                  offset, self.widths["record"]], outputs=[obs.i4h_surgical_contact], device=dev)

    def clear(self, obs, mask):
        for kind, (name, _) in obs._observable_fields().items():
            array = getattr(obs, name)
            if kind in KINDS and array is not None and array.size:
                wp.launch(clear_counts if kind == ObservableKind.I4H_SURGICAL_COUNT else clear_worlds, dim=array.size, inputs=[mask, array.size // self.scene.envs],
                          outputs=[array], device=self.scene.device)


@wp.kernel
def add_wrench(impulse: wp.array2d(dtype=wp.vec3), moment: wp.array2d(dtype=wp.vec3), scale: float,
               offset: int, width: int, out: wp.array(dtype=wp.spatial_vector)):
    e, k = wp.tid()
    row = e * width + offset + k
    out[row] = out[row] - wp.spatial_vector(impulse[e, k] * scale, moment[e, k] * scale)


@wp.kernel
def copy_depth(depth: wp.array(dtype=float), count: wp.array(dtype=int), offset: int, width: int,
               out: wp.array(dtype=ContactDepth)):
    e = wp.tid()
    value = ContactDepth()
    value.depth = depth[e]
    value.count = count[e]
    out[e * width + offset] = value


@wp.kernel
def copy_counts(count: wp.array2d(dtype=int), capacity: int, enabled: int, offset: int, width: int,
                out: wp.array(dtype=SurgicalCount)):
    e, p = wp.tid()
    value = SurgicalCount()
    value.capacity = capacity
    if enabled != 0:
        value.found = count[e, p]
        value.valid = wp.min(value.found, capacity)
        value.overflow = wp.max(value.found - capacity, 0)
    out[e * width + offset + p] = value


@wp.kernel
def copy_records(count: wp.array2d(dtype=int), enabled: int,
                 tri: wp.array3d(dtype=int), tet: wp.array3d(dtype=int), vertices: wp.array3d(dtype=wp.vec3i),
                 bary: wp.array3d(dtype=wp.vec3), pen: wp.array3d(dtype=float), coord: wp.array3d(dtype=float),
                 normal: wp.array3d(dtype=wp.vec3), q: wp.array2d(dtype=wp.vec3), sphere: wp.array2d(dtype=wp.vec3),
                 a: wp.array(dtype=int), b: wp.array(dtype=int), offset: int, width: int,
                 out: wp.array(dtype=SurgicalContact)):
    e, p, r = wp.tid()
    value = SurgicalContact()
    value.triangle = -1
    value.tet = -1
    value.vertices = wp.vec3i(-1)
    if enabled != 0 and r < wp.min(count[e, p], tri.shape[2]):
        value.valid = 1
        value.triangle = tri[e, p, r]
        value.tet = tet[e, p, r]
        value.vertices = vertices[e, p, r]
        value.barycentric = bary[e, p, r]
        value.normal = normal[e, p, r]
        value.penetration = pen[e, p, r]
        value.capsule_coordinate = coord[e, p, r]
        v = value.vertices
        bc = value.barycentric
        value.surface_point_completed = q[e, v[0]] * bc[0] + q[e, v[1]] * bc[1] + q[e, v[2]] * bc[2]
        value.capsule_point_completed = wp.lerp(sphere[e, a[p]], sphere[e, b[p]], value.capsule_coordinate)
    out[e * width + offset + p * tri.shape[2] + r] = value


@wp.kernel
def clear_worlds(mask: wp.array(dtype=wp.bool), width: int, out: wp.array(dtype=Any)):
    i = wp.tid()
    if mask[i // width]:
        out[i] = type(out[i])()


@wp.kernel
def clear_counts(mask: wp.array(dtype=wp.bool), width: int, out: wp.array(dtype=SurgicalCount)):
    i = wp.tid()
    if mask[i // width]:
        value = SurgicalCount()
        value.capacity = out[i].capacity
        out[i] = value
