# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Instrument-owned declarations and checked aliases of Newton storage.

Local sphere/primitive indices remain kernel-local integers. Only identity
columns carry Newton references. Binding never initializes physical history.
"""
from functools import partial
from types import SimpleNamespace
from weakref import ref

import newton
import numpy as np
import warp as wp
from i4h_physics.storage import Domain, Field, Owner, defer_declaration
from i4h_surgical.systems.instruments.geometry import InstrumentLayout

LIVE_FIELDS = ("max_velocity", "energy_threshold", "fire_threshold")
STATE_TYPES = {
    "root": wp.vec3, "shaft_q": wp.quat, "wrist_q": wp.quat, "jaw_q": wp.quat,
    "jaw_angle": wp.float32, "jaw_delta": wp.float32,
    "held": wp.int32, "grasp_pending": wp.int32, "released": wp.int32,
    "energized": wp.int32, "fire_on": wp.int32, "fired": wp.int32,
    "sphere_q": wp.vec3, "sphere_q_prev": wp.vec3, "sphere_active": wp.int32,
}


def geometry_values(scene):
    """Host values only; the same flattened local geometry used by the kernels."""
    records = list(getattr(scene, "instruments", {}).values())
    n = len(records)
    rest = np.array([r.initial_position for r in records], np.float32).reshape(n, 3)
    zero3 = np.zeros(3, np.float32)
    layout = InstrumentLayout(records)
    specs, owner = layout.specs, layout.owner
    prim_a, prim_b = layout.prim_a, layout.prim_b
    prim_owner, prim_kind, prim_starts = layout.prim_owner, layout.prim_kind, layout.prim_starts
    jaw_root_a, jaw_root_b = layout.jaw_root_a, layout.jaw_root_b
    S, P = len(specs), len(prim_a)
    instrument, sphere, primitive = "instrument", "sphere", "primitive"

    def floats(values, dtype, tail=()):
        return dtype, np.asarray(values, np.float32).reshape((n,) + tail)

    def ints(values):
        return wp.int32, np.asarray(values, np.int32).reshape(n)

    def sfloats(values, dtype, tail=()):
        return dtype, np.asarray(values, np.float32).reshape((S,) + tail)

    def sints(values, count=S):
        return wp.int32, np.asarray(values, np.int32).reshape(count)

    values = {}
    values["trocar"] = (instrument, floats([r.trocar if r.trocar is not None else zero3 for r in records], wp.vec3, (3,)))
    values["has_trocar"] = (instrument, ints([r.trocar is not None for r in records]))
    values["initial_position"] = (instrument, floats(rest, wp.vec3, (3,)))
    values["initial_rotation"] = (instrument, floats([r.initial_rotation for r in records], wp.quat, (4,)))
    values["jaw_open"] = (instrument, floats([r.jaw_open for r in records], wp.float32))
    values["jaw_closed"] = (instrument, floats([r.jaw_closed for r in records], wp.float32))
    values["jaw_response"] = (instrument, floats([r.jaw_response for r in records], wp.float32))
    values["grasp_threshold"] = (instrument, floats([r.grasp_threshold for r in records], wp.float32))
    values["wrist_pivot"] = (instrument, floats([r.wrist_pivot for r in records], wp.vec3, (3,)))
    values["jaw_anchor"] = (instrument, floats([r.jaw_anchor for r in records], wp.vec3, (3,)))
    values["wrist_end"] = (instrument, floats([r.wrist_end if r.wrist_end is not None else zero3 for r in records], wp.vec3, (3,)))
    values["has_wrist_end"] = (instrument, ints([r.wrist_end is not None for r in records]))
    values["wrist_end_bind"] = (instrument, floats([r.wrist_end_bind for r in records], wp.mat44, (4, 4)))
    values["shaft_spacing"] = (instrument, floats([r.shaft_spacing for r in records], wp.float32))
    values["prim_start"] = (instrument, ints(prim_starts))
    values["prim_counts"] = (instrument, ints([len(r.prims) for r in records]))
    values["jaw_root_a"] = (instrument, ints(jaw_root_a))
    values["jaw_root_b"] = (instrument, ints(jaw_root_b))
    values["sphere_instrument"] = (sphere, sints(owner))
    values["sphere_role"] = (sphere, sints([s.role for s in specs]))
    values["sphere_jaw_sign"] = (sphere, sfloats([s.jaw_sign for s in specs], wp.float32))
    values["sphere_local"] = (sphere, sfloats([s.local for s in specs], wp.vec3, (3,)))
    values["sphere_bind"] = (sphere, sfloats([s.bind for s in specs], wp.mat44, (4, 4)))
    values["sphere_radius"] = (sphere, sfloats([s.radius for s in specs], wp.float32))
    values["sphere_chain"] = (sphere, sints([s.chain for s in specs]))
    values["sphere_chain_index"] = (sphere, sints([s.chain_index for s in specs]))
    values["sphere_chain_count"] = (sphere, sints([s.chain_count for s in specs]))
    values["prim_a"] = (primitive, sints(prim_a, P))
    values["prim_b"] = (primitive, sints(prim_b, P))
    values["prim_instrument"] = (primitive, sints(prim_owner, P))
    values["prim_kind"] = (primitive, sints(prim_kind, P))
    return values, (n, S, P), tuple(r.name for r in records)


def layout_signature(scene):
    values, counts, names = geometry_values(scene)
    return (scene.envs, counts, names, tuple((key, domain, dtype, data.shape, data.tobytes())
            for key, (domain, (dtype, data)) in values.items()))


def declare_instruments(scene, *, system_id="instrument_control", max_velocity=2.5,
                        energy_threshold=0.5, fire_threshold=0.5):
    """Declare a direct Scene's instruments before allocation/schema registration."""
    if not isinstance(system_id, str) or not system_id.isidentifier():
        raise ValueError("instrument system_id must be a nonempty identifier")
    defer_declaration(scene, system_id, publish_instruments,
                      dict(max_velocity=max_velocity, energy_threshold=energy_threshold, fire_threshold=fire_threshold,
                           expected_signature=layout_signature(scene)))


def publish_instruments(scene, declarations, system_id, *, max_velocity=2.5,
                        energy_threshold=0.5, fire_threshold=0.5, expected_signature=None):
    if expected_signature is not None and layout_signature(scene) != expected_signature:
        raise RuntimeError("instrument layout changed after declaration; reconstruct the model")
    values, (N, S, P), names = geometry_values(scene)
    E = scene.envs
    domains = {}

    def domain(kind, shape, owners, reference, worlds):
        d = Domain("system", f"{system_id}_{kind}", f"i4h:{system_id}_{kind}_rows", shape, reference)
        declarations.domain(d, owners=np.asarray(owners, np.int32).reshape(shape),
                            worlds=np.broadcast_to(np.asarray(worlds, np.int32), shape))
        domains[kind] = d
        return d

    root = domain("instrument", (N,), np.full(N, -1), "world", -1)
    sphere = domain("sphere", (S,), values["sphere_instrument"][1][1], root.frequency, -1)
    domain("primitive", (P,), values["prim_instrument"][1][1], root.frequency, -1)
    domain("runtime", (E, N), np.tile(np.arange(N), (E, 1)), root.frequency, np.arange(E)[:, None])
    domain("sphere_runtime", (E, S), np.tile(np.arange(S), (E, 1)), sphere.frequency, np.arange(E)[:, None])
    world = Domain("system", system_id, newton.Model.AttributeFrequency.WORLD, (E,))
    declarations.domain(world)
    fields = {}

    def add(key, d, owner, dtype, data, width=1):
        shape = d.shape + ((width,) if width != 1 else ())
        f = Field(d, key, owner, dtype, shape, "world, local row" if owner != Owner.MODEL else "local geometry or world",
                  "selected-world existing reset kernels" if owner != Owner.MODEL else "preserve live Model values",
                  row_width=width, attribute_name=f"{system_id}_{key}")
        declarations.add(f, data)
        fields[key] = f

    for key, (kind, (dtype, data)) in values.items():
        add(key, domains[kind], Owner.MODEL, dtype, data)
    settings = dict(max_velocity=max_velocity, energy_threshold=energy_threshold, fire_threshold=fire_threshold)
    for key, value in settings.items():
        # These parameters are per-world scalars, exactly as the pose kernel expects.
        if np.asarray(value).ndim != 0:
            raise ValueError(f"{key} must be a scalar declaration; edit its WORLD array after binding")
        add(key, world, Owner.MODEL, wp.float32, np.full(E, value, np.float32))
    for key, dtype in STATE_TYPES.items():
        d = domains["sphere_runtime" if key.startswith("sphere_") else "runtime"]
        add(key, d, Owner.STATE, dtype, np.zeros(d.shape + tuple(getattr(dtype, "_shape_", ())),
                                               np.int32 if dtype is wp.int32 else np.float32))
    add("actions", domains["runtime"], Owner.CONTROL, wp.float32, np.zeros((E, N, 10), np.float32), 10)
    from i4h_physics.mount_storage import publish_mountable
    poses = np.tile(np.concatenate((values["initial_position"][1][1], values["initial_rotation"][1][1]), axis=1),
                    (E, 1)).reshape(E, N, 7)
    mountable = publish_mountable(scene, declarations, system_id, names, root, domains["runtime"], fields["root"], poses)
    fields.update(mounted=mountable.mounted, mount_pose=mountable.mount_pose)
    record = dict(fields=fields, domains=domains, settings=tuple(settings.items()), signature=layout_signature(scene),
                  counts=(N, S, P), names=names, envs=E)
    declarations.register_package(records=((scene, "instrument_domains", {system_id: record}),))
    declarations.initializer(partial(initialize_instruments, scene_ref=ref(scene), record=record))
    return record


def validate_layout(scene, record):
    if layout_signature(scene) != record["signature"]:
        raise RuntimeError("instrument layout changed; reconstruct the model")


def initialize_instruments(model, *, scene_ref, record):
    """Resolve this Model's own templates; never retain its arrays in the builder."""
    from i4h_surgical.systems.instruments.instrument_control import reset_instruments
    scene = scene_ref()
    if scene is not None:
        validate_layout(scene, record)
    E = record["envs"]
    fields = {}
    for key, f in record["fields"].items():
        backing = getattr(model.i4h, f.name)
        fields[key] = backing.view(f.dtype).flatten()[:int(np.prod(f.shape))].reshape(f.shape)
    N, S, _ = record["counts"]
    runtime = SimpleNamespace(**fields, count=N, sphere_count=S)
    mask = wp.ones(E, dtype=wp.bool, device=model.device)
    reset_instruments(runtime, SimpleNamespace(envs=E, device=model.device), mask)


def bind_instruments(system):
    scene, sid = system.scene, system.id
    record = getattr(scene, "instrument_domains", {}).get(sid)
    if record is None:
        raise RuntimeError(f"instrument {sid!r} storage missing; call declare_instruments before allocation")
    validate_layout(scene, record)
    scene.storage.check()
    if tuple((key, getattr(system, key)) for key in LIVE_FIELDS) != record["settings"]:
        raise ValueError("instrument declaration settings changed; reconstruct the model")
    fields = SimpleNamespace(**{key: scene.attr(f.domain, f.key) for key, f in record["fields"].items()})
    system.fields = fields
    scene.storage.register_package(identities=((system, "fields", fields),))
    for key in record["fields"]:
        array = getattr(fields, key)
        setattr(system, key, array)
        scene.storage.register_package(identities=((system, key, array), (fields, key, array)))
    scene.storage.register_package(checks=(partial(validate_layout, scene, record),))
