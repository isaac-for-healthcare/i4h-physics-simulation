# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physics-owned rod declarations. Kernel indices remain body-local integers."""

from functools import partial
from types import SimpleNamespace
from weakref import ref

import numpy as np
import warp as wp
from i4h_physics.storage import Domain, Field, Owner, defer_declaration, scene_record

DIRECT_LIVE = ("young", "torsion", "bend", "twist", "angular_damping")
PBD_LIVE = ("stretch", "bend", "twist", "angular_damping")
DIRECT_REGIONS = dict(
    young_multiplier=1.0,
    torsion_multiplier=1.0,
    bend_multiplier=1.0,
    twist_multiplier=1.0,
    curvature_x=0.0,
    curvature_y=0.0,
)
PBD_REGIONS = dict(stretch=1.0, bend=1.0, twist=1.0)


def geometry(body, backend):
    from i4h_physics.systems.rods.rod_pbd import chain_topology, rest_darboux_quats

    if body.rod is not None:
        m = body.rod
        if m.frames.shape[0] != 1:
            raise ValueError("rod systems support a single mesh variant")
        seg, pair = chain_topology(body.count)
        frames, inv_mass = m.frames, m.inv_mass_q
        if backend == "pbd":
            frames, inv_mass = frames[:, :-1].copy(), inv_mass[:, :-1].copy()
        rd = m.rest_darboux if backend == "direct" else np.stack([rest_darboux_quats(f, pair) for f in frames])
    elif body.tree is not None and backend == "pbd":
        m = body.tree
        if m.frames.shape[0] != 1:
            raise ValueError("rod systems support a single mesh variant")
        seg, pair = m.edges[0], m.bend_pairs[0]
        frames, inv_mass = m.frames, m.inv_mass_q
        rd = np.stack([rest_darboux_quats(f, pair) for f in frames])
    else:
        raise ValueError(f"rod_{backend}: body needs a compatible rod or centreline tree")
    return dict(
        frames0=(wp.quat, frames),
        inv_mass_q=(wp.float32, inv_mass),
        rest_len=(wp.float32, m.rest_len),
        rest_darboux=(wp.vec3 if backend == "direct" else wp.quat, rd),
        **(dict(seg=(wp.vec2i, seg[None]), pair=(wp.vec2i, pair[None])) if backend == "pbd" else {}),
    )


def body_signature(body, backend):
    mesh = body.rod if body.rod is not None else body.tree
    keys = (
        ("frames", "inv_mass_q", "rest_len", "rest_darboux")
        if body.rod is not None
        else ("frames", "inv_mass_q", "rest_len", "edges", "bend_pairs", "root_arc", "radii", "material_radius")
    )
    arrays = tuple(
        (k, None if getattr(mesh, k, None) is None else (getattr(mesh, k).shape, getattr(mesh, k).tobytes()))
        for k in keys
    )
    membership = body.regions
    return (
        body.name,
        body.count,
        backend,
        arrays,
        None if membership is None else (membership.names, membership.edge.tobytes()),
    )


class Record:
    """Host-only field declarations with explicit row identities and original views."""

    def __init__(self, scene, declarations, sid, body=None, *, kind="system"):
        self.sid, self.body_name, self.kind = sid, None if body is None else body.name, kind
        self.fields, self.domains = {}, {}
        self.scene_ref = ref(scene)
        self.declarations = declarations
        self.settings = {}

    @property
    def body(self):
        scene = self.scene_ref()
        return None if self.body_name is None else scene.body(self.body_name)

    def add(self, key, owner, dtype, values, *, runtime=False, nodes=None, width=1, group=None, flat=False):
        scene = self.scene_ref()
        data = np.asarray(values)
        tail = len(getattr(dtype, "_shape_", ()))
        shape = data.shape[:-tail] if tail else data.shape
        rowshape = shape[:-1] if width != 1 else shape
        group = group or key
        if group not in self.domains:
            domain = Domain(self.kind, f"{self.sid}_{group}", f"i4h:{self.sid}_{group}_rows", rowshape, "particle")
            worlds = (
                np.broadcast_to(
                    np.arange(scene.envs, dtype=np.int32).reshape((scene.envs,) + (1,) * (len(rowshape) - 1)), rowshape
                )
                if runtime
                else np.full(rowshape, -1, np.int32)
            )
            body = self.body
            if nodes is None or body is None:
                owners = np.full(rowshape, -1, np.int32)
            else:
                local = np.broadcast_to(np.asarray(nodes, np.int32), rowshape).copy()
                offset = body.offset

                def owners():
                    s = self.scene_ref()
                    starts = np.asarray([scene_record(s).starts[e] + offset for e in range(s.envs)])
                    base = starts.reshape((s.envs,) + (1,) * (len(rowshape) - 1)) if runtime else starts[0]
                    return np.where(local < 0, -1, local + base)

            self.declarations.domain(domain, worlds=worlds, owners=owners)
            self.domains[group] = domain
        domain = self.domains[group]
        f = Field(
            domain,
            key,
            owner,
            dtype,
            (int(np.prod(shape)),) if flat else shape,
            "world/local row" if runtime else "shared local row",
            "existing explicit episode reset" if owner is not Owner.MODEL else "persistent Model values",
            row_width=width,
            attribute_name=f"{self.sid}_{key}",
        )
        self.declarations.add(f, data.reshape(f.shape + tuple(getattr(dtype, "_shape_", ()))))
        self.fields[key] = f
        return f

    def live(self, key, value):
        value = np.asarray(value, np.float32)
        self.settings[key] = value.shape, value.tobytes()
        data = np.broadcast_to(value, (self.scene_ref().envs,) + value.shape).copy()
        self.add(key, Owner.MODEL, wp.float32, data, runtime=True, width=value.size if value.ndim else 1, group="world")


def alias(scene, owner, name, value):
    setattr(owner, name, value)
    scene.storage.register_package(identities=((owner, name, value),))


def route_alias(scene, owner, name, value):
    scene.storage.register_package(identities=((owner, name, value),), replace=True)


def bind_record(system, record):
    scene = system.scene
    scene.storage.check()
    for key, expected in record.settings.items():
        value = np.asarray(getattr(system, key), np.float32)
        if (value.shape, value.tobytes()) != expected:
            raise ValueError(f"{system.id}.{key}: declaration settings changed; reconstruct the model")
    fields = SimpleNamespace()
    alias(scene, system, "fields", fields)
    for key, f in record.fields.items():
        a = scene.attr(f.domain, f.key)
        alias(scene, fields, key, a)
        alias(scene, system, key, a)
    return fields


def model_fields(model, record):
    return SimpleNamespace(
        **{
            k: getattr(model.i4h, f.name).view(f.dtype).flatten()[: int(np.prod(f.shape))].reshape(f.shape)
            for k, f in record.fields.items()
        }
    )


def shared_rod(scene, declarations, body, backend):
    records = getattr(scene, "rod_body_domains", {})
    if body.name in records:
        r = records[body.name]
        if r.backend != backend:
            raise ValueError("rod material: mismatched backend")
        return r
    r = Record(scene, declarations, scene.body_domains[body.name].instance + "_rod", body, kind="body")
    r.backend = backend
    values = geometry(body, backend)
    r.S = body.count - 1 if backend == "direct" else values["seg"][1].shape[1]
    r.K = r.S if backend == "direct" else values["pair"][1].shape[1]
    r.F = values["frames0"][1].shape[1]
    for key, (dtype, data) in values.items():
        count = data.shape[1]
        nodes = (
            np.arange(count)
            if backend == "direct"
            else (
                values["seg"][1][0, :, 0]
                if key not in ("pair", "rest_darboux")
                else values["seg"][1][0, values["pair"][1][0, :, 0], 0]
            )
        )
        r.add(key, Owner.MODEL, dtype, data, nodes=nodes)
    E = scene.envs
    for key, count, dtype, data in (
        ("stretch_scale", r.S, wp.float32, np.ones((E, r.S), np.float32)),
        ("bend_scale", r.K, wp.float32, np.ones((E, r.K), np.float32)),
        ("bend_offset", r.K, wp.vec3, np.zeros((E, r.K, 3), np.float32)),
    ):
        r.add(
            key,
            Owner.CONTROL,
            dtype,
            data,
            runtime=True,
            nodes=(
                np.arange(count)
                if backend == "direct"
                else values["seg"][1][0, :, 0]
                if key == "stretch_scale"
                else values["seg"][1][0, values["pair"][1][0, :, 0], 0]
            ),
            group="segments_runtime" if key == "stretch_scale" else "pairs_runtime",
        )
    r.solver = r.provider = None
    declarations.register_package(records=((scene, "rod_body_domains", {body.name: r}),))
    return r


def declare_rod_direct(scene, body, *, system_id="rod_direct", **settings):
    defer_declaration(
        scene,
        system_id,
        publish_rod,
        dict(body=body, backend="direct", settings=settings, expected=body_signature(scene.body(body), "direct")[:-1]),
    )


def declare_rod_pbd(scene, body, *, system_id="rod_pbd", **settings):
    defer_declaration(
        scene,
        system_id,
        publish_rod,
        dict(body=body, backend="pbd", settings=settings, expected=body_signature(scene.body(body), "pbd")[:-1]),
    )


def publish_rod(scene, declarations, system_id, *, body, backend, settings, expected=None):
    from i4h_physics.systems.rods.rod_direct import RodDirect
    from i4h_physics.systems.rods.rod_pbd import RodPbd

    cls = RodDirect if backend == "direct" else RodPbd
    body = scene.body(body)
    if expected is not None and body_signature(body, backend)[:-1] != expected:
        raise RuntimeError("rod layout changed after declaration; reconstruct the model")
    shared = shared_rod(scene, declarations, body, backend)
    if shared.solver is not None:
        raise ValueError("duplicate rod solver for shared channels")
    if body.regions is not None and shared.provider is None:
        raise ValueError("regional body requires matching material before solver")
    r = Record(scene, declarations, system_id, body)
    r.shared = shared
    r.backend = backend
    r.signature = body_signature(body, backend)
    r.inertia = settings.get("inertia", "unit")
    if r.inertia not in ("unit", "physical"):
        raise ValueError("rod_direct.inertia must be 'unit' or 'physical'")
    for key in DIRECT_LIVE if backend == "direct" else PBD_LIVE:
        r.live(key, settings.get(key, getattr(cls, key)))
    E, F = scene.envs, shared.F
    frames = geometry(body, backend)["frames0"][1]
    nodes = np.arange(F) if backend == "direct" else geometry(body, backend)["seg"][1][0, :, 0]
    r.add(
        "frame",
        Owner.STATE,
        wp.quat,
        np.tile(frames[0], (E, 1)).reshape(E, F, 4),
        runtime=True,
        nodes=nodes,
        group="frames_runtime",
        flat=True,
    )
    r.add(
        "omega",
        Owner.STATE,
        wp.vec3,
        np.zeros((E, F, 3), np.float32),
        runtime=True,
        nodes=nodes,
        group="frames_runtime",
        flat=True,
    )
    if backend == "direct":
        r.add("inertia_local", Owner.MODEL, wp.vec3, np.zeros((F, 3), np.float32), nodes=nodes)
        declarations.initializer(partial(initialize_inertia, scene_ref=ref(scene), record=r))
    else:
        r.add("unit_twist", Owner.MODEL, wp.float32, np.ones((E, shared.K), np.float32), runtime=True)
    shared.solver = system_id
    declarations.register_package(records=((scene, "rod_domains", {system_id: r}),))
    return r


def initialize_inertia(model, *, scene_ref, record):
    from i4h_physics.systems.rods.rod_direct import cylinder_inv_inertia

    scene, body = scene_ref(), record.body
    if scene is None:
        raise RuntimeError("rod authoring expired before finalization")
    start = scene_record(scene).starts[0] + body.offset
    # Read this finalized Model's exact i4h inverse mass and native radius, preserving the original float64 arithmetic.
    proxy = SimpleNamespace(
        field=lambda name, b: (model.i4h.particle_inv_mass if name == "inv_mass" else model.particle_radius)[
            start : start + b.count
        ].reshape((1, b.count))
    )
    fields = model_fields(model, record)
    fields.inertia_local.assign(
        cylinder_inv_inertia(proxy, body, model_fields(model, record.shared).rest_len.numpy()[0])
    )


def validate_rod(scene, record):
    if body_signature(scene.body(record.body_name), record.backend) != record.signature:
        raise RuntimeError("rod topology or layout changed; reconstruct the model")


def bind_shared(scene, body, shared):
    if body.rod_controls is None:
        from i4h_physics.regions import RodControls

        controls = RodControls(None, None, None)
        alias(scene, body, "rod_controls", controls)
        for key in ("stretch_scale", "bend_scale", "bend_offset"):
            f = shared.fields[key]
            alias(scene, controls, key, scene.attr(f.domain, f.key))
    return body.rod_controls


def bind_rod(system, backend):
    scene = system.scene
    system.id = system.id or "rod_" + backend
    r = getattr(scene, "rod_domains", {}).get(system.id)
    if r is None:
        raise RuntimeError(f"rod {system.id!r} storage missing; call declare_rod_{backend} before allocation")
    if r.body_name != system.body.name or r.backend != backend:
        raise ValueError("incompatible rod declaration")
    validate_rod(scene, r)
    if backend == "direct" and system.inertia != r.inertia:
        raise ValueError("rod_direct.inertia changed; reconstruct the model")
    bind_record(system, r)
    alias(scene, system, "body", system.body)
    alias(scene, system, "variant", system.body.variant)
    for key, f in r.shared.fields.items():
        if f.owner is Owner.MODEL:
            alias(scene, system, key, scene.attr(f.domain, f.key))
    controls = bind_shared(scene, system.body, r.shared)
    for key in ("bend_offset",) if backend == "direct" else ("stretch_scale", "bend_scale", "bend_offset"):
        alias(scene, system, key, getattr(controls, key))
    scene.storage.register_package(checks=(partial(validate_rod, scene, r),))
    return r


def declare_rod_direct_material(scene, body, *, system_id="rod_direct_material"):
    defer_declaration(scene, system_id, publish_material, dict(body=body, backend="direct", settings={}))


def declare_rod_pbd_material(scene, body, *, system_id="rod_pbd_material", radius_weighting=False):
    defer_declaration(
        scene,
        system_id,
        publish_material,
        dict(body=body, backend="pbd", settings=dict(radius_weighting=radius_weighting)),
    )


def publish_material(scene, declarations, system_id, *, body, backend, settings, solver_specs=None):
    from i4h_physics.systems.rods.material import _membership

    body = scene.body(body)
    if solver_specs is None:
        solver_specs = [
            dict(values)
            for callback, values in scene_record(scene).deferred.values()
            if callback is publish_rod
        ]
    matches = [spec for spec in solver_specs if spec["body"] == body.name]
    if len(matches) != 1 or matches[0]["backend"] != backend:
        raise ValueError("rod material requires exactly one matching solver declaration")
    shared = shared_rod(scene, declarations, body, backend)
    if shared.provider is not None or shared.solver is not None:
        raise ValueError("duplicate provider or material must precede solver")
    membership = _membership(body, shared.S)
    r = Record(scene, declarations, system_id, body)
    r.backend, r.shared = backend, shared
    r.signature = body_signature(body, backend)
    r.defaults = DIRECT_REGIONS.copy() if backend == "direct" else PBD_REGIONS.copy()
    r.add("edge", Owner.MODEL, wp.int32, membership.edge)
    for key, value in r.defaults.items():
        r.add(
            "region_" + key,
            Owner.MODEL,
            wp.float32,
            np.full((scene.envs, len(membership.names)), value, np.float32),
            runtime=True,
            group="regions_runtime",
        )
    if backend == "pbd":
        seg = geometry(body, backend)["seg"][1][0]
        baseline = body.tree.material_radius if body.tree is not None else None
        if baseline is None:
            baseline = body.tree.radii if body.tree is not None else body.rest.radius[:1]
        radius = 0.5 * (baseline[0, seg[:, 0]] + baseline[0, seg[:, 1]])
        r.add("radius_baseline", Owner.MODEL, wp.float32, (radius / radius.max()) ** 3, nodes=seg[:, 0])
    shared.provider = system_id
    declarations.register_package(records=((scene, "rod_material_domains", {system_id: r}),))
    return r


def bind_material(system, backend):
    from i4h_physics.regions import RegionProperties

    scene, body = system.scene, system.body
    system.id = system.id or "rod_" + backend + "_material"
    r = getattr(scene, "rod_material_domains", {}).get(system.id)
    if r is None:
        raise RuntimeError("rod material storage missing; declare material before allocation")
    if r.body_name != body.name or r.backend != backend:
        raise ValueError("incompatible material declaration")
    if r.shared.solver is None:
        raise ValueError("rod material: missing matching solver")
    validate_rod(scene, r)
    bind_record(system, r)
    controls = bind_shared(scene, body, r.shared)
    props = RegionProperties(
        body.regions,
        system.edge,
        {key: getattr(system, "region_" + key) for key in r.defaults},
        r.defaults,
        body.rod.rest_len[0] if backend == "direct" else None,
        body.rod.rest_darboux[0] if backend == "direct" else None,
    )
    alias(scene, props, "edge", system.edge)
    expected = dict(props.tables)

    def check_tables():
        if not isinstance(props.tables, dict) or props.tables.keys() != expected.keys() or any(props.tables[k] is not a for k, a in expected.items()):
            raise RuntimeError("regional table replaced; bind a fresh runtime")

    scene.storage.register_package(checks=(check_tables,))
    alias(scene, props, "tables", props.tables)
    alias(scene, body, "region_properties", props)
    m = SimpleNamespace(backend=backend, active=system.enabled, S=r.shared.S, K=r.shared.K, Ed=r.shared.S)
    for key in ("rest_len", "rest_darboux") if backend == "direct" else ("pair",):
        f = r.shared.fields[key]
        alias(scene, m, key, scene.attr(f.domain, f.key))
    if backend == "pbd":
        alias(scene, m, "radius_baseline", system.radius_baseline)
        for key, count in (("stretch", m.S), ("bend", m.K), ("twist", m.K)):
            setattr(m, key, wp.empty((scene.envs, count), dtype=float, device=scene.device))
    else:
        m.preferred_rest = wp.empty((scene.envs, m.Ed), dtype=wp.vec3, device=scene.device)
        for key in ("young_multiplier", "torsion_multiplier", "bend_multiplier", "twist_multiplier"):
            setattr(m, key, wp.empty((scene.envs, m.Ed), dtype=float, device=scene.device))
    for key, value in vars(m).items():
        if isinstance(value, wp.array):
            alias(scene, m, key, value)
    alias(scene, body, "rod_material", m)
    alias(scene, system, "controls", controls)
    scene.storage.register_package(checks=(partial(validate_rod, scene, r),))
    return r
