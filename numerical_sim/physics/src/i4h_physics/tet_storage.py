# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only declarations for per-tet mechanics and physical-surface fields.

Schema registration precedes world creation; completion resolves native owners.
Tet mechanics and physical Surface bind declared aliases.
Variant topology is shared, and mutable masks retain their world-major layout.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp
from i4h_physics.storage import Domain, Field, Owner


def tet_domain(declarations, body, system_id, particle_starts, *, variants):
    """Declare row identity using real owning native particle indices.

    ``particle_starts[e]`` is this body's first native particle in world e, after
    host prefixes. Shared variant rows have world=-1 and refer to a representative
    instance's first tet vertex; unused variants and padding have owner=-1.
    Body-local vertex/edge/neighbor columns themselves are never remapped.
    """
    mesh = body.tet
    selected = np.asarray(body.variant_of_env, dtype=np.int32)
    V, T = mesh.active0.shape
    shape = (V, T) if variants else (selected.size, T)
    worlds = np.full(shape, -1, np.int32) if variants else np.broadcast_to(
        np.arange(selected.size, dtype=np.int32)[:, None], shape)

    def resolve_owners():
        starts = np.asarray(particle_starts() if callable(particle_starts) else particle_starts, dtype=np.int64)
        if starts.shape != selected.shape or np.any(starts < 0):
            raise ValueError("particle_starts must give each world's authored body start")
        if variants:
            owners = np.full(shape, -1, np.int64)
            for v in range(V):
                instances = np.flatnonzero(selected == v)
                if instances.size:
                    owners[v] = starts[instances[0]] + mesh.tets[v, :, 0]
            owners[mesh.active0 == 0] = -1
        else:
            owners = starts[:, None] + mesh.tets[selected, :, 0]
            owners[mesh.active0[selected] == 0] = -1
        return owners

    frequency = f"i4h:{system_id}_tet_variant" if variants else f"i4h:{system_id}_tet"
    domain = Domain("system", system_id, frequency, shape, "particle")
    owners = resolve_owners if callable(particle_starts) else resolve_owners()
    declarations.domain(domain, worlds=worlds, owners=owners)
    return domain


def declare_tet_fields(declarations, body, system_id, particle_starts):
    """Persistent Tet topology, masks and materials on explicit row domains."""
    mesh = body.tet
    variant = tet_domain(declarations, body, system_id, particle_starts, variants=True)
    state = tet_domain(declarations, body, system_id, particle_starts, variants=False)
    for key, dtype, values in (("tets", wp.vec4i, mesh.tets), ("active0", wp.int32, mesh.active0),
                               ("rest_volume", wp.float32, mesh.rest_volume)):
        declarations.add(Field(variant, key, Owner.MODEL, dtype, variant.shape,
                               "variant, body-local tet", "preserve authored values"), values)
    declarations.add(Field(state, "active", Owner.STATE, wp.int32, state.shape,
                           "world, body-local tet", "active0[variant_of_env]"), mesh.active0[body.variant_of_env])
    declarations.add(Field(state, "native_tet", Owner.MODEL, wp.int32, state.shape,
                           "world, body-local tet; not a native Newton tet", "constant -1"), np.full(state.shape, -1, np.int32))
    # Part ids are shared across variants in the existing layout, not V*T rows.
    T = mesh.part.size
    parts = Domain("system", system_id, f"i4h:{system_id}_tet_part", (T,), "particle")
    owners = np.full(T, -1, np.int64)
    if len(body.variant_of_env) and not callable(particle_starts):
        owners = np.asarray(particle_starts)[0] + mesh.tets[body.variant_of_env[0], :, 0]
        owners = np.where(mesh.active0[body.variant_of_env[0]] != 0, owners, -1)
    if callable(particle_starts):
        def owners():
            if not len(body.variant_of_env):
                return np.full(T, -1, np.int64)
            v = body.variant_of_env[0]
            return np.where(mesh.active0[v] != 0, np.asarray(particle_starts())[0] + mesh.tets[v, :, 0], -1)
    declarations.domain(parts, worlds=np.full(T, -1, np.int32), owners=owners)
    declarations.add(Field(parts, "tet_part", Owner.MODEL, wp.int32, (T,),
                           "shared body-local tet", "preserve authored local part ids"), mesh.part)
    edge_variant = edge_domain(declarations, body, system_id, particle_starts, variants=True)
    edge_state = edge_domain(declarations, body, system_id, particle_starts, variants=False)
    for key, dtype, values in (("edges", wp.vec2i, mesh.edges), ("edge_active0", wp.int32, mesh.edge_active0),
                               ("edge_rest", wp.float32, mesh.edge_rest)):
        declarations.add(Field(edge_variant, key, Owner.MODEL, dtype, edge_variant.shape,
                               "variant, body-local edge", "preserve authored values"), values)
    declarations.add(Field(edge_state, "edge_active", Owner.STATE, wp.int32, edge_state.shape,
                           "world, body-local edge", "edge_active0[variant_of_env]"), mesh.edge_active0[body.variant_of_env])
    edge_parts = Domain("system", system_id, f"i4h:{system_id}_edge_part", (mesh.edge_part.size,), "particle")
    def edge_owners():
        if not len(body.variant_of_env):
            return np.full(edge_parts.shape, -1, np.int64)
        v = body.variant_of_env[0]
        starts = particle_starts() if callable(particle_starts) else particle_starts
        return np.where(mesh.edge_active0[v] != 0, np.asarray(starts)[0] + mesh.edges[v, :, 0], -1)
    declarations.domain(edge_parts, worlds=np.full(edge_parts.shape, -1, np.int32), owners=edge_owners)
    declarations.add(Field(edge_parts, "edge_part", Owner.MODEL, wp.int32, edge_parts.shape,
                           "shared body-local edge", "preserve authored local part ids"), mesh.edge_part)
    E, P = len(body.variant_of_env), len(mesh.spring)
    materials = Domain("system", system_id, f"i4h:{system_id}_material", (E, P), "particle")
    def material_owners():
        starts = np.asarray(particle_starts() if callable(particle_starts) else particle_starts)
        owners = np.full((E, P), -1, np.int64)
        for e, v in enumerate(body.variant_of_env):
            for part in range(P):
                rows = np.flatnonzero((mesh.part == part) & (mesh.active0[v] != 0))
                if rows.size:
                    owners[e, part] = starts[e] + mesh.tets[v, rows[0], 0]
                else:
                    rows = np.flatnonzero((mesh.edge_part == part) & (mesh.edge_active0[v] != 0))
                    if rows.size:
                        owners[e, part] = starts[e] + mesh.edges[v, rows[0], 0]
        return owners
    declarations.domain(materials, worlds=np.broadcast_to(np.arange(E, dtype=np.int32)[:, None], (E, P)),
                        owners=material_owners)
    for key in MATERIAL_FIELDS:
        declarations.add(Field(materials, key, Owner.MODEL, wp.float32, materials.shape,
                               "world, body-local part", "live material; preserved on reset"),
                         np.tile(np.asarray(getattr(mesh, key), np.float32)[None], (E, 1)))
    return variant, state, parts


def declare_surface_tet_fields(declarations, body, system_id, particle_starts, *, tet_body, tet_variant):
    """Use the resolved Tet's row domain; keep Surface attribute names distinct.

    Body identity is checked, not just equal mesh sizes. Callers resolve the
    authored Surface.tet reference before invoking this helper.
    """
    if body is not tet_body:
        raise ValueError("Surface and its referenced Tet must resolve to the same body")
    if tet_variant.shape != body.tet.active0.shape or (tet_variant, "tets") not in declarations.fields:
        raise ValueError("expected the referenced Tet's declared variant domain")
    domain = tet_variant
    for key, width in (("tet_edges", 6), ("face_neighbor", 4), ("face_tri", 4)):
        name = f"{system_id}_{key}"
        declarations.add(Field(domain, name, Owner.MODEL, wp.int32, (*domain.shape, width),
                               "variant, body-local tet, local edge/face slot", "preserve authored local indices",
                               row_width=width, attribute_name=name), getattr(body.tet, key))
    return domain


MATERIAL_FIELDS = ("spring", "spring_damping", "volume", "volume_damping")
TET_FIELDS = ("tets", "active0", "rest_volume", "tet_part", "edges", "edge_active0", "edge_rest",
              "edge_part", *MATERIAL_FIELDS, "native_tet", "active", "edge_active")


def edge_domain(declarations, body, system_id, particle_starts, *, variants):
    mesh = body.tet
    selected = np.asarray(body.variant_of_env, np.int32)
    V, S = mesh.edge_active0.shape
    shape = (V, S) if variants else (len(selected), S)
    worlds = np.full(shape, -1, np.int32) if variants else np.broadcast_to(
        np.arange(len(selected), dtype=np.int32)[:, None], shape)
    def owners():
        starts = np.asarray(particle_starts() if callable(particle_starts) else particle_starts)
        if starts.shape != selected.shape or np.any(starts < 0):
            raise ValueError("particle_starts must give each world's authored body start")
        result = np.full(shape, -1, np.int64)
        for row in range(shape[0]):
            instances = np.flatnonzero(selected == row) if variants else [row]
            if len(instances):
                e = instances[0]
                v = row if variants else selected[e]
                result[row] = np.where(mesh.edge_active0[v] != 0, starts[e] + mesh.edges[v, :, 0], -1)
        return result
    domain = Domain("system", system_id, f"i4h:{system_id}_edge" + ("_variant" if variants else ""), shape, "particle")
    declarations.domain(domain, worlds=worlds, owners=owners)
    return domain


def publish_tet(scene, declarations, body, system_id, particle_starts):
    domains = declare_tet_fields(declarations, body, system_id, particle_starts)
    record = {"body": body, "variant": domains[0], "state": domains[1], "parts": domains[2]}
    record["fields"] = {key: domain for domain, key in declarations.fields
                        if domain.kind == "system" and domain.instance == system_id and key in TET_FIELDS}
    record.update(edge_variant=record["fields"]["edges"], edge_state=record["fields"]["edge_active"],
                  edge_parts=record["fields"]["edge_part"], materials=record["fields"]["spring"])
    declarations.register_package(records=((scene, "tet_domains", {system_id: record}),))
    return record


def declare_tet(scene, body, *, system_id="tet"):
    """Predeclare direct-Scene Tet storage before allocation/schema registration."""
    if hasattr(scene, "declarations") or scene.q is not None:
        raise RuntimeError("declare_tet before allocation or schema registration; reconstruct the model")
    body = scene.body(body) if isinstance(body, str) else body
    if not any(body is b for b in scene.bodies) or body.tet is None:
        raise ValueError("declare_tet requires a Scene body with a tet mesh")
    system_id = system_id or "tet"
    pending = scene.__dict__.setdefault("_tet_declarations", {})
    if system_id in pending:
        raise ValueError(f"duplicate Tet declaration {system_id!r}")
    pending[system_id] = body.name


def tet_fields(scene, system_id="tet"):
    """Cached checked aliases for one declared Tet, with no device allocation."""
    system_id = system_id or "tet"
    records = getattr(scene, "tet_domains", {})
    if system_id not in records:
        raise RuntimeError(f"Tet {system_id!r} storage is missing; call declare_tet before allocation and reconstruct the model")
    scene.storage.check()
    cache = scene.__dict__.setdefault("_tet_fields", {})
    if system_id not in cache:
        record = records[system_id]
        fields = SimpleNamespace(body=record["body"], domains=record, variant=record["body"].variant,
                                 **{key: scene.attr(domain, key) for key, domain in record["fields"].items()})
        cache[system_id] = fields
        scene.storage.register_package(identities=((fields.body, "variant", fields.variant),))
        for key in (*TET_FIELDS, "variant"):
            scene.storage.register_package(identities=((fields, key, getattr(fields, key)),))
    return cache[system_id]


def alias_tet_fields(scene, target, fields, names):
    """Compatibility attributes are checked along with the Newton backing buffers."""
    for name in names:
        value = getattr(fields, name)
        setattr(target, name, value)
        scene.storage.register_package(identities=((target, name, value),))


def tet_layout(mesh):
    """Freeze dimensions and local topology; live device materials remain editable."""
    if mesh is None:
        return None
    names = ("tets", "active0", "rest_volume", "part", "edges", "edge_active0", "edge_rest", "edge_part", *MATERIAL_FIELDS)
    return tuple((key, np.asarray(getattr(mesh, key)).shape, np.asarray(getattr(mesh, key)).dtype.str,
                  None if key in MATERIAL_FIELDS else np.asarray(getattr(mesh, key)).tobytes()) for key in names)


def estimate_tet_bytes(scene):
    """Model, one State, templates and identities, including empty backing rows."""
    bodies = ([record["body"] for record in scene.tet_domains.values()] if hasattr(scene, "tet_domains") else
              [scene.body(name) for name in getattr(scene, "_tet_declarations", {}).values()])
    total = 0
    for body in bodies:
        V, T = body.tet.active0.shape
        S, E, P = body.tet.edge_active0.shape[1], scene.envs, len(body.tet.spring)
        vt, et, t, vs, es, s, ep = (max(1, n) for n in (V*T, E*T, T, V*S, E*S, S, E*P))
        total += 24*vt + 4*t + 16*vs + 4*s + 4*et + 16*ep + 8*(et+es)
        total += 8*(vt+et+t+vs+es+s+ep)
    return total
