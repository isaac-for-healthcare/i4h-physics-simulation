# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton-owned physical Surface rows; authored identities never follow cuts."""
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from i4h_physics.storage import Domain, Field, Owner, scene_record
from i4h_physics.tet_storage import declare_surface_tet_fields, tet_fields

MODEL_FIELDS = ("tet_edges", "face_neighbor", "face_tri", "tris0", "tri_tet0", "tri_part",
                "support0", "edge_support0", "heat_enabled", "deletable", "cut_start", "cut_capacity")
STATE_FIELDS = ("tris", "tri_tet", "support", "edge_support", "cut_count", "overflow", "deleted", "dirty")
SURFACE_FIELDS = MODEL_FIELDS + STATE_FIELDS


def publish_surface(scene, declarations, system_id, tet_id, particle_starts):
    tet = scene.tet_domains.get(tet_id)
    if tet is None:
        raise ValueError("Surface.tet must reference a declared Tet; reconstruct the model")
    body, mesh = tet["body"], tet["body"].tet
    E, V, F, N, P = scene.envs, body.variants, mesh.tris.shape[1], body.count, len(body.parts)
    if F < P:
        raise ValueError(f"surface: body {body.name!r} has fewer triangle slots ({F}) than parts ({P})")
    selected = np.asarray(body.variant_of_env)
    fields = {}
    declare_surface_tet_fields(declarations, body, system_id, particle_starts,
                               tet_body=body, tet_variant=tet["variant"])
    for key in MODEL_FIELDS[:3]:
        fields[key] = (tet["variant"], f"{system_id}_{key}")

    def domain(label, shape, local, *, variants=False, shared=False, world_rows=False):
        d = Domain("system", system_id, f"i4h:{system_id}_{label}", shape, "particle")
        worlds = np.full(shape, -1, np.int32) if variants or shared else np.broadcast_to(
            np.arange(E, dtype=np.int32)[:, None], shape)
        def owners():
            starts = np.asarray(particle_starts(), np.int64)
            result = np.full(shape, -1, np.int64)
            if shared:
                if E:
                    ids = local[selected[0]]
                    result[:] = np.where(ids >= 0, starts[0] + ids, -1)
            else:
                for row in range(shape[0]):
                    instances = np.flatnonzero(selected == row) if variants else [row]
                    if len(instances):
                        e = instances[0]
                        ids = local[row if variants or world_rows else selected[e]]
                        result[row] = np.where(ids >= 0, starts[e] + ids, -1)
            return result
        declarations.domain(d, worlds=worlds, owners=owners)
        return d

    tris = mesh.tris
    valid = (tris[:, :, 0] != tris[:, :, 1]) & (tris[:, :, 0] != tris[:, :, 2]) & (tris[:, :, 1] != tris[:, :, 2])
    face_ids = np.where(valid, tris[:, :, 0], -1)
    fv = domain("face_variant", (V, F), face_ids, variants=True)
    fs = domain("face", (E, F), face_ids)
    fp = domain("face_part", (F,), face_ids, shared=True)
    particle_ids = np.full((V, N), -1, np.int64)
    for v in range(V):
        instances = np.flatnonzero(selected == v)
        if instances.size:
            particle_ids[v] = np.where((body.rest.flags[instances[0]] & 1) != 0, np.arange(N), -1)
    pv = domain("support_variant", (V, N), particle_ids, variants=True)
    ps = domain("support", (E, N), np.where((body.rest.flags & 1) != 0, np.arange(N), -1), world_rows=True)
    part_ids = np.full((V, P), -1, np.int64)
    for v in range(V):
        for p in range(P):
            rows = np.flatnonzero((mesh.part == p) & (mesh.active0[v] != 0))
            if rows.size:
                part_ids[v, p] = mesh.tets[v, rows[0], 0]
            else:
                rows = np.flatnonzero((mesh.edge_part == p) & (mesh.edge_active0[v] != 0))
                if rows.size:
                    part_ids[v, p] = mesh.edges[v, rows[0], 0]
    parts = domain("part", (P,), part_ids, shared=True)
    world = Domain("system", system_id, newton.Model.AttributeFrequency.WORLD, (E,))
    declarations.domain(world)

    def add(key, d, values, dtype=wp.int32):
        owner = Owner.STATE if key in STATE_FIELDS else Owner.MODEL
        name = f"{system_id}_{key}"
        shape = (E*N,) if key == "support" else d.shape
        declarations.add(Field(d, name, owner, dtype, shape, "authored body-local rows",
                               "selected-world defaults" if owner is Owner.STATE else "preserve live Model values",
                               attribute_name=name), values)
        fields[key] = (d, name)
    for key, d, values, dtype in (
        ("tris0", fv, mesh.tris, wp.vec3i), ("tri_tet0", fv, mesh.tri_tet, wp.int32),
        ("tri_part", fp, mesh.tri_part, wp.int32), ("support0", pv, mesh.support0, wp.int32),
        ("edge_support0", tet["edge_variant"], mesh.edge_support0, wp.int32),
        ("tris", fs, mesh.tris[selected], wp.vec3i), ("tri_tet", fs, mesh.tri_tet[selected], wp.int32),
        ("support", ps, mesh.support0[selected].reshape(-1), wp.int32),
        ("edge_support", tet["edge_state"], mesh.edge_support0[selected], wp.int32),
        ("cut_count", tet["materials"], np.zeros((E, P), np.int32), wp.int32)):
        add(key, d, values, dtype)
    for key in ("heat_enabled", "deletable", "cut_start", "cut_capacity"):
        add(key, parts, getattr(mesh, "heat" if key == "heat_enabled" else key))
    for key in ("overflow", "deleted", "dirty"):
        add(key, world, np.zeros(E, np.int32))
    record = dict(body=body, tet_id=tet_id, variant=tet["variant"], fields=fields,
                  domains=(fv, fs, fp, pv, ps, parts, world))
    declarations.register_package(records=((scene, "surface_domains", {system_id: record}),))
    return record


def support_domain(scene, body):
    """The ``(E, count)`` per-particle row domain of the Surface declared on ``body``, or None.

    Packages add particle-indexed history of that body to these existing rows.
    """
    for record in getattr(scene, "surface_domains", {}).values():
        if record["body"] is body:
            return record["domains"][4]
    return None


def declare_surface(scene, *, tet_id="tet", system_id="surface"):
    """Predeclare after the matching Tet and before allocation/schema registration."""
    if hasattr(scene, "declarations") or scene.q is not None:
        raise RuntimeError("declare_surface before allocation or schema registration; reconstruct the model")
    system_id, tet_id = system_id or "surface", tet_id or "tet"
    if tet_id not in scene_record(scene).tets:
        raise ValueError("declare_surface requires matching Tet predeclaration; reconstruct the model")
    pending = scene_record(scene).surfaces
    if system_id in pending:
        raise ValueError(f"duplicate Surface declaration {system_id!r}")
    pending[system_id] = tet_id


def surface_fields(scene, system_id="surface"):
    """Cached checked physical topology aliases; binding never initializes State."""
    system_id = system_id or "surface"
    records = getattr(scene, "surface_domains", {})
    if system_id not in records:
        raise RuntimeError(f"Surface {system_id!r} storage is missing; call declare_surface before allocation and reconstruct the model")
    from i4h_physics.scene_storage import validate_scene
    validate_scene(scene)
    scene.storage.check()
    cache = scene_record(scene).surface_fields
    if system_id not in cache:
        record = records[system_id]
        fields = SimpleNamespace(body=record["body"], domains=record, tet_id=record["tet_id"],
                                 tet_fields=tet_fields(scene, record["tet_id"]),
                                 **{key: scene.attr(*pair) for key, pair in record["fields"].items()})
        cache[system_id] = fields
        for key in (*SURFACE_FIELDS, "tet_fields"):
            scene.storage.register_package(identities=((fields, key, getattr(fields, key)),))
    return cache[system_id]


def surface_layout(mesh):
    if mesh is None:
        return None
    names = ("tet_edges", "face_neighbor", "face_tri", "tris", "tri_tet", "tri_part", "support0",
             "edge_support0", "heat", "deletable", "cut_start", "cut_capacity")
    return tuple((key, np.asarray(getattr(mesh, key)).shape, np.asarray(getattr(mesh, key)).dtype.str,
                  np.asarray(getattr(mesh, key)).tobytes()) for key in names)


def estimate_surface_bytes(scene):
    """Additional Surface fields, two State copies and unshared row identities."""
    if hasattr(scene, "surface_domains"):
        bodies = [r["body"] for r in scene.surface_domains.values()]
    else:
        bodies = [scene.body(scene_record(scene).tets[tet])
                  for tet in scene_record(scene).surfaces.values()]
    total = 0
    for body in bodies:
        m = body.tet
        V, T = m.active0.shape
        E, F, N, S, P = scene.envs, m.tris.shape[1], body.count, m.edges.shape[1], len(body.parts)
        vt, vf, f, vn, vs, p, ef, en, es, ep = (max(1, x) for x in (V*T, V*F, F, V*N, V*S, P, E*F, E*N, E*S, E*P))
        total += 56*vt + 16*vf + 4*f + 4*vn + 4*vs + 16*p
        total += 2*(16*ef + 4*en + 4*es + 4*ep + 12*E)
        total += 8*(vf + ef + f + vn + en + p)
    return total
