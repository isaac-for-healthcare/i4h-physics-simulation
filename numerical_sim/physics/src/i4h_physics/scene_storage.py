# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared Scene declarations, completed after all component worlds exist."""
import newton
import numpy as np
import warp as wp
from i4h_physics.particle_storage import declare_particles
from i4h_physics.storage import Domain, Field, Owner


def layout(scene):
    """Frozen capacities, mappings, transforms and mesh-local geometry layouts."""
    from i4h_physics.surface_storage import surface_layout
    from i4h_physics.tet_storage import tet_layout
    return (scene.envs, tuple((b.name, b.offset, b.count, b.pool, b.variants,
            tuple((key, getattr(b.rest, key).shape) for key in ("q0", "inv_mass", "radius", "flags")),
            b.variant_of_env.tobytes(), tet_layout(b.tet), surface_layout(b.tet),
            tuple((name, p.index, p.particle_start, p.particle_count, p.tri_start, p.tri_count) for name, p in b.parts.items()))
            for b in scene.bodies + scene.pools),
            tuple((m.name, m.points.shape, m.indices.shape) for m in scene.static_meshes),
            tuple(getattr(scene, "_tet_declarations", {}).items()),
            tuple(getattr(scene, "_surface_declarations", {}).items()),
            tuple(getattr(scene, "_deferred_storage", {}).items()),
            getattr(scene, "_mount_declaration", None))


def declare_scene(declarations, scene):
    scene.declarations = declarations
    scene.particle_domain = declare_particles(declarations, scene)
    scene.pool_domain = domain = Domain("scene", "pool", "i4h:pool_rows", (scene.pool_count,), "particle")
    worlds = np.concatenate([np.repeat(np.arange(scene.envs, dtype=np.int32), b.count)
                             for b in scene.pools]) if scene.pools else np.empty(0, np.int32)
    declarations.domain(domain, worlds=worlds, owners=np.full(domain.shape, -1, np.int32))
    for key, source, owner, dtype in (
        ("q0", "q0", Owner.MODEL, wp.vec3), ("inv_mass", "inv_mass", Owner.MODEL, wp.float32),
        ("radius", "radius", Owner.MODEL, wp.float32), ("flags0", "flags", Owner.MODEL, wp.int32),
        ("q", "q0", Owner.STATE, wp.vec3), ("qd", None, Owner.STATE, wp.vec3),
        ("q_prev", "q0", Owner.STATE, wp.vec3), ("flags", "flags", Owner.STATE, wp.int32),
    ):
        shape = domain.shape + ((3,) if dtype is wp.vec3 else ())
        rows = [getattr(b.rest, source).reshape((-1,) + shape[1:]) for b in scene.pools] if source else []
        values = np.concatenate(rows) if rows else np.zeros(shape, np.int32 if dtype is wp.int32 else np.float32)
        declarations.add(Field(domain, key, owner, dtype, domain.shape,
                               "body, world, local particle", "selected-world authored pool defaults"), values)
    scene.body_domains = {}
    for index, body in enumerate(scene.bodies + scene.pools):
        # Declaration order, independent of system ids or arbitrary asset names.
        domain = Domain("body", f"body_{index}", newton.Model.AttributeFrequency.WORLD, (scene.envs,))
        declarations.domain(domain)
        declarations.add(Field(domain, "variant", Owner.MODEL, wp.int32, domain.shape,
                               "world", "persistent body variant mapping"), body.variant_of_env)
        scene.body_domains[body.name] = domain
    scene.static_domains = {}
    for index, mesh in enumerate(scene.static_meshes):
        domains = []
        for key, dtype, values in (("points", wp.vec3, mesh.points),
                                   ("indices", wp.int32, mesh.indices.reshape(-1))):
            domain = Domain("system", f"mesh_{index}_{key}", f"i4h:mesh_{index}_{key}_rows", (len(values),), "particle")
            declarations.domain(domain, worlds=np.full(domain.shape, -1, np.int32),
                                owners=np.full(domain.shape, -1, np.int32))
            declarations.add(Field(domain, key, Owner.MODEL, dtype, domain.shape,
                                   "mesh-local", "static authored geometry"), values)
            domains.append(domain)
        scene.static_domains[mesh.name] = tuple(domains)
    scene.gravity_domain = domain = Domain("system", "scene", newton.Model.AttributeFrequency.WORLD, (scene.envs,))
    declarations.domain(domain)
    declarations.add(Field(domain, "gravity", Owner.MODEL, wp.vec3, domain.shape,
                           "world", "explicit component-frame gravity refresh"), np.zeros((scene.envs, 3), np.float32))
    if len(scene.static_domains) != len(scene.static_meshes):
        raise ValueError("static mesh names must be unique")
    scene.tet_domains = {}
    from i4h_physics.tet_storage import publish_tet
    for system_id, name in getattr(scene, "_tet_declarations", {}).items():
        body = scene.body(name)
        publish_tet(scene, declarations, body, system_id,
                    lambda body=body: [scene._storage_starts[e] + body.offset for e in range(scene.envs)])
    scene.surface_domains = {}
    from i4h_physics.surface_storage import publish_surface
    for system_id, tet_id in getattr(scene, "_surface_declarations", {}).items():
        body = scene.tet_domains[tet_id]["body"]
        publish_surface(scene, declarations, system_id, tet_id,
                        lambda body=body: [scene._storage_starts[e] + body.offset for e in range(scene.envs)])
    scene._storage_starts = {}
    scene._storage_layout = layout(scene)
    scene._storage_worlds = set()
    scene._storage_completed = False
    for key, (callback, settings) in getattr(scene, "_deferred_storage", {}).items():
        callback(scene, declarations, key, **dict(settings))


def register_scene_finalizers(builder, scene):
    from i4h_physics.newton_compat import require_model_finalizer
    register = require_model_finalizer(builder)
    def gravity_finalizer(builder, model, attribute):
        if not scene._storage_completed:
            raise RuntimeError("complete Scene storage before finalization")
        validate_scene(scene)
        from i4h_physics.newton_model import component_gravity
        data = (wp.array(component_gravity(model, scene), dtype=wp.vec3, device=model.device)
                if scene.env_rotated else model.gravity[:scene.envs])
        model.add_attribute(attribute.name, data, attribute.frequency, attribute.assignment,
                            attribute.namespace, attribute.references)
    register("i4h:scene_gravity", gravity_finalizer)


def complete_scene(builder, scene):
    """Direct-builder completion, also used by CaseAuthoring.complete; idempotent."""
    if layout(scene) != scene._storage_layout:
        raise RuntimeError("Scene capacities, body mappings or geometry layout changed; reconstruct the model")
    if scene._storage_worlds != set(range(scene.envs)) or builder.world_count != scene.envs:
        raise RuntimeError("Scene completion requires all authored worlds")
    if scene.env_xforms is not None and any(x is None for x in scene.env_xforms):
        raise RuntimeError("Scene completion requires complete world transforms")
    transforms = None if scene.env_xforms is None else tuple((p.tobytes(), q.tobytes()) for p, q in scene.env_xforms)
    if scene._storage_completed and transforms != scene._storage_transforms:
        raise RuntimeError("Scene world transforms changed; reconstruct the model")
    scene.declarations.author_rows(builder)
    scene._storage_transforms = transforms
    scene._storage_completed = True


def validate_scene(scene):
    if not scene._storage_completed:
        raise RuntimeError("complete Scene storage before binding")
    transforms = None if scene.env_xforms is None else tuple(
        None if x is None else (x[0].tobytes(), x[1].tobytes()) for x in scene.env_xforms)
    if layout(scene) != scene._storage_layout or transforms != scene._storage_transforms:
        raise RuntimeError("Scene layout or world transforms changed; reconstruct the model")
