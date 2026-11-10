# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physics-owned host mount declarations. Packages supply only field descriptors."""
from dataclasses import dataclass
from functools import partial
from weakref import ref

import newton
import numpy as np
import warp as wp

from i4h_physics.storage import Domain, Field, Owner

IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


@dataclass(frozen=True)
class Mountable:
    system_id: str
    names: tuple[str, ...]
    target_domain: Domain
    mounted: Field
    mount_pose: Field
    root: Field
    root_reaction: bool


def configuration(mounts, two_way):
    """Immutable, ordered host label specifications and normalized authored offsets."""
    if not isinstance(two_way, bool):
        raise TypeError("two_way must be bool")
    if mounts is None:
        mounts = {}
    if not isinstance(mounts, dict):
        raise TypeError("mounts must be a mapping")
    entries = []
    for name, spec in mounts.items():
        if not isinstance(name, str) or not isinstance(spec, dict) or set(spec) - {"body", "offset"} or "body" not in spec:
            raise ValueError(f"mounts[{name!r}] must be {{'body': label, 'offset': ((x, y, z), (qx, qy, qz, qw))}}")
        if not isinstance(spec["body"], str) or not spec["body"]:
            raise ValueError("mount body must be a nonempty label")
        try:
            p, q = spec.get("offset", IDENTITY)
            p, q = np.asarray(p, np.float32), np.asarray(q, np.float32)
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or not np.linalg.norm(q) > 0:
                raise ValueError
            # Preserve the original Warp quaternion normalization exactly.
            q = np.asarray(wp.normalize(wp.quat(*map(float, q))), np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("mount offset must be a finite position (3,) and nonzero quaternion (4,)") from exc
        entries.append((name, spec["body"], tuple(map(float, p)), tuple(map(float, q))))
    return tuple(entries), two_way


def declare_mounts(scene, mounts, *, two_way=False):
    """Direct Scene path: after target declarations, before schema registration."""
    if hasattr(scene, "declarations") or scene.q is not None:
        raise RuntimeError("declare mounts before allocation or schema registration; reconstruct the model")
    if hasattr(scene, "_mount_declaration"):
        raise ValueError("duplicate mount declaration")
    scene._mount_declaration = configuration(mounts, two_way)


def mounted_defaults(builder, *, scene_ref, names):
    scene = scene_ref()
    configured = {entry[0] for entry in getattr(scene, "_mount_declaration", ((), False))[0]}
    return np.asarray([int(name in configured) for name in names] or [0], np.int32)


def _reject_control_clear(model=None):
    raise RuntimeError(
        "Control.clear() would zero mount-pose quaternions; clear the packed/named "
        "action views instead, or use the explicit episode reset (mount poses persist)"
    )


def _mount_control(factory, requires_grad=None, clone_variables=True):
    control = factory(requires_grad=requires_grad, clone_variables=clone_variables)
    control.clear = _reject_control_clear
    return control


def protect_control_clear(model):
    """Keep the pinned Newton blanket clear from corrupting mount transforms.

    The pinned Control has no custom-clear hook. Wrap only this finalized Model's
    factory, so every ordinary/template-alias Control rejects clear before writes,
    even before binding. Newton classes and retained dependencies stay unchanged.
    The builder holds this host callback, never the finalized Model or its arrays.
    """
    model.control = partial(_mount_control, model.control)


def publish_mountable(scene, declarations, system_id, names, target_domain, runtime_domain, root, poses, *, root_reaction=True):
    """Publish the mount interface without importing a package implementation."""
    records = getattr(scene, "mountable_domains", {})
    if not records:
        declarations.initializer(protect_control_clear)
    if system_id in records:
        raise ValueError(f"duplicate mountable control {system_id!r}")
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError("duplicate mountable target names")
    domain = Domain("system", f"{system_id}_mountable", f"i4h:{system_id}_mountable_rows", (max(len(names), 1),), "world")
    declarations.domain(domain, worlds=np.full(domain.shape, -1, np.int32), owners=np.full(domain.shape, -1, np.int32))
    mounted = Field(domain, "mounted", Owner.MODEL, wp.int32, domain.shape, "local target; absent padding if empty",
                    "persistent explicit in-place mount edits", attribute_name=f"{system_id}_mounted")
    declarations.add(mounted, partial(mounted_defaults, scene_ref=ref(scene), names=names))
    pose = Field(runtime_domain, "mount_pose", Owner.CONTROL, wp.transform, (scene.envs, len(names)),
                 "world, local target", "preserve mount commands", attribute_name=f"{system_id}_mount_pose")
    declarations.add(pose, poses)
    record = Mountable(system_id, names, target_domain or domain, mounted, pose, root, bool(root_reaction))
    declarations.register_package(records=((scene, "mountable_domains", {system_id: record}),))
    return record


def find_body(host, label, world):
    worlds = host.body_world
    worlds = worlds.numpy() if isinstance(worlds, wp.array) else np.asarray(worlds)
    hits = [int(b) for b in np.flatnonzero(worlds == world)
            if host.body_label[b] == label or str(host.body_label[b]).endswith("/" + label)]
    if len(hits) != 1:
        raise ValueError(f"mount body {label!r}: {len(hits)} bodies match in world {world}, expected one")
    return hits[0]


def resolve_bodies(builder, entries, envs):
    return np.asarray([[find_body(builder, entry[1], e) for entry in entries] for e in range(envs)], np.int32)


def body_owners(builder_ref, entries, envs):
    return resolve_bodies(builder_ref(), entries, envs)


def environment_values(builder, *, scene_ref):
    scene = scene_ref()
    return np.asarray([(*p, *q) for p, q in (scene.env_xforms or [IDENTITY] * scene.envs)], np.float32)


def group_entries(descriptors, config):
    entries, two_way = config
    grouped = {sid: [] for sid in descriptors}
    for entry in entries:
        name = entry[0]
        owners = [r for r in descriptors.values() if name in r.names]
        if len(owners) != 1:
            known = [n for r in descriptors.values() for n in r.names]
            raise ValueError(f"mounts: no mount target {name!r} (mountable: {known})" if not owners
                             else f"mounts: {name!r} names {len(owners)} controls")
        descriptor = owners[0]
        if two_way and not descriptor.root_reaction:
            raise ValueError(f"{descriptor.system_id}: two_way needs the root's hold, measured with rod_direct only")
        grouped[descriptor.system_id].append(entry)
    return {sid: tuple(rows) for sid, rows in grouped.items() if rows}


def publish_mounts(scene, declarations, builder):
    config = getattr(scene, "_mount_declaration", None)
    if config is None:
        return
    if hasattr(scene, "mount_domains"):
        raise ValueError("duplicate coupling declarations")
    groups = []
    descriptors = getattr(scene, "mountable_domains", {})
    for sid, entries in group_entries(descriptors, config).items():
        descriptor = descriptors[sid]
        M, E = len(entries), scene.envs
        index = np.asarray([descriptor.names.index(entry[0]) for entry in entries], np.int32)
        shared = Domain("mount", f"{sid}_mount", f"i4h:{sid}_mount_rows", (M,), descriptor.target_domain.frequency)
        declarations.domain(shared, worlds=np.full(M, -1, np.int32), owners=index)
        runtime = Domain("mount", f"{sid}_mount_runtime", f"i4h:{sid}_mount_runtime_rows", (E, M), newton.Model.AttributeFrequency.BODY)
        declarations.domain(runtime, worlds=np.broadcast_to(np.arange(E, dtype=np.int32)[:, None], (E, M)),
                            owners=partial(body_owners, ref(builder), entries, E))
        fields = {}
        for key, domain, owner, dtype, values, reference in (
            ("target", shared, Owner.MODEL, wp.int32, index, None),
            ("offset", shared, Owner.MODEL, wp.transform, np.asarray([(*r[2], *r[3]) for r in entries], np.float32), None),
            ("body", runtime, Owner.MODEL, wp.int32, partial(resolve_bodies, entries=entries, envs=E), newton.Model.AttributeFrequency.BODY),
            ("force", runtime, Owner.STATE, wp.vec3, np.zeros((E, M, 3), np.float32), None),
            ("torque", runtime, Owner.STATE, wp.vec3, np.zeros((E, M, 3), np.float32), None),
        ):
            f = Field(domain, key, owner, dtype, domain.shape, "local target" if key == "target" else "shared mount or world, mount",
                      "clear selected pending reactions only" if owner is Owner.STATE else "persistent mapping",
                      references=reference, attribute_name=f"{sid}_mount_{key}")
            declarations.add(f, values)
            fields[key] = f
        groups.append((sid, entries, fields))
    domain = Domain("mount", "mounts", "i4h:mounts_rows", (scene.envs,), "world")
    declarations.domain(domain, worlds=np.arange(scene.envs, dtype=np.int32), owners=np.full(scene.envs, -1, np.int32))
    f = Field(domain, "env_xform", Owner.MODEL, wp.transform, domain.shape, "world", "persistent component transform")
    declarations.add(f, partial(environment_values, scene_ref=ref(scene)))
    declarations.register_package(records=((scene, "mount_domains", dict(config=config, groups=tuple(groups), env_xform=f)),))
    declarations.initializer(partial(validate_model_mapping, groups=tuple(groups), envs=scene.envs))


def validate_model_mapping(model, *, groups, envs):
    """Reject host entity changes between completion and retained-builder finalization."""
    for sid, entries, fields in groups:
        expected = resolve_bodies(model, entries, envs)
        actual = getattr(model.i4h, fields["body"].name).numpy().reshape(expected.shape)
        if not np.array_equal(actual, expected):
            raise RuntimeError("host mount body layout changed; reconstruct the model")
