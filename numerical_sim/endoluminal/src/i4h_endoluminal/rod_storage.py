# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal-owned catheter, layout and drag storage declarations."""

from functools import partial

import numpy as np
import warp as wp

from i4h_physics.rod_storage import Record, alias, bind_record, body_signature
from i4h_physics.storage import Owner, defer_declaration

CATHETER_LIVE = ("track_length", "track_stiffness", "max_delta", "rates", "lower", "upper")


def declare_catheter_control(scene, rod, *, track_length, system_id="catheter_control", **settings):
    defer_declaration(
        scene, system_id, publish_catheter_control, dict(rod=rod, settings=dict(track_length=track_length, **settings))
    )


def publish_catheter_control(scene, declarations, system_id, *, rod, settings):
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl

    solver = getattr(scene, "rod_domains", {}).get(rod)
    if solver is None:
        raise ValueError("catheter_control.rod must name an earlier declared rod solver")
    body = solver.body
    if body.rod is None:
        raise TypeError("catheter_control drives a straight rod asset, not a centreline tree")
    r = Record(scene, declarations, system_id, body)
    r.kind_name, r.rod = "catheter_control", rod
    r.backend = solver.backend
    r.signature = body_signature(body, solver.backend)
    r.tip_edges = settings.get("tip_edges", 0)
    if not isinstance(r.tip_edges, int) or not 0 <= r.tip_edges < solver.shared.K:
        raise ValueError("catheter_control.tip_edges outside declared bend range")
    for key in CATHETER_LIVE:
        value = settings.get(key, getattr(CatheterControl, key, None))
        if key in ("rates", "lower", "upper") and np.asarray(value).shape != (3,):
            raise ValueError(f"catheter_control.{key} must have 3 entries")
        r.live(key, value)
    m = body.rod
    for key, dtype, data in (
        ("track_start", wp.vec3, m.start),
        ("track_direction", wp.vec3, m.direction),
        ("base_orientation", wp.quat, np.ascontiguousarray(m.frames[:, 0])),
    ):
        r.add(key, Owner.MODEL, dtype, data, nodes=np.zeros(1, np.int32), group="track")
    E = scene.envs
    r.add("actions", Owner.CONTROL, wp.float32, np.zeros((E, 3), np.float32), runtime=True, width=3, group="world")
    r.add("targets", Owner.STATE, wp.vec3, np.zeros((E, 3), np.float32), runtime=True, group="world")
    r.add(
        "root",
        Owner.STATE,
        wp.vec3,
        m.start[body.variant_of_env].reshape(E, 1, 3).astype(np.float32),
        runtime=True,
        nodes=np.zeros((E, 1), np.int32),
    )
    from i4h_physics.mount_storage import publish_mountable
    v = body.variant_of_env
    poses = np.concatenate((m.start[v], m.frames[v, 0]), axis=1).reshape(E, 1, 7).astype(np.float32)
    mountable = publish_mountable(scene, declarations, system_id, (body.name,), None, r.fields["root"].domain,
                                 r.fields["root"], poses, root_reaction=solver.backend == "direct")
    r.fields.update(mounted=mountable.mounted, mount_pose=mountable.mount_pose)
    declarations.register_package(records=((scene, "endoluminal_domains", {system_id: r}),))
    return r


def declare_control_layout(scene, centreline, *, system_id="control_layout", rod=None):
    defer_declaration(scene, system_id, publish_control_layout, dict(centreline=centreline, rod=rod, expected=layout_signature(scene.body(centreline), scene.envs)))


def layout_signature(body, E):
    return E, repr(body.anatomy.layouts)


def publish_control_layout(scene, declarations, system_id, *, centreline, rod=None, expected=None):
    from i4h_endoluminal.systems.rods.control_layout import layout_values

    body = scene.body(centreline)
    if expected is not None and layout_signature(body, scene.envs) != expected:
        raise RuntimeError("control layout changed after declaration; reconstruct the model")
    shared = getattr(scene, "rod_body_domains", {}).get(body.name)
    if body.tree is None or shared is None or shared.backend != "pbd" or shared.solver is None:
        raise ValueError("control_layout requires an earlier rod_pbd on the same tree body")
    if rod is not None and rod != shared.solver:
        raise ValueError("control_layout.rod must name the same tree solver")
    r = Record(scene, declarations, system_id, body)
    r.kind_name, r.backend = "control_layout", "pbd"
    r.signature = body_signature(body, "pbd")
    r.layout_signature = layout_signature(body, scene.envs)
    for key, (dtype, data) in layout_values(body, scene.envs).items():
        r.add(key, Owner.MODEL, dtype, data, runtime=key == "layout_of_env")
    declarations.register_package(records=((scene, "endoluminal_domains", {system_id: r}),))
    return r


def declare_drag(scene, body, *, system_id="drag", max_delta=0.01):
    defer_declaration(scene, system_id, publish_drag, dict(body=body, max_delta=max_delta))


def publish_drag(scene, declarations, system_id, *, body, max_delta=0.01):
    body = scene.body(body)
    if body.tree is None:
        raise ValueError("drag: body is not a centreline tree")
    r = Record(scene, declarations, system_id, body)
    r.kind_name, r.backend = "drag", "pbd"
    r.signature = body_signature(body, "pbd")
    r.live("max_delta", max_delta)
    E, P = scene.envs, body.count
    for key, dtype in (("active", wp.int32), ("node", wp.int32), ("target", wp.vec3)):
        r.add(
            key,
            Owner.CONTROL,
            dtype,
            np.zeros((E, 3) if dtype is wp.vec3 else (E,), np.float32 if dtype is wp.vec3 else np.int32),
            runtime=True,
            group="world",
        )
    for key, dtype in (("weights", wp.float32), ("baseline", wp.vec3)):
        r.add(
            key,
            Owner.STATE,
            dtype,
            np.zeros((E, P, 3) if dtype is wp.vec3 else (E, P), np.float32),
            runtime=True,
            nodes=np.arange(P),
            group="nodes_runtime",
            flat=True,
        )
    declarations.register_package(records=((scene, "endoluminal_domains", {system_id: r}),))
    return r


def validate_layout(scene, record):
    body = scene.body(record.body_name)
    if body_signature(body, record.backend) != record.signature:
        raise RuntimeError("endoluminal topology changed; reconstruct the model")
    if record.kind_name == "control_layout" and layout_signature(body, scene.envs) != record.layout_signature:
        raise RuntimeError("control layout changed; reconstruct the model")


def bind_endoluminal(system, kind):
    scene = system.scene
    system.id = system.id or kind
    r = getattr(scene, "endoluminal_domains", {}).get(system.id)
    if r is None:
        raise RuntimeError(f"{kind} storage missing; call declare_{kind} before allocation")
    body = system.centreline if kind == "control_layout" else system.body
    if r.kind_name != kind or r.body_name != body.name:
        raise ValueError("incompatible endoluminal storage declaration")
    if kind == "catheter_control" and r.rod != system.rod.id:
        raise ValueError("catheter rod declaration changed")
    validate_layout(scene, r)
    bind_record(system, r)
    alias(scene, system, "centreline" if kind == "control_layout" else "body", body)
    if kind != "drag":
        alias(scene, system, "controls", body.rod_controls)
    scene.storage.register_package(checks=(partial(validate_layout, scene, r),))
    return r
