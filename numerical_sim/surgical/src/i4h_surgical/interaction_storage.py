# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only declarations for surgical interaction and tissue history.

Centreline nodes, hook anchors, grasp and clip slots, the tool-share residue, heat, burn, coagulation, surface
colour, bleeding, emission and smoke are Newton State. Templates
carry each array's construction value; the systems' own resets are unchanged and clear only what they cleared.
Capacities come from the authored scene and the system settings, never from system instances.
"""
import numpy as np
import warp as wp
from i4h_physics.history_storage import RECORDS, Layout, bind_history, published
from i4h_physics.storage import defer_declaration, scene_record
from i4h_physics.surface_storage import support_domain


def tools(scene):
    return len(getattr(scene, "instruments", {}))


def runtime(scene, instruments):
    """The control system's ``(E, n)`` runtime rows and the instrument names, for owner references."""
    record = getattr(scene, "instrument_domains", {}).get(instruments)
    if record is None:
        raise ValueError(f"interaction storage: {instruments!r} is not a declared instrument_control; declare it first")
    return record["domains"]["runtime"], record["names"]


def tool_rows(scene, shape, names, tool):
    """Owner rows in the control's runtime domain: one tool for every row, or the tool on axis 1."""
    E, n = scene.envs, len(names)
    worlds = np.arange(E, dtype=np.int64).reshape((E,) + (1,) * (len(shape) - 1))
    index = np.arange(n, dtype=np.int64).reshape((1, n) + (1,) * (len(shape) - 2)) if tool is None else tool
    return np.broadcast_to(worlds * n + index, shape)


def particle_starts(scene, body):
    return lambda: [scene_record(scene).starts[e] + body.offset for e in range(scene.envs)]


def residue_domain(layout, body, starts):
    """Existing surface rows of the body when there are any; else this system's own particle rows."""
    scene = layout.scene
    d = support_domain(scene, body)
    if d is not None:
        return d
    E, N = scene.envs, body.count
    def owners():
        first = np.asarray(starts() if callable(starts) else starts, np.int64)
        return first[:, None] + np.arange(N, dtype=np.int64)[None]
    return layout.domain("particle", (E, N), "particle", owners)


def slot_fields(layout, d, ints, vectors=(), floats=(), triples=()):
    for key, default in ints:
        layout.add(key, d, wp.int32, default=default)
    for key in floats:
        layout.add(key, d, wp.float32, default=0.0)
    for key in vectors:
        layout.add(key, d, wp.vec3, default=0.0)
    for key in triples:
        layout.add(key, d, wp.vec3i, default=-1)


# --- centreline ----------------------------------------------------------------------------------------------

def publish_centreline(scene, declarations, system_id, *, body):
    b = scene.body(body) if isinstance(body, str) else body
    c = getattr(b, "centrelines", None)
    if c is None:
        raise ValueError(f"centreline: body {b.name!r} has no centrelines")
    E, K = scene.envs, c.clamp_start.shape[1]
    variant = np.asarray(b.variant_of_env)
    layout = Layout(scene, declarations, system_id, "centreline")
    node = layout.domain("node", (E, K), "world", np.full((E, K), -1, np.int32))
    layout.add("positions", node, wp.vec3, values=np.asarray(c.rest, np.float32)[variant])
    layout.add("active_count", node, wp.int32, values=np.asarray(c.clamp_count, np.int32)[variant])
    return layout.publish(body=b, nodes=K)


def declare_centreline(scene, *, body, system_id="centreline"):
    defer_declaration(scene, system_id, publish_centreline, dict(body=body))


# --- instrument contact residue -----------------------------------------------------------------------------

def publish_contact(scene, declarations, system_id, *, surface, records=8):
    surfaces = getattr(scene, "surface_domains", {})
    if surface not in surfaces:
        raise ValueError(f"instrument_contact: {surface!r} is not a declared surface; declare it first")
    body = surfaces[surface]["body"]
    layout = Layout(scene, declarations, system_id, "instrument_contact")
    n = max(tools(scene), 1)
    layout.residue(support_domain(scene, body), scene.envs * body.count, n)
    return layout.publish(body=body, surface=surface, records=int(records), tools=n)


def declare_instrument_contact(scene, *, surface="surface", records=8, system_id="instrument_contact"):
    defer_declaration(scene, system_id, publish_contact, dict(surface=surface, records=records))


# --- hook ----------------------------------------------------------------------------------------------------

def publish_hook(scene, declarations, system_id, *, instruments, instrument):
    rows, names = runtime(scene, instruments)
    if instrument not in names:
        raise ValueError(f"hook: unknown instrument {instrument!r}; known: {list(names)}")
    layout = Layout(scene, declarations, system_id, "hook")
    shape = (scene.envs, 2)
    anchor = layout.domain("anchor", shape, rows.frequency, tool_rows(scene, shape, names, names.index(instrument)))
    slot_fields(layout, anchor, (("tet", -1), ("active", 0)), vectors=("bary", "radial"), floats=("friction_coord",),
                triples=("verts",))
    return layout.publish(instrument=instrument)


def declare_hook(scene, *, instrument, instruments="instrument_control", system_id="hook"):
    defer_declaration(scene, system_id, publish_hook, dict(instruments=instruments, instrument=instrument))


# --- grasp ---------------------------------------------------------------------------------------------------

def grasp_slots(scene, records):
    """Slots per instrument: every jaw capsule can hold all of its contact records."""
    from i4h_surgical.instrument_storage import geometry_values
    from i4h_surgical.systems.instruments.instrument_control import PRIM_KINDS
    values, (n, _, _), _ = geometry_values(scene)
    kinds, starts, counts = (np.asarray(values[key][1][1]).reshape(-1) for key in ("prim_kind", "prim_start", "prim_counts"))
    jaw = (kinds == PRIM_KINDS["jaw_left"]) | (kinds == PRIM_KINDS["jaw_right"])
    return max([int(jaw[starts[k]:starts[k] + counts[k]].sum()) * int(records) for k in range(n)] + [1])


def publish_grasp(scene, declarations, system_id, *, instruments, contacts):
    contact = getattr(scene, RECORDS, {}).get(contacts)
    if contact is None or contact["kind"] != "instrument_contact":
        raise ValueError(f"grasp: {contacts!r} is not a declared instrument_contact; declare it first")
    rows, names = runtime(scene, instruments)
    E, n, G = scene.envs, len(names), grasp_slots(scene, contact["records"])
    layout = Layout(scene, declarations, system_id, "grasp")
    slot = layout.domain("slot", (E, n, G), rows.frequency, tool_rows(scene, (E, n, G), names, None))
    slot_fields(layout, slot, (("prim", -1), ("tet", -1), ("active", 0)), vectors=("radial", "prev_axis", "bary"),
                floats=("t", "gap"))
    layout.add("verts", slot, wp.vec3i, default=0)
    for key in ("count", "pending"):
        layout.add(key, rows, wp.int32, default=0)
    body = contact["body"]
    layout.residue(support_domain(scene, body), E * body.count, max(n, 1))
    return layout.publish(slots=G, body=body, tools=n)


def declare_grasp(scene, *, instruments="instrument_control", contacts="instrument_contact", system_id="grasp"):
    defer_declaration(scene, system_id, publish_grasp, dict(instruments=instruments, contacts=contacts))


# --- volumetric grasp ----------------------------------------------------------------------------------------

def publish_volumetric_grasp(scene, declarations, system_id, *, instruments, body, capacity=64, starts=None):
    b = scene.body(body) if isinstance(body, str) else body
    rows, names = runtime(scene, instruments)
    E, n, G = scene.envs, len(names), int(capacity)
    layout = Layout(scene, declarations, system_id, "volumetric_grasp")
    slot = layout.domain("slot", (E, n, G), rows.frequency, tool_rows(scene, (E, n, G), names, None))
    slot_fields(layout, slot, (("particle", -1), ("sphere", 0), ("active", 0)), vectors=("rest",))
    for key in ("count", "pending", "claim"):
        layout.add(key, rows, wp.int32, default=0)
    layout.residue(residue_domain(layout, b, starts or particle_starts(scene, b)), E * b.count, max(n, 1))
    return layout.publish(slots=G, body=b, tools=n)


def declare_volumetric_grasp(scene, *, body, capacity=64, instruments="instrument_control", system_id="volumetric_grasp"):
    defer_declaration(scene, system_id, publish_volumetric_grasp, dict(instruments=instruments, body=body, capacity=capacity))


# --- clipping ------------------------------------------------------------------------------------------------

def publish_clipping(scene, declarations, system_id, *, centreline, instruments, instrument, slots=4):
    centre = getattr(scene, RECORDS, {}).get(centreline)
    if centre is None or centre["kind"] != "centreline":
        raise ValueError(f"clipping: {centreline!r} is not a declared centreline; declare it first")
    rows, names = runtime(scene, instruments)
    if instrument not in names:
        raise ValueError(f"clipping: unknown instrument {instrument!r}; known: {list(names)}")
    E, G = scene.envs, int(slots)
    layout = Layout(scene, declarations, system_id, "clipping")
    slot = layout.domain("slot", (E, G), rows.frequency, tool_rows(scene, (E, G), names, names.index(instrument)))
    slot_fields(layout, slot, (("node", -1), ("active", 0)))
    layout.add("count", layout.world(), wp.int32, default=0)
    layout.add("clipped", centre["domains"]["node"], wp.int32, default=0)  # latched per centreline node
    return layout.publish(slots=G, nodes=centre["nodes"], instrument=instrument)


def declare_clipping(scene, *, instrument, centreline="centreline", instruments="instrument_control", slots=4,
                     system_id="clipping"):
    defer_declaration(scene, system_id, publish_clipping,
                      dict(centreline=centreline, instruments=instruments, instrument=instrument, slots=slots))


# --- thermal and appearance ----------------------------------------------------------------------------------

def surface_record(scene, surface, user):
    record = getattr(scene, "surface_domains", {}).get(surface)
    if record is None:
        raise ValueError(f"{user}: {surface!r} is not a declared surface; declare it first")
    return record


def publish_thermal(scene, declarations, system_id, *, surface, starts=None):
    """Heat and burn per tet on the system's own tet rows; coagulation on the surface's particle rows."""
    from i4h_surgical.tet_storage import declare_thermal_tet_fields
    record = surface_record(scene, surface, "thermal")
    body = record["body"]
    layout = Layout(scene, declarations, system_id, "thermal")
    tet = layout.domains["tet"] = declare_thermal_tet_fields(declarations, body, system_id, starts or particle_starts(scene, body))
    for key in ("heat", "burn"):
        layout.fields[key] = (declarations.fields[tet, key], None, tet.shape)
    layout.add("coag", record["domains"][4], wp.float32, default=0.0, flat=True)
    return layout.publish(body=body, surface=surface)


def declare_thermal(scene, *, surface="surface", system_id="thermal"):
    defer_declaration(scene, system_id, publish_thermal, dict(surface=surface))


def publish_appearance(scene, declarations, system_id):
    """The surgical surface's colour per particle, on that surface's own particle rows."""
    record = surface_record(scene, system_id, "appearance")
    layout = Layout(scene, declarations, system_id, "appearance")
    layout.domains["support"] = record["domains"][4]
    layout.add("visual", record["domains"][4], wp.vec4f, default=0.0, flat=True)
    return layout.publish(body=record["body"])


def declare_appearance(scene, *, system_id="surface"):
    """For a surgical Surface built on a Scene directly; call after ``declare_surface``."""
    defer_declaration(scene, f"{system_id}:appearance", lambda scene, declarations, key: publish_appearance(scene, declarations, system_id), {})


# --- bleeding, emitter, smoke, suction ------------------------------------------------------------------------

def publish_bleeding(scene, declarations, system_id, *, centreline):
    centre = published(scene, centreline, "centreline", "bleeding")
    layout = Layout(scene, declarations, system_id, "bleeding")
    node = centre["domains"]["node"]
    for key, dtype in (("cut", wp.int32), ("source", wp.int32), ("acc", wp.float32)):
        layout.add(key, node, dtype, default=0)
    layout.add("serial", layout.world(), wp.int32, default=0)
    return layout.publish(nodes=centre["nodes"], centreline=centreline)


def declare_bleeding(scene, *, centreline="centreline", system_id="bleeding"):
    defer_declaration(scene, system_id, publish_bleeding, dict(centreline=centreline))


def publish_emitter(scene, declarations, system_id):
    layout = Layout(scene, declarations, system_id, "emitter")
    world = layout.world()
    layout.add("acc", world, wp.float32, default=0)
    layout.add("serial", world, wp.int32, default=0)
    return layout.publish()


def declare_emitter(scene, *, system_id="emitter"):
    defer_declaration(scene, system_id, publish_emitter, {})


def publish_smoke(scene, declarations, system_id, *, count=512):
    E, n = scene.envs, int(count)
    layout = Layout(scene, declarations, system_id, "smoke")
    particle = layout.domain("particle", (E, n), "world", np.full((E, n), -1, np.int32))
    for key in ("pos_age", "vel_life"):
        layout.add(key, particle, wp.vec4f, default=0, flat=True)
    world = layout.world()
    for key, dtype in (("cursor", wp.int32), ("acc", wp.float32), ("serial", wp.int32)):
        layout.add(key, world, dtype, default=0)
    return layout.publish(count=n)


def declare_smoke(scene, *, count=512, system_id="smoke"):
    defer_declaration(scene, system_id, publish_smoke, dict(count=count))


def publish_suction(scene, declarations, system_id, *, fluid):
    pools = published(scene, fluid, "fluid", "suction")
    layout = Layout(scene, declarations, system_id, "suction")
    layout.add("removed", pools["domains"]["pool"], wp.int32, default=0)
    return layout.publish(pools=pools["pools"], fluid=fluid)


def declare_suction(scene, *, fluid="fluid", system_id="suction"):
    defer_declaration(scene, system_id, publish_suction, dict(fluid=fluid))


# --- binding -------------------------------------------------------------------------------------------------

bind_interaction = bind_history
