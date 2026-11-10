# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""One construction-only sample; invoke each variant in its own fresh process.

Measures the current C2 checkpoint at one or 64 worlds, including declared
initialization templates, each live/checkpoint container and legacy residue.
No stepping, capture, timing retry, or final performance qualification occurs.
"""

import argparse
import hashlib
import inspect
import json
import resource
import subprocess
import time
from pathlib import Path

import newton
import warp as wp
from i4h_physics import author_case
from i4h_physics import config as physical
from i4h_physics.stepper import Stepper
from i4h_surgical import read_case
from i4h_surgical.config import LOADERS, SYSTEMS


def byte_count(arrays):
    # These containers hold contiguous allocations; pointer identity removes aliases.
    sizes = {}
    for array in arrays:
        if array.ptr:
            # A native allocation can also appear as a shorter Scene alias
            # (gravity excludes the native global-world row).
            sizes[array.ptr] = max(sizes.get(array.ptr, 0), array.size * wp.types.type_size_in_bytes(array.dtype))
    return sum(sizes.values())



def tet_accounting(scene):
    """Logical Tet bytes separated from identity and inaccessible backing rows."""
    result = {}
    for sid, record in scene.tet_domains.items():
        mesh = record["body"].tet
        V, T = mesh.active0.shape
        E, S, P = scene.envs, mesh.edge_active0.shape[1], len(mesh.spring)
        model = 24*V*T + 4*T + 16*V*S + 4*S + 4*E*T + 16*E*P
        state = 4*E*(T+S)
        old_model = 24*V*T + 4*T + 4*E*T
        domains = {domain for domain in record["fields"].values()}
        padding = sum(wp.types.type_size_in_bytes(field.backing_dtype) for (domain, _), field in scene.declarations.fields.items()
                      if record["fields"].get(field.key) == domain and domain.count == 0)
        result[sid] = dict(variants=V, tets=T, edges=S, parts=P, model_data_bytes=model,
                           each_state_bytes=state, initialization_template_bytes=state,
                           c22_additive_model_bytes=old_model, c22_additive_state_bytes=4*E*T,
                           c22_additive_template_bytes=4*E*T,
                           new_model_data_bytes=model-old_model, new_state_bytes=4*E*S,
                           new_template_bytes=4*E*S,
                           identity_column_bytes=sum(max(1, d.count)*8 for d in domains),
                           empty_domain_field_bytes=padding,
                           removed_legacy_bytes=model-4*E*T+state)
    return result

def surface_accounting(scene):
    """Surface data and new identities; reused Tet domains are counted there."""
    from i4h_physics.storage import Owner
    result = {}
    for sid, record in scene.surface_domains.items():
        body = record["body"]
        mesh = body.tet
        V, T = mesh.active0.shape
        E, F, N, S, P = scene.envs, mesh.tris.shape[1], body.count, mesh.edges.shape[1], len(body.parts)
        model = 56*V*T + 16*V*F + 4*F + 4*V*N + 4*V*S + 16*P
        state = 16*E*F + 4*E*N + 4*E*S + 4*E*P + 12*E
        padding = {owner.value: 0 for owner in Owner}
        for pair in record["fields"].values():
            field = scene.declarations.fields[pair]
            if field.domain.count == 0:
                padding[field.owner.value] += wp.types.type_size_in_bytes(field.backing_dtype)
        result[sid] = dict(variants=V, tets=T, edges=S, faces=F, particles=N, parts=P,
                           model_data_bytes=model, each_state_bytes=state, initialization_template_bytes=state,
                           c23_additive_model_bytes=56*V*T, new_model_data_bytes=model-56*V*T,
                           new_state_bytes=state, new_template_bytes=state,
                           identity_column_bytes=sum(max(1, d.count)*8 for d in record["domains"] if d.custom),
                           shared_tet_domains=[scene.tet_domains[record["tet_id"]][k].frequency
                                               for k in ("variant", "edge_variant", "edge_state", "materials")],
                           empty_domain_field_bytes=padding, empty_template_bytes=padding["state"],
                           removed_legacy_bytes=model+state)
    return result


def instrument_accounting(scene):
    """Logical ownership, padding, identities and temporary finalization storage."""
    from i4h_physics.storage import Owner
    result = {}
    for sid, record in getattr(scene, "instrument_domains", {}).items():
        E, (N, S, P) = scene.envs, record["counts"]
        logical = {owner.value: 0 for owner in Owner}
        padding = dict(logical)
        for field in record["fields"].values():
            logical[field.owner.value] += field.domain.count * wp.types.type_size_in_bytes(field.backing_dtype)
            if not field.domain.count:
                padding[field.owner.value] += wp.types.type_size_in_bytes(field.backing_dtype)
        assert logical["state"] == 92*E*N + 28*E*S
        assert logical["control"] == 68*E*N
        result[sid] = dict(instruments=N, spheres=S, primitives=P, logical_bytes=logical,
                           initialization_template_bytes=logical["state"]+logical["control"],
                           empty_domain_field_bytes=padding, empty_template_bytes=padding["state"]+padding["control"],
                           identity_column_bytes=sum(max(1, d.count)*8 for d in {f.domain for f in record["fields"].values()} if d.custom),
                           temporary_finalization_bytes=E,
                           removed_legacy_bytes=sum(logical.values()),
                           model_fields={k: f.domain.count*wp.types.type_size_in_bytes(f.backing_dtype)
                                         for k, f in record["fields"].items() if f.owner is Owner.MODEL})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declarations", action="store_true", help="retained compatibility flag; declarations are required")
    parser.add_argument("--worlds", type=int, choices=(1, 64), default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                         "--format=csv,noheader"], text=True)
    wp.init()
    device = "cuda:0"
    wp.synchronize_device(device)
    base_bytes = wp.get_mempool_used_mem_current(device)
    timings = {}
    started = phase = time.perf_counter()
    cfg = read_case("chole_tet_med")
    cfg["assets"].update({name: {"type": "instrument", **record} for name, record in cfg.get("instruments", {}).items()})
    authored = author_case(cfg, loaders={**physical.LOADERS, **LOADERS}, systems={**physical.SYSTEMS, **SYSTEMS},
                           device=device, envs=args.worlds)
    authored.scene.check_memory(authored.scene.estimate_bytes(construction=True))
    builder = newton.ModelBuilder()
    phase = time.perf_counter()
    authored.declare(builder)
    timings["schema_registration_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    for world in range(args.worlds):
        builder.begin_world(gravity=authored.scene.gravity_vector)
        authored.add_world(builder, world)
        builder.end_world()
    timings["authored_worlds_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    authored.complete(builder)
    declarations = authored.storage
    timings["row_authoring_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    model = builder.finalize(device=device)
    model.particle_grid = None
    wp.synchronize_device(device)
    timings["finalize_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    state, control = model.state(), model.control()
    scene, systems = authored.bind(model, state, control)
    sim = Stepper.create(scene, systems, control=control, **physical.sim_settings(cfg, False))
    views = scene.storage
    wp.synchronize_device(device)
    timings["runtime_binding_seconds"] = time.perf_counter() - phase
    timings["total_construction_seconds"] = time.perf_counter() - started
    retained = wp.get_mempool_used_mem_current(device) - base_bytes
    peak = wp.get_mempool_used_mem_high(device) - base_bytes
    # Additional containers are measured after the construction peak, separately.
    extra_state, extra_control = model.state(), model.control()
    wp.synchronize_device(device)
    extra_bytes = wp.get_mempool_used_mem_current(device) - base_bytes - retained
    rows = sum(max(1, d.count) for d in declarations._domains if d.custom) if declarations else 0
    source_files = [Path(physical.__file__), Path(inspect.getfile(type(sim))), Path(inspect.getfile(type(authored)))]
    native = Path(wp.__file__).parent / "bin" / "warp.so"
    def buffers(container):
        found = {}
        for name, value in vars(container).items():
            if isinstance(value, wp.array):
                found[name] = value
            elif isinstance(value, newton.Model.AttributeNamespace):
                found.update({f"{name}:{key}": array for key, array in vars(value).items() if isinstance(array, wp.array)})
        return found

    model_arrays = buffers(model)
    templates = {key: array for key, array in model_arrays.items()
                 if model.attribute_assignment.get(key) in (newton.Model.AttributeAssignment.STATE, newton.Model.AttributeAssignment.CONTROL)
                 or key in ("particle_q", "particle_qd")}
    model_buffers = {key: array for key, array in model_arrays.items() if key not in templates}
    shared = model.control(clone_variables=False)
    residue = {}
    for cls in ("InstrumentContact", "Grasp", "VolumetricGrasp"):
        for key in ("delta_k", "delta_jaw"):
            entries = {system.id or str(index): getattr(system, key) for index, system in enumerate(systems)
                       if type(system).__name__ == cls}
            residue[f"{cls}.{key}"] = {"bytes": byte_count(entries.values()), "owner": "legacy system State",
                                        "instances": {sid: a.size * wp.types.type_size_in_bytes(a.dtype) for sid, a in entries.items()},
                                        "newton_template_bytes": 0, "checkpoint_bytes": 0}
    report = dict(
        workload="chole_tet_med", worlds=args.worlds, instrument_contact_records=next(s["records"] for s in cfg["systems"] if s["type"] == "instrument_contact"),
        variant="C3.3 Newton-owned mounted flags, mount poses and pending reactions",
        scope="construction only; no steady-state performance claim",
        timings=timings, custom_domain_rows=rows, declared_fields=len(declarations.fields) if declarations else 0,
        completion_microseconds_per_row=(1e6 * timings["row_authoring_seconds"] / rows) if rows else None,
        retained_warp_bytes=retained, construction_peak_warp_bytes=peak,
        additional_state_control_bytes=extra_bytes, host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        declared_bytes=views.allocated_bytes(),
        container_bytes=dict(model_buffers=byte_count(model_buffers.values()), initialization_templates=byte_count(templates.values()),
                             live_state=byte_count(buffers(state).values()), live_control=byte_count(buffers(control).values()),
                             checkpoint_state=byte_count(buffers(extra_state).values()), checkpoint_control=byte_count(buffers(extra_control).values()),
                             intentionally_shared_control=byte_count(buffers(shared).values()),
                             shared_control_template_overlap=byte_count(a for a in buffers(shared).values() if a.ptr in {v.ptr for v in templates.values()})),
        pool_storage=dict(logical_particles=scene.pool_count,
                          model_fields_bytes=max(1, scene.pool_count) * 24,
                          state_fields_bytes=max(1, scene.pool_count) * 40,
                          initialization_template_bytes=max(1, scene.pool_count) * 40,
                          identity_column_bytes=max(1, scene.pool_count) * 8,
                          inaccessible_padding_rows=int(scene.pool_count == 0)),
        tet_storage=tet_accounting(scene),
        surface_storage=surface_accounting(scene),
        instrument_storage=instrument_accounting(scene),
        ordinary_shared_live_control=dict(same_container=True, additional_bytes=0,
                                         aliases_template=control is shared,
                                         buffer_bytes=byte_count(buffers(control).values())),
        instrument_pointers={sid: {key: dict(template=getattr(model.i4h, f.name).ptr,
                                            live=getattr(state.i4h if f.owner.value == "state" else control.i4h, f.name).ptr,
                                            checkpoint=getattr(extra_state.i4h if f.owner.value == "state" else extra_control.i4h, f.name).ptr,
                                            template_alias=getattr(shared.i4h, f.name).ptr if f.owner.value == "control" else None)
                                   for key, f in record["fields"].items() if f.owner.value != "model"}
                             for sid, record in getattr(scene, "instrument_domains", {}).items()},
        residue_arrays=residue, residue_total_bytes=sum(row["bytes"] for row in residue.values()),
        native_model_buffer_bytes={key: array.size * wp.types.type_size_in_bytes(array.dtype) for key, array in model_buffers.items()},
        removed_allocations=["Scene main.q0 clone", "Scene main.inv_mass clone", "Scene main.flags0 clone", "Scene main.q_prev clone",
                             "Model.i4h.particle_q0 custom rest copy", "Scene._allocate_block all eight pool arrays",
                             "Scene.allocate body variant uploads", "Scene.allocate rotated gravity upload",
                             "Scene.static_wp_mesh geometry uploads",
                             "Tet topology/default masks/part uploads", "Tet live mask allocations and constructor reset",
                             "Tet four live material uploads", "Surface twelve Model uploads",
                             "Surface eight State allocations and constructor reset mask",
                             "Instrument geometry and live Model parameters", "Instrument State and packed Control runtime allocations",
                             "Instrument constructor reset", "Instrument mounted flags and mount-pose runtime allocations",
                             "Instrument finalization temporary mount flags and poses"],
        remaining_duplicates=["native inverse mass versus exact i4h inverse mass", "native flags versus i4h-semantic default flags",
                              "State history/flags initialization templates", "pool State initialization templates (40 bytes/row)", "Tet mask initialization templates", "Surface State initialization templates", "Thermal heat/burn", "Appearance legacy", "Instrument State/Control initialization templates"],
        native_counts=dict(particles=model.particle_count, tets=model.tet_count, triangles=model.tri_count),
        compute_processes_at_entry=processes,
        warp_native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        source_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    )
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    # Keep all measured allocations alive until after the report is emitted.
    assert extra_state is not state and extra_control is not control and sim.scene is scene
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
