# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""One fresh-process aorta_shell construction sample; no simulation or capture."""

import argparse
import importlib.util
import json
import pathlib
import resource
import subprocess
import time

import newton
import warp as wp
from i4h_endoluminal.config import LOADERS, SYSTEMS, read_case
from i4h_physics import author_case
from i4h_physics import config as physical
from i4h_physics.stepper import Stepper
from i4h_physics.storage import Owner

root = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--worlds", type=int, choices=(1, 64), required=True)
p.add_argument("--out", type=pathlib.Path, required=True)
args = p.parse_args()
assert not args.out.exists()
spec = importlib.util.spec_from_file_location(
    "construction_accounting", root / "scripts/measure_stage_c_construction.py"
)
account = importlib.util.module_from_spec(spec)
spec.loader.exec_module(account)
processes = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
)
wp.init()
device = "cuda:0"
wp.synchronize_device(device)
base = wp.get_mempool_used_mem_current(device)
started = time.perf_counter()
timings = {}
cfg = read_case("aorta_shell")
a = author_case(
    cfg,
    loaders={**physical.LOADERS, **LOADERS},
    systems={**physical.SYSTEMS, **SYSTEMS},
    device=device,
    envs=args.worlds,
)
a.scene.check_memory(a.scene.estimate_bytes(construction=True))
b = newton.ModelBuilder()
t = time.perf_counter()
a.declare(b)
timings["schema_registration_seconds"] = time.perf_counter() - t
t = time.perf_counter()
for world in range(args.worlds):
    b.begin_world(gravity=a.scene.gravity_vector)
    a.add_world(b, world)
    b.end_world()
timings["authored_worlds_seconds"] = time.perf_counter() - t
t = time.perf_counter()
a.complete(b)
timings["row_authoring_seconds"] = time.perf_counter() - t
t = time.perf_counter()
m = b.finalize(device=device)
m.particle_grid = None
wp.synchronize_device(device)
timings["finalize_seconds"] = time.perf_counter() - t
t = time.perf_counter()
s, c = m.state(), m.control()
scene, systems = a.bind(m, s, c)
sim = Stepper.create(scene, systems, control=c, **physical.sim_settings(cfg, False))
wp.synchronize_device(device)
timings["runtime_binding_seconds"] = time.perf_counter() - t
timings["total_construction_seconds"] = time.perf_counter() - started
retained = wp.get_mempool_used_mem_current(device) - base
peak = wp.get_mempool_used_mem_high(device) - base
s2, c2 = m.state(), m.control()
wp.synchronize_device(device)
extra = wp.get_mempool_used_mem_current(device) - base - retained
shared = m.control(clone_variables=False)


def buffers(container):
    result = {}
    for key, value in vars(container).items():
        if isinstance(value, wp.array):
            result[key] = value
        elif isinstance(value, newton.Model.AttributeNamespace):
            result.update({f"{key}:{k}": v for k, v in vars(value).items() if isinstance(v, wp.array)})
    return result


ma = buffers(m)
templates = {
    k: v
    for k, v in ma.items()
    if m.attribute_assignment.get(k)
    in (newton.Model.AttributeAssignment.STATE, newton.Model.AttributeAssignment.CONTROL)
    or k in ("particle_q", "particle_qd")
}
model = {k: v for k, v in ma.items() if k not in templates}
records = {}
pointers = {}
for family in ("rod_body_domains", "rod_domains", "rod_material_domains", "endoluminal_domains"):
    for sid, record in getattr(scene, family, {}).items():
        fields = record.fields
        logical = {o.value: 0 for o in Owner}
        padding = dict(logical)
        for f in fields.values():
            size = wp.types.type_size_in_bytes(f.backing_dtype)
            logical[f.owner.value] += f.domain.count * size
            if not f.domain.count:
                padding[f.owner.value] += size
        records[f"{family}:{sid}"] = dict(
            logical_bytes=logical,
            padding_bytes=padding,
            initialization_template_bytes=logical["state"] + logical["control"],
            template_padding_bytes=padding["state"] + padding["control"],
            identity_bytes=sum(max(1, d.count) * 8 for d in record.domains.values() if d.custom),
        )
        for key, f in fields.items():
            if f.owner is Owner.MODEL:
                continue
            live = s if f.owner is Owner.STATE else c
            checkpoint = s2 if f.owner is Owner.STATE else c2
            ptrs = dict(
                template=getattr(m.i4h, f.name).ptr,
                live=getattr(live.i4h, f.name).ptr,
                checkpoint=getattr(checkpoint.i4h, f.name).ptr,
                template_alias=getattr(shared.i4h, f.name).ptr if f.owner is Owner.CONTROL else None,
            )
            assert len({ptrs["template"], ptrs["live"], ptrs["checkpoint"]}) == 3
            if f.owner is Owner.CONTROL:
                assert ptrs["template_alias"] == ptrs["template"]
            pointers[f"{family}:{sid}:{key}"] = ptrs
report = dict(
    workload="aorta_shell",
    worlds=args.worlds,
    scope="post-freeze construction only; no stepping or capture",
    timings=timings,
    retained_warp_bytes=retained,
    construction_peak_warp_bytes=peak,
    additional_state_control_bytes=extra,
    host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    declared_bytes=scene.storage.allocated_bytes(),
    rod_and_control_storage=records,
    pointers=pointers,
    container_bytes={
        k: account.byte_count(v.values())
        for k, v in dict(
            model_buffers=model,
            initialization_templates=templates,
            live_state=buffers(s),
            live_control=buffers(c),
            checkpoint_state=buffers(s2),
            checkpoint_control=buffers(c2),
            intentionally_shared_control=buffers(shared),
        ).items()
    },
    all_container_bytes_deduplicated=account.byte_count(
        [
            *ma.values(),
            *buffers(s).values(),
            *buffers(c).values(),
            *buffers(s2).values(),
            *buffers(c2).values(),
            *buffers(shared).values(),
        ]
    ),
    native_counts=dict(particles=m.particle_count, tets=m.tet_count, triangles=m.tri_count),
    compute_processes_at_entry=processes,
)
with args.out.open("x") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
assert sim.scene is scene
print(json.dumps(report, indent=2))
