# SPDX-License-Identifier: Apache-2.0
"""Reproducible standalone numerical campaign; no Arena or Workflow dependencies."""

import argparse
import json
from pathlib import Path

import numpy as np
from i4h_surgical.config import load_case, read_case
from i4h_surgical.grasp_geometry import normals, select_targets
from i4h_surgical.grasp_probe import GraspTargetProbe
from i4h_surgical.grasp_snapshot import HarnessSnapshot, identity
from i4h_surgical.grasp_teacher import GraspTeacher


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def diagnostics(q, body, rest):
    a, b, c, d = np.moveaxis(q[:, body.tet.tets[0]], 2, 0)
    signed = np.sum(np.cross(b - a, c - a) * (d - a), axis=-1) / 6
    ratio = signed / body.tet.rest_volume[0]
    pins = body.pin.index[0]
    drift = np.linalg.norm(q[:, pins] - body.pin.rest[0], axis=-1)
    return dict(
        finite=bool(np.isfinite(q).all()),
        inverted=int((ratio < 0).sum()),
        near_degenerate=int((np.abs(ratio) < 0.01).sum()),
        minimum_volume_ratio=float(ratio.min()),
        attachment_max_drift_m=float(drift.max()),
        attachment_mean_drift_m=float(drift.mean()),
        displacement_max_m=float(np.linalg.norm(q - rest, axis=-1).max()),
    )


def settle(sim, probe, out):
    q = sim.scene.field("q", probe.body)
    previous = q.numpy().copy()
    trace, positions = [], []
    streak = 0
    for step in range(1, 1801):
        sim.step()
        current = q.numpy()
        speed = float(np.linalg.norm(current - previous, axis=-1).max() / sim.dt)
        previous = current.copy()
        streak = streak + 1 if speed < 0.005 else 0
        row = dict(
            step=step,
            time_s=step * sim.dt,
            max_speed_m_s=speed,
            streak=streak,
            **diagnostics(current, probe.body, probe.rest),
        )
        trace.append(row)
        positions.append(current)
        if not row["finite"]:
            break
        if step >= 180 and streak >= 60:
            break
    np.savez_compressed(
        out / "settling.npz",
        positions=np.asarray(positions),
        actions=np.repeat(probe.ic.actions.numpy()[None], len(positions), axis=0),
    )
    with (out / "settling.jsonl").open("w") as f:
        for row in trace:
            f.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
    return dict(
        status="settled" if step >= 180 and streak >= 60 else "not_settled", **trace[-1]
    )


def reproducibility(sim, snapshot, counters):
    trials, exact = [], []
    for _ in range(3):
        snapshot.restore(counters)
        exact.append(all(snapshot.exact().values()))
        sim.step()
        trials.append(sim.scene.field("q", sim.scene.body("tissue")).numpy())
    errors = [float(np.linalg.norm(q - trials[0], axis=-1).max()) for q in trials[1:]]
    snapshot.restore(counters)
    return dict(
        exact_state_control=exact,
        max_pointwise_differences_m=errors,
        passed=bool(all(exact) and max(errors) <= 0.00001),
    ), trials[0]


def manifest(sim, probe, snapshot):
    b, ic = probe.body, probe.ic
    q = sim.scene.field("q", b).numpy()[0]
    jaw_ids = probe.jaws.numpy()
    sphere_q = ic.sphere_q.numpy()[0]
    active = ic.sphere_active.numpy()[0, jaw_ids] != 0
    centre = sphere_q[jaw_ids[active]].mean(axis=0)
    trocar = list(sim.scene.instruments.values())[probe.k].trocar
    targets, facing = select_targets(
        probe.rest, q, probe.triangles, probe.eligible, trocar, centre
    )
    normal, _ = normals(q, probe.triangles)
    roots = ic.actions.numpy()[0, 0, :3]
    starts = [roots + [x, 0, 0] for x in (0, -0.005, 0.005)]
    goals = list(starts)
    measurements = []
    radii = ic.sphere_radius.numpy()
    pr = sim.scene.field("radius", b).numpy()[0]
    for t in targets:
        nearest = int(
            jaw_ids[active][
                np.argmin(np.linalg.norm(sphere_q[jaw_ids[active]] - q[t], axis=1))
            ]
        )
        envelope = float(radii[nearest] + pr[t])
        pre = roots + q[t] + normal[t] * (envelope + 0.005) - sphere_q[nearest]
        closure = roots + q[t] + normal[t] * (envelope - 0.00025) - sphere_q[nearest]
        retract = closure + [0, 0.01, 0]
        goals.extend([pre, closure, retract])
        measurements.append(
            dict(
                vertex=t,
                patch=probe.patch_ids[t, : probe.patch_counts[t]].tolist(),
                settled_position=q[t].tolist(),
                normal=normal[t].tolist(),
                facing=float(facing[t]),
                nearest_sphere=nearest,
                pregrasp_root_goal=pre.tolist(),
                closure_root_goal=closure.tolist(),
                retract_root_goal=retract.tolist(),
            )
        )
    goals = np.asarray(goals)
    workspace = [list(goals.min(axis=0) - 0.02), list(goals.max(axis=0) + 0.02)]
    return dict(
        snapshot_hash=snapshot.hash,
        targets=measurements,
        median_surface_edge_m=probe.edge_length,
        eligible_before_facing=int(probe.eligible.sum()),
        grasp_centre=centre.tolist(),
        starting_commands=[v.tolist() for v in starts],
        workspace=workspace,
        workspace_label="controller sanity bound, not an independently validated safety or reachability constraint",
        selection="nearest settled grasp centre, then rest-space farthest point within 30 mm; ties by vertex ID",
    )


def attempt(sim, probe, snapshot, counters, target, offset, frozen, directory):
    snapshot.restore(counters, probe)
    if not all(snapshot.exact().values()):
        raise RuntimeError("exact restoration failed before scoring")
    probe.select(target)
    probe.sample()
    selected = probe.read()
    normal = selected["geometry"][0, 1]
    trocar = sim.scene.instruments["grasper"].trocar
    toward = trocar - selected["geometry"][0, 0]
    valid = (
        np.linalg.norm(normal) > 0.5
        and np.dot(normal, toward / max(np.linalg.norm(toward), 1.0e-20)) >= 0.25
    )
    initial = probe.ic.actions.numpy()[0, 0].copy()
    teacher = GraspTeacher(initial, frozen["workspace"], sim.dt)
    if not valid:
        teacher.fail("restored_eligibility_disagreement")
    # Start variants are open commands; tissue is always restored from the same snapshot.
    start_goal = initial[:3] + [offset, 0, 0]
    rows, actions, positions, geometries, displacements, residuals, identities = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    acquisition = None
    force = next(s for s in sim.systems if s.id == "tool_force")
    while not teacher.done:
        m = {k: v[0] for k, v in selected.items()}
        if teacher.step % 2 == 0:
            sphere = int(m["nearest"])
            previous_phase = teacher.phase
            if np.linalg.norm(teacher.action[:3] - start_goal) > 1.0e-7 and not getattr(
                teacher, "start_ready", False
            ):
                delta = start_goal - teacher.action[:3]
                teacher.action[:3] += delta * min(
                    1.0, 0.001 / max(np.linalg.norm(delta), 1.0e-20)
                )
            else:
                teacher.start_ready = True
                teacher.command(
                    m,
                    float(probe.ic.sphere_radius.numpy()[sphere]),
                    float(sim.scene.field("radius", probe.body).numpy()[0, target]),
                    probe.ic.jaw_angle.numpy()[0, 0],
                )
            if teacher.phase == "capture" and previous_phase != "capture":
                probe.arm()
            if teacher.phase == "release" and teacher.release_step is None:
                teacher.release_step = teacher.step
            a = probe.ic.actions.numpy()
            a[0, 0] = teacher.action
            probe.ic.actions.assign(a)
        if teacher.done:
            break
        actions.append(probe.ic.actions.numpy().reshape(sim.scene.envs, 10))
        sim.step()
        counters["step"] += 1
        counters["time_s"] += sim.dt
        probe.sample()
        selected = probe.read()
        m = {k: v[0] for k, v in selected.items()}
        q = sim.scene.field("q", probe.body).numpy()
        diag = diagnostics(q, probe.body, probe.rest)
        sc = int(probe.surface.count.numpy()[0, 0])
        vc = int(probe.volume.count.numpy()[0, 0])
        sa = int(np.count_nonzero(probe.surface.active.numpy()[0, 0]))
        va = int(np.count_nonzero(probe.volume.active.numpy()[0, 0]))
        teacher.observe(m, probe.ic.root.numpy()[0, 0], sc, vc, sa, va, diag["finite"])
        if teacher.acquisition_step is not None and acquisition is None:
            acquisition = {
                key: selected[key][0].tolist()
                for key in (
                    "original_particle",
                    "original_sphere",
                    "original_rest",
                    "membership",
                )
            }
        rows.append(
            dict(
                step=teacher.step,
                phase=teacher.phase,
                gap_m=float(m["gap"]),
                surface_count=sc,
                volume_count=vc,
                surface_active=sa,
                volume_active=va,
                acquisition_size=int(m["counts"][0]),
                patch_size=int(m["counts"][1]),
                valid=int(m["counts"][2]),
                retained=int(m["counts"][3]),
                invalid=int(m["counts"][4]),
                max_radial_residual_m=float(m["stats"][0]),
                mean_radial_residual_m=float(m["stats"][1]),
                destination_error_m=None
                if teacher.destination is None
                else float(np.linalg.norm(m["geometry"][0] - teacher.destination)),
                jaw_angle=float(probe.ic.jaw_angle.numpy()[0, 0]),
                root=probe.ic.root.numpy()[0, 0].tolist(),
                tool_force_sim_units=force.force.numpy()[0, 0].tolist(),
                jaw_load_sim_units=float(force.jaw_load.numpy()[0, 0]),
                **diag,
            )
        )
        positions.append(q)
        geometries.append(m["geometry"])
        displacements.append(m["displacement"])
        residuals.append(m["residual"])
        identities.append(m["lost"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "trace.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
    np.savez_compressed(
        directory / "physical_trace.npz",
        actions=actions,
        positions=positions,
        geometry=geometries,
        displacement=displacements,
        residual=residuals,
        invalid_original_bindings=identities,
    )
    result = dict(
        target=target,
        offset_x_m=offset,
        mode="captured" if sim.cuda_graph else "eager",
        status=teacher.phase,
        failure=teacher.failure,
        steps=teacher.step,
        closure=teacher.closure,
        acquisition=acquisition,
        acquisition_step=teacher.acquisition_step,
        acquisition_size=0
        if acquisition is None
        else int(sum(np.array(acquisition["original_particle"]) >= 0)),
        destination=None
        if teacher.destination is None
        else teacher.destination.tolist(),
        release_command_step=teacher.release_step,
        surface_clear_step=teacher.surface_clear_step,
        volume_clear_step=teacher.volume_clear_step,
    )
    write_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="restore a compatible canonical bundle instead of settling again",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = read_case("liver_pinned_grasp_1x", data=args.data)
    sim = load_case(cfg, device=args.device, envs=args.worlds)
    sim.prepare()
    probe = GraspTargetProbe(sim.physics)
    assert probe.body.count == 1986 and int((probe.body.pin.index[0] >= 0).sum()) == 391
    assert probe.ic.count == 1 and probe.ic.actions.reshape(
        (args.worlds, 10)
    ).shape == (args.worlds, 10)
    report = dict(
        hosted_compatibility="pending exact final Workflows runtime and recorder integration",
        functional_diagnostics="inversions and soft attachment drift reported without mechanics tuning",
    )
    ident = identity(
        sim.physics, dict(case=cfg, device=args.device, worlds=args.worlds), args.data
    )
    write_json(out / "identity.json", ident)
    counters = dict(step=0, time_s=0.0)
    if args.snapshot is None:
        report["settling"] = settle(sim, probe, out)
        print("SETTLING", report["settling"], flush=True)
        if report["settling"]["status"] != "settled":
            report["qualification"] = "blocked_not_settled"
            write_json(out / "report.json", report)
            return
        counters.update(
            step=report["settling"]["step"],
            time_s=report["settling"]["time_s"],
            settling=report["settling"],
        )
    snapshot = HarnessSnapshot(sim.physics, ident, counters)
    if args.snapshot is not None:
        snapshot.load(args.snapshot)
        evidence = snapshot.counters.get("settling", {})
        if (
            evidence.get("status") != "settled"
            or evidence.get("step", 0) < 180
            or evidence.get("streak", 0) < 60
            or evidence.get("max_speed_m_s", 1) >= 0.005
        ):
            raise ValueError("bundle lacks the required canonical settling evidence")
        snapshot.restore(counters, probe)
        report["settling"] = dict(
            status="loaded_canonical_snapshot",
            snapshot_hash=snapshot.hash,
            original_evidence=evidence,
        )
    snapshot.write(out / "settled_snapshot.npz")
    snapshot.load(out / "settled_snapshot.npz")
    report["reproducibility"] = {}
    references = []
    for captured in (False, True):
        if captured and not sim.scene.device.is_cuda:
            continue
        sim.cuda_graph = captured
        sim.prepare()
        result, reference = reproducibility(sim, snapshot, counters)
        report["reproducibility"]["captured" if captured else "eager"] = result
        references.append(reference)
    snapshot.restore(counters)

    frozen = manifest(sim, probe, snapshot)
    write_json(out / "target_workspace_manifest.json", frozen)
    comparison = float(np.linalg.norm(references[-1] - references[0], axis=-1).max())
    report["eager_captured_first_step_max_distance_m"] = comparison
    if comparison > 0.00001 or not all(
        v["passed"] for v in report["reproducibility"].values()
    ):
        report["qualification"] = "blocked_reproducibility"
        write_json(out / "report.json", report)
        return
    results = []
    for captured in (False, True):
        if captured and not sim.scene.device.is_cuda:
            report["captured_status"] = "unavailable_on_cpu"
            continue
        sim.cuda_graph = captured
        sim.prepare()
        for candidate in frozen["targets"]:
            for offset in (0.0, -0.005, 0.005):
                t = candidate["vertex"]
                directory = (
                    out / f"{'captured' if captured else 'eager'}_v{t}_x{offset:+.3f}"
                )
                result = attempt(
                    sim, probe, snapshot, counters, t, offset, frozen, directory
                )
                results.append(result)
                write_json(out / "attempt_results.json", results)
                print(
                    "ATTEMPT",
                    result["mode"],
                    t,
                    offset,
                    result["status"],
                    result["failure"],
                    flush=True,
                )
    supported = [
        c["vertex"]
        for c in frozen["targets"]
        if len([r for r in results if r["target"] == c["vertex"]]) == 6
        and all(r["status"] == "passed" for r in results if r["target"] == c["vertex"])
    ]
    report.update(
        results=results,
        supported_targets=supported,
        reusable_target_set_support=len(supported) >= 4,
        capture_count_histogram={
            str(n): sum(r["acquisition_size"] == n for r in results)
            for n in sorted({r["acquisition_size"] for r in results})
        },
    )
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
