# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze layout-independent products before migration, using six fresh processes.

Dtype, shape and bytes must match by default. Only products demonstrated to vary
in the frozen repetitions use the original factor (3) and their measured envelope.
No tolerance floor is permitted. Each archive has an immutable JSON manifest with
product hashes, repetition hashes, software, hardware and imported source identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import warp as wp

CASES = ("two_worlds", "chole_tet_med", "aorta_static", "aorta_shell", "tissue_contact", "aorta_centreline")
# Scope v3 retains the explicit physical/history/measurement contract. Never enumerate
# arbitrary system buffers: scratch and broadphase candidates are implementation details.
BASELINE_SCOPE = "physical-history-measurements-v3"
SYSTEM_FIELDS = {
    "Particles": "", "Bounds": "", "Attachments": "", "Connectors": "",
    "Tet": "active edge_active",
    "Surface": "active edge_active support edge_support deleted cut_count cut_start overflow tris tri_tet dirty",
    "Shell": "", "RodDirect": "frame omega", "RodPbd": "frame omega",
    "RodDirectMaterial": "", "RodPbdMaterial": "", "ControlLayout": "",
    "Fluid": "cursor alive_count overflow",
    "FluidContact": "", "StaticContact": "",
    "PartContact": "contact_count overflow_count",
    "RodContainment": "contact_count contact_depth",
    "TetContainment": "contact_count contact_depth",
    "ShellContainment": "contact_count contact_depth",
    "GraphContainment": "contact_count contact_depth overflow ref_tree ref_samples built",
    "Drag": "active baseline",
    "CatheterControl": "actions mount_pose mounted rates root targets impulse moment",
    "InstrumentControl": "actions mount_pose mounted root shaft_q wrist_q jaw_q jaw_closed jaw_angle sphere_q sphere_q_prev held fire_on fired released grasp_pending energized",
    "InstrumentContact": "impulse impulse_count jaw_impulse moment record_overflow",
    "Grasp": "count pending impulse jaw_impulse moment",
    "VolumetricGrasp": "count pending impulse jaw_impulse moment",
    "Hook": "active verts tet bary friction_coord radial impulse moment",
    "Clipping": "count clipped",
    "Thermal": "heat burn coag",
    "Cutting": "cut_on cut_tets",
    "StretchBreaking": "max_stretch",
    "Centreline": "positions active_count",
    "Bleeding": "acc serial sealed cut clipped sources",
    "Emitter": "acc serial spawned",
    "Suction": "removed",
    "Smoke": "acc serial cursor pos_age vel_life alive_count",
    "ContactPaint": "paint painted",
    "ToolForce": "force jaw_load count",
}
SLOT_FIELDS = {
    "Grasp": "prim tet verts t radial prev_axis bary gap active",
    "VolumetricGrasp": "particle sphere rest active",
    "Clipping": "node active",
}


def canonical_slots(fields, *, discard_inactive=True):
    """Sort complete records within each world/tool, preserving multiplicity and dtypes.

    Inactive grasp payload is never read by a hold; it is overwritten on capture.
    Keep clip nodes even after a tear: slots are consumed permanently. Hook entry
    and exit have different roles and are deliberately not interchangeable slots.
    """
    shape = fields["active"].shape
    result = {name: np.zeros_like(value) for name, value in fields.items()}
    if shape[-1] == 0:
        return result
    for group in np.ndindex(shape[:-1]):
        ids = np.arange(shape[-1])
        if discard_inactive:
            ids = ids[fields["active"][group] != 0]
        keys = np.concatenate([value[group].reshape(shape[-1], -1)[ids] for value in fields.values()], axis=1)
        order = np.lexsort(tuple(keys[:, column] for column in reversed(range(keys.shape[1]))))
        for name, value in fields.items():
            result[name][group][:len(ids)] = value[group][ids[order]]
    return result


def controls_and_bounds(sim):
    from i4h_isaaclab.views import action_bounds, controls

    ctrl = controls(sim)
    if not ctrl:
        return [], None, None
    low, high = action_bounds(sim, 0.5)
    return ctrl, (low + high) / 2.0, (high - low) / 2.0


def apply_actions(ctrl, mid, half, envs: int, step: int) -> None:
    if not ctrl:
        return
    cols = mid.shape[0]
    t = step / 60.0
    phase = np.arange(cols, dtype=np.float32)[None, :] * 0.7 + np.arange(envs, dtype=np.float32)[:, None] * 1.3
    u = 0.6 * np.sin(2.0 * np.pi * 0.5 * t + phase)
    physical = (mid[None, :] + half[None, :] * u).astype(np.float32)
    offset = 0
    for s in ctrl:
        width = int(np.prod(s.actions.shape[1:]))
        s.actions.assign(physical[:, offset : offset + width].reshape(s.actions.shape))
        offset += width


def scenario(name: str, steps: int) -> dict:
    """Fixed phase schedule, shared by recording and replay; no data-dependent checkpoint selection."""
    if name == "chole_tet_med":
        return {"kind": "decisive_grasp", "approach_steps": 100, "close_step": 80,
                "engaged_checkpoint": 120, "retract_steps": 30, "retract_fraction": 0.1,
                "hold_steps": steps, "reach": 0.85}
    if name == "aorta_shell":
        return {"kind": "settled_shell", "drive_steps": steps, "settle_steps": 360, "rates": [0.3, 0.0, 0.0]}
    if name == "aorta_centreline":
        return {"kind": "settled_containment", "drive_steps": steps, "settle_steps": 1800,
                "rates": [0.3, 0.0, 0.0], "rebuild_steps": 1, "post_rebuild_steps": 180}
    return {"kind": "sinusoidal", "steps": steps}


def run_scenario(sim, name, ctrl, mid, half, steps):
    plan = scenario(name, steps)
    if plan["kind"] == "decisive_grasp":
        control = next(s for s in ctrl if type(s).__name__ == "InstrumentControl")
        grasp = next(s for s in sim.systems if type(s).__name__ == "Grasp")
        body = sim.scene.body("tissue")
        part = body.parts["gallbladder"]
        target = body.rest.q0[:, part.particle_start:part.particle_start + part.particle_count].mean(axis=1)
        actions = control.actions.numpy()
        tool = control.names.index("grasper")
        initial = actions[:, tool, :3].copy()
        target = initial + plan["reach"] * (target - initial)
        lifted = target + plan["retract_fraction"] * (initial - target)
        # Approach, establish the grasp, retract a little, then hold; all other tools retain their defaults.
        for k in range(plan["engaged_checkpoint"] + plan["retract_steps"] + steps):
            if k < plan["engaged_checkpoint"]:
                actions[:, tool, :3] = initial + (target - initial) * min((k + 1) / plan["approach_steps"], 1.0)
            else:
                actions[:, tool, :3] = target + (lifted - target) * min(
                    (k + 1 - plan["engaged_checkpoint"]) / plan["retract_steps"], 1.0)
            actions[:, tool, 7] = float(k >= plan["close_step"])
            control.actions.assign(actions)
            sim.step()
            if k + 1 >= plan["engaged_checkpoint"] and not (grasp.count.numpy()[:, tool] > 0).all():
                raise RuntimeError(f"{name}: grasp must be engaged throughout the recorded hold (step {k + 1})")
    elif plan["kind"] in ("settled_containment", "settled_shell"):
        control = next(s for s in ctrl if type(s).__name__ == "CatheterControl")
        kind = "GraphContainment" if plan["kind"] == "settled_containment" else "ShellContainment"
        containment = next(s for s in sim.systems if type(s).__name__ == kind)
        control.actions.assign(np.tile(np.asarray(plan["rates"], np.float32), (sim.scene.envs, 1)))
        for _ in range(steps):
            sim.step()
        control.actions.zero_()
        for _ in range(plan["settle_steps"]):
            sim.step()
        if plan["kind"] == "settled_containment":
            # Invalidate on both sources, rebuild normally, then allow a fixed settling period.
            containment.rebuild.fill_(1)
            for _ in range(plan["rebuild_steps"] + plan["post_rebuild_steps"]):
                sim.step()
            if containment.rebuild.numpy().any():
                raise RuntimeError(f"{name}: containment reference was not rebuilt")
        if not (containment.contact_count.numpy() > 0).all():
            raise RuntimeError(f"{name}: settled checkpoint must still exercise containment in every world")
    else:
        for k in range(steps):
            apply_actions(ctrl, mid, half, sim.scene.envs, k)
            sim.step()


# Device object handles are process-local addresses, not numerical products.
HANDLE_ARRAYS = {"bvh_ids", "mesh_ids"}


def arrays(value, prefix):
    """Traverse explicit product records and containers, without following system refs."""
    if isinstance(value, wp.array):
        yield prefix, value.numpy().copy()
    elif isinstance(value, np.ndarray):
        yield prefix, value.copy()
    elif isinstance(value, dict):
        for name, child in value.items():
            yield from arrays(child, f"{prefix}/{name}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from arrays(child, f"{prefix}/{index}")


def snapshot(sim) -> dict[str, np.ndarray]:
    scene, out = sim.scene, {}
    for body in list(scene.bodies) + list(scene.pools):
        for field in scene.main.FIELDS:
            out[f"body/{body.name}/{field}"] = scene.field(field, body).numpy().copy()
    out["scene/gravity"] = scene.gravity.numpy().copy()
    for system in sim.systems:
        if not system.id:
            raise ValueError("baseline systems require an explicit stable logical id")
        kind = type(system).__name__
        if kind not in SYSTEM_FIELDS:
            raise ValueError(f"baseline scope needs an explicit review for {kind}")
        for name in SYSTEM_FIELDS[kind].split():
            out.update(arrays(getattr(system, name), f"system/{system.id}/{name}"))
        if kind == "Surface" and hasattr(system, "appearance"):
            out.update(arrays(system.appearance.visual, f"system/{system.id}/visual"))
        if kind in SLOT_FIELDS:
            fields = {name: getattr(system, name).numpy().copy() for name in SLOT_FIELDS[kind].split()}
            for name, value in canonical_slots(fields, discard_inactive=kind != "Clipping").items():
                out[f"system/{system.id}/slots/{name}"] = value
        if kind == "Hook":
            for name in ("verts", "tet", "bary", "friction_coord", "radial"):
                out[f"system/{system.id}/{name}"][system.active.numpy() == 0] = 0
    return out


def source_identity():
    import i4h_physics

    workspace = pathlib.Path(__file__).resolve().parents[1]
    root = pathlib.Path(i4h_physics.__file__).resolve().parent
    expected = workspace / "physics/src/i4h_physics"
    if root != expected:
        raise RuntimeError(f"wrong implementation imported: {root}; expected {expected}")
    files = {}
    for member in ("physics", "isaaclab", "standalone", "endoluminal", "surgical", "assessment"):
        for path in sorted((workspace / member / "src").rglob("*")):
            if path.suffix in (".py", ".yaml", ".slang"):
                files[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"root": str(workspace), "files": files,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip(),
            "python": platform.python_version(), "baseline_scope": BASELINE_SCOPE,
            "recorder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "software": {n: importlib.metadata.version(n) for n in ("warp-lang", "newton", "numpy", "pyyaml")},
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()}


def digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def run_case(name: str, envs: int, steps: int, device: str) -> dict[str, np.ndarray]:
    from i4h_isaaclab.composition import load_case, read_case

    sim = load_case(read_case(name), device=device, envs=envs)
    ctrl, mid, half = controls_and_bounds(sim)
    sim.step()  # initialize working products before the first reset checkpoint
    sim.reset()
    data = {f"initial/{k}": v for k, v in snapshot(sim).items()}
    edited = False
    for body in sim.scene.bodies:
        table = body.region_properties
        if table is not None:
            parameter = next(k for k in table.tables if not k.startswith('curvature'))
            table.edit({parameter: 1.05}, table.membership.names[0], envs=np.array([0], dtype=np.int32))
            edited = True
    if edited:
        data.update({f"regional_edit/{k}": v for k, v in snapshot(sim).items()})
    run_scenario(sim, name, ctrl, mid, half, steps)
    wp.synchronize()
    data.update({f"stepped/{k}": v for k, v in snapshot(sim).items()})
    sim.reset([0])
    wp.synchronize()
    data.update({f"masked_reset/{k}": v for k, v in snapshot(sim).items()})
    bad = non_finite(data)
    if bad:
        raise RuntimeError(f"{name}: non-finite values in {bad}; a baseline or a comparison run must be finite")
    return data


def non_finite(data: dict) -> list[str]:
    """Keys of float arrays holding a NaN or an infinity."""
    return [k for k, a in data.items() if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all()]


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference for floats, fraction of differing entries for ints."""
    if not a.size:
        return 0.0
    if np.issubdtype(a.dtype, np.integer):
        return float(np.mean(a != b))
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def noise_envelope(runs: list[dict]) -> dict[str, np.ndarray]:
    return {f"noise/{key}": np.float64(max((difference(r[key], s[key]) for i, r in enumerate(runs) for s in runs[i + 1 :]),
                                           default=0.0))
            for key in runs[0]}


def compare(ref: dict, cur: dict, atol: float, int_mismatch: float, factor: float) -> list[str]:
    if atol != 0 or int_mismatch != 0 or factor != 3:
        raise ValueError("frozen baseline requires zero floors and factor 3")
    failures = []
    noise = {k[len("noise/"):]: float(v) for k, v in ref.items() if k.startswith("noise/")}
    ref = {k: v for k, v in ref.items() if not k.startswith("noise/")}
    for key in sorted(set(ref) | set(cur)):
        if key not in cur or key not in ref:
            failures.append(f"{key}: {'missing now' if key not in cur else 'new key'}")
            continue
        a, b = ref[key], cur[key]
        if a.dtype != b.dtype:
            failures.append(f"{key}: dtype {a.dtype} -> {b.dtype}")
            continue
        if a.shape != b.shape:
            failures.append(f"{key}: shape {a.shape} -> {b.shape}")
            continue
        if non_finite({key: a}) or non_finite({key: b}):
            failures.append(f"{key}: non-finite")
            continue
        d, n = difference(a, b), noise.get(key, 0.0)
        if n == 0:
            if a.tobytes() != b.tobytes():
                failures.append(f"{key}: bytes changed")
            continue
        if np.issubdtype(a.dtype, np.integer):
            tol = max(int_mismatch, factor * n)
            bad = d > tol
            print(f"  {'FAIL' if bad else 'ok':4s} {key}: {d:.2%} entries differ (tol {tol:.2%}, noise {n:.2%})")
        else:
            tol = max(atol, factor * n)
            bad = d > tol or not (np.isfinite(a).all() and np.isfinite(b).all())  # never accept NaN or inf, matching or not
            print(f"  {'FAIL' if bad else 'ok':4s} {key}: max |diff| {d:.3e} (tol {tol:.1e}, noise {n:.1e})")
        if bad:
            failures.append(f"{key}: {d:.3e}")
    return failures


def comparison_policy(name, manifest):
    """Only explicitly accepted chole references use three complete replays."""
    policy = manifest.get("comparison_policy")
    if policy is None:
        return {"kind": "single", "replays": 1, "required_passes": 1,
                "fresh_processes": True, "retain_all_trials": True}
    expected = {"kind": "majority-of-three", "replays": 3, "required_passes": 2,
                "fresh_processes": True, "retain_all_trials": True}
    disposition = manifest.get("reference_disposition", {})
    if (name != "chole_tet_med" or policy != expected or disposition.get("status") != "approved"
            or disposition.get("capacity") != 256):
        raise ValueError("three-replay policy requires the approved capacity-256 chole reference")
    return dict(expected)


def summarize_replays(policy, trials):
    """Count whole passing replays, never a majority independently per product."""
    if len(trials) != policy["replays"]:
        raise ValueError("all scheduled replays must be retained before deciding")
    passed = sum(trial["passed"] is True for trial in trials)
    valid = all(trial.get("valid", True) for trial in trials)
    return {"policy": policy, "trials": trials, "passed_replays": passed,
            "valid": valid, "passed": valid and passed >= policy["required_passes"]}


def repeat_in_process(name: str, args, k: int, *, retained_path=None):
    """One run in a fresh process: repeats in one process underestimate the spread of the chaotic cases."""
    with tempfile.TemporaryDirectory() as tmp:
        path = retained_path or pathlib.Path(tmp) / f"{name}_{k}.npz"
        cmd = [sys.executable, __file__, "--dump", str(path), "--cases", name, "--envs", str(args.envs),
               "--steps", str(args.steps), "--device", args.device]
        if retained_path is None:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        else:
            with path.with_suffix(".log").open("x") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        metadata = json.loads(path.with_suffix(".json").read_text())
        return dict(np.load(path)), metadata


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path, help="record into this directory")
    mode.add_argument("--compare", type=pathlib.Path, help="compare against this directory")
    mode.add_argument("--dump", type=pathlib.Path, help=argparse.SUPPRESS)  # one run of one case (a repeat)
    p.add_argument("--cases", nargs="*", default=list(CASES))
    p.add_argument("--envs", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeat", type=int, default=6, help="runs per case when recording (noise envelope)")
    p.add_argument("--atol", type=float, default=0.0, help="must remain zero")
    p.add_argument("--int-mismatch", type=float, default=0.0, help="integer mismatch floor (fraction)")
    p.add_argument("--factor", type=float, default=3.0, help="tolerance in multiples of the recorded noise")
    p.add_argument("--results", type=pathlib.Path, help="fresh comparison evidence directory (leaves reference untouched)")
    args = p.parse_args(argv)
    if args.repeat != 6 or args.atol != 0 or args.int_mismatch != 0 or args.factor != 3:
        p.error("requires six repetitions, zero tolerance floors and factor 3")
    wp.config.quiet = True
    wp.init()
    if args.dump:
        identity = source_identity()
        np.savez(args.dump, **run_case(args.cases[0], args.envs, args.steps, args.device))
        args.dump.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n")
        return 0
    if args.results:
        if not args.compare:
            p.error("--results requires --compare")
        args.results.mkdir(parents=True, exist_ok=False)
    failed = []
    for name in args.cases:
        print(f"{name}: envs={args.envs} steps={args.steps}", flush=True)
        if args.out:
            if (args.out / f"{name}.npz").exists():
                raise FileExistsError("frozen baselines cannot be overwritten")
            repeats = [repeat_in_process(name, args, k) for k in range(args.repeat)]
            runs = [r for r, identity in repeats]
            identity = source_identity()
            if any(meta != identity for _, meta in repeats):
                raise RuntimeError("fresh processes imported different source/software/hardware")
            args.out.mkdir(parents=True, exist_ok=True)
            for run in runs:
                errors = compare(runs[0], run, 0, 0, 3)
                structural = [e for e in errors if "shape" in e or "dtype" in e or "key" in e or "non-finite" in e]
                if structural:
                    raise RuntimeError(structural)
            envelope = noise_envelope(runs)
            manifest = {"source": identity, "envs": args.envs, "steps": args.steps, "scenario": scenario(name, args.steps), "repeat_count": args.repeat, "factor": 3,
                        "products": {k: {"shape": list(v.shape), "dtype": v.dtype.str,
                                         "sha256": [digest(r[k]) for r in runs],
                                         "noise": float(envelope[f"noise/{k}"])} for k, v in runs[0].items()}}
            (args.out / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for i, run in enumerate(runs):
                np.savez_compressed(args.out / f"{name}.repeat{i}.npz", **run)
            np.savez_compressed(args.out / f"{name}.npz", **runs[0], **envelope)
            print(f"  recorded {len(runs[0])} arrays over {len(runs)} runs")
        else:
            manifest = json.loads((args.compare / f"{name}.json").read_text())
            identity = source_identity()
            if manifest["source"].get("baseline_scope") != BASELINE_SCOPE:
                raise RuntimeError("baseline scope changed; record a new archive without replacing the original")
            if identity["software"] != manifest["source"]["software"] or identity["gpu"] != manifest["source"]["gpu"]:
                raise RuntimeError("baseline software/hardware identity changed")
            if manifest.get("scenario") != scenario(name, args.steps) or manifest.get("repeat_count") != 6:
                raise RuntimeError("baseline scenario or repetition contract changed")
            if manifest.get("factor") != 3:
                raise RuntimeError("baseline factor changed")
            if args.envs != manifest["envs"] or args.steps != manifest["steps"]:
                raise RuntimeError("baseline workload changed")
            policy = comparison_policy(name, manifest)
            ref = dict(np.load(args.compare / f"{name}.npz"))
            products = manifest["products"]
            if set(products) != {k for k in ref if not k.startswith("noise/")}:
                raise RuntimeError("archive does not match frozen manifest products")
            for key, product in products.items():
                if (digest(ref[key]) != product["sha256"][0] or ref[key].dtype.str != product["dtype"]
                        or list(ref[key].shape) != product["shape"] or float(ref[f"noise/{key}"]) != product["noise"]):
                    raise RuntimeError(f"archive does not match frozen manifest: {key}")
            report = (args.results or args.compare) / f"{name}.candidate.{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]}.{time.time_ns()}.json"
            report.write_text(json.dumps({"source": identity, "baseline_source": manifest["source"]}, indent=2) + "\n")
            if non_finite({k: v for k, v in ref.items() if not k.startswith("noise/")}):
                raise RuntimeError(f"{name}: the baseline holds non-finite values; record it again")
            trials = []
            for k in range(policy["replays"]):
                path = report.with_name(f"{report.stem}.trial{k}.npz")
                errors = []
                valid = True
                try:
                    data, imported = repeat_in_process(name, args, k, retained_path=path)
                    if imported != identity:
                        raise RuntimeError("candidate source identity changed during comparison")
                    errors = compare(ref, data, args.atol, args.int_mismatch, args.factor)
                except (subprocess.CalledProcessError, RuntimeError) as error:
                    errors = [str(error)]
                    valid = False
                trial = {"trial": k, "passed": not errors, "valid": valid,
                         "failures": errors, "products": str(path)}
                trials.append(trial)
                path.with_suffix(".comparison.json").write_text(json.dumps(trial, indent=2) + "\n")
                print(f"  replay {k}: {'FAIL' if errors else 'PASS'}", flush=True)
                for error in errors:
                    print(f"  FAIL {error}")
            summary = summarize_replays(policy, trials)
            report.write_text(json.dumps({"source": identity, "baseline_source": manifest["source"],
                                          **summary}, indent=2) + "\n")
            if not summary["passed"]:
                failed.append(name)
    if failed:
        print(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
