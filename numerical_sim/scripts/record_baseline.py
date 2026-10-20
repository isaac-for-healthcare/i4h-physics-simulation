# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze layout-independent products before migration, using three fresh processes.

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
SYSTEM_ARRAYS = ("frame", "omega", "active", "edge_active")


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
        for name in ("region_properties", "rod_material", "rod_controls", "regions", "tet", "rod", "tree", "shell", "pin", "connectors", "centrelines", "skin"):
            record = getattr(body, name, None)
            if record is not None:
                for key, value in vars(record).items():
                    out.update(arrays(value, f"body/{body.name}/{name}/{"anatomical_radii" if name == "tree" and key == "material_radius" else key}"))
    out["scene/gravity"] = scene.gravity.numpy().copy()
    for system in sim.systems:
        if not system.id:
            raise ValueError("baseline systems require an explicit stable logical id")
        for name, value in vars(system).items():
            if name == "appearance":
                # Explicit product move: Surface.visual -> domain-owned Appearance.visual.
                out.update(arrays(value.visual, f"system/{system.id}/visual"))
            elif name not in HANDLE_ARRAYS:
                legacy = {"target_response": "vessel_response", "source_max_delta": "catheter_max_delta",
                          "target_max_delta": "vessel_max_delta"}.get(name, name)
                out.update(arrays(value, f"system/{system.id}/{legacy}"))
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
            "python": platform.python_version(),
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
            name = next(k for k in table.tables if not k.startswith('curvature'))
            table.edit({name: 1.05}, table.membership.names[0], envs=np.array([0], dtype=np.int32))
            edited = True
    if edited:
        data.update({f"regional_edit/{k}": v for k, v in snapshot(sim).items()})
    for k in range(steps):
        apply_actions(ctrl, mid, half, envs, k)
        sim.step()
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


def repeat_in_process(name: str, args, k: int) -> dict[str, np.ndarray]:
    """One run in a fresh process: repeats in one process underestimate the spread of the chaotic cases."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / f"{name}_{k}.npz"
        cmd = [sys.executable, __file__, "--dump", str(path), "--cases", name, "--envs", str(args.envs),
               "--steps", str(args.steps), "--device", args.device]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
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
    p.add_argument("--repeat", type=int, default=3, help="runs per case when recording (noise envelope)")
    p.add_argument("--atol", type=float, default=0.0, help="must remain zero")
    p.add_argument("--int-mismatch", type=float, default=0.0, help="integer mismatch floor (fraction)")
    p.add_argument("--factor", type=float, default=3.0, help="tolerance in multiples of the recorded noise")
    args = p.parse_args(argv)
    if args.repeat != 3 or args.atol != 0 or args.int_mismatch != 0 or args.factor != 3:
        p.error("requires three repetitions, zero tolerance floors and factor 3")
    wp.config.quiet = True
    wp.init()
    if args.dump:
        identity = source_identity()
        np.savez(args.dump, **run_case(args.cases[0], args.envs, args.steps, args.device))
        args.dump.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n")
        return 0
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
            manifest = {"source": identity, "envs": args.envs, "steps": args.steps, "factor": 3,
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
            if identity["software"] != manifest["source"]["software"] or identity["gpu"] != manifest["source"]["gpu"]:
                raise RuntimeError("baseline software/hardware identity changed")
            if args.envs != manifest["envs"] or args.steps != manifest["steps"]:
                raise RuntimeError("baseline workload changed")
            data, imported = repeat_in_process(name, args, 0)
            if imported != identity:
                raise RuntimeError("candidate source identity changed during comparison")
            ref = dict(np.load(args.compare / f"{name}.npz"))
            products = manifest["products"]
            if set(products) != {k for k in ref if not k.startswith("noise/")}:
                raise RuntimeError("archive does not match frozen manifest products")
            for key, product in products.items():
                if (digest(ref[key]) != product["sha256"][0] or ref[key].dtype.str != product["dtype"]
                        or list(ref[key].shape) != product["shape"] or float(ref[f"noise/{key}"]) != product["noise"]):
                    raise RuntimeError(f"archive does not match frozen manifest: {key}")
            report = args.compare / f"{name}.candidate.{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]}.{time.time_ns()}.json"
            report.write_text(json.dumps({"source": identity, "baseline_source": manifest["source"]}, indent=2) + "\n")
            if non_finite({k: v for k, v in ref.items() if not k.startswith("noise/")}):
                raise RuntimeError(f"{name}: the baseline holds non-finite values; record it again")
            failures = compare(ref, data, args.atol, args.int_mismatch, args.factor)
            report.write_text(json.dumps({"source": identity, "baseline_source": manifest["source"],
                                          "failures": failures}, indent=2) + "\n")
            np.savez_compressed(report.with_suffix(".npz"), **data)
            if failures:
                for failure in failures:
                    print(f"  FAIL {failure}")
                failed.append(name)
    if failed:
        print(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
