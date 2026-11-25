# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage A probes, run with an isolated interpreter; never install or change pins.

Each probe runs in a fresh process. A failed probe, timeout, or abnormal process
exit fails qualification, even if it printed a success message before exiting.
Reports and full logs go into a new directory, never over an earlier result.
This is dependency qualification, not qualification of the planned i4h adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REVISIONS = {
    "newton": "91fb115d685a96311b2f1bac5527754ee4e9c21b",
    "main": "dc01ec356052f2acd04246752a92a55162f6cd68",
    "3.0": "e53cd288727a5ca3dae7187d1d415b4d32b67e11",
}
VERSIONS = {"newton": "1.8.0rc1", "warp-lang": "1.17.0", "mujoco": "3.17.0", "mujoco-warp": "3.17.0"}


def source_identity(path: Path, revision: str) -> dict:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if actual != revision:
        raise ValueError(f"{path}: expected {revision}, found {actual}")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True)
    if dirty:
        raise ValueError(f"qualification source has tracked changes: {path}\n{dirty}")
    return {"path": str(path.resolve()), "revision": actual}


def identity(args) -> dict:
    packages = {}
    for name, module in (("newton", "newton"), ("warp-lang", "warp"), ("mujoco", "mujoco"),
                         ("mujoco-warp", "mujoco_warp"), ("numpy", "numpy")):
        dist = metadata.distribution(name)
        version = dist.version
        if name in VERSIONS and version != VERSIONS[name]:
            raise ValueError(f"{name}: expected {VERSIONS[name]}, found {version}")
        packages[name] = {"version": version, "import": importlib.import_module(module).__file__,
                          "direct_url": json.loads(dist.read_text("direct_url.json") or "null")}
    sources = {"newton": source_identity(args.newton_source, REVISIONS["newton"])}
    # Refuse an unrelated wheel which happens to have the same version string.
    direct = packages["newton"]["direct_url"]
    if direct is None or direct["url"] != args.newton_source.resolve().as_uri():
        raise ValueError("install Newton from --newton-source in the isolated environment")
    source_hash = hashlib.sha256()
    package_root = Path(packages["newton"]["import"]).parent
    source_root = args.newton_source / "newton"
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        digest = hashlib.sha256(source.read_bytes()).digest()
        installed = package_root / relative
        if not installed.is_file() or hashlib.sha256(installed.read_bytes()).digest() != digest:
            raise ValueError(f"installed Newton source differs from pinned checkout: {relative}")
        source_hash.update(str(relative).encode() + digest)
    packages["newton"]["source_sha256"] = source_hash.hexdigest()
    if args.stack != "core":
        sources["lab"] = source_identity(args.lab_source, REVISIONS[args.stack])
        for name in ("isaaclab", "isaaclab_newton"):
            module = importlib.import_module(name)
            imported = Path(module.__file__).resolve()
            if not imported.is_relative_to(args.lab_source.resolve()):
                raise ValueError(f"wrong {name} imported: {imported}")
            packages[name] = {"version": metadata.version(name), "import": str(imported)}
        import torch
        packages["torch"] = {"version": torch.__version__, "import": torch.__file__, "cuda": torch.version.cuda}
    import warp as wp
    wp.init()
    dev = wp.get_device(args.device)
    return {"python": sys.executable, "python_version": sys.version, "sys_path": sys.path,
            "sources": sources, "packages": packages, "device": str(dev), "device_name": dev.name,
            "installed_packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
            "capture": dev.is_cuda, "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "I4H_DATA")},
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                             "--format=csv,noheader"], text=True).strip() if dev.is_cuda else None}


def particles(device: str) -> None:
    import newton
    import numpy as np
    import warp as wp
    from newton.solvers import SolverXPBD

    builder = newton.ModelBuilder(gravity=(0., 0., -9.81))
    for env in range(2):
        builder.begin_world()
        for mass in (1., 0.):
            builder.add_particle(pos=wp.vec3(float(env), mass, 1.), vel=wp.vec3(0.), mass=mass, radius=.01)
        builder.end_world()
    model = builder.finalize(device=device)
    state, out, independent = model.state(), model.state(), model.state()
    control, other_control = model.control(), model.control()
    assert state.particle_q.ptr != independent.particle_q.ptr
    assert control is not other_control
    assert model.tet_count == model.tri_count == model.edge_count == 0
    solver = SolverXPBD(model, iterations=2)
    q0 = state.particle_q.numpy().copy()

    def step():
        state.clear_forces()
        solver.step(state, out, control, None, .01)
        wp.copy(state.particle_q, out.particle_q)
        wp.copy(state.particle_qd, out.particle_qd)

    step()  # compile before capture
    state.particle_q.assign(q0)
    state.particle_qd.zero_()
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            step()
        for _ in range(4):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(4):
            step()
    q = state.particle_q.numpy()
    expected = q0.copy()
    expected[::2, 2] -= 9.81 * .01**2 * (1 + 2 + 3 + 4)
    np.testing.assert_allclose(q, expected, atol=2.e-6, rtol=0)
    np.testing.assert_array_equal(independent.particle_q.numpy(), q0)
    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state, world_mask=mask, flags=0)
    np.testing.assert_array_equal(state.particle_q.numpy(), q)


def rigid(device: str) -> None:
    import newton
    import numpy as np
    import warp as wp
    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    # MuJoCo CPU simulates only its template world; masked history and capture
    # are qualified on the two-world MuJoCo-Warp path below.
    for env in range(2 if wp.get_device(device).is_cuda else 1):
        builder.begin_world()
        body = builder.add_link(xform=wp.transform(wp.vec3(float(env), 0., 1.), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=.05)
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        builder.end_world()
    model = builder.finalize(device=device)
    state = model.state()
    control = model.control()
    solver = SolverMuJoCo(model, njmax=32, nconmax=16, separate_worlds=True,
                          use_mujoco_cpu=not model.device.is_cuda)
    q0 = state.body_q.numpy().copy()
    state.clear_forces()
    solver.step(state, state, control, None, .01)
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            state.clear_forces()
            solver.step(state, state, control, None, .01)
        wp.capture_launch(capture.graph)
    q = state.body_q.numpy()
    assert np.isfinite(q).all() and np.all(q[:, 2] < q0[:, 2])
    if model.device.is_cuda:
        warm = np.full(solver.mjw_data.qacc_warmstart.shape, 17., dtype=np.float32)
        solver.mjw_data.qacc_warmstart.assign(warm)
        mask = wp.array([True, False, False], dtype=wp.bool, device=device)
        joint_q = state.joint_q.numpy().copy()
        solver.reset(state, world_mask=mask, flags=0)
        actual = solver.mjw_data.qacc_warmstart.numpy()
        np.testing.assert_array_equal(actual[0], 0.)
        np.testing.assert_array_equal(actual[1], warm[1])
        np.testing.assert_array_equal(state.joint_q.numpy(), joint_q)
        empty = wp.zeros(3, dtype=wp.bool, device=device)
        solver.reset(state, world_mask=empty, flags=0)
        np.testing.assert_array_equal(solver.mjw_data.qacc_warmstart.numpy(), actual)


def host(device: str, observation: bool) -> None:
    import numpy as np
    import warp as wp
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg, XPBDSolverCfg

    cfg = SimulationCfg(device=device, physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=device != "cpu"))
    with launch_simulation(cfg, {"headless": True, "device": device}):
        with build_simulation_context(sim_cfg=cfg) as sim:
            builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics))
            expected = []
            for env in range(2):
                builder.begin_world()
                start = builder.particle_count
                rotation = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), env * .7)
                xform = wp.transform(wp.vec3(env * 3., 0., 1.), rotation)
                points = [wp.transform_point(xform, wp.vec3(x, y, 0.)) for x, y in ((0., 0.), (.1, 0.), (0., .1))]
                expected.append(np.asarray(points, dtype=np.float32))
                for point in points:
                    builder.add_particle(pos=point, vel=wp.vec3(0.), mass=1., radius=.01)
                if observation:
                    # The native observation probe only: no surgical elements are
                    # authored into Newton's tet or triangle domains.
                    builder._record_volume_deformable_object(
                        f"/World/env_{env}/Tissue", (start, builder.particle_count), (0, 0))
                builder.end_world()
            if observation:
                import isaaclab.sim as sim_utils
                from isaaclab.assets import DeformableObject, DeformableObjectCfg
                for env in range(2):
                    sim_utils.create_prim(f"/World/env_{env}/Tissue", "Xform")
                asset = DeformableObject(DeformableObjectCfg(prim_path="/World/env_.*/Tissue", spawn=None))
            sim.reset()
            backend = sim.backend_manager.backend if hasattr(sim, "backend_manager") else sim.physics_manager.backend
            assert backend.model.world_count == 2
            assert backend.model.tet_count == backend.model.tri_count == backend.model.edge_count == 0
            if observation:
                assert asset.is_initialized
                values = asset.data.nodal_pos_w.warp.numpy()
                np.testing.assert_allclose(values, np.asarray(expected), atol=1.e-7, rtol=0)
            initial = backend.state_0.particle_q.numpy().copy()
            for _ in range(3):
                sim.step(render=False)
            stepped = backend.state_0.particle_q.numpy()
            assert np.isfinite(stepped).all() and np.all(stepped[:, 2] < initial[:, 2])
            if observation:
                asset.update(.01)
                np.testing.assert_array_equal(asset.data.nodal_pos_w.warp.numpy(), stepped.reshape(2, 3, 3))
                asset.reset([0])  # Lab's asset reset is a no-op; it must preserve authored state.
                np.testing.assert_array_equal(backend.state_0.particle_q.numpy(), stepped)
            # Re-finalize the retained builder; no second authoring pass.
            sim.reset()
            backend = sim.backend_manager.backend if hasattr(sim, "backend_manager") else sim.physics_manager.backend
            assert backend.model.particle_count == 6
            np.testing.assert_array_equal(backend.state_0.particle_q.numpy(), initial)


def run_probe(name: str, args) -> None:
    if name == "identity":
        result = identity(args)
        with (args.out / "identity.json").open("x") as output:
            json.dump(result, output, indent=2)
        print(json.dumps(result))
    elif name == "particles":
        particles(args.device)
    elif name == "rigid":
        rigid(args.device)
    else:
        host(args.device, name == "observations")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=("core", "main", "3.0"), required=True)
    parser.add_argument("--newton-source", type=Path, required=True)
    parser.add_argument("--lab-source", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300, help="timeout of each isolated probe in seconds")
    parser.add_argument("--probe", choices=("identity", "particles", "rigid", "host", "observations"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stack != "core" and args.lab_source is None:
        parser.error("--lab-source is required for a Lab stack")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.probe:
        run_probe(args.probe, args)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    # Snapshot the probe so later edits cannot alter a running qualification or
    # make its recorded script hash impossible to reproduce.
    script = Path(__file__).read_bytes()
    snapshot = args.out / "probe.py"
    snapshot.write_bytes(script)
    command = [sys.executable, "-u", str(snapshot.resolve()), "--stack", args.stack,
               "--newton-source", str(args.newton_source.resolve()), "--device", args.device,
               "--out", str(args.out.resolve())]
    if args.lab_source is not None:
        command.extend(["--lab-source", str(args.lab_source.resolve())])
    probes = ["identity", "particles", "rigid"]
    if args.stack != "core":
        probes.extend(["host", "observations"])
    report = {"stack": args.stack, "device": args.device, "command": command,
              "script_sha256": hashlib.sha256(script).hexdigest(), "checks": {}}
    for probe in probes:
        if probe != "identity" and report["checks"]["identity"]["status"] != "passed":
            report["checks"][probe] = {"status": "not_run", "reason": "stack identity failed"}
            continue
        started = time.perf_counter()
        with (args.out / f"{probe}.log").open("x") as log:
            try:
                result = subprocess.run(command + ["--probe", probe], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout, check=False)
                check = {"status": "passed" if result.returncode == 0 else "failed", "exit_code": result.returncode}
            except subprocess.TimeoutExpired:
                check = {"status": "failed", "reason": "timeout"}
        check["seconds"] = time.perf_counter() - started
        report["checks"][probe] = check
        print(f"{probe}: {check['status']}", flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(any(check["status"] != "passed" for check in report["checks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
