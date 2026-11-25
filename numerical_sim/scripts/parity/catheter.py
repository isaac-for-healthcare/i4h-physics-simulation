# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catheter parity (F5): the workflow's catheter rod against i4h_endoluminal's, same physical rod, same commands.

Reference: ``catheter_vasculature_solver`` at physics repo ``5005619`` on Warp 1.17,
the current workflow pin. Each archive records the loaded package's exact revision,
working-tree changes, effective configuration, commands and full sampled state.
Historical e4bed31 archives cannot qualify the current migration.

    $REF/bin/python scripts/parity/catheter.py run --engine reference --out parity/reference.npz
    $PY scripts/parity/catheter.py run --engine ours --out parity/ours.npz
    $PY scripts/parity/catheter.py compare parity/reference.npz parity/ours.npz --report parity/report.json

Each run also repeats every scenario at half the substep (``<scenario>@half``) for the refinement rows. ``compare``
evaluates physics/component-contracts.md section 4, prints one PASS/FAIL row per bound plus informational rows, and exits
non-zero on a FAIL; it decides nothing by itself.

Scenarios, all free-space (no vessel), the rod straight along +x from (0, 0, 0.5):
- ``sag``: a cantilever under gravity, 5 s; tip trajectory and the steady root reaction.
- ``tip_bend``: the tip command ramps to a 0.8 rad polyline turn over 1 s and holds 1 s, no gravity; curvature.
- ``twist``: a 0.6 rad tip bend, then the root turns at 1.5 rad/s for 1 s and holds; the tip's angle about the axis.
- ``insertion``: the root advances at 0.05 m/s for 2 s under gravity; tip trajectory.
Both roots ride a straight rail along +x: the reference through ``set_root_pose_gpu`` (as the workflow's route-railed
introducer drives it), ours through ``catheter_control``'s track.
- ``tube_straight``, ``tube_bend``: the root advances 0.2 m at 0.05 m/s into a rigid 2 mm lumen, straight or turning
  90 degrees on a 5 cm radius, no gravity. Both sides contain the wire with the same one-sided centreline capsule test
  (the reference's ``CenterlineVesselRuntime``, ``two_way`` off, no interior pull; ours ``graph_containment`` on a
  pinned centreline tree, no spline) alternated with the rod solve 32 times per substep.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Rod:
    """The workflow's CatheterRodSpec / XPBDRodSolverCfg defaults (SI)."""

    length: float = 0.4
    segments: int = 120
    radius: float = 5.0e-4
    density: float = 7800.0
    young: float = 1.0e9
    poisson: float = 0.3
    bend: float = 0.1  # multiplier on E*I
    twist: float = 0.4  # multiplier on G*J
    damping: float = 0.01  # per substep, linear and angular: v *= 1 - damping
    tip_length: float = 0.025
    dt: float = 1.0 / 120.0
    substeps: int = 1

    @property
    def seg(self) -> float:
        return self.length / self.segments

    @property
    def node_mass(self) -> float:
        return self.density * math.pi * self.radius**2 * self.seg

    @property
    def tip_edges(self) -> int:
        return int(math.ceil(self.tip_length / self.seg - 1e-9))

    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))


GRAVITY = (0.0, 0.0, -9.81)
START = (0.0, 0.0, 0.5)
STRIDE = 5  # record every STRIDE physics steps
RATE = 4.0  # ours: catheter_control.rates, the largest command per second (m/s or rad/s)
INERTIA = "physical"  # ours: rod_direct.inertia (--inertia)
TIP_MAPPING = "polyline"  # ours: catheter_control.tip_mapping (--tip-mapping)


REFERENCE_REVISION = "50056196775fc0ed8cca03bf16144fe71d183ca7"
SCENARIOS = ("sag", "insertion", "tip_bend", "twist", "tube_straight", "tube_bend")
TRACE_FIELDS = ("positions", "orientations", "velocities", "angular_velocities", "reaction",
                "commands", "time", "vessel_displacement")


def source_identity(engine):
    """Identify the imported code, rather than the checkout invoking the script."""
    package = importlib.import_module("catheter_vasculature_solver" if engine == "reference" else "i4h_endoluminal")
    path = Path(package.__file__).resolve()
    root = next((p for p in path.parents if (p / ".git").exists()), None)
    if root is None:
        raise ValueError(f"parity requires a versioned source checkout: {path}")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    revision = git("rev-parse", "HEAD")
    if engine == "reference" and revision != REFERENCE_REVISION:
        raise ValueError(f"reference must be {REFERENCE_REVISION}, imported {revision} at {path}")
    import warp as wp
    return {"revision": revision, "module": str(path), "root": str(root),
            "status": git("status", "--porcelain", "--untracked-files=no"), "warp": wp.__version__,
            "numpy": np.__version__, "torch": importlib.metadata.version("torch"),
            "python": sys.version, "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def sampled_trace():
    return {name: [] for name in TRACE_FIELDS}


def finish_trace(trace):
    return {name: np.asarray(values) for name, values in trace.items()}


def scenarios(rod: Rod):
    """name -> (gravity, steps, command(step) -> (push m/s, rotate rad/s, tip polyline turn rad))."""
    hz = round(1.0 / rod.dt)

    def ramp(k, start, length, value):
        return value * min(max((k - start) / length, 0.0), 1.0)

    return {
        "tube_straight": ((0.0, 0.0, 0.0), 4 * hz, lambda k: (0.05, 0.0, 0.0)),
        "tube_bend": ((0.0, 0.0, 0.0), 4 * hz, lambda k: (0.05, 0.0, 0.0)),
        "sag": (GRAVITY, 5 * hz, lambda k: (0.0, 0.0, 0.0)),
        "tip_bend": ((0.0, 0.0, 0.0), 2 * hz, lambda k: (0.0, 0.0, ramp(k, 0, hz, 0.8))),
        "twist": ((0.0, 0.0, 0.0), 3 * hz,
                  lambda k: (0.0, 1.5 if hz // 2 <= k < hz // 2 + hz else 0.0, ramp(k, 0, hz // 2, 0.6))),
        "insertion": (GRAVITY, 2 * hz, lambda k: (0.05, 0.0, 0.0)),
    }


TUBE_RADIUS = 0.002  # lumen radius (m), the wire's 0.5 mm inside
COUPLING = 32  # contact/solve alternations per substep (the workflow's contact_coupling_iterations; ours: iterations)


def tube(name):
    """The lumen centreline of a ``tube_*`` scenario (None otherwise), from behind the root along +x, 5 mm samples:
    straight to x 0.85, or straight to x 0.45, a 90 degree turn to +y on a 5 cm radius and 0.3 m along +y. The rod
    starts inside the straight part and is inserted 0.2 m, so in the bend its tip goes round the corner."""
    if not name.startswith("tube_"):
        return None
    x, y, z = START
    if name == "tube_straight":
        xs = np.arange(-0.05, 0.85 + 1e-9, 0.005)
        return np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], axis=1)
    straight = np.arange(-0.05, 0.45 + 1e-9, 0.005)
    angles = np.linspace(0.0, math.pi / 2, 17)[1:]
    arc = np.stack([0.45 + 0.05 * np.sin(angles), 0.05 - 0.05 * np.cos(angles)], axis=1)
    ys = np.arange(0.055, 0.35 + 1e-9, 0.005)
    xy = np.concatenate([np.stack([straight, np.zeros_like(straight)], 1), arc, np.stack([np.full_like(ys, 0.5), ys], 1)])
    return np.concatenate([xy + np.float64([x, y]), np.full((len(xy), 1), z)], axis=1)


def quat_mul(a, b):
    """xyzw quaternion product a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


# --- engines --------------------------------------------------------------------------------------------------------

def run_reference(rod: Rod, gravity, steps, command, physical_inertia=True, lumen=None,
                  initial_positions=None, initial_frames=None, root_pose=None, vessel_settings=None, sample_stride=STRIDE):
    import torch  # noqa: F401  (the solver's host arrays)
    import warp as wp
    from catheter_vasculature_solver import (
        CathRodSolver,
        RodConfig,
        RodGeometryConfig,
        RodMaterialConfig,
        RodSolverConfig,
    )

    cfg = RodConfig(material=RodMaterialConfig(young_modulus=rod.young, shear_modulus=None, poisson_ratio=rod.poisson,
                                               density=rod.density, damping=rod.damping, bend_stiffness=rod.bend,
                                               twist_stiffness=rod.twist),
                    geometry=RodGeometryConfig(num_segments=rod.segments, segment_length=rod.seg, radius=rod.radius),
                    solver=RodSolverConfig(dt=rod.dt, num_substeps=rod.substeps, gravity=tuple(gravity)), device="cuda:0")
    vessel = None if lumen is None else reference_lumen(lumen, rod, vessel_settings)
    wall_rest = None if vessel is None else vessel.positions_per_env[0].copy()
    solver = CathRodSolver(cfg, num_envs=1, floor_z=None, initial_height=START[2], collision_mesh=None,
                           track_start=np.float32(START), track_dir=np.float32([1.0, 0.0, 0.0]), track_length=rod.length,
                           tip_num_edges=rod.tip_edges, particle_radius=rod.radius, segment_length=rod.seg,
                           collision_enabled=False, track_enabled=False, containment_cleanup_iterations=128,
                           containment_cleanup_rounds=8, contact_coupling_iterations=COUPLING,
                           centerline_runtime=vessel)
    solver.set_cuda_graph_enabled(False)
    if initial_positions is not None:
        ws = solver._bws if solver._bws is not None else solver._ws
        for name in ("positions", "predicted_positions", "rest_positions"):
            getattr(ws, name).assign(np.float32(initial_positions))
        for name in ("orientations", "predicted_orientations", "prev_orientations", "rest_orientations"):
            getattr(ws, name).assign(np.float32(initial_frames))
    if physical_inertia:  # newton_catheter_physics._apply_physical_rotational_inertia
        m = rod.node_mass
        inv = 1.0 / (m * (3.0 * rod.radius**2 + rod.seg**2) / 12.0)
        polar = 2.0 / (m * rod.radius**2)
        ws = solver._bws if solver._bws is not None else solver._ws
        if isinstance(ws.inv_inertia_local_diag, wp.array):
            ws.inv_inertia_local_diag.fill_(wp.vec3(inv, inv, polar))
        else:
            ws.inv_inertia_local_diag = wp.vec3(inv, inv, polar)
    # The root rides a straight rail along +x (the workflow's RouteRailedIntroducer on a straight route): its pose is
    # the rail point at the inserted arc, its rest frame turned about the rail by the twist.
    base = solver.orientations.detach().cpu().numpy().reshape(-1, 4)[0].astype(np.float64)
    arc = twist = 0.0
    trace = sampled_trace()
    started = time.perf_counter()
    for k in range(steps):
        push, rotate, tip = command(k)
        arc += push * rod.dt
        twist += rotate * rod.dt
        turn = np.array([math.sin(twist / 2.0), 0.0, 0.0, math.cos(twist / 2.0)])
        if root_pose is None:
            position, orientation = np.float32([[START[0] + arc, START[1], START[2]]]), np.float32([quat_mul(turn, base)])
        else:
            position, orientation = root_pose(k, solver.position_array.numpy(), solver.orientations.cpu().numpy())
        solver.set_root_pose_gpu(position, orientation)
        solver.set_tip_bend(tip)
        solver.step(rod.dt)
        if k % sample_stride == sample_stride - 1:
            workspace = solver._bws if solver._bws is not None else solver._ws
            trace["positions"].append(solver.position_array.numpy().reshape(-1, 3).copy())
            trace["orientations"].append(workspace.orientations.numpy().reshape(-1, 4).copy())
            trace["velocities"].append(solver.velocity_array.numpy().reshape(-1, 3).copy())
            trace["angular_velocities"].append(workspace.angular_velocities.numpy().reshape(-1, 3).copy())
            trace["reaction"].append(solver.proximal_wrench().numpy().reshape(-1, 6)[0].copy())
            trace["commands"].append([push, rotate, tip])
            trace["time"].append((k + 1) * rod.dt)
            trace["vessel_displacement"].append(np.empty((0, 3)) if lumen is None else
                                                vessel.positions_per_env[0] - wall_rest)
    result = finish_trace(trace)
    result["elapsed_s"] = time.perf_counter() - started
    result["case"] = json.dumps(asdict(cfg))
    return result


def reference_lumen(points, rod: Rod, settings=None):
    """A rigid, one-sided centreline lumen (the workflow's vessel without wall response or interior pull)."""
    from catheter_vasculature_solver.vessel_deformation import (
        CenterlineData,
        CenterlineDynamicsParams,
        CenterlineVesselRuntime,
        build_centerline_tree,
    )

    if isinstance(points, dict):
        edges, points = np.asarray(points["edges"], np.int32), np.asarray(points["points"], np.float32)
    else:
        edges = np.column_stack((np.arange(len(points) - 1), np.arange(1, len(points))))
    n = len(edges)
    radii = np.full(n, TUBE_RADIUS)
    tree = build_centerline_tree(CenterlineData(points[edges[:, 0]], points[edges[:, 1]], np.zeros(n, np.int32), radii, radii, radii, radii))
    settings = dict(settings or {})
    params = CenterlineDynamicsParams(**settings.pop("dynamics", {"endpoints_locked": True}))
    contact = {"two_way": False, "max_distance": 0.05, "interior_deadband": 1.0, "interior_stiffness": 0.0,
               "catheter_max_delta": 0.005, "collision_iterations": 2, **settings}
    return CenterlineVesselRuntime.from_tree(tree, device="cuda:0", params=params, catheter_radius=rod.radius, **contact)


def add_lumen(scene, name, points, radius, edges=None, dynamic=False):
    """Ours: the same lumen as a fully pinned centreline tree with an open root."""
    from i4h_endoluminal.loaders.centreline import add_centreline_tree

    points = np.float32(points)
    edges = (np.stack([np.arange(len(points) - 1), np.arange(1, len(points))], axis=1) if edges is None
             else np.asarray(edges, np.int32))
    pinned = np.flatnonzero(np.bincount(edges.ravel(), minlength=len(points)) == 1) if dynamic else range(len(points))
    return add_centreline_tree(scene, name, points, edges, float(radius), 0, pinned_nodes=pinned,
                               pinned_segments=(0,) if dynamic else (), mass=1.0, inertia=1.0, open_root=True,
                               precise_rest_frames=dynamic)


def run_ours(rod: Rod, gravity, steps, command, lumen=None, initial_positions=None, initial_frames=None,
             root_pose=None, vessel_settings=None, sample_stride=STRIDE):
    from i4h_host.composition import load_case

    area_moment = math.pi * rod.radius**4 / 4.0
    # reference compliance L / (4 E (b I) dt^2) == ours 1 / (E k L dt^2): k = 4 b I / L^2 (twist: J = 2I, G)
    case = {
        "sim": {"device": "cuda:0", "envs": 1, "dt": rod.dt, "substeps": rod.substeps, "iterations": 1,
                "cuda_graph": False, "gravity": list(gravity)},
        "assets": {"catheter": {"type": "rod", "start": list(START), "direction": [1.0, 0.0, 0.0],
                                "count": rod.segments + 1, "segment_length": rod.seg, "radius": rod.radius,
                                "mass": rod.node_mass}},
        "systems": [
            {"type": "particles", "damping": 1.0 - rod.damping},
            {"type": "rod_direct", "id": "rod", "body": "catheter", "young": rod.young, "torsion": rod.shear,
             "bend": 4.0 * rod.bend * area_moment / rod.seg**2, "twist": 4.0 * rod.twist * 2.0 * area_moment / rod.seg**2,
             "angular_damping": rod.damping, "inertia": INERTIA},
            {"type": "catheter_control", "rod": "rod", "track_length": 0.0, "tip_edges": rod.tip_edges, "tip_mapping": TIP_MAPPING,
             "rates": [RATE] * 3, "lower": [-1.0e9, -1.0e9, -1.0e9], "upper": [1.0e9, 1.0e9, 1.0e9],
             "max_delta": 1.0},
        ],
    }
    if initial_positions is not None:
        case["assets"]["catheter"].update(startup_positions=np.asarray(initial_positions).tolist(),
                                         startup_frames=np.asarray(initial_frames).tolist(),
                                         rest_lengths=[rod.seg] * rod.segments)
    if root_pose is not None:
        case["systems"][-1].update(root_mode="external_pose", target_mode="absolute")
    maps = {}
    if lumen is not None:  # the rigid one-sided lumen, matched to the reference's (linear capsules, no spline)
        from i4h_endoluminal.loaders import LOADERS
        from i4h_endoluminal.systems import SYSTEMS

        case["sim"]["iterations"] = COUPLING
        lumen_points = lumen["points"] if isinstance(lumen, dict) else lumen
        case["assets"]["lumen"] = {"type": "lumen", "points": np.asarray(lumen_points).tolist(), "radius": TUBE_RADIUS}
        if isinstance(lumen, dict):
            case["assets"]["lumen"]["edges"] = np.asarray(lumen["edges"]).tolist()
        case["systems"].append({"type": "graph_containment", "body": "catheter", "centreline": "lumen", "two_way": False,
                                "spline": False, "candidates": False, "open_root": True, "max_distance": 0.05,
                                "iterations": 2, "source_max_delta": 0.005, "target_max_delta": 0.005})
        if vessel_settings is not None:
            settings = vessel_settings
            dynamics = settings["dynamics"]
            case["assets"]["lumen"]["dynamic"] = True
            particles, elastic, control, contact = case["systems"]
            particles["body"] = "catheter"
            particles["id"] = "catheter_particles"
            contact.update(two_way=settings["two_way"], target_response=settings["vessel_response"],
                           interior_deadband=settings["interior_deadband"], interior_stiffness=settings["interior_stiffness"])
            case["systems"] = [particles,
                {"type": "particles", "id": "wall_particles", "body": "lumen", "gravity_scale": 0.0, "damping_at_finalize": True,
                 "damping": 1.0 - dynamics["linear_damping"]},
                {"type": "rod_pbd", "body": "lumen", "scheme": "colored", "pre_sweeps": dynamics["iterations"], "sweeps": dynamics["iterations"],
                 "stretch": dynamics["stretch_stiffness"], "bend": dynamics["bend_stiffness"], "twist": dynamics["twist_stiffness"],
                 "angular_damping": dynamics["angular_damping"], "max_delta": settings["vessel_max_delta"], "limit_motion": True},
                contact, elastic, control]
        maps = {"loaders": {**LOADERS, "lumen": add_lumen}, "systems": SYSTEMS}
    sim = load_case(case, **maps)
    cc = next(s for s in sim.systems if type(s).__name__ == "CatheterControl")
    q = sim.scene.field("q", sim.scene.body("catheter"))
    wall_rest = None if lumen is None else sim.scene.field("q", sim.scene.body("lumen")).numpy()[0].copy()
    sdt = rod.dt / rod.substeps
    trace = sampled_trace()
    tip_prev = 0.0
    started = time.perf_counter()
    for k in range(steps):
        push, rotate, tip = command(k)
        # catheter_control integrates normalised rates (clamped to [-1, 1], times ``rates``): insertion and rotation as
        # given, the tip angle as its rate this step
        rates = np.float64([push, rotate, (tip - tip_prev) / rod.dt]) / RATE
        if np.abs(rates).max() > 1.0:
            raise ValueError(f"step {k}: a command exceeds catheter_control's rate {RATE}")
        if root_pose is not None:
            position, orientation = root_pose(k, q.numpy()[0], cc.rod.frame.numpy())
            cc.write_root_poses(np.asarray(position).reshape(1, 3), np.asarray(orientation).reshape(1, 4))
            cc.absolute_targets.assign(np.float32([[0.0, 0.0, tip]]))
        else:
            cc.actions.assign(np.float32([rates]))
        tip_prev = tip
        sim.step()
        if k % sample_stride == sample_stride - 1:
            rod_system = cc.rod
            trace["positions"].append(q.numpy()[0].copy())
            trace["orientations"].append(rod_system.frame.numpy().reshape(-1, 4).copy())
            trace["velocities"].append(sim.scene.field("qd", cc.body).numpy()[0].copy())
            trace["angular_velocities"].append(rod_system.omega.numpy().reshape(-1, 3).copy())
            trace["commands"].append([push, rotate, tip])
            trace["time"].append((k + 1) * rod.dt)
            trace["vessel_displacement"].append(np.empty((0, 3)) if lumen is None else
                                                sim.scene.field("q", sim.scene.body("lumen")).numpy()[0] - wall_rest)
            scale = -1.0 / (sdt * sdt * rod.substeps)  # the driver's imposed impulse -> the rod's wrench on the holder
            trace["reaction"].append(np.concatenate([cc.impulse.numpy()[0, 0], cc.moment.numpy()[0, 0]]) * scale)
    result = finish_trace(trace)
    result["elapsed_s"] = time.perf_counter() - started
    result["case"] = json.dumps(case)
    return result


def run(engine: str, rod: Rod, out_path=None, **kw) -> dict:
    out = {"rod": json.dumps(asdict(rod)), "engine": engine, "schema": 2,
           "source": json.dumps(source_identity(engine)),
           "effective_config": json.dumps({"inertia": INERTIA if engine == "ours" else
                                          ("physical" if kw.get("physical_inertia", True) else "unit"),
                                          "tip_mapping": TIP_MAPPING if engine == "ours" else "polyline",
                                          "distance_sweeps": 0 if engine == "ours" else 128,
                                          "coupling_iterations": COUPLING, "stride": STRIDE,
                                          "cuda_graph": False, "vessel": "rigid, one-sided, no interior pull"})}
    out["complete"] = False
    for half in (False, True):
        r = Rod(**{**asdict(rod), "substeps": rod.substeps * 2}) if half else rod
        for name, (gravity, steps, command) in scenarios(r).items():
            print(f"{engine} {name}{'@half' if half else ''}: starting", flush=True)
            lumen = tube(name)
            trace = (run_reference(r, gravity, steps, command, lumen=lumen, **kw) if engine == "reference"
                              else run_ours(r, gravity, steps, command, lumen=lumen))
            key = name + ("@half" if half else "")
            out.update({key + "/" + field: value for field, value in trace.items()})
            if out_path is not None:
                np.savez(out_path, **out)
            print(f"{engine} {key}: tip {trace['positions'][-1, -1]}, reaction {trace['reaction'][-1]}", flush=True)  # force, moment
    out["complete"] = True
    return out


# --- metrics --------------------------------------------------------------------------------------------------------

def curvature(frames):
    """(T, N-2) turning angle per unit length at each interior node."""
    e = np.diff(frames, axis=1)
    e /= np.linalg.norm(e, axis=2, keepdims=True)
    cos = np.clip((e[:, 1:] * e[:, :-1]).sum(axis=2), -1.0, 1.0)
    return np.arccos(cos) / np.linalg.norm(np.diff(frames, axis=1), axis=2)[:, 1:]


def tip_roll(frames):
    """(T,) the tip's angle about the x axis (its lateral offset's direction in the y-z plane)."""
    d = frames[:, -1] - frames[:, 0]
    return np.unwrap(np.arctan2(d[:, 2], d[:, 1]))


def validate_archives(ref, ours):
    """Fail closed on stale, incomplete, nonfinite or mismatched qualification inputs."""
    for engine, data in (("reference", ref), ("ours", ours)):
        if not bool(data.get("complete", False)):
            raise ValueError(f"{engine}: incomplete campaign")
        if int(data.get("schema", 0)) != 2 or str(data.get("engine")) != engine:
            raise ValueError(f"{engine}: requires a schema-2 {engine} archive")
        source = json.loads(str(data["source"]))
        if engine == "reference" and source["revision"] != REFERENCE_REVISION:
            raise ValueError("wrong reference revision")
        if engine == "reference" and source["status"]:
            raise ValueError("reference source has tracked modifications")
        rod = Rod(**json.loads(str(data["rod"])))
        for half in ("", "@half"):
            for name, (_, steps, command) in scenarios(rod).items():
                sample_steps = range(STRIDE - 1, steps, STRIDE)
                T, P = len(sample_steps), rod.segments + 1
                wall = tube(name)
                W = 0 if wall is None else len(wall)
                shapes = {"positions": (T, P, 3), "orientations": (T, P, 4), "velocities": (T, P, 3),
                          "angular_velocities": (T, P, 3), "reaction": (T, 6), "commands": (T, 3),
                          "time": (T,), "vessel_displacement": (T, W, 3)}
                for field in TRACE_FIELDS:
                    key = name + half + "/" + field
                    if key not in data or not np.isfinite(data[key]).all():
                        raise ValueError(f"{engine}: missing or nonfinite {key}")
                    if data[key].shape != shapes[field]:
                        raise ValueError(f"{engine}: invalid shape for {key}: {data[key].shape}, expected {shapes[field]}")
                if not np.array_equal(data[name + half + "/commands"], np.asarray([command(k) for k in sample_steps])):
                    raise ValueError(f"{engine}: commands differ from the frozen campaign: {name}{half}")
                if not np.array_equal(data[name + half + "/time"], [(k + 1) * rod.dt for k in sample_steps]):
                    raise ValueError(f"{engine}: sample times differ from the frozen campaign: {name}{half}")
    if str(ref["rod"]) != str(ours["rod"]):
        raise ValueError("rod configurations differ")
    for half in ("", "@half"):
        for name in SCENARIOS:
            key = name + half + "/"
            for field in TRACE_FIELDS:
                if ref[key + field].shape != ours[key + field].shape:
                    raise ValueError(f"trace shapes differ: {key + field}")
            for field in ("commands", "time"):
                if not np.array_equal(ref[key + field], ours[key + field]):
                    raise ValueError(f"trace inputs differ: {key + field}")


def compare(ref: dict, ours: dict) -> list[dict]:
    """physics/component-contracts.md section 4: each parity metric within its bound; at half the substep each metric's
    change at most half its bound (refinement). Plus informational rows (``pass`` None): each engine's own trajectory
    change at half the substep. Both engines damp velocity by ``1 - damping`` per substep, so halving the substep
    doubles the damping per second; a change there is the damping model's, shared by both."""
    validate_archives(ref, ours)
    rod = Rod(**json.loads(str(ref["rod"])))
    L = rod.length
    rows = []

    def warm(x):  # after warm-up: the first fifth dropped
        return x[len(x) // 5:]

    def parity(suffix):
        a, b = (lambda k: ref[k + suffix + "/positions"]), (lambda k: ours[k + suffix + "/positions"])
        out = []
        for name, (rms_bound, max_bound) in (("sag", (0.01, 0.02)), ("insertion", (0.02, 0.04))):
            d = np.linalg.norm(warm(a(name))[:, -1] - warm(b(name))[:, -1], axis=1)
            out += [(name, "tip RMS / L", np.sqrt((d**2).mean()) / L, rms_bound),
                    (name, "tip max / L", d.max() / L, max_bound)]
        for name in ("tube_straight", "tube_bend"):  # the contract's insertion through a tube and a bend: RMS only
            if name + suffix + "/positions" in ref and name + suffix + "/positions" in ours:
                d = np.linalg.norm(warm(a(name))[:, -1] - warm(b(name))[:, -1], axis=1)
                out.append((name, "tip RMS / L", np.sqrt((d**2).mean()) / L, 0.02))
        ka, kb = warm(curvature(a("tip_bend"))), warm(curvature(b("tip_bend")))
        out.append(("tip_bend", "curvature rel L2", np.linalg.norm(ka - kb) / max(np.linalg.norm(ka), 1e-12), 0.10))
        ra, rb = warm(tip_roll(a("twist"))), warm(tip_roll(b("twist")))
        ra, rb = ra - ra[0], rb - rb[0]  # the turn since warm-up
        out.append(("twist", "tip angle RMS / peak", np.sqrt(((ra - rb) ** 2).mean()) / max(np.abs(ra).max(), 1e-12), 0.05))
        fa, fb = ref["sag" + suffix + "/reaction"][-1], ours["sag" + suffix + "/reaction"][-1]
        out.append(("sag", "root reaction rel error", np.linalg.norm(fa[:3] - fb[:3]) / max(np.linalg.norm(fa[:3]), 1e-12), 0.10))
        if fa.shape[-1] == 6 and fb.shape[-1] == 6:  # the moment about the root (the reaction bound applies)
            out.append(("sag", "root moment rel error", np.linalg.norm(fa[3:] - fb[3:]) / max(np.linalg.norm(fa[3:]), 1e-12), 0.10))
        return out

    full, half = parity(""), parity("@half")
    for name, metric, value, bound in full:
        rows.append({"scenario": name, "metric": metric, "value": float(value), "bound": bound, "pass": bool(value <= bound)})
    for (name, metric, value, bound), (_, _, value_half, _) in zip(full, half):
        change = abs(value_half - value)
        rows.append({"scenario": name + " refinement", "metric": metric + " change at half substep", "value": float(change),
                     "bound": bound / 2.0, "pass": bool(change <= bound / 2.0)})
    for engine, data in (("reference", ref), ("ours", ours)):
        for name in ("sag", "insertion", "tip_bend", "twist", "tube_straight", "tube_bend"):
            if name + "/positions" not in data:
                continue
            d = np.linalg.norm(data[name + "/positions"][:, -1] - data[name + "@half/positions"][:, -1], axis=1)
            rows.append({"scenario": f"{name} self ({engine})", "metric": "tip RMS / L at half substep (info)",
                         "value": float(np.sqrt((d**2).mean()) / L), "bound": None, "pass": None})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=("reference", "ours"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--unit-inertia", action="store_true", help="reference: keep the solver's legacy identity inertia")
    r.add_argument("--inertia", default="physical", choices=("unit", "physical"), help="ours: rod_direct.inertia")
    r.add_argument("--tip-mapping", default="polyline", choices=("per_edge", "polyline"), help="ours: catheter_control.tip_mapping")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("ours")
    c.add_argument("--report")
    args = parser.parse_args(argv)
    if args.command == "run":
        global INERTIA, TIP_MAPPING
        INERTIA, TIP_MAPPING = args.inertia, args.tip_mapping
        kw = {"physical_inertia": not args.unit_inertia} if args.engine == "reference" else {}
        np.savez(args.out, **run(args.engine, Rod(), out_path=args.out, **kw))
        return 0
    ref, ours = dict(np.load(args.reference)), dict(np.load(args.ours))
    rows = compare(ref, ours)
    for x in rows:
        verdict = "info" if x["pass"] is None else ("PASS" if x["pass"] else "FAIL")
        bound = "" if x["bound"] is None else f"  (bound {x['bound']:.4g})"
        print(f"{verdict:<5} {x['scenario']:<26} {x['metric']:<44} {x['value']:.4g}{bound}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rows, f, indent=2)
    return 0 if all(x["pass"] is not False for x in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
