# Endoluminal Solver

Standalone Python package for catheter/guidewire rod dynamics, focused on vasculature navigation.

This package is designed to run independently: it contains its own solver modules, data/config types, and runtime entry points under `src/catheter_vasculature_solver`.

## Standalone package scope

- Provides GPU-accelerated XPBD/Cosserat rod simulation primitives.
- Includes vessel-aware catheter extensions (track guidance + mesh containment).
- Vendors the `SolverXPBDRod` rod solver in-tree and exposes an optional Newton bridge to it (see [Optional Newton bridge](#optional-newton-bridge)).
- Exposes the rod through Newton's `SolverBase` interface, so it runs under Isaac Lab's Newton Manager.
- Keeps solver logic self-contained in package modules (no imports from the digital twin packages).

## Package layout

- `src/catheter_vasculature_solver/rod_data.py`
  - Dataclass configuration surface: `RodConfig`, `RodMaterialConfig`, `RodGeometryConfig`, `RodSolverConfig`.
  - Runtime state container `RodData` with torch + Warp interop buffers.
- `src/catheter_vasculature_solver/rod_kernels.py`
  - Warp kernels for prediction, constraints (stretch/shear/bend/twist), collisions, friction, and utility metrics.
- `src/catheter_vasculature_solver/rod_solver.py`
  - High-level solver loop and orchestration for Newton iteration, direct solve path, and collision integration.
- `src/catheter_vasculature_solver/xpbd_rod_solver.py`
  - Self-contained XPBD direct solver implementation (embedded Warp kernels, no external Newton requirement).
- `src/catheter_vasculature_solver/cath_rod_solver.py`
  - Catheter-in-vessel extensions: vessel containment paths and track-guided insertion behavior.
- `src/catheter_vasculature_solver/newton_rod_bridge.py`
  - Particle-buffer marshalling between a Newton `State` and the rod solver. No `newton` import, so it is unit-testable on its own.
- `src/catheter_vasculature_solver/newton_rod_solver.py`
  - `NewtonCathRodSolver`: `CathRodSolver` behind Newton's `SolverBase`. Needs only released Newton (`Model`, `State`, `SolverBase`).
- `src/catheter_vasculature_solver/newton_xpbd_rod/`
  - Vendored `SolverXPBDRod` rod solver (Apache-2.0, from [newton-physics/newton#1981](https://github.com/newton-physics/newton/pull/1981)), kept in-tree because it ships in no Newton release.
- `src/catheter_vasculature_solver/newton_xpbd_rod_wrapper.py`
  - Optional wrapper adapting `RodConfig` to the vendored `SolverXPBDRod`.
- `src/catheter_vasculature_solver/isaaclab_integration/`
  - Isaac Lab Newton Manager glue: catheter-only, or coupled MJWarp (rigid) + catheter rod.
  - Configs: `XPBDRodSolverCfg`, `CoupledMJWarpXPBDRodSolverCfg`.
  - Managers: `NewtonXPBDRodManager`, `NewtonCoupledMJWarpXPBDRodManager`.
  - Builder helpers: `add_catheter_rod_to_builder`, `rod_config_from_solver_cfg`.

## Install

From this package directory:

```bash
pip install -e .
```

That is all you need for `XPBDRodSolver` and `CathRodSolver`, which are the supported solvers.

### Optional Newton bridge

For `NewtonXPBDRodSolver`, `NewtonCathRodSolver` and the Isaac Lab managers:

The Newton path (`newton_rod_solver`, `isaaclab_integration`) needs only a released `newton` for `Model`, `State` and `SolverBase`. It does **not** need `SolverXPBDRod`, which exists solely on the closed, unmerged `newton-physics/newton` PR #1981 and is in no installable `newton` — not PyPI, not any `beta-*` tag, not upstream `main`, and not the `newton` shipped with Isaac Lab. The legacy `newton_xpbd_rod_wrapper` still requires that symbol and is therefore unsupported; `NewtonCathRodSolver` replaces it.

That resolves from PyPI and needs no git references. The rod solver is vendored in-tree under `catheter_vasculature_solver.newton_xpbd_rod`, so the extra only supplies Newton's `Model`, `State`, `Control`, `Contacts`, `ModelBuilder` and `SolverBase`. The bound is closed at both ends for compatibility rather than API reasons — see the comment above the extra in `pyproject.toml`.

The solver is vendored rather than imported from `newton.solvers` because `SolverXPBDRod` ships in **no** Newton release: it exists only on the head of [newton-physics/newton#1981](https://github.com/newton-physics/newton/pull/1981), closed unmerged on 2026-05-25, and is absent from Newton `main`. Resolution is deliberately not "Newton first, vendored second" — which implementation ran would then depend on the host environment, and identical inputs could integrate differently.

The Isaac Lab managers additionally need Isaac Lab `develop` plus an editable install of `source/isaaclab_newton` in the same environment. Builder helpers still import without Isaac Lab.

## Runtime dependencies

- Required: `numpy`, `torch`, `warp-lang`
- Optional: `newton>=1.2.1,<1.3` via the `[newton]` extra (for `NewtonXPBDRodSolver`, `newton_rod_solver` and builder spawn)
- Optional: `isaaclab` + `isaaclab_newton` (only for `isaaclab_integration` managers/configs)

If you only use `XPBDRodSolver` / `CathRodSolver`, you do not need Newton or Isaac Lab installed.

## Public API usage

```python
from catheter_vasculature_solver import RodConfig, XPBDRodSolver

cfg = RodConfig()
cfg.geometry.num_segments = 24
cfg.solver.newton_iterations = 4

solver = XPBDRodSolver(cfg)
for _ in range(100):
    solver.step(cfg.solver.dt)

positions = solver.positions
```

Vessel-aware variant:

```python
from catheter_vasculature_solver import RodConfig, CathRodSolver
```

## Isaac Lab Newton Manager

Isaac Lab's [Newton Manager Abstraction](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction.html) drives a Newton `SolverBase` and nothing else, so `NewtonCathRodSolver` wraps `CathRodSolver` in that interface. Track guidance, vessel containment and centerline vessel deformation keep running inside the wrapped solver — the Newton integration adds coupling without giving up behavior.

Wiring is three steps: author the rod's particles on the `ModelBuilder`, tell the manager which particles they are, and select the solver config.

```python
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg
from catheter_vasculature_solver.isaaclab_integration import (
    NewtonXPBDRodManager,
    XPBDRodSolverCfg,
    add_catheter_rod_to_builder,
)

# 1 + 2. Scene setup, before ModelBuilder.finalize():
particle_range = add_catheter_rod_to_builder(builder, rod_config, lock_root=True)
NewtonXPBDRodManager.register_rod(particle_range)

# 3. Solver selection; class_type dispatch resolves the manager.
sim_cfg = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=XPBDRodSolverCfg(num_segments=24, solver_backend="block_thomas"),
        num_substeps=4,
        use_cuda_graph=False,
    )
)
```

With track and collision disabled, the manager constructs an `XPBDRodSolver`. Vessel meshes, insertion tracks and centerline runtimes are GPU buffers rather than config values, so scenes needing them must supply a configured rod (or a `rod_factory` that builds it from the finalized model and solver config):

```python
NewtonXPBDRodManager.register_rod(particle_range, rod=CathRodSolver(...))
```

For a robot or tool sharing the Newton step with the catheter, swap in `CoupledMJWarpXPBDRodSolverCfg`, which nests the same rod config alongside `MJWarpSolverCfg` and follows Isaac Lab's coupled-manager pattern (the same shape as MJWarp + VBD).

Exercise the `SolverBase` path without Isaac Sim:

```bash
python examples/isaaclab_newton_manager_sketch.py
```

Notes:

- `NewtonXPBDRodManager` sets `_needs_collision_pipeline = False`: vessel containment runs in the rod solver's own kernels, which ignore Newton contact buffers. The coupled manager enables the pipeline because rigid bodies need it.
- Both managers require `NewtonCfg(use_cuda_graph=False)`: outer graph replay bypasses the Python pending-control latch and would ignore new commands. Rod-internal graph capture is also disabled by the Newton wrapper.
- Substeps multiply: `NewtonCfg.num_substeps` times `XPBDRodSolverCfg.num_substeps`. Leave the rod's at `1` unless you want both.
- The rod's device must match the Newton model's; the wrapper raises if they differ.
- `coupling_mode="two_way"` requires `drive_body_name` or `drive_body_index` and feeds the rod's lagged proximal reaction into that body.
- `drive_root_sync=True` holds the rod's root on `drive_mount_local` of that body inside every substep, between the rigid and rod solves, so the root follows the solved body pose without lag. Command the body, not the root: the solver raises if a task also latches a root command on the rod. With one coupling iteration this is a staggered explicit step, stable while `k·dt²/M < 4` for rod stiffness `k` against holder mass `M`.
- `drive_coupling_iterations > 1` (two-way only) re-solves each substep from its start with an updated reaction, as Newton's `Config.iterations`, so the holder's force and the rod's vessel contacts are solved together. `drive_relaxation_mode="aitken"` adapts the relaxation between iterations within `drive_relaxation_min`/`drive_relaxation_max`. Each iteration saves and restores the rod's full state, vessel included.
- `drive_proxy_mass=True` (with `drive_root_sync` and at least two iterations) makes the root a proxy of the holder, as Newton's proxies do: it carries the drive link's mass times `drive_proxy_mass_scale` and starts each substep with the holder's rewound free velocity, while containment, track guidance and friction still treat it as held. With two iterations it converges at any stiffness; only translation is a proxy.
- Containment (static mesh and deforming centerline), track guidance and per-env reset all run on the batched path, so `num_envs > 1` simulates the same physics as `num_envs = 1`. A deforming vessel must be built with a matching `num_envs` so each env owns its own wall; `CathRodSolver` raises on a mismatch rather than letting envs share one.

Outside Isaac Lab, with `pip install -e ".[newton-coupled]"` (Newton 1.8 or newer, which needs a newer Warp than Isaac Lab pins), `build_coupled_rod_solver` in `catheter_vasculature_solver.newton_coupled_rod` couples the rod through Newton's own `SolverCoupledProxy` from `newton.solvers.experimental.coupled`. The rigid entry owns the bodies, joints and shapes, and the rod entry owns the rod's particles. Each drive body becomes a proxy in the rod's view, carrying the rigid solver's articulated effective mass. Newton runs the iterations and the fixed or Aitken relaxation. The rod places its root on the proxy and reports its proximal reaction as the harvested wrench:

```python
from newton.solvers import SolverMuJoCo
from catheter_vasculature_solver.newton_coupled_rod import ROD_ENTRY, build_coupled_rod_solver

solver = build_coupled_rod_solver(
    model, rod, particle_range,
    rigid_solver=lambda view: SolverMuJoCo(view),
    drive_bodies=[flange_body], mount_local=[(0.0, 0.0, 0.05)],
    iterations=2, relaxation_mode="aitken",
)
solver.step(state_0, state_1, control, contacts, dt)
rod_solver = solver.solver(ROD_ENTRY)
```

On the pinned Isaac Lab `ffff603` revision, the scene's reset callback must explicitly call the active manager's `reset_rod(world_mask=...)` after authoring any rigid-body reset and before stepping again. Pass one mask entry per environment, rather than a list of environment IDs: `[False, True, False]` resets only environment 1. `None` resets all environments; an all-zero mask changes nothing.

Use `NewtonXPBDRodManager.reset_rod(...)` for rod-only scenes and `NewtonCoupledMJWarpXPBDRodManager.reset_rod(...)` for coupled scenes. Both publish only the selected rod particles and preserve other environments' queued rate or pose commands. The coupled entry point also clears only selected feedback history and resets rigid-solver internals when that solver exposes a reset method, preserving scene-authored body and joint state with `flags=0`. Newer Isaac Lab revisions invoke the managers' `_reset_solver_internals` hooks automatically, so their normal reset path needs no additional explicit call.

## Inter-package integration (with `patient_digital_twin`)

Use the digital twin package to build a patient twin from CT, then initialize solver constraints from its bundle.

`patient_digital_twin` ships in the [Isaac for Healthcare digital twin repository](https://github.com/isaac-for-healthcare/i4h-digital-twin) under `patient-digital-twin`. A patient-twin bundle holds the CT (`volume.npy`), a vessel mask on the CT grid (`vessel_mask.npy`), a navigation centerline graph (`centerline_points.npy`, `centerline_edges.npy`, `centerline_radii.npy`), and the anatomy USD (`patient_anatomy.usdc`). Points are in the scan's physical frame and unit, which `patient_twin.yaml` declares along with `meters_per_unit`; read it rather than assuming millimeters.

1) Build a bundle from CT. NV-Segment segments the requested vessels; to start from existing labels instead, see the digital twin README:

```bash
python -m patient_digital_twin \
  --source nvsegment --input /path/to/ct.nii.gz \
  --bundle-root /path/to/NV-Segment-CTMR/NV-Segment-CTMR \
  --classes aorta --format bundle --output /tmp/patient_bundle
```

1) Load the navigation centerline and derive an insertion track:

```python
from pathlib import Path

import numpy as np
import yaml

bundle = Path("/tmp/patient_bundle")
manifest = yaml.safe_load((bundle / "patient_twin.yaml").read_text())
artifacts, to_meters = manifest["artifacts"], manifest["meters_per_unit"]
points = np.load(bundle / artifacts["centerline_points"]) * to_meters  # (N, 3), meters
edges = np.load(bundle / artifacts["centerline_edges"])  # (E, 2) point-index pairs

# Insert from a graph endpoint (choose the one at your access site) along its edge.
start = int(np.flatnonzero(np.bincount(edges.ravel(), minlength=len(points)) == 1)[0])
edge = edges[(edges == start).any(axis=1)][0]
track_start = points[start]
track_dir = points[edge[edge != start][0]] - track_start
track_dir = track_dir / (np.linalg.norm(track_dir) + 1e-12)
track_length = float(np.linalg.norm(points - track_start, axis=1).max())
```

1) (Optional) Build a vessel collision mesh from the bundle's vessel mask, in the same frame:

```python
import warp as wp
from patient_digital_twin.geometry import mask_to_mesh, transform_points

volume = yaml.safe_load((bundle / artifacts["volume_metadata"]).read_text())
axes = volume["output"]["array_axes"]  # array order of the CT and masks, e.g. "ijk"
mask = np.load(bundle / artifacts["vessel_mask"]).transpose([axes.index(c) for c in "kji"])
voxels, faces = mask_to_mesh(mask)  # closed surface in voxel (i, j, k) coordinates
voxel_to_scan = np.asarray(manifest["transforms"]["voxel_to_scan"])
verts = transform_points(voxels, voxel_to_scan) * to_meters
if np.linalg.det(voxel_to_scan[:3, :3]) < 0:
    faces = faces[:, ::-1]  # keep normals outward on a mirrored grid
vessel_mesh = wp.Mesh(
    points=wp.array(verts.astype(np.float32), dtype=wp.vec3, device="cuda"),
    indices=wp.array(faces.flatten(), dtype=int, device="cuda"),
)
```

1) Run catheter simulation with vessel-aware solver:

```python
from catheter_vasculature_solver import RodConfig, CathRodSolver

cfg = RodConfig()
solver = CathRodSolver(
    cfg,
    collision_mesh=vessel_mesh,
    track_start=track_start,
    track_dir=track_dir,
    track_length=track_length,
    tip_num_edges=10,
    particle_radius=0.002,
    segment_length=cfg.geometry.segment_length,
)
```

This integration is optional: the solver package remains fully usable without digital twin inputs.

## Vessel deformation

Static vessel containment is available today via `CathRodSolver`, and compliant walls are simulated from the vessel centerline.

**Primary path:** branching **centerline Cosserat** (`pbd_cosserat` + `centerline_tree`) + live tapered-tube containment.
**Secondary:** Newton cloth / tet soft walls.

Foundation modules live under `catheter_vasculature_solver.vessel_deformation`. Live tapered-tube containment, two-way contact and surface skinning are wired into `CathRodSolver` through `CenterlineVesselRuntime`; cloth/tet remain gated. See `examples/centerline_vessel_deformation.py` for an end-to-end run.

Multi-environment vessels: pass `num_envs` to `CenterlineVesselRuntime.from_tree` and every env gets its own deformable wall, so two-way contact in one env cannot move another's. Vessel memory therefore scales with `num_envs`, while containment stays one batched launch per collision iteration. `env_positions` and `env_radii` give each env its own geometry and lumen widths over a shared topology, and `solver.reset(env_ids)` restores the selected envs' rod *and* vessel to rest. See `examples/multi_env_rl_vessels.py`.

Different patients in one batch: `CenterlineVesselRuntime.from_trees(trees, env_trees=...)` gives each env its own vessel tree, with its own node and edge counts and its own entry. Every env's vessel is padded to the largest tree, so the batch keeps one fixed-stride layout and containment stays one launch; padded nodes carry no mass and padded edges are never searched. Pair it with per-env guidance on `CathRodSolver`: `track_start` / `track_dir` accept `(num_envs, 3)`, and `track_path` accepts one route per env (lengths may differ). `add_catheter_rod_to_builder` takes per-env `start`, `direction` or `positions` for each env's starting shape. A batched env reproduces its patient simulated alone (`tests/test_per_env_anatomy.py`).

Mesh collision is single-anatomy only: `collision_mesh` is one `wp.Mesh` queried by every env, so the solver refuses it alongside per-env vessels or differing per-env routes. Use `collision_mesh=None` and centerline containment for mixed-patient batches.

Routes of different lengths can get catheters of different lengths: `env_segment_lengths` on the solver sets each env's segment rest length while every rod keeps `num_segments` segments. Pass the same values to `add_catheter_rod_to_builder` so Newton's particle spacing and masses agree with the rod's. Default positions are then spaced by each env's length; explicit `positions` must already be, or the builder raises, because the bridge treats Newton positions as authoritative and a mismatched rod would snap to its rest length on the first step.

The launch count is independent of `num_envs` (45 kernel launches per step here), so env-steps/s grows with the batch even though wall-clock time per step does not stay flat. Measured on one RTX A6000 with the configuration of `examples/multi_env_rl_vessels.py`: a 24-segment catheter against a 16-segment two-way vessel (`vessel_response=0.6`), two Cosserat iterations per step, 20 warm-up steps then the median of three 500-step runs:

| `num_envs` | steps/s | env-steps/s | vs. 1 env |
| --- | --- | --- | --- |
| 1 | 961 | 961 | 1.0x |
| 16 | 756 | 12 093 | 12.6x |
| 64 | 637 | 40 787 | 42.4x |
| 256 | 318 | 81 459 | 84.8x |

Small batches are bound by fixed CPU launch overhead, because CUDA-graph capture is disabled while a vessel is attached (see `_can_use_cuda_graph`), so adding envs there is nearly free. Large batches become GPU-bound: the catheter's block-tridiagonal solve (`solve` in `rod_linear_system.py`) runs one rod per thread in double precision and takes about 83% of GPU time at 256 envs. A mixed-patient batch (`from_trees` with per-env routes and rod lengths) runs at the same rate as a uniform one.

```python
from catheter_vasculature_solver import VesselDeformationConfig
from catheter_vasculature_solver.vessel_deformation import (
    CenterlineData,
    build_centerline_tree,
    CosseratRod,  # lazy; requires Warp
)

# Safe default — current static behavior
cfg = VesselDeformationConfig(deformation="off")

# Build a centerline tree from digital twin branch samples
# tree = build_centerline_tree(CenterlineData(starts=..., ends=..., branch_ids=..., ...))
```

## Notes for standalone use

- The package follows a standard `src/` Python layout and can be reused in other projects via editable install or wheel build.
- Solver variants share the same configuration schema, so backends can be swapped without changing high-level parameter wiring.
- The vasculature digital twin package is complementary but not required to run this solver package.
