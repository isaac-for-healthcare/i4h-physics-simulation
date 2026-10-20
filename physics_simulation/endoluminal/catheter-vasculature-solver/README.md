# Endoluminal Solver

Standalone Python package for catheter/guidewire rod dynamics, focused on vasculature navigation.

This package is designed to run independently: it contains its own solver modules, data/config types, and runtime entry points under `src/catheter_vasculature_solver`.

## Standalone package scope

- Provides GPU-accelerated XPBD/Cosserat rod simulation primitives.
- Includes vessel-aware catheter extensions (track guidance + mesh containment).
- Vendors the `SolverXPBDRod` rod solver in-tree and exposes an optional Newton bridge to it (see [Optional Newton bridge](#optional-newton-bridge)).
- Exposes the rod through Newton's `SolverBase` interface, so it runs under Isaac Lab's Newton Manager.
- Keeps solver logic self-contained in package modules (no cross-package imports into `vasculature-digital-twin`).

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
- Containment (static mesh and deforming centerline), track guidance and per-env reset all run on the batched path, so `num_envs > 1` simulates the same physics as `num_envs = 1`. A deforming vessel must be built with a matching `num_envs` so each env owns its own wall; `CathRodSolver` raises on a mismatch rather than letting envs share one.

On the pinned Isaac Lab `ffff603` revision, the scene's reset callback must explicitly call the active manager's `reset_rod(world_mask=...)` after authoring any rigid-body reset and before stepping again. Pass one mask entry per environment, rather than a list of environment IDs: `[False, True, False]` resets only environment 1. `None` resets all environments; an all-zero mask changes nothing.

Use `NewtonXPBDRodManager.reset_rod(...)` for rod-only scenes and `NewtonCoupledMJWarpXPBDRodManager.reset_rod(...)` for coupled scenes. Both publish only the selected rod particles and preserve other environments' queued rate or pose commands. The coupled entry point also clears only selected feedback history and resets rigid-solver internals when that solver exposes a reset method, preserving scene-authored body and joint state with `flags=0`. Newer Isaac Lab revisions invoke the managers' `_reset_solver_internals` hooks automatically, so their normal reset path needs no additional explicit call.

## Inter-package integration (with `vasculature-digital-twin`)

Use the digital twin package to generate patient-specific CT artifacts, then initialize solver constraints from those artifacts.

These packages ship in the [Isaac for Healthcare digital twin repository](https://github.com/isaac-for-healthcare/i4h-digital-twin) under `patient-digital-twin`. Two pieces matter for interoperability: `vasculature_digital_twin` provides the `vdt-*` commands used below, and `imaging_to_mesh` converts a vessel mask into the vessel USD (`convert_mask_to_usd`) that carries the lumen geometry into simulation.

1) Generate CT cache + vessel artifacts:

```bash
vdt-preprocess-ct --nifti /path/to/ct.nii.gz --output-dir /tmp/ct_cache
vdt-segment-vessels --ct-dir /tmp/ct_cache
```

1) Load centerline artifacts and derive an insertion track:

```python
import numpy as np

pts_mm = np.load("/tmp/ct_cache/centerline_points_mm.npy")  # (N, 3), millimeters
track_start = pts_mm[0] / 1000.0  # convert to meters
track_dir = pts_mm[1] - pts_mm[0]
track_dir = track_dir / (np.linalg.norm(track_dir) + 1e-12)
track_length = float(np.linalg.norm((pts_mm[-1] - pts_mm[0]) / 1000.0))
```

1) (Optional) Build a vessel collision mesh from the digital twin mask:

```python
from vasculature_digital_twin.vasculature import extract_vessel_mesh

vessel_mask = np.load("/tmp/ct_cache/vessel_mask.npy")
vessel_mesh = extract_vessel_mesh(
    vessel_mask=vessel_mask,
    spacing_zyx_mm=(1.0, 1.0, 1.0),  # replace with metadata spacing for accurate scale
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

Multi-environment vessels: pass `num_envs` to `CenterlineVesselRuntime.from_tree` and every env gets its own deformable wall, so two-way contact in one env cannot move another's. Vessel memory therefore scales with `num_envs`, while containment stays one batched launch per collision iteration. `env_positions` gives each env its own anatomy (shared topology, per-env geometry), and `solver.reset(env_ids)` restores the selected envs' rod *and* vessel to rest. See `examples/multi_env_rl_vessels.py`.

Because the launch count is independent of `num_envs`, wall-clock time per step is nearly flat and env-steps/s scales close to linearly. Measured on one RTX A6000 with a 24-segment catheter against a 16-segment two-way vessel, two Cosserat iterations per step:

| `num_envs` | steps/s | env-steps/s | vs. 1 env |
| --- | --- | --- | --- |
| 1 | 847 | 847 | 1.0x |
| 16 | 787 | 12 592 | 14.9x |
| 64 | 786 | 50 324 | 59.4x |
| 256 | 719 | 183 976 | 217.3x |

Per-step cost is dominated by fixed CPU launch overhead here, because CUDA-graph capture is disabled while a vessel is attached (see `_can_use_cuda_graph`), which is precisely why adding envs is close to free.

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
