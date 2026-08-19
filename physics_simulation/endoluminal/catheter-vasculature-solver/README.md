# Endoluminal Solver

Standalone Python package for catheter/guidewire rod dynamics, focused on vasculature navigation.

This package is designed to run independently: it contains its own solver modules, data/config types, and runtime entry points under `src/catheter_vasculature_solver`.

## Standalone package scope

- Provides GPU-accelerated XPBD/Cosserat rod simulation primitives.
- Includes vessel-aware catheter extensions (track guidance + mesh containment).
- Exposes an optional bridge to Newton's `SolverXPBDRod` backend.
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
- `src/catheter_vasculature_solver/newton_xpbd_rod_wrapper.py`
  - Optional wrapper around Newton's `SolverXPBDRod` when that runtime is available.
- `src/catheter_vasculature_solver/isaaclab_integration/`
  - **Path B** Isaac Lab Newton Manager glue: coupled MJWarp (rigid) + `SolverXPBDRod` (catheter).
  - Configs: `XPBDRodSolverCfg`, `CoupledMJWarpXPBDRodSolverCfg`.
  - Managers: `NewtonXPBDRodManager`, `NewtonCoupledMJWarpXPBDRodManager`.
  - Builder helpers: `register_xpbd_rod_builder_attributes`, `add_catheter_rod_to_builder`.

## Install

From this package directory:

```bash
pip install -e .
```

Optional Newton backend:

```bash
pip install -e ".[newton]"
```

Isaac Lab Path B (coupled manager) needs Isaac Lab `develop` + Newton with `SolverXPBDRod` installed in that env (not pip-published as a simple extra). Builder helpers still import without Lab.

## Runtime dependencies

- Required: `numpy`, `torch`, `warp-lang`
- Optional: `newton` (only for `NewtonXPBDRodSolver` / builder spawn)
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

Newton bridge variant (optional dependency):

```python
from catheter_vasculature_solver import RodConfig, NewtonXPBDRodSolver
```

## Isaac Lab Newton Manager — Path B (coupled rigid + catheter)

Use this when a robot/tool must share a Newton step with the catheter rod (MJWarp for rigid bodies, `SolverXPBDRod` for the Cosserat rod). This follows the [Newton Manager Abstraction](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction.html) coupled-solver pattern (same idea as MJWarp + VBD).

```python
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from catheter_vasculature_solver import RodConfig
from catheter_vasculature_solver.isaaclab_integration import (
    CoupledMJWarpXPBDRodSolverCfg,
    XPBDRodSolverCfg,
    add_catheter_rod_to_builder,
    register_xpbd_rod_builder_attributes,
)

sim_cfg = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=CoupledMJWarpXPBDRodSolverCfg(
            coupling_mode="one_way",  # or "two_way"
            rigid_solver_cfg=MJWarpSolverCfg(),
            rod_solver_cfg=XPBDRodSolverCfg(solver_backend="block_thomas"),
            soft_contact_ke=5.0e3,
            soft_contact_mu=0.5,
        ),
        num_substeps=4,
    )
)

# During scene / ModelBuilder setup (before finalize):
# register_xpbd_rod_builder_attributes(builder)
# add_catheter_rod_to_builder(builder, RodConfig(...), lock_root=True)
```

Dry-run the wiring checklist (prints the Lab snippet if Lab is missing):

```bash
python examples/path_b_coupled_newton_manager_sketch.py
```

Notes:

- Do **not** assign `CathRodSolver` to `NewtonManager._solver` — it is not a Newton `SolverBase`.
- Vessel SDF / track guidance from `CathRodSolver` are optional post-step hooks via `NewtonCoupledMJWarpXPBDRodManager.register_vessel_track(...)`.
- Prefer Newton collision meshes for the vessel lumen when sharing contacts with tools (`_needs_collision_pipeline=True`).

## Inter-package integration (with `vasculature-digital-twin`)

Use the digital twin package to generate patient-specific CT artifacts, then initialize solver constraints from those artifacts.

These packages ship in the [Isaac for Healthcare digital twin repository](https://github.com/isaac-for-healthcare/i4h-digital-twin-internal) under `patient-digital-twin`. Two pieces matter for interoperability: `vasculature_digital_twin` provides the `vdt-*` commands used below, and `imaging_to_mesh` converts a vessel mask into the vessel USD (`convert_mask_to_usd`) that carries the lumen geometry into simulation.

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

Foundation modules live under `catheter_vasculature_solver.vessel_deformation`. Live tapered-tube containment, two-way contact and surface skinning are wired into `CathRodSolver` through `CenterlineVesselRuntime` (single-environment path only); cloth/tet remain gated. See `examples/centerline_vessel_deformation.py` for an end-to-end run.

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
