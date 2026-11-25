# Numerical simulation documentation

Start here for architecture, package ownership, maintained guides and migration.
Package READMEs introduce their local APIs; this index connects them.

## Reading order

1. [Workspace setup](README.md) and [contribution checks](CONTRIBUTING.md).
2. Architecture and package ownership below.
3. [Component contracts](physics/component-contracts.md): ownership, hosting,
   continuation, qualification exclusions and deferred defects.
4. The guide for the domain or host you are using.

## Maintained guides and package ownership

| Package | Owns | Guides |
| --- | --- | --- |
| [physics](physics/README.md) | Scene, staged systems, solver, binding and reset | [Component contracts](physics/component-contracts.md), [performance](physics/performance.md) |
| [surgical](surgical/README.md) | Tissue, instruments, appearance and grasp measurements | [Liver grasp contract](surgical/liver-grasp-contract.md), [experiment](examples/liver_grasp/README.md) |
| [endoluminal](endoluminal/README.md) | Vessels, catheter control and reach measurements | [Vessel regions](endoluminal/vessel-regions.md), [catheter parity](endoluminal/catheter-parity.md) |
| [assessment](assessment/README.md) | Executor, history and rewards | [Component contracts](physics/component-contracts.md) |
| [isaaclab](isaaclab/README.md) | Lab hosts, gym tasks and cameras | [Versius](isaaclab/versius.md), [ReachTarget](isaaclab/reach-target-example.md), [Versius results](examples/versius/README.md) |
| [host](host/README.md) | Neutral composition, camera geometry, appearance and controls | [Composition example](examples/composition/README.md) |

The [standalone application](https://github.com/isaac-for-healthcare/i4h-standalone-simulator) owns rendering, devices, calibration,
teleoperation, UI, XR and recording. Its optional `i4h-isaaclab-standalone` bridge
owns Lab presentation and teleoperation. Follow that repository for application
setup and contribution instructions. Physics and headless Lab need neither.
The [asset manifest](data/asset-manifest.json) and [asset instructions](data/README.md)
remain separate from package payloads.

## Architecture

Physics and assessment are neutral packages. Domains depend on physics, and import
neither hosts nor each other. Standalone owns device adapters, teleoperation, input
calibration, UI, rendering, XR and capture. Isaac Lab owns environments, tasks and
cameras; its cameras operate without standalone. Environment configurations declare no
teleop device by default; `i4h_isaaclab_standalone.devices.configure_teleop(cfg)` installs the `i4h`
device, whose case then follows the environment case.

`PhysicsBinding` owns physical lifetime and masked reset. Host teardown and replacement
invalidate bindings. Assessment evaluates every completed physics sample, before rewards
and termination checks. Empty reset selections leave assessment and host pacing alone.
`Stepper.step()` advances one standalone physics sample. In Lab, the native
manager advances physics and owns decimation; the hosted binding has no `step()`. Per-sample reward pulses and action totals retain
their existing meanings.

Loader and constructor maps are explicit. Ordinary owning sidecars carry presentation;
physical surfaces can render with host defaults without appearance metadata. Structural
edits require preparation before capture; staged kernel order is preserved.

Catheter material keys follow the backend: direct rods expose `young`, `torsion`, `bend`
and `twist`; PBD rods expose `stretch`, `bend` and `twist` multipliers. These are backend
fields, with no conversion between physical stiffness and PBD multipliers.

Mesh Euler angles use Rx @ Ry @ Rz; input orientation offsets use Rz @ Ry @ Rx. Explicit
helper names state these orders; existing YAML interpretation and compatibility aliases
are retained. There is no `i4h_sim` compatibility package.

Host adapters pass their episode horizon (`max_episode_length * step_dt`) through
`horizon_s` when constructing assessment. The earlier host horizon or procedure
deadline wins; completion takes precedence at that boundary. Terminal exports identify
`timeout_source` and the effective `timeout_s`. External hosts must supply their
horizon explicitly; assessment has no host dependency.

## State ownership and continuation

Particles, tet and surface topology, instruments, rods, catheter commands, mounts and the episode history of every
system are fields of the Newton Model, State and Control, declared on the builder before finalize. TipEvidence and
Procedure state are host-owned assessment state and are not part of it.

| Operation | Call | Effect |
| --- | --- | --- |
| Episode reset | `Stepper.reset`, `PhysicsBinding.reset`, `I4hSolver.restore_episode` | Selected worlds return to authored state, history and command defaults |
| Cache reset | `I4hSolver.reset(state, mask, flags=0)` | Derived caches are refreshed; history is kept |
| Continuation | `save_i4h` / `restore_i4h`, `PhysicsBinding.save` / `restore`, `I4hComponent.save` / `restore` | i4h rows are copied between containers of one Model; no reset runs |

Continuation excludes rigid solver history (including MuJoCo-Warp's), host time and episode counters, assessment,
procedure and tip evidence, renderer state and recordings. Details are in [component contracts](physics/component-contracts.md).

## Renamed and removed cases

| Before | Now |
| --- | --- |
| `aorta`, `aorta_reach` | `aorta_static`, `aorta_static_reach` |
| `aorta_def`, `aorta_def_reach` | `aorta_shell`, `aorta_shell_reach` |
| `I4h-AortaDefReachTarget-Direct` | `I4h-ReachTarget-Direct` with `env_cfg.case = "aorta_shell_reach"` and `observe = "catheter"` |
| `i4h_physics/data/vessels/{vascular_templates,instrument_regions}.yaml` | `i4h_endoluminal/data/` |

The airway, airrc, colon, porcine CTA, S0065 and Syndaver cases are no longer shipped. Gym task ids
`I4h-Aorta-Direct` and `I4h-ReachTarget-Direct` are unchanged and now load `aorta_static` and `aorta_static_reach`.

## Standalone migration

| Previous symbol or command | Replacement |
| --- | --- |
| `i4h_isaaclab.composition` | `i4h_host.composition` |
| `i4h_isaaclab.appearance` | `i4h_host.appearance` |
| `i4h_isaaclab.camera.camera` | `i4h_host.camera.camera` |
| `i4h_isaaclab.camera.rig.resolve_tip`, `TIP_DEFAULTS` | `i4h_host.camera.tip` |
| `i4h_isaaclab.camera.rig.TipCamera` | `i4h_standalone.render.tip.TipCamera` |
| `i4h_isaaclab.views.controls` | `i4h_host.controls.controls` |
| `i4h_standalone.devices` | `i4h_isaaclab_standalone.devices` |
| `i4h_standalone.teleop` | `i4h_isaaclab_standalone.teleop` |
| `i4h_standalone.cartesian` | `i4h_isaaclab_standalone.cartesian` |
| `i4h_standalone.render.versius.Viewer` | `i4h_isaaclab_standalone.viewer.Viewer` |
| `i4h_isaaclab.versius.status.StatusPanel`, `guidance` | `i4h_isaaclab_standalone.status` |
| `scripts/isaaclab_run.py --render gl ...` | `i4h-lab-run --render gl ...` |
| `scripts/isaaclab_teleop.py ...` | `i4h-lab-teleop ...` |

Headless upstream `scripts/isaaclab_run.py` remains available. Its installed implementation
is `i4h_isaaclab.run`; `main(..., viewer_factory=..., view_loop_factory=...)` explicitly
accepts optional presentation. Old viewer invocations name the replacement command.
Task measurements and `failure_status` remain in `i4h_isaaclab.versius.status`.
No forwarding modules recreate dependencies between hosts. Ordinary app APIs and
`i4h-sim` remain available; case/YAML/action/recording contracts retain their formats.

No forwarding modules recreate dependencies between hosts. The application owns
its lockfile and setup; numerical packages retain their own pins. Historical
validation is separate from current verification.

## Historical reports and audit tools

Tracked checkpoint reports, plans, inventories, retired patches and qualification
records are available unchanged at baseline `8020858babd4ad3c2d1cebff7c75bcb2f2bd904c`. There is no local history archive.
Ignored raw evidence in `.tmp_tests/` and local artifacts is separate from those
tracked reports; Git cannot recover evidence that was never committed. Historical
acceptance does not establish current numerical, rendering, device or policy readiness.

Recover a tracked report from the repository root:

```bash
git show 8020858babd4ad3c2d1cebff7c75bcb2f2bd904c:numerical_sim/docs/stage-c44.md > /tmp/stage-c44.md
```

`scripts/inventory_stage_c.py --out /tmp/allocation-census.json` is an optional
source allocation census; it does not certify ownership or continuation.
`scripts/check_stage_c_inventory.py` requires an explicit `--inventory PATH`.
To verify an old inventory, use a matching historical checkout and that checkpoint's
checker and source, for example from that checkout's `numerical_sim/`:

```bash
python3 scripts/check_stage_c_inventory.py --root "$PWD" --inventory docs/stage-c-inventory.json
```

Do not regenerate old inventories or hashes to make them match current source.
Historical qualification JSONs and certificate/reference hashes are retained in
Git at the same baseline. Current checks belong to the patch being reviewed.

| Historical record | Disposition |
| --- | --- |
| [qualification/c30-core.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/qualification/c30-core.json) | Git history only |
| [qualification/c30-main.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/qualification/c30-main.json) | Git history only |
| [qualification/c30-lab30.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/qualification/c30-lab30.json) | Git history only |
| [qualification/approved-references.sha256](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/qualification/approved-references.sha256) | Git history only |
| [qualification/c43-certificate.sha256](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/qualification/c43-certificate.sha256) | Git history only |
| [qualification/c44-certificate.sha256](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/qualification/c44-certificate.sha256) | Git history only |

## Original file dispositions

The table accounts for every file in the former documentation directory. Maintained
guides and required data moved; architecture and migration were consolidated above.
The remaining reports, inventories and patches are retained only in Git.

| Original | Disposition |
| --- | --- |
| [docs/asset-manifest.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/asset-manifest.json) | Relocated to [data/asset-manifest.json](data/asset-manifest.json) |
| [docs/catheter-parity.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/catheter-parity.md) | Relocated to [endoluminal/catheter-parity.md](endoluminal/catheter-parity.md) |
| [docs/component-contracts.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/component-contracts.md) | Relocated to [physics/component-contracts.md](physics/component-contracts.md) |
| [docs/current-stack-assessment-20261116.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/current-stack-assessment-20261116.md) | Git history only |
| [docs/dlv-010-29.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/dlv-010-29.md) | Git history only |
| [docs/internal-pr21-pr23-migration.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/internal-pr21-pr23-migration.md) | Git history only |
| [docs/liver-consolidation-results.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/liver-consolidation-results.md) | Results summarized in [liver example](examples/liver_grasp/README.md); report in Git |
| [docs/liver-grasp-contract.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/liver-grasp-contract.md) | Relocated to [surgical/liver-grasp-contract.md](surgical/liver-grasp-contract.md) |
| [docs/liver-grasp-results.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/liver-grasp-results.md) | Results summarized in [liver example](examples/liver_grasp/README.md); report in Git |
| [docs/liver-grasp-workflows-integration-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/liver-grasp-workflows-integration-plan.md) | Git history only |
| [docs/liver-policy-results.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/liver-policy-results.md) | Results summarized in [liver example](examples/liver_grasp/README.md); report in Git |
| [docs/newton18-lab30-cloner.patch](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/newton18-lab30-cloner.patch) | Relocated to [isaaclab/patches/newton18-lab30-cloner.patch](isaaclab/patches/newton18-lab30-cloner.patch) |
| [docs/newton18-lab30.patch](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/newton18-lab30.patch) | Git history only |
| [docs/newton18-qualification.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/newton18-qualification.md) | Git history only |
| [docs/package-migration.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/package-migration.md) | Consolidated in this index |
| [docs/performance.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/performance.md) | Relocated to [physics/performance.md](physics/performance.md) |
| [docs/pr12-coupling-qualification.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/pr12-coupling-qualification.md) | Git history only |
| [docs/reach-target-example.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/reach-target-example.md) | Relocated to [isaaclab/reach-target-example.md](isaaclab/reach-target-example.md) |
| [docs/stage-b-b1-followup.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b-b1-followup.md) | Git history only |
| [docs/stage-b-exclusive-timing.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b-exclusive-timing.md) | Git history only |
| [docs/stage-b-host.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b-host.md) | Git history only |
| [docs/stage-b-parity-investigation.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b-parity-investigation.md) | Git history only |
| [docs/stage-b-refresh-audit.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b-refresh-audit.md) | Git history only |
| [docs/stage-b-v3.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b-v3.md) | Git history only |
| [docs/stage-b.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-b.md) | Git history only |
| [docs/stage-c-closure.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c-closure.md) | Rules extracted into [component contracts](physics/component-contracts.md); report in Git |
| [docs/stage-c-field-contracts.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c-field-contracts.md) | Git history only |
| [docs/stage-c-inventory-c1-accepted.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c-inventory-c1-accepted.json) | Git history only |
| [docs/stage-c-inventory.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c-inventory.json) | Git history only |
| [docs/stage-c-scratch-audit.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c-scratch-audit.md) | Git history only |
| [docs/stage-c.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c.md) | Git history only |
| [docs/stage-c21-closure.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c21-closure.md) | Git history only |
| [docs/stage-c21-followup.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c21-followup.md) | Git history only |
| [docs/stage-c22.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c22.md) | Git history only |
| [docs/stage-c23.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c23.md) | Git history only |
| [docs/stage-c24.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c24.md) | Git history only |
| [docs/stage-c25.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c25.md) | Git history only |
| [docs/stage-c3-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c3-plan.md) | Git history only |
| [docs/stage-c30.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c30.md) | Git history only |
| [docs/stage-c31.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c31.md) | Git history only |
| [docs/stage-c32.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c32.md) | Git history only |
| [docs/stage-c33.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c33.md) | Git history only |
| [docs/stage-c4-inventory.json](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c4-inventory.json) | Git history only |
| [docs/stage-c4-inventory.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c4-inventory.md) | Git history only |
| [docs/stage-c4-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c4-plan.md) | Git history only |
| [docs/stage-c40.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c40.md) | Git history only |
| [docs/stage-c41-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c41-plan.md) | Git history only |
| [docs/stage-c41.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c41.md) | Git history only |
| [docs/stage-c42-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c42-plan.md) | Git history only |
| [docs/stage-c42.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c42.md) | Git history only |
| [docs/stage-c43-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c43-plan.md) | Git history only |
| [docs/stage-c43.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c43.md) | Git history only |
| [docs/stage-c44-plan.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c44-plan.md) | Git history only |
| [docs/stage-c44.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/stage-c44.md) | Git history only |
| [docs/standalone-split-validation.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/standalone-split-validation.md) | Git history only |
| [docs/standalone-split.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/standalone-split.md) | Consolidated in this index |
| [docs/versius-consolidation-results.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/versius-consolidation-results.md) | Results summarized in [Versius example](examples/versius/README.md); report in Git |
| [docs/versius.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/versius.md) | Relocated to [isaaclab/versius.md](isaaclab/versius.md) |
| [docs/vessel-regions.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/vessel-regions.md) | Relocated to [endoluminal/vessel-regions.md](endoluminal/vessel-regions.md) |
| [docs/warp117-hashgrid-capture.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/warp117-hashgrid-capture.md) | Git history only |
| [docs/warp117-hashgrid-capture.patch](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/warp117-hashgrid-capture.patch) | Git history only |
| [docs/xr.md](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/xr.md) | Git history only |
