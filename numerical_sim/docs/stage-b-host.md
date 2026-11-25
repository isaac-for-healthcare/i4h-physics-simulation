# Stage B host replacement evidence

The v3 prerequisite was qualified before changing host code. Its six-repeat
final-B1 archive is `.tmp_tests/stage-b-v3/baseline-v3-final/`; its immutable
index and the complete pre-change checks are described in
[stage-b-v3.md](stage-b-v3.md). Zero floors, factor three and slot canonicalization
remain unchanged. No diagnostic v2 envelope was promoted.

## Implementation

Main adapters are `I4hNewtonSolver` and `I4hCoupledNewtonSolver` in
`i4h_isaaclab/newton_solver.py`. `I4hSolverCfg` chooses one when constructed; their
capabilities never change dynamically. `lab30.py` implements the release bridge.
The common component-owned context is in `newton_manager.py`; numerical coupling
is in `i4h_physics/coupling.py`. Both releases call the same authoring, binding,
preparation, cache refresh and episode restoration functions as standalone.

Lab allocates model/state/control, initializes contacts, advances time and owns
capture. Main builds the solver unbound and binds in a one-shot solver-init
callback registered on `PHYSICS_READY`. Release 3.0 binds in solver construction
and completes output allocation in contact initialization. The retained builder
keeps immutable authoring; hard rebuilds create fresh runtime systems without
adding entities again. A new builder gets fresh authoring.

The component owns lifecycle subscriptions. Stop, rebuild, failed initialization
and close invalidate old bindings. Main's small manager subclass only validates
live buffers before the inherited step and orders host views afterward; it does
not implement a numerical loop. Release-specific class manager access is confined
to `lab30.py`. Device-resident assessment runs in main's adapter `post_step` or
3.0's solver-step hook, once per physics step. General post-step callbacks are
not used for assessment. Terminal observations precede physical restoration,
host reset events and assessment reset.

Hosted `PhysicsBinding` exposes scene, systems, host state/control, measurements
and restoration, with no hosted step, capture owner or independent clock.
`NewtonSim`, the alternate backend and its CLI switches are deleted. Explicit
legacy recording metadata can select the sole Newton host path while preserving
physical action layout; insufficient metadata fails. Native deformable views
remain disabled for existing component-frame particles.

Pure i4h rejects external collision inputs and native force-contact requests.
Coupled adapters delegate native rigid contact allocation and legacy force export
to MuJoCo-Warp. Requested `CONTACT_F` observables use distinct wrapper and child
containers allocated before capture. Captured arrays, output containers and
requested output kinds cannot be replaced silently. Mounted force and moment
accounting and the one-physics-step reaction delay are unchanged.

## Environments and reproducibility

All results below live under `.tmp_tests/stage-b-host/`. Environments and source
checkouts supplied for qualification were not synchronized or repinned:

- Newton `1.8.0rc1`, source `91fb115d685a96311b2f1bac5527754ee4e9c21b`;
  Warp `1.17.0`; MuJoCo and MuJoCo-Warp `3.17.0`.
- Main: `dc01ec356052f2acd04246752a92a55162f6cd68`, `main-env`.
- Release: `e53cd288727a5ca3dae7187d1d415b4d32b67e11` plus the explicitly
  recorded [cloner delta](newton18-lab30-cloner.patch), in the isolated
  `lab30-cloner` worktree, `lab30-env`. Unmodified e53cd288 is not claimed to
  pass the environment-cloner path.
- Standalone: the qualified `core-env`. A separate new `wheel-env` installs
  the built packages with the existing workspace dependency lock.

The private `run_lab.py` selects the corresponding supplied interpreter and
places the six current package source roots and the chosen Lab source roots on
`PYTHONPATH`. It supplies `I4H_DATA`. To replay with those imports already set:

```bash
I4H_DATA="$PWD/data" "$PY" scripts/qualify_stage_b.py \
  --host main --baseline .tmp_tests/stage-b-v3/baseline-v3-final \
  --out /path/to/fresh/main-result
```

Use `--host lab30` or `--host standalone` with its matching interpreter/imports.
Every case runs in a fresh process. The harness checks reference integrity,
scenario, repetition count, hardware/software identity and the unchanged v3
comparison; it writes candidate results separately and rejects reused result
directories. The stepping facade exists only in this test harness and forwards
to Lab's step. Production bindings have no stepping method.

## Verification

Initial full v3 replays passed all six cases on main, patched 3.0 and standalone
(`main-v3/`, `lab30-v3/`, `standalone-v3-assets/`). The first standalone invocation
omitted `I4H_DATA`; its missing-asset failures are retained in `standalone-v3/`
and are not numerical comparisons. The final-source main replay failed only `chole_tet_med`, on eight stepped
force/moment products. Its largest tool-force difference was 15.98 against a
7.9 envelope limit; jaw load differed by 4.863 against 2.6. These are positive-noise
products; zero-noise products remained byte-identical. The failed replay is
retained in `main-v3-final/`. The final-source patched 3.0 and standalone replays passed all six
cases (`lab30-v3-final/`, `standalone-v3-final/`). The earlier passing main sample does not supersede
this failure. No new envelope or tolerance is substituted.

A subsequent paired chole diagnostic ran six fresh processes per source with the
identical final recorder, RC1 environment, scenario and frozen v3 gate:

| Source | Passing runs | Failed products |
| --- | ---: | --- |
| Preserved pre-change | 6/6 | none |
| Preserved final B1 | 5/6 | 16 force/moment products in one run |
| Main replacement | 6/6 | none in this diagnostic; the earlier holdout still failed |

The B1 failure includes exactly the same eight stepped grasp/contact/tool-force
products as the retained main failure, plus their masked-reset counterparts.
Tool-force difference was 14.58 and jaw-load difference 4.961. No zero-noise
product failed. This demonstrates that the failure pattern exists in accepted B1
without host replacement; it does not by itself close the frozen numerical gate
or establish an exact failure rate. All 18 candidates and results are in
`paired-chole/`. The v3 archive and all 48 indexed reference files are unchanged.
No enlarged envelope, tolerance adjustment or automatic acceptance is applied.

The patched 3.0 host suite passed **80 tests, 6 skipped**, covering direct and
manager environments, components, recording, teleoperation and new adapter
contracts. Main's final complete host suite passed **81 tests, 6 skipped**
(`main-host-final.log`). Its initial run had one stale test import of the removed
backend; that failure and its correction remain recorded. After the
last configuration/output checks, the focused component and adapter suite passed
**12 tests on each release**. This includes three actual hard rebuilds without
entity/subscription growth, stale handles, changed captured buffers, pure sensor
rejection, real native rigid forces and distinct observable ownership.

Existing runtime tests exercise eager/captured controls, decimation greater than
one, assessment cadence, terminal/reset-event ordering, side-stream writes,
consumer rebinding, coupled internal rigid substeps, and partial pending-reaction
restoration. These are actual Lab execution tests, not import-only probes.

The first full core/package run in the preserved core environment finished
725 passed, 47 skipped and three failures. Two migration issues were corrected:
neutral coupling index/error names and the test's obsolete Lab revision
expectation. The third was stale installed `i4h-physics` dependency metadata in
that preserved environment; a fresh installation reads the correct declared RC1
pin. No supplied environment was rewritten to fix its metadata. The full suite
in the fresh installation then passed **728 tests, 47 skipped**
(`installed-suite.log`). Final packages are in `dist-complete/`; the final
installed-root, CPU and focused package/coupling checks are logged separately.

The CPU synthetic example passes. Six wheels and six source distributions build
and pass the package/asset-boundary checker. A new isolated wheel installation
passes all six installed-root/version checks and the CPU synthetic example from
outside the checkout. Focused Ruff and `git diff --check` pass.

## Remaining Stage B gates

This adapter migration does not declare the whole Stage B complete. Optional
surgical geometry export and custom wrench observables, their overflow and full
analytic wrench matrix, and the complete performance qualification remain
separate work. Existing component measurement views and native rigid force
observables work; native shape identities or surgical contact forces are not
invented. No Stage C persistent-data or command/schema migration has begun.

The earlier coupling qualification has a preserved normal-reaction timestep
refinement failure (approximately 5.1% against the unchanged 5% bound). The final
rerun also failed that row, at 5.2%; all other probe and catheter rows passed
(`coupling-final.json`). Passing adapter tests does not waive it. The five Stage A cross-stack physical findings
remain unresolved. No Newton 1.6.1/RC1 parity is claimed.

Hosted timing samples use identical tissue/catheter workloads at 1 and 64 worlds,
20 warmup steps, 200 timed physics steps, captured execution and zero commands.
Construction and peak Warp memory are separate from warmed step medians. The
preserved pre-change source supplies the patched 3.0 comparison; main has no
pre-change adapter, so its first working measurements are a host-specific
reference only. Other user catheter processes were active on the GPU during
this session, limiting any final 5% performance claim. They were left untouched.

The samples below all stayed within the 5% bound; no increase in Warp's live
allocation was measured across any 200-step replay. They are bounded workload
evidence, with the shared-GPU limitation above, not a completed exclusive-hardware
performance qualification. Times are milliseconds; construction is seconds;
memory is peak Warp MB. Each triple is pre-change 3.0 / replacement 3.0 / main.

| Case | Worlds | Step median (before / 3.0 / main) | 3.0 change | Construction | Peak Warp MB |
| --- | ---: | --- | ---: | --- | --- |
| tissue_contact | 1 | 0.363 / 0.362 / 0.361 | -0.29% | 0.775 / 0.679 / 0.996 | 16.779 / 16.784 / 16.784 |
| tissue_contact | 64 | 0.530 / 0.529 / 0.528 | -0.27% | 0.795 / 0.702 / 1.000 | 16.900 / 17.123 / 17.123 |
| aorta_static | 1 | 8.010 / 7.979 / 7.963 | -0.40% | 0.849 / 0.757 / 1.052 | 16.822 / 24.959 / 24.959 |
| aorta_static | 64 | 10.528 / 10.455 / 10.270 | -0.69% | 0.862 / 0.768 / 1.070 | 20.486 / 37.460 / 37.460 |
