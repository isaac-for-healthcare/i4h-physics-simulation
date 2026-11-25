# Isaac Lab refactor: current-source assessment, 2026-11-16

**Recommendation:** retain the existing Stage C closure on its pinned stack and use this refactor's `I4hSolver` as the custom-physics foundation. A focused WORLD-row compatibility patch enables the requested current-source tissue/contact and dynamic-reaction probes. The bounded current-source assessment and the final patch regression checks pass, with no unresolved blocker within that scope. This does not extend the frozen numerical qualification to all current-stack behavior. Migrating v2, replacing the coupler and redesigning scheduling are separate work.

## Baseline and scope

Stage C was already closed by the user and published on `i4h-isaaclab/manager-env` at `b921eca5b12546fdcf51b0a6b9b3849291874a75`. This assessment began with a clean refactor worktree. The later request authorizes focused compatibility work and a recommendation, not another commit or push.

| Checkout | HEAD at assessment start | Local changes |
| --- | --- | --- |
| This refactor | `b921eca5b12546fdcf51b0a6b9b3849291874a75` | Initially clean; the compatibility patch and this report are uncommitted |
| Current Isaac Lab main | `a9c1eb5de328c61e285532d72cb6164b9d0c16f0` | Existing render, camera, simulation-context and deletion-presentation changes |
| Current Newton soft-solvers worktree | `ba27cbe6caad3ed6d852469658031e1990c12037` | Existing builder, deformable-visual, viewer and deletion-presentation changes |
| Original numerical_sim checkout | `f772f330951a8311efca9995dc2707139b196f0a` | Existing standalone CLI and untracked local work |
| i4h-v2-physics-simulation | `16882a4da24a2457f81a7180b4730ffe354906f3` | Existing topology, anatomy, interaction and training work |

The exact paths, statuses and SHA-256 hashes of modified/untracked files are in `.tmp_tests/current-stack-assessment-20261116/before.json`. Results apply to those local source trees, including their existing changes, rather than clean checkouts of HEAD alone. Neither original numerical_sim nor v2 was executed or changed for this assessment.

The frozen Stage C campaign remains [C4.4](stage-c44.md), on `95c8f4c`, with Newton `7f0c13af`, Lab main `4cfdfcfa`, patched Lab 3.0 `e53cd288`, patched Warp 1.17 and MuJoCo/MuJoCo-Warp 3.17. Its **35/36 groups, 46/48 replays, 17/18 chole trials**, and core/main/3.0 suite counts **1,261/123/122** are historical qualification results. They are not results for today's dependencies or for this patch. The accepted `aorta_shell` catheter-impulse discrepancy, chole trial failures and all [closure limitations](stage-c-closure.md) remain recorded.

## Isolation and compatibility change

A new environment under `.tmp_tests/current-stack-assessment-20261116/env` uses the retained main environment's third-party packages through a `.pth` file. `PYTHONPATH` selects the requested current Newton and Lab sources and this refactor's six packages. `PYTHONDONTWRITEBYTECODE=1` avoids writing bytecode into the retained dependencies. No dependency installation, pin change or sync was performed in the qualification environments.

This is a **current-source compatibility check with retained third-party dependencies**, not qualification of a newly resolved dependency lock or the local Lab `.venv`.

`imports.json` records the interpreter and actual module files:

| Import | Actual source |
| --- | --- |
| `newton` | `/home/pkorzeniowsk/Projects/newton/newton-1.9-worktrees/i4h-v2-soft-solvers/newton/__init__.py` |
| `isaaclab` | `/home/pkorzeniowsk/Projects/isaaclab/isaaclab-main/source/isaaclab/isaaclab/__init__.py` |
| `isaaclab_newton` | `/home/pkorzeniowsk/Projects/isaaclab/isaaclab-main/source/isaaclab_newton/isaaclab_newton/__init__.py` |
| `i4h_physics`, `i4h_surgical`, `i4h_isaaclab` | This worktree's `numerical_sim/<package>/src/` |
| Warp, Torch, MuJoCo, MuJoCo-Warp | Retained `stage-c30-20261109/main-env` site-packages |

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver `615.71.09`, 97,887 MiB reported VRAM (`gpu.csv`).

Versions imported: Newton `1.9.0.dev0`, Lab `46.1.1`, Lab-Newton `17.2.3`, Warp `1.17.0`, Torch `2.11.0+cu130`, MuJoCo/MuJoCo-Warp `3.17.0`. Warp's native library SHA-256 is `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`, matching the retained qualification package.

The first current-source smoke failed during construction: `native domain with foreign rows requires an explicit placement`. Current Newton sizes native WORLD attributes to `world_count + 1`, including a trailing global row; the qualified revision used `world_count`.

The focused patch:

- Allows this specific WORLD layout in `storage.Views`, exposing only local-world rows to systems. Other foreign native layouts still require explicit placement.
- Marks the trailing global WORLD row foreign in continuation, even when the shared-row mask slot is selected.
- Sizes gravity initialization to Newton's declared WORLD storage; component rotation and refresh affect local worlds only.
- Tests local-view isolation, gravity/global-row preservation and CPU/CUDA continuation with distinct poisoned source/destination bytes. Existing byte accounting now includes the actual WORLD backing.
- Updates the live allocation inventory's source hashes and site locations: still **454 sites in 192 files**, with unchanged ownership classifications. Adds one reasoned Newton-private-access allowance for querying the WORLD count.

No numerical kernel, staged hook order, adapter protocol, dependency pin or old certificate was changed. This patch is subsequent work and is not covered by the frozen C4.4 source certificate.

## Fresh execution results

Evidence directory: `.tmp_tests/current-stack-assessment-20261116`. Every launched Python command has a `*-command.json`, log and exit-code file. `integration_probe.py` is the bounded measurement harness; it imports the repository's existing block/probe and dynamic-holder fixtures.

The harness constructs `I4hEnv` with two environments, verifies that its binding aliases Lab's actual backend State and Control, and steps through `env.sim.step(render=False)`. The backend is `i4h_isaaclab.newton_solver.I4hHostManager`, reached through Lab's normal physics backend. Each rollout runs 60 physics steps. CUDA graph presence is asserted, not inferred from configuration.

| Measurement | Eager | CUDA graph |
| --- | --- | --- |
| Maximum tissue displacement caused by env 0's instrument, relative to no-contact rollout | About `0.02955` case units | About `0.02955` case units |
| Maximum env 1 difference from its no-contact rollout | `5.70e-8` | `6.12e-8` |
| Selected env 0 episode reset | Positions exactly authored | Positions exactly authored |
| Unselected env 1 reset preservation | Positions, velocities, previous positions and commands bitwise unchanged | Same |
| Particle positions and velocities | Finite | Finite |
| Dynamic holder displacement attributable to two-way reaction, worlds 0/1 | About `0.00852 / 0.00882` | About `0.00852 / 0.00882` |
| Pending vertical reactions, worlds 0/1 | About `-8.48 / -8.82` case force units | About `-8.48 / -8.82` |

The contact comparison uses `atol=1e-6` for unaffected-env positions; it does not claim bitwise identical independent runs. Partial reset itself is checked bitwise. Execution-mode comparison uses `rtol=1e-4, atol=1e-5`; maximum componentwise position differences were `6.71e-8` for the baseline and `5.96e-8` under contact. The dynamic two-way holder position difference between modes was `7.45e-9`. Exact values and pre-reset impulses are in `integration-results.json`. These are case units, not a physical calibration.

The dynamic fixture is a 1-unit-mass holder with a driven prismatic joint and a mounted probe. Its rigid shape collision is disabled, so the comparison isolates the custom tissue reaction. Both modes check finite body/particle state, zero pending reactions in one-way mode, nonzero opposing reactions in two-way mode, and selected reset of the pending wrench while preserving the other world's wrench.

| Fresh check | Result |
| --- | --- |
| Current-source six-module Lab/solver/component/coupling selection | **123 passed, 7 skipped**, 126.75 s |
| Current-source storage, scene storage and bounded continuation selection | **64 passed, 13 deselected**, 3.49 s |
| Bounded Lab integration harness | **Passed**, eight 60-step rollouts |
| Full pinned core regression | **1,261 passed, 51 skipped, 0 failed**, 1,240.83 s |
| Generated CPU example on pinned stack | **Passed**: generated rod/tet/contact/fluid fixture |
| Package build and distribution inspection | All six packages built as wheels and source distributions; package sources present, no environments/evidence bundled |
| Allocation inventory / private-access checker | **454 sites / 51 listed reaches**, passed |
| Kernels in changed source files versus HEAD | AST-identical; all other production files unchanged |
| Historical C4.4 and approved-reference hashes | **316 entries passed** |
| Changed-file hooks | **Passed**, including the final report |

The two new continuation tests skip on the pinned Newton revision because it has no global WORLD row; both pass on the current revision. Other core skips include unavailable Lab, rendering and XR dependencies or opt-in rendering checks; the full reasons are retained in `pinned-core.log`. The successful bounded probe does not establish full current-stack numerical or timing qualification.

Initial failures are retained: `smoke.log` contains the WORLD construction failure; `storage.log` contains the changed WORLD memory-accounting expectation. The first measurement harness also failed because it passed a body name to `scene.field`, which requires a Body object; this was a harness error, recorded in `integration.log`. The successful intermediate `integration-v2.log` sampled impulses after reset; the final harness samples them before reset. No failed observation was removed or treated as a successful run.

## Actual rigid-soft force path

[CoupledSolver](../physics/src/i4h_physics/coupling.py) implements:

1. Apply the previous completed step's pending mount wrench to rigid `State.body_f`.
2. Step MuJoCo-Warp, which leaves the i4h particles alone.
3. Read each mounted tool's pose from the rigid body.
4. Run `I4hSolver`'s ordered systems.
5. Accumulate each mount's next pending reaction.

| Path | Implemented behavior | Validation boundary |
| --- | --- | --- |
| Kinematic instrument to tissue | Custom contact/grasp constraints deform particles; no dynamic rigid response without a mounted two-way coupler | Contact deformation freshly measured |
| Mounted instrument contact to rigid body | Delayed two-way force and torque through `Mounts` and `body_f` | Dynamic contact response freshly measured through Lab, eager and captured |
| Surface Grasp, VolumetricGrasp and Hook to rigid body | These publish mass-weighted applied corrections and moments; `MountGroup.sources` includes them for the associated control and sums them into the same delayed path | Source-inspected; no new articulated dynamic-grasp demonstration |
| Catheter root/guide to rigid body | Driver publishes its root reaction into the same mounted path when enabled | Existing coupling tests; not the tissue contact demo |
| ToolForce / observables | Measurement outputs; ToolForce can apply haptic gain and cap | Reading an output alone does not apply a rigid reaction |

For each source, the pending force is `-sum(impulse)/(sdt² * substeps)` and torque is the corresponding negative moment. Here `impulse` stores accumulated mass-weighted positional corrections, not a directly measured SI impulse. The wrench is rotated to world space and the root-to-centre-of-mass lever arm is added. The coupling uses raw source products, bypassing ToolForce's gain/cap.

Thus the refactor has an implemented, demonstrated dynamic contact-reaction path. It is explicitly delayed; it is not simultaneous constraint solving or proof of stable, calibrated Versius interaction. Full-arm grasp/release, controller stiffness and timestep sensitivity, force/reward scaling, long-horizon reset behavior and RL training remain validation work. Rigid solver history is still outside i4h continuation. Native Newton coupler migration is not required by Stage C closure.

## What fits the architecture, and what stays separate

The three code lines must remain distinct:

- **Original numerical_sim:** the source of the custom mechanisms, with earlier Scene/System-owned runtime history and older lifecycle/host integration. Do not use it to replace this refactor's storage and continuation work.
- **This refactor:** the existing staged `SolverBase` implementation for custom tissue, rods, fluids, heat and surgical interactions, with Newton-owned Model/State/Control and three distinct lifecycle operations. This is the preferred reuse point.
- **v2:** native Newton mechanics extended by `SolverXPBDSoftAttachments` (and a VBD attachment variant), plus GPU tet-removal integration and separate robot/task code. Its XPBD override applies attachment constraints after native particle deltas; its optional `contact_body_wrench` is a readout restricted to a kinematic, joint-free soft-body view. That readout alone does not prove dynamic reaction transfer. This assessment did not rerun v2's behavior or checkpoints.

Proposed convergence, as subsequent work:

1. Keep v2's current solver, topology behavior, tasks and checkpoints as the baseline. Add an opt-in solver/component configuration backed by this refactor, initially on an independent tissue/contact fixture.
2. Reuse `I4hSolver` and its custom distance/volume constraints. Do not reimplement them inside `SolverXPBDSoftAttachments`, run both tissue solvers on the same particles, or introduce a callback framework in Newton XPBD.
3. Establish one owner for particle integration, tissue constraints and contact response. Preserve authored coordinates, fixed particles, masses, command units, reset ordering and data placement. The refactor requires its bound State/Control and in-place stepping; it is not automatically compatible with arbitrary native coupler ModelViews.
4. Resolve topology deliberately. v2 tet removal changes native tet activity, mass/support and boundary presentation; the refactor uses custom tet domains and its own live-surface masks rather than native mechanical tets. A small explicit bridge or a deliberately ported operation must preserve deletion region, full contact surface, selected reset and capture behavior before replacement is qualified. Replacing a solver class alone cannot do this.
5. Reuse one Lab solver/component configuration and the existing `state_products` / `is_control` declarations. `I4hComponent.state()` and command views supply reusable observation/action terms. Additional physics fields do not automatically become standard Lab assets, observations or rewards. Keep task-specific policy layout, normalization, rewards and episode semantics explicit; validate checkpoint compatibility before switching a trained task.
6. Validate articulated dynamic contact and grasp, topology/reset, eager/captured behavior and bounded task behavior before comparing or retraining policies. Native coupler integration can follow a separate ownership/projection experiment if needed.

Replace concrete-class dependencies only when an actual alternative system needs a small data interface. That interface must specify arrays, layout, units, lifetime, update stage and reset behavior; removing `isinstance` alone is insufficient. Split detection/response only for an actual independent experiment. Keep the single ordered system list until a concrete scheduling requirement demonstrates a need for overrides. None of these changes is necessary for the bounded capability demonstrated here.

## Remaining limits and disposition

- Preserve the two explicitly deferred P2 defects: standalone binding setters bypass capture invalidation, and observable publishing retains removed systems. The probe uses fixed configuration and fixed system membership, so it does not exercise those defects or waive them.
- Preserve every Stage C discrepancy and exclusion, including `aorta_shell` impulse, accepted chole trial failures, HDF5 replay, larger-case CUDA bitwise continuation, tet-containment fresh-runtime resume, physical calibration and external-force/contact-export follow-ups.
- No current-source frozen reference campaign, timing campaign, full current-source core suite, clean dependency-lock qualification, rendering/haptics/XR qualification, Versius RL training or v2 migration was performed.
- The required checks for this bounded assessment have completed without unresolved failures. Retain the historical Stage C closure, and treat this tested compatibility patch as subsequent work. Broader deployment and v2 convergence remain separate acceptance scopes.

## Reproduction

Run from `numerical_sim/`. The ignored evidence directory must be retained; the launcher and probe are local evidence, not installed package APIs. Commands use the existing retained environments without resolving or changing their dependencies. Command JSON records contain expanded interpreter and import paths.

```bash
E=.tmp_tests/current-stack-assessment-20261116
python3 "$E/run.py" integration-final "$PWD/$E/integration_probe.py"
python3 "$E/run.py" current-host -m pytest \
  tests/runtime/test_lab_adapters.py tests/runtime/test_solver.py \
  tests/runtime/test_component.py tests/runtime/test_isaaclab_env.py \
  tests/runtime/test_manager_env.py tests/runtime/test_coupling.py \
  -q -p no:cacheprovider --basetemp=/tmp/i4h-current-host
python3 "$E/run.py" storage-final -m pytest \
  tests/packages/test_storage.py tests/packages/test_scene_storage.py \
  tests/packages/test_continuation.py \
  -k 'not chole and not catheter and not pool_rows' \
  -q -p no:cacheprovider --basetemp=/tmp/i4h-current-storage-final
python3 "$E/pinned_run.py" core pinned-core -m pytest \
  tests/runtime tests/packages -q -ra -p no:cacheprovider \
  --basetemp=/tmp/i4h-pinned-core-assessment
python3 "$E/pinned_run.py" core pinned-example examples/core_synthetic.py --device cpu
uv build --all-packages --out-dir "$E/dist"
python3 scripts/check_stage_c_inventory.py
python3 scripts/check_private_reaches.py
```

From the repository root:

```bash
pre-commit run --files $(git diff --name-only) numerical_sim/docs/current-stack-assessment-20261116.md
```

`tested-source.json` identifies the nine changed source/test/checker/inventory files; their hashes were unchanged after testing and final hooks. `after.json` confirms unchanged revisions, statuses and local file hashes for Lab, Newton, original numerical_sim and v2. `assessment-evidence.sha256` hashes the local assessment receipts, harnesses and logs. The 316 historical certificate/reference hashes were checked again after the regression. No commit, push, tag or v2 migration was performed.
