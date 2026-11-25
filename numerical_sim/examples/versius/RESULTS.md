# Versius integration results — 2026-11-17

The two profiles passed the required software/mechanical checks below on this
machine. Physical input-device operation is **unverified**: neither `libHD.so` nor
`follou_sdk` was available in the example environment. No convergence or learned
policy success claim is made. No commit, push, tag, v2 edit or anatomy rescaling
was performed.

## Evidence and environment

All local evidence is under `numerical_sim/.tmp_tests/versius/`. Commands below
are relative to `numerical_sim/`. `stack.json` records actual dependency imports,
source revisions and dirty-file hashes. Every scene directory contains
`before.json`, `after.json`, `scene.json` and `result.json`; failed launches retain
`failure.json` where the final exception wrapper was present. Later commands have
expanded interpreter/import-path/exit-code receipts under `commands/`.

- Lab main: `a9c1eb5de328c61e285532d72cb6164b9d0c16f0` (existing local edits retained).
- Newton soft-solvers: `ba27cbe6caad3ed6d852469658031e1990c12037` (existing local edits retained).
- Retained Warp 1.17 binary SHA-256:
  `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.
- RSL-RL `5.5.1`, MuJoCo/MuJoCo-Warp `3.17.0`, Python `3.12.12`.
- Imported Torch `2.11.0+cu130`. GPU: RTX PRO 6000 Blackwell workstation.

Before/after comparisons confirmed unchanged Lab/Newton source hashes, referenced
USD/anatomy asset hashes and Warp binary hash. The reviewed WORLD-row patch's
eight source/test/checker hashes still match its original `tested-source.json`.
The live allocation inventory was refreshed only for the separate PPO helper's
source hash and five shifted line numbers; ownership classifications remain
unchanged. Frozen Stage C evidence and its numerical limitations remain intact.

The additional six-line `i4h_isaaclab.agents.runner_dict` compatibility change
uses current configs directly when Lab's removed migration helper is absent.
The existing migration path is retained when that helper exists. This example's
PPO config is task-local.

## Construction and scripted task gates

The robot is a real articulation driven by native implicit actuators and
Jacobian IK. The component retains all authored chole geometry/material/support,
connector, boundary and static-contact settings. Unit tests compare those entries
to the original case. Contact-free settling is checked on every startup step.
Both clear and settled approach IK errors must be at most `0.005`.

| Profile | Eager, one world | Graph, replicated worlds | Final graph training fixture |
| --- | --- | --- | --- |
| Block | 10/10 success | 12/12, four worlds | 16/16, eight worlds |
| Original chole | 10/10 success | 10/10, two worlds | 10/10, two worlds |

The final fixtures are `final-block/` and `final-chole/`; their preserved
`scripted-gate.json` includes task metadata. Earlier evidence is in
`block-eager-v3/`, `block-graph4/`, `chole-eager/` and `chole-graph2/`.
`trajectory.json` records measured indentation, actual TCP and contact diagnostics.
Success thresholds, hold duration and episode length were unchanged.

```bash
python3 examples/versius/run.py scripted --scene block --envs 8 --graph --episodes 16 --output .tmp_tests/versius/final-block
python3 examples/versius/run.py scripted --scene chole --envs 2 --graph --episodes 10 --output .tmp_tests/versius/final-chole
```

The arm remains reachable after original-chole settling. Example baseline patch
coordinates were approximately `(0.469, -0.099, -0.0333)` in the one-world scene;
its original height near `0.2` changed under authored mechanics. This is a saved
settled baseline, not an assumption of zero ongoing tissue motion. In the
160-step no-contact probe the chole patch moved up to about `0.00068`, below the
success indentation band. Measurements remain in simulation units.

## Mechanical response, isolation and numerical drift

The `mechanics` workflow runs the same 160-step Cartesian commands: approach,
hold and withdrawal. Two-way, one-way and repeat runs use the same commands.
`compare.py` checks an arm response above three times measured repeat noise,
positive opposing hold reaction and the block's trajectory tolerance. It writes
`comparisons.json`.

| Metric, simulation units | Block | Original chole |
| --- | --- | --- |
| Peak measured indentation, eager two-way | 0.008418 | 0.005827 |
| Peak reaction magnitude | 2.493 | 33.682 |
| Median opposing world-Z reaction during hold | +2.440 | +25.012 |
| Maximum two-way / one-way TCP response | 0.00081697 | 0.00917612 |
| Maximum repeat-run TCP noise | 0.000000766 | 0.00005778 |
| Maximum eager / graph TCP drift | 0.000001055 | 0.00006325 |
| Maximum eager / graph patch-depth drift | 0.000000566 | 0.00014308 |

Both profiles had 91 contact steps in the two-way mechanical run, followed by
withdrawal. The reaction opposes the tool's downward movement. Block eager/graph
TCP trajectories pass `rtol=1e-4, atol=1e-5`. Chole is assessed through measured
contact/deformation/task metrics; **bitwise equivalence is not claimed**.

Four-world block and two-world chole contact/baseline comparisons recorded zero
contact force/count in every untouched peer. Independent-rollout peer particle
position differences were `4.47e-8` for block and `0.003856` for chole; the latter's
patch difference was `0.0001721`. The chole full-tissue difference is retained as
numerical rollout drift, not described as bitwise isolation. Selected reset,
which compares the same live allocation before/after without advancing physics,
preserves peer state bitwise as described below.

Mechanical evidence directories: `mech-block-{two,one,repeat,graph}/`,
`mech-chole-{eager,one,repeat,graph}/`, `ownership-block-{contact,baseline}/` and
`ownership-chole-baseline/`.

```bash
python3 examples/versius/run.py mechanics --scene block --coupling one_way --output .tmp_tests/versius/mech-block-one
python3 examples/versius/run.py mechanics --scene chole --envs 2 --graph --output .tmp_tests/versius/mech-chole-graph
python3 examples/versius/run.py mechanics --scene chole --envs 2 --graph --no-contact --output .tmp_tests/versius/ownership-chole-baseline
python3 examples/versius/run.py python examples/versius/compare.py .tmp_tests/versius
```

## Lifecycle and task semantics

`final-block-lifecycle/` (four worlds, captured) and `final-chole-lifecycle/`
(two worlds, captured) pass selected reset, hard rebuild and stale binding checks.
Every world first acquires nonzero pending contact reaction. Resetting world 0
preserves each peer's positions, velocities, previous positions, public state
products, tool commands, robot state, Cartesian commands and task/controller
history; pending reactions also remain bitwise unchanged. Repeated restoration
is idempotent. A hard rebuild creates fresh robot/tissue checkpoints and permits
continued finite stepping. Old `PhysicsBinding` operations reject use.

Raw Torch aliases and the low-level `CommandViews` array container do not revoke
already exposed memory when a binding is invalidated. The example reacquires
views through the current checked component/binding; callers must do likewise.
The stale-handle gate is tested through the checked binding API, not by expecting
an existing tensor to become unusable.

Unit tests cover measured-deformation success, no success from target depth
alone, hold-counter idempotence under repeated observation/reward/termination
reads, failure precedence, nonfinite state, actual TCP distance, timeout behavior
and terminal-state PPO bootstrapping. Repository host tests also exercise terminal
observations across automatic reset.

```bash
python3 examples/versius/run.py lifecycle --scene block --envs 4 --graph --output .tmp_tests/versius/final-block-lifecycle
python3 examples/versius/run.py lifecycle --scene chole --envs 2 --graph --output .tmp_tests/versius/final-chole-lifecycle
python3 examples/versius/run.py python -m pytest examples/versius/tests -q
```

## PPO, playback and teleoperation

Both final graph fixtures completed **five PPO updates**, with eight block or
two chole environments. The algorithm asserts finite losses and parameters.
Policies changed parameters, saved and reloaded new checkpoints, and completed
200-step playback with finite observations/actions. Separate-process playback
also passed for each checkpoint. Metadata sidecars reject incompatible profiles,
component descriptions, observation layouts and control settings. These are
smoke-test policies, not trained success-rate results.

- `final-block/i4h_solver_versius_block_24obs.pt`
- `final-chole/i4h_solver_versius_chole_24obs.pt`

```bash
python3 examples/versius/run.py train --scene block --envs 8 --graph --updates 5 --steps 200 --output .tmp_tests/versius/final-block
python3 examples/versius/run.py train --scene chole --envs 2 --graph --updates 5 --steps 200 --output .tmp_tests/versius/final-chole
python3 examples/versius/run.py play --scene block --graph --steps 200 --checkpoint .tmp_tests/versius/final-block/i4h_solver_versius_block_24obs.pt --output .tmp_tests/versius/final-block-play
python3 examples/versius/run.py play --scene chole --graph --steps 200 --checkpoint .tmp_tests/versius/final-chole/i4h_solver_versius_chole_24obs.pt --output .tmp_tests/versius/final-chole-play
```

Keyboard GL startup/stepping and a 190-step replay passed. Replay exceeded the
ordinary five-second task horizon without timeout reset. Input tests exercise
keyboard motion, clutch, focus loss, reset, invalid samples and recovery without
jumps. Follou and HDAPI adapters pass simulated-input tests, including force-output
disabling and cleanup. Physical devices were unavailable and remain unverified.
The sample replay contains tracking loss and an intentionally discontinuous
recovery position to exercise re-anchoring.

`block-gl-v4.mp4` is a visually inspected 1280×720, 160-frame, 5.33-second Newton
OpenGL recording. It shows the native arm, closed tool, explicitly transformed
tissue mesh, target marker and cyan contact proxy. Native visual geometry is
explicitly imported even though no Lab visualizer is configured. Kit/RTX,
force feedback and full shaft/jaw tissue collision remain outside scope.

## Repository and package checks

| Check | Result |
| --- | --- |
| Current-source repository/example selection | 198 passed, 2 skipped |
| Final example-specific suite | 26 passed |
| Existing direct/manager PPO smoke selection after helper fix | 5 passed |
| Generated CPU rod/tet/contact/fluid example | Passed |
| Six core package wheel/sdist builds | Passed |
| Separate example wheel/sdist build | Passed |
| Live allocation inventory | 454 sites / 192 source files; passed |
| Private-access checker | 51 listed reaches; passed |
| Third-party lock dry run | 95 packages checked; no changes needed |
| Changed-file hooks | Passed |

The two skips are Lab-3.0-specific manager-name tests on current Lab main;
software acceptance does not imply physical-device qualification. The full frozen
Stage C numerical campaign was not rerun or reclassified.

These historical checkpoint commands are retained as evidence, not current instructions.

```bash
python3 examples/versius/run.py python -m pytest examples/versius/tests tests/runtime/test_component.py tests/runtime/test_manager_env.py tests/runtime/test_isaaclab_env.py tests/runtime/test_coupling.py tests/runtime/test_solver.py tests/runtime/test_lab_adapters.py tests/packages/test_storage.py tests/packages/test_scene_storage.py -q
python3 examples/versius/run.py python examples/core_synthetic.py --device cpu
# Historical checkpoint only: the Stage C inventory checker is retired.
python3 scripts/check_private_reaches.py
uv build --all-packages --out-dir .tmp_tests/versius/dist
uv build --directory examples/versius --out-dir ../../.tmp_tests/versius/example-dist
```

From the repository root, run hooks on the new example, `agents.py` and live
inventory. Build outputs were inspected for expected package code and exclusion
of environments/runtime evidence. Final source/evidence hashes and comparison
metrics are retained in `results.json` and the local evidence bundle.

## Retained development failures

Early runs exposed and corrected a float64 target assignment, a redundant
startup `sim.forward()` that disrupted iterative IK, and an example-side WXYZ
conversion despite current Lab's XYZW convention. The first mechanical run
therefore deformed tissue without meeting the task distance criterion; it was
not counted as a passing gate.

Initial GL attempts exposed an unsupported scalar-color tuple, early pyglet
headless initialization, and missing native visuals when no Lab renderer was
requested. The final native-geometry recording was inspected after fixing these.
A bootstrap script initially named `setup.py` conflicted with setuptools and was
renamed `bootstrap.py` before the successful package build. Early hooks required
import/style corrections. The first repository regression had two failures due
to Lab's removed config migration helper; all affected PPO tests pass after the
narrow compatibility fallback. An initial stale-array assertion was replaced
with the documented checked-binding test; raw alias limitations are recorded
above. A diagnostic initially queried the historical singular `graph` property; current
Lab uses a `graphs` mapping. The corrected check observed actual graph execution
and confirmed eager replay remained uncaptured. Development logs are retained
in the local evidence directory.

## Local Follou SDK follow-up

The local package `/home/pkorzeniowsk/Projects/i4h/follou-sdk` was subsequently
supplied and verified with `uv run --with` through the launcher. Both the SDK's
`MiniMou` adapter and Torch import successfully in uv's temporary environment.
The launcher now preserves that active environment. This import check did not
open or move a physical device; physical operation remains unverified.

## Live status UI follow-up

Added an always-visible task panel using Newton's public UI callback API and a
pinned example-local `imgui-bundle==1.93.0` dependency. It shows actual component
contact diagnostics, measured goal checks and hold history without changing
physics, reset ordering or policy observations. Added close-up/overview camera
controls and explicit input pause/reset buttons. Physics pause diagnostics now
name nonfinite state, excessive tissue indentation or excessive tool depth and
print only on transition into the paused state.

The example suite passes **29 tests**, including read-only status snapshots,
correct failure reasons and one-message frozen-state behavior. A rendered eager
block episode and captured block/chole episodes passed scripted success. GUI
recordings were rendered with `PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl` while
keeping the viewer GUI enabled; image inspection verified the contact guidance,
close-up view and red physics-fault reason. A downward replay stopped physics at
step 27 after measured indentation reached about `0.01236`, while the UI continued
rendering through frame 100. It logged the exact failure once.

Evidence: `.tmp_tests/versius/status-followup/`, `status-ui-v3/` and
`status-ui-v3.mp4`. The first renderer check exposed the missing optional ImGui
package; after installing it in the task environment, an EGL UI check also
required selecting PyOpenGL's EGL platform explicitly. No external Newton/Lab
source or retained qualification environment was changed.

## Reference life-size anatomy follow-up

Added `chole-1x` from `life-size-chole` revision
`298f6e0b632541519d807b624b51b433a9df3769`. The reference worktree remains clean and
unchanged. The bundled YAML is byte-identical to its `chole_tet_med_1x.yaml`;
`life_size_reference.json` records that file and the relevant loader/solver hashes.
The existing original-scale `chole` and synthetic `block` profiles are preserved.

Minimal supporting options were ported into the current Newton-owned-storage
implementation: tet-bundle geometry scale (including connector rest lengths) and
`Tet.volume_epsilon`, with backward-compatible defaults of 1 and `1e-8`.
No legacy solver/storage implementation was imported. The scaled cutoff
`4.16493e-12` preserves exactly the original volume-constraint eligibility mask.
The live allocation census remains 454 sites / 192 source files; only source
hashes and shifted line references require refresh, with ownership unchanged.

Rest geometry extents in component coordinates:

| Part | X | Y | Z |
| --- | --- | --- | --- |
| Liver fragment | 0.260901 | 0.159695 | 0.177277 |
| Fat | 0.025165 | 0.078418 | 0.066698 |
| Gallbladder | 0.027257 | 0.070993 | 0.078986 |

Tests verify 1/7 vertex/radius/cavity/connector scaling, identical connectivity,
pins, masses and material arrays, and reference-specific boundary/contact settings.
The reference adds liver/cavity contact, sets tissue velocity to 0.5, and keeps
masses/materials unchanged. The native Versius/tool asset remains at its original
size. This is the reference's approximate life-size geometry, not a calibrated
physical material model.

Validation evidence is under `.tmp_tests/versius/life-size/`:

- Focused loader, tet PBD/XPBD, contact, storage, solver, coupling, component and
  example regression: **143 passed**.
- Two-world captured selected reset, nonzero pending reaction preservation, hard
  rebuild and stale binding rejection: passed.
- Open-loop approach/hold/withdrawal: finite state, 92 contact steps, peak measured
  indentation about 0.000331 and peak reaction magnitude about 17.09.
- Original block and chole captured scripted regressions: one success each.
- **1x scripted poke gate failed**: eager one-world and captured two-world runs
  each recorded 0/10 successes and 10 failures. Maximum recorded indentation was
  about 0.000374 / 0.000376, below the unchanged 0.0015–0.0045 success band.
  No material retuning, altered thresholds or one-way substitution was used to
  obtain a passing gate. PPO for this profile remains blocked by that gate.
- The UI labels the geometry scale and uncalibrated mechanics. Its initial panel
  placement is clamped to the current window on appearance, avoiding a saved
  layout leaving the status panel offscreen after changing viewport size.

```bash
python3 examples/versius/run.py construct --scene chole-1x --output .tmp_tests/versius/life-size/construct
python3 examples/versius/run.py scripted --scene chole-1x --episodes 10 --output .tmp_tests/versius/life-size/eager
python3 examples/versius/run.py scripted --scene chole-1x --envs 2 --graph --episodes 10 --output .tmp_tests/versius/life-size/graph2
python3 examples/versius/run.py lifecycle --scene chole-1x --envs 2 --graph --output .tmp_tests/versius/life-size/lifecycle
```

The failed scripted runs are retained. Geometry/teleoperation support does not
constitute 1x RL-task qualification. The reference's unchanged particle masses and
PBD coefficients must not be described as physically calibrated merely because
the geometry is smaller. A diagnostic size-report command initially used NumPy's
removed array `.ptp()` method; its corrected `np.ptp()` output is retained too.
