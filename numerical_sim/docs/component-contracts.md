# Component contracts: i4h physics inside a host Newton scene

These contracts bind the work that makes i4h physics a component of a host scene (an Isaac Lab / workflow scene with robots):

- **F1:** particle range and frames
- **F2:** coupled stepping, attachments, reactions
- **F3:** component handle

They come from the feasibility spike recorded at the end, and every test of those items checks against them.

## 1. State ownership

**Particles.**

- The component owns one contiguous particle range per world: `[base + e*S + col, base + e*S + col + W)`, where
  - `W` is our particle count per world;
  - `S` is the host's per-world particle stride;
  - `col` is our column inside the world;
  - `base` is the count of host global particles in front.
- The range is recorded when our particles are added in the per-world builder hook, and checked against `model.particle_world_start` after finalize.
- Worlds with different host particle counts are rejected.
- Only `I4hSolver` integrates our range. Host particles and global particles belong to the host's solvers.

**Rigid solvers.**

- **MuJoCo-Warp** does not read or write particle state, so it is the rigid solver the coupled manager supports.
- **Featherstone is rejected with an explicit error.** It integrates every active particle, and it also evaluates particle–particle and particle–body contact forces, writing the latter into `body_f`. Running it beside `I4hSolver` would integrate our particles twice and push the robot from our particles outside the coupling.
- Particles are integrated **exactly once per physics step**, which a gravity-only regression test checks.

**Buffers.**

- The coupled manager runs with `_use_single_state = True`: per substep, the rigid solver steps `state_0` in place, then `I4hSolver` steps `state_0` in place.
- Lab never swaps buffers, so every reference we hold stays valid through odd and even substep counts, graph capture and replay, and partial resets: system arrays, `scene` views, torch views, captured step and reset graphs.
- The spike showed in-place and double-buffered rigid steps give bitwise-identical bodies for MuJoCo-Warp and Featherstone. It also showed MuJoCo-Warp never writes particles to the output buffer, so a double-buffered schedule would lose our particles at the next swap.

**Rebuild.**

- A hard reset (`sim.reset(soft=False)`) finalizes a new model and states and rebuilds both solvers.
- The previous binding is invalidated (`PhysicsBinding`), so methods on that binding raise when used. Previously returned Torch tensors cannot detect invalidation; discard them.
- The component handle re-fetches its views on its rebuild callback; consumers read views through the handle, not cached arrays.

**Reset.**

- A masked reset resets our rows (positions, velocities, `q_prev`, flags, system products) and the host rigid solver's selected worlds in one call.
- An empty selection changes nothing.

## 2. Coordinate frames

- **Component frame.** Our particles, and every spatial setting of the case, live in the component frame: the case's coordinates. It is the same in every world. Spatial settings include rod frames, the catheter track, instrument initial poses, trocars and bind matrices, static and collision meshes, and cameras and tip poses. Nothing inside the component is transformed per env, so all of it stays consistent by construction.
- **World placement.** Each world has a transform `T_e = env_origin_e ∘ pose`: `env_origin_e` is Lab's per-env origin and orientation, and `pose` is the component's placement in the env.
- **Boundaries.** Only quantities crossing the boundary are transformed:
  - **Attachments (rigid → us):** the mounted body's world pose `body_q[b]` maps to the component frame as `T_e⁻¹ · body_q[b] · mount`.
  - **Reactions (us → rigid):** a force or torque computed in the component frame is rotated by `R_e` into the world frame before it is written to `body_f`.
  - **Gravity:** the component sees `R_eᵀ g_world,e`, per world. Gravity is no longer aliased to `model.gravity`.
  - **Host-side consumers of our particle positions** (Lab visualizers, a host renderer) apply `T_e`. Our own tiled camera renders in the component frame.
- **Observations and commands** are in the component frame. Instrument position commands are component-frame poses; catheter rates are frame-free.

## 3. Coupling schedule (two-way)

- **Timestep:** the coupling happens once per Lab physics step. Lab runs with `num_substeps = 1`, and the case's own substeps run inside `I4hSolver.step`. Coupling inside our substeps would change the frame structure the systems rely on (instrument pose updates, catheter target integration, step-averaged products), and with it the standalone physics.
- **Order inside one physics step:**
  1. Lab clears forces (end of the previous step).
  2. The coupling wrench is written to `body_f`.
  3. The rigid solver steps in place.
  4. The attachment poses are read from the new `body_q`.
  5. `I4hSolver` steps in place.
  6. The coupling buffer is filled.
- **Delay:** the reaction applied to the rigid body at step `k` is the one computed at step `k-1`. This one-step lag is deliberate and documented. A same-step implicit coupling would need the rigid solver's effective mass; it is not part of v0.9.
- **Buffer:**
  - A coupling wrench per mount (force and torque), separate from `ToolForce`.
  - `ToolForce` remains the step-averaged, capped haptic output and is never used for coupling.
  - The buffer is filled after `I4hSolver` steps, from the frame-averaged products of every tool system (contact, both grasps, hook), and cleared when it is written to `body_f`, or when its world is reset.
  - It has no gains and no cap.
  - Torque needs the point of application, which `impulse` loses, so each of those systems also publishes `moment`: `Σ (x_i − root) × m_i Δx_i` about the tool root, under the same conditions as `impulse`. It writes no particle state.
- **Units and sign:**
  - Numerical forces and moments use the case's mass, time and coordinate scales. The previous SI (N, N·m) description is conditional on a consistently SI-scaled case; it does not establish physical calibration for shipped assets.
  - The force is the force *on the rigid body*.
  - Torque is about the body's centre of mass, Newton's `body_f` convention: `τ = τ_mount + (p_mount − p_com) × f`.
- **Contributions:** each mount sums the constraint impulses that its own attachment applies to the deformable (`Σ m·Δx / dt²` over the nodes the mount drives) once:
  - the catheter root, or the instrument capsules for contact;
  - the guide or track;
  - wall contact. This reaches the arm only through the root or instrument constraint, never as a separate term.

**Mounts (F2b).**

- `I4hEnvCfg.mounts` / `I4hSolverCfg.mounts`: `{instrument: {"body": label, "offset": ((x, y, z), (qx, qy, qz, qw))}}`.
  - The label names one body per world: the body's label or its last path element.
  - `offset` is the instrument root's pose in the body frame.
- A mounted instrument's pose is `T_e⁻¹ · body_q · offset`, read after the rigid step. Its pose actions, velocity limit and trocar are ignored; grip, energy and fire still come from the actions.
- A catheter is mounted by its rod body's name. The mount frame replaces the rest root frame: the track starts at the mount's position, the root's rest orientation is the mount's rotation, and the track direction is the rest tangent carried into it. A mount at the rest root frame drives as unmounted, up to float rounding. Insertion, rotation and tip bend still come from the actions.
- The catheter's reaction is the root's hold plus the guide's snaps:
  - **Root hold.** With `rod_direct`, the root's hold is the negated stretch multipliers of the first edge: the impulse the rod's constraints would give the locked root. `rod_pbd` has no multipliers, so `two_way` with it is rejected.
  - **Guide snaps.** The guide's contribution is `Σ m·Δx` of its snaps, with their moment about the root.
  - **The root's bending and twisting moment:** the rotational Jacobian contraction of the first edge's multipliers, as the reference's `proximal_wrench` does. It is meaningful only with `rod_direct.inertia: physical`: with the unit inertia, a light rod's frames carry its weight's torque instead of the root (F5, docs/catheter-parity.md).
  - **Checks.** The tests check the hold against the weight of a cantilever, and the PBD momentum identity step by step.
- `one_way` drives the mounts only; `two_way` also writes the reactions.

## 3b. Component handle and composition (F3)

**Handle.** `i4h_isaaclab.component`: `I4hComponentCfg` holds assets, systems, instruments, `sim`, `pose`, explicit `loaders`/`systems_registry`, `mounts` and `coupling_mode`. `I4hComponent` uses it to:

- `configure(env_cfg.sim)`: the host's `MJWarpSolverCfg` becomes the rigid solver;
- `construction(device, envs)`: wraps the host env's construction;
- serve views and commands from the current binding: `particle_range`, `positions`, `state`, `action_bounds`, `write_actions`, `reset`, `sync_views`.

**Rebuild.** A hard rebuild rebinds the handle and runs its `on_rebuild` callbacks. The old binding raises when used.

**Hosts without replicated assets.** A host scene with no replicated assets of its own replicates an empty env root, as `I4hEnv` does; otherwise no world hook runs.

**Declarations.** Systems declare what a host reads, and `views` names no system class:

- `is_control` with `action_bounds(workspace)`, and optionally `recorded_actions(base)`;
- `state_products`.

**Explicit maps.** Either map (`loaders`, `systems`) replaces installed-domain discovery: the physics package's types plus exactly the given ones.

**Pose.** `pose` composes with Lab's env frame into `T_e`.

## 3c. Supported GPU state access

For a hosted component, call `component.sync_views()` and then
`component.positions("aorta")` (replace `aorta` with the case's body name).
The result is a zero-copy `torch.float32` tensor of shape `(E, 3*n)` on the
component's device, normally `cuda:<ordinal>`. Use `.view(E, n, 3)` for xyz.
`component.positions(body, part)` selects a named part. Omit the body only when
there is exactly one integrated body. No implicit concatenation or world transform
is performed. `component.state()` maps `<system id>.<product>` to `(E, k)` views;
each product retains its declared dtype, including integer counters.

For a standalone `Stepper`, use `sim.scene.field("q", sim.scene.body("aorta"))`:
a Warp `vec3` array of shape `(E, n)` on `sim.scene.device`.
`wp.to_torch(...)` exposes `(E, n, 3)` float32 without a host copy. `qd` uses the
same layout for velocities. `n` is capacity; mesh variants may have inactive spare
particles. Use the ACTIVE bit (`flags & 1`) of the body-selected `flags` field
(an `(E, n)` int32 array) when consuming variable topology.

Positions are in the **component frame and case-coordinate units**. Apply `T_e`
only when exporting to host world coordinates. A shell body's positions are shell
particles. A centreline body's positions are centreline nodes, **not the skinned
surface-mesh vertices**. The render surface is a separate derived skin product;
its topology and vertex count need not match the physics body's view. A static
mesh is not an integrated body's particle observation.

Stepping and masked reset update the current allocation in place. A reset updates
only selected environments; an empty selection changes nothing. The manager's
`after_step` orders Torch's current stream after Warp. Outside that callback,
including after reset, call `component.sync_views()` before reading with Torch.
It is a device-stream dependency, not a host synchronization. A standalone consumer
must order its own CUDA stream after the producing Warp stream (or synchronize that
stream before reading). Do not read concurrently with the physics worker. Clone a
view after synchronization if a stable snapshot is needed.

A hard rebuild invalidates the old binding and allocates new state. Reacquire
positions and state from the rebound component, optionally in `on_rebuild`.
Old raw tensors do not raise automatically and must not be reused. Closing its
host likewise ends the supported lifetime of these views.

### Physical-unit evidence and calibration

The source contains conflicting physical-unit descriptions: `aorta_centreline.yaml`
says the source mesh is in mm, uses `scale: 0.01`, and describes one scene unit as
about 10 cm. `aorta_static.yaml` loads the same vessel at `scale: 0.01` but labels
its coordinates as metres and its 192-node, 0.05-unit-segment catheter as 9.55 m.
`docs/xr.md` also describes the chole XR scale as one scene unit per metre.
The loader applies configured scale/rotation/translation numerically. These
case-specific comments do not establish a common physical conversion across assets
or settle QA's physical-unit concern. Report **case-coordinate units**; no definitive
millimetre conversion, geometry rescaling or mechanics change is made for rc3.
Regional coefficients in `instrument_regions.yaml` and `vascular_templates.yaml`
are uncalibrated numerical defaults. No physical calibration reference has yet been
selected; calibration and its acceptance criteria remain owner decisions.

## 4. Parity tolerances (catheter, F5)

These bounds are fixed before any candidate is evaluated. `L` is the catheter length; all metrics are taken over the scenario's trajectory after warm-up.

| Scenario | Metric | Bound |
| --- | --- | --- |
| Cantilever sag under gravity | tip position RMS and max | ≤ 1% L and ≤ 2% L |
| Commanded tip bend (including the polyline-turn mapping) | curvature along the rod, relative L2 | ≤ 10% |
| Twist response (rotational inertia) | tip angle time series, RMS | ≤ 5% of the peak angle |
| Insertion through a straight tube and a bend | tip position RMS | ≤ 2% L |
| Root reaction, steady push | reaction magnitude, relative error | ≤ 10% |
| Refinement (each scenario, half substep) | change of every metric above | ≤ half its bound |

**Two-way balance (F2).** In steady state the arm's reaction balances minus the deformable's momentum change plus external contacts, within 5% of the reaction magnitude. Halving the physics step changes the reaction by ≤ 5%.

## 5. Dependency-check exceptions

`uv pip check` must be clean, apart from exactly these three messages, which come from Isaac Lab 6864dd1's own lockfile:

- `open3d` requires `ipywidgets>=8.0.4`;
- `numba` requires `numpy>=1.22,<2.5`;
- `cmeel-boost` requires `numpy>=2.3,<2.4`.

## 6. Spike record (2026-10-06, Lab 6864dd1, Newton 1.6.1)

A 2-world model, each world with a revolute pendulum and 5 particles, 50 steps at 1/240 s:

- **MuJoCo-Warp:** particles untouched; in-place and double-buffered `body_q` identical (max difference 0); the output buffer's particles are never written.
- **Featherstone:** particles integrated (they fall under gravity); in-place and double-buffered `body_q` identical.
- **Featherstone with `integrate_particles` disabled:** particles untouched; `body_q` identical to the default. It still evaluates particle contact forces into `particle_f`/`body_f` (`solver_featherstone.py:552-557`), which is why it is rejected.


## 7. Internal navigation commands (migration worktree)

`catheter_control.root_mode` defaults to `track`. `external_pose` takes a full
batch through `control.write_root_poses(positions, rotations)` or
`component.write_root_poses(..., control="catheter_control")`. Positions are
(E,3), rotations are xyzw (E,4), in component coordinates. NumPy, Warp float32
scalar/vector arrays and Torch float32 tensors are accepted. Device sources must
be on the component device. Torch's current producer stream is ordered before
Warp; callers using another Warp producer stream must establish its dependency.
Destination command buffers retain their identity across writes and resets.

External poses own both translation and rotation without slew or rate integration;
the first two action columns are inactive, tip steering remains active. Rigid
mount ownership and external-pose ownership are mutually exclusive. Selected
reset restores authored startup root commands. `track_path` is an immutable
component-frame polyline; `distal_free_nodes` is independent of steering tip edges.
Without these settings the existing straight guide behavior is preserved.

Rod assets accept independent `startup_positions`, `startup_frames`,
`rest_lengths` and `rest_curvature` (dimensionless Darboux vectors). Placement is
restored on reset; placement on a curve does not author material curvature.
Linear graph containment accepts per-environment `interior_deadband` (default 1)
and `interior_stiffness` (default 0). Interior pull acts on the wire only and is
not wall contact. Nonzero interior stiffness requires `spline: false`.

For the internal migration, legacy trajectory comparisons in section 4 are
**diagnostics**, not release gates. Analytic limits, coupling checks and the
existing bent-lumen bound remain gates; see `migration/internal-endoluminal.md`.
