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
- Lab never swaps buffers, so every reference we hold stays valid through odd and even substep counts, graph capture and replay, and partial resets: system arrays, `scene` views, torch views and Lab's captured step graph. Hosted bindings do not own capture graphs.
- The spike showed in-place and double-buffered rigid steps give bitwise-identical bodies for MuJoCo-Warp and Featherstone. It also showed MuJoCo-Warp never writes particles to the output buffer, so a double-buffered schedule would lose our particles at the next swap.

**Rebuild.**

- A hard reset (`sim.reset(soft=False)`) finalizes a new model and states and rebuilds both solvers.
- The previous binding is invalidated (`PhysicsBinding`), so methods on that binding raise when used. Previously returned Torch tensors cannot detect invalidation; discard them.
- The component handle re-fetches its views on its rebuild callback; consumers read views through the handle, not cached arrays.

**Reset.**

- A masked reset resets our rows (positions, velocities, `q_prev`, flags, system products) and the host rigid solver's selected worlds in one call.
- A hosted binding may supply `prepare_episode_reset(world_mask)`. After mask selection and before numerical restoration, the Lab host evaluates current holder joints and mount poses only for selected worlds. Write or randomize the holder joints before calling the component reset. Mounted catheters restore their rest geometry and frames at that current mount pose.
- An empty selection changes nothing.
- This is the episode reset. The cache reset (`I4hSolver.reset(flags=0)`, which Lab calls after state is written through its asset API) refreshes derived caches and keeps history.

**Continuation.**

- `I4hSolver.save_i4h(dst_state, dst_control, mask=None)` and `restore_i4h(src_state, src_control, mask=None)` copy between the bound containers and another State and Control of the same Model. `PhysicsBinding.save` / `restore` and `I4hComponent.save` / `restore` forward to them and accept `env_ids`.
- **Copied:** every declared State and Control field and the scene's own rows of `particle_q`, `particle_qd` and `particle_f`. **Never written:** foreign host particles, padding rows, body and joint fields, Model fields and initialization templates.
- **Mask:** `(E+1,)` bool; the last slot selects rows shared by all worlds. `None` selects everything.
- **Rejected:** the bound containers themselves, containers of another Model or another finalize, replaced arrays, and a Control that shares its buffers with the Model templates (`model.control(clone_variables=False)`).
- **Recipe:** bind, prepare, then restore. A restore runs no reset, cache refresh or preparation. Call both at a completed step boundary, outside graph capture.
- **Side effect:** both calls rebuild the contact search trees from current positions. A runtime that saves differs at rounding level from one that never saved. With a partial mask, unselected worlds keep their State and Control bytes but may differ at rounding level in later steps.
- **Hosted order:** write the host's own state, call the simulation's `forward()`, and restore the i4h rows last. Otherwise Lab's cache reset follows the restore.
- **Not saved:** a rigid solver's own history (including MuJoCo-Warp's), host time and episode counters, assessment, procedure and tip evidence, renderer state and recordings. With a rigid solver the host restores body state itself.
- **Bitwise resume** is tested on the CPU for every case, and on CUDA for the rod and mount fixtures only; chole and the aorta cases are not repeatable on CUDA.

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
  - **The root's bending and twisting moment:** the rotational Jacobian contraction of the first edge's multipliers, as the reference's `proximal_wrench` does. It is meaningful only with `rod_direct.inertia: physical`: with the unit inertia, a light rod's frames carry its weight's torque instead of the root (F5, [catheter parity](../endoluminal/catheter-parity.md)).
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

**Procedure.** `I4hComponentCfg.procedure` (`from_case(case, procedure=True)`) makes the component own the case's
assessment.

- Every solver build constructs it with the host's `horizon_s`, set before construction to
  `max_episode_length * step_dt`.
- Main uses solver-adapter `post_step`; patched Lab 3.0 uses its solver-step bridge. Both update it once per physics step, including each iteration of a captured decimation batch. Host stream ordering runs after the batch.
- `component.reset()` stays physical only. The host calls `reset_procedure()` after its reset events, so the
  procedure samples the state those events left.
- `i4h_isaaclab.manager` builds the manager-based env on this.

## 3c. Supported GPU state access

For a hosted component, call `component.sync_views()` and then
`component.positions("aorta")` (replace `aorta` with the case's body name).
The result is a zero-copy `torch.float32` tensor of shape `(E, 3*n)` on the
component's device, normally `cuda:<ordinal>`. Use `.view(E, n, 3)` for xyz.
After explicit teleports through position views or native State arrays, call
`component.initialize_particle_history(env_ids, mask=...)` before stepping
(supply either ids or a mask). It orders Torch writes before Warp and updates
only selected owned particle history, including private pools. Velocities, flags,
topology, commands and foreign/unselected rows remain unchanged. Raw view writes
cannot be detected automatically. Direct `PhysicsBinding` callers have the same
method and must order position writes onto Warp's stream themselves.

`I4hEnv` and manager-based `I4hHostMixin` complete this operation after startup and reset events. Fresh component construction
and rebuild callbacks complete writes on the current Torch stream automatically;
callbacks writing on another stream must call the method on that stream before
returning. Binding an independent/restored State and cache-only solver reset do
not initialize history. Restore saved history after construction callbacks.

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
The [historical XR guide](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/xr.md) also describes the chole XR scale as one scene unit per metre.
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

`uv pip check` must be clean, apart from exactly these two messages, which come from Isaac Lab v3.0.0's own lockfile (6864dd1 also reported `open3d` requires `ipywidgets>=8.0.4`):

- `numba` requires `numpy>=1.22,<2.5`;
- `cmeel-boost` requires `numpy>=2.3,<2.4`.

## 6. Spike record (2026-10-06, Lab 6864dd1, Newton 1.6.1)

A 2-world model, each world with a revolute pendulum and 5 particles, 50 steps at 1/240 s:

- **MuJoCo-Warp:** particles untouched; in-place and double-buffered `body_q` identical (max difference 0); the output buffer's particles are never written.
- **Featherstone:** particles integrated (they fall under gravity); in-place and double-buffered `body_q` identical.
- **Featherstone with `integrate_particles` disabled:** particles untouched; `body_q` identical to the default. It still evaluates particle contact forces into `particle_f`/`body_f` (`solver_featherstone.py:552-557`), which is why it is rejected.

## Stage B solver observables

`I4hSolverCfg.observables` defaults to `()`. Request custom kinds from
`i4h_physics.observables.ObservableKind` before host construction. Standalone
callers allocate `sim.observables = sim.solver.observables(kinds)` before
`sim.prepare()`. Subsets are allocated with `output.select(kinds)` before
capture and share the selected arrays; omitted products are not refreshed.
Allocating another output container after preparation requires preparation again
so episode-reset capture includes it. Replacing arrays, captured containers or
source buffers is rejected. A host hard rebuild invalidates the old binding and
allocates new outputs.

| Kind / array | Row domain | Meaning |
| --- | --- | --- |
| `I4H_WRENCH` / `i4h_wrench` | world, control system, tool | `wp.spatial_vector(force, moment)`, reaction on the tool; moment about the instrument/driver root, in component coordinates and case units. Includes unmounted tools. |
| `I4H_CONTACT_DEPTH` / `i4h_contact_depth` | world, containment system | Struct `depth`, `count`: copies the existing last-substep measurement. Count semantics remain those of the source (nodes or samples pushed); they are not unique frame contacts. |
| `I4H_SURGICAL_COUNT` / `i4h_surgical_count` | world, instrument-contact system, capsule | Struct `found`, `valid=min(found,capacity)`, `capacity`, `overflow=max(found-capacity,0)`, for the final recording event of the final substep. This overflow is distinct from the source's sum over all substeps. |
| `I4H_SURGICAL_CONTACT` / `i4h_surgical_contact` | world, instrument-contact system, capsule, stored slot | Struct described below. Opt-in; request `I4H_SURGICAL_COUNT` alongside it to report truncation. |

`output.row_keys[kind]` is an immutable tuple of row identities, derived from
Model output-layout metadata registered before finalization. Tool keys are
`(world, control_system_id, tool_index, name_or_rod_system_id)`; the last field
is the instrument name or the catheter control's referenced rod-system ID. Containment keys are
`(world, system_id, 0)`. Capsule keys are
`(world, contact_system_id, capsule_index, tool_index, control_system_id,
surface_system_id, capacity)`; record keys replace the final capacity with the
stored slot. Triangle/tet/vertex IDs belong to that surface system's body.
These are i4h identities, not native Newton shape IDs. Model metadata contains
no migrated persistent physics state.

Raw wrenches sum each enabled reaction source once in authored order using
`-sum(impulse)/(substep_dt * step_dt)`, with identical moment scaling. Contact,
surface grasp, volumetric grasp and hook contribute to instruments; the catheter
control already accumulates its root hold and guide contributions. Its
`root_reaction` capability retains its existing meaning: the direct rod solver
measures the root hold; a PBD driver does not acquire a new root reaction through
this export. `ToolForce` remains the separate gain-weighted and capped haptic
product. Reading/exporting a wrench neither consumes nor clears pending coupling
reactions and does not change their one-step delay. Coupling alone rotates the
reaction into world coordinates and shifts root moment to body COM.

Surgical records contain `valid`, body-local `triangle`, `tet`, `vertices`,
`barycentric`, `normal`, `penetration`, `capsule_coordinate`,
`surface_point_completed`, and `capsule_point_completed`. The last two fields are
reconstructed from completed-step positions. They are **not points at the
recording instant**: constraints, motion and cuts can occur after recording.
Recorded IDs, normal, penetration, barycentrics and stored order are retained,
including an event whose tissue was cut later in the substep. The capsule point
is on its axis. The existing tip-capsule coverage is unchanged; wrist/shaft rows
have zero valid records. No per-contact force is invented. Invalid payload is
zeroed (IDs are `-1` during publication); only `valid` slots may be consumed.

All requested buffers are allocated before capture. The publication kernels read
physics products and write only these separate output buffers, after the existing
five-hook step. Disabled sources publish zero outputs/counts; capacities remain
layout metadata. Episode reset clears selected worlds' measurements, preserving
capacity and unselected worlds. Empty masks do nothing. Unrequested kinds have
`None` arrays; requested empty domains have zero-length arrays.

The coupled wrapper routes custom arrays to i4h and native `CONTACT_F` to
MuJoCo-Warp, including subset selection. Pure i4h continues to reject native
`CONTACT_F`. Surgical export must not be interpreted as native rigid contacts.

See [vessel regions](../endoluminal/vessel-regions.md) for the aorta branch mapping,
sidecar schema and material multiplier calculations. Territory calibration remains deferred.

## Historical Stage C acceptance and verification limits

Stage C was accepted on 2026-11-16 under the rule allowing only `aorta_shell`
catheter-impulse reference failures. The final frozen campaign passed 35/36
reference groups and 17/18 chole trials (each group at least 2/3).
That acceptance is historical; it does not verify today's source or fix the deferred
defects below. The ownership and continuation rules remain the maintained contract.
See the [closure report](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/stage-c-closure.md) and
[final checkpoint](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/stage-c44.md) for exact stack and endpoints.

## Storage ownership and lifecycle

Newton Model, State and Control hold the fields below, declared on the builder before finalize. Systems hold views of them. The table is by data family, not per system. Its chole rows were checked against the declarations of a bound `chole_tet_med` runtime; the rod and catheter row is taken from the C3.2 report.

| Data | Owner | Since |
| --- | --- | --- |
| Particle positions, velocities, forces, previous positions and live flags | State | C2.1 |
| Tet mechanics storage and topology masks | State | C2.3 |
| Surface live topology and support counters | State | C2.4 |
| Fluid pool particles | State | C2.2 |
| Instrument poses, jaw state and commands | State and Control | C3.1 |
| Rod frames, angular velocity, material tables; catheter commands | State, Model and Control | C3.2 |
| Mount flags and transforms | Model and Control | C3.3 |
| Centreline positions and counts; Hook, Grasp, VolumetricGrasp and Clipping slots; the six tool residue arrays | State | C4.1 |
| Heat, burn, coagulation, surface colour; fluid, bleeding, emitter, smoke and suction counters; graph containment candidates | State | C4.2 |
| Rest geometry, per-variant tables, parameters, initialization templates | Model | C2 onward |

Not Newton-owned, by decision or by kind:

- **TipEvidence and Procedure state** are host-owned assessment state (user decision, C4.2).
- **Scratch** rewritten before it is read inside a step: constraint deltas, contact records, containment `sides`, fluid neighbour lists and the hash grid, `heat_next`.
- **Outputs** of a completed step: impulses, moments, ToolForce force and jaw load, observables.
- **Search trees** of FluidContact, PartContact, ShellContainment and TetContainment: built at binding, refit while stepping, rebuilt by save and restore.

The [historical allocation inventory](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/stage-c-inventory.json) records the reviewed source at its checkpoint. It is not a census or verification of the current tree.

## Three reset-like operations

| Operation | Call | Guarantee |
| --- | --- | --- |
| Episode reset | `Stepper.reset`, `PhysicsBinding.reset`, `I4hSolver.restore_episode` | Selected worlds return to authored state, history and command defaults; unselected worlds are bitwise unchanged; an empty selection does nothing |
| Cache reset | `I4hSolver.reset(state, mask, flags=0)` | Derived caches are refreshed from current state; history is kept |
| Continuation | `save_i4h` / `restore_i4h` and their binding and component forms | Every declared State and Control field and the scene's own particle rows are copied between containers of one Model; no reset or preparation runs |

Each system's reset keeps its exact earlier behaviour, including what it leaves alone: stale Grasp, VolumetricGrasp and Hook slot bytes, the drag node and target, mounted flags and the instrument's current pose. Three lifecycle writes outside episode reset remain as named exceptions: Hook's disabled `prepare` zeroes `active`, GraphContainment's `prepare` forces a rebuild, and Centreline rewrites its positions and counts from the current particles at binding.

## Continuation contract

- **Recipe:** bind, prepare, then restore.
- **Hosted order:** write the host's own state, call the simulation's `forward()`, and restore the i4h rows last. Otherwise Isaac Lab's cache reset follows the restore and replaces graph containment's saved rebuild request.
- **Side effect:** save and restore rebuild the contact search trees. A runtime that saves differs at rounding level from one that never saved.
- **Masks:** a masked call leaves unselected worlds' State and Control bytes unchanged. Bitwise resume is claimed for unmasked calls only.
- **Bitwise resume, tested:** on the CPU for the rod, mount, chole, `aorta_centreline` and `aorta_shell` cases; on CUDA for the rod and mount fixtures only. Chole and both aorta cases are not repeatable on CUDA, so there the tests check copy exactness.
- **Excluded:** a rigid solver's own history (including MuJoCo-Warp's), host time and episode counters, assessment, procedure and tip evidence, renderer state, and HDF5 recordings. With a rigid solver the host restores body state itself.

## Retained failures

None of these was attributed; the user declined attribution work for the first.

| Failure | Where seen | Evidence |
| --- | --- | --- |
| `aorta_shell` catheter impulse above three times its noise floor | Zero to two of six groups per campaign (none in C4.0 and C4.2; two in C4.1 and C4.3; Lab 3.0 eager in C4.4) | exact endpoints for C3.1 to C4.4 in the C4.4 certificate |
| Chole trial failures within the 2-of-3 rule | One to four of 18 trials per campaign; four in C4.3, with eight products not seen before or since | each certificate's `failures` list |
| Clipping count intermittent (`test_tracker_and_clip`) | Earlier campaigns; did not recur in C4.2 to C4.4 | [C3.0](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/stage-c30.md) through [C4.1](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/stage-c41.md) |
| `two_worlds` fluid count intermittent | Earlier campaigns; did not recur in C4.2 to C4.4 | same |
| `test_chole_batched_matches_single` over its 0.2 limit | Once in the C4.3 development selection; single trials over the limit in C4.0 and C4.1 | [C4.3](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/docs/stage-c43.md) |

Raw certificates and approved references are in git-ignored `.tmp_tests`, separate from tracked reports. Historical hash lists are retained only in [Git](https://github.com/isaac-for-healthcare/i4h-physics-simulation/tree/44c62f1ae44241cb034f3ef194558c04c47a1572/numerical_sim/qualification/); those hashes do not publish the raw evidence.

## Unverified

- **HDF5 recording and replay**: deferred by user choice; `h5py` is absent from the retained environments.
- **Tet containment** resume through a fresh runtime: no authored case uses it.
- **CUDA bitwise resume** for chole and the aorta cases.
- **Physical calibration**: forces, lengths and rod stiffness are in case units.
- **Devices, haptics, XR and rendering** are outside every Stage C campaign.

## Known limits of the code

- **Third-party private access** remains in the Isaac Lab 3.0 bridge, the env hooks, the storage layer (Newton custom-attribute internals) and Warp array views. Each is tied to a pinned revision; `scripts/check_private_reaches.py` lists them.
- **The standalone renderer** shares private names inside its own package.
- **Construction time** at 64 worlds is about 15 s for chole, of which about 12 s is authoring custom rows.
- **Memory:** chole retains 1.36 GB of Warp memory at 64 worlds, including initialization templates.

## Deferred beyond Stage C

- **P2: standalone binding settings bypass capture invalidation** (`src/i4h_physics/bindings.py`). Setting `sim.physics.substeps`, `iterations` or `debug` writes the solver's value rather than the Stepper's. After preparing four substeps, setting `sim.physics.substeps = 5` permits stepping without recapture: a gravity-only probe produced `z=-0.006` eagerly and `z=-0.00625` captured, although both reported five substeps. Calling `prepare()` reverts the setting. Until fixed, change settings on the standalone Stepper (`sim.substeps`, `sim.iterations`, `sim.debug`) and call `sim.prepare()`.
- **P2: observable publishing retains removed systems** (`src/i4h_physics/observables.py`). The publisher caches source instances when allocated. Removing an instrument-contact system and its force consumer, then calling `prepare()` and `step()`, still exported the removed source's stale force (approximately `-14,400` in the review probe). Until fixed, reconstruct the runtime when changing system membership. The fix should reject membership changes after binding or rebuild and validate the publisher's sources.
- External `particle_f` consumption.
- Contact geometry export and `soft_contact_capacity`.
- Physical rod units.
- Native Rod and VBD mechanics, and migration to Newton's coupled solver.
- A bulk-authoring request to Newton for the 64-world construction time.
- An archive location for the references and certificates.
