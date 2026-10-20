# Contributing

Use Python 3.12 and the pinned workspace dependencies. Run `uv sync --locked --group dev`.
Supply production data explicitly through `I4H_DATA` or `--data`; no checkout discovery
or asset substitution is performed. Unconfigured external-case tests skip; configured
missing or broken assets fail.

Run `uv run pytest tests/runtime tests/packages -q`, the generated example with
`uv run python examples/core_synthetic.py --device cpu`, and `uv build --all-packages`.
Check import boundaries and inspect built wheels and source distributions before review.
Preserve staged execution, case keys, task IDs, numerical contracts and asset bytes.
Do not weaken assertions or tolerances to pass a failure. Keep raw qualification
results privately and summarize passed, failed, skipped and unavailable checks clearly.

## Implementation conventions

- **Simplicity over generality.** No field registries, no access declarations, no
  symbolic recorders, no signature caches, no provenance tracking, no frozen
  dataclasses for descriptions. If a helper is used once, inline it. The pinned Warp examples and Newton XPBD solvers are the style reference.
- **Kernels take explicit arrays.** Inputs and outputs are separate arguments. Never
  read-modify-write an array that another thread of the same launch reads. Atomics
  into an accumulator that is consumed by a later launch are fine. Arrays in `inputs`
  are never written; an array the owning thread reads and writes goes in `outputs`
  only, so `--debug` can check `inputs ∩ outputs = ∅`.
- **Batching is one indexing rule.** Kernels launch with `dim=(E, n)` and index a body's
  particles through its `(E, n)` views, `scene.field("q", body)[env, local]`; they never
  see the layout (the main block is world-major, `Scene.layout(body)` gives the flat
  `(offset, env_stride)` for the rare host code that builds flat index tables). A
  system's own per-env buffers are indexed `env * n + local`. `n` is a capacity: envs with a smaller mesh variant
  mark their spare elements inactive. Materials are `(E, parts)`. Immutable mesh data
  (topology, rest geometry, connectors, pins, initial masks) is stored per variant
  `(V, ...)` and read through the body's `variant_of_env`; mutable state (positions,
  active masks) is per env. Fixed-capacity per-env buffers (contacts, grasp slots, clip
  slots, deletion counters, fluid pool cursors) are `E × capacity` indexed the same
  way. One system list serves all envs; never replicate Python systems or launches
  per env.
- **Capture safety.** Inside `step()` device launches only. No `.numpy()`, no
  allocation, no host branches on device data. Topology is fixed-capacity with active
  flags, so cutting and clipping run inside the graph. Structural changes (substeps,
  iterations, `enabled`, capacities) go through `sim.prepare()`; a system lists them
  in `structural = (...)` and the runner rejects live writes to them. `prepare()` calls
  every system's `prepare(scene)` before the capture; a system whose buffers are sized
  by a structural value raises there instead of silently keeping the old size.
- **Renderers are not systems.** They read a `Snapshot` between steps (`publish(snap)`
  copies the drawn env into shared vertex buffers and runs the normal kernels on the
  renderer's stream, bracketed by the snapshot's ready and release events; `draw()`
  encodes and presents) and never run inside the graph. `publish()` with no argument
  reads the live scene on the current stream (tests, Isaac Lab's GL view). The `render:`
  key is read by the app only (`--render noop|slang|gl` overrides `render.backend`); Isaac
  Lab and headless runs ignore it. Kernels ported from the renderer follow the same
  inputs/outputs rule as the systems.
  The post chain (`render/post.py`) runs every pass every frame and a toggle only changes a
  uniform (the pass list is fixed for the surface's life); `read_frame()` is the display frame,
  `read_hdr()` the linear composite. Dataset capture reads through shared buffers and pinned
  host copies on the render stream, never `to_numpy` on a device texture per frame.
- **One thread issues unqualified Warp calls.** Warp's current stream is process-global,
  not thread-local. The physics thread keeps it (systems, `Stepper`, drivers are
  unchanged); the render thread passes `stream=` on every copy and launch and, while
  the worker runs, never calls `.numpy()`, `ScopedStream`, or allocates and frees Warp
  arrays outside the ring lock (allocations are `cudaMallocAsync` on the current stream
  and would join a capture). A cross-stream copy must be contiguous, or Warp
  synchronises the two streams.
- **Instruments are kinematic and batched.** `instruments:` entries (kind grasper,
  scissors, clip_applier, hook, suction or sphere) become one `instrument_control` system:
  actions `(E, n, 10)` = `[position, quaternion xyzw, grip, energy, fire]` per env and
  instrument (grip closes the jaw, energy is the cautery level, fire is the discrete
  trigger: a clip applier applies a clip; scissors cut while their jaw closes), poses and
  collision spheres per env, capsule primitives shared (a scissors' cutting chain has none).
  Contact, grasp, volumetric grasp, hook, thermal, cutting, clipping and contact paint read
  its products through `refs`. The `inputs:` key and `--input scripted|keyboard|devices|noop`
  belong to the app's `InstrumentDriver` (keyboard: Tab cycles the tools, Space grip, Shift
  energy, F fire, M/V/B paint; a hook's handle or Space energises it, since it has no jaw;
  in `devices` mode `inputs.pickup` lets a device put its tool down or take the nearest with a
  key, `input:` being the initial holding); Isaac Lab writes the actions array itself.
- **Standalone adapters carry UI.** `app/panel_adapters.py` supplies `Knob` rows over systems' live arrays and
  structural int/bool fields that are safe to change through `prepare()` (never a field
  whose `prepare` raises); ranges are hand-written on the adapter. The app's `Panels`
  (`app/ui.py`) binds them to slangpy's retained widgets: callbacks only record pending
  changes, `update()` applies them on the render thread before `acquire()`, every simulator
  write goes through the worker queue (live arrays assigned from host mirrors read once
  before the worker starts; structural edits batched into one `prepare()`). `render/*` owns
  the ImGui context and event gating only; it never imports `app/*`.
- **One thread does all of a device's I/O.** Haptics has two halves: the `tool_force` system publishes the
  per-tool reaction (the exact mass-weighted displacement each source applied, through the per-instrument `delta_k`
  channel and the surgical `mechanics.tool_delta.apply_average_tools` kernel); the app's servo thread (`app/haptics.py`, the app-only `haptics:` key, like
  `render:`) owns the live devices, sends the conditioned force and publishes the pose sample the physics thread's
  `InputManager.poll()` returns. The physics thread stages the drawn env's force row into pinned slots with an
  asynchronous copy and an event; the servo queries the event under the ring lock and makes no other Warp call. An
  instrument-box handle is a property of a pose device (`settings.handle: {box, channel}`, `input/avatar.py`): the box
  itself has no pose and binds no tool; it supplies grip and pedals and receives the tool's `jaw_load` as the handle force.
- **Isaac Lab drives the loop.** The simulator owns no loop, thread or window.
  `Stepper.step()` advances one physics sample; `NewtonSim.step()` calls the manager
  with its configured decimation. `reset(env_ids)` resets selected envs; per-env inputs
  and outputs are device arrays viewable with `wp.to_torch`; device is a constructor
  parameter. Every system implements masked `reset`.
- **Systems do not reach into other systems.** Shared data lives in `Scene`. A system
  that needs another system's product (a BVH, contact records, instrument spheres, the
  live surface) names the field in `refs`; the loader resolves the YAML id to the
  instance and the system reads a public attribute. A system that works on one asset
  takes `body:` and uploads the host records the loader attached to that `Body`
  (`TetMesh`, `ConnectorSet`, `PinSet`, `CentrelineSet`, `RodMesh`, `ShellMesh`, `CentrelineTree`,
  `SkinMesh`; domain records are attached by their owning loaders; `ParticleRest` is uploaded by
  `Scene.allocate()`); it owns the device arrays. Topology changes go through the
  `surface` system: it owns the live per-env triangles, support counters; appearance belongs to the surgical sidecar, and `deactivate_tet` is the one way to remove a tet. No string registries, no
  `__getattr__` forwarding.
- **Defaults live on the system class** as dataclass fields; YAML states only what
  differs. The config loader is `SYSTEMS[type](scene, **entry)` in list order and
  nothing else. Unknown keys fail through the dataclass constructor. Never author a
  YAML key before a system reads it.
- **Port, do not reinvent.** Copy kernels from `the approved omnisurg source` and
  `the approved omniendo source` verbatim, then apply the `(env, local)` prologue and
  remove `newton._src` imports. Cite the source `file:line` in a one-line comment at
  the top of each ported kernel block. Host-side wrappers (cutting, emitters, clip
  attach) are rewritten as kernels, not ported.
- **Public Newton API only** (`newton.ModelBuilder`, `newton.Model`,
  `newton.viewer.*`). No `newton._src` imports in package sources.
- **Verification is a runnable case plus a few asserts.** Each qualification step ends with a command that runs a case. Tests are short pytest files
  checking finite values, env isolation on `envs: 2`, graph replay equal to
  uncaptured stepping, and the Isaac Lab stub loop. Float atomics make runs differ at
  the 1e-7 level, so comparisons across two runs use a tolerance well below what an
  indexing leak would produce; checks within one run (masked reset) stay bitwise. Preserve raw qualification evidence and frozen manifests required by the migration plan. Commit, push and tag only when requested.
- Keep haptics (`follou`), Slang (`slangpy`), OVRTX and Isaac Lab imports lazy so
  headless runs need only `warp`, `newton` and `numpy`.
- No hard-coded developer paths in package sources. Case YAMLs may use `${DATA}` which
  resolves to explicit `--data`, then `I4H_DATA`; packages ship no case assets.
  Production assets remain external; never discover a checkout or substitute geometry.
