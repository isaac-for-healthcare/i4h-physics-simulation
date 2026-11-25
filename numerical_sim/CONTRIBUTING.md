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

Application rendering, input, haptics, UI, recording and XR contribution guidance
lives in the [standalone repository](https://github.com/isaac-for-healthcare/i4h-standalone-simulator).

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
- **Renderers are not systems.** Hosts read completed physics data between steps;
  rendering and device I/O stay outside CUDA capture. See the standalone repository
  for its threading, stream and presentation conventions.
- **Isaac Lab drives the loop.** The simulator owns no loop, thread or window.
  `Stepper.step()` advances one standalone physics sample; in Isaac Lab the native manager
  advances physics and owns decimation. `reset(env_ids)` resets selected envs; per-env inputs
  and outputs are device arrays viewable with `wp.to_torch`; device is a constructor
  parameter. Every system implements masked `reset`.
- **Systems do not reach into other systems.** Shared data lives in `Scene`. A system
  that needs another system's product (a BVH, contact records, instrument spheres, the
  live surface) names the field in `refs`; the loader resolves the YAML id to the
  instance and the system reads a public attribute. A system that works on one asset
  takes `body:` and uploads the host records the loader attached to that `Body`
  (`TetMesh`, `ConnectorSet`, `PinSet`, `CentrelineSet`, `RodMesh`, `ShellMesh`, `CentrelineTree`,
  `SkinMesh`; domain records are attached by their owning loaders; `ParticleRest` is uploaded by
  `Scene.allocate()`). State and Control arrays are Newton's: a system declares them in
  `declare_storage` before finalize and binds views; it allocates only scratch, outputs and tables. Topology changes go through the
  `surface` system: it owns the live per-env triangles, support counters; appearance belongs to the surgical sidecar, and `deactivate_tet` is the one way to remove a tet. No string registries, no
  `__getattr__` forwarding.
- **Three operations, kept apart.** Episode reset (`restore_episode`, `Stepper.reset`) restores authored state,
  history and command defaults of the selected worlds. Cache reset (`I4hSolver.reset(flags=0)`) refreshes derived
  caches and keeps history. Continuation (`save_i4h`, `restore_i4h`) copies State and Control rows between containers
  and runs neither. A system with a search tree built at binding implements `rebuild_caches`.
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
  indexing leak would produce; checks within one run (masked reset) stay bitwise. Preserve raw qualification evidence. Historical manifests and reports are linked from
  [the documentation index](DOCUMENTATION.md#historical-reports-and-audit-tools). Commit, push and tag only when requested.
- Keep haptics (`follou`), Slang (`slangpy`), OVRTX and Isaac Lab imports lazy so
  headless runs need only `warp`, `newton` and `numpy`.
- No hard-coded developer paths in package sources. Case YAMLs may use `${DATA}` which
  resolves to explicit `--data`, then `I4H_DATA`; packages ship no case assets.
  Production assets remain external; never discover a checkout or substitute geometry.
