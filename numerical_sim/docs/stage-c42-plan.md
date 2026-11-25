# C4.2 plan: tissue, fluid and containment history

Implementer: Claude, with advisor review of the diff and the report. Starts from commit `6c2dd2b` (C4.1 closed). Approved by the user on 2026-11-10. The C4 masterplan is [stage-c4-plan.md](stage-c4-plan.md).

## Context

After C4.1, 29 State arrays are still allocated by systems. C4.2 moves 25 of them into Newton State: Thermal, surgical appearance, the fluid counters, Bleeding, Emitter, Smoke, Suction and GraphContainment. The other four belong to TipEvidence and stay host-owned by user decision. After C4.2 a checkpoint of State and Control holds heat, burns, coagulation, surface colour, fluid emission history and smoke, so C4.3 can build the public save and restore call on complete data.

User decisions (2026-11-10):

- **TipEvidence stays host-owned.** Its four arrays are reclassified as assessment state and excluded from the continuation claim, next to Procedure state. No code change.
- **GraphContainment migrates as State and `prepare` stays unchanged.** If the forced rebuild at `prepare` changes results, resume through a fresh binding is reported as not bitwise for cases with graph containment. No hook change.
- Carried over: resets keep their exact behaviour; `aorta_shell` catheter-impulse failures are retained without attribution; C4.0 timing accepted, no timing campaign; HDF5 deferred; bitwise resume claims are CPU-only, with deterministic binding, view-identity and isolation checks on CUDA.

Unchanged: every numerical kernel, every step, reset, refresh and prepare hook (AST audit against `6c2dd2b`), capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins, retained environments, historical certificates.

## What migrates

| Order | System | Arrays | Row domain | Template |
| ---: | --- | --- | --- | --- |
| 1 | Thermal (`surgical/.../instruments/thermal.py`) | `heat`, `burn` | existing `i4h:<id>_tet` domain `(E,T)`, already declared and unused | zero |
| 1 | Thermal | `coag` | the referenced surface's support rows `(E,N)`, flat `(E*N,)` | zero |
| 2 | Appearance (`surgical/.../mechanics/surface.py`) | `visual` (vec4) | same support rows, flat `(E*N,)` | zero |
| 3 | Fluid (`physics/.../fluids/fluid.py`) | `cursor`, `overflow` | new pool-slot rows `(E,nf)` per fluid system | zero |
| 4 | Bleeding | `cut`, `source`, `acc`; `serial` | centreline node rows `(E,K)`; WORLD | zero |
| 5 | Emitter | `acc`, `serial` | WORLD | zero |
| 6 | Smoke | `pos_age`, `vel_life` (vec4); `cursor`, `acc`, `serial` | new smoke rows `(E,count)`, flat; WORLD | zero |
| 7 | Suction | `removed` | Fluid's pool-slot rows | zero |
| 8 | GraphContainment (`physics/.../contact/graph_containment.py`) | `cand_idx`; `cand_count`, `cand_overflow`, `ref_samples`; `ref_tree`; `rebuild`, `built` | new sample rows `(E,n)` (`cand_idx` with `row_width = capacity`); new tree node rows `(E,Pt)`; WORLD | `cand_idx` -1, `rebuild` 1, others zero |

Out of scope, with the evidence: `heat_next`, fluid neighbour lists and the `HashGrid` (rebuilt every substep), the `FluidContact` BVH (no arrays of history; see step 7), `alive_count`, `sealed`, `sources`, `spawned`, `hot` (rewritten each step; zero after a fresh binding until the next `step_end`, stated in the report), GraphContainment `max_disp` and correction accumulators (self-clearing). `Smoke.centre` keeps its last value when there is no source; verify in step 3 whether any kernel reads it before writing. If one does, it migrates with Smoke and the inventory is corrected; otherwise it stays an output.

## Design

All fields use uniform defaults (`Declarations.add(..., default=)`), so only the new row domains add authoring cost. Kernel signatures do not change: every view keeps the shape and dtype the kernels take today.

### Thermal and appearance

- `heat`, `burn`: `Thermal.declare_storage` (`thermal.py:54`) already declares them through `declare_thermal_tet_fields` (`surgical/src/i4h_surgical/tet_storage.py:11`). Bind them with `scene.attr` after the constructor's type checks and delete the legacy allocations. This removes live memory. Switch the two declarations to uniform defaults.
- `coag`, `visual`: declared on the surface's own support domain, selected by surface id (`scene.surface_domains[<id>]["domains"][4]`), not by `support_domain(body)`, which returns the first surface on a body.
- Surgical `Surface` gets a `declare_storage` override that calls the physical one and adds `visual`. `Appearance` binds a view instead of allocating, still after `super().__post_init__()` (pinned by `tests/packages/test_surface_owned_storage.py:204`). The physical `Surface` stays appearance-free.
- Declarations go into a new section of `surgical/src/i4h_surgical/interaction_storage.py` (reuse `Layout`, `bind_interaction`); direct Scene callers get `declare_thermal` and `declare_appearance` through `defer_declaration`.
- Writers (Cutting, StretchBreaking, ContactPaint, Thermal) and readers (`isaaclab/camera/tiled.py`, `standalone/render/snapshots.py`, `slang.py`) keep reaching `surface.appearance.visual`; one backing, identity registered. Renderers hold the pointer for one binding, as they already do for triangles.

### Fluids

- New `physics/src/i4h_physics/fluid_storage.py` (host-only, public storage API): `publish_fluid` creates the `(E,nf)` pool-slot domain keyed by system id (two Fluid systems in one scene exist in `tests/packages/test_scene_storage.py`) and declares `cursor`, `overflow`. `Fluid.declare_storage` returns a record with the domain and `nf` for dependants.
- Suction declares `removed` on Fluid's domain; Bleeding declares on the centreline record's node domain and WORLD; Emitter on WORLD; Smoke publishes its `(E,count)` domain with flat vec4 fields. These live in `interaction_storage.py`. `Layout` moves from `i4h_surgical` into `physics/src/i4h_physics/storage.py` (re-exported where it is used), because Fluid and GraphContainment are physics-package systems and must not import the surgical package. It gains a flat-shape option for Smoke, `coag` and `visual`.
- `Bleeding.clipped` stays the alias of `Clipping.clipped`; `Smoke` keeps reading `Thermal.heat`, now Newton-backed, so Thermal migrates first.

### GraphContainment

- Declarations in a new `physics/src/i4h_physics/containment_storage.py`: sample rows owned by the catheter particles, tree node rows owned by the tree particles, WORLD for `rebuild` and `built`. `cand_idx` uses `row_width = capacity` so the case authors 765 rows per world instead of 48,960.
- `endoluminal/.../systems/__init__.py::compatible` copies only `refs` and `author`, so `lumen_containment` would never reach a new `declare_storage`. Extend it to forward the hook with the translated keys.
- `prepare` keeps `rebuild.fill_(1)`, as a named exception like Hook's.
- **Neutrality measurement first, on unchanged code** (CPU, `aorta_centreline`), three arms: two uninterrupted runs of N steps (control: they must agree bitwise), and one run with a `prepare()` at the midpoint. Compare particle and rod State bitwise, and check that the candidate or reference arrays do differ between arms after the midpoint, so the measurement is not vacuous. The result decides the wording of the claim, not the design. If no difference appears, the report says "no difference observed on this trajectory": the spline projection cannot be shown neutral from the code.
- The resume test for this case compares the fresh binding with the same runtime that called `prepare()` at the save point. That comparison is bitwise whatever the neutrality result.

### TipEvidence

No code change. `docs/stage-c4-inventory.{md,json}` and the allocation inventory record the four arrays as assessment state, host-owned by user decision.

## Steps

1. Save the plan as `docs/stage-c42-plan.md`, lint, commit.
2. Pre-edit audit against the C4.1 manifest; evidence directory `.tmp_tests/stage-c42-<date>/`. Still on unchanged production code: run the GraphContainment neutrality measurement; add `aorta_centreline` to the construction measurement script, commit that, and take its 1- and 64-world samples as the pre-migration reference.
3. Pre-migration pins on unchanged production code, committed separately, extending `tests/packages/interaction_helpers.py`: masked reset bytes of all 25 arrays, cache-only preparation preserving them (GraphContainment `rebuild` is the recorded exception), `Smoke.centre` read-before-write check (kernels and host readers such as the renderers), Bleeding's `cut` carried from `step_end` to the next `step_start`, RNG `serial` continuity.
4. Storage layer: move `Layout`, add its flat-shape option, and test the combinations no code uses yet (in `tests/packages/test_storage.py`): vec4 with a uniform default, a non-zero default with `row_width` (`cand_idx` = -1), vec2 on WORLD (`built`), an integer default of 1 (`rebuild`), and zero-row domains (`T == 0`, no fluid pools).
5. Migrate in table order, running each system's tests and the ownership suites after each. Update direct-Scene fixtures: `fluid_only` in `tests/runtime/test_blocks.py`, `lumen_scene` in `tests/runtime/test_centreline.py`, `scripts/parity/catheter.py`, any surgical `Surface` or Thermal built by hand.
6. New `tests/packages/test_tissue_owned_storage.py` (CPU eager, CUDA eager, captured): templates, names, pointers, shapes, owners and identities for the 25 arrays; alias identity for `visual` across its four writers and for `Smoke`'s heat input; replaced-array, late, duplicate and changed-capacity rejection; two Fluid systems; masked, empty and repeated resets.
7. Resume tests, bitwise on CPU: chole with Thermal, Bleeding, Emitter, Suction, Smoke and fluids enabled (remove them from `LATER`), with non-zero heat, burn, coag, colour, live blood and smoke at the save point, comparing Newton containers including appearance; `aorta_centreline` with graph containment, comparing according to the neutrality result. On CUDA: binding changes no byte, views are the bound State, containers are isolated, also under capture.
   If CPU fluid stepping is not repeatable (twin runtimes restored from one checkpoint disagree), the claim is narrowed, not the comparison. `FluidContact` builds its BVH in the constructor from bind-time positions and only refits afterwards (`physics/.../contact/fluid_contact.py:62`), so a fresh binding gets a different tree; this test is what shows whether that changes results.
8. Re-anchor the allocation inventory (25 `S` sites removed, TipEvidence reclassified, new view sites reviewed), update `scripts/measure_stage_c_construction.py` accounting, and add `aorta_centreline` as a construction workload, since its storage now changes.
9. Development selection in the three environments, then commit with hooks passing, freeze, and run one campaign with the C4.1 launcher, driver and auditor: full core suite, both six-module host selections, 36 groups, 48 replays, 18 chole trials, six construction samples (chole, `aorta_shell`, `aorta_centreline` at 1 and 64 worlds). No retries; failures retained; independent audit.
10. Report `docs/stage-c42.md`, status prepended to `docs/stage-c.md`, advisor review, documents-only commit, no push. Stop for review before C4.3.

## Verification

- Auditor recomputes suite, group, replay and chole counts from logs and retained arrays; no production file differs between freeze manifest and final commit.
- Construction against C4.1 (chole 64 worlds: 14.935 s, 1,361,783,246 retained bytes). Expected: templates for `visual`, `coag` and smoke rows added, legacy `heat` and `burn` removed; new rows per world about 512 smoke plus `nf` for chole and `two_worlds`, about 765 plus `Pt` for `aorta_centreline`. Every byte delta explained by the auditor.
- Kernel and hook AST audit unchanged against `6c2dd2b`.
- `pre-commit run --files $(git ls-files numerical_sim)` passes with a clean status.

## Stop conditions (report and wait)

- A failed reference group other than `aorta_shell` catheter impulse, or a chole group below 2 of 3. `two_worlds`, chole and `aorta_centreline` all hold migrated State now, so a new failure in any of them is the main risk signal.
- A suite failure other than the two known intermittents. The `two_worlds` fluid count intermittent touches migrated counters; a recurrence is retained and reported with its values.
- Chole or `aorta_centreline` construction time at 64 worlds more than 10% above the pre-migration sample.
- CPU resume differs from uninterrupted execution while twin restored runtimes agree with each other: report the first diverging array and the system that owns it (the `FluidContact` BVH is the known candidate); do not loosen the comparison or disable the system.
- A cross-step reader of data classified as scratch or output that the pins expose.
- Appearance cannot be declared without changing when `Appearance` is created relative to reference validation.

## Not in C4.2

Public save and restore, removal of the remaining private storage reaches, merging the ownership test helpers, and the bind-time BVH of Shell and Tet containment (a fresh binding builds a different tree; to be examined in C4.3).
