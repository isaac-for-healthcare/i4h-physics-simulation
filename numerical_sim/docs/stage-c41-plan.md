# C4.1 detailed plan: interaction history and residue

Implementer: Claude (user decision, 2026-11-10), with an advisor review of the diff and the report before completion. Starts from commit `ebd86e2`. It details the second checkpoint of [the C4 masterplan](stage-c4-plan.md) and uses [the C4.0 inventory](stage-c4-inventory.md).

## Context

After C4.0, 65 State arrays are still allocated by systems (`numerical_sim/docs/stage-c4-inventory.md`). C4.1 moves the 36 that hold surgical interaction history: Centreline, the six residue arrays, Hook, Grasp, VolumetricGrasp and Clipping. All live in `chole_tet_med`; the timing and rod workloads hold none. Afterwards a fresh runtime bound to a saved State and Control resumes an episode with a held grasp, a set hook and an applied clip, closing the "mounted-tool episode restore remains partial" caveat carried since C3.1.

Binding decisions: residue migrates as State unchanged; resets keep their exact behaviour; the C4.0 timing result is accepted, so no timing campaign; HDF5 stays deferred; `aorta_shell` catheter-impulse failures are retained without attribution; a residue-owning system on a body without a surface system fails loudly at declaration.

Unchanged: every numerical kernel (including `apply_average_tools` in `surgical/.../mechanics/tool_delta.py`), hook order, capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins, retained environments, historical certificates.

## What migrates

| Order | System (`surgical/src/i4h_surgical/systems/`) | Arrays | Domain | Template |
| ---: | --- | --- | --- | --- |
| 1 | `mechanics/centreline.py` | `positions`, `active_count` | centreline node `(E,K)`, published once per centreline system | `rest[variant]`, `clamp_count[variant]` |
| 2 | `contact/instrument_contact.py` | `delta_k`, `delta_jaw` | surface support rows `(E,Nb)`, tool width | zero |
| 3 | `instruments/hook.py` | `verts`, `tet`, `bary`, `friction_coord`, `radial`, `active` | hook anchor `(E,2)` | `verts`, `tet` = -1, others zero |
| 4 | `instruments/grasp.py` | `prim`, `tet`, `t`, `radial`, `prev_axis`, `verts`, `bary`, `gap`, `active`; `count`, `pending`; `delta_k`, `delta_jaw` | slot `(E,n,G)`; instrument runtime `(E,n)`; support rows | `prim`, `tet` = -1, others zero |
| 5 | `instruments/volumetric_grasp.py` | `particle`, `sphere`, `rest`, `active`; `count`, `pending`, `claim`; `delta_k`, `delta_jaw` | slot `(E,n,G)`; instrument runtime `(E,n)`; support rows | `particle` = -1, others zero |
| 6 | `instruments/clipping.py` | `node`, `active`; `count`; `clipped` | clip slot `(E,G)`; WORLD; centreline node `(E,K)` | `node` = -1, others zero |

Verified out of scope: contact records (`record_count`, `rec_*`) are written in InstrumentContact's `iteration` after `substep_start` zeroes the count, and every reader (Grasp, Hook, Thermal, Smoke) runs in `substep_end` or `step_end` of the same step. `impulse`, `moment`, `jaw_impulse` are outputs; `delta`, `delta_count` are self-clearing scratch.

## Design

### Declarations

- New host-only module `surgical/src/i4h_surgical/interaction_storage.py`, modelled on `surgical/src/i4h_surgical/instrument_storage.py`: one `publish_*` / `bind_*` pair per system, using `defer_declaration`, `Declarations.domain/add/register_package` and `Views.register_package(identities=, checks=)` from `physics/src/i4h_physics/storage.py`. No private storage access.
- Each system gains `declare_storage(cls, authoring, declarations, spec, records)` (invoked by `physics/src/i4h_physics/authoring.py:72` in spec order, with earlier systems' records). Capacities come from specs and records, never from instances:
  - Grasp `G = max(jaw capsules per instrument x records, 1)` from the instrument record (`scene.instrument_domains`, primitive kinds via `geometry_values`) and the referenced `instrument_contact` spec's `records`.
  - VolumetricGrasp `capacity`, Clipping `slots`, Hook's fixed 2 from their own specs; Centreline `K` from the body's authored centrelines.
  - InstrumentContact gains a `declare_storage` returning a record (body, surface id, records, tool count) for its dependants.
- Disabled systems declare storage too. Direct Scene callers get `declare_*` helpers. Missing, duplicate, late or changed declarations fail explicitly.
- Slot and anchor rows reference the control system's instrument runtime domain. Tet, vertex and particle indices stay body-local kernel indices with no Newton reference. Shared rows use world -1, runtime rows real worlds.

### Residue arrays without new rows

Chole at 64 worlds already authors 6.9 million custom rows in about 13 s; a row per particle and tool for three systems would add about 5.4 million. So:

- Declare residue fields on the surface system's existing per-particle row domain `i4h:<surface>_support`, shape `(E,Nb)` (`physics/src/i4h_physics/surface_storage.py:67`), as `float32` with `row_width = 3 * max(n,1)`.
- `Field` allows `row_width` on scalars only, and `Views` exposes scalar layouts (`storage.py` around line 433). Add one option so `Views` exposes the same memory as `wp.array2d(dtype=wp.vec3)` of shape `(E*Nb, max(n,1))`, built like the existing placement branch (`wp.array(ptr=..., copy=False)` with a retained backing reference). Layout equals today's arrays, so kernel signatures and indexing do not change. Test pointer, stride, shape, dtype and capacity on CPU and CUDA.
- No surface system on the body: declaration raises a clear error naming it. If an existing test or case builds VolumetricGrasp, Grasp or InstrumentContact without a surface, stop and report instead of changing that fixture's intent.
- If the support domain cannot be shared without touching kernel indexing, stop and report the row and construction cost of a dedicated domain.

### Initialization and binding

- Templates carry the constructor fills in the table. They differ from post-reset bytes for fields a reset leaves alone; both are preserved exactly.
- Centreline templates are computed on the host from the authored tables; no finalizer kernel. The constructor's `reset` call goes away.
- Centreline's constructor `refresh` rewrites `positions` and `active_count` from current particles at binding. Pin today's behaviour first (bind to a moved State and to a State saved after a step; record which bytes change). Remove it only if every host path already reaches `I4hSolver.reset(flags=0)` (which calls `refresh` on all systems, `physics/src/i4h_physics/sim.py:178`) before the first step; Lab main's forward path and Lab 3.0's `_reset_solver_internals` do, the standalone `Stepper` path must be checked. Otherwise keep it as a named exception limited to derived values. Report which applied.
- Constructors bind views instead of allocating. Binding and cache-only preparation preserve all State and Control bytes.

### Aliases and lifecycle

- Hook: `entry_active`, `entry_tet` and InstrumentContact's `suppress_active`, `suppress_verts` stay column-0 views of the `(E,2)` Newton-backed arrays; register an identity for each; test under CUDA capture and for replacement rejection.
- `Clipping.clipped` is declared on the centreline node domain; Bleeding keeps aliasing it (its own fields join in C4.2).
- Hook's disabled-`prepare` zero of `active` stays, as a named exception to the preserve-bytes rule (removing it changes contact suppression). Grasp, VolumetricGrasp and Hook resets keep their partial clears. Residue is never cleared by reset.

## Steps

1. Commit this plan.
2. Pre-edit audit against the C4.0 manifest (`.tmp_tests/stage-c40-20261110/certificate/source-manifest.json`), evidence directory `.tmp_tests/stage-c41-<date>/`.
3. Pre-migration tests on unchanged production code, committed separately: extend `tests/packages/test_storage_residue.py`, `tests/runtime/test_instruments.py`, `test_volumetric_grasp.py`, `test_clipping.py`, `test_centreline.py` to record stale slot bytes after reset, Hook's disabled-`prepare` zero, Centreline constructor writes on a moved State, and clip, hook and grasp state surviving cache-only preparation.
4. Storage layer: the typed vec3 view option in `physics/src/i4h_physics/storage.py`, with tests in `tests/packages/test_storage.py`.
5. Migrate in table order, one system per commit-sized change, running that system's tests and the ownership suite after each: Centreline, InstrumentContact residue, Hook, Grasp, VolumetricGrasp, Clipping. Update direct-Scene fixtures (`held_probe` in `tests/runtime/test_coupling.py`, `tests/runtime/helpers.py`, `examples/composition`) with the new `declare_*` calls.
6. New `tests/packages/test_interaction_owned_storage.py` (CPU eager, CUDA eager, captured): templates, domains, identity columns, owners, shapes, strides, pointers for all 36 arrays; two control systems and two worlds; bind, prepare, cache reset preserve bytes; masked, empty, repeated episode resets; rejection of replaced arrays, late or duplicate declarations, changed capacities; scratch poisoning of `delta`, `delta_count` and contact records.
7. Resume test in the same module: save State and Control with an active grasp, an active volumetric grasp, a set hook with a pierced exit anchor, an applied clip and non-zero residue; bind a fresh runtime; step both; compare bitwise with uninterrupted execution; include a mounted tool with a pending reaction via the `ExposedRigid` solver from `tests/packages/test_mount_owned_storage.py`. Fixtures exclude Thermal, Bleeding, fluids and Smoke (C4.2). Move `authored`, `runtime`, `snapshot`, `copy_container` from the three existing ownership test modules into one shared helper.
8. Re-anchor the allocation inventory (adapt `.tmp_tests/stage-c40-20261110/reconcile_inventory.py`): 36 `S` sites removed, added view sites reviewed, `scripts/check_stage_c_inventory.py` passes. Update `scripts/measure_stage_c_construction.py` accounting for the new templates and identity columns, since its assertions are exact.
9. Development selection in all three environments. The pre-freeze list names every test that builds these systems directly or through a factory: `test_coupling.py` (`held_probe`), `helpers.py` builders, `test_instruments.py`, `test_volumetric_grasp.py`, `test_clipping.py`, `test_centreline.py`, `test_thermal.py`, `test_scissors.py`, `test_procedure_sequence.py`, `test_procedures.py`, `test_stretch_breaking.py`, `test_two_worlds.py`, `test_isaaclab_env.py`, `examples/composition`.
10. Commit with hooks passing, freeze the clean commit, run one campaign with the C4.0 launcher, driver and auditor (`.tmp_tests/stage-c40-20261110/certificate/`, paths and expected deltas adapted): full core suite, both six-module host selections, 36 groups, 48 replays, 18 chole trials, four construction samples. No retries; failures retained; independent audit under the source guard. The campaign also qualifies the Lab 3.0 manager rename in `ebd86e2`.
11. Report `numerical_sim/docs/stage-c41.md` (lint-clean), status prepended to `docs/stage-c.md`, advisor review, documents-only commit, no push. Stop for review before C4.2.

## Verification

- Suite, group, replay and chole counts recomputed from logs and retained arrays by the auditor; no production file differs between the freeze manifest and the final commit.
- Construction at 1 and 64 worlds against C4.0 (chole 64 worlds: 15.0 s, 1,219,312,846 retained bytes). Expected growth at 64 worlds: about 129 MB residue template plus about 12 MB slot templates; live storage unchanged; custom row count grows only by slot, anchor and centreline node rows. Every delta must be explained to the byte by the auditor.
- Resume test bitwise on all three execution modes.
- `pre-commit run --files $(git ls-files numerical_sim)` passes and leaves a clean status; excluded historical documents and patches unchanged.

## Stop conditions (report and wait for the user)

- A failed reference group other than `aorta_shell` catheter impulse, or a chole group below 2 of 3. Chole exercises every migrated system, so a new chole failure pattern is the main risk signal.
- A suite failure other than the two known intermittents (clipping count, `two_worlds` fluid count).
- Chole construction time at 64 worlds more than 10% above C4.0.
- Residue fields cannot share the support domain without a kernel indexing change, or an existing fixture uses a residue-owning system without a surface.
- Resume is not bitwise for a migrated system: report the first diverging array, do not loosen the comparison.
- A cross-step reader of unmigrated data that the inventory missed.
