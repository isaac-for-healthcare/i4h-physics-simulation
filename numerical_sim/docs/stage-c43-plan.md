# C4.3 plan: continuation contract

Implementer: Claude, with advisor review of the diff and the report. Starts from commit `09bec9f`. Approved by the user on 2026-11-11.

## Context

After C4.2 the allocation inventory has no State site left on a system owner, but there is no public way to save and restore an episode. Tests copy whole containers with private helpers, and a freshly bound runtime does not resume bitwise where a contact system holds a BVH built at binding. C4.3 adds the public save and restore call, makes bind-time caches reproducible at the save and restore boundary, removes the remaining private storage reaches, and proves continuation in one test module. C4.4 (timing gate and closure) follows.

User decisions (2026-11-11):

- **C4.2 is closed.**
- **BVHs are rebuilt in place by the save and restore calls.** Step code and approved references stay untouched. A runtime that saves diverges at rounding level from one that never saved; this is stated in the docstring and report.
- **The masked variant claims bytes only.** Unselected worlds' State and Control bytes are unchanged at the call. Bitwise resume is claimed for unmasked save and restore only, because Warp rebuilds a whole BVH and so also rebuilds unselected worlds' subtrees.
- Carried over: resets unchanged; TipEvidence and Procedure state excluded; GraphContainment `prepare` unchanged; `aorta_shell` impulse failures retained without attribution; no timing campaign before C4.4; HDF5 deferred; no push.

Unchanged: every existing numerical kernel and every step, reset, refresh and prepare hook (AST audit against `09bec9f`; new kernels and one new hook are additions), capacity 256, comparator, zero floors, factor three, approved references, chole 2-of-3, dependency pins, retained environments, historical certificates, and `docs/stage-c-field-contracts.md` byte-for-byte.

## Deviations from the masterplan, stated up front

1. **The recipe is bind, prepare, then restore**, not "bind a fresh runtime to a saved State". `prepare` and `refresh` keep their history writes (GraphContainment `rebuild`, Hook's disabled zero), so the saved bytes are copied in after them. Restore never calls `reset(flags=0)`, `refresh` or `prepare`.
2. **Tip evidence is not in the continuation test** (host-owned assessment state, C4.2 decision).
3. **Bitwise claims are stated per case and mode**, not globally. Rod and mount resumes are already bitwise in all three modes; chole on CUDA is not repeatable.
4. **Rigid body state is the host's.** i4h copies its declared fields and its owned particle rows; the mounted-tool test copies `body_q`, `body_qd`, `body_f` itself.

## Design

### Public call

On `I4hSolver` (`physics/src/i4h_physics/sim.py`), forwarded by `CoupledSolver`, `Stepper` and `PhysicsBinding`:

- `save_i4h(dst_state, dst_control, mask=None)` copies from the bound containers; `restore_i4h(src_state, src_control, mask=None)` copies into them. `mask` is the existing `(E+1,)` `wp.bool` world mask (`select_envs` in `stepper.py` builds it); the last slot selects shared rows (world -1). `None` means all worlds and shared rows.
- Copied: every declared `Owner.STATE` and `Owner.CONTROL` field, and the owned rows of native `particle_q`, `particle_qd`, `particle_f`. Foreign host particles, padding rows, body and joint fields, Model fields (including `mounted`) and templates are never written.
- Both calls end with the new system hook `rebuild_caches(scene)` (default no-op in `systems/base.py`), also for an empty mask only if rows were copied. Restore then clears outputs through the existing `publisher.clear`.
- Validation before any write: the other containers carry this Model's storage token (`i4h.storage_*_id`, the check `Views.__init__` already does), `Views.check()` passes (replaced arrays rejected), and a Control whose backing aliases the Model template (`model.control(clone_variables=False)`, pointer test as in `Views.allocated_bytes`) is rejected as a restore or save destination.
- Runs eagerly at a completed step boundary, outside captured graphs. No device allocation after construction.
- Docstring lists the exclusions: rigid solver history including MuJoCo-Warp's, host time and episode counters, assessment, Procedure and tip evidence, renderer state, HDF5 replay; and the two side effects (tree rebuild, masked perturbation).

### Row copy

- New `physics/src/i4h_physics/continuation.py` (host-only planning plus one kernel). At binding it builds, per row domain, an `int32` row-to-world map on the device: the existing `model.i4h.<domain>_world` column for custom domains; the row index for native WORLD; for the native particle domain a map from the placement `(base, stride, col)` with -2 for foreign rows. One kernel copies row `r` when the map is not -2 and `mask[world or E]` is set, over a non-owning `uint8` `(rows, row_bytes)` view of source and destination, so one kernel serves every dtype and `row_width`. Pool rows are body-major, which the map handles.
- Views for the bound containers are created at binding; views for the other containers are cached by container identity on first use.
- `Declarations` gains the public accessors this needs (`rows(domain)` for world and owner columns, `authored`, `default(field)`), which also serve the private-reach removal below.
- Before coding: list which STATE and CONTROL fields live on shared rows (world -1) and confirm the mask-slot rule is right for each; check whether either Lab host's live Control aliases the template. If one does, stop and report instead of rejecting it.

### Bind-time caches

- **Measure first, on unchanged code, CPU:** for `tissue_contact` (chole and `tissue_contact` case), ShellContainment (`aorta_shell`) and TetContainment (a tet-containment case; `tests/runtime/test_tet_containment.py` fixture if no case has one), compare an uninterrupted runtime with a freshly bound one restored by container copy, with and without a manual `bvh.rebuild()` at the save point. Assert the tree is queried (candidates or contacts non-zero). FluidContact's result exists (`.tmp_tests/stage-c42-20261110/fluid-contact-bvh-resume.txt`).
- `rebuild_caches` is implemented for FluidContact (`fluid_contact.py`), PartContact (`part_contact.py`), ShellContainment and TetContainment: recompute bounds with the system's existing `_bounds` and call `self.bvh.rebuild()`. All four get it whatever the measurement shows, since tree order provably reaches accumulation order or tie-breaking in each; the measurement decides what the report claims.
- GraphContainment needs nothing new: its cache is State and restore copies `rebuild` as saved. Centreline's bind-time write is idempotent. Fluid's `HashGrid` is rebuilt every substep.
- Renderer BVHs (`isaaclab/camera/tiled.py`) are outside the claim.

### Private reaches

In scope, replaced by public API with no behaviour change:

- `authoring.py` `storage._authored`; `Views` reading `Declarations._names/_domains/_token/_defaults`.
- `scene.__dict__.setdefault(...)` for `_deferred_storage`, `_tet_declarations`, `_tet_fields`, `_surface_declarations`, `_surface_fields`, and `scene._mount_declaration`: one declared `Scene.storage` record with named members.
- `scene._storage_starts` read by `containment_storage.py`, `rod_storage.py` and surgical `interaction_storage.py`: a public `scene_storage` accessor. The C4.1 and C4.2 reports claimed public-API-only declarations; the C4.3 report corrects that.
- `coupling.py` reaching `ours._check_state` and `ours._all`: public names on `I4hSolver`.

Listed in the report and left for closure (C4.4): host-layer reaches into binding and solver (`binding._prepared`, `_runtime`, `_output_arrays`, `solver._command_views`, `simulator._mask`, recording and assessment privates) and all reaches into Isaac Lab's own managers.

## Steps

0. **Close C4.2:** disposition at the top of `docs/stage-c.md` (closed with the stated deviations; one failed chole trial retained), documents-only commit.
1. Save this plan as `docs/stage-c43-plan.md`, lint, commit.
2. Pre-edit audit against the C4.2 manifest; evidence directory `.tmp_tests/stage-c43-<date>/`. Run the BVH measurements and the two pre-coding checks above. Take pre-change construction samples (six, as in C4.2).
3. **Test helper merge, own commit, production code untouched:** new `tests/packages/ownership_helpers.py` with `MODES`, `arrays`, `snapshot`, `copy_container(source, destination)`, `first_difference` and one rigid test solver (inverse mass as a parameter, replacing `ExposedRigid` and `HeavyRigid`). The rod, mount, instrument, interaction and tissue modules import them; scene-specific `authored` and `runtime` builders stay local. Collected test ids and pass counts must be identical before and after.
4. Pre-migration pins, own commit: empty and repeated reset masks for the 25 C4.2 arrays (not done in C4.2); bytes written by `prepare` and by `reset(flags=0)` on a stepped State, per system, as the record of what restore must not route through.
5. Storage accessors and private-reach removal, running the ownership suites after each module.
6. `continuation.py`, `save_i4h` and `restore_i4h`, `rebuild_caches`, forwarding on `CoupledSolver`, `Stepper`, `PhysicsBinding`; `I4hComponent` exposes them for hosts.
7. New `tests/packages/test_continuation.py`, all three modes unless noted:
   - **Copy exactness:** after save, destination equals source on every copied row and is untouched elsewhere (foreign particles, padding, bodies, Model, templates); save then restore is the identity; no allocation (Warp mempool counters) on repeated calls.
   - **Masks:** partial, empty, repeated and device masks; unselected worlds and, without the last slot, shared rows are bitwise unchanged; pool rows of an unselected world unchanged.
   - **Rejections:** containers of another Model, of a second finalize of the same builder, a replaced array, a template-aliased Control; nothing written before the error.
   - **Resume through a fresh runtime** (bind, prepare, restore) and **save, advance, restore, replay on the same runtime**, bitwise against uninterrupted execution where the case is repeatable: chole with grasp, volumetric grasp, hook, clip, cut, heat, burn, coagulation, colour, blood and smoke (CPU); the mounted tool with a pending reaction (CPU); `aorta_centreline` with graph containment, catheter commands and an active drag (CPU, and CUDA if twins agree); `aorta_shell` and the tet-containment fixture (per the measurement); rod and mount cases (all modes). State that the script cannot reach by stepping is injected with the C4.2 helpers and listed in the report.
   - **Twin check per case and mode:** two runtimes restored from one checkpoint must agree before a resume comparison counts; where they do not, the test checks copy exactness, view identity and isolation only.
   - **Hosted path:** in the Lab selections, a restore through `I4hComponent` followed by a step, and a check that neither host runs its cache reset after it.
   - The C4.1 and C4.2 resume tests switch from `copy_container` to the public call and drop the manual BVH rebuild.
8. Re-anchor the allocation inventory (row maps and views are new sites; status string updated; code `A` recorded in the JSON, not in the hashed legend) and the exact byte accounting in both construction scripts.
9. Development selection in the three environments, commit with hooks passing, freeze, one campaign with the C4.2 launcher, driver and auditor (15 jobs: six construction samples, six qualify groups, three suites). No retries; failures retained; independent audit.
10. Report `docs/stage-c43.md`, status prepended to `docs/stage-c.md`, advisor review, documents-only commit, no push. Stop for review before C4.4.

## Verification

- Auditor recomputes suite, group, replay and chole counts; no production file differs between the freeze manifest and the final commit.
- AST audit: the 222 kernels and 151 hooks of `09bec9f` unchanged; additions listed by name.
- Reference groups must be unaffected: no reference script calls save or restore, so any new failure pattern is a stop condition.
- Construction against C4.2 (chole 64 worlds 14.137 s, 1,360,663,246 bytes; `aorta_centreline` 64 worlds 0.493 s, 49,404,950 bytes): growth is the row maps only, explained to the byte.
- `pre-commit run --files $(git ls-files numerical_sim)` passes with a clean status.

## Stop conditions (report and wait)

- A failed reference group other than `aorta_shell` catheter impulse, a chole group below 2 of 3, or a suite failure other than the two known intermittents.
- CPU twins agree but resume differs from uninterrupted execution after the tree rebuild: report the first diverging array and its owner; do not loosen the comparison.
- A Lab host's live Control aliases the template, or a host runs a cache reset that would follow a restore.
- A STATE or CONTROL field on shared rows for which the mask-slot rule is wrong.
- In-place `rebuild()` does not reproduce a freshly constructed tree on CPU for one of the four systems.
- Construction time at 64 worlds more than 10% above the pre-change sample for chole or `aorta_centreline`.
- Removing a private reach requires changing a hook or a kernel.

## Not in C4.3

The timing gate, closure housekeeping and the documentation rewrite (C4.4); host-layer private reaches; per-world BVHs; HDF5.
