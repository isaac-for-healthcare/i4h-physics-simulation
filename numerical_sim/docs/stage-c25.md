# C2.5 — C2 closeout and C3 handoff

**C2 closed for progression; C3 not started.** The user-supplied C2.5 plan
accepts the two C2.4 failed reference groups for progression: standalone captured
`aorta_static` masked-reset impulse and Lab 3.0 eager `aorta_shell`
stepped/masked-reset impulse. This changes checkpoint disposition only. The
original [C2.4 report](stage-c24.md), certificates, endpoints, construction
samples, development failures and historical tables remain unchanged.

Lab 3.0 captured chole trial 0 remains failed (19 product comparisons); that
case group passes the existing 2-of-3 policy. No failed trial is relabeled,
replaced or retried. Closure does not assert numerical perfection, full
fresh-runtime continuation, general save/restore, or final steady-step
performance qualification. This work stops at C2.5, without production changes,
ownership migrations, commit or push.

## Inherited evidence independently checked

The frozen C2.4 manifest remains
`91c74d1a7fbf720811c4ce84d121477678e3b810b66e2c42be898cd6cf816566`.
All **9,376 files** match. The new [read-only auditor][audit] checks the retained
source maps, paired startup/exit receipts, imported package paths, job exits,
approved reference integrity, all individual endpoint archives, construction
sample provenance and allocation inventory. It does not execute the old
certificate writers or produce new construction/reference samples.

[The inherited-evidence audit][audit-result] independently recomputes comparisons
from all 48 endpoint archives with the frozen comparator (zero floors, factor 3).
It reproduces each report's complete failure list, not just its pass flag:

| Inherited measure | Recomputed result |
| --- | ---: |
| Case groups | 34/36 pass |
| Individual replays | 45/48 pass |
| Chole trials | 17/18 pass |
| Failed individual replays | 3 |
| Failed product comparisons | 22 |
| Matched reference-worker source maps / endpoint archives | 48/48 |
| Paired Python startup/exit receipts | 82 |
| File-backed import observations | 504 |

Each of the six host/mode chole groups has exactly three numbered fresh-process
trials and a distinct worker PID per trial. Lab 3.0 captured is 2/3; the other
five groups are 3/3. Both construction samples have distinct process receipts
and match their recorded world counts, contact capacity 256, source hashes and
qualified Warp binary. Non-file module observations remain explicit. For the
three `python -m pytest` processes, Python replaces startup `argv[0] == '-m'`
with pytest's `__main__.py` at exit; the remaining arguments and process identity
match. The audit checks this specific interpreter behavior.

| Inherited C2.4 suite | Recorded result | Exit |
| --- | --- | ---: |
| Core packages/runtime | 940 passed, 47 skipped | 0 |
| Lab 3.0 host suite | 109 passed, 8 skipped | 0 |
| Main host suite | 112 passed, 5 skipped | 0 |

These are inherited suite results, not C2.5 full-suite reruns. The accepted
reference failures remain individually linked in [the original endpoint
extraction](../.tmp_tests/stage-c24-20261106/failed-comparison-values.json).
The original certificate's failed integrated-gate flags remain false.

## C2 completion matrix

Each cell below links implementation and focused test evidence. Ownership means
persistent physical storage; existing scratch and legacy histories keep their
previous owners. C2.5 adds coverage and auditing only.

| Family | Ownership and initialization | Binding and preservation | Explicit reset | Stale-buffer and layout rejection | Consumers |
| --- | --- | --- | --- | --- | --- |
| Native particles | [Declarations][particle]: native Model rest positions; exact inverse mass/default flags in Model; history/live flags in State. [Exact bytes, authored padding/variants and foreign offsets][particle-tests]. | [Scene aliases][scene] bind selected native rows without writing history. [Host preservation and retained-builder independence][author-tests]. | [Selected particle reset/history initialization][scene], after position writes. [Partial, empty, repeated and foreign-row checks][particle-tests]. | [Storage identity/layout checks][storage] and [Scene layout validation][scene-storage]. [Replaced host buffers][author-tests], [nested replacement tests][storage-tests]. | [Particle mechanics][particles] uses Scene aliases. [Main-block ownership, strided views and snapshots][blocks-tests]; [coupling isolation][coupling-tests]. |
| Private pools | [Scene declarations][scene-storage]: four Model fields and four State fields, body-major rows; no native particles added. [Independent States, shared/default Controls, gaps and empty padding][scene-tests]. | [Scene.allocate][scene] aliases all eight arrays and body views. [Pool byte/pointer and Fluid span checks][scene-tests]. | [Scene.reset][scene] preserves unselected pools and restores selected authored defaults. [Full/partial/empty/repeated reset tests][scene-tests]. | [Frozen capacities/layout][scene-storage] and [checked views][storage]. [Replacement and reconstruction tests][scene-tests]. | [Fluid][fluid] uses pool spans; [fluid-only eager/captured and held-snapshot tests][blocks-tests]. |
| Scene metadata | [Model-owned variants, static geometry and component gravity][scene-storage]; shared static identities use world/owner -1. [Direct completion, geometry and empty-domain tests][scene-tests]. | [Scene binding and static mesh aliases][scene] preserve Model edits; unrotated gravity aliases native storage and rotated gravity has its own storage. [Gravity/retained-builder tests][scene-tests]. | [Episode reset][scene] leaves Model metadata intact; [live variant-table reset tests][surface-tests] verify that selected State reads current defaults. | [Layout/transform signature][scene-storage] rejects late changes; [storage checks][storage] reject replacements. [Geometry and undeclared-storage tests][scene-tests]. | [Surface/Tet variant aliases][surface-storage], [static mesh consumers][scene]; [direct construction][scene-tests] and [component-frame mounted coupling][coupling-tests]. |
| Tet mechanics | [Tet declarations][tet-storage]: topology/defaults/materials in Model; active tet/edge masks in State; `native_tet=-1`, no native topology. [Ownership, identities and exact defaults][tet-tests], [domain tests][tet-domain-tests]. | [Tet constructor][tet] binds checked aliases without reset; scratch stays runtime-owned. [Restored masks/materials/foreign rows][tet-tests]. | [Tet.reset][tet] reads live Model defaults/current variants. [Selected/empty/repeated reset and Model-edit tests][tet-tests]. | [tet_fields/alias_tet_fields][tet-storage] and [every migrated buffer/cache replacement test][tet-tests]. | [Deletion consumers][tet-consumers], [containment][containment], [contact][contact] and [PBD/XPBD scratch/capture tests][tet-tests] use selected masks and domains. |
| Physical Surface | [Twelve Model/eight State fields][surface-storage]; authored triangles/supports initialize templates, counters start zero; immutable authored identities. [Defaults, repeated instances, empty backing and accounting][surface-tests]. | [Physical constructor][surface] performs no upload/reset/rebuild. [All restored State bytes and Model edits][surface-tests]; [new pending second-deletion rebind][closeout-test]. | [Surface.reset][surface] restores selected physical fields and clears bookkeeping, without resetting Tet masks. [Current live variants, empty/repeated masks and other-State/template preservation][surface-tests]. | [surface_fields and layout signature][surface-storage]; [all fields and cached aliases, late/missing/wrong declarations][surface-tests]. | [Surface rebuild/deactivate kernels][surface], [deletion isolation][tet-consumers], [containment][containment], [cut-face contact][contact], [snapshot aliases][snapshots]. [New nonzero-capacity emission][closeout-test] supplements [existing overflow/rebind coverage][surface-consumers]. |

Native counts remain unchanged and local topology integers receive no Newton
reference offsets. Tet/Surface consumers retain their original ordering and
compatibility aliases. Physical Surface's three additive/runtime table duplicates
are removed; Thermal and Appearance remain outside C2 physical ownership.

## New focused verification and supplemental certificate

[test_c2_closeout.py][closeout-test] uses physical Tet/Surface runtimes sharing
one Model, with independent selected, uninterrupted reference and untouched
States. An incidence oracle built directly from authored tetrahedra chooses two
nonadjacent interior deletions. The first exposes **4** actual cut faces; the
second leaves dirty work pending and requires **8** exposed faces. Capacity is
128 throughout, with no Model capacity edit. This is a small regression fixture;
the approved contact capacity of 256 is unchanged.

Only the selected runtime is rebound and prepared. The test checks exact bytes
before rebuilding, including already emitted slots, supports, masks, native
particle history and bookkeeping. It compares complete face sets by owner and
oriented triangle (cyclic rotation allowed, reversed winding rejected), without
requiring atomic slot order. It checks the expected cut count, live owners and
vertices, incidence-derived vertex/edge supports, unselected worlds, zero
overflow, cleared dirty state, the untouched State and all i4h Model fields and
initialization templates. Captured cases launch an actual captured rebuild and
also execute the prepared mechanics graph; eager cases execute eager rebuilds
and mechanics. Existing zero-capacity overflow tests remain separate and unchanged.

| New guarded check | Result |
| --- | --- |
| Standalone: PBD/XPBD × CPU eager/CUDA eager/CUDA captured | 6 passed |
| Lab 3.0 qualified environment: same six combinations | 6 passed |
| Main qualified environment: same six combinations | 6 passed |
| Existing focused core ownership, direct construction, snapshots, deletion consumers, containment and coupling | 279 passed, 1 skipped; 5 warnings |
| Inherited provenance/endpoints/accounting audit | Passed |
| Current allocation inventory | 587 sites across 181 files; passed |

The three six-case runs qualify this physical regression in each environment;
they do not claim new full host integration coverage. The core selection is
explicitly recorded in [the runner][runner] and [per-job commands/results][certificate].
Its results overlap inherited suites and must not be added as unique-test totals.

Supplemental manifest SHA256:
`2932e8e9a7fc006d90a810bf5ed17aaa4bf320550205a7d6edc138263151daca`.
It covers **10,130 files**, including the original 9,376 source entries, the new regression and audit/runner
sources, and retained C2.4 artifacts and historical inventory/removal records.
Every Python job, including nested processes, inherits the fail-closed
[startup/exit guard][guard]. The runner also verifies all hashes before and after
each job. Logs, JUnit results, paired receipts and the [supplemental certificate][certificate]
are retained in the [C2.5 evidence directory][evidence]. All six jobs pass with
eight matched startup/exit receipt pairs. The single skip is
`test_slang_contact_panel_activation` (requires `I4H_RENDER_TESTS=1` and slangpy).
Five existing coupling-fixture warnings report Newton inertia corrections;
these remain in the focused log.

Development observations are retained separately: an initial output-path mistake
prevented test creation/collection; the corrected standalone preflight passed
six tests. Auditor development retained a missing-parenthesis syntax error,
an overly strict `argv[0]` equality assertion, and an intentionally checked raw
Model-byte sum that double-counted the gravity alias. The latter two were audit
assumption errors resolved against receipts and frozen allocation source, not
runtime or numerical regressions. All development logs remain present. Ruff
passes for the new test and harnesses. No frozen C2.4 test or artifact was edited.

## Reconciled construction accounting (inherited samples)

No new samples were taken. [The audit][audit-result] reconciles C2.4 with the
retained C2.3 one-world and 64-world samples using the frozen formulas and
per-buffer listings. Allocation totals use maximum backing extent per aliased
pointer, as checked by the existing focused
`test_measurement_accounts_maximum_alias_extent_and_tet_formulas`.

| Retained C2.4 bytes | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model buffers | 6,125,893 | 93,207,265 |
| Initialization templates | 1,499,948 | 95,996,672 |
| Live State | 1,567,292 | 100,306,688 |
| Additional checkpoint State | 1,567,292 | 100,306,688 |
| Live / checkpoint / intentionally shared Control | 0 / 0 / 0 | 0 / 0 / 0 |
| Shared-Control/template overlap | 0 | 0 |
| Private-pool Model / each State / templates | 294,912 / 491,520 / 491,520 | 18,874,368 / 31,457,280 / 31,457,280 |
| Pool identities | 98,304 | 6,291,456 |
| Tet identities | 1,172,688 | 25,800,144 |
| Surface identities (excluding reused Tet domains) | 532,760 | 12,663,032 |
| Six legacy residue arrays | 2,020,320 | 129,300,480 |
| Residue Newton template / checkpoint bytes | 0 / 0 | 0 / 0 |
| Retained Warp storage | 27,805,529 | 1,218,389,558 |

Native gravity backing is 24 / 780 bytes. Its Scene alias is 12 / 768 bytes
at the same pointer, so summing the named Model-buffer listings would overcount
by 12 / 768 bytes. Taking the maximum extent preserves the native world -1 row.
The recorded Model totals already apply this rule; no original result changes.
Live State exceeds initialization templates by native particle force storage
(67,344 / 4,310,016 bytes). Checkpoint State is a separate allocation measured
after retained construction storage, not a general checkpoint implementation.

| Surface reconciliation bytes | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model data | 1,565,856 | 1,565,856 |
| Each State / initialization templates, each | 437,920 | 28,026,880 |
| Already-additive three Model tables | 1,054,088 | 1,054,088 |
| New Model declarations | 511,768 | 511,768 |
| New identity columns | 532,760 | 12,663,032 |
| Removed legacy persistent arrays | 2,003,776 | 29,592,736 |
| Retained-storage delta versus C2.3 | -83,408 | +39,635,824 |

The delta is new Model declarations + new identities + two State-sized copies
(live and template) − removed legacy arrays. Existing additive Model tables
are counted once. The physical constructor now aliases all twelve Model/eight
State fields, with no legacy uploads or State allocations. Empty-domain backing
is zero in these chole samples; the focused empty-domain tests separately check
inaccessible Model/State rows, State template padding and identity backing.

Thermal heat/burn still have additive Newton live State, initialization-template
and legacy Thermal copies, each **150,584 / 9,637,376 bytes**. Additional Newton
checkpoint State also contains its declared Thermal fields. Appearance remains
legacy. Native/exact inverse mass and default flags still have **22,448 /
1,436,672 bytes per copy**. The six residue arrays each retain **336,720 /
21,550,080 bytes**, owned by legacy InstrumentContact, Grasp and VolumetricGrasp;
there are no new residue template/checkpoint copies. Their total is already
included in retained Warp storage and is not added again.

The current inventory remains **587 sites in 181 files**. The accepted C1
inventory (627/178), every historical removal record, and the frozen C2.4
inventory remain unchanged. The supplemental manifest hashes them; the current
checker validates contracts, reviewed hashes and the allocation-site census.

## C3 handoff boundary

C3 may begin only as a subsequent task. Packed/named command ownership, instrument
and catheter controls, rod state/materials and coupling mount/reaction storage
remain unmigrated. Preserve valid reset poses, bidirectional command aliases,
recording replay, mounted-control ordering and the existing one-step reaction
delay when that work is planned and implemented. Existing coupling tests passing
here establish compatibility of C2 storage, not completion of C3.

Remaining Thermal, Appearance, fluid/emitter and interaction histories and general
save/restore/fresh-runtime continuation belong to later work. Keep the six residue
arrays and their accounting explicit. The final exclusive steady-step performance
campaign is also later work. No API, schema, kernel, layout, reset policy,
dependency, comparator or tolerance changed during C2.5.

[evidence]: ../.tmp_tests/stage-c25-20261107/
[audit]: ../.tmp_tests/stage-c25-20261107/audit_inherited.py
[audit-result]: ../.tmp_tests/stage-c25-20261107/inherited-audit.json
[runner]: ../.tmp_tests/stage-c25-20261107/run_supplement.py
[guard]: ../.tmp_tests/stage-c25-20261107/sitecustomize.py
[certificate]: ../.tmp_tests/stage-c25-20261107/supplemental-certificate.json
[closeout-test]: ../tests/runtime/test_c2_closeout.py
[particle]: ../physics/src/i4h_physics/particle_storage.py
[particle-tests]: ../tests/packages/test_particle_storage.py
[scene]: ../physics/src/i4h_physics/scene.py
[scene-storage]: ../physics/src/i4h_physics/scene_storage.py
[scene-tests]: ../tests/packages/test_scene_storage.py
[storage]: ../physics/src/i4h_physics/storage.py
[storage-tests]: ../tests/packages/test_storage.py
[author-tests]: ../tests/packages/test_authoring.py
[particles]: ../physics/src/i4h_physics/systems/mechanics/particles.py
[fluid]: ../physics/src/i4h_physics/systems/fluids/fluid.py
[blocks-tests]: ../tests/runtime/test_blocks.py
[coupling-tests]: ../tests/runtime/test_coupling.py
[tet-storage]: ../physics/src/i4h_physics/tet_storage.py
[tet]: ../physics/src/i4h_physics/systems/mechanics/tet.py
[tet-tests]: ../tests/packages/test_tet_owned_storage.py
[tet-domain-tests]: ../tests/packages/test_tet_storage.py
[tet-consumers]: ../tests/runtime/test_tet_state_consumers.py
[surface-storage]: ../physics/src/i4h_physics/surface_storage.py
[surface]: ../physics/src/i4h_physics/systems/mechanics/surface.py
[surface-tests]: ../tests/packages/test_surface_owned_storage.py
[surface-consumers]: ../tests/runtime/test_surface_state_consumers.py
[containment]: ../tests/runtime/test_tet_containment.py
[contact]: ../tests/runtime/test_tissue_contact.py
[snapshots]: ../tests/runtime/test_app_loop.py
