# C2.4 — Newton-owned physical Surface storage

**Implemented; checkpoint open.** Final suites pass **940/47 skipped** core,
**109/8** Lab 3.0 and **112/5** main. References pass **34/36 case groups**,
with **17/18 chole trials** passing under the unchanged majority policy. Two new
impulse failures remain unaccepted. This work stops at
C2.4 without commit or push. C2.3 is accepted for progression by the user-supplied
plan; its [two certificates, all failures, construction samples and direct-caller
correction](stage-c23.md) remain preserved. No C2.4 failure waiver is inferred.

## Implementation and scope

Physical Surface binds all twelve Model and eight State fields through
[surface_storage.py](../physics/src/i4h_physics/surface_storage.py).
`declare_surface(scene, tet_id="tet", system_id="surface")` follows the matching
Tet predeclaration before allocation/schema registration. CaseAuthoring uses the
same publisher. `surface_fields` returns cached checked aliases, body, domains
and referenced Tet fields without allocating device storage. Empty constructor
IDs resolve to `surface`; repeated instances use distinct explicit IDs.

The referenced Tet's variant-tet, edge-default, live-edge and material domains
are shared. Surface declares six custom domains: variant-face, world-face,
shared-face-part, variant-particle-support, world-particle-support and shared-part.
Shared rows use world -1. Native-particle owners resolve after world creation,
including host prefixes; unused variants, padding and initially unused cut slots
keep owner -1. Face owners use the first vertex of an authored nondegenerate
triangle; particle-support rows use authored active particles. Shared face/part
rows use the first world. These identities remain immutable when cut slots fill.
Local topology and slot integers have no Newton reference offsets, and no native
Newton topology is introduced. Overflow, deleted and dirty use native WORLD.

The physical constructor performs no uploads, reset or rebuild. New State
templates initialize triangles, owners and supports from authored variant rows
and zero all four bookkeeping fields. Binding/preparation preserve selected
State bytes, including pending dirty work, cut counts and latched overflow.
Explicit selected-world reset reads live Model defaults through the current body
variant table and clears selected bookkeeping without independently resetting
Tet masks. Authored topology, part mappings, cut ranges and dimensions are frozen;
changed authoring requires reconstruction. Every migrated array and cached
compatibility alias participates in stale-buffer checks. Valid Model edits survive
binding/reset. Existing contacts, containment, deletion, snapshots and rendering
continue through these checked aliases with unchanged relationships and scratch.

[The source audit](../.tmp_tests/stage-c24-20261106/source-audit.json) proves all
four Surface kernels/functions and explicit reset/step-end bodies unchanged.
Step-end still rebuilds authored slots, emits atomic exposed faces, then clears
dirty; overflow latches and counts can exceed cut capacity. Tests compare newly
emitted face sets semantically without sorting production slots or changing
reference comparisons. Pins, qualified Warp binaries, approved references,
tolerances, capacity 256 and coupling limits are unchanged.

Surgical Appearance allocates zeroed legacy history only after physical
reference/declaration validation; explicit appearance reset remains. Thermal
history still follows legacy construction. Appearance, Thermal, commands, rods,
coupling and the six residue arrays retain their owners. Physical Tet/Surface
rebinding is supported; no full fresh-runtime continuation or general save/restore
API is claimed.

## Pre-freeze verification and retained development failures

[All logs and artifacts](../.tmp_tests/stage-c24-20261106/) are retained.
The corrected first selection passed **68 tests** (direct construction, snapshot,
Tet ownership, Surface and coupling). New CPU/CUDA ownership tests passed **58**.
The expanded selection retained **436 passed, 2 skipped, 1 failed**: the existing
CaseAuthoring regression expected `storage_records["skin"]["variant"]`, which
was missing from the new record. Restoring that compatibility key preserves the
existing shared-Tet-domain identity assertion. The subsequent ownership, consumer,
current-variant-reset and fresh-process-memory selection passed **105 tests**.
The final pre-freeze selection passed **62 tests**, including every migrated
buffer and compatibility alias and PBD/XPBD cut/rebind/capture checks. Selections
overlap and are not additive unique-test totals.

Both focused host preflights passed: **109 passed / 8 skipped** Lab 3.0 and
**112 passed / 5 skipped** main. Both existing fresh-process memory probes passed
before freeze; they are allocation probes, not chole construction samples.
Alias-aware accounting counts the maximum extent for each shared pointer;
Surface formulas are checked against actual logical bound buffers on CPU/CUDA,
including inaccessible empty-domain backing separately.

Coverage includes exact defaults, shared/unused variants, multiple bodies,
inactive padding/reserved slots, repeated Surfaces, empty domains, independent
States and default/shared Controls, deliberately restored topology/supports and
all bookkeeping, partial/empty/repeated resets using a changed live variant
table, preserved Model edits, untouched other State/templates/native rows,
missing/late/wrong declarations, incompatible buffers, changed authoring and
retained-builder idempotence. Cuts, thermal deletion, stretch breaking, contact
and containment suites pass; new deletion assertions preserve the other State
and templates. Cut-slot bytes, pending dirty work, overflow and immutable face
identities survive physical rebinding. Snapshot and render compatibility remain
covered by the existing alias paths; interactive render tests retain their
existing optional-environment skips.

Earlier development logs also retain an initial layout failure caused by treating
Body.parts dictionary keys as Part records (57 failed, 11 passed); it was fixed
before the corrected first selection. A `python` executable lookup and two
working-directory mistakes prevented their intended edit/test commands; corrected
commands ran successfully. These are development/harness observations, not
reference retries. Ruff passes. The inventory checker reports **587 sites in
181 files**, zero new device-allocation sites and eleven removed sites. Those
sites cover the shared integer upload helper, the triangle-default upload,
eight State allocations and the temporary all-world reset mask: all twenty
physical arrays plus the mask are removed. [Removed-site contracts and links](../.tmp_tests/stage-c24-20261106/inventory-delta.json)
remain alongside the accepted C1 inventory.

## Frozen-source campaign

The frozen manifest covers production, tests, qualification and measurement
scripts, pinned installed Newton/Warp sources and binaries, qualified host source,
approved references, allocation contracts and inventory. Every worker and nested
Python process has startup/exit guards. The six approved cases run eager and
captured on standalone, Lab 3.0 and main. Each host/mode retains exactly three
fresh chole trials under the unchanged majority policy. The full core and both
focused host suites run on the same manifest. No declined attribution campaign
or replacement construction/reference samples are run.

Manifest SHA256: `91c74d1a7fbf720811c4ce84d121477678e3b810b66e2c42be898cd6cf816566`.
The [final certificate](../.tmp_tests/stage-c24-20261106/certificate/final-source-certificate.json)
verifies **9,376 files**, all **48 reference workers'** package source maps,
**82 Python process startup/exit receipts**, and
**504 file-backed import observations**. Non-file module observations are
retained explicitly. Every job passes source guards, including failed numerical
workers. Both construction and all eighteen chole samples are unique fresh
processes; no sample is retried or replaced.

| Host | Captured six-case group | Eager six-case group | Chole captured / eager |
| --- | --- | --- | --- |
| Standalone | FAIL: aorta_static impulse | PASS | 3/3 / 3/3 |
| Lab 3.0 | PASS | FAIL: aorta_shell impulse | 2/3 / 3/3 |
| Main | PASS | PASS | 3/3 / 3/3 |

**34/36 case groups** and **45/48 individual replays** pass. **17/18 chole
trials** pass; Lab 3.0 captured is 2/3 and all other groups are 3/3. Exactly three
trials per host/mode are retained under the unchanged
majority-of-three policy. The two failed case groups are new C2.4 observations,
separate from the accepted C2.3 results. Every failed report, endpoint array and
source receipt is retained.

| New C2.4 failed comparison | Index (world, tool, component) | Maximum absolute difference | Unchanged tolerance |
| --- | --- | ---: | ---: |
| lab30-eager, aorta_shell, masked_reset impulse | (1, 0, 0) | 6.4711410232121125e-05 | 4.6263936383184046e-05 |
| lab30-eager, aorta_shell, stepped impulse | (1, 0, 0) | 6.4711410232121125e-05 | 4.6263936383184046e-05 |
| standalone-captured, aorta_static, masked_reset impulse | (1, 0, 1) | 2.0954757928848267e-09 | 1.7462298274040222e-09 |

Lab 3.0 captured chole trial 0 also failed 19 product comparisons, including
grasp/contact impulses and moments, tool force/jaw load and volumetric grasp
products. Its [full failed report](../.tmp_tests/stage-c24-20261106/certificate/lab30-captured/chole_tet_med.trial0/chole_tet_med.json)
and endpoint arrays remain intact. Trials 1 and 2 pass, so the case group passes
the unchanged majority policy; trial 0 is not relabeled as a pass. All 19 failed
comparisons are included in the endpoint extraction linked below.

An initial progress/reporting summary mistakenly inferred 3/3 from the passing
chole group. The final certificate audit caught trial 0; a reporting assertion
failed before publishing final counts. The corrected counts are 17/18 chole and
45/48 individual replays. No runtime source or sample changed, and no trial was
rerun. [Reporting errors and corrections](../.tmp_tests/stage-c24-20261106/reporting-observations.json)
are recorded separately, including an endpoint-extraction scan that initially
included synthetic recorder fixtures before being restricted to actual campaign
directories. No runtime source or certificate manifest changed.

These numbers are [read-only extractions from retained endpoints](../.tmp_tests/stage-c24-20261106/failed-comparison-values.json),
not an attribution experiment or a finding of harmlessness. Both exceed the
unchanged reference comparator. No automatic acceptance follows from C2.3's
progression decision or from the passing suites.

| Frozen suite | Final result |
| --- | --- |
| Full core packages/runtime | **940 passed, 47 skipped** |
| Lab 3.0 focused host suite | **109 passed, 8 skipped** |
| Main focused host suite | **112 passed, 5 skipped** |

**C2.4 is implemented; its checkpoint remains open because the two numerical
failures are unaccepted.** All requested campaign jobs and accounting audits are
complete. Work stops at C2.4; no next ownership slice, attribution campaign,
commit or push is performed. No full fresh-runtime continuation or steady-step
performance qualification is claimed.


## C2.4 construction and ownership accounting

Exactly one fresh-process chole construction sample at one world and one at 64
worlds is retained on the frozen runtime. Neither is retried. Both entry listings
record Blender PIDs 1270430 (1,716 MiB) and 1350619 (1,826 MiB). These are
construction observations under contention, not steady-step timing qualification.
Raw JSON retains phase timings, peak RSS, Warp peak/retained bytes, containers,
per-buffer sizes, source/binary hashes and the entry process listings.

Per Surface, excluding identities and inaccessible backing:

| Ownership | Logical bytes |
| --- | --- |
| Model | `56VT + 16VF + 4F + 4VN + 4VS + 16P` |
| Each live/checkpoint State | `16EF + 4EN + 4ES + 4EP + 12E` |
| Initialization templates | another full State-sized allocation |
| Already additive before C2.4 | `56VT` Model bytes |
| New Model declarations | `16VF + 4F + 4VN + 4VS + 16P` |
| New custom identity columns | `8(VF+EF+F+VN+EN+P)` |
| Removed legacy persistent arrays | full Model plus one State |

Reused Tet domain identities are counted under Tet, once. Empty custom domains
keep their inaccessible one-row backing; data padding is reported by owner,
State padding also has a template copy, and identity backing is counted at its
maximum extent. Native WORLD counters add no custom identity columns.

## New C2.4 samples

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 1.525733 | 15.500262 |
| Schema registration (s) | 0.008029 | 0.063677 |
| Native world authoring (s) | 0.009048 | 0.707305 |
| Custom row authoring (s) | 0.634921 | 12.854822 |
| Finalization (s) | 0.089629 | 1.640777 |
| Binding and runtime construction (s) | 0.679014 | 0.123480 |
| Peak host RSS (KiB) | 872,296 | 2,712,008 |
| Construction Warp peak (bytes) | 28,011,773 | 1,231,594,844 |
| Retained Warp storage (bytes) | 27,805,529 | 1,218,389,558 |
| Newton Model buffers (bytes) | 6,125,893 | 93,207,265 |
| Initialization templates (bytes) | 1,499,948 | 95,996,672 |
| Live State (bytes) | 1,567,292 | 100,306,688 |
| Live Control (bytes) | 0 | 0 |
| Checkpoint State (bytes) | 1,567,292 | 100,306,688 |
| Checkpoint Control (bytes) | 0 | 0 |
| Intentionally shared Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays (bytes) | 2,020,320 | 129,300,480 |

The chole Surface has V=1, T=18,823, S=30,038, F=18,456, N=5,612 and P=3.
There are no empty Surface domains in these samples.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Surface Model data (bytes) | 1,565,856 | 1,565,856 |
| Each Surface State (bytes) | 437,920 | 28,026,880 |
| Surface initialization templates (bytes) | 437,920 | 28,026,880 |
| Already additive C2.3 Model (bytes) | 1,054,088 | 1,054,088 |
| New Model declarations (bytes) | 511,768 | 511,768 |
| New Surface identity columns (bytes) | 532,760 | 12,663,032 |
| Removed legacy persistent arrays (bytes) | 2,003,776 | 29,592,736 |

Surface's three additive/runtime duplicates disappear. Thermal's heat/burn copies
remain explicit: additive Newton live State, initialization templates and legacy
Thermal State each retain 150,584 bytes at one world / 9,637,376 bytes at 64 worlds.
Appearance stays legacy. Native/exact i4h inverse mass and default flags still
each have 22,448 / 1,436,672 bytes per copy. The six legacy residue arrays remain
exactly **2,020,320 / 129,300,480 bytes**, with zero new Newton templates or
checkpoint copies. Live and additional checkpoint States are separate; all
Control totals and intentionally shared-Control overlap are zero for this case.

Relative to C2.3, retained storage changes by **-83,408 bytes** at one world and
**+39,635,824 bytes** at 64 worlds. This reconciles exactly as new Model fields,
new Surface identities and two copies of new State (live and template), minus
all legacy physical Surface arrays. Existing additive tables are not counted as
new declarations. Native particles remain 5,612 / 359,168 with zero native
triangles/tets/edges. Historical construction tables below are carried unchanged;
their earlier checkpoint wording describes the historical samples only.

## C2.3 construction accounting

Exactly one fresh-process chole sample at one world and one at 64 worlds is taken
on the frozen final runtime. Samples retain phase timing, contention, host peak
RSS, Warp peak, retained bytes and per-container storage. They are not retried.
No exclusive steady-step performance claim follows from construction timing.

Per Tet system, excluding identity columns and empty backing:

| Ownership | Logical bytes |
| --- | --- |
| Model data | `24VT + 4T + 16VS + 4S + 4ET + 16EP` |
| Each live/checkpoint State | `4E(T+S)` |
| Initialization templates | another `4E(T+S)` |
| Already additive C2.2 Model | `24VT + 4T + 4ET` |
| Already additive C2.2 State/templates, each | `4ET` |
| Newly declared Model data | `16VS + 4S + 16EP` |
| Newly declared State/templates, each | `4ES` |
| Removed legacy persistent allocations | `24VT + 4T + 16VS + 4S + 16EP + 4E(T+S)` |

The measurement JSON separates Model buffers, initialization templates, each
live/checkpoint State and Control, intentionally shared Control and its overlap,
Tet declarations and remaining duplicates. Surface's three face tables and
Thermal heat/burn retain additive and legacy copies. Native versus exact i4h
inverse mass/default flags and State initialization templates remain explicit.

## Historical C2.3 construction samples (carried unchanged)

These are the two new samples, distinct from all historical tables below.
Both entry compute-process listings were empty. They are single construction
samples, not distributions or an exclusive steady-step qualification.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.789688 | 12.371675 |
| Schema registration (s) | 0.005637 | 0.051703 |
| Native world authoring (s) | 0.009141 | 0.748344 |
| Custom row authoring (s) | 0.529705 | 9.991822 |
| Finalization (s) | 0.073758 | 1.401022 |
| Binding and runtime construction (s) | 0.068726 | 0.062566 |
| Peak host RSS (KiB) | 729,168 | 2,272,284 |
| Construction Warp peak (bytes) | 28,095,181 | 1,191,959,020 |
| Retained Warp storage (bytes) | 27,888,937 | 1,178,753,734 |
| Newton Model buffers (bytes) | 5,081,365 | 80,032,465 |
| Newton initialization templates (bytes) | 1,062,028 | 67,969,792 |
| Live Newton State (bytes) | 1,129,372 | 72,279,808 |
| Live Newton Control (bytes) | 0 | 0 |
| Additional checkpoint State (bytes) | 1,129,372 | 72,279,808 |
| Additional checkpoint Control (bytes) | 0 | 0 |
| Intentionally shared Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays, total (bytes) | 2,020,320 | 129,300,480 |

Each sample retains its [raw JSON and phase log](../.tmp_tests/stage-c23-20261106/certificate/).
The chole Tet has V=1, T=18,823, S=30,038, P=3. No empty Tet rows are present.

| Tet accounting (bytes) | 1 world | 64 worlds |
| --- | ---: | ---: |
| Model data | 1,203,144 | 5,949,564 |
| Each State | 195,444 | 12,508,416 |
| Initialization templates | 195,444 | 12,508,416 |
| Already additive C2.2 Model | 602,336 | 5,345,732 |
| Already additive C2.2 State/templates, each | 75,292 | 4,818,688 |
| New Model data | 600,808 | 603,832 |
| New State/templates, each | 120,152 | 7,689,728 |
| All Tet world/owner identity columns | 1,172,688 | 25,800,144 |
| Removed legacy persistent allocations | 1,323,296 | 13,639,292 |

| Remaining duplicate ownership (bytes per listed copy) | 1 world | 64 worlds |
| --- | ---: | ---: |
| Surface three face tables: additive Newton Model / legacy Surface Model, each | 1,054,088 | 1,054,088 |
| Thermal heat/burn: additive Newton live State / initialization templates / legacy Thermal State, each | 150,584 | 9,637,376 |
| Native / exact i4h inverse mass, each | 22,448 | 1,436,672 |
| Native / i4h default flags, each | 22,448 | 1,436,672 |

Initialization templates are retained deliberately; they are separate from live
State ownership and checkpoint copies. The additional checkpoint State includes
the additive Thermal declarations, but legacy history has not migrated into
those declarations. Surface declarations remain shared on Model. The six residue arrays remain only in legacy runtime
State and add zero Newton template/checkpoint bytes.

Relative to C2.2, retained storage increases by **238,752 bytes** at one world
and **18,205,596 bytes** at 64 worlds. This reconciles exactly as newly declared
edge/material Model data plus new domain identities plus two copies of the new
edge-mask State (live and template), less the removed legacy Tet allocations.
The sample does not hide additive declarations already present in C2.2. Native
particle counts remain 5,612 / 359,168, with zero native tets/triangles/edges.

## Historical C2.2 construction samples (carried unchanged)

Exactly one fresh process per world count, under the frozen source and qualified
patched binary. These samples are never retried for timing. They measure
construction under contention, without a steady-step performance claim.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.668543 | 9.587114 |
| Schema registration (s) | 0.004977 | 0.042819 |
| Native world authoring (s) | 0.008574 | 0.749086 |
| Custom row authoring (s) | 0.403617 | 7.445778 |
| Finalization (s) | 0.066183 | 1.164136 |
| Binding and runtime construction (s) | 0.086903 | 0.074281 |
| Construction Warp peak (bytes) | 27,856,429 | 1,173,753,424 |
| Retained Warp storage (bytes) | 27,650,185 | 1,160,548,138 |
| Newton Model buffers (bytes) | 3,759,621 | 63,567,033 |
| Newton initialization templates (bytes) | 941,876 | 60,280,064 |
| Live Newton State (bytes) | 1,009,220 | 64,590,080 |
| Live Newton Control (bytes) | 0 | 0 |
| Additional checkpoint State (bytes) | 1,009,220 | 64,590,080 |
| Additional checkpoint Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays, total (bytes) | 2,020,320 | 129,300,480 |

The Model-buffer row above includes the full native gravity allocation. The
frozen measurement helper's pointer-keyed dictionary keeps the shorter alias's
extent (E vec3 rows) instead of the native allocation's extent (E+1 rows),
undercounting that raw subtotal by 12 bytes in each JSON. The raw reports remain
unchanged: they retain both native extents (24/780 bytes) and alias extents
(12/768 bytes), proving the correction. The
[accounting correction](../.tmp_tests/stage-c22-20261106/construction-accounting-correction.json)
adds those 12 bytes without another sample. Initialization-template, State,
Control, peak and retained totals are unaffected. The original helper's frozen source
remains archived as evidence of the raw counter's limitation. The current helper
now counts the maximum extent among arrays sharing a pointer; it reports the
correct full native allocation in future samples.

Both entry listings record the concurrent vascular Python process PID 1078020
using 791 MiB. Peak host RSS, phase timings, full array bytes, and binary hashes
are retained in each JSON. The additional checkpoint State/Control containers
are measured after retained construction storage, separately from its peak.
Intentionally shared Control and shared-template overlap are both zero bytes for
this case, because its commands still use legacy owners.

| Pool accounting | 1 world | 64 worlds |
| --- | ---: | ---: |
| Logical particles | 12,288 | 786,432 |
| Model fields (bytes) | 294,912 | 18,874,368 |
| Each State (bytes) | 491,520 | 31,457,280 |
| Initialization templates (bytes) | 491,520 | 31,457,280 |
| World/owner identity columns (bytes) | 98,304 | 6,291,456 |

The static-geometry row identities add 552,248 Model bytes at either world count.
Relative to the historical C2.1 sample, retained bytes increase by 1,142,072 at
one world and 38,300,984 at 64 worlds: exactly the pool templates, pool identity
columns and static-geometry identity columns. The former live pool/variant/
geometry allocations are replaced, and their ownership changes; deleting those
sites does not remove the need for their data. Native particle counts remain
5,612 / 359,168, with zero native tets/triangles. The six residue arrays remain
legacy State, included once in retained storage and absent from the Newton
initialization templates and checkpoint containers.

## Historical C2.1 final-source construction sample (carried unchanged)

The following earlier sample is historical. All earlier intermediate tables
remain unchanged in [the C2.1 record](stage-c21-closure.md).

One fresh process at each world count on the complete startup/reset/rebuild
implementation, under the same final source manifest as all integrated gates.
Every sample is retained; these timings are not distributions or a performance
verdict. Buffer totals match both intermediate sources.

| Measurement | 1 world | 64 worlds |
| --- | ---: | ---: |
| Total construction (s) | 0.476171 | 5.442159 |
| Schema registration (s) | 0.003752 | 0.021898 |
| Native world authoring (s) | 0.008576 | 0.718893 |
| Custom row authoring (s) | 0.254328 | 3.771190 |
| Finalization (s) | 0.046083 | 0.540506 |
| Binding and runtime construction (s) | 0.060326 | 0.279208 |
| Construction Warp peak (bytes) | 26,714,357 | 1,135,452,440 |
| Retained Warp storage (bytes) | 26,508,113 | 1,122,247,154 |
| Newton Model buffers (bytes) | 2,438,605 | 37,472,653 |
| Newton initialization templates (bytes) | 450,356 | 28,822,784 |
| Live Newton State (bytes) | 517,700 | 33,132,800 |
| Live Newton Control (bytes) | 0 | 0 |
| Additional checkpoint State (bytes) | 517,700 | 33,132,800 |
| Additional checkpoint Control (bytes) | 0 | 0 |
| Shared Control/template overlap (bytes) | 0 | 0 |
| Six legacy residue arrays, total (bytes) | 2,020,320 | 129,300,480 |

Both final samples record Solvecraft PID 65109 using 1,071 MiB at entry; no
other compute process appears in those two entry listings. They remain
construction measurements under contention, not an exclusive steady-step
campaign. JSON reports retain per-array counts, peak RSS, source and binary
hashes. Residue templates/checkpoint copies remain zero because those six
legacy fields have not migrated; their 129,300,480 bytes are included in
retained Warp storage. Model particle_q is counted once among initialization
templates because Newton uses it to initialize State positions, and the new
q0 alias adds no bytes. Native counts and the remaining duplicates are unchanged.

