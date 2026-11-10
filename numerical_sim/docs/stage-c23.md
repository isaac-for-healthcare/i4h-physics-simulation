# C2.3 — Newton-owned Tet mechanics storage

**Accepted for progression to C2.4 by the user-supplied C2.4 plan (2026-11-06).**
Both certificates, all failures, construction samples and the direct-caller
correction below remain unchanged historical evidence. This acceptance supplies
no automatic waiver for new C2.4 failures.

The following is the preserved C2.3 checkpoint report:

**Implemented; checkpoint open.** Final suites pass: **878 passed / 47 skipped**
core, **109 / 8** Lab 3.0, **112 / 5** main. Numerical references pass 33/36
case groups, including all 18 chole trials. Three new `aorta_shell` impulse
failures remain unaccepted. Stop before C2.4.

C2.2 is **accepted for progression** by the user-supplied implementation plan.
Its [certificate, failed observations, samples and accounting corrections](stage-c22.md)
remain intact. That acceptance gives C2.3 no automatic failure waiver. This slice
stops before C2.4. No commit or push is made.

## Implementation boundary

`physics/src/i4h_physics/tet_storage.py` extends the existing declarations with
explicit edge variant, world-edge, shared edge-part and world-material domains.
Tet's thirteen Model fields and two State masks alias `scene.attr` through the
cached `tet_fields(scene, system_id)` record. The existing `i4h:{system_id}_{field}`
names and tet-part layout are preserved. There are no native Newton tets, edges
or triangles. Local indices and `native_tet=-1` receive no reference offsets.
World/owner columns resolve after native world creation, including foreign
prefixes/strides. Shared variants use world -1; unused variants and inactive
padding use owner -1. Empty domains retain one inaccessible backing row.

CaseAuthoring publishes domain records before runtime construction. Direct Scene
callers predeclare with `declare_tet(scene, body, system_id="tet")`, before any
allocation or schema registration, and use the existing completion lifecycle.
Empty direct constructor IDs resolve to `tet`; repeated Tet instances use explicit
matching IDs and independent declarations. They share the body's C2.2 Model
variant table. Missing declarations, changed topology, body mappings or field
layouts require reconstruction. Retained-builder completion is idempotent.

New States clone authored active0/edge_active0 templates. Tet construction and
preparation preserve selected State masks and live Model material edits. Explicit
selected-world reset gathers both live Model defaults through the current body
variant table. The four material aliases retain existing live-parameter checks.
Backing buffers, cached views and compatibility aliases participate in stale
binding checks. Solver delta/count, XPBD multipliers and PBD fallback buffers
remain local; clearing and multiplier lifetime are unchanged. The source audit
verifies all four Tet numerical kernels against the pre-C2.3 source.

Physical Surface aliases the referenced cached Tet fields. TetContainment,
Thermal, Cutting, StretchBreaking, Grasp and Hook consume those fields without
changing reference validation or system ordering. VolumetricGrasp and Centreline
consume no migrated topology/mask field and have no Tet reference: they retain
their existing particle flags/positions and shared Body.variant path. Deletion's
particle-state effects therefore follow the selected Newton State.

Physical Surface storage, including its three additive per-tet declarations,
and Thermal history retain their legacy owners. A rebound Tet preserves its
masks while Surface initializes its own legacy history. **This is a partial
migration, not full fresh-runtime continuation.** Commands, rods, coupling and
all six residue arrays remain outside the slice. The residue remains exactly
2,020,320 bytes at one world and 129,300,480 bytes at 64 worlds, with no new
Newton initialization templates or checkpoint copies.

## Verification before source freeze

[All logs and artifacts](../.tmp_tests/stage-c23-20261106/) are retained.
The initial 21-test declaration selection and 44 new ownership tests passed.
The expanded packages/consumer selection passed **393 tests, 1 skipped**.
The final ownership/accounting selection passed **51 tests**; the selected-State
cutting/thermal/stretch and fresh-process memory preflight passed **8 tests**.
After the final alias/layout guards, final preflight passed **83 tests**. These
selections overlap; they are not presented as an additive unique-test total.

Coverage includes CPU/CUDA exact defaults, multiple bodies, unused variants,
inactive padding and empty domains, repeated Tet IDs, independent States,
default/shared Controls, deliberately restored masks, live material edits,
partial/empty/repeated resets, foreign native rows, direct construction,
retained builders, missing declarations, changed capacities, incompatible
buffers, all fifteen migrated-field aliases and eager/captured PBD/XPBD scratch
lifetime. Selected-State consumer tests show cuts, thermal deletion and stretch
breaking mutate only the chosen State and preserve another State and templates.
Existing TetContainment liveness and reset tests pass in the expanded selection.

The alias-aware container counter is tested with both alias orderings and takes
the maximum backing extent per pointer. Logical Tet accounting is independently
compared with actual bound buffers. Both existing fresh-process memory probes
pass before freezing source. No construction sample is taken for this audit.
Ruff passes. The inventory checker reports **598 allocation/helper sites in
180 source files**, with zero new device-allocation sites and fifteen removed
sites: fourteen legacy persistent Tet allocations plus the constructor's temporary
all-world reset mask. Their prior field contracts and links are retained in the
[inventory delta](../.tmp_tests/stage-c23-20261106/inventory-delta.json).

Development observations retained separately: an expanded test invocation named
nonexistent `test_grasp.py`/`test_hook.py` paths and collected no tests; the corrected
selection passed. Ruff caught one unused import in the new fixture, subsequently
removed. These are harness observations, not numerical reference retries.

## Accepted C2.2 observations, separate from C2.3

C2.2's final frozen campaign retained main captured `aorta_shell` stepped impulse
and main eager `aorta_static` masked-reset impulse failures. Standalone chole was
2/3 in each mode, with both failed trials retained; other hosts were 3/3 in each
mode. Final C2.2 suites passed 823/47 skipped core, 109/8 skipped Lab and 112/5
skipped main. Earlier failed core memory probes, the corrected harness, original
and final manifests, and the 12-byte native-gravity accounting correction remain
in that record. No C2.2 result is relabeled as a pass here.

## C2.3 frozen-source campaign

The original campaign uses one source manifest and startup/exit guards on every
worker and nested Python process. It uses the unchanged six approved references, tolerances,
capacity 256, dependency pins, qualified patched Warp binary and coupling limits.
Each host/mode retains exactly three fresh chole trials under the existing
majority policy. No attribution campaign or replacement trial is run.

Original full-campaign source manifest SHA256:
`4c817b7bdf9904aa5cb024856c4a1aad9afd86507a0f4c7d8706583063aa70a6`.
The [manifest](../.tmp_tests/stage-c23-20261106/certificate/source-manifest.json)
covers **9,373 files**, including production sources, tests, measurement and
qualification harnesses, installed dependencies, native Warp binaries, qualified
host sources, approved references and the reviewed allocation contracts/inventory.

| Host | Captured six-case group | Eager six-case group | Chole captured / eager |
| --- | --- | --- | --- |
| Standalone | FAIL: aorta_shell impulse | PASS | 3/3 / 3/3 |
| Lab 3.0 | PASS | PASS | 3/3 / 3/3 |
| Main | FAIL: aorta_shell impulse | FAIL: aorta_shell impulse | 3/3 / 3/3 |

All **18/18 chole trials** pass; exactly three fresh trials per host/mode are
retained. Across the six references, **33/36 case groups** and **45/48 individual
replays** pass. These are new C2.3 observations, separate from accepted C2.2
failures. Every failed endpoint array, comparator report and process receipt is
retained; no failed sample is replaced.

| New C2.3 failed comparison | Index (world, tool, component) | Maximum absolute difference | Unchanged tolerance |
| --- | --- | ---: | ---: |
| standalone-captured, masked_reset impulse | (1, 0, 0) | 0.0001257743169844616 | 4.626393638318405e-05 |
| standalone-captured, stepped impulse | (1, 0, 0) | 0.0001257743169844616 | 4.626393638318405e-05 |
| main-captured, stepped impulse | (0, 0, 0) | 9.964626678993227e-05 | 4.626393638318405e-05 |
| main-eager, stepped impulse | (0, 0, 0) | 9.699642760097049e-05 | 4.626393638318405e-05 |

The numerical failures keep the checkpoint open. The table only extracts values
from retained endpoints; it does not establish cause or harmlessness, and no
attribution campaign is run. The final suite results and source certificate are
recorded below; the passing suites do not waive these reference failures.


## Direct-caller correction after the original campaign

The original full core suite retained **874 passed, 47 skipped, 4 failed**.
Original Lab 3.0 retained **105 passed, 8 skipped, 4 failed**; main retained
**108 passed, 5 skipped, 4 failed**. All four failures in each suite came from
`test_coupling.held_probe`, an indirect direct-Scene builder that constructed Tet
through `build_systems` without predeclaring it. Each failed with the required
missing-declaration/reconstruct error. This caller was missed in the initial
migration; it is not a waived test or a numerical-reference failure.

After every original job finished with matching startup/exit guards, the fixture
was updated with `declare_tet(scene, "block")` before schema registration. Its
assertions, tolerances, coupling behavior and all production code are unchanged.
The focused coupling selection then passed **9 tests**. Ruff and the allocation
inventory checker pass. The original fixture and every failed suite log remain
archived; the original campaign certificate remains intact.

The [final source proof](../.tmp_tests/stage-c23-20261106/certificate-reviewed/inherited-runtime-proof.json)
verifies that this test file is the only changed file from the original manifest.
All three suites run again because all execute this fixture. The eight original
construction/reference jobs execute no changed source, and their measured runtime,
scenario, reference and measurement hashes match the final manifest exactly.
There are still exactly two construction samples and eighteen fresh chole trials;
none is rerun or replaced. This correction is unrelated to the accounting helper,
whose pre-freeze audit and raw samples remain unchanged.

## Final certificate and checkpoint disposition

Final test/runtime manifest SHA256:
`99f65af402456e2bdedb9e507e514ac0051b33ca272d9b6e7e03c4209e30a7e9`.
It preserves the original runtime map and adds the corrected caller, its archived
pre-correction source and the final-suite guard/runner. Production source,
numerical scenarios and construction measurements remain unchanged.

The [final certificate](../.tmp_tests/stage-c23-20261106/certificate-reviewed/final-source-certificate.json)
verifies **9,376 files**, all 48 original reference workers' package source maps,
**82 final/inherited Python process startup/exit receipts**, and **504 file-backed
import observations**. The non-file module observation is retained explicitly.
Including the original failed-suite processes, **102 process receipts** are
preserved across both test-source maps. The source proof confirms that neither
reference workers nor construction workers execute the changed test fixture.

| Suite | Final result |
| --- | --- |
| Full core packages/runtime | **878 passed, 47 skipped** |
| Lab 3.0 focused host suite | **109 passed, 8 skipped** |
| Main focused host suite | **112 passed, 5 skipped** |

All final suite processes exited successfully and passed startup/exit source
guards. Both original and final certificates remain available. No reporting-only
accounting correction, new reference trial or replacement construction sample
was needed.

**C2.3 is implemented; its checkpoint remains open.** The new standalone captured,
main captured and main eager `aorta_shell` impulse failures exceed the unchanged
reference tolerance. C2.2 acceptance does not waive them. All eighteen chole
trials pass. C2.4 has not started; no full fresh-runtime continuation or steady-step
performance qualification is claimed. No commit or push is made.

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

## New C2.3 construction samples

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

